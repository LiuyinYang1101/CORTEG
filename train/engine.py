# ecog_finger/train/engine.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import corr_per_dim, nanmean, mse as mse_np


# -------------------------
# Helpers
# -------------------------
def _pull_aux_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Try common locations for aux losses / diagnostics:
      - model.backbone.last_aux
      - model.last_aux
    NOTE:
      This is for *logging only*. The optimized objective MUST be defined by step_fn.
    """
    aux: Dict[str, torch.Tensor] = {}
    bb = getattr(model, "backbone", None)
    if bb is not None:
        d = getattr(bb, "last_aux", None)
        if isinstance(d, dict):
            aux.update({k: v for k, v in d.items() if torch.is_tensor(v)})

    d2 = getattr(model, "last_aux", None)
    if isinstance(d2, dict):
        aux.update({k: v for k, v in d2.items() if torch.is_tensor(v)})
    return aux


def _to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _autocast(enabled: bool, *, device: torch.device, amp_dtype: str):
    """
    AMP context.
    - On CUDA: bf16 recommended on H100 for stability.
    - On CPU: disabled context.
    """
    if not enabled or device.type != "cuda":
        return torch.autocast(device_type="cpu", enabled=False)

    dtype = torch.bfloat16 if amp_dtype.lower() in {"bf16", "bfloat16"} else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype, enabled=True)


def _scalarize_for_log(v: Any) -> Optional[float]:
    """
    Convert a value to a python float for logging.
    - If tensor: only accept scalar tensors (numel==1).
    - If non-tensor: best-effort float conversion.
    Returns None if it should be skipped.
    """
    if torch.is_tensor(v):
        if v.numel() != 1:
            return None
        return float(v.detach().item())

    try:
        return float(v)
    except Exception:
        return None


@dataclass
class EngineConfig:
    use_amp: bool = False
    amp_dtype: str = "bf16"  # "bf16" or "fp16"
    accum_iter: int = 1
    max_norm: float = 1.0
    skip_nonfinite: bool = True


# -------------------------
# Train
# -------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    cfg: EngineConfig,
    step_fn: Callable[[nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]],
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    post_backward_fn: Optional[Callable[[nn.Module], Optional[Dict[str, float]]]] = None,
) -> Dict[str, float]:
    """
    IMPORTANT:
      - step_fn defines the optimized objective via out["loss_total"] (preferred) or out["loss"] (fallback).
      - Engine does NOT automatically add aux losses.
      - Engine only *logs* model/backbone.last_aux (scalars only).
    """
    model.train()
    amp = bool(cfg.use_amp and device.type == "cuda")

    # GradScaler only needed for fp16; for bf16 it's unnecessary.
    use_scaler = amp and (cfg.amp_dtype.lower() in {"fp16", "float16"})

    if scaler is None:
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
        except Exception:
            scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    optimizer.zero_grad(set_to_none=True)

    total = defaultdict(float)
    n = 0
    skipped = 0
    accum_iter = max(1, int(cfg.accum_iter))

    for it, batch in enumerate(loader):
        batch = _to_device(batch, device)
        bs = int(batch["y"].shape[0])

        with _autocast(amp, device=device, amp_dtype=cfg.amp_dtype):
            out = step_fn(model, batch)

            # step_fn should provide these; keep fallbacks for older code
            loss_main = out["loss_main"] if "loss_main" in out else out["loss"]
            loss_total = out.get("loss_total", out.get("loss", loss_main))

        # ----- non-finite guard on TOTAL objective -----
        if cfg.skip_nonfinite and (not torch.isfinite(loss_total).item()):
            skipped += 1
            continue

        n += bs

        # ---- log main & total loss ----
        if torch.is_tensor(loss_main) and loss_main.numel() == 1:
            total["loss_main"] += float(loss_main.detach().item()) * bs
        if torch.is_tensor(loss_total) and loss_total.numel() == 1:
            total["loss_total"] += float(loss_total.detach().item()) * bs

        # ---- log step_fn outputs except y_hat ----
        # Only scalar tensors (or scalar numbers) are logged.
        for k, v in out.items():
            if k in {"y_hat"}:
                continue
            fv = _scalarize_for_log(v)
            if fv is None:
                continue
            if np.isfinite(fv):
                total[k] += fv * bs

        # ---- log aux diagnostics (scalars only) ----
        aux = _pull_aux_dict(model)
        for k, v in aux.items():
            fv = _scalarize_for_log(v)
            if fv is None:
                continue
            if np.isfinite(fv):
                total[k] += fv * bs

        # backward on scaled TOTAL objective
        loss_scaled = loss_total / float(accum_iter)

        if use_scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        do_step = ((it + 1) % accum_iter == 0) or ((it + 1) == len(loader))
        if do_step:
            if cfg.max_norm and cfg.max_norm > 0:
                if use_scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.max_norm))

            # Post-backward hook (e.g. OGM-GE gradient modulation)
            if post_backward_fn is not None:
                pb_info = post_backward_fn(model)
                if isinstance(pb_info, dict):
                    for k, v in pb_info.items():
                        fv = _scalarize_for_log(v)
                        if fv is not None and np.isfinite(fv):
                            total[k] += fv * bs

            if use_scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

    out_epoch = {k: v / max(1, n) for k, v in total.items()}

    # For backward compatibility:
    # - "loss" reported as loss_total
    if "loss_total" in out_epoch:
        out_epoch["loss"] = out_epoch["loss_total"]
    else:
        out_epoch["loss"] = out_epoch.get("loss_main", float("nan"))

    out_epoch["n"] = float(n)
    out_epoch["skipped_batches"] = float(skipped)
    return out_epoch


# -------------------------
# Eval
# -------------------------
@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    step_fn: Callable[[nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]],
    use_amp: bool = False,
    amp_dtype: str = "bf16",
) -> Dict[str, Any]:
    """
    Evaluation computes metrics (corr/mse) and also logs scalar losses.

    IMPORTANT:
      - Uses step_fn outputs for loss_main/loss_total (does NOT auto-add aux)
      - Also logs model/backbone.last_aux scalars for diagnostics.
    """
    model.eval()
    amp = bool(use_amp and device.type == "cuda")

    by_sid_pred = defaultdict(list)
    by_sid_true = defaultdict(list)

    # scalar tracking
    total = defaultdict(float)
    n_total = 0

    for batch in loader:
        batch = _to_device(batch, device)
        bs = int(batch["y"].shape[0])

        with _autocast(amp, device=device, amp_dtype=amp_dtype):
            out = step_fn(model, batch)
            y_hat = out["y_hat"]

            loss_main = out["loss_main"] if "loss_main" in out else out.get("loss", None)
            loss_total = out.get("loss_total", out.get("loss", loss_main))

            if torch.is_tensor(loss_main) and loss_main.numel() == 1:
                total["loss_main"] += float(loss_main.detach().item()) * bs
            if torch.is_tensor(loss_total) and loss_total.numel() == 1:
                total["loss_total"] += float(loss_total.detach().item()) * bs

            # log step_fn scalars (except y_hat)
            for k, v in out.items():
                if k in {"y_hat"}:
                    continue
                fv = _scalarize_for_log(v)
                if fv is None:
                    continue
                if np.isfinite(fv):
                    total[k] += fv * bs

            # log aux diagnostics
            aux = _pull_aux_dict(model)
            for k, v in aux.items():
                fv = _scalarize_for_log(v)
                if fv is None:
                    continue
                if np.isfinite(fv):
                    total[k] += fv * bs

        n_total += bs

        # cast to fp32 before numpy
        sids = batch["sid"].detach().cpu().numpy().astype(int)
        y_true = batch["y"].detach().to(torch.float32).cpu().numpy()
        y_pred = y_hat.detach().to(torch.float32).cpu().numpy()

        for sid, yt, yp in zip(sids, y_true, y_pred):
            by_sid_true[sid].append(yt)
            by_sid_pred[sid].append(yp)

    summary = {}
    scores = []
    mses = []

    for sid in sorted(by_sid_true.keys()):
        yt = np.stack(by_sid_true[sid], axis=0)
        yp = np.stack(by_sid_pred[sid], axis=0)

        c = corr_per_dim(yp, yt)
        cm = nanmean(c)
        m = mse_np(yp, yt)

        summary[int(sid)] = {
            "corr": c,
            "corr_mean": float(cm) if np.isfinite(cm) else float("nan"),
            "mse": float(m) if np.isfinite(m) else float("nan"),
            "n": int(yt.shape[0]),
        }
        scores.append(cm)
        mses.append(m)

    scalars = {k: (v / max(1, n_total)) for k, v in total.items()}

    return {
        "by_sid": summary,
        "score": float(np.nanmean(scores)) if len(scores) else float("nan"),
        "score_mse": float(np.nanmean(mses)) if len(mses) else float("nan"),
        **scalars,  # may include loss_main/loss_total/loss_sp/kl_*/etc
    }
