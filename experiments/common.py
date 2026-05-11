"""
Shared utilities for ECoG experiment scripts.

Provides helpers used across multiple training entry points:
  - reproducibility (set_seed, seed_worker)
  - multi-loader evaluation (evaluate_multi_loader)
  - result formatting (format_subject_table)
  - model saving (save_merged_model, save_test_trajectories)
  - config parsing (safe_parse_model_kwargs)
"""
from __future__ import annotations

import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from train.engine import evaluate
from models.steegformer.pretrained import parse_model_kwargs as _parse_model_kwargs
from paths import get_data_root, get_output_root

# Default subject list for the Stanford ECoG dataset
STANFORD_SUBJECTS: List[str] = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set random seeds for Python, NumPy, and PyTorch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker init function for reproducible data loading."""
    wseed = torch.initial_seed() % (2 ** 32)
    np.random.seed(wseed)
    random.seed(wseed)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_multi_loader(
    model: nn.Module,
    loaders,
    device: torch.device,
    step_fn,
    use_amp: bool,
) -> Dict[str, Any]:
    """Evaluate over a list of per-subject DataLoaders and merge results.

    Each loader contains data for one subject.  Results are aggregated
    into a single dict with the same structure as ``train.engine.evaluate``.

    Args:
        model: The model to evaluate.
        loaders: List of DataLoaders, one per subject.
        device: Target device.
        step_fn: Callable(model, batch) → dict with 'y_hat'.
        use_amp: Whether to use automatic mixed precision.

    Returns:
        Dict with keys: 'by_sid', 'score' (mean corr), 'score_mse',
        and weighted-average scalar loss keys.
    """
    merged_summary: Dict[int, Any] = {}
    all_scores: List[float] = []
    all_mses: List[float] = []
    total_metrics: Dict[str, float] = {}
    total_n = 0

    for loader in loaders:
        res = evaluate(model, loader, device, step_fn=step_fn, use_amp=use_amp)
        for sid, rec in res["by_sid"].items():
            merged_summary[sid] = rec
            all_scores.append(rec["corr_mean"])
            all_mses.append(rec["mse"])
        n_this = sum(r["n"] for r in res["by_sid"].values())
        for k, v in res.items():
            if k not in ("by_sid", "score", "score_mse") and isinstance(v, (int, float)):
                total_metrics[k] = total_metrics.get(k, 0.0) + v * n_this
        total_n += n_this

    final = {k: v / max(1, total_n) for k, v in total_metrics.items()}
    final["by_sid"] = merged_summary
    final["score"] = float(np.nanmean(all_scores)) if all_scores else float("nan")
    final["score_mse"] = float(np.nanmean(all_mses)) if all_mses else float("nan")
    return final


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------

def format_subject_table(
    val_report: Dict[str, Any],
    subjects: Optional[List[str]] = None,
) -> str:
    """Format per-subject evaluation results as a human-readable table.

    Args:
        val_report: Output of ``evaluate_multi_loader`` or ``evaluate``.
        subjects: Optional list of subject name strings indexed by sid.
                  If None, the integer sid is printed instead.

    Returns:
        Multi-line string ready to print.
    """
    lines = [
        f"{'sid':>3}  {'sub':<4}  {'n':>5}  {'corr_mean':>9}  {'corr[5]':<35}  {'mse':>9}",
        "-" * 75,
    ]
    for sid in sorted(val_report["by_sid"].keys()):
        rec = val_report["by_sid"][sid]
        sub_name = subjects[sid] if (subjects and sid < len(subjects)) else str(sid)
        corr_str = "[" + ",".join(
            [f"{c: .3f}" if np.isfinite(c) else "  nan" for c in rec["corr"]]
        ) + "]"
        lines.append(
            f"{sid:>3}  {sub_name:<4}  {rec['n']:>5}  {rec['corr_mean']:>9.4f}"
            f"  {corr_str:<35}  {rec['mse']:>9.5f}"
        )
    lines.append(f"SCORE = {val_report['score']:.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model saving
# ---------------------------------------------------------------------------

def save_merged_model(
    model: nn.Module,
    save_path: str,
    lora_r: int = 0,
    lora_alpha: int = 0,
) -> None:
    """Merge LoRA adapters into base weights and save the state dict.

    If ``lora_r > 0``, the LoRA update ``(B @ A) * (alpha / r)`` is added
    to each adapted weight matrix and the adapter attributes are removed so
    the saved checkpoint is a standard dense model.

    Args:
        model: The model to save (modified in-place for LoRA merging).
        save_path: Destination ``.pth`` file path.
        lora_r: LoRA rank used during training (0 = no LoRA).
        lora_alpha: LoRA alpha scaling factor.
    """
    model.eval()
    if lora_r > 0:
        scaling = lora_alpha / lora_r
        print(f"Merging LoRA weights (scaling={scaling:.2f}) before saving...")
        with torch.no_grad():
            for _name, module in model.named_modules():
                if hasattr(module, "lora_A") and hasattr(module, "lora_B") and hasattr(module, "weight"):
                    module.weight.add_((module.lora_B @ module.lora_A) * scaling)
                    delattr(module, "lora_A")
                    delattr(module, "lora_B")
                    if hasattr(module, "lora_dropout"):
                        delattr(module, "lora_dropout")
                    if hasattr(module, "scaling"):
                        delattr(module, "scaling")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to: {save_path}")


def save_test_trajectories(
    model: nn.Module,
    loaders,
    subjects: List[str],
    step_fn,
    device: torch.device,
    save_path: str,
) -> None:
    """Run inference on test loaders and save predicted/true trajectories.

    Saves a compressed ``.npz`` file with keys ``{subject}_pred`` and
    ``{subject}_true`` for each subject.

    Args:
        model: Trained model.
        loaders: List of test DataLoaders, one per subject.
        subjects: Subject name strings (indexed by loader position).
        step_fn: Callable(model, batch) → dict with 'y_hat'.
        device: Target device.
        save_path: Destination ``.npz`` file path.
    """
    model.eval()
    results: Dict[str, np.ndarray] = {}
    print(f"Generating test trajectories for {len(loaders)} subjects...")
    with torch.no_grad():
        for i, loader in enumerate(loaders):
            sub_name = subjects[i] if i < len(subjects) else f"sub_{i}"
            preds, trues = [], []
            for batch in loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                out = step_fn(model, batch)
                preds.append(out["y_hat"].cpu().numpy())
                trues.append(batch["y"].cpu().numpy())
            if preds:
                results[f"{sub_name}_pred"] = np.concatenate(preds, axis=0)
                results[f"{sub_name}_true"] = np.concatenate(trues, axis=0)
    np.savez_compressed(save_path, **results)
    print(f"Trajectories saved to: {save_path}")


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def safe_parse_model_kwargs(model_kwargs_json: str) -> Dict[str, Any]:
    """Parse a JSON string of model kwargs, returning {} for empty/null input.

    Args:
        model_kwargs_json: JSON string, file path, or empty/null string.

    Returns:
        Parsed dict, or empty dict if input is absent.
    """
    if model_kwargs_json is None:
        return {}
    s = str(model_kwargs_json).strip()
    if s == "" or s.lower() in {"none", "null"}:
        return {}
    return _parse_model_kwargs(s)
