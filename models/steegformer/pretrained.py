from __future__ import annotations

import os
import json
from typing import Dict, Any, Optional

import torch
import torch.nn as nn


def parse_model_kwargs(model_kwargs_json: str) -> Dict[str, Any]:
    if not model_kwargs_json:
        return {}
    if os.path.exists(model_kwargs_json):
        with open(model_kwargs_json, "r") as f:
            return json.load(f)
    # A path-looking argument that does not exist is a user error, not inline
    # JSON. Say so, rather than failing later inside json.loads().
    if model_kwargs_json.strip().endswith(".json"):
        raise FileNotFoundError(
            "model_kwargs_json looks like a file path but does not exist: "
            f"{model_kwargs_json!r} (cwd={os.getcwd()}). "
            "Run the scripts from the repository root, or pass an absolute path."
        )
    return json.loads(model_kwargs_json)



def _safe_torch_load(path: str, *, trust_checkpoint: bool) -> Any:
    """
    PyTorch >=2.6 changed torch.load default weights_only=True.
    - If trust_checkpoint=True: load with weights_only=False (may execute pickle code).
    - Else: try safe weights_only=True first (may fail if ckpt contains objects like argparse.Namespace).
    """
    if trust_checkpoint:
        return torch.load(path, map_location="cpu", weights_only=False)

    # Try safe load first
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # If the user doesn't trust the ckpt, we should not silently fall back.
        raise


def _extract_state_dict(ckpt: Any, ckpt_key: str) -> Dict[str, torch.Tensor]:
    """
    Supports:
      - ckpt[ckpt_key] as state_dict
      - ckpt["state_dict"]
      - raw state_dict dict(str->tensor)
    """
    if isinstance(ckpt, dict) and ckpt_key in ckpt and isinstance(ckpt[ckpt_key], dict):
        sd = ckpt[ckpt_key]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict) and all(isinstance(k, str) for k in ckpt.keys()):
        # raw state_dict
        sd = ckpt
    else:
        keys = list(ckpt.keys()) if isinstance(ckpt, dict) else None
        raise ValueError(f"Checkpoint format not understood. type={type(ckpt)} keys={keys}")
    # Ensure tensors only
    sd2 = {k: v for k, v in sd.items() if torch.is_tensor(v)}
    return sd2


def load_pretrained_with_report(
    model_or_backbone: nn.Module,
    ckpt_path: str,
    *,
    ckpt_key: str = "model",
    strict: bool = False,
    strip_prefix: str = "",
    print_max_keys: int = 80,
    save_report_json: Optional[str] = None,
    trust_checkpoint: bool = True,
    verbose: bool = True,
) -> torch.nn.modules.module._IncompatibleKeys:
    """
    Robust checkpoint loader (based on your old working function), but adds:
      - clear printing of missing/unexpected keys
      - JSON report dump
      - explicit control of torch.load safety

    It attempts to load only the relevant backbone/encoder weights by:
      - stripping prefix (strip_prefix)
      - removing DDP prefix "module."
      - optionally removing wrapper prefix "backbone."
      - dropping decoder-only keys
      - remapping old channel embedding key -> new embedding table when possible

    Returns:
      msg from load_state_dict (missing_keys, unexpected_keys)
    """

    if not ckpt_path:
        raise ValueError("ckpt_path is required")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Accept passing wrapper model; prefer its .backbone if present
    backbone = getattr(model_or_backbone, "backbone", model_or_backbone)

    report: Dict[str, Any] = {
        "ckpt_path": ckpt_path,
        "ckpt_key": ckpt_key,
        "strict": bool(strict),
        "strip_prefix": strip_prefix,
        "trust_checkpoint": bool(trust_checkpoint),
        "steps": [],
        "remaps": [],
        "dropped_keys_count": 0,
        "dropped_keys_sample": [],
    }

    if verbose:
        print(f"[pretrained] path={ckpt_path}", flush=True)

    ckpt = _safe_torch_load(ckpt_path, trust_checkpoint=trust_checkpoint)
    sd = _extract_state_dict(ckpt, ckpt_key=ckpt_key)

    report["raw_num_keys"] = int(len(sd))
    report["raw_sample_keys"] = list(sorted(sd.keys()))[: min(50, len(sd))]

    def step(name: str, before: int, after: int, extra: Optional[Dict[str, Any]] = None):
        rec = {"name": name, "before": int(before), "after": int(after)}
        if extra:
            rec.update(extra)
        report["steps"].append(rec)

    # ---- strip_prefix ----
    if strip_prefix:
        before = len(sd)
        sd = {(k[len(strip_prefix):] if k.startswith(strip_prefix) else k): v for k, v in sd.items()}
        step("strip_prefix", before, len(sd), {"strip_prefix": strip_prefix})

    # ---- DDP module. ----
    if any(k.startswith("module.") for k in sd.keys()):
        before = len(sd)
        sd = {k[len("module."):]: v for k, v in sd.items()}
        step("strip_ddp_module", before, len(sd))

    # ---- wrapper backbone. ----
    # only keep backbone.* if present
    if any(k.startswith("backbone.") for k in sd.keys()):
        before = len(sd)
        sd = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
        step("strip_wrapper_backbone", before, len(sd))

    # ---- drop decoder keys ----
    drop_prefixes = (
        "decoder_", "decoder.", "decoder_blocks.", "decoder_norm.", "decoder_pred.",
        "dec_", "dec.",
        "mask_token",
        "decoder_embed.",
    )
    before = len(sd)
    dropped = [k for k in sd.keys() if k.startswith(drop_prefixes)]
    if dropped:
        sd = {k: v for k, v in sd.items() if not k.startswith(drop_prefixes)}
    step("drop_decoder_like_keys", before, len(sd), {"dropped": len(dropped)})
    report["dropped_keys_count"] = int(len(dropped))
    report["dropped_keys_sample"] = dropped[: min(200, len(dropped))]

    # ---- remap old channel embedding transformation -> new embedding weight ----
    # This is heuristic and only done if shapes look compatible.
    old_k = "enc_channel_emd.channel_transformation.weight"
    new_k = "enc_channel.emb.weight"
    if old_k in sd and hasattr(backbone, "enc_channel") and hasattr(backbone.enc_channel, "emb"):
        w = sd[old_k]
        tgt = backbone.enc_channel.emb.weight
        before = len(sd)

        mapped = False
        mode = None
        if w.shape == tgt.shape:
            sd[new_k] = w
            mapped = True
            mode = "direct"
        elif w.t().shape == tgt.shape:
            sd[new_k] = w.t()
            mapped = True
            mode = "transpose"
        else:
            # try partial row copy if feature dim matches
            w2 = w.t() if (w.ndim == 2 and w.t().shape[1] == tgt.shape[1]) else w
            if w2.ndim == 2 and w2.shape[1] == tgt.shape[1]:
                tmp = tgt.detach().clone()
                rows = min(tmp.shape[0], w2.shape[0])
                tmp[:rows] = w2[:rows]
                sd[new_k] = tmp
                mapped = True
                mode = f"partial_rows_{rows}"
                if verbose:
                    print(f"[pretrained] Partial channel-emb load rows=0..{rows-1}", flush=True)

        if mapped:
            report["remaps"].append({
                "from": old_k,
                "to": new_k,
                "mode": mode,
                "src_shape": list(w.shape),
                "tgt_shape": list(tgt.shape),
            })
            # remove old key so it doesn't become unexpected
            sd.pop(old_k, None)
            step("remap_channel_embedding", before, len(sd), {"mode": mode})
        else:
            report["remaps"].append({
                "from": old_k,
                "to": new_k,
                "mode": "failed",
                "src_shape": list(w.shape),
                "tgt_shape": list(tgt.shape),
            })
            if verbose:
                print(f"[pretrained] WARNING: cannot map {old_k} {tuple(w.shape)} -> {new_k} {tuple(tgt.shape)}", flush=True)

    # ---- temporal pe keys: don't force-load old ones unless you implement explicit remap ----
    # Your previous function *removed* these keys; keep the same behavior.
    before = len(sd)
    popped = []
    for k in ("enc_temporal_emd.pe", "dec_temporal_emd.pe", "enc_time.pe"):
        if k in sd:
            popped.append(k)
            sd.pop(k, None)
    if popped:
        step("drop_temporal_pe_keys", before, len(sd), {"dropped": len(popped)})
        report["dropped_temporal_pe_keys"] = popped

    # ---- filter shape mismatches (e.g. hi_only changes patch_size) ----
    if not strict:
        model_sd = backbone.state_dict()
        mismatched = [k for k in list(sd.keys())
                      if k in model_sd and sd[k].shape != model_sd[k].shape]
        for k in mismatched:
            sd.pop(k)
        if mismatched:
            step("drop_shape_mismatch", before, len(sd),
                 {"dropped": mismatched})
            report["shape_mismatch_keys"] = mismatched

    # ---- actual load ----
    msg = backbone.load_state_dict(sd, strict=bool(strict))

    missing = list(msg.missing_keys)
    unexpected = list(msg.unexpected_keys)

    report["loaded_num_keys"] = int(len(sd))
    report["missing_keys"] = missing
    report["unexpected_keys"] = unexpected
    report["missing_count"] = int(len(missing))
    report["unexpected_count"] = int(len(unexpected))

    if verbose:
        print(f"[pretrained] strict={strict} missing={len(missing)} unexpected={len(unexpected)}", flush=True)

        if missing:
            print(f"[pretrained] --- missing keys (showing up to {print_max_keys}) ---", flush=True)
            for k in missing[: max(0, int(print_max_keys))]:
                print(f"  MISSING: {k}", flush=True)

        if unexpected:
            print(f"[pretrained] --- unexpected keys (showing up to {print_max_keys}) ---", flush=True)
            for k in unexpected[: max(0, int(print_max_keys))]:
                print(f"  UNEXPECTED: {k}", flush=True)

    # ---- save report ----
    if save_report_json:
        os.makedirs(os.path.dirname(save_report_json), exist_ok=True)
        with open(save_report_json, "w") as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"[pretrained] report saved: {save_report_json}", flush=True)

    return msg
