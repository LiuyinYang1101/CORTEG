"""
Shared utilities for ECoG experiment scripts.

Provides helpers used across multiple training entry points:
  - reproducibility (set_seed, seed_worker)
  - multi-loader evaluation (evaluate_multi_loader)
  - result formatting (format_subject_table)
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
