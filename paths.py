"""
Centralised path resolution for CORTEG.

Uses environment variables with sensible local defaults so that:
  - Local runs need zero configuration (defaults point to ~/workspace/...)
  - HPC runs set env vars in the SLURM script header

Priority: CLI arg (--data_root etc.) > env var > local default.
"""
from __future__ import annotations

import os

_LOCAL_DATA_ROOT = os.path.expanduser("~/workspace/datasets/stanford_ecog")
_LOCAL_PRETRAINED_ROOT = os.path.expanduser("~/workspace/datasets/pretrained_eeg_mae")
_LOCAL_OUTPUT_ROOT = os.path.expanduser("~/workspace/outputs/corteg")


def _env(*names: str, default: str = "") -> str:
    """First environment variable that is set, else `default`."""
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return default


def get_data_root(override: str = "") -> str:
    """Resolve the Stanford ECoG dataset root directory."""
    if override:
        return override
    return _env("CORTEG_DATA_ROOT", "ECOG_DATA_ROOT", default=_LOCAL_DATA_ROOT)


def get_pretrained_root(override: str = "") -> str:
    """Resolve the root directory for pretrained EEG-MAE checkpoints."""
    if override:
        return override
    return _env("CORTEG_PRETRAINED_ROOT", "ECOG_PRETRAINED_ROOT", default=_LOCAL_PRETRAINED_ROOT)


def get_output_root(override: str = "") -> str:
    """Resolve the root directory for experiment outputs."""
    if override:
        return override
    return _env("CORTEG_OUTPUT_ROOT", "ECOG_OUTPUT_ROOT", default=_LOCAL_OUTPUT_ROOT)


def resolve_pretrained_path(path: str) -> str:
    """Resolve a pretrained checkpoint path.

    If the path is absolute, return as-is. If relative, prepend the
    pretrained root directory.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return os.path.join(get_pretrained_root(), path)
