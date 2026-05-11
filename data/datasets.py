"""PyTorch Dataset classes for single-stream and dual-stream ECoG inputs."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

import numpy as np
import torch
from torch.utils.data import Dataset


class SingleStreamDataset(Dataset):
    """
    Returns dict:
      {"x": [C,...], "y": [5], "sid": int}
    """
    def __init__(self, x: np.ndarray, y: np.ndarray, sid: int):
        assert len(x) == len(y)
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.sid = int(sid)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"x": self.x[idx], "y": self.y[idx], "sid": self.sid}


class HiLoAddDataset(Dataset):
    """
    Returns dict:
      {"x_raw": [C,T_raw], "x_hi": [C,T_hi], "y": [5], "sid": int}
    """
    def __init__(self, x_raw: np.ndarray, x_hi: np.ndarray, y: np.ndarray, sid: int):
        assert len(x_raw) == len(x_hi) == len(y)
        self.x_raw = torch.as_tensor(x_raw, dtype=torch.float32)
        self.x_hi = torch.as_tensor(x_hi, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.sid = int(sid)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"x_raw": self.x_raw[idx], "x_hi": self.x_hi[idx], "y": self.y[idx], "sid": self.sid}


class PairedLatentDataset(Dataset):
    """
    Returns dict:
      {"x_low": [C,T_low], "x_high": [...], "y": [5], "sid": int}
    """
    def __init__(self, x_low: np.ndarray, x_high: np.ndarray, y: np.ndarray, sid: int):
        assert len(x_low) == len(x_high) == len(y)
        self.x_low = torch.as_tensor(x_low, dtype=torch.float32)
        self.x_high = torch.as_tensor(x_high, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)
        self.sid = int(sid)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {"x_low": self.x_low[idx], "x_high": self.x_high[idx], "y": self.y[idx], "sid": self.sid}
