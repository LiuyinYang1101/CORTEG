"""PyTorch Dataset for the dual-stream (broadband + high-gamma) ECoG input."""
from __future__ import annotations
from typing import Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset


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
