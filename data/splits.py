
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class TailSplit:
    """Last `val_ratio` as validation (time-ordered)."""
    val_ratio: float = 0.1

    def split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n <= 1:
            return np.arange(n), np.array([], dtype=int)
        n_val = max(1, int(round(n * float(self.val_ratio))))
        n_val = min(n_val, n-1)
        idx = np.arange(n)
        tr = idx[:-n_val]
        va = idx[-n_val:]
        return tr, va
