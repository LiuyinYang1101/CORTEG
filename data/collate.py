"""Collate functions and per-subject electrode coordinate bank."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np
import torch


@dataclass
class SubjectXYZBank:
    """
    Stores per-subject electrode xyz in meters.
    xyz_m_by_sid: list where index is sid -> np.ndarray[C,3] float32 meters
    """
    xyz_m_by_sid: List[np.ndarray]

    @staticmethod
    def from_mm(xyz_mm_by_sid: List[np.ndarray]) -> "SubjectXYZBank":
        xyz_m = []
        for x in xyz_mm_by_sid:
            if x is None:
                xyz_m.append(None)
            else:
                x = np.asarray(x, dtype=np.float32)
                xyz_m.append(x / 1000.0)
        return SubjectXYZBank(xyz_m_by_sid=xyz_m)

    def get_m(self, sid: int) -> np.ndarray:
        x = self.xyz_m_by_sid[int(sid)]
        if x is None:
            raise ValueError(f"No ecog_xyz for sid={sid}")
        return x


def make_collate_fn(xyz_bank: Optional[SubjectXYZBank]):
    """
    Collate items that are dicts and attach:
      - sid: LongTensor[B]
      - ecog_xyz: FloatTensor[B,C,3] (meters), if xyz_bank is provided
    """
    def collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        # stack keys except sid
        sids = torch.as_tensor([int(b["sid"]) for b in batch], dtype=torch.long)
        out["sid"] = sids

        # stack y always
        out["y"] = torch.stack([b["y"] for b in batch], dim=0)

        # optional inputs
        for k in ["x", "x_raw", "x_hi", "x_low", "x_high"]:
            if k in batch[0]:
                out[k] = torch.stack([b[k] for b in batch], dim=0)

        # Forward any extra integer keys (e.g., task_id)
        _handled = {"sid", "y", "x", "x_raw", "x_hi", "x_low", "x_high"}
        for k in batch[0]:
            if k not in _handled and isinstance(batch[0][k], (int, float)):
                out[k] = torch.tensor([b[k] for b in batch])

        if xyz_bank is not None:
            # assume consistent C across subjects (enforced elsewhere)
            xyz_list = [xyz_bank.get_m(int(s)) for s in sids.tolist()]
            out["ecog_xyz"] = torch.as_tensor(np.stack(xyz_list, axis=0), dtype=torch.float32)
        else:
            out["ecog_xyz"] = None

        return out

    return collate
