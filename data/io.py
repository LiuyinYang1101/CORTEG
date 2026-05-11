"""Data loading utilities for ECoG feature pickles and electrode location mats."""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import scipy.io


@dataclass
class SubjectData:
    """Canonical subject record loaded from *_features.pkl (+ optional electrode_loc.mat)."""
    subject: str
    X_feat_tr: np.ndarray  # (N,C,200,2) [high, low]
    y_tr: np.ndarray       # (N,5)
    X_feat_te: np.ndarray  # (N,C,200,2)
    y_te: np.ndarray       # (N,5)
    X_raw_tr: np.ndarray   # (N,C,128)
    X_raw_te: np.ndarray   # (N,C,128)
    ecog_xyz_mm: Optional[np.ndarray] = None  # (C,3) in mm


def load_features_pkl(pkl_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a 6-array features pickle: (X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te)."""
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(pkl_path)
    with open(pkl_path, "rb") as f:
        obj = pickle.load(f)
    if not (isinstance(obj, (tuple, list)) and len(obj) == 6):
        raise ValueError(f"Expected 6 arrays in {pkl_path}, got type={type(obj)} len={len(obj) if isinstance(obj,(tuple,list)) else 'NA'}")
    X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te = obj
    return X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te


def load_ecog_xyz_mm(mat_path: str, key: str = "electrodes") -> np.ndarray:
    """Load electrode XYZ coordinates (mm) from a .mat file. Returns float32 array of shape (C, 3)."""
    if not os.path.exists(mat_path):
        raise FileNotFoundError(mat_path)
    mat = scipy.io.loadmat(mat_path)
    if key not in mat:
        raise KeyError(f"Missing key '{key}' in {mat_path}. Keys={list(mat.keys())}")
    xyz = mat[key]
    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"Expected (C,3) xyz in mm, got {xyz.shape} from {mat_path}")
    return xyz.astype(np.float32, copy=False)


def load_subject(
    file_root: str,
    subject: str,
    *,
    pkl_suffix: str = "_features.pkl",
    mat_suffix: str = "_electrode_loc.mat",
    require_xyz: bool = False,
) -> SubjectData:
    """Load one subject's features and (optionally) electrode locations into a SubjectData record."""
    pkl_path = os.path.join(file_root, f"{subject}{pkl_suffix}")
    X_feat_tr, y_tr, X_feat_te, y_te, X_raw_tr, X_raw_te = load_features_pkl(pkl_path)

    mat_path = os.path.join(file_root, f"{subject}{mat_suffix}")
    ecog_xyz_mm = None
    if os.path.exists(mat_path):
        ecog_xyz_mm = load_ecog_xyz_mm(mat_path)
    elif require_xyz:
        raise FileNotFoundError(f"Missing electrode locations for {subject}: {mat_path}")

    # shape checks (canonical contract)
    if X_feat_tr.ndim != 4 or X_feat_tr.shape[-1] != 2:
        raise ValueError(f"{subject}: X_feat_tr must be (N,C,200,2), got {X_feat_tr.shape}")
    if X_raw_tr.ndim != 3:
        raise ValueError(f"{subject}: X_raw_tr must be (N,C,128), got {X_raw_tr.shape}")
    if y_tr.ndim != 2 or y_tr.shape[1] != 5:
        raise ValueError(f"{subject}: y_tr must be (N,5), got {y_tr.shape}")

    C = int(X_raw_tr.shape[1])
    if ecog_xyz_mm is not None and int(ecog_xyz_mm.shape[0]) != C:
        raise ValueError(f"{subject}: xyz has C={ecog_xyz_mm.shape[0]} but X_raw has C={C}")

    return SubjectData(
        subject=subject,
        X_feat_tr=np.asarray(X_feat_tr),
        y_tr=np.asarray(y_tr),
        X_feat_te=np.asarray(X_feat_te),
        y_te=np.asarray(y_te),
        X_raw_tr=np.asarray(X_raw_tr),
        X_raw_te=np.asarray(X_raw_te),
        ecog_xyz_mm=ecog_xyz_mm,
    )
