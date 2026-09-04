"""Adapters for the three intracranial foundation models CORTEG is compared against.

BrainBERT, the Population Transformer (PopT) and Brant are third-party models.
Neither their code nor their weights are redistributed here -- BrainBERT's
repository carries no LICENSE at all -- so each is reached through a clone and a
checkpoint you provide, named by environment variable:

    BrainBERT   BRAINBERT_REPO     git clone https://github.com/czlwang/BrainBERT
                BRAINBERT_WEIGHTS  stft_large_pretrained.pth, from the Google Drive
                                   link in that repository's README
    PopT        POPT_REPO          git clone https://github.com/czlwang/PopulationTransformer
                POPT_WEIGHTS       pretrained_popt_brainbert_stft.pth, from
                                   huggingface.co/PopulationTransformer/popt_brainbert_stft
                                   (PopT also needs BRAINBERT_REPO: it is defined
                                   over frozen BrainBERT embeddings)
    Brant       BRANT_SRC          Brant_src/ from huggingface.co/Daoze/Brant
                BRANT_WEIGHTS      the checkpoint linked from that model card

The preprocessing here is not ours to choose: each model is fed exactly what it
was pretrained on, because a foundation-model comparison is only meaningful if
each model sees its own native input. Every constant below is traced to the
upstream repository or paper, and changing one silently changes the baseline.

One implementation detail worth knowing: BrainBERT and PopT both ship top-level
packages called ``models``, ``data`` and ``tasks``, which collide head-on with
this repository's. ``_import_scope`` evicts the colliding names from sys.modules
for the duration of the import and restores them afterwards. Without it,
``import models`` inside BrainBERT silently resolves to CORTEG's package.
"""

from __future__ import annotations

import contextlib
import os
import sys
from typing import Optional

import numpy as np

try:                      # torch is required to run a model, not to import this module
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:       # pragma: no cover
    torch = nn = F = None


def _env_path(name: str, what: str) -> str:
    """Resolve a third-party path from the environment, or explain how to get it."""
    p = os.environ.get(name, "")
    if not p or not os.path.exists(p):
        raise SystemExit(
            f"{what} is not available.\n"
            f"  Set ${name} to your own copy — this repository does not\n"
            f"  redistribute third-party code or weights. See the Intracranial\n"
            f"  foundation models section of README.md for the download sources.\n"
            f"  Currently: ${name}={p!r}")
    return p


@contextlib.contextmanager
def _import_scope(prefixes, *repos):
    """Import from a third-party clone whose package names collide with ours.

    BrainBERT and PopT both ship top-level `models`, `data` and `tasks` packages.
    Two things are needed for `import models` to reach theirs rather than this
    repository's: the clone must come FIRST on sys.path, and the already-imported
    colliding modules must be evicted so the import actually re-resolves. Doing
    only one of the two silently returns CORTEG's package.

    This module deliberately lives at the repository root rather than inside
    `models/`, so that evicting `models` can never pull the ground out from under
    the module doing the evicting.
    """
    saved = {k: v for k, v in sys.modules.items() if k.split(".")[0] in prefixes}
    for k in saved:
        del sys.modules[k]
    added = [r for r in repos if r and r not in sys.path]
    for r in reversed(added):
        sys.path.insert(0, r)
    try:
        yield
    finally:
        for r in added:
            if r in sys.path:
                sys.path.remove(r)
        for k in list(sys.modules):
            if k.split(".")[0] in prefixes:
                del sys.modules[k]
        sys.modules.update(saved)



# ==========================================================================
# Shared preprocessing (BrainBERT's pipeline; PopT reuses it verbatim)
# ==========================================================================

PRETRAIN_FS = 2048          # Hz
STFT_NPERSEG = 400
STFT_NOVERLAP = 350         # hop = 50 samples
STFT_FREQ_CUTOFF = 40       # keep first 40 freq bins
STFT_ZSCORE_CLIP = 10       # frames dropped each end after per-freq z-score
HIDDEN_DIM = 768            # masked_tf_model_large
INPUT_DIM = 40              # == STFT_FREQ_CUTOFF
POOL_HALF = 5               # center pooling uses out[:, mid-5:mid+5]

# Mains harmonics BrainBERT notch-filters (data/h5_data_reader.py freqs_to_filter).
_NOTCH_FREQS = [60, 120, 180, 240, 300, 360]


# ----------------------------------------------------------------------------


_BB_COLLIDE = ("models", "preprocessors", "criterions", "data", "tasks", "util")
_PT_COLLIDE = ("models", "preprocessors", "criterions", "datasets", "data", "tasks", "util")


def brainbert_repo() -> str:
    return _env_path("BRAINBERT_REPO", "The BrainBERT repository")


def brainbert_weights() -> str:
    return _env_path("BRAINBERT_WEIGHTS", "The BrainBERT stft_large checkpoint")


def popt_repo() -> str:
    return _env_path("POPT_REPO", "The Population Transformer repository")


def brant_src() -> str:
    return _env_path("BRANT_SRC", "Brant_src/ from the Brant model card")


def brant_weights() -> str:
    return _env_path("BRANT_WEIGHTS", "The Brant checkpoint")


def _ensure_repo_on_path() -> None:
    r = brainbert_repo()
    if r not in sys.path:
        sys.path.insert(0, r)


def _ensure_popt_on_path() -> None:
    for r in (popt_repo(), brainbert_repo()):
        if r not in sys.path:
            sys.path.insert(0, r)


def _notch_filter(x: np.ndarray, fs: float, freqs=_NOTCH_FREQS, Q: int = 30) -> np.ndarray:
    """Notch out mains harmonics that fall below Nyquist. x: (..., T)."""
    from scipy import signal as sps
    nyq = fs / 2.0
    y = x
    for f0 in freqs:
        if f0 >= nyq - 1.0:      # above (or at) Nyquist -> skip
            continue
        w0 = f0 / nyq
        b, a = sps.iirnotch(w0, Q)
        y = sps.lfilter(b, a, y, axis=-1)
    return y


def _zscore_time(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-score over the time axis (last). x: (..., T)."""
    mn = x.mean(axis=-1, keepdims=True)
    sd = x.std(axis=-1, keepdims=True)
    sd = np.where(sd < eps, 1.0, sd)
    return (x - mn) / sd


def _laplacian_xyz(x: np.ndarray, xyz_mm: np.ndarray, k: int = 2) -> np.ndarray:
    """XYZ-nearest-neighbor Laplacian re-ref (analogue of BrainBERT's shank Laplacian).

    x: (C, T) for ONE window; xyz_mm: (C, 3). Returns (C, T): each channel minus
    the mean of its k nearest spatial neighbors. BrainBERT uses the 2 same-shank
    neighbors; we use the 2 nearest by Euclidean distance (no shank info on grids).
    """
    C = x.shape[0]
    if xyz_mm is None or xyz_mm.shape[0] != C:
        raise ValueError(
            f"reref='laplacian_xyz' needs xyz_mm of shape ({C},3); got "
            f"{None if xyz_mm is None else xyz_mm.shape}"
        )
    d2 = ((xyz_mm[:, None, :] - xyz_mm[None, :, :]) ** 2).sum(-1)  # (C,C)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, axis=1)[:, :k]                            # (C,k) nearest idxs
    nbr_mean = x[nn].mean(axis=1)                                 # (C,T)
    return x - nbr_mean


def _car(x: np.ndarray) -> np.ndarray:
    """Common-average reference. x: (C, T) -> (C, T)."""
    return x - x.mean(axis=0, keepdims=True)


def _resample_to_pretrain(x: np.ndarray, fs: float) -> np.ndarray:
    """Resample (..., T) from fs -> PRETRAIN_FS using scipy.signal.resample (Fourier)."""
    if int(round(fs)) == PRETRAIN_FS:
        return x
    from scipy import signal as sps
    T = x.shape[-1]
    n_out = int(round(T * PRETRAIN_FS / float(fs)))
    return sps.resample(x, n_out, axis=-1)


def _stft_spec(x_1d: np.ndarray) -> np.ndarray:
    """BrainBERT STFT for a single 1-D signal already at PRETRAIN_FS.

    Reproduces preprocessors/stft.py get_stft(..., normalizing='zscore'):
      |STFT| -> keep first 40 freq bins -> z-score per-freq over time
      -> drop first/last 10 frames -> transpose to (T_frames, 40).
    Returns float32 (T_frames, 40).
    """
    from scipy import signal as sps
    f, t, Zxx = sps.stft(
        x_1d, PRETRAIN_FS, nperseg=STFT_NPERSEG, noverlap=STFT_NOVERLAP,
        return_onesided=True,
    )
    Zxx = np.abs(Zxx[:STFT_FREQ_CUTOFF])             # (40, T_frames)
    # per-freq z-score over time (matches stft.py zscore(axis=-1))
    mn = Zxx.mean(axis=-1, keepdims=True)
    sd = Zxx.std(axis=-1, keepdims=True)
    sd = np.where(sd == 0, 1.0, sd)
    Zxx = (Zxx - mn) / sd
    if Zxx.std() == 0:
        Zxx = np.ones_like(Zxx)
    # boundary clip (stft.py: Zxx[:, 10:-10]); guard against too-short windows
    if Zxx.shape[1] > 2 * STFT_ZSCORE_CLIP:
        Zxx = Zxx[:, STFT_ZSCORE_CLIP:-STFT_ZSCORE_CLIP]
    Zxx = np.nan_to_num(Zxx, nan=0.0)
    return np.transpose(Zxx).astype(np.float32)      # (T_frames, 40)


def _min_samples_for_frames(min_frames: int) -> int:
    """Min #samples (at PRETRAIN_FS) so STFT yields >= min_frames after clip+pool."""
    need_frames = min_frames + 2 * STFT_ZSCORE_CLIP
    # scipy stft frame count ~= 1 + (n - nperseg) / (nperseg - noverlap), padded.
    hop = STFT_NPERSEG - STFT_NOVERLAP
    return STFT_NPERSEG + (need_frames - 1) * hop


def preprocess_window(
    x_cw: np.ndarray,
    fs: float,
    xyz_mm: Optional[np.ndarray],
    reref: str,
) -> np.ndarray:
    """Full BrainBERT preprocessing for ONE window of all electrodes.

    x_cw: (C, T) raw ECoG at `fs`. Returns (C, T_frames, 40) STFT specs.
    Order matches the repo: notch -> re-ref -> per-channel z-score (raw) ->
    resample to 2048 Hz -> per-channel STFT (which z-scores again per-freq).
    """
    x = np.asarray(x_cw, dtype=np.float64)
    # 1) notch mains harmonics below Nyquist (pre-resample, on native fs)
    x = _notch_filter(x, fs)
    # 2) re-reference
    if reref == "laplacian_xyz":
        x = _laplacian_xyz(x, xyz_mm, k=2)
    elif reref == "car":
        x = _car(x)
    elif reref == "none":
        pass
    else:
        raise ValueError(f"reref must be laplacian_xyz|car|none, got {reref!r}")
    # 3) per-channel raw z-score over time
    x = _zscore_time(x)
    # 4) reflect-pad so the window survives STFT clip+pool, then resample to 2048 Hz
    x = _resample_to_pretrain(x, fs)               # (C, T2048)
    need = _min_samples_for_frames(2 * POOL_HALF + 1)
    if x.shape[-1] < need:
        pad = need - x.shape[-1]
        x = np.pad(x, ((0, 0), (pad // 2, pad - pad // 2)), mode="reflect")
    # 5) per-channel STFT
    specs = np.stack([_stft_spec(x[c]) for c in range(x.shape[0])], axis=0)
    return specs                                    # (C, T_frames, 40)


# ----------------------------------------------------------------------------
# Model loading (the demo / SpecPretrained code path)



# ==========================================================================
# BrainBERT — frozen per-electrode spectrogram encoder
# ==========================================================================

def _maybe_unbundle(weights_path: str) -> None:
    """Repair the gdown download in place if it is the ZIP *bundle*, not a ckpt.

    GOTCHA: Google Drive id 14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q downloads a ZIP that
    *contains* two checkpoints (`stft_large_pretrained.pth`, `superlet_large_pretrained.pth`)
    rather than a torch file. A real torch .pth zip has an `archive/` entry; the
    bundle's entries are the two .pth filenames. If we detect the bundle at
    `weights_path`, extract `stft_large_pretrained.pth` and replace the file.
    """
    import zipfile
    if not zipfile.is_zipfile(weights_path):
        return
    with zipfile.ZipFile(weights_path) as z:
        names = z.namelist()
        is_torch_ckpt = any(n.startswith("archive/") or n.endswith("data.pkl")
                            for n in names)
        if is_torch_ckpt:
            return  # it's a normal torch zip checkpoint, leave it
        inner = "stft_large_pretrained.pth"
        if inner not in names:
            return  # unknown zip; let torch.load raise a clear error
        dst_dir = os.path.dirname(weights_path)
        z.extract(inner, dst_dir)
    extracted = os.path.join(os.path.dirname(weights_path), inner)
    os.replace(extracted, weights_path)  # overwrite the bundle with the real ckpt


def load_brainbert(weights_path: str = "", device: str = "cuda"):
    """Load the frozen pretrained BrainBERT (demo cell 9-10 path).

        init_state = torch.load(ckpt)
        upstream   = models.build_model(init_state["model_cfg"])
        upstream.load_weights(init_state["model"])

    Returns the eval()-mode model on `device`. Raises FileNotFoundError if the
    checkpoint is missing; see the module docstring for the download source.
    """
    import torch
    weights_path = weights_path or brainbert_weights()
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"BrainBERT weights not found at {weights_path}\n"
            f"Download with:\n"
            f"  mkdir -p {os.path.dirname(weights_path)}\n"
            f"  gdown 14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q -O {weights_path}\n"
            f"(the download is a ZIP bundle of stft+superlet; this loader "
            f"auto-extracts the stft checkpoint.)"
        )
    _maybe_unbundle(weights_path)
    # NOTE: the checkpoint stores an OmegaConf `model_cfg` object, so torch>=2.6's
    # default weights_only=True rejects it -> must pass weights_only=False.
    init_state = torch.load(weights_path, map_location="cpu", weights_only=False)
    upstream_cfg = init_state["model_cfg"]            # OmegaConf object in the ckpt
    if getattr(upstream_cfg, "name", None) == "debug_model":
        upstream_cfg.name = "masked_tf_model"        # (spec_pretrained.py does this)
    with _import_scope(_BB_COLLIDE, brainbert_repo()):
        import models  # noqa: E402  (BrainBERT model registry; needs torch only)
        model = models.build_model(upstream_cfg)
    model.load_weights(init_state["model"])
    model.eval().to(device)
    for p in model.parameters():                     # frozen extractor
        p.requires_grad_(False)
    return model


def _build_random_model(device: str = "cpu"):
    """Build an *un-trained* large model (for --self_test without weights/omegaconf)."""
    cfg = SimpleNamespace(
        name="masked_tf_model", hidden_dim=HIDDEN_DIM, layer_dim_feedforward=3072,
        layer_activation="gelu", nhead=12, encoder_num_layers=6, input_dim=INPUT_DIM,
    )
    with _import_scope(_BB_COLLIDE, brainbert_repo()):
        import models  # noqa: E402
        model = models.build_model(cfg)
    model.eval().to(device)
    return model


# ----------------------------------------------------------------------------
# Public API


def brainbert_embeddings(
    x_raw: np.ndarray,
    fs: float = 128.0,
    xyz_mm: Optional[np.ndarray] = None,
    device: str = "cuda",
    reref: str = "laplacian_xyz",
    pool: str = "default",
    batch_size: int = 256,
    model=None,
) -> np.ndarray:
    """Extract frozen BrainBERT embeddings for our (N, C, T) raw ECoG.

    Args:
      x_raw    : (N, C, T) raw ECoG (e.g. X_raw_tr (N, C, 128)).
      fs       : sampling rate of x_raw (Hz). Our data: 128.
      xyz_mm   : (C, 3) electrode coords in mm. REQUIRED if reref='laplacian_xyz'.
      device   : torch device.
      reref    : 'laplacian_xyz' (faithful analogue, needs xyz) | 'car' | 'none'.
      pool     : 'default' -> BrainBERT center-10-frame mean -> (N, C, 768);
                 'mean'    -> mean over ALL frames           -> (N, C, 768);
                 'none'    -> keep frames                    -> (N, C, T_frames, 768).
      batch_size: #(window,electrode) STFT specs per model forward.
      model    : preloaded model (else load_brainbert is called).

    Returns:
      pool in {default, mean}: (N, C, 768).  Flatten to (N, C*768) or mean over
        the C axis to (N, 768) before a linear/ridge regression head.
      pool == 'none'         : (N, C, T_frames, 768).

    The model is frozen and run under torch.no_grad().
    """
    import torch

    x_raw = np.asarray(x_raw)
    if x_raw.ndim != 3:
        raise ValueError(f"x_raw must be (N, C, T); got {x_raw.shape}")
    N, C, _ = x_raw.shape

    if model is None:
        model = load_brainbert(device=device)

    # 1) Preprocess every window -> list of (C, T_frames, 40); T_frames is constant.
    specs = np.stack(
        [preprocess_window(x_raw[n], fs, xyz_mm, reref) for n in range(N)], axis=0
    )                                                 # (N, C, T_frames, 40)
    T_frames = specs.shape[2]

    # 2) Flatten (N,C) into a batch of single-electrode spectrograms and run model.
    flat = specs.reshape(N * C, T_frames, INPUT_DIM)  # (N*C, T_frames, 40)
    outs = []
    for i in range(0, flat.shape[0], batch_size):
        chunk = torch.from_numpy(flat[i:i + batch_size]).float().to(device)
        mask = torch.zeros(chunk.shape[:2], dtype=torch.bool, device=device)
        with torch.no_grad():
            rep = model.forward(chunk, mask, intermediate_rep=True)  # (b, T_frames, 768)
        outs.append(rep.detach().cpu())
    rep = torch.cat(outs, dim=0).numpy()              # (N*C, T_frames, 768)

    # 3) Pool.
    if pool == "default":                             # BrainBERT center-10 mean
        mid = T_frames // 2
        lo, hi = max(0, mid - POOL_HALF), mid + POOL_HALF
        emb = rep[:, lo:hi].mean(axis=1)              # (N*C, 768)
        return emb.reshape(N, C, HIDDEN_DIM)
    elif pool == "mean":
        emb = rep.mean(axis=1)                        # (N*C, 768)
        return emb.reshape(N, C, HIDDEN_DIM)
    elif pool == "none":
        return rep.reshape(N, C, T_frames, HIDDEN_DIM)
    raise ValueError(f"pool must be default|mean|none, got {pool!r}")


# ----------------------------------------------------------------------------
# Self-test



# ==========================================================================
# Population Transformer — population [CLS] over BrainBERT embeddings
# ==========================================================================

POPT_HF_REPO = "PopulationTransformer/popt_brainbert_stft"
POPT_HF_FILENAME = "pretrained_popt_brainbert_stft.pth"
POPT_HF_CONFIG = "pt_custom_model.yaml"  # also in the repo; same dims as embedded model_cfg

# --- BrainBERT upstream weights (Google Drive) ------------------------------ #
# PopT's per-electrode encoder is BrainBERT STFT-large `stft_large_pretrained.pth`.
# NOT on HF: BrainBERT README points to a Google Drive file.
#   https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view
# Download once (e.g. `gdown 14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q`) and place at
# brainbert_weights() below. This is the SAME weight file the sibling BrainBERT
# adapter needs, so they should share it.
BRAINBERT_GDRIVE_ID = "14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q"
BRAINBERT_WEIGHTS_FILENAME = "stft_large_pretrained.pth"
# Default expected location (overridable via env var). Kept under datasets, not in git.

# Architecture constants (verified from checkpoint state_dict shapes).
POPT_INPUT_DIM = 768   # BrainBERT embedding dim (in_proj: 512x768)
POPT_HIDDEN_DIM = 512  # PopT token / population embedding dim  -> OUTPUT D
POPT_PE_MAX_LEN = 5000  # positional-encoding table length: coords must be in [0, 5000)


# --------------------------------------------------------------------------- #
# Lazy PopT model loader


def download_popt_checkpoint() -> str:
    """Download the PopT checkpoint from HF and return the local path.

    Equivalent CLI:
        huggingface-cli download PopulationTransformer/popt_brainbert_stft \
            pretrained_popt_brainbert_stft.pth
    """
    from huggingface_hub import hf_hub_download

    return hf_hub_download(POPT_HF_REPO, POPT_HF_FILENAME)


def load_popt_model(device: str = "cuda", ckpt_path: Optional[str] = None):
    """Build PtModelCustom from the checkpoint's embedded `model_cfg` and load weights.

    The checkpoint stores both the weights (`["model"]`) and the config
    (`["model_cfg"]`), so we do not need Hydra. Returns an eval-mode model on `device`.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required (conda env: eeg311).")
    from omegaconf import OmegaConf

    if ckpt_path is None:
        ckpt_path = download_popt_checkpoint()
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not (isinstance(ckpt, dict) and "model" in ckpt and "model_cfg" in ckpt):
        raise ValueError(
            f"Unexpected PopT checkpoint format at {ckpt_path}: "
            f"expected dict with 'model' and 'model_cfg' keys."
        )
    cfg = OmegaConf.create(ckpt["model_cfg"])
    with _import_scope(_PT_COLLIDE, popt_repo(), brainbert_repo()):
        import models as popt_models  # registers pt_model_custom
        model = popt_models.build_model(cfg)        # PtModelCustom
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing or unexpected:
        # Should be empty for the official checkpoint; surface anything odd.
        print(f"[extract_popt] load_state_dict missing={missing} unexpected={unexpected}")
    model.eval().to(device)
    return model


# --------------------------------------------------------------------------- #
# Coordinate conversion: continuous MNI mm  ->  integer L/I/P indices in [0, 5000)
# --------------------------------------------------------------------------- #


def mni_mm_to_lip_indices(
    xyz_mm: np.ndarray,
    *,
    max_len: int = POPT_PE_MAX_LEN,
    offset: int = 128,
) -> np.ndarray:
    """Map continuous MNI XYZ (mm) to the integer (L, I, P) index triplet PopT expects.

    PopT/BrainTreebank coords are integer voxel-ish indices fed straight into a
    sinusoidal positional-encoding lookup table of length `max_len`. Our Stanford/
    Ghent coords are continuous MNI millimetres (roughly [-90, 90]). We therefore:

      1. Re-orient axes to PopT's L/I/P convention:
            L (Left)     = -X   (MNI +X is Right; L grows leftward)
            I (Inferior) = -Z   (MNI +Z is Superior; I grows downward)
            P (Posterior)= -Y   (MNI +Y is Anterior; P grows backward)
      2. Round to the nearest integer millimetre and add `offset` (default 128) so
         the typical [-90, 90] mm range becomes non-negative integers ~[38, 218].
      3. Clip to [0, max_len-1] for safety.

    NOTE: This is an APPROXIMATE alignment, not a recovery of BrainTreebank's exact
    FreeSurfer voxel indices (those are unavailable for our subjects). It preserves
    relative inter-electrode geometry at ~1 mm resolution, which is what PopT's
    spatial positional encoding actually uses. See POPT_SETUP.md "Fairness risks".

    Args:
        xyz_mm: (C, 3) float array, MNI millimetres, columns = [X, Y, Z].
    Returns:
        (C, 3) int64 array, columns = [L, I, P], values in [0, max_len-1].
    """
    xyz = np.asarray(xyz_mm, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz_mm must be (C,3), got {xyz.shape}")
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    L = -x
    I = -z
    P = -y
    lip = np.stack([L, I, P], axis=1)
    lip = np.rint(lip).astype(np.int64) + int(offset)
    lip = np.clip(lip, 0, max_len - 1)
    return lip


# --------------------------------------------------------------------------- #
# Per-electrode BrainBERT embeddings (sibling adapter)
# --------------------------------------------------------------------------- #
def _brainbert_per_electrode_embeddings(
    x_raw: np.ndarray, fs: int, device: str
) -> np.ndarray:
    """Get per-electrode BrainBERT embeddings (N, C, 768) via the sibling adapter.

    Expects `external_fms/adapters/extract_brainbert.py` to expose
    `extract_embeddings(x_raw, fs, device)`.

    TODO(sibling): if the BrainBERT adapter's signature/return shape differs from the
    assumption below, adjust this shim. Accepted return shapes:
        (N, C, 768)  -> used as-is
        (C, 768)     -> treated as a single window (N==1)
        (N*C, 768)   -> reshaped if `_n`/`_c` recoverable (not assumed here)
    """
    try:
        from external_fms.adapters.extract_brainbert import extract_embeddings  # type: ignore
    except Exception:
        try:
            # Fallback if adapters dir is on sys.path directly.
            from extract_brainbert import extract_embeddings  # type: ignore
        except Exception as e:
            raise ImportError(
                "TODO: BrainBERT adapter not found. Expected "
                "`external_fms/adapters/extract_brainbert.py` exposing "
                "`extract_embeddings(x_raw, fs, device) -> (N, C, 768)`. "
                f"Underlying error: {e}"
            )

    emb = extract_embeddings(x_raw, fs, device)
    if torch is not None and isinstance(emb, torch.Tensor):
        emb = emb.detach().cpu().numpy()
    emb = np.asarray(emb)
    if emb.ndim == 2:  # (C, 768) single window
        emb = emb[None, ...]
    if emb.ndim != 3 or emb.shape[-1] != POPT_INPUT_DIM:
        raise ValueError(
            f"BrainBERT adapter returned shape {emb.shape}; expected (N, C, {POPT_INPUT_DIM})."
        )
    return emb.astype(np.float32, copy=False)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def popt_embeddings(
    x_raw: np.ndarray,
    xyz: np.ndarray,
    fs: int = 128,
    device: Optional[str] = None,
    *,
    coords_are_mni_mm: bool = True,
    batch_size: int = 64,
    per_electrode_embeddings: Optional[np.ndarray] = None,
    popt_model=None,
    coord_offset: int = 128,
) -> np.ndarray:
    """Run the BrainBERT -> PopT pipeline and return per-window population embeddings.

    Args:
        x_raw: (N, C, T) raw ECoG. N windows, C electrodes, T samples (e.g. 128 @ 128 Hz).
        xyz:   (C, 3) electrode coordinates. By default interpreted as MNI millimetres
               and converted to PopT's integer L/I/P indices via `mni_mm_to_lip_indices`.
               Set `coords_are_mni_mm=False` to pass already-integer L/I/P indices.
        fs:    sampling rate of x_raw (default 128). Forwarded to the BrainBERT adapter,
               which is responsible for the 2048 Hz pretrain mismatch (resample/STFT).
        device: "cuda"/"cpu" (default: cuda if available).
        coords_are_mni_mm: convert xyz from mm to LIP indices when True.
        batch_size: PopT forward batch size over windows.
        per_electrode_embeddings: optional precomputed (N, C, 768) to skip BrainBERT.
        popt_model: optional preloaded PtModelCustom (else loaded from HF).
        coord_offset: integer offset added in mm->index conversion (see that fn).

    Returns:
        (N, POPT_HIDDEN_DIM) == (N, 512) float32 population embeddings ([CLS] token).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required (conda env: eeg311).")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x_raw = np.asarray(x_raw)
    if x_raw.ndim != 3:
        raise ValueError(f"x_raw must be (N, C, T), got {x_raw.shape}")
    N, C, _T = x_raw.shape

    xyz = np.asarray(xyz)
    if xyz.ndim != 2 or xyz.shape != (C, 3):
        raise ValueError(f"xyz must be (C, 3) matching x_raw C={C}, got {xyz.shape}")

    # 1) per-electrode BrainBERT embeddings -> (N, C, 768)
    if per_electrode_embeddings is not None:
        emb = np.asarray(per_electrode_embeddings, dtype=np.float32)
        if emb.shape != (N, C, POPT_INPUT_DIM):
            raise ValueError(
                f"per_electrode_embeddings must be {(N, C, POPT_INPUT_DIM)}, got {emb.shape}"
            )
    else:
        emb = _brainbert_per_electrode_embeddings(x_raw, fs, device)
        if emb.shape[0] != N or emb.shape[1] != C:
            raise ValueError(
                f"BrainBERT embeddings {emb.shape} do not match (N={N}, C={C})."
            )

    # 2) coords -> integer L/I/P indices in [0, 5000)
    if coords_are_mni_mm:
        lip = mni_mm_to_lip_indices(xyz, offset=coord_offset)
    else:
        lip = np.rint(np.asarray(xyz)).astype(np.int64)
        lip = np.clip(lip, 0, POPT_PE_MAX_LEN - 1)

    # 3) PopT model
    if popt_model is None:
        popt_model = load_popt_model(device=device)

    coords_t = torch.as_tensor(lip, dtype=torch.long, device=device)        # (C, 3)
    coords_t = coords_t.unsqueeze(0)                                         # (1, C, 3)
    seq_id_base = torch.zeros(1, C, dtype=torch.long, device=device)        # (1, C)

    outs = np.empty((N, POPT_HIDDEN_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, N, batch_size):
            stop = min(start + batch_size, N)
            b = stop - start
            chunk = torch.as_tensor(emb[start:stop], dtype=torch.float32, device=device)  # (b, C, 768)
            # prepend [CLS] token of ones (matches dataset: torch.ones(1, embed_dim))
            cls = torch.ones(b, 1, POPT_INPUT_DIM, dtype=torch.float32, device=device)
            inputs = torch.cat([cls, chunk], dim=1)                          # (b, 1+C, 768)
            # all-valid mask (we never pad electrodes within a subject)
            pad_mask = torch.zeros(b, 1 + C, dtype=torch.bool, device=device)
            coords = coords_t.expand(b, C, 3)
            seq_id = seq_id_base.expand(b, C)
            rep = popt_model.forward(
                inputs, pad_mask, (coords, seq_id), intermediate_rep=True
            )                                                               # (b, 1+C, 512)
            cls_out = rep[:, 0, :]                                           # (b, 512)
            outs[start:stop] = cls_out.detach().cpu().numpy()
    return outs


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #



# ==========================================================================
# Brant — 505 M-parameter iEEG foundation model
# ==========================================================================

BRANT_HF_REPO = "Daoze/Brant"
BRANT_SRC_SUBDIR = "Brant_src"                       # contains the official model code
BRANT_PRE_MODEL = "Brant_src/pretrain/pre_model.py"  # official architecture
BRANT_PRE_UTILS = "Brant_src/pretrain/pre_utils.py"  # official PSD power calc
BRANT_EMB_FN = "Brant_src/utils.py:get_emb"          # official embedding helper (per model card)

# Local checkpoint path (set after you download the .pth from the Google-Drive link).
# Override with env var BRANT_WEIGHTS or the --weights CLI flag.
# Resolved lazily through brant_weights(); see the module docstring.
# Approx checkpoint size: Brant is >500M params (505.69M in Fig.1). fp32 -> ~2.0 GB.
BRANT_PARAM_COUNT_M = 505.69

# ----------------------------------------------------------------------------
# Brant fixed hyper-parameters (from the paper — do NOT change for a fair baseline)
# ----------------------------------------------------------------------------
BRANT_FS = 250            # Hz, pretrain sampling rate
BRANT_PATCH_LEN = 1500    # M, samples per patch (6 s @ 250 Hz)
BRANT_PATCH_STRIDE = 1500 # S, non-overlapping (stride == patch len)
BRANT_D_MODEL = 2048      # D, embedding dim (temporal & spatial encoders)
BRANT_TEMPORAL_LAYERS = 12
BRANT_SPATIAL_LAYERS = 5
BRANT_FFN = 3072
BRANT_HEADS = 16
BRANT_N_BANDS = 8

# 8 PSD bands (Hz) — exactly as in the paper "Frequency encoding" paragraph.
BRANT_BANDS_HZ = (
    (4.0, 8.0),     # theta
    (8.0, 13.0),    # alpha
    (13.0, 30.0),   # beta
    (30.0, 50.0),   # gamma1
    (50.0, 70.0),   # gamma2
    (70.0, 90.0),   # gamma3
    (90.0, 110.0),  # gamma4
    (110.0, 128.0), # gamma5
)

# Flip to True once the official Brant_src/pretrain/pre_model.py + pre_utils.py are
# placed alongside this file (or importable) so the released checkpoint loads exactly.


def resample_to_brant_fs(x: np.ndarray, fs: int) -> np.ndarray:
    """Resample (N, C, T) from `fs` Hz to BRANT_FS (250 Hz) along the time axis.

    Uses scipy.signal.resample (Fourier method). This is the documented fairness
    bridge for fs mismatch — Brant was pretrained at 250 Hz; our data is 128 Hz.
    """
    if fs == BRANT_FS:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample

    N, C, T = x.shape
    T_new = int(round(T * BRANT_FS / fs))
    out = resample(x.astype(np.float64), T_new, axis=-1)
    return out.astype(np.float32)


def make_patches(x_250: np.ndarray) -> np.ndarray:
    """Cut (N, C, T) @250Hz into Brant patches -> (N, L, C, M).

    M = BRANT_PATCH_LEN (1500), stride S = BRANT_PATCH_STRIDE (1500).
    L = floor((T - M) / S) + 1 if T >= M, else 0.

    NOTE: Our 1-s windows resampled to 250 Hz are only 250 samples < M=1500, so a
    *single* window yields L=0 patches. We zero-pad the tail up to one full patch
    (M samples) so that L >= 1. This pads ~1250 samples of zeros per 6-s patch
    (see FAIRNESS — this is the central preprocessing risk for short-window data).
    """
    N, C, T = x_250.shape
    M, S = BRANT_PATCH_LEN, BRANT_PATCH_STRIDE
    if T < M:
        pad = M - T
        x_250 = np.pad(x_250, ((0, 0), (0, 0), (0, pad)), mode="constant")
        T = M
    L = (T - M) // S + 1
    patches = np.empty((N, L, C, M), dtype=np.float32)
    for li in range(L):
        s = li * S
        patches[:, li] = x_250[:, :, s : s + M]
    return patches


def band_log_psd(patches: np.ndarray, fs: int = BRANT_FS) -> np.ndarray:
    """Compute the 8-band log-PSD power weights for every patch (Eq.1).

    Input  : patches (N, L, C, M)
    Output : (N, L, C, 8)  =  P(i) = log( sum_{w in band(i)} PSD(w) ) for i=1..8.

    # TODO(official-code): the paper (Eq.1) only specifies P(i)=log sum PSD(w);
    # the exact PSD estimator (Welch window length / overlap / detrend) is defined
    # in Brant_src/pretrain/pre_utils.py (GATED). We use scipy.signal.welch with a
    # full-patch window (nperseg=M) Hann, which is the standard single-segment PSD
    # and is the most faithful default for a fixed-length patch. Confirm against
    # pre_utils.py once HF access is granted.
    """
    from scipy.signal import welch

    N, L, C, M = patches.shape
    flat = patches.reshape(-1, M)
    # nperseg = M -> single Hann-windowed periodogram per patch (no averaging).
    freqs, psd = welch(flat, fs=fs, nperseg=M, noverlap=0, detrend="constant", axis=-1)
    out = np.empty((flat.shape[0], BRANT_N_BANDS), dtype=np.float32)
    eps = 1e-12
    for i, (lo, hi) in enumerate(BRANT_BANDS_HZ):
        m = (freqs >= lo) & (freqs < hi)
        band_sum = psd[:, m].sum(axis=-1)
        out[:, i] = np.log(band_sum + eps)
    return out.reshape(N, L, C, BRANT_N_BANDS)


# ============================================================================
# Reconstructed Brant model (paper-faithful; load-compatible pending official code)
# ============================================================================
if torch is not None:

    class _TransformerEncoderStack(nn.Module):
        def __init__(self, depth: int):
            super().__init__()
            layer = nn.TransformerEncoderLayer(
                d_model=BRANT_D_MODEL,
                nhead=BRANT_HEADS,
                dim_feedforward=BRANT_FFN,
                batch_first=True,
                norm_first=False,
                activation="gelu",
            )
            self.enc = nn.TransformerEncoder(layer, num_layers=depth)

        def forward(self, x):  # x: (B, seq, D)
            return self.enc(x)

    class BrantReconstructed(nn.Module):
        """Paper-faithful Brant encoder (Eqs.1-3, Sec.2).

        Forward: patches (B, L, C, M) + band_logpsd (B, L, C, 8) -> z (B, L, C, D).

        WARNING: random-init unless the official checkpoint is mapped onto these
        modules. Used by --self_test for shape validation only.
        # TODO(official-code): align module names with pre_model.py so the released
        # state_dict loads. Likely the official names differ (e.g. patch_embed,
        # band_embed, temporal_transformer, spatial_transformer).
        """

        def __init__(self):
            super().__init__()
            self.w_proj = nn.Linear(BRANT_PATCH_LEN, BRANT_D_MODEL)        # W_proj (Eq.3)
            self.band_embed = nn.Parameter(torch.randn(BRANT_N_BANDS, BRANT_D_MODEL))  # f_i (Eq.2)
            # W_pos is (L, D); L is data-dependent, so use a generous max and slice.
            self._max_L = 64
            self.w_pos = nn.Parameter(torch.randn(self._max_L, BRANT_D_MODEL))
            self.temporal_encoder = _TransformerEncoderStack(BRANT_TEMPORAL_LAYERS)
            self.spatial_encoder = _TransformerEncoderStack(BRANT_SPATIAL_LAYERS)

        def freq_encoding(self, band_logpsd):  # (B, L, C, 8) -> (B, L, C, D)
            w = F.softmax(band_logpsd, dim=-1)            # softmax over 8 bands (Eq.2)
            return torch.einsum("blck,kd->blcd", w, self.band_embed)

        def forward(self, patches, band_logpsd):
            B, L, C, M = patches.shape
            assert L <= self._max_L, f"L={L} exceeds max positional length {self._max_L}"
            proj = self.w_proj(patches)                   # (B, L, C, D)
            pos = self.w_pos[:L].view(1, L, 1, BRANT_D_MODEL)
            freq = self.freq_encoding(band_logpsd)        # (B, L, C, D)
            h_in = proj + pos + freq                      # input encoding (Eq.3)

            # Temporal encoder: attend over the L patches within each channel.
            ht = h_in.permute(0, 2, 1, 3).reshape(B * C, L, BRANT_D_MODEL)
            ht = self.temporal_encoder(ht)
            ht = ht.reshape(B, C, L, BRANT_D_MODEL).permute(0, 2, 1, 3)  # (B, L, C, D)

            # Spatial encoder: for each time index, attend over the C channels.
            hs = ht.reshape(B * L, C, BRANT_D_MODEL)
            hs = self.spatial_encoder(hs)
            z = hs.reshape(B, L, C, BRANT_D_MODEL)        # final repr z (Sec.2)
            return z


def _load_brant_model(weights: str, device: str, allow_random_init: bool = False):
    """Build Brant and load the pretrained checkpoint.

    If `_USE_OFFICIAL_CODE`, import the official pre_model.py (drop it next to this
    file or make it importable) and load the released state_dict exactly. Otherwise
    fall back to the reconstructed module (random init unless a compatible state_dict
    is supplied) — only valid for shape testing.

    Weights-absent policy: a REAL run (allow_random_init=False, the default) HARD-FAILS
    with a clear FileNotFoundError so random-init numbers can never be reported as a
    baseline. Only the shape/wiring paths (self_test, or the runner's explicit
    --debug_brant_random_init) pass allow_random_init=True to permit random init.
    """
    weights = weights or brant_weights()
    if _USE_OFFICIAL_CODE:
        # TODO(official-code): from Brant_src.pretrain.pre_model import Brant (or similar)
        #   model = Brant(**cfg); ckpt = torch.load(weights, map_location="cpu")
        #   model.load_state_dict(ckpt["model"] or ckpt)  # exact key TBD
        raise NotImplementedError(
            "Set up official Brant_src/pretrain/pre_model.py first; see BRANT_SETUP.md."
        )
    have_weights = bool(weights) and os.path.exists(weights)
    # Fail BEFORE building the ~505M-param model so a real run with an absent
    # checkpoint errors fast (and never silently reports random-init numbers).
    if not have_weights and not allow_random_init:
        raise FileNotFoundError(
            f"Brant weights not found at '{weights}'; see external_fms/BRANT_SETUP.md "
            f"to obtain the gated checkpoint (HF Daoze/Brant + Google-Drive .pth) and "
            f"set --weights / $BRANT_WEIGHTS. Pass allow_random_init=True (runner: "
            f"--debug_brant_random_init) ONLY for shape validation, never for reported "
            f"numbers."
        )
    state = None
    if have_weights:
        # torch.load BEFORE building the ~505M-param model so a corrupt / wrong-format
        # checkpoint also fails fast. NOTE: the released Brant weights ship as a zip
        # of TWO separate encoder .pt files (pre_trained_weights/time_encoder.pt +
        # channel_encoder.pt), NOT a single state_dict — so a plain torch.load of that
        # .pth raises here. Real loading of the two-encoder format requires the
        # official Brant_src/pretrain/pre_model.py (set _USE_OFFICIAL_CODE=True); see
        # BRANT_SETUP.md.
        try:
            ckpt = torch.load(weights, map_location="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to torch.load Brant checkpoint '{weights}' ({type(e).__name__}: "
                f"{e}). The released weights are a zip of two encoder .pt files "
                f"(time_encoder.pt + channel_encoder.pt), not a single state_dict, and "
                f"the reconstructed module is not key-compatible — set "
                f"_USE_OFFICIAL_CODE=True with the official pre_model.py to load them "
                f"(see external_fms/BRANT_SETUP.md). For a shape check only, pass "
                f"allow_random_init=True (runner: --debug_brant_random_init)."
            ) from e
        state = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt

    model = BrantReconstructed().to(device).eval()
    if state is not None:
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[extract_brant] WARNING load_state_dict strict=False: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys. "
                f"Reconstructed module is NOT key-compatible with the official "
                f"checkpoint — set _USE_OFFICIAL_CODE=True and drop in the official "
                f"Brant_src/pretrain/pre_model.py (see BRANT_SETUP.md). Until then these "
                f"embeddings are effectively RANDOM-INIT and NOT a valid baseline."
            )
    else:
        print(
            f"[extract_brant] WARNING: no weights at '{weights}'. Using RANDOM-INIT "
            f"Brant — embeddings are NOT a valid baseline (wiring/shape check only)."
        )
    return model


# ============================================================================
# Public API
# ============================================================================
@torch.no_grad() if torch is not None else (lambda f: f)
def brant_embeddings(
    x_raw: np.ndarray,
    fs: int = 128,
    device: Optional[str] = None,
    weights: str = "",
    batch_size: int = 8,
    allow_random_init: bool = False,
) -> np.ndarray:
    """Extract frozen-Brant embeddings from our ECoG windows.

    Args:
        x_raw : (N, C, T) raw ECoG (Stanford T=128 @128Hz; Ghent C-chan @128Hz).
        fs    : sampling rate of x_raw (default 128). Resampled to 250 Hz internally.
        device: "cuda" / "cpu" (auto if None).
        weights: path to the pretrained Brant .pth (Google-Drive download).
        batch_size: windows per forward pass (Brant is large; keep small).
        allow_random_init: if True, permit RANDOM-INIT Brant when `weights` is absent
            (shape/wiring validation only). Default False -> a missing checkpoint is a
            hard FileNotFoundError so random numbers can't be reported as a baseline.

    Returns:
        z : np.ndarray, shape (N, L, C, 2048).  L = #patches/window (1 for our 1-s
            windows after 250 Hz resample + tail-pad). Pool with `pool_embeddings`.
    """
    if torch is None:
        raise RuntimeError("PyTorch not available; cannot run Brant.")
    assert x_raw.ndim == 3, f"expected (N,C,T), got {x_raw.shape}"
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    x250 = resample_to_brant_fs(x_raw, fs)        # (N, C, T')
    patches = make_patches(x250)                  # (N, L, C, M)
    blogpsd = band_log_psd(patches)               # (N, L, C, 8)

    model = _load_brant_model(weights, device, allow_random_init=allow_random_init)

    N = patches.shape[0]
    outs = []
    for s in range(0, N, batch_size):
        e = min(s + batch_size, N)
        p = torch.from_numpy(patches[s:e]).to(device)
        b = torch.from_numpy(blogpsd[s:e]).to(device)
        z = model(p, b)                           # (B, L, C, D)
        outs.append(z.float().cpu().numpy())
    return np.concatenate(outs, axis=0)           # (N, L, C, 2048)


def pool_embeddings(z: np.ndarray, mode: str = "mean_lc") -> np.ndarray:
    """Pool (N, L, C, D) Brant embeddings for a regression head.

    mode:
      "mean_lc"  -> (N, D)      mean over patches and channels (default).
      "mean_l"   -> (N, C, D)   mean over patches only (channel-aware head).
      "flatten"  -> (N, L*C*D)  no pooling (large; only for tiny C).
    """
    if mode == "mean_lc":
        return z.mean(axis=(1, 2))
    if mode == "mean_l":
        return z.mean(axis=1)
    if mode == "flatten":
        N = z.shape[0]
        return z.reshape(N, -1)
    raise ValueError(f"unknown pool mode {mode}")


# ============================================================================
# Self test
