"""Adapters for the three intracranial foundation models CORTEG is compared against.

BrainBERT, the Population Transformer (PopT) and Brant are third-party models.
Neither their code nor their weights are redistributed here -- BrainBERT's
repository carries no LICENSE at all -- so each is reached through a clone and a
checkpoint you provide, named by environment variable:

    BrainBERT   BRAINBERT_REPO     git clone https://github.com/czlwang/BrainBERT
                BRAINBERT_WEIGHTS  stft_large_pretrained.pth, from the Google Drive
                                   link in that repository's README
    PopT        POPT_REPO          git clone https://github.com/czlwang/PopulationTransformer
                POPT_WEIGHTS       optional: a local pretrained_popt_brainbert_stft.pth.
                                   Unset, the same file is fetched from
                                   huggingface.co/PopulationTransformer/popt_brainbert_stft,
                                   which is where the paper's copy came from
                                   (PopT also needs BRAINBERT_REPO: it is defined
                                   over frozen BrainBERT embeddings)
    Brant       BRANT_SRC          Brant_src/ from huggingface.co/Daoze/Brant
                BRANT_WEIGHTS      the directory holding time_encoder.pt and
                                   channel_encoder.pt, linked from that model card

Two Python packages are needed beyond the core install. The BrainBERT and PopT
checkpoints both pickle an OmegaConf config, so neither loads without
``omegaconf``; the PopT download also uses ``huggingface_hub``. Both come with
``pip install -e .[ieeg-fm]``.

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
from typing import Optional, Tuple

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


def popt_weights() -> Optional[str]:
    """A local PopT checkpoint from ``POPT_WEIGHTS``, or None when it is unset.

    Unlike the other accessors, unset is not an error: PopT has a public
    download (``download_popt_checkpoint``), and that is the file the paper
    used. Set but pointing nowhere IS an error, so a typo cannot silently turn
    an offline run into a network fetch.
    """
    if not os.environ.get("POPT_WEIGHTS"):
        return None
    return _env_path("POPT_WEIGHTS", "The PopT checkpoint named by POPT_WEIGHTS")


def brant_src_dir() -> str:
    return _env_path("BRANT_SRC", "Brant_src/ from the Brant model card")


def brant_weights_dir() -> str:
    """Directory holding time_encoder.pt and channel_encoder.pt.

    The released weights are two separate encoder files, not one state_dict.
    """
    return _env_path("BRANT_WEIGHTS", "The Brant weights directory "
                                      "(time_encoder.pt + channel_encoder.pt)")


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


def _resample_factors(fs_in: int, fs_out: int):
    """Integer (up, down) such that fs_in * up / down == fs_out."""
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    return int(fs_out) // g, int(fs_in) // g


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


# ==========================================================================
# BrainBERT — frozen per-electrode spectrogram encoder
# ==========================================================================

# The weights are not on Hugging Face; BrainBERT's README points to this Google
# Drive file. PopT's per-electrode encoder is the same checkpoint.
#   https://drive.google.com/file/d/14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q/view
BRAINBERT_GDRIVE_ID = "14ZBOafR7RJ4A6TsurOXjFVMXiVH6Kd_Q"
BRAINBERT_WEIGHTS_FILENAME = "stft_large_pretrained.pth"


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
        inner = BRAINBERT_WEIGHTS_FILENAME
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
            f"  gdown {BRAINBERT_GDRIVE_ID} -O {weights_path}\n"
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
    """Frozen BrainBERT embeddings for (N, C, T) raw intracranial windows.

    Args:
      x_raw    : (N, C, T) raw iEEG windows at `fs`.
      fs       : sampling rate of x_raw (Hz), measured. Pass it explicitly: the
                 windows are resampled to BrainBERT's 2048 Hz from this rate,
                 and every caller in this repository does.
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

    Memory. Each forward batch is pooled as soon as it comes back, and the
    spectrograms are built one window at a time as the batches need them. The
    earlier version kept every unpooled (N*C, T_frames, 768) output and then
    concatenated it, twice that at peak: ~390 GB for the largest BrainTreebank
    subject on Task B, where the pooled result is ~1 GB. The batches, their
    order and the numpy float32 pooling are unchanged, so the output is
    bit-identical to pooling after concatenation (tests/test_btb_fm.py checks
    this). Only pool='none' still holds every frame, because it returns them.
    """
    import torch

    if pool not in ("default", "mean", "none"):
        raise ValueError(f"pool must be default|mean|none, got {pool!r}")
    x_raw = np.asarray(x_raw)
    if x_raw.ndim != 3:
        raise ValueError(f"x_raw must be (N, C, T); got {x_raw.shape}")
    N, C, _ = x_raw.shape
    if N * C == 0:
        raise ValueError(f"x_raw has no windows or no electrodes: {x_raw.shape}")

    if model is None:
        model = load_brainbert(device=device)

    def spec_rows():
        # The rows of the old (N*C, T_frames, 40) array, in the same (n, c) order.
        for n in range(N):
            yield from preprocess_window(x_raw[n], fs, xyz_mm, reref)

    rows, outs = spec_rows(), []
    T_frames = lo = hi = None
    for i in range(0, N * C, batch_size):
        batch = np.stack([next(rows) for _ in range(min(batch_size, N * C - i))])
        if T_frames is None:
            T_frames = batch.shape[1]
            mid = T_frames // 2                       # BrainBERT centre-10 pooling
            lo, hi = max(0, mid - POOL_HALF), mid + POOL_HALF
        elif batch.shape[1] != T_frames:
            raise ValueError(f"windows give {batch.shape[1]} and {T_frames} STFT frames; "
                             f"every window must have the same length")
        chunk = torch.from_numpy(batch).float().to(device)
        mask = torch.zeros(chunk.shape[:2], dtype=torch.bool, device=device)
        with torch.no_grad():
            rep = model.forward(chunk, mask, intermediate_rep=True)  # (b, T_frames, 768)
        rep = rep.detach().cpu().numpy()
        if pool == "default":
            outs.append(rep[:, lo:hi].mean(axis=1))   # (b, 768)
        elif pool == "mean":
            outs.append(rep.mean(axis=1))             # (b, 768)
        else:
            outs.append(rep)                          # (b, T_frames, 768)
    rep = np.concatenate(outs, axis=0)
    if pool == "none":
        return rep.reshape(N, C, T_frames, HIDDEN_DIM)
    return rep.reshape(N, C, HIDDEN_DIM)


# ==========================================================================
# Population Transformer — population [CLS] over BrainBERT embeddings
# ==========================================================================

POPT_HF_REPO = "PopulationTransformer/popt_brainbert_stft"
POPT_HF_FILENAME = "pretrained_popt_brainbert_stft.pth"

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

    The checkpoint is `ckpt_path` if given, else ``POPT_WEIGHTS`` if set, else
    the Hugging Face download. The parameters are left trainable (requires_grad
    as built); the frozen arm runs under torch.no_grad and the fine-tune arms
    choose what to train themselves.
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to run PopT (pip install torch).")
    from omegaconf import OmegaConf

    ckpt_path = ckpt_path or popt_weights() or download_popt_checkpoint()
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
        print(f"[popt] load_state_dict missing={missing} unexpected={unexpected}")
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
    sinusoidal positional-encoding lookup table of length `max_len`. The Stanford
    and Ghent coordinates are continuous MNI millimetres (roughly [-90, 90]), so:

      1. Re-orient axes to PopT's L/I/P convention:
            L (Left)     = -X   (MNI +X is Right; L grows leftward)
            I (Inferior) = -Z   (MNI +Z is Superior; I grows downward)
            P (Posterior)= -Y   (MNI +Y is Anterior; P grows backward)
      2. Round to the nearest integer millimetre and add `offset` (default 128) so
         the typical [-90, 90] mm range becomes non-negative integers ~[38, 218].
      3. Clip to [0, max_len-1] for safety.

    NOTE: This is an APPROXIMATE alignment, not a recovery of BrainTreebank's exact
    FreeSurfer voxel indices (those do not exist for these subjects). It preserves
    relative inter-electrode geometry at ~1 mm resolution, which is what PopT's
    spatial positional encoding actually uses, but the absolute indices are not
    the distribution PopT was pretrained on -- a known handicap for PopT on the
    Stanford and Ghent data that does not arise on BrainTreebank, where the
    native voxel indices are passed with ``coords_are_mni_mm=False``.

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
# Per-electrode BrainBERT embeddings (PopT's input)
# --------------------------------------------------------------------------- #
def _brainbert_per_electrode_embeddings(
    x_raw: np.ndarray, fs: int, device: str, xyz_mm=None, batch_size: int = 64
) -> np.ndarray:
    """Per-electrode BrainBERT embeddings, (N, C, 768).

    PopT is defined over frozen BrainBERT embeddings, so it calls straight into
    the BrainBERT section of this module.
    """
    emb = brainbert_embeddings(
        x_raw, fs=fs, xyz_mm=xyz_mm, device=device,
        reref="laplacian_xyz" if xyz_mm is not None else "car",
        pool="default", batch_size=batch_size)
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
        x_raw: (N, C, T) raw ECoG. N windows, C electrodes, T samples at `fs`.
               May be None when `per_electrode_embeddings` is given, since the raw
               signal is then never read.
        xyz:   (C, 3) electrode coordinates. By default interpreted as MNI millimetres
               and converted to PopT's integer L/I/P indices via `mni_mm_to_lip_indices`.
               Set `coords_are_mni_mm=False` to pass already-integer L/I/P indices.
        fs:    true sampling rate of x_raw; pass it explicitly (every caller in this
               repository does). Forwarded to the BrainBERT adapter, which resamples
               to BrainBERT's 2048 Hz. The paper's Stanford FM results use the native
               1 kHz windows (data/stanford_native.py); the 128 Hz stream of the
               CORTEG pickles has no content above 64 Hz and does not reproduce them.
        device: "cuda"/"cpu" (default: cuda if available).
        coords_are_mni_mm: convert xyz from mm to LIP indices when True.
        batch_size: PopT forward batch size over windows.
        per_electrode_embeddings: optional precomputed (N, C, 768) to skip BrainBERT.
        popt_model: optional preloaded PtModelCustom (else `load_popt_model`).
        coord_offset: integer offset added in mm->index conversion (see that fn).

    Returns:
        (N, POPT_HIDDEN_DIM) == (N, 512) float32 population embeddings ([CLS] token).
    """
    if torch is None:
        raise RuntimeError("PyTorch is required to run PopT (pip install torch).")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if x_raw is None:
        if per_electrode_embeddings is None:
            raise ValueError("give x_raw, per_electrode_embeddings, or both")
        N, C = np.shape(per_electrode_embeddings)[:2]
    else:
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
        # Pass the coordinates through: BrainBERT re-references Laplacian over
        # electrode neighbours, and omitting them silently falls back to CAR,
        # which is not what the published PopT numbers used.
        emb = _brainbert_per_electrode_embeddings(
            x_raw, fs, device, xyz_mm=xyz, batch_size=batch_size)
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


# ==========================================================================
# Brant — 505 M-parameter iEEG foundation model
#
# Uses the OFFICIAL Brant_src/pretrain/pre_model.py. The released weights are two
# separate encoder files (time_encoder.pt + channel_encoder.pt), not one
# state_dict, so nothing but the official module is key-compatible with them. An
# earlier reconstruction of the architecture loaded them with strict=False,
# printed a warning that the result was effectively random-init, and returned the
# model anyway -- exactly the warn-and-continue failure this repository exists to
# remove. That path is gone; a mismatch now raises.
# ==========================================================================

BRANT_FS = 250            # pretrain sampling rate (Hz)
BRANT_PATCH_LEN = 1500    # samples per patch (6.0 s @ 250 Hz), the fixed atomic unit
BRANT_MAX_PATCHES = 15    # pretrain max L == positional-encoding rows
BRANT_D_MODEL = 2048


def load_brant(brant_src: str, weights_dir: str, device: str, n_patches: int = 1):
    """Instantiate the OFFICIAL Brant TimeEncoder + ChannelEncoder and load the
    released pretrained weights strictly (0 missing / 0 unexpected).

    The official code lives in <brant_src> and <brant_src>/pretrain; we add both to
    sys.path and import `pretrain.pre_model`. We do NOT import Brant_src/utils.py
    (it needs torchmetrics); the embedding fn is inlined (get_emb) below.

    `n_patches` (L): the official InputEmbedding.forward adds the FULL (15, 2048)
    positional encoding (`input_emb += self.positional_encoding`), which only
    broadcasts correctly when the data carries exactly 15 patches. For L < 15 the
    documented fix is to use `positional_encoding[:seq_len]`; we implement that by
    slicing the PE PARAMETER to its first L rows at load time (equivalent, and it
    keeps the official forward byte-unchanged). L=1 (default) => one 6 s patch.
    """
    for p in (brant_src, os.path.join(brant_src, "pretrain")):
        if p not in sys.path:
            sys.path.insert(0, p)
    from pretrain.pre_model import TimeEncoder, ChannelEncoder  # official

    et = TimeEncoder(in_dim=1500, d_model=2048, dim_feedforward=3072, seq_len=15,
                     n_layer=12, nhead=16, band_num=8, project_mode='linear',
                     learnable_mask=True)
    ec = ChannelEncoder(out_dim=1500, d_model=2048, dim_feedforward=3072,
                        n_layer=5, nhead=16)

    def _strip(sd):
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        return {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}

    tp = os.path.join(weights_dir, "time_encoder.pt")
    cp = os.path.join(weights_dir, "channel_encoder.pt")
    if not (os.path.exists(tp) and os.path.exists(cp)):
        raise FileNotFoundError(
            f"Brant weights not found under {weights_dir} "
            f"(need time_encoder.pt + channel_encoder.pt)")
    r1 = et.load_state_dict(_strip(torch.load(tp, map_location="cpu")), strict=True)
    r2 = ec.load_state_dict(_strip(torch.load(cp, map_location="cpu")), strict=True)
    n_params = (sum(p.numel() for p in et.parameters())
                + sum(p.numel() for p in ec.parameters())) / 1e6
    print(f"  [brant] strict load OK: time_encoder={r1}, channel_encoder={r2}")
    print(f"  [brant] params={n_params:.2f}M (paper: 505.69M)")

    # PE-slice fix so L != 15 forwards run (official forward adds full (15,2048) PE).
    L = int(n_patches)
    if not (1 <= L <= BRANT_MAX_PATCHES):
        raise ValueError(f"--context_patches must be in [1,{BRANT_MAX_PATCHES}], got {L}")
    pe = et.input_embedding.positional_encoding
    if L != BRANT_MAX_PATCHES:
        et.input_embedding.positional_encoding = nn.Parameter(
            pe.data[:L].clone(), requires_grad=False)
        print(f"  [brant] sliced positional_encoding {tuple(pe.shape)} -> ({L}, 2048) "
              f"(== official PE[:seq_len] for L={L} < 15)")
    et = et.to(device).eval()
    ec = ec.to(device).eval()
    return et, ec


def compute_power(data: np.ndarray, fs: int = BRANT_FS) -> np.ndarray:
    """Brant's exact 8-band log10-PSD power (pre_utils.py::compute_power, verbatim,
    only fixing the intended `scipy.signal` import). data (..., M) -> (..., 8).

    P(i) = log10( sum_{w in band(i)} periodogram_PSD(w) + 1 ), bands split at
    [4,8,13,30,50,70,90,110,128] Hz. Matches the distribution the pretrained
    band-embeddings + softmax expect — using this verbatim is the FAIR choice.
    """
    from scipy import signal as sig
    f, Pxx = sig.periodogram(data, fs)
    f_thres = [4, 8, 13, 30, 50, 70, 90, 110, 128]
    poses = []
    for fi in range(len(f_thres) - 1):
        c1 = np.where(f_thres[fi] < f)[0]
        c2 = np.where(f_thres[fi + 1] >= f)[0]
        poses.append(np.intersect1d(c1, c2))
    ori = Pxx.shape[:-1]
    Pxx = Pxx.reshape(-1, len(f))
    bs = [np.sum(Pxx[:, bp], axis=-1) + 1 for bp in poses]
    bs = [np.log10(x)[:, np.newaxis] for x in bs]
    bs = np.concatenate(bs, axis=-1)
    return bs.reshape(ori + (8,))


def _get_emb(x: torch.Tensor, power: torch.Tensor, et, ec) -> torch.Tensor:
    """Inlined official embedding fn (per the Brant model card / utils.py:get_emb,
    without importing utils.py). x (B,C,15,1500), power (B,C,15,8) -> (B,C,15,2048).

    Differentiable. `get_emb` below is the no-grad wrapper the frozen arms use. A
    caller that trains through Brant must call this one instead: through the
    wrapper the trunk output has grad_fn=None, so every trainable parameter
    would silently receive no gradient. (This release has no such caller; the
    paper's Brant fine-tuning rows are not included.)
    """
    b, c, s, seg = x.shape
    tz = et(mask=None, data=x, power=power, need_mask=False)      # (B*C, 15, 2048)
    d = tz.shape[-1]
    tz = tz.reshape(b, c, s, d).transpose(1, 2).reshape(b * s, c, d)  # (B*15, C, 2048)
    emb, _ = ec(tz)                                               # (B*15, C, 2048)
    return emb.reshape(b, s, c, d).transpose(1, 2)               # (B, C, 15, 2048)


def get_emb(x, power, et, ec):
    """`_get_emb` under torch.no_grad(): the frozen arms' entry point.

    A wrapper rather than a second implementation, so the two cannot drift
    apart. Written as a function rather than ``torch.no_grad()(_get_emb)`` so
    that importing this module does not need torch.
    """
    with torch.no_grad():
        return _get_emb(x, power, et, ec)


# ===========================================================================
# Brant input: the whole recording at 250 Hz, then 6 s patches cut from it
#
# This is how the paper's Brant rows were produced. The recording is resampled
# ONCE, at the exact measured rate, and each patch is a slice of that stream
# ending at the task window's right edge. Resampling each cut window on its own
# (`brant_embeddings`, below) is not equivalent: the 250 Hz grid then starts at
# each window's first native sample (up to ~2 ms off the stream's grid), the
# integer-rounded rate is ~25 ppm off, and the anti-alias filter differs, which
# moves the 110-128 Hz band power most. Measured on one BrainTreebank electrode,
# the two patches differ by a median relative L2 of 0.03-0.05, depending on the
# events sampled.
# ===========================================================================


def brant_resample_ratio(fs: float) -> Tuple[int, int]:
    """(up, down) taking the MEASURED rate `fs` to 250 Hz, as the paper runs did.

    BrainTreebank rates are not integers (e.g. 2049.948 Hz), so rounding fs to
    an integer first (5/41 for that subject) resamples to ~250.006 Hz. The ratio
    is instead built from the rate itself, with its denominator limited to 10000.
    """
    from fractions import Fraction
    fr = Fraction(BRANT_FS, 1) / Fraction(float(fs)).limit_denominator(10000)
    return fr.numerator, fr.denominator


def brant_resample_fir(up: int, down: int) -> np.ndarray:
    """The anti-alias FIR ``scipy.signal.resample_poly`` designs for (up, down).

    Its own default, reproduced so that it can be designed once per recording
    rather than once per channel: with exact ratios `down` reaches ~8e6, the
    filter has ~1.6e8 taps, and designing it is most of the cost. Passing it as
    ``window=`` gives resample_poly bit-identical output (it copies the array,
    then scales it by `up`, exactly as it does its own design).
    """
    from scipy.signal import firwin
    max_rate = max(up, down)
    return firwin(2 * 10 * max_rate + 1, 1.0 / max_rate, window=("kaiser", 5.0))


def brant_stream_at_250(read_channel, n_channels: int, n_samples: int,
                        fs: float) -> np.ndarray:
    """Resample a whole recording to Brant's 250 Hz, one channel at a time.

    Args:
        read_channel: callable, ``read_channel(i)`` -> the full 1-D signal of
            channel i at `fs`. Called once per channel, so peak memory is one
            native channel plus the (T250, C) float32 result.
        n_channels, n_samples: C and the native length every channel has.
        fs: MEASURED sampling rate.

    Returns (T250, C) float32, T250 = ceil(n_samples * up / down).
    """
    from scipy.signal import resample_poly
    up, down = brant_resample_ratio(fs)
    n250 = int(np.ceil(n_samples * up / down))
    fir = brant_resample_fir(up, down) if (up, down) != (1, 1) else None
    out = np.empty((n250, int(n_channels)), dtype=np.float32)
    for ci in range(int(n_channels)):
        sig = np.asarray(read_channel(ci), dtype=np.float64)
        if fir is None:
            r = sig.astype(np.float32)
        else:
            r = resample_poly(sig, up, down, window=fir).astype(np.float32)
        k = min(len(r), n250)
        out[:k, ci] = r[:k]
        out[k:, ci] = 0.0
        del sig, r
    return out


def brant_right_edges_250(right_native, fs: float) -> np.ndarray:
    """Native-rate right-edge sample indices -> 250 Hz stream indices."""
    return np.round(np.asarray(right_native) * (BRANT_FS / float(fs))).astype(np.int64)


def _context_patches(stream250: np.ndarray, end250: int, n_patches: int
                     ) -> Tuple[np.ndarray, bool]:
    """Build one anchor's (C, L, 1500) patch tensor: the L*1500 samples ENDING at
    end250 (exclusive), cut into L temporal 6 s patches (patch 0 oldest, L-1 newest,
    ending exactly at the target). If fewer real samples precede end250, the front is
    LEFT-FILLED by tiling the available real signal (np.pad mode='wrap') — never
    zeros. For the default L=1 this is simply the real 6 s window ending at the
    target. Returns (patches, padded?).
    """
    ctx_samples = int(n_patches) * BRANT_PATCH_LEN
    T = stream250.shape[0]
    end250 = int(min(max(end250, 1), T))
    lo = end250 - ctx_samples
    if lo >= 0:
        ctx = stream250[lo:end250]                                    # (ctx_samples, C)
        padded = False
    else:
        avail = stream250[0:end250]                                   # (<ctx_samples, C)
        pad = ctx_samples - avail.shape[0]
        # tile REAL available signal to fill the front; 'wrap' works for any pad size
        ctx = np.pad(avail, ((pad, 0), (0, 0)), mode="wrap")
        padded = True
    C = ctx.shape[1]
    # (ctx_samples, C) -> (C, ctx_samples) -> (C, L, 1500): patch 0 oldest, L-1 newest
    return np.ascontiguousarray(ctx.T).reshape(C, int(n_patches), BRANT_PATCH_LEN), padded


def brant_stream_embeddings(stream250: np.ndarray, ends250, device: str = "cuda",
                            batch_size: int = 16, n_patches: int = 1,
                            brant_src: str = "", weights_dir: str = "",
                            model=None) -> np.ndarray:
    """Frozen Brant embeddings from a 250 Hz stream, (N, C, 2048).

    Each anchor reads the L*1500 samples ending at ``ends250[i]`` (exclusive),
    cut with `_context_patches`; for L=1 that is the one 6 s patch ending there.
    The channel axis is kept, so per-electrode readouts are possible; patches are
    averaged (a no-op for L=1). This is the paper's BrainTreebank Brant arm:
    the same batching (16 anchors), the same 8-band power and the same encoders.

    Args:
        stream250: (T250, C) float32, from `brant_stream_at_250`.
        ends250: (N,) right-edge indices into the stream, from
            `brant_right_edges_250`.
        model: optional (time_encoder, channel_encoder) from `load_brant`,
            loaded with the same `n_patches`.
    """
    ends = np.asarray(ends250, dtype=np.int64)
    C = stream250.shape[1]
    et, ec = model if model is not None else load_brant(
        brant_src or brant_src_dir(), weights_dir or brant_weights_dir(),
        device, n_patches=n_patches)
    embs = np.empty((len(ends), C, BRANT_D_MODEL), dtype=np.float32)
    n_padded = 0
    for b0 in range(0, len(ends), batch_size):
        pats = []
        for e in ends[b0:b0 + batch_size]:
            p, was_padded = _context_patches(stream250, int(e), n_patches)
            pats.append(p)
            n_padded += int(was_padded)
        # C-contiguous (B, C, L, 1500): Brant's pre_model.py calls .view() on it.
        xb = np.stack(pats, axis=0).astype(np.float32)
        pw = compute_power(xb, BRANT_FS).astype(np.float32)          # (B, C, L, 8)
        z = get_emb(torch.from_numpy(xb).to(device), torch.from_numpy(pw).to(device),
                    et, ec)                                          # (B, C, L, 2048)
        embs[b0:b0 + len(pats)] = z.mean(dim=2).float().cpu().numpy()
    if n_padded:
        print(f"  [brant] {n_padded}/{len(ends)} anchors had under {n_patches * 6} s of "
              f"history; their front was filled by tiling real signal", flush=True)
    return embs


def brant_embeddings(x_raw: np.ndarray, fs: float, device: str = "cuda",
                     brant_src: str = "", weights_dir: str = "",
                     batch_size: int = 8) -> np.ndarray:
    """Frozen Brant embeddings from pre-cut windows, (N, C, 2048).

    Brant's atomic unit is a 1500-sample patch at 250 Hz, i.e. **6 seconds**, so
    windows handed to this function must already carry at least 6 s; the patch
    used is the last 6 s of each window. Passing a shorter window would zero-pad
    the patch and quietly measure something else, so it raises instead.

    This resamples each window on its own, with integer-rounded rate factors. It
    is a convenience for callers that only have cut windows, and it only
    approximates the paper's BrainTreebank Brant arm, which resampled the whole
    recording once: use `brant_stream_at_250` + `brant_stream_embeddings` to
    reproduce that (experiments/run_ieeg_fm_baselines.py does).

    Args:
        x_raw: (N, C, T) raw windows at `fs` Hz, T >= 6 s worth.
        fs: MEASURED sampling rate. Never assume 2048 -- sub_9 runs at ~1019 Hz,
            and assuming otherwise stretches the patch to 12 s, past the embargo.
    """
    from scipy.signal import resample_poly

    x = np.asarray(x_raw, dtype=np.float32)
    if x.ndim != 3:
        raise ValueError(f"x_raw must be (N, C, T), got {x.shape}")
    n, c, T = x.shape

    up, down = _resample_factors(int(round(fs)), BRANT_FS)
    s250 = resample_poly(x, up, down, axis=-1).astype(np.float32)
    if s250.shape[-1] < BRANT_PATCH_LEN:
        raise ValueError(
            f"windows are {s250.shape[-1]} samples at {BRANT_FS} Hz "
            f"({s250.shape[-1] / BRANT_FS:.2f} s) but Brant's patch is "
            f"{BRANT_PATCH_LEN} ({BRANT_PATCH_LEN / BRANT_FS:.1f} s). Give it its "
            f"native context: 6 s ending where the task window ends.")
    s250 = s250[..., -BRANT_PATCH_LEN:]                    # the 6 s ending at the window end

    et, ec = load_brant(brant_src or brant_src_dir(), weights_dir or brant_weights_dir(),
                        device, n_patches=1)
    out = []
    for i in range(0, n, batch_size):
        chunk = s250[i:i + batch_size]                     # (B, C, 1500)
        power = compute_power(chunk.reshape(-1, BRANT_PATCH_LEN))
        power = power.reshape(chunk.shape[0], c, 1, -1)    # (B, C, 1, 8)
        xt = torch.from_numpy(chunk).to(device).unsqueeze(2)          # (B, C, 1, 1500)
        pt = torch.from_numpy(power).float().to(device)
        emb = get_emb(xt, pt, et, ec)                      # (B, C, 1, 2048)
        out.append(emb.squeeze(2).float().cpu().numpy())
    return np.concatenate(out, axis=0)
