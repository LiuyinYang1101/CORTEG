"""BrainBERT and the Population Transformer on Stanford finger-flexion regression.

This produces the BrainBERT and PopT rows of Table 1 and their Stanford cells
in Tables 18-19 (App. A.11). Brant has its own runner,
`experiments/run_brant_regression.py`, because it needs a 6 s context rather
than a 1 s window.

Input. Each model reads the RAW 1 kHz ECoG, not CORTEG's 128 Hz stream: the
native files come from `data/stanford_native.py` (see there to build them),
and window n of those files pairs with target n of the CORTEG pickle, so
splits, window order and targets are CORTEG's. Each 1 s window ends exactly at
its target. BrainBERT's own preprocessing (`ieeg_fm.preprocess_window`: mains
notch, Laplacian re-reference over the two nearest electrodes, per-channel
z-score, resampling to 2048 Hz, STFT) runs on every window. PopT is defined
over frozen BrainBERT embeddings and adds its population transformer on top;
Stanford's coordinates are subject-space millimetres, so its positional
encoding is an approximation (see `ieeg_fm.mni_mm_to_lip_indices`).

Adaptations (Table 18 rows) and regimes (Table 19 rows):

  --mode probe  --head linear     frozen FM, one embedding per window, MSE
                                  linear head. BrainBERT: its per-electrode
                                  embeddings averaged over electrodes;
                                  PopT: the population [CLS] token.
  --mode finetune                 the last N transformer blocks (default 2) and a
                                  linear head trained end to end at --ft_lr /
                                  --lr. The frozen front end (BrainBERT's
                                  spectrograms; PopT's BrainBERT embeddings) is
                                  computed once and cached.
  --mode probe  --head temporal   a 2-layer BiLSTM over time-ordered sequences
                                  of the frozen per-window embeddings, with a
                                  per-subject linear readout.

  --train_mode pooled       one model for all subjects, scored per subject
               per_subject  one model per subject
               loo          train on the other eight, then continue training
                            on the held-out subject's training split

Every regime is scored the same way: Pearson r per finger on the test split,
averaged over the five fingers, then over subjects.

Recorded settings. The defaults below are the values the paper runs used,
resolved per adaptation when a flag is left unset:

  probe      epochs 60,  patience 20, warmup 10, lr 1e-3, batch 256
  finetune   epochs 60,  patience 20, warmup 5,  lr 1e-3 (head), ft_lr 1e-4
             (unfrozen blocks), unfreeze_last_n 2, batch 256
  temporal   epochs 100, patience 20, warmup 10, lr 1e-3, batch 256 windows
             (4 sequences of 64), hidden 128 per direction
  all        AdamW, weight decay 1e-4, warmup-cosine schedule to min_lr 1e-6,
             early stopping on validation r (the last 10% of the training
             windows), MSE loss, AMP on CUDA, re-reference laplacian_xyz;
             targets and embeddings z-scored with training statistics only.

These are the foundation models' own budgets, not CORTEG's (which trains for
100 epochs with patience 90 at batch 64); changing them to CORTEG's does not
reproduce the paper's numbers. The paper ran probe and fine-tune under all three
regimes (seeds 42, 0 and 1 for pooled and per-subject; 42 for LOO) and the
temporal head pooled and per-subject at seed 42. The one exception is pooled
BrainBERT fine-tuning, which Table 19 does not report (see below).
`scripts/table1_ieeg_fm_stanford.sh` runs that grid.

Where results go. With no --save_root, a run writes to
$CORTEG_OUTPUT_ROOT/ieeg_fm_regression/<adaptation>/Stanford/<fm>/<train_mode>/seed<S>/.
<adaptation> is probe, ft or temporal for the recorded settings. Any other
value of a setting that changes a number is appended to it (e.g.
probe_reref=car, ft_unfreeze_last_n=6, temporal_epochs=2), so such a run never
writes into a paper cell's folder. That includes the device: the paper ran on
GPUs with AMP, and a run records the device type it actually used and AMP only
where it is applied, so a CPU run (fp32; its numbers differ measurably) is
named e.g. probe_use_amp=False. A run on a subject subset or with
--max_windows goes under ieeg_fm_regression_smoke/ instead, in a folder that
also names the subset and the cap. --skip_if_done skips a run whose result
file already records this exact run: the same subjects and the same value of
every such setting, the device type included.

Where this differs from what a reader might assume (all as in the paper runs;
flags marked * select the alternative, which is not what the paper reports):

  * BrainBERT pooling (--bb_pool*). BrainBERT's released recipe averages the
    centre ten of the spectrogram's 22 frames. For a 1 s window ending at the
    target those frames are centred 0.39-0.61 s into the window, so the pooled
    embedding describes the signal ~0.5 s BEFORE the target. That is
    BrainBERT's own recipe, and it is what Tables 1, 18 and 19 use, but it is
    mis-timed for a causal regression target. --bb_pool last10 averages the
    last ten frames instead (centred 0.54-0.76 s, ~0.35 s before the target;
    the last ten frames of the STFT are dropped by BrainBERT's own edge clip).
    It applies to PopT too, whose inputs are these embeddings.
  * The temporal head is BIDIRECTIONAL within each chunk of 64 windows, so a
    window's prediction also sees up to 63 later windows (2.5 s). This favours
    the foundation model. --temporal_direction uni* makes it causal.
  * Pooled probe training is not reseeded after extraction. When a run
    extracts any embeddings itself (a cold cache), the FMs are built first,
    which draws from the CPU RNG, so the linear head starts from another
    initialisation than on a warm cache: the same seed gives a different
    number, by how much is not bounded (on tiny smoke data the gap was
    large). Whether the archived pooled-probe runs read a warm cache or
    extracted cold is not recorded, so their per-seed values (seed 42:
    BrainBERT 0.031795, PopT 0.047346) may not be reproducible exactly.
    `scripts/table1_ieeg_fm_stanford.sh` runs its pooled-probe cells on a warm
    cache: its temporal cells fill the same cache first. --reseed_pooled_head*
    reseeds before the head is built, which makes the result independent of
    the cache: it gives the number a warm-cache run gives (nothing draws from
    the RNG between the seed and the head on a warm cache), but is named
    apart (probe_reseed). Per-subject, LOO, temporal and fine-tune runs
    reseed and are unaffected.
  * The fine-tune readout is a single linear head on the electrode-averaged
    BrainBERT embedding, or on PopT's [CLS] token.
  * Pooled BrainBERT fine-tuning holds every subject's spectrograms and
    backpropagates through two BrainBERT blocks for all electrodes: it needs
    more than 32 GB of GPU memory, and a similar amount of host RAM.

Paper values on Stanford (mean +/- SD over the nine subjects, ddof=1; seed 42
unless noted; the three-seed cells average each subject over seeds first):

  Table 18 (best regime per adaptation)       Table 19 cells
  BrainBERT probe     0.044 +/- 0.051  LOO     probe pooled  BB 0.030, PopT 0.040 (3 seeds)
  BrainBERT last-2 FT 0.047 +/- 0.038  LOO     probe per-sub BB 0.039, PopT 0.049 (3 seeds)
  BrainBERT temporal  0.053 +/- 0.056  per-sub probe LOO     BB 0.044, PopT 0.050
  PopT probe          0.050 +/- 0.045  LOO     FT pooled     BB ---+, PopT 0.041 (3 seeds)
  PopT last-2 FT      0.041 +/- 0.046  pooled  FT per-sub    BB 0.033, PopT 0.037 (3 seeds)
  PopT temporal       0.063 +/- 0.046  per-sub FT LOO       BB 0.047, PopT 0.036

The Table 1 rows are the best adaptation: BrainBERT 0.053 +/- 0.056 and PopT
0.063 +/- 0.046, both the temporal head, per subject.

  + Table 19 prints "---" for Stanford pooled BrainBERT fine-tuning: its
    footnote says the run is incomplete. Only the seed-42 run (r = 0.039) was
    archived. The authors' revision notes give seeds 0 and 1 as 0.030 and
    0.036, from cluster runs whose outputs were never retrieved. The cell is
    therefore not a paper value, and it is not a candidate for Table 18, whose
    BrainBERT fine-tune row is the LOO cell. The script runs it only when
    asked (RUN_BB_FT_POOLED=1, all three seeds, more than 32 GB of GPU
    memory).

Not in this release: full-backbone fine-tuning with a per-electrode readout,
joint fine-tuning of the FM with the temporal head, the representation
analysis, and all Ghent (audio) cells -- the Ghent recordings are private.

Third-party code and weights are not redistributed; see ieeg_fm.py for the
environment variables (BRAINBERT_REPO, BRAINBERT_WEIGHTS, POPT_REPO, and
optionally POPT_WEIGHTS).

Example (the Table 1 PopT cell):

    python -m experiments.run_ieeg_fm_regression --fm popt \\
        --train_mode per_subject --head temporal --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import ieeg_fm
from data.stanford_native import NativeSubject, native_root as resolve_native_root
from experiments.common import STANFORD_SUBJECTS, set_seed
from paths import get_data_root, get_output_root
from train.earlystop import EarlyStopper
from train.lr_schedule import WarmupCosineLR
from train.metrics import corr_per_dim

DATASET = "Stanford"
D_OUT = 5                    # five fingers
BB_BATCH = 256               # (window, electrode) spectrograms per BrainBERT forward
POPT_BATCH = 64              # windows per PopT forward
EXTRACT_CHUNK = 512           # windows per BrainBERT call; a multiple of both batches

# The recorded value of every setting that changes a number, per adaptation.
# epochs and warmup_epochs are filled in from here when left unset
# (resolve_budget); a run that departs from any of them is named after the
# departure (adaptation_tag), so it never lands in a paper cell's folder.
_RECIPE_COMMON = {"early_stop_patience": 20, "batch_size": 256, "lr": 1e-3,
                  "weight_decay": 1e-4, "min_lr": 1e-6, "val_ratio": 0.1,
                  "reref": "laplacian_xyz", "use_amp": True}
RECIPE = {
    "probe": {"epochs": 60, "warmup_epochs": 10, **_RECIPE_COMMON},
    "finetune": {"epochs": 60, "warmup_epochs": 5, **_RECIPE_COMMON,
                 "ft_lr": 1e-4, "unfreeze_last_n": 2},
    "temporal": {"epochs": 100, "warmup_epochs": 10, **_RECIPE_COMMON,
                 "seq_len": 64, "temporal_hidden": 128},
}
MLP_RECIPE = {"mlp_hidden": 256, "mlp_dropout": 0.1}   # --head mlp only (not in the paper)

# Arguments that cannot change a number: where things are read and written,
# and the subject flags (runs are compared on the resolved list). The device
# is a setting: CPU and GPU runs give different numbers, so `resolve_device`
# records the device type actually used, and AMP only where it is applied.
NOT_SETTINGS = frozenset({"native_root", "data_root", "emb_cache", "save_root",
                          "skip_if_done", "subjects", "skip_subjects"})
RESULT_NAME = {"pooled": "results_pooled.json", "per_subject": "results_persub.json",
               "loo": "_loo_done.json"}
PAPER_DIR, SMOKE_DIR = "ieeg_fm_regression", "ieeg_fm_regression_smoke"


# ===========================================================================
# Run bookkeeping: settings, result folders, skip-if-done
# (also used by experiments/run_brant_regression.py)
# ===========================================================================

def resolve_device(args) -> torch.device:
    """The device a run uses, written back into args before anything is named.

    --device auto becomes the device type it resolves to, and use_amp becomes
    whether AMP is actually applied (on CUDA only). The paper cells ran on
    GPUs with AMP; a CPU run is fp32 and gives measurably different numbers.
    Recording it this way names a CPU run <adaptation>_use_amp=False, so it
    never lands in a paper cell's folder, --skip_if_done never takes it for a
    GPU run, and scripts/aggregate_fm.py lists it apart from the paper cells.
    """
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.device = device.type
    args.use_amp = bool(args.use_amp) and device.type == "cuda"
    return device


def json_args(args) -> Dict:
    """vars(args), JSON-ready: every value an int, float, str, bool or None."""
    return {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
            for k, v in vars(args).items()}


def run_settings(args, ignore=NOT_SETTINGS) -> Dict:
    """The arguments that can change a number (paths and subject flags left out)."""
    return {k: v for k, v in json_args(args).items() if k not in ignore}


def same_value(a, b) -> bool:
    """Equal settings; numbers compared with a relative tolerance of 1e-9."""
    if any(isinstance(v, (bool, str)) or v is None for v in (a, b)):
        return a == b
    return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(b)))


def same_settings(got: Dict, want: Dict) -> bool:
    return got.keys() == want.keys() and all(same_value(got[k], want[k]) for k in want)


def _value_str(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def departures(args, recipe: Dict) -> str:
    """'_<name>=<value>' for every setting that differs from its recorded value.

    A value left unset (None) is resolved from the recipe, so it never departs.
    """
    out = ""
    for name, want in recipe.items():
        got = getattr(args, name, want)
        if got is not None and not same_value(got, want):
            out += f"_{name}={_value_str(got)}"
    return out


def result_dir(args, subjects: List[str], tag: str, fm: str) -> str:
    """--save_root, or the default result folder of a run.

    A subject subset or a window / anchor cap is a smoke run, and goes under
    SMOKE_DIR, so it can neither be mistaken for a paper cell nor be picked up
    by `scripts/aggregate_fm.py` pointed at the paper folders. Its folder also
    names the subset and the caps (e.g. probe_subs=mv+wm_w32), so smoke runs
    of one cell on different subsets or caps do not overwrite each other.
    """
    if args.save_root:
        return args.save_root
    max_windows = getattr(args, "max_windows", 0) or 0
    max_anchors = getattr(args, "max_anchors", 0) or 0
    subset = sorted(subjects) != sorted(STANFORD_SUBJECTS)
    smoke = subset or max_windows > 0 or max_anchors > 0
    if subset:
        tag += "_subs=" + "+".join(subjects)
    if max_windows > 0:
        tag += f"_w{int(max_windows)}"
    if max_anchors > 0:
        tag += f"_a{int(max_anchors)}"
    return os.path.join(get_output_root(), SMOKE_DIR if smoke else PAPER_DIR, tag,
                        DATASET, fm, args.train_mode, f"seed{args.seed}")


def finished_result(out_dir: str, train_mode: str, settings: Dict,
                    subjects: List[str]) -> Optional[str]:
    """The result file in out_dir if it already records this exact run, else None.

    It must cover exactly `subjects` and record the same value for every
    setting in `settings`; a setting it does not record counts as different,
    so a doubtful file is re-run rather than trusted. For a `_loo_done.json`
    that carries no settings of its own (the paper runs' format), those of a
    held-out fold are used.
    """
    path = os.path.join(out_dir, RESULT_NAME[train_mode])
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if sorted((d.get("per_subject") or {}).keys()) != sorted(subjects):
        return None
    got = d.get("args") or {}
    if not got and train_mode == "loo":
        for sub in sorted(subjects):
            try:
                with open(os.path.join(out_dir, f"ft_{sub}", "results_persub.json")) as f:
                    got = json.load(f).get("args") or {}
                break
            except (OSError, ValueError):
                continue
    if not all(k in got and same_value(got[k], v) for k, v in settings.items()):
        return None
    return path


# ===========================================================================
# Embedding caches
# ===========================================================================

def _suffixes(nmax: int, bb_pool: str) -> Tuple[str, str]:
    pool = "" if bb_pool == "center10" else f"_{bb_pool}"
    cap = f"_n{int(nmax)}" if nmax and nmax > 0 else ""
    return pool, cap


def _cache_path(emb_cache: str, fm: str, sub: str, split: str, fs: float,
                reref: str, nmax: int = 0, bb_pool: str = "center10") -> str:
    """Probe embedding cache. The name is the one the paper runs wrote.

    The debug cap (--max_windows) and a non-default pooling are folded in, so a
    capped or differently pooled array can never be mistaken for the real one.
    """
    pool, cap = _suffixes(nmax, bb_pool)
    return os.path.join(
        emb_cache, f"{fm}_{DATASET}_{sub}_{split}_fs{int(round(fs))}_{reref}{pool}{cap}.npz")


def _frontend_cache_path(emb_cache: str, fm: str, sub: str, split: str, fs: float,
                         reref: str, nmax: int = 0, bb_pool: str = "center10") -> str:
    """Fine-tune front-end cache, a separate namespace from the probe cache.

    BrainBERT's front end is the spectrograms, which do not depend on pooling;
    PopT's is the pooled BrainBERT embeddings, which do.
    """
    pool, cap = _suffixes(nmax, bb_pool)
    if fm == "brainbert":
        pool = ""
    return os.path.join(
        emb_cache,
        f"{fm}_frontend_{DATASET}_{sub}_{split}_fs{int(round(fs))}_{reref}{pool}{cap}.npz")


def expected_windows(ns: NativeSubject, split: str, nmax: int) -> int:
    """Windows a split yields under the --max_windows cap (read from the index only)."""
    n = int(ns.right_edges(split).size)
    return min(n, int(nmax)) if nmax and nmax > 0 else n


def check_cache_rows(path: str, arrays: Dict[str, np.ndarray], want: int) -> None:
    """A cached array must have one row per window of the run reading it.

    Cache names carry every setting that changes their content, but a file
    written by an older build, or copied in from elsewhere, can still disagree;
    reading it would silently train and score on the wrong windows.
    """
    bad = {k: a.shape[0] for k, a in arrays.items() if a.shape[0] != want}
    if bad:
        raise RuntimeError(
            f"{path}: rows {bad}, but this run has {want} windows for this split. "
            f"The file belongs to another run (a capped smoke run or an older build); "
            f"delete it and re-run.")


def _save_atomic(path: str, **arrays) -> None:
    """Write via a per-process temp file, so concurrent cells never read half a file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def _maybe_car_fallback(reref: str, xyz_mm: Optional[np.ndarray], sub: str) -> str:
    """laplacian_xyz needs usable coordinates; otherwise fall back to CAR.

    Resolved BEFORE the cache key is built, so the file name states the
    montage actually used. (All nine Stanford subjects keep laplacian_xyz.)
    """
    if reref != "laplacian_xyz":
        return reref
    if xyz_mm is None:
        print(f"  [reref] {sub}: no coordinates -> car")
        return "car"
    xyz = np.asarray(xyz_mm)
    n_nan = int(np.isnan(xyz).any(axis=1).sum())
    n_zero = int(np.all(xyz == 0, axis=1).sum())
    uniq = np.unique(np.round(xyz, 3), axis=0).shape[0]
    if n_nan > 0 or uniq < 3 or n_zero == xyz.shape[0]:
        print(f"  [reref] {sub}: degenerate coordinates (nan={n_nan} zero={n_zero} "
              f"unique={uniq}) -> car")
        return "car"
    return reref


# ===========================================================================
# BrainBERT pooling
# ===========================================================================

def bb_pool_slice(t_frames: int, mode: str) -> slice:
    """Which BrainBERT output frames are averaged into one embedding.

    center10 is BrainBERT's released recipe (`out[:, mid-5:mid+5]`), used for
    every paper number; last10 is the frames nearest the target. See the module
    docstring for their timing.
    """
    half = ieeg_fm.POOL_HALF
    if mode == "center10":
        mid = t_frames // 2
        return slice(max(0, mid - half), mid + half)
    if mode == "last10":
        return slice(max(0, t_frames - 2 * half), t_frames)
    raise ValueError(f"bb_pool must be center10|last10, got {mode!r}")


def brainbert_per_electrode(x_raw: np.ndarray, fs: float, xyz_mm: np.ndarray,
                            reref: str, device: str, bb_pool: str) -> np.ndarray:
    """Frozen per-electrode BrainBERT embeddings, (N, C, 768).

    Called EXTRACT_CHUNK windows at a time, because last10 pooling asks
    `ieeg_fm.brainbert_embeddings` for every frame of every (window,
    electrode): about 40 GB for a 64-electrode subject's training split. The
    chunk is a multiple of the forward batch, so every forward pass sees
    exactly the batch it would see in one call, and the model is built once
    per split, as in the paper runs.
    """
    model = ieeg_fm.load_brainbert(device=device)
    pool = "default" if bb_pool == "center10" else "none"
    out = []
    for i in range(0, x_raw.shape[0], EXTRACT_CHUNK):
        rep = ieeg_fm.brainbert_embeddings(
            x_raw[i:i + EXTRACT_CHUNK], fs=fs, xyz_mm=xyz_mm, device=device,
            reref=reref, pool=pool, batch_size=BB_BATCH, model=model)
        if pool == "none":
            rep = rep[:, :, bb_pool_slice(rep.shape[2], bb_pool)].mean(axis=2)
        out.append(np.asarray(rep, dtype=np.float32))
    return np.concatenate(out, axis=0)


# ===========================================================================
# Frozen embeddings (probe and temporal head), with caching
# ===========================================================================

def _extract(fm: str, ns: NativeSubject, split: str, reref: str, device: str,
             bb_pool: str, nmax: int) -> np.ndarray:
    x = ns.windows(split, nmax)
    per_elec = brainbert_per_electrode(x, ns.fs, ns.xyz_mm, reref, device, bb_pool)
    if fm == "brainbert":
        return per_elec.mean(axis=1)                                  # (N, 768)
    # PopT on the same per-electrode embeddings, with the same re-reference.
    return ieeg_fm.popt_embeddings(
        x, np.asarray(ns.xyz_mm, dtype=np.float32), fs=int(round(ns.fs)),
        device=device, coords_are_mni_mm=True, batch_size=POPT_BATCH,
        per_electrode_embeddings=per_elec)                             # (N, 512)


def _tail_split(n: int, val_ratio: float) -> Tuple[slice, slice]:
    """Validation = the last val_ratio of the training windows (time order)."""
    n_val = min(max(1, int(round(n * val_ratio))), n - 1)
    return slice(0, n - n_val), slice(n - n_val, n)


def get_subject_embeddings(fm: str, ns: NativeSubject, args, device: str,
                           emb_cache: Optional[str]) -> Dict:
    """Frozen train/test embeddings for one subject, a tail-validation split and
    targets z-scored on the training part only.

    On a cache hit neither the windows nor the pickle are read.
    """
    eff_reref = _maybe_car_fallback(args.reref, ns.xyz_mm, ns.sub)
    got = {}
    for split in ("train", "test"):
        cpath = (_cache_path(emb_cache, fm, ns.sub, split, ns.fs, eff_reref,
                             args.max_windows, args.bb_pool) if emb_cache else None)
        if cpath and os.path.exists(cpath):
            d = np.load(cpath)
            emb, y = d["emb"].astype(np.float32), d["y"].astype(np.float32)
            check_cache_rows(cpath, {"emb": emb, "y": y},
                             expected_windows(ns, split, args.max_windows))
            print(f"  [cache hit] {os.path.basename(cpath)} emb={emb.shape}")
            got[split] = (emb, y)
            continue
        y = ns.targets()[0 if split == "train" else 1]
        if args.max_windows and args.max_windows > 0:
            y = y[: int(args.max_windows)]
        emb = np.asarray(_extract(fm, ns, split, eff_reref, device, args.bb_pool,
                                  args.max_windows), dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if cpath:
            _save_atomic(cpath, emb=emb, y=y)
            print(f"  [cache write] {os.path.basename(cpath)} emb={emb.shape}")
        got[split] = (emb, y)

    emb_tr, y_tr = got["train"]
    emb_te, y_te = got["test"]
    tr, va = _tail_split(emb_tr.shape[0], args.val_ratio)
    ymu, ysd = fit_zscore(y_tr[tr])
    y_tr_z, y_te_z = apply_zscore(y_tr, ymu, ysd), apply_zscore(y_te, ymu, ysd)
    return {"sub": ns.sub,
            "emb_tr": emb_tr[tr], "y_tr": y_tr_z[tr],
            "emb_va": emb_tr[va], "y_va": y_tr_z[va],
            "emb_te": emb_te, "y_te": y_te_z}


# ===========================================================================
# Per-window heads (linear / mlp), MSE only, scored by Pearson r
# ===========================================================================

class LinearHead(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.fc = nn.Linear(d_in, d_out)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, d_in: int, d_out: int, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, d_out))

    def forward(self, x):
        return self.net(x)


def build_head(head: str, d_in: int, d_out: int, hidden: int, dropout: float) -> nn.Module:
    if head == "linear":
        return LinearHead(d_in, d_out)
    if head == "mlp":
        return MLPHead(d_in, d_out, hidden=hidden, dropout=dropout)
    raise ValueError(f"per-window head must be linear|mlp, got {head!r}")


def fit_zscore(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mu = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return mu.astype(np.float32), sd.astype(np.float32)


def apply_zscore(x: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((x - mu) / sd).astype(np.float32)


def train_head(emb_tr, y_tr, emb_va, y_va, d_out: int, args, device: torch.device,
               init_state: Optional[Dict[str, torch.Tensor]] = None) -> nn.Module:
    """MSE head on z-scored embeddings; early stopping on validation r."""
    fmu, fsd = fit_zscore(emb_tr)
    Xtr = apply_zscore(emb_tr, fmu, fsd)
    Xva = apply_zscore(emb_va, fmu, fsd)

    model = build_head(args.head, Xtr.shape[1], d_out, args.mlp_hidden,
                       args.mlp_dropout).to(device)
    if init_state is not None:
        model.load_state_dict(init_state, strict=True)
    # The feature scaler travels with the model so evaluation re-applies it.
    model._feat_mu = torch.from_numpy(fmu).to(device)
    model._feat_sd = torch.from_numpy(fsd).to(device)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                           max_epochs=args.epochs, min_lr=args.min_lr)
    early = EarlyStopper(patience=args.early_stop_patience)
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    n, bs = Xtr_t.shape[0], args.batch_size
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = torch.mean((model(Xtr_t[idx]) - ytr_t[idx]) ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).float().cpu().numpy()
        if early.step(float(np.nanmean(corr_per_dim(pv, y_va))), model):
            break
    early.restore(model)
    return model


def predict(model: nn.Module, emb: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    X = (emb - model._feat_mu.cpu().numpy()) / model._feat_sd.cpu().numpy()
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32)).to(device)).float().cpu().numpy()


def score_subject(pred: np.ndarray, y_true: np.ndarray) -> Dict:
    """Pearson r per finger; the subject's score is their mean."""
    corr = corr_per_dim(pred, y_true)
    return {"corr_mean": float(np.nanmean(corr)), "corr": [float(c) for c in corr],
            "mse": float(np.mean((pred - y_true) ** 2)), "n": int(y_true.shape[0])}


# ===========================================================================
# Results
# ===========================================================================

def _sd(per_subject: Dict) -> Optional[float]:
    """Sample SD (ddof=1) over subjects, the paper's +/-; None for one subject."""
    vals = [r["corr_mean"] for r in per_subject.values()]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else None


def _fmt(score: float, sd: Optional[float]) -> str:
    return f"{score:.4f}" + ("" if sd is None else f" +/- {sd:.4f}")


def write_results(path: str, train_mode: str, score: float, score_mse: float,
                  per_subject: Dict, args, n_trainable: Optional[int] = None) -> None:
    """Result json: the paper's `score` (mean over subjects) plus its SD (ddof=1)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {
        "fm": args.fm, "dataset": DATASET, "train_mode": train_mode,
        "mode": args.mode, "head": args.head,
        "unfreeze_last_n": args.unfreeze_last_n if args.mode == "finetune" else None,
        "score": float(score), "score_sd": _sd(per_subject),
        "n_subjects": len(per_subject), "score_mse": float(score_mse),
        "per_subject": per_subject,
        "args": json_args(args),
    }
    if n_trainable is not None:
        obj["n_trainable"] = int(n_trainable)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\nResults saved: {path}  (score={_fmt(score, obj['score_sd'])})")


def write_loo_done(out_dir: str, per_subject: Dict, args,
                   n_trainable: Optional[int] = None) -> Tuple[str, float]:
    """The LOO summary: mean held-out r over subjects. Also marks the cell done.

    Unlike the paper runs' summaries it records the run's arguments, so
    --skip_if_done can tell which run it summarises.
    """
    score = float(np.nanmean([r["corr_mean"] for r in per_subject.values()]))
    path = os.path.join(out_dir, "_loo_done.json")
    obj = {"fm": args.fm, "dataset": DATASET, "train_mode": "loo",
           "mode": args.mode, "head": args.head, "score": score,
           "score_sd": _sd(per_subject), "per_subject": per_subject,
           "n_subjects": len(per_subject), "args": json_args(args)}
    if n_trainable is not None:
        obj["n_trainable"] = int(n_trainable)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\n[loo] {len(per_subject)} held-out subjects: "
          f"{_fmt(score, obj['score_sd'])}  ({path})")
    return path, score


def _summarise(per_subject: Dict) -> Tuple[float, float]:
    scores = [r["corr_mean"] for r in per_subject.values()]
    mses = [r["mse"] for r in per_subject.values()]
    return float(np.nanmean(scores)), float(np.nanmean(mses))


# ===========================================================================
# Regimes: frozen probe
# ===========================================================================

def run_pooled(subjects: List[Dict], d_out, args, device, out_dir):
    """One head on every subject's training windows; scored per subject.

    Not reseeded here by default, as in the paper runs, so the head's
    initialisation depends on whether extraction built the FMs first (see the
    module docstring). --reseed_pooled_head makes it independent of that.
    """
    if args.reseed_pooled_head:
        set_seed(args.seed)
    cat = lambda k: np.concatenate([s[k] for s in subjects], axis=0)  # noqa: E731
    model = train_head(cat("emb_tr"), cat("y_tr"), cat("emb_va"), cat("y_va"),
                       d_out, args, device)
    per_subject = {s["sub"]: score_subject(predict(model, s["emb_te"], device), s["y_te"])
                   for s in subjects}
    score, mse = _summarise(per_subject)
    path = os.path.join(out_dir, "results_pooled.json")
    write_results(path, "pooled", score, mse, per_subject, args)
    return path, score


def run_per_subject(subjects: List[Dict], d_out, args, device, out_dir):
    per_subject = {}
    for s in subjects:
        set_seed(args.seed)
        model = train_head(s["emb_tr"], s["y_tr"], s["emb_va"], s["y_va"], d_out, args, device)
        per_subject[s["sub"]] = rec = score_subject(predict(model, s["emb_te"], device), s["y_te"])
        print(f"  {s['sub']}: r={rec['corr_mean']:.4f}")
    score, mse = _summarise(per_subject)
    path = os.path.join(out_dir, "results_persub.json")
    write_results(path, "per_subject", score, mse, per_subject, args)
    return path, score


def stage1_key(args, others: List[str], device) -> Dict:
    """What a LOO stage-1 head depends on: the training subjects, every setting
    that changes a number, and the device type (CPU and GPU runs differ)."""
    return {"others": ",".join(others), "device_type": torch.device(device).type,
            **run_settings(args)}


def load_stage1(path: str, key: Dict, device) -> Optional[Dict]:
    """A saved stage-1 checkpoint, only if it was trained exactly as `key` says.

    The paper runs reused any checkpoint found in the fold folder; that is
    kept for resumed runs, but a checkpoint from another run (other subjects,
    a smoke run, other settings) is retrained rather than reused.
    """
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    saved = ckpt.get("key") if isinstance(ckpt, dict) else None
    if not isinstance(saved, dict) or not same_settings(saved, key):
        print(f"  [loo] {path} was trained by another run; retraining stage 1")
        return None
    return ckpt


def run_loo(subjects: List[Dict], d_out, args, device, out_dir):
    """Stage 1 on the other subjects, stage 2 continues on the held-out subject.

    Stage 2 re-fits the feature scaler on the held-out subject and starts from
    the stage-1 weights. Stage 1 is saved per fold, and reused by a run that
    matches it exactly (`load_stage1`).
    """
    per_subject = {}
    for held in subjects:
        sub = held["sub"]
        ft_dir = os.path.join(out_dir, f"ft_{sub}")
        os.makedirs(ft_dir, exist_ok=True)
        stage1_path = os.path.join(ft_dir, "stage1_head.pt")
        others = [s for s in subjects if s["sub"] != sub]
        cat = lambda k: np.concatenate([s[k] for s in others], axis=0)  # noqa: E731
        emb_tr = cat("emb_tr")
        key = stage1_key(args, [s["sub"] for s in others], device)

        set_seed(args.seed)
        ckpt = load_stage1(stage1_path, key, device)
        if ckpt is not None:
            print(f"  [loo {sub}] stage 1 reused: {stage1_path}")
            model1 = build_head(args.head, emb_tr.shape[1], d_out,
                                args.mlp_hidden, args.mlp_dropout).to(device)
            model1.load_state_dict(ckpt["state_dict"], strict=True)
        else:
            model1 = train_head(emb_tr, cat("y_tr"), cat("emb_va"), cat("y_va"),
                                d_out, args, device)
            torch.save({"state_dict": model1.state_dict(),
                        "feat_mu": model1._feat_mu.cpu(),
                        "feat_sd": model1._feat_sd.cpu(), "key": key}, stage1_path)

        set_seed(args.seed)
        model2 = train_head(held["emb_tr"], held["y_tr"], held["emb_va"], held["y_va"],
                            d_out, args, device, init_state=dict(model1.state_dict()))
        rec = score_subject(predict(model2, held["emb_te"], device), held["y_te"])
        per_subject[sub] = rec
        write_results(os.path.join(ft_dir, "results_persub.json"), "loo",
                      rec["corr_mean"], rec["mse"], {sub: rec}, args)
        print(f"  [loo {sub}] held-out r={rec['corr_mean']:.4f}")
    return write_loo_done(out_dir, per_subject, args)


# ===========================================================================
# Temporal head: a BiLSTM over time-ordered sequences of frozen embeddings
# ---------------------------------------------------------------------------
# Each subject's train / validation / test windows are chunked INDEPENDENTLY
# into non-overlapping sequences of --seq_len consecutive windows, so no
# sequence crosses a subject or a split. The ragged last chunk is zero-padded
# and masked out of the loss; evaluation flattens the chunks back into window
# order and scores exactly like the per-window heads.
# ===========================================================================

class TemporalHead(nn.Module):
    """2-layer LSTM over (B, L, d_in) -> per-window (B, L, d_out).

    One linear readout per subject, chosen per sequence by an integer id
    (pooled training); one readout otherwise.
    """

    def __init__(self, d_in: int, d_out: int, hidden: int = 128, n_subjects: int = 1,
                 num_layers: int = 2, dropout: float = 0.1, bidirectional: bool = True):
        super().__init__()
        self.d_out = int(d_out)
        self.n_subjects = int(n_subjects)
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=(dropout if num_layers > 1 else 0.0))
        width = (2 if bidirectional else 1) * hidden
        self.readout = nn.ModuleList(
            [nn.Linear(width, self.d_out) for _ in range(self.n_subjects)])

    def forward(self, x, sid=None):
        out, _ = self.lstm(x)
        if self.n_subjects == 1 or sid is None:
            return self.readout[0](out)
        pred = out.new_zeros(out.shape[0], out.shape[1], self.d_out)
        sid = sid.view(-1)
        for s in torch.unique(sid):
            m = sid == s
            pred[m] = self.readout[int(s.item())](out[m])
        return pred


def _chunk_sequences(emb: np.ndarray, y: np.ndarray, seq_len: int
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(N, D) + (N, d) in time order -> X (S, L, D), Y (S, L, d), mask (S, L)."""
    emb = np.ascontiguousarray(emb, dtype=np.float32)
    y = np.ascontiguousarray(y, dtype=np.float32)
    N, D = emb.shape
    L = int(seq_len)
    S = (N + L - 1) // L
    X = np.zeros((S, L, D), dtype=np.float32)
    Y = np.zeros((S, L, y.shape[1]), dtype=np.float32)
    M = np.zeros((S, L), dtype=bool)
    for si in range(S):
        lo, hi = si * L, min(si * L + L, N)
        X[si, :hi - lo] = emb[lo:hi]
        Y[si, :hi - lo] = y[lo:hi]
        M[si, :hi - lo] = True
    return X, Y, M


def _seq_predict_windows(model, X, sid_arr, device, use_amp: bool, seq_bs: int,
                         n_windows: int) -> np.ndarray:
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, X.shape[0], seq_bs):
            xb = torch.from_numpy(X[i:i + seq_bs]).to(device)
            sb = torch.from_numpy(sid_arr[i:i + seq_bs]).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pb = model(xb, sb)
            outs.append(pb.float().cpu().numpy())
    return np.concatenate(outs, axis=0).reshape(-1, model.d_out)[:n_windows]


def _seq_batch(args) -> int:
    """Sequences per batch, so a batch holds ~--batch_size windows."""
    return max(1, args.batch_size // int(args.seq_len))


def _train_temporal(model, Xtr, Ytr, Mtr, SIDtr, val_blocks: List[Dict], args,
                    device) -> nn.Module:
    """Masked MSE; early stopping on validation r averaged over subjects."""
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                           max_epochs=args.epochs, min_lr=args.min_lr)
    early = EarlyStopper(patience=args.early_stop_patience)
    Xtr_t, Ytr_t = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Mtr_t, SIDtr_t = torch.from_numpy(Mtr), torch.from_numpy(SIDtr)
    n, d_out, seq_bs = Xtr.shape[0], Ytr.shape[-1], _seq_batch(args)

    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, seq_bs):
            idx = perm[i:i + seq_bs]
            xb, yb = Xtr_t[idx].to(device), Ytr_t[idx].to(device)
            mb = Mtr_t[idx].to(device).unsqueeze(-1).float()
            sb = SIDtr_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                diff = (model(xb, sb).float() - yb) ** 2
                loss = (diff * mb).sum() / (mb.sum().clamp_min(1.0) * d_out)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        vcorrs = []
        for b in val_blocks:
            sidv = np.full(b["Xva"].shape[0], b["sid"], dtype=np.int64)
            pv = _seq_predict_windows(model, b["Xva"], sidv, device, use_amp, seq_bs, b["n_va"])
            vcorrs.append(float(np.nanmean(corr_per_dim(pv, b["y_va"]))))
        if early.step(float(np.nanmean(vcorrs)), model):
            break
    early.restore(model)
    return model


def _seq_block(s: Dict, sid: int, fmu, fsd, seq_len: int) -> Dict:
    """Z-score (training statistics) and chunk one subject's three splits."""
    Xtr, Ytr, Mtr = _chunk_sequences(apply_zscore(s["emb_tr"], fmu, fsd), s["y_tr"], seq_len)
    Xva, _, _ = _chunk_sequences(apply_zscore(s["emb_va"], fmu, fsd), s["y_va"], seq_len)
    Xte, _, _ = _chunk_sequences(apply_zscore(s["emb_te"], fmu, fsd), s["y_te"], seq_len)
    return {"sub": s["sub"], "sid": int(sid), "Xtr": Xtr, "Ytr": Ytr, "Mtr": Mtr,
            "Xva": Xva, "n_va": int(s["y_va"].shape[0]), "y_va": s["y_va"],
            "Xte": Xte, "n_te": int(s["y_te"].shape[0]), "y_te": s["y_te"]}


def _temporal_head(d_in: int, d_out: int, n_subjects: int, args) -> TemporalHead:
    return TemporalHead(d_in, d_out, hidden=args.temporal_hidden, n_subjects=n_subjects,
                        dropout=0.1, bidirectional=args.temporal_direction == "bi")


def _stack(blocks: List[Dict]):
    cat = lambda k: np.concatenate([b[k] for b in blocks], axis=0)  # noqa: E731
    sid = np.concatenate([np.full(b["Xtr"].shape[0], b["sid"], dtype=np.int64)
                          for b in blocks], axis=0)
    return cat("Xtr"), cat("Ytr"), cat("Mtr"), sid


def _score_blocks(model, blocks, args, device) -> Dict:
    use_amp = bool(args.use_amp) and device.type == "cuda"
    out = {}
    for b in blocks:
        sidv = np.full(b["Xte"].shape[0], b["sid"], dtype=np.int64)
        pred = _seq_predict_windows(model, b["Xte"], sidv, device, use_amp,
                                    _seq_batch(args), b["n_te"])
        out[b["sub"]] = score_subject(pred, b["y_te"])
    return out


def run_pooled_temporal(subjects: List[Dict], d_out, args, device, out_dir):
    """One shared LSTM with per-subject readouts over every subject's sequences."""
    set_seed(args.seed)
    emb_all = np.concatenate([s["emb_tr"] for s in subjects], axis=0)
    fmu, fsd = fit_zscore(emb_all)
    blocks = [_seq_block(s, i, fmu, fsd, args.seq_len) for i, s in enumerate(subjects)]
    Xtr, Ytr, Mtr, SID = _stack(blocks)
    model = _temporal_head(emb_all.shape[1], d_out, len(blocks), args).to(device)
    print(f"  [temporal pooled] seq_len={args.seq_len} hidden={args.temporal_hidden} "
          f"{args.temporal_direction} sequences={Xtr.shape[0]}")
    model = _train_temporal(model, Xtr, Ytr, Mtr, SID, blocks, args, device)
    per_subject = _score_blocks(model, blocks, args, device)
    score, mse = _summarise(per_subject)
    path = os.path.join(out_dir, "results_pooled.json")
    write_results(path, "pooled", score, mse, per_subject, args)
    return path, score


def run_per_subject_temporal(subjects: List[Dict], d_out, args, device, out_dir):
    per_subject = {}
    for s in subjects:
        set_seed(args.seed)
        fmu, fsd = fit_zscore(s["emb_tr"])
        b = _seq_block(s, 0, fmu, fsd, args.seq_len)
        model = _temporal_head(b["Xtr"].shape[2], d_out, 1, args).to(device)
        model = _train_temporal(model, b["Xtr"], b["Ytr"], b["Mtr"],
                                np.zeros(b["Xtr"].shape[0], dtype=np.int64), [b], args, device)
        per_subject.update(_score_blocks(model, [b], args, device))
        print(f"  {s['sub']}: r={per_subject[s['sub']]['corr_mean']:.4f}")
    score, mse = _summarise(per_subject)
    path = os.path.join(out_dir, "results_persub.json")
    write_results(path, "per_subject", score, mse, per_subject, args)
    return path, score


def run_loo_temporal(subjects: List[Dict], d_out, args, device, out_dir):
    """Shared LSTM on the other subjects; the held-out subject gets a fresh
    readout on the transferred LSTM. (Not run for the paper.)"""
    per_subject = {}
    for held in subjects:
        sub = held["sub"]
        ft_dir = os.path.join(out_dir, f"ft_{sub}")
        os.makedirs(ft_dir, exist_ok=True)
        others = [s for s in subjects if s["sub"] != sub]

        set_seed(args.seed)
        emb_all = np.concatenate([s["emb_tr"] for s in others], axis=0)
        fmu, fsd = fit_zscore(emb_all)
        oblocks = [_seq_block(s, i, fmu, fsd, args.seq_len) for i, s in enumerate(others)]
        stage1 = _temporal_head(emb_all.shape[1], d_out, len(oblocks), args).to(device)
        if oblocks:
            Xtr, Ytr, Mtr, SID = _stack(oblocks)
            stage1 = _train_temporal(stage1, Xtr, Ytr, Mtr, SID, oblocks, args, device)

        set_seed(args.seed)
        hmu, hsd = fit_zscore(held["emb_tr"])
        hb = _seq_block(held, 0, hmu, hsd, args.seq_len)
        stage2 = _temporal_head(emb_all.shape[1], d_out, 1, args).to(device)
        stage2.lstm.load_state_dict(stage1.lstm.state_dict())
        stage2 = _train_temporal(stage2, hb["Xtr"], hb["Ytr"], hb["Mtr"],
                                 np.zeros(hb["Xtr"].shape[0], dtype=np.int64),
                                 [hb], args, device)
        rec = _score_blocks(stage2, [hb], args, device)[sub]
        per_subject[sub] = rec
        write_results(os.path.join(ft_dir, "results_persub.json"), "loo",
                      rec["corr_mean"], rec["mse"], {sub: rec}, args)
        print(f"  [loo {sub}] held-out r={rec['corr_mean']:.4f}")
    return write_loo_done(out_dir, per_subject, args)


# ===========================================================================
# Last-N fine-tuning
# ---------------------------------------------------------------------------
# The frozen front end is computed once and cached: BrainBERT's per-electrode
# spectrograms (N, C, 22, 40) for BrainBERT, the frozen BrainBERT embeddings
# (N, C, 768) for PopT (BrainBERT stays frozen under PopT). What trains is the
# last N transformer blocks of the model and a linear head.
# ===========================================================================

def _brainbert_frontend(x_raw, fs, xyz_mm, reref) -> np.ndarray:
    """BrainBERT's preprocessing of every window, (N, C, T_frames, 40)."""
    return np.stack([ieeg_fm.preprocess_window(x_raw[n], fs, xyz_mm, reref)
                     for n in range(x_raw.shape[0])], axis=0).astype(np.float32)


def _load_or_compute_frontend(fm, ns: NativeSubject, split, reref, device, emb_cache,
                              args) -> np.ndarray:
    cpath = (_frontend_cache_path(emb_cache, fm, ns.sub, split, ns.fs, reref,
                                  args.max_windows, args.bb_pool) if emb_cache else None)
    if cpath and os.path.exists(cpath):
        fe = np.load(cpath)["fe"].astype(np.float32)
        check_cache_rows(cpath, {"fe": fe}, expected_windows(ns, split, args.max_windows))
        print(f"  [frontend cache hit] {os.path.basename(cpath)} fe={fe.shape}")
        return fe
    x = ns.windows(split, args.max_windows)
    if fm == "brainbert":
        fe = _brainbert_frontend(x, ns.fs, ns.xyz_mm, reref)
    else:
        fe = brainbert_per_electrode(x, ns.fs, ns.xyz_mm, reref, device, args.bb_pool)
    fe = np.asarray(fe, dtype=np.float32)
    if cpath:
        _save_atomic(cpath, fe=fe)
        print(f"  [frontend cache write] {os.path.basename(cpath)} fe={fe.shape}")
    return fe


def get_subject_frontend(fm: str, ns: NativeSubject, args, device: str,
                         emb_cache: Optional[str]) -> Dict:
    """Cached front end for train/test, tail validation, z-scored targets, and
    the subject's PopT coordinate indices."""
    eff_reref = _maybe_car_fallback(args.reref, ns.xyz_mm, ns.sub)
    fe_tr = _load_or_compute_frontend(fm, ns, "train", eff_reref, device, emb_cache, args)
    fe_te = _load_or_compute_frontend(fm, ns, "test", eff_reref, device, emb_cache, args)
    y_tr, y_te = ns.targets()
    if args.max_windows and args.max_windows > 0:
        y_tr, y_te = y_tr[: args.max_windows], y_te[: args.max_windows]
    tr, va = _tail_split(fe_tr.shape[0], args.val_ratio)
    ymu, ysd = fit_zscore(y_tr[tr])
    y_tr_z, y_te_z = apply_zscore(y_tr, ymu, ysd), apply_zscore(y_te, ymu, ysd)
    return {"sub": ns.sub,
            "fe_tr": fe_tr[tr], "y_tr": y_tr_z[tr],
            "fe_va": fe_tr[va], "y_va": y_tr_z[va],
            "fe_te": fe_te, "y_te": y_te_z,
            "xyz_mm": np.asarray(ns.xyz_mm, dtype=np.float32),
            "lip": ieeg_fm.mni_mm_to_lip_indices(np.asarray(ns.xyz_mm, dtype=np.float32))}


def _unfreeze_last(model: nn.Module, layers, extra_norm, n: int) -> int:
    """Freeze everything, then unfreeze the last n blocks (and a final norm if any)."""
    for p in model.parameters():
        p.requires_grad_(False)
    k = max(0, min(int(n), len(layers)))
    for layer in list(layers)[len(layers) - k:]:
        for p in layer.parameters():
            p.requires_grad_(True)
    if extra_norm is not None:
        for p in extra_norm.parameters():
            p.requires_grad_(True)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class BrainBERTBack(nn.Module):
    """Spectrograms (B, C, Tf, 40) -> BrainBERT with its last N blocks trainable
    -> pooled frames -> electrode mean (B, 768) -> linear head."""

    def __init__(self, bb_model, d_out: int, unfreeze_last_n: int, bb_pool: str = "center10"):
        super().__init__()
        self.bb = bb_model
        self.bb_pool = bb_pool
        self.n_trainable_fm = _unfreeze_last(
            bb_model, bb_model.transformer.layers,
            getattr(bb_model.transformer, "norm", None), unfreeze_last_n)
        self.head = nn.Linear(ieeg_fm.HIDDEN_DIM, d_out)

    def forward(self, specs):
        B, C, Tf, F = specs.shape
        flat = specs.reshape(B * C, Tf, F)
        mask = torch.zeros(flat.shape[:2], dtype=torch.bool, device=flat.device)
        rep = self.bb.forward(flat, mask, intermediate_rep=True)      # (B*C, Tf, 768)
        emb = rep[:, bb_pool_slice(Tf, self.bb_pool)].mean(dim=1)
        return self.head(emb.reshape(B, C, ieeg_fm.HIDDEN_DIM).mean(dim=1))


class PopTBack(nn.Module):
    """Frozen BrainBERT embeddings (B, C, 768) + coordinates -> PopT with its last
    N blocks trainable -> [CLS] (B, 512) -> linear head.

    Coordinates are passed per batch, so one pooled model serves subjects with
    different electrode layouts; a batch never mixes subjects.
    """

    def __init__(self, pt_model, d_out: int, unfreeze_last_n: int, lip_indices: np.ndarray):
        super().__init__()
        self.pt = pt_model
        self.n_trainable_fm = _unfreeze_last(
            pt_model, pt_model.transformer_encoder.layers,
            getattr(pt_model.transformer_encoder, "norm", None), unfreeze_last_n)
        self.head = nn.Linear(ieeg_fm.POPT_HIDDEN_DIM, d_out)
        self.register_buffer("default_lip",
                             torch.as_tensor(np.asarray(lip_indices), dtype=torch.long))

    def forward(self, emb, coords=None):
        B, C, _ = emb.shape
        cls = torch.ones(B, 1, ieeg_fm.POPT_INPUT_DIM, dtype=emb.dtype, device=emb.device)
        inputs = torch.cat([cls, emb], dim=1)
        pad_mask = torch.zeros(B, 1 + C, dtype=torch.bool, device=emb.device)
        if coords is None:
            coords = self.default_lip.unsqueeze(0).expand(B, C, 3)
        else:
            coords = coords.to(device=emb.device, dtype=torch.long)
            if coords.shape != (B, C, 3):
                raise ValueError(f"coords {tuple(coords.shape)} != (B={B}, C={C}, 3)")
        seq_id = torch.zeros(B, C, dtype=torch.long, device=emb.device)
        rep = self.pt.forward(inputs, pad_mask, (coords, seq_id), intermediate_rep=True)
        return self.head(rep[:, 0, :])


def build_finetune_model(fm: str, subj: Dict, d_out: int, args, device) -> nn.Module:
    """A fresh pretrained model with its last N blocks trainable, plus a head."""
    if fm == "brainbert":
        model = BrainBERTBack(ieeg_fm.load_brainbert(device=str(device)), d_out,
                              args.unfreeze_last_n, bb_pool=args.bb_pool)
    else:
        model = PopTBack(ieeg_fm.load_popt_model(device=str(device)), d_out,
                         args.unfreeze_last_n, subj["lip"])
    model = model.to(device)
    tot = sum(p.numel() for p in model.parameters())
    tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [finetune {subj['sub']}] trainable {tr:,} of {tot:,} "
          f"(head + last {args.unfreeze_last_n} blocks)")
    return model


def _frontend_stats(fe_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Per-feature (last axis) z-score statistics pooled over subjects.

    Subjects have different electrode counts, so the arrays are reduced to
    sums rather than concatenated.
    """
    D = fe_list[0].shape[-1]
    total, s1, s2 = 0, np.zeros(D), np.zeros(D)
    for fe in fe_list:
        flat = fe.reshape(-1, D).astype(np.float64)
        total += flat.shape[0]
        s1 += flat.sum(axis=0)
        s2 += (flat * flat).sum(axis=0)
    mu = s1 / total
    sd = np.sqrt(np.maximum(s2 / total - mu * mu, 0.0))
    sd = np.where(sd < 1e-8, 1.0, sd)
    shape = (1,) * (fe_list[0].ndim - 1) + (D,)
    return mu.astype(np.float32).reshape(shape), sd.astype(np.float32).reshape(shape)


def _block(s: Dict, key_fe: str, key_y: Optional[str], fm: str, fmu, fsd) -> Dict:
    """One subject's windows for training or prediction.

    The front end is z-scored per batch (`_rows`) rather than copied whole: the
    values are the same, and a normalised copy of every subject's front end
    would double the host memory fine-tuning needs.
    """
    return {"fe": s[key_fe], "mu": fmu, "sd": fsd,
            "y": None if key_y is None else torch.from_numpy(np.asarray(s[key_y], np.float32)),
            "coords": (torch.as_tensor(np.asarray(s["lip"]), dtype=torch.long)
                       if fm == "popt" else None),
            "sub": s["sub"], "n": s[key_fe].shape[0]}


def _rows(blk: Dict, idx) -> torch.Tensor:
    return torch.from_numpy(((blk["fe"][idx] - blk["mu"]) / blk["sd"]).astype(np.float32))


def _forward(model, fm, xb, coords_C, device, use_amp):
    with torch.amp.autocast("cuda", enabled=use_amp):
        if fm == "popt":
            coords = coords_C.unsqueeze(0).expand(xb.shape[0], coords_C.shape[0], 3).to(device)
            return model(xb, coords=coords)
        return model(xb)


def _predict_block(model, fm, blk, device, use_amp: bool, batch_size: int) -> np.ndarray:
    outs = []
    with torch.no_grad():
        for i in range(0, blk["n"], batch_size):
            xb = _rows(blk, slice(i, i + batch_size)).to(device)
            outs.append(_forward(model, fm, xb, blk["coords"], device, use_amp).float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def train_finetune(fm, subjects: List[Dict], model, args, device) -> nn.Module:
    """End-to-end MSE training of the unfrozen blocks (--ft_lr) and head (--lr).

    Batches never mix subjects (their electrode counts differ); the visiting
    order of batches is shuffled across subjects every epoch.
    """
    fmu, fsd = _frontend_stats([s["fe_tr"] for s in subjects])
    model._fe_mu, model._fe_sd = fmu, fsd
    tr_blocks = [_block(s, "fe_tr", "y_tr", fm, fmu, fsd) for s in subjects]
    va_blocks = [_block(s, "fe_va", "y_va", fm, fmu, fsd) for s in subjects]

    head_ids = {id(p) for p in model.head.parameters()}
    fm_params = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    opt = torch.optim.AdamW([{"params": fm_params, "lr": args.ft_lr},
                             {"params": list(model.head.parameters()), "lr": args.lr}],
                            weight_decay=args.weight_decay)
    sched = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                           max_epochs=args.epochs, min_lr=args.min_lr)
    early = EarlyStopper(patience=args.early_stop_patience)
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    bs = args.batch_size

    for _ in range(args.epochs):
        model.train()
        plan = []
        for bi, blk in enumerate(tr_blocks):
            perm = np.random.permutation(blk["n"])
            plan += [(bi, perm[i:i + bs]) for i in range(0, blk["n"], bs)]
        np.random.shuffle(plan)
        for bi, widx in plan:
            blk = tr_blocks[bi]
            xb = _rows(blk, widx).to(device)
            yb = blk["y"][torch.from_numpy(widx)].to(device)
            opt.zero_grad(set_to_none=True)
            pred = _forward(model, fm, xb, blk["coords"], device, use_amp)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = torch.mean((pred.float() - yb) ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        pv = np.concatenate([_predict_block(model, fm, b, device, use_amp, bs)
                             for b in va_blocks], axis=0)
        yv = np.concatenate([b["y"].numpy() for b in va_blocks], axis=0)
        if early.step(float(np.nanmean(corr_per_dim(pv, yv))), model):
            break
    early.restore(model)
    return model


def predict_finetune(model, s: Dict, device, fm: str, batch_size: int = 256) -> np.ndarray:
    """Test predictions for one subject (without autocast, as in the paper runs)."""
    model.eval()
    return _predict_block(model, fm, _block(s, "fe_te", None, fm, model._fe_mu, model._fe_sd),
                          device, False, batch_size)


def run_pooled_ft(subjects: List[Dict], d_out, args, device, out_dir):
    set_seed(args.seed)
    model = build_finetune_model(args.fm, subjects[0], d_out, args, device)
    model = train_finetune(args.fm, subjects, model, args, device)
    per_subject = {s["sub"]: score_subject(predict_finetune(model, s, device, args.fm), s["y_te"])
                   for s in subjects}
    score, mse = _summarise(per_subject)
    path = os.path.join(out_dir, "results_pooled.json")
    write_results(path, "pooled", score, mse, per_subject, args,
                  n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad))
    return path, score


def run_per_subject_ft(subjects: List[Dict], d_out, args, device, out_dir):
    per_subject, n_tr = {}, 0
    for s in subjects:
        set_seed(args.seed)
        model = build_finetune_model(args.fm, s, d_out, args, device)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        model = train_finetune(args.fm, [s], model, args, device)
        per_subject[s["sub"]] = rec = score_subject(
            predict_finetune(model, s, device, args.fm), s["y_te"])
        print(f"  {s['sub']}: r={rec['corr_mean']:.4f}")
        del model
    score, mse = _summarise(per_subject)
    path = os.path.join(out_dir, "results_persub.json")
    write_results(path, "per_subject", score, mse, per_subject, args, n_trainable=n_tr)
    return path, score


def run_loo_ft(subjects: List[Dict], d_out, args, device, out_dir):
    """Stage 1 on the other subjects (PopT built with the held-out subject's
    coordinates as its fallback; batches always carry their own), stage 2
    continues the same model on the held-out subject."""
    per_subject, n_tr = {}, 0
    for held in subjects:
        sub = held["sub"]
        ft_dir = os.path.join(out_dir, f"ft_{sub}")
        os.makedirs(ft_dir, exist_ok=True)
        others = [s for s in subjects if s["sub"] != sub]
        set_seed(args.seed)
        model = build_finetune_model(args.fm, held, d_out, args, device)
        n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if others:
            model = train_finetune(args.fm, others, model, args, device)
        set_seed(args.seed)
        model = train_finetune(args.fm, [held], model, args, device)
        rec = score_subject(predict_finetune(model, held, device, args.fm), held["y_te"])
        per_subject[sub] = rec
        write_results(os.path.join(ft_dir, "results_persub.json"), "loo",
                      rec["corr_mean"], rec["mse"], {sub: rec}, args, n_trainable=n_tr)
        print(f"  [loo {sub}] held-out r={rec['corr_mean']:.4f}")
        del model
    return write_loo_done(out_dir, per_subject, args, n_trainable=n_tr)


# ===========================================================================
# CLI
# ===========================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BrainBERT / PopT on Stanford finger-flexion regression (native 1 kHz).")
    p.add_argument("--fm", type=str, required=True, choices=["brainbert", "popt"])
    p.add_argument("--train_mode", type=str, default="pooled",
                   choices=["pooled", "per_subject", "loo"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mode", type=str, default="probe", choices=["probe", "finetune"],
                   help="probe: frozen FM + head. finetune: the last --unfreeze_last_n "
                        "blocks train with a linear head.")
    p.add_argument("--head", type=str, default="linear", choices=["linear", "mlp", "temporal"],
                   help="linear (the paper's probe and fine-tune readout), mlp, or temporal "
                        "(the BiLSTM over window sequences; probe only).")
    p.add_argument("--unfreeze_last_n", type=int, default=2)
    p.add_argument("--ft_lr", type=float, default=1e-4,
                   help="learning rate of the unfrozen FM blocks (the head uses --lr)")

    # data and paths
    p.add_argument("--subjects", type=str, default="", help="comma-separated subset")
    p.add_argument("--skip_subjects", type=str, default="")
    p.add_argument("--data_root", type=str, default="",
                   help="CORTEG pickles, for the targets (default paths.get_data_root())")
    p.add_argument("--native_root", type=str, default="",
                   help="built <sub>_native1k.npz files (default $CORTEG_NATIVE1K_ROOT, else "
                        "<data_root>/native_1k/built with --data_root if given; build with "
                        "python -m data.stanford_native)")
    p.add_argument("--emb_cache", type=str, default="",
                   help="folder for cached embeddings / front ends (reused when present)")
    p.add_argument("--save_root", type=str, default="",
                   help="result folder (default $CORTEG_OUTPUT_ROOT/ieeg_fm_regression/"
                        "<adaptation>/Stanford/<fm>/<train_mode>/seed<seed>; subject subsets "
                        "and --max_windows runs go under ieeg_fm_regression_smoke/)")
    p.add_argument("--skip_if_done", action="store_true",
                   help="exit at once if the result folder already holds this exact run "
                        "(same subjects, same value of every setting)")
    p.add_argument("--max_windows", type=int, default=0,
                   help="smoke tests only: first N windows per split (0 = all)")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                   help="the paper ran on a GPU with AMP; a CPU run (fp32) is recorded with "
                        "use_amp False and named <adaptation>_use_amp=False")

    # extraction
    p.add_argument("--reref", type=str, default="laplacian_xyz",
                   choices=["laplacian_xyz", "car", "none"])
    p.add_argument("--bb_pool", type=str, default="center10", choices=["center10", "last10"],
                   help="BrainBERT frames averaged per embedding. center10 is BrainBERT's "
                        "recipe and the paper's; last10 is nearer the target (not in the paper).")

    # heads
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--mlp_dropout", type=float, default=0.1)
    p.add_argument("--seq_len", type=int, default=64,
                   help="temporal head: windows per non-overlapping sequence")
    p.add_argument("--temporal_hidden", type=int, default=128,
                   help="temporal head: LSTM width per direction")
    p.add_argument("--temporal_direction", type=str, default="bi", choices=["bi", "uni"],
                   help="bi is the paper's (sees later windows within a sequence); "
                        "uni is causal (not in the paper)")
    p.add_argument("--reseed_pooled_head", action="store_true",
                   help="pooled probe only: reseed before the head is built, so the result "
                        "does not depend on whether the embedding cache was warm (not in "
                        "the paper; the paper runs did not reseed)")

    # training (epochs and warmup default to the recorded budget of the adaptation)
    p.add_argument("--epochs", type=int, default=None,
                   help="default: 60 probe / fine-tune, 100 temporal head")
    p.add_argument("--early_stop_patience", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=None,
                   help="default: 10 probe / temporal, 5 fine-tune")
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--use_amp", dest="use_amp", action="store_true", default=True,
                   help="mixed precision on CUDA (default; the paper runs used it)")
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    return p


def _recipe_key(args) -> str:
    return "temporal" if args.head == "temporal" else args.mode


def resolve_budget(args) -> argparse.Namespace:
    """Fill epochs / warmup left unset with the adaptation's recorded budget."""
    recipe = RECIPE[_recipe_key(args)]
    for name in ("epochs", "warmup_epochs"):
        if getattr(args, name) is None:
            setattr(args, name, recipe[name])
    return args


def adaptation_tag(args) -> str:
    """Result sub-folder: probe, ft or temporal for the recorded settings.

    Every departure is named in it: the head, the pooling, the temporal
    direction and pooled-probe reseeding as a short suffix, and any other
    recorded setting as _<name>=<value> (e.g. probe_reref=car,
    ft_unfreeze_last_n=6, probe_use_amp=False for a CPU run; see
    `resolve_device`).
    """
    tag = "temporal" if args.head == "temporal" else ("ft" if args.mode == "finetune" else "probe")
    if args.head == "mlp":
        tag += "_mlp"
    if args.bb_pool != "center10":
        tag += f"_{args.bb_pool}"
    if args.head == "temporal" and args.temporal_direction != "bi":
        tag += f"_{args.temporal_direction}"
    if getattr(args, "reseed_pooled_head", False):
        tag += "_reseed"
    tag += departures(args, RECIPE[_recipe_key(args)])
    if args.head == "mlp":
        tag += departures(args, MLP_RECIPE)
    return tag


def resolve_save_root(args, subjects: Optional[List[str]] = None) -> str:
    """--save_root, or the default folder (see `result_dir`)."""
    subs = resolve_subjects(args) if subjects is None else subjects
    return result_dir(args, subs, adaptation_tag(args), args.fm)


def resolve_subjects(args) -> List[str]:
    subs = list(STANFORD_SUBJECTS)
    if args.subjects:
        want = [s.strip() for s in args.subjects.split(",") if s.strip()]
        unknown = [s for s in want if s not in subs]
        if unknown:
            raise SystemExit(f"unknown Stanford subjects: {unknown}")
        subs = [s for s in subs if s in want]
    if args.skip_subjects:
        skip = {s.strip() for s in args.skip_subjects.split(",") if s.strip()}
        subs = [s for s in subs if s not in skip]
    return subs


def main(argv=None):
    args = resolve_budget(build_argparser().parse_args(argv))
    if args.mode == "finetune" and args.head != "linear":
        raise SystemExit(
            "--mode finetune trains a linear readout. Fine-tuning jointly with the "
            "temporal head, and other fine-tune readouts, are not part of this release.")
    if args.reseed_pooled_head and not (args.train_mode == "pooled" and args.mode == "probe"
                                        and args.head != "temporal"):
        raise SystemExit("--reseed_pooled_head applies to the pooled frozen probe only; "
                         "every other regime already reseeds before its head is built.")
    set_seed(args.seed)

    # Before anything is named or compared: records the device and AMP as used.
    device = resolve_device(args)
    device_str = device.type

    subjects = resolve_subjects(args)
    if not subjects:
        raise SystemExit("no subjects selected")
    if args.train_mode == "loo" and len(subjects) < 2:
        raise SystemExit("--train_mode loo needs at least two subjects: one held out, "
                         "the others to train stage 1 on")
    out_dir = resolve_save_root(args, subjects)
    if args.skip_if_done:
        done = finished_result(out_dir, args.train_mode, run_settings(args), subjects)
        if done:
            print(f"[done] {done} already records this run; skipping")
            with open(done) as f:
                return json.load(f).get("score")
    data_root = args.data_root or get_data_root()
    nroot = resolve_native_root(args.native_root, args.data_root)
    os.makedirs(out_dir, exist_ok=True)
    emb_cache = args.emb_cache or None
    if emb_cache:
        os.makedirs(emb_cache, exist_ok=True)

    print("=" * 64)
    print(f"iEEG-FM regression: fm={args.fm} mode={args.mode} head={args.head} "
          f"train_mode={args.train_mode} seed={args.seed}")
    print(f"  device={device_str} amp={args.use_amp} epochs={args.epochs} "
          f"patience={args.early_stop_patience} warmup={args.warmup_epochs} "
          f"bb_pool={args.bb_pool}")
    print(f"  subjects ({len(subjects)}): {subjects}")
    print(f"  native_root={nroot}\n  save_root={out_dir}\n  emb_cache={emb_cache}")
    print("=" * 64)

    t0 = time.time()
    per_subj: List[Dict] = []
    for sub in subjects:
        print(f"\n[load+extract] {sub}")
        ns = NativeSubject(sub, nroot, data_root)
        if args.mode == "finetune":
            se = get_subject_frontend(args.fm, ns, args, device_str, emb_cache)
            print(f"  front end train={se['fe_tr'].shape} val={se['fe_va'].shape} "
                  f"test={se['fe_te'].shape}")
        else:
            se = get_subject_embeddings(args.fm, ns, args, device_str, emb_cache)
            print(f"  emb train={se['emb_tr'].shape} val={se['emb_va'].shape} "
                  f"test={se['emb_te'].shape}")
        per_subj.append(se)
        del ns
    print(f"\n[extract] {time.time() - t0:.0f}s; training")

    if args.mode == "finetune":
        run = {"pooled": run_pooled_ft, "per_subject": run_per_subject_ft,
               "loo": run_loo_ft}[args.train_mode]
    elif args.head == "temporal":
        run = {"pooled": run_pooled_temporal, "per_subject": run_per_subject_temporal,
               "loo": run_loo_temporal}[args.train_mode]
    else:
        run = {"pooled": run_pooled, "per_subject": run_per_subject,
               "loo": run_loo}[args.train_mode]
    path, score = run(per_subj, D_OUT, args, device, out_dir)
    print(f"\n{'=' * 64}\nDONE {args.fm}/{args.mode}/{args.head}/{args.train_mode} "
          f"score={score:.4f}\n{path}\n{'=' * 64}")
    return score


if __name__ == "__main__":
    main()
