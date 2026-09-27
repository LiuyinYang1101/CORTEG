"""Native 1 kHz Stanford fingerflex windows for the intracranial-FM comparison.

The CORTEG data contract (`data/io.py`) carries the Stanford ECoG at 128 Hz.
That is what CORTEG itself reads, but it is the wrong input for the
intracranial foundation models: BrainBERT and PopT were pretrained on 2048 Hz
spectrograms and Brant on 250 Hz patches, and everything above the 64 Hz
Nyquist of the 128 Hz stream -- the high-gamma band BrainBERT leans on -- is
gone. The paper therefore evaluates them on the raw 1 kHz recordings
("Stanford is evaluated at its native 1 kHz", App. A.11). This module is the
bridge between the two.

For each subject it writes ``<sub>_native1k.npz`` holding the raw 1 kHz signal
of the channels CORTEG kept, plus the right-edge sample index of every
CORTEG window, so that window n of the native file pairs with target n of
``<sub>_features.pkl``. Nothing is re-windowed or re-labelled: the targets,
the split and the window order all stay CORTEG's.

Building (once per subject; reads the raw download the preprocessing tutorial
already asks for, plus the CORTEG pickle)::

    python -m data.stanford_native                      # all nine subjects
    python -m data.stanford_native --subjects bp,mv     # or: --subjects bp mv

  --data_root  folder of <sub>_features.pkl / <sub>_electrode_loc.mat, the
               same --data_root the runners take; default paths.get_data_root()
               ($CORTEG_DATA_ROOT). The two defaults below are relative to it,
               so a build with --data_root D is found by a run with
               --data_root D. (--pkl_root is an alias.)
  --raw_root   folder of <sub>/<sub>_fingerflex.mat (Stanford Digital
               Repository zk881ps0522, the same download as
               data/stanford_preprocessing); default $CORTEG_STANFORD_RAW_ROOT,
               else <data_root>/raw/Stanford, where
               data/stanford_preprocessing/tutorial_Stanford.md suggests putting it
  --out_root   where the npz files go; default $CORTEG_NATIVE1K_ROOT, else
               <data_root>/native_1k/built, where the runners look (native_root()
               below)

How the alignment is established, per subject:

  1. Channels. CORTEG dropped bad channels, so the pickle has C <= Cn of the
     raw file's channels. Each CORTEG electrode is matched to the raw file's
     ``locs`` by nearest neighbour (distances are ~1e-6 mm for every subject),
     which recovers the kept channels in CORTEG's order.
  2. Split. The paper's rule: recordings longer than 600 s train on the first
     400 s, otherwise on the first two thirds. Windows are 1 s (1000 samples)
     with a 40-sample stride, the target read at the window's right edge.
  3. Offsets. The window grid's phase is not stored anywhere, so it is
     recovered: the 128 Hz stream is re-derived from the raw signal with the
     original recipe (60/120/180 Hz band-stop, CAR, anti-alias decimation to
     128 Hz) and correlated against the pickle's X_raw over a spread of
     windows, separately for the train and the test segment. The test offset
     is floored so that no test window's lookback reaches into training time.
  4. Gate. The file is written only if the window counts equal the pickle's,
     every window lies inside the recording and its own segment, and the
     median alignment correlation is >= 0.95. For the nine subjects it is
     0.9984-0.9991, and no proof window is below 0.996. A silently misaligned
     benchmark file is the worst outcome here, so failure refuses to write
     rather than warning.

The re-derived 128 Hz stream is only the alignment proof. What goes to disk is
the RAW signal: no notch, no re-reference, no scaling -- each foundation model
applies its own preprocessing. Note that it is in the recording's ADC counts,
not the microvolts of the CORTEG pickle (the MATLAB step scales by 0.0298).
BrainBERT and PopT z-score every window and do not see the difference; Brant's
input is scale-sensitive, and its paper number was produced on these counts.

The builder is a port of the script that produced the files behind every
Stanford foundation-model number in the paper; its output is array-for-array
identical to those files.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

# `python data/stanford_native.py` as well as `python -m data.stanford_native`.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from paths import get_data_root  # noqa: E402

SUBJECTS = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]
FS = 1000                     # native sampling rate (Hz)
WIN = 1000                    # 1 s lookback (samples at 1 kHz)
STRIDE = 40                   # 0.04 s stride, the 25 Hz data-glove rate

# The 128 Hz re-derivation used only for the alignment proof.
RESAMPLE_UP, RESAMPLE_DOWN = 16, 125     # 1000 * 16 / 125 = 128 Hz
NOTCH_FREQS = (60.0, 120.0, 180.0)
NOTCH_BW = 2.0                # band-stop half-width (Hz)
NOTCH_ORDER = 3               # Butterworth order
N_PROOF = 50                  # proof windows spread over train + test
TRAIN_SEARCH = range(-20, 21)       # offset search around the train seed
TEST_COARSE = range(-200, 2400)     # offset search around the test boundary
ALIGN_MIN = 0.95              # median proof correlation required to write

NPZ_SUFFIX = "_native1k.npz"
SUMMARY_NAME = "_build_summary.json"


# ─────────────────────────────── locations ──────────────────────────────────
def native_root(override: str = "", data_root: str = "") -> str:
    """Folder holding the built ``<sub>_native1k.npz`` files.

    Priority: explicit argument > $CORTEG_NATIVE1K_ROOT >
    <data_root>/native_1k/built, where data_root is the one given (a runner's
    --data_root), else paths.get_data_root().
    """
    if override:
        return override
    return os.environ.get("CORTEG_NATIVE1K_ROOT") or os.path.join(
        get_data_root(data_root), "native_1k", "built")


def raw_root(override: str = "", data_root: str = "") -> str:
    """Folder of the raw ``<sub>/<sub>_fingerflex.mat`` downloads.

    Priority: explicit argument > $CORTEG_STANFORD_RAW_ROOT >
    <data_root>/raw/Stanford (the location the preprocessing tutorial suggests).
    """
    if override:
        return override
    return os.environ.get("CORTEG_STANFORD_RAW_ROOT") or os.path.join(
        get_data_root(data_root), "raw", "Stanford")


def parse_subjects(values: List[str]) -> List[str]:
    """Subjects given space- or comma-separated (``bp mv`` or ``bp,mv``), checked."""
    subs = [s.strip() for v in values for s in v.split(",") if s.strip()]
    unknown = [s for s in subs if s not in SUBJECTS]
    if unknown or not subs:
        raise SystemExit(f"unknown Stanford subjects: {unknown or values}; "
                         f"choose from {','.join(SUBJECTS)}")
    return list(dict.fromkeys(subs))


def native_npz_path(root: str, sub: str) -> str:
    return os.path.join(root, f"{sub}{NPZ_SUFFIX}")


def _missing(sub: str, path: str, data_root: str = "") -> FileNotFoundError:
    root_flag = f" --data_root {data_root}" if data_root else ""
    return FileNotFoundError(
        f"native 1 kHz file for {sub} not found: {path}\n"
        f"  Build it from the raw fingerflex download with\n"
        f"      python -m data.stanford_native --subjects {sub}{root_flag}\n"
        f"  or point --native_root / $CORTEG_NATIVE1K_ROOT at an existing build.")


# ─────────────────────────────── loading ────────────────────────────────────
def build_native_windows(data_kept: np.ndarray, right_edges: np.ndarray,
                         win_len: int = WIN) -> np.ndarray:
    """(N, C, win_len) windows cut from a (T, C) stream.

    Window n covers ``data_kept[s - (win_len - 1) : s + 1]`` for right-edge
    index s = right_edges[n]: the edge sample is INCLUDED, so the window ends
    exactly at the target time and nothing after it is read.
    """
    data_kept = np.ascontiguousarray(np.asarray(data_kept, dtype=np.float32))
    T, C = data_kept.shape
    right_edges = np.asarray(right_edges).reshape(-1)
    X = np.empty((right_edges.shape[0], C, int(win_len)), dtype=np.float32)
    for i, s in enumerate(right_edges):
        s = int(s)
        lo, hi = s - (int(win_len) - 1), s + 1
        if lo < 0 or hi > T:
            raise ValueError(f"window {i}: right edge {s} gives [{lo}, {hi}), "
                             f"outside the stream of length {T}")
        X[i] = data_kept[lo:hi, :].T
    return X


class NativeSubject:
    """One subject's native 1 kHz file, paired with CORTEG's targets.

    Everything is lazy: opening reads only the small arrays (fs, window length,
    coordinates). The pickle -- 1-4 GB per subject, needed only for the
    targets -- is read on the first call to `targets`, and windows are cut on
    demand. A caller whose embeddings are already cached therefore never pays
    for either.
    """

    def __init__(self, sub: str, root: str = "", data_root: str = ""):
        self.sub = sub
        self.root = native_root(root, data_root)
        self.data_root = data_root or get_data_root()
        self.path = native_npz_path(self.root, sub)
        if not os.path.exists(self.path):
            raise _missing(sub, self.path, data_root)
        self._npz = np.load(self.path, allow_pickle=False)
        files = set(self._npz.files)
        need = {"data_kept", "win_starts_tr", "win_starts_te", "xyz"}
        if not need <= files:
            raise ValueError(f"{self.path} lacks {sorted(need - files)}; rebuild it")
        self.fs = float(self._npz["fs"]) if "fs" in files else float(FS)
        self.win_len = int(self._npz["win_len"]) if "win_len" in files else WIN
        self.xyz_mm = np.asarray(self._npz["xyz"], dtype=np.float32)   # (C,3), CORTEG order
        self._y: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._stream: Optional[np.ndarray] = None

    def right_edges(self, split: str) -> np.ndarray:
        key = {"train": "win_starts_tr", "test": "win_starts_te"}[split]
        return np.asarray(self._npz[key]).reshape(-1).astype(np.int64)

    def stream(self) -> np.ndarray:
        """The raw (T, C) stream both splits index into (ADC counts, float32)."""
        if self._stream is None:
            self._stream = np.ascontiguousarray(self._npz["data_kept"], dtype=np.float32)
        return self._stream

    def targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """(y_tr, y_te) from the CORTEG pickle; window n <-> target n."""
        if self._y is None:
            from data.io import load_subject
            sd = load_subject(self.data_root, self.sub, require_xyz=False)
            y_tr = np.asarray(sd.y_tr, dtype=np.float32)
            y_te = np.asarray(sd.y_te, dtype=np.float32)
            del sd
            n_tr, n_te = self.right_edges("train").size, self.right_edges("test").size
            if y_tr.shape[0] != n_tr or y_te.shape[0] != n_te:
                raise AssertionError(
                    f"{self.sub}: pickle has {y_tr.shape[0]}/{y_te.shape[0]} train/test "
                    f"windows, native file {n_tr}/{n_te}; they must pair one to one")
            self._y = (y_tr, y_te)
        return self._y

    def windows(self, split: str, nmax: int = 0) -> np.ndarray:
        """(N, C, win_len) raw windows for `split`; the first `nmax` if nmax > 0."""
        re = self.right_edges(split)
        if nmax and nmax > 0:
            re = re[: int(nmax)]
        X = build_native_windows(self.stream(), re, self.win_len)
        if X.shape[1] != self.xyz_mm.shape[0]:
            raise AssertionError(f"{self.sub}: windows have C={X.shape[1]}, "
                                 f"coordinates C={self.xyz_mm.shape[0]}")
        return X


def load_native_windows(sub: str, root: str = "", data_root: str = "") -> Dict:
    """Convenience: every array for one subject, materialised.

    Returns {x_tr, y_tr, x_te, y_te, xyz_mm, fs}; x_* are (N, C, 1000) raw 1 kHz
    windows ending at each target, y_* the (N, 5) finger targets.
    """
    ns = NativeSubject(sub, root, data_root)
    y_tr, y_te = ns.targets()
    return {"x_tr": ns.windows("train"), "y_tr": y_tr,
            "x_te": ns.windows("test"), "y_te": y_te,
            "xyz_mm": ns.xyz_mm, "fs": ns.fs}


# ─────────────────────────────── building ───────────────────────────────────
def _bandstop(x: np.ndarray) -> np.ndarray:
    from scipy.signal import butter, filtfilt
    y = x
    for f0 in NOTCH_FREQS:
        b, a = butter(NOTCH_ORDER,
                      [(f0 - NOTCH_BW) / (FS / 2), (f0 + NOTCH_BW) / (FS / 2)],
                      btype="bandstop")
        y = filtfilt(b, a, y, axis=0)
    return y


def derive_lfs_continuous(raw_kept: np.ndarray) -> np.ndarray:
    """Band-stop then CAR over the kept channels, still at 1 kHz.

    There is deliberately no separate 1-64 Hz band-pass: resample_poly's
    anti-alias FIR in `lfs_window` is the only low-pass the original 128 Hz
    stream had. Adding one lowers the proof correlation. This is how the recipe
    was pinned down: in that diagnostic on bp, the per-channel proof
    correlation was 0.997 as here, 0.985 with an FIR band-pass and 0.922 with
    a Butterworth one. (The gate's median over proof windows on bp is 0.999.)
    """
    xs = _bandstop(raw_kept)
    return xs - xs.mean(axis=1, keepdims=True)


def lfs_window(xf: np.ndarray, s: int) -> np.ndarray:
    """The 128-sample, 128 Hz window whose 1 kHz right edge is sample s."""
    from scipy.signal import resample_poly
    return resample_poly(xf[s - WIN + 1: s + 1], RESAMPLE_UP, RESAMPLE_DOWN, axis=0)


def per_channel_corr(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pearson r per column of two (T, C) arrays."""
    A = A - A.mean(axis=0)
    B = B - B.mean(axis=0)
    num = (A * B).sum(axis=0)
    den = np.sqrt((A * A).sum(axis=0) * (B * B).sum(axis=0)) + 1e-12
    return num / den


def split_boundary(total_samples: int) -> Tuple[int, str]:
    """(boundary, rule): train is [0, boundary), test is [boundary, T)."""
    if total_samples > 600 * FS:
        return 400 * FS, "first400s"
    return (total_samples * 2) // 3, "first2/3"


def median_corr_for_offset(xf, offset: int, probe_n, X_raw, T: int) -> float:
    vals = []
    for n in probe_n:
        s = offset + int(n) * STRIDE
        if s - WIN + 1 < 0 or s + 1 > T:
            return -1.0
        vals.append(np.median(per_channel_corr(lfs_window(xf, s), X_raw[n].T)))
    return float(np.median(vals)) if vals else -1.0


def tune_offset(xf, seed: int, search, probe_n, X_raw, T: int,
                min_off: Optional[int] = None) -> Tuple[int, float]:
    """Offset in seed + search maximising the median proof correlation.

    `min_off` is a hard floor. For the test segment it is the smallest right
    edge whose 1 s lookback starts at or after the split boundary; anything
    lower would read training-time signal into a test window, so a spurious
    correlation peak there can never be chosen.
    """
    best_off, best_c = None, -1.0
    for delta in search:
        off = seed + delta
        if min_off is not None and off < min_off:
            continue
        c = median_corr_for_offset(xf, off, probe_n, X_raw, T)
        if c > best_c:
            best_off, best_c = off, c
    if best_off is None:
        best_off = seed if min_off is None else max(seed, min_off)
    return best_off, best_c


def process_subject(sub: str, raw_dir: str, pkl_root: str, out_root: str,
                    verbose: bool = True) -> dict:
    """Build, verify and (only if every check passes) write one subject's npz."""
    import scipy.io
    from scipy.spatial import cKDTree
    from data.io import load_subject

    nat_path = os.path.join(raw_dir, sub, f"{sub}_fingerflex.mat")
    if not os.path.exists(nat_path):
        raise FileNotFoundError(
            f"{nat_path} not found. --raw_root (or $CORTEG_STANFORD_RAW_ROOT; default "
            f"<data_root>/raw/Stanford) must be the folder of <sub>/<sub>_fingerflex.mat "
            f"from the Stanford fingerflex download.")
    if verbose:
        print(f"\n=== {sub} ===\n  loading {nat_path}", flush=True)
    mat = scipy.io.loadmat(nat_path)
    data = mat["data"].astype(np.float64)                 # (T, Cn) ADC counts
    locs = np.asarray(mat["locs"], dtype=np.float64)      # (Cn, 3)
    T, Cn = data.shape

    sd = load_subject(pkl_root, sub, require_xyz=True)
    X_tr, X_te = sd.X_raw_tr, sd.X_raw_te                 # (N, C, 128)
    N_tr, N_te, C = X_tr.shape[0], X_te.shape[0], X_tr.shape[1]
    xyz = sd.ecog_xyz_mm.astype(np.float64)

    # (1) channels, in CORTEG's order
    dist, kept_idx = cKDTree(locs).query(xyz)
    kept_idx = kept_idx.astype(np.int64)
    if np.unique(kept_idx).size != C:
        print(f"  [WARN] {sub}: nearest-neighbour match is not one-to-one", flush=True)
    raw_kept = data[:, kept_idx]

    # (2) split, (3) offsets tuned against the 128 Hz stream
    boundary, rule = split_boundary(T)
    xf = derive_lfs_continuous(raw_kept)
    # The first window needs a full lookback (right edge >= WIN - 1); the phase
    # the original pipeline used sits 40 samples later, at WIN + 39.
    probe_tr = np.unique(np.linspace(0, N_tr - 1, min(N_tr, 12)).astype(int))
    tr_off, tr_c = tune_offset(xf, WIN + 39, TRAIN_SEARCH, probe_tr, X_tr, T)
    seed_te = boundary + WIN - 1
    probe_te = np.unique(np.linspace(0, N_te - 1, min(N_te, 12)).astype(int))
    te_off, te_c = tune_offset(xf, seed_te, TEST_COARSE, probe_te, X_te, T,
                               min_off=seed_te)

    win_starts_tr = tr_off + np.arange(N_tr, dtype=np.int64) * STRIDE
    win_starts_te = te_off + np.arange(N_te, dtype=np.int64) * STRIDE
    bounds_ok = bool(win_starts_tr.min() - WIN + 1 >= 0 and win_starts_tr.max() + 1 <= T
                     and win_starts_te.min() - WIN + 1 >= 0 and win_starts_te.max() + 1 <= T)
    train_in_region = bool(win_starts_tr.max() + 1 <= boundary)
    test_in_region = bool(win_starts_te.min() - WIN + 1 >= boundary)

    # (4) proof over windows spread across both segments
    idx_tr = np.unique(np.linspace(0, N_tr - 1, min(N_tr, N_PROOF // 2)).astype(int))
    idx_te = np.unique(np.linspace(0, N_te - 1, min(N_te, N_PROOF - N_PROOF // 2)).astype(int))
    proof = [np.median(per_channel_corr(lfs_window(xf, int(win_starts_tr[n])), X_tr[n].T))
             for n in idx_tr]
    proof += [np.median(per_channel_corr(lfs_window(xf, int(win_starts_te[n])), X_te[n].T))
              for n in idx_te]
    median_align, min_align = float(np.median(proof)), float(np.min(proof))
    del xf
    gc.collect()

    checks = {
        "bounds_ok": bounds_ok,
        "train_in_region": train_in_region,
        "test_in_region": test_in_region,
        "N_tr_match": bool(win_starts_tr.size == N_tr),
        "N_te_match": bool(win_starts_te.size == N_te),
        "align_ok": bool(median_align >= ALIGN_MIN),
    }
    wrote = all(checks.values())
    out_path = native_npz_path(out_root, sub)
    if wrote:
        os.makedirs(out_root, exist_ok=True)
        np.savez_compressed(
            out_path,
            data_kept=raw_kept.astype(np.float32),        # (T, C) raw, CORTEG channel order
            win_starts_tr=win_starts_tr.astype(np.int64),  # (N_tr,) right-edge sample index
            win_starts_te=win_starts_te.astype(np.int64),  # (N_te,)
            fs=np.int64(FS), win_len=np.int64(WIN), stride=np.int64(STRIDE),
            kept_idx=kept_idx, our_C=np.int64(C), native_C=np.int64(Cn),
            xyz=xyz.astype(np.float32), match_dist=dist.astype(np.float32),
            boundary=np.int64(boundary),
            train_offset=np.int64(tr_off), test_offset=np.int64(te_off),
        )
    else:
        failed = [k for k, v in checks.items() if not v]
        print(f"  [ABORT] {sub}: not writing {out_path}; failed: {', '.join(failed)}",
              flush=True)

    row = {
        "subject": sub, "our_C": int(C), "native_C": int(Cn),
        "max_xyz_match_dist_mm": round(float(dist.max()), 6),
        "total_s": round(T / FS, 2), "split_rule": rule, "boundary_sample": int(boundary),
        "N_tr": int(N_tr), "N_te": int(N_te),
        "train_offset": int(tr_off), "test_offset": int(te_off),
        "train_search_corr": round(float(tr_c), 4), "test_search_corr": round(float(te_c), 4),
        "median_align_corr": round(median_align, 4), "min_align_corr": round(min_align, 4),
        **checks, "wrote": bool(wrote),
        "file": os.path.basename(out_path) if wrote else None,
    }
    if verbose:
        print(f"  C={C} (raw {Cn})  max match dist={dist.max():.1e} mm  split={rule}\n"
              f"  train offset {tr_off} (r={tr_c:.3f})  test offset {te_off} (r={te_c:.3f})\n"
              f"  median alignment r={median_align:.4f} (min {min_align:.4f})  "
              f"{'wrote ' + out_path if wrote else 'NOT written'}", flush=True)
    del data, raw_kept, X_tr, X_te, sd
    gc.collect()
    return row


def _merge_summary(out_root: str, rows: List[dict]) -> str:
    """Update the per-subject build summary without dropping earlier subjects."""
    path = os.path.join(out_root, SUMMARY_NAME)
    old = []
    if os.path.exists(path):
        try:
            with open(path) as f:
                old = json.load(f)
        except (OSError, ValueError):
            old = []
    by_sub = {r.get("subject"): r for r in old if isinstance(r, dict)}
    by_sub.update({r["subject"]: r for r in rows})
    os.makedirs(out_root, exist_ok=True)
    with open(path, "w") as f:
        json.dump([by_sub[s] for s in sorted(by_sub)], f, indent=2)
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build native 1 kHz Stanford windows aligned to the CORTEG pickles.")
    ap.add_argument("--subjects", nargs="+", default=SUBJECTS,
                    help="space- or comma-separated (default: all nine)")
    ap.add_argument("--data_root", "--pkl_root", dest="data_root", default="",
                    help="folder of the CORTEG <sub>_features.pkl, as the runners' --data_root "
                         "(default paths.get_data_root()); the defaults of --raw_root and "
                         "--out_root are relative to it")
    ap.add_argument("--raw_root", default="",
                    help="folder of <sub>/<sub>_fingerflex.mat "
                         "(default $CORTEG_STANFORD_RAW_ROOT, else <data_root>/raw/Stanford)")
    ap.add_argument("--out_root", default="",
                    help="output folder (default $CORTEG_NATIVE1K_ROOT, else "
                         "<data_root>/native_1k/built, where the runners look)")
    args = ap.parse_args(argv)
    subjects = parse_subjects(args.subjects)

    # Resolved exactly as the runners resolve them from their --data_root.
    raw_dir = raw_root(args.raw_root, args.data_root)
    pkl_root = get_data_root(args.data_root)
    out_root = native_root(args.out_root, args.data_root)
    rows = [process_subject(s, raw_dir, pkl_root, out_root) for s in subjects]
    path = _merge_summary(out_root, rows)
    print(f"\nsummary: {path}")
    bad = [r["subject"] for r in rows if not r["wrote"]]
    if bad:
        print(f"NOT written (failed a check): {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
