"""BrainTreebank: loading, feature extraction, and leakage-free temporal splits.

BrainTreebank (Wang et al., 2024) is public sEEG from 10 patients watching films,
with a word-level transcript. The paper scores two binary tasks:

* **Task A**, sentence-initial vs mid-sentence word (``load_events``). Both
  classes are words. Upstream BrainBERT/PopT's ``sentence_onset`` task is a
  different contrast: sentence-initial words against silence.
* **Task B**, word vs non-word (``word_nonword_events``), the BrainBERT/PopT
  benchmark: word onsets against the centres of 1 s word-free tiles.

Both keep at most ``max_per_class`` events per class, drawn with a fixed seed.

The raw recordings are ~52 GB and are not redistributed here. Download them from
https://braintreebank.dev/ and point ``BTB_DATA_ROOT`` at the directory holding
``all_subject_data/``, ``electrode_labels/``, ``localization/``,
``subject_metadata/``, ``subject_timings/`` and ``transcripts/``.

Three things this module is careful about, each of which was a real bug:

1. **Sampling rate is measured, not assumed.** Nine subjects run at a nominal 2048 Hz
   (measured 2038-2050 Hz) but ``sub_9`` runs at about 1019 Hz. Hardcoding 2048 silently mis-scales every window
   length and every frequency band for that subject.
2. **Events outside the trigger range are dropped, not clamped.** ``np.interp``
   clamps out-of-range inputs to the end value, which collapses every late event
   onto one identical window -- 47 % of ``sub_6``'s events, with both labels.
   ``build_time_to_sample`` returns NaN instead so callers drop them.
3. **Splits are strictly causal with an embargo.** See ``forward_chaining_split``.

The feature transform matches the Stanford pipeline so the same CORTEG model runs
on both datasets: a low stream at 128 Hz and a high-frequency-activity envelope
at 200 Hz, tokenised in patches of 16 and 25 samples. Over CORTEG's 1.5 s window
that is 192 and 300 samples, i.e. 12 tokens per electrode from each stream.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

import paths


def btb_root() -> str:
    """Root of the BrainTreebank download (``BTB_DATA_ROOT``)."""
    return os.environ.get(
        "BTB_DATA_ROOT", os.path.expanduser("~/workspace/datasets/braintreebank"))


def btb_output_root() -> str:
    """Where BrainTreebank runs write caches and results."""
    return os.path.join(paths.get_output_root(), "braintreebank")


# =========================================================================
# Raw data loading
# =========================================================================

def load_electrode_map(root, subj):
    with open(f"{root}/electrode_labels/{subj}/electrode_labels.json", encoding="utf-8") as fh:
        names = json.load(fh)
    return {n: i for i, n in enumerate(names)}, names


def load_localization_mni(root, subj):
    """name -> (X, Y, Z) float coords in MILLIMETRES, SHARED MNI frame.

    Read from ``<root>/localization/elec_coords_full.csv`` (one combined file for
    ALL subjects), filtering rows where ``Subject == subj`` and keying by the
    ``Electrode`` column.  The CSV column ORDER is ``Z,X,Y`` but the columns are
    used here by NAME as (X, Y, Z).  Coordinates are MNI millimetres (range
    ~[-85, 107]), so unlike ``load_localization`` they ARE co-registered across
    subjects and can align channels in pooled training.
    """
    coords = {}
    with open(f"{root}/localization/elec_coords_full.csv") as f:
        for row in csv.DictReader(f):
            if row.get("Subject") != subj:
                continue
            try:
                coords[row["Electrode"]] = (
                    float(row["X"]),
                    float(row["Y"]),
                    float(row["Z"]),
                )
            except (ValueError, KeyError):
                continue
    return coords


def load_localization(root, subj):
    """name -> (L, I, P) integer voxel coordinates, per-subject FreeSurfer frame.

    These are NOT co-registered across subjects, so they cannot align channels in
    pooled training -- use ``load_localization_mni`` for that. They are here
    because the intracranial-FM arms select and re-reference electrodes in this
    frame, and the published FM numbers depend on that choice: the voxel and MNI
    electrode sets differ in count on 4 of the 10 subjects.
    """
    coords = {}
    with open(f"{root}/localization/{subj}/depth-wm.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                coords[row["Electrode"]] = (
                    int(round(float(row["L"]))),
                    int(round(float(row["I"]))),
                    int(round(float(row["P"]))),
                )
            except (ValueError, KeyError):
                continue
    return coords


def build_time_to_sample(root, subj, trial):
    """Interp movie_time(sec) -> h5 sample index from the trial timings csv."""
    mt, idx = [], []
    with open(f"{root}/subject_timings/{subj}_{trial}_timings.csv") as f:
        for row in csv.DictReader(f):
            try:
                mt.append(float(row["movie_time"]))
                idx.append(float(row["index"]))
            except (ValueError, KeyError):
                continue
    mt, idx = np.asarray(mt), np.asarray(idx)
    order = np.argsort(mt)
    mt, idx = mt[order], idx[order]

    def t2s(t):
        """movie_time -> h5 sample. Returns NaN OUTSIDE the trigger range.

        `np.interp` clamps out-of-range inputs to the end values, which silently collapsed every
        late event onto one identical window: sub_6 844/1800 events (47%) with BOTH labels, sub_10
        323, sub_2 30, sub_9 17. Upstream drops such words (`trial_data_reader.py:61-64`). We return
        NaN so callers can drop them explicitly instead of fabricating duplicates.
        """
        t = np.asarray(t, dtype=np.float64)
        out = np.interp(t, mt, idx)
        return np.where((t >= mt[0]) & (t <= mt[-1]), out, np.nan)

    t2s.movie_time_range = (float(mt[0]), float(mt[-1]))
    return t2s


def estimate_fs(root, subj, trial):
    """Sampling rate from the MEDIAN LOCAL SLOPE of index vs movie_time.

    Robust to gaps in the timings file (a whole-span estimate is not: it reads 3553 Hz for sub_6).
    Verified: nine subjects are 2048 Hz, sub_9 is 1019 Hz. Hardcoding 2048 silently mis-scaled every
    window length and frequency band for sub_9.
    """
    mt, idx = [], []
    with open(f"{root}/subject_timings/{subj}_{trial}_timings.csv") as f:
        for row in csv.DictReader(f):
            try:
                mt.append(float(row["movie_time"])); idx.append(float(row["index"]))
            except (ValueError, KeyError):
                continue
    mt, idx = np.asarray(mt), np.asarray(idx)
    o = np.argsort(mt); mt, idx = mt[o], idx[o]
    dm, di = np.diff(mt), np.diff(idx)
    ok = dm > 1e-6
    return float(np.median(di[ok] / dm[ok]))


def load_events(root, movie, max_per_class, seed):
    """Task A events: (starts_sec, labels), balanced, in temporal order.

    Every timed transcript word is a candidate, labelled 1 if it begins a
    sentence (``is_onset``) and 0 otherwise, so both classes are words. At most
    ``max_per_class`` of each class are drawn with ``RandomState(seed)``.
    """
    starts, lab = [], []
    with open(f"{root}/transcripts/{movie}/features.csv") as f:
        for row in csv.DictReader(f):
            try:
                s = float(row["start"])
                o = float(row["is_onset"])
            except (ValueError, KeyError):
                continue
            if not np.isfinite(s):
                continue
            starts.append(s)
            lab.append(1 if o >= 0.5 else 0)
    starts, lab = np.asarray(starts), np.asarray(lab)
    rng = np.random.RandomState(seed)
    pos = np.where(lab == 1)[0]
    neg = np.where(lab == 0)[0]
    k = min(len(pos), len(neg), max_per_class)
    pos = rng.permutation(pos)[:k]
    neg = rng.permutation(neg)[:k]
    keep = np.sort(np.concatenate([pos, neg]))  # keep temporal order for split
    return starts[keep], lab[keep]


def extract_windows(root, subj, trial, ch_idx, starts_sec, t2s, fs, pre_sec, win_sec):
    """Load (N, C, T) raw windows. Reads needed electrodes fully into RAM, then slices."""
    import h5py

    win = int(round(win_sec * fs))
    pre = int(round(pre_sec * fs))
    centers_f = np.asarray(t2s(starts_sec), dtype=np.float64)
    in_range = np.isfinite(centers_f)          # NaN => event outside the trigger range
    centers = np.where(in_range, centers_f, 0).astype(np.int64)
    s0 = centers + pre
    h5 = h5py.File(f"{root}/all_subject_data/{subj}_{trial}.h5", "r")
    T_total = h5["data/electrode_0"].shape[0]
    valid = in_range & (s0 >= 0) & (s0 + win <= T_total)
    s0 = s0[valid]
    N = len(s0)
    C = len(ch_idx)
    x = np.empty((N, C, win), dtype=np.float32)
    for ci, ei in enumerate(ch_idx):
        sig = h5[f"data/electrode_{ei}"][:]  # full electrode signal in RAM
        for ni, s in enumerate(s0):
            x[ni, ci] = sig[s : s + win]
        del sig
    h5.close()
    return x, valid


# ---------------------------------------------------------------------------
# Task B: word vs non-word
#
# The second BrainTreebank endpoint, and the upstream Population Transformer
# benchmark. Positives are word onsets; negatives are non-overlapping 1 s tiles
# intersecting no word. Both are reported as the WINDOW CENTRE, because upstream
# centres its 5 s window. An arm reading that centred 5 s window has a 5.0 s
# overlap footprint; CORTEG reads [centre, centre+1.5], so its footprint stays
# 1.5 s. Candidates are filtered with the 5 s window either way, so every arm
# scores the same events. Transcribed from PopulationTransformer's
# data/trial_data_reader.py.
# ---------------------------------------------------------------------------
TILE_SEC = 1.0        # upstream interval_duration
WIN_SEC = 5.0         # upstream duration (centred)


def _words(root, movie):
    lo, hi = [], []
    with open(f"{root}/transcripts/{movie}/features.csv") as f:
        for r in csv.DictReader(f):
            try:
                a, b = float(r["start"]), float(r["end"])
            except (ValueError, KeyError, TypeError):
                continue
            if np.isfinite(a) and np.isfinite(b) and b > a:
                lo.append(a); hi.append(b)
    o = np.argsort(lo)
    return np.asarray(lo)[o], np.asarray(hi)[o]


def _trigger_range(root, subj, trial):
    mt = []
    with open(f"{root}/subject_timings/{subj}_{trial}_timings.csv") as f:
        for r in csv.DictReader(f):
            try:
                mt.append(float(r["movie_time"]))
            except (ValueError, KeyError):
                continue
    return float(min(mt)), float(max(mt))


def word_nonword_events(root, subj, trial, movie, max_per_class, seed,
                        tile_sec=TILE_SEC, win_sec=WIN_SEC, neg_mode="upstream",
                        max_silence_sec=10.0):
    """Return (event_times, labels, meta). Event time is the WINDOW CENTRE (upstream is centred).

    Positives: word onsets (centre = onset, matching upstream's est_idx alignment).
    Negatives: non-overlapping `tile_sec` tiles intersecting no word; centre = tile centre.

    `win_sec` only filters candidates: an event is kept if the `win_sec` window
    centred on it fits inside the trigger range. It is not the window a model
    reads. Keep the default 5.0 s for every arm, CORTEG's 1.5 s arm included:
    that is what the paper runs did, and it gives all arms the same events.

    `neg_mode="upstream"` (the paper) draws negatives uniformly from all
    word-free tiles, so many fall in long silences such as credits;
    `"short_silence"` draws them only from pauses shorter than
    `max_silence_sec`. `meta` reports the fraction of selected negatives in
    silences of 20 s or more, and of more than 60 s.
    """
    w_lo, w_hi = _words(root, movie)
    if w_lo.size == 0:
        raise RuntimeError(f"no word intervals for {movie}")
    t0, t1 = _trigger_range(root, subj, trial)
    half = win_sec / 2.0

    # positives: word onsets whose FULL centred window lies inside the trigger range
    pos = w_lo[(w_lo - half >= t0) & (w_lo + half <= t1)]

    # negatives: NON-OVERLAPPING tiles (upstream tiles, not a sliding grid)
    n_tiles = int(np.floor((t1 - t0) / tile_sec))
    tl = t0 + np.arange(n_tiles) * tile_sec
    th = tl + tile_sec
    # keep a tile iff it intersects NO word: a[0] < b[1] and a[1] > b[0]
    j_hi = np.searchsorted(w_lo, th)            # words starting before tile end
    j_lo = np.searchsorted(w_hi, tl, "right")   # words ending after tile start
    clean = j_hi <= j_lo
    ctr = (tl + th) / 2.0
    neg_all = ctr[clean]
    # the 5 s window about the centre must still fit the trigger range
    neg_all = neg_all[(neg_all - half >= t0) & (neg_all + half <= t1)]

    # how concentrated are the candidates in long silences? (the confound we must report)
    def silence_len(c):
        i = np.searchsorted(w_hi, c, "right")
        left = w_hi[i - 1] if i > 0 else t0
        j = np.searchsorted(w_lo, c)
        right = w_lo[j] if j < w_lo.size else t1
        return right - left
    sil = np.array([silence_len(c) for c in neg_all]) if neg_all.size else np.array([])

    rng = np.random.RandomState(seed)
    k = min(pos.size, neg_all.size, int(max_per_class))
    if k < 50:
        raise RuntimeError(f"{subj}/{movie}: too few events (pos={pos.size} neg={neg_all.size} "
                           f"max_per_class={max_per_class}); Task B needs at least 50 per class")

    if neg_mode == "short_silence" and sil.size:
        # SECONDARY arm: negatives drawn ONLY from SHORT pauses (< max_silence_sec), so the class is
        # brief inter-speech gaps rather than end credits. This is the contrast that isolates
        # speech-vs-silence from "detect the quiet part of the film".
        #
        # NOTE: an earlier attempt sampled evenly across silence-length RANK deciles. That is a
        # no-op -- equal-size rank bins sampled equally reproduce the original distribution exactly
        # (measured: 0.40 -> 0.39 of negatives in >=20 s silences). Restricting by DURATION works.
        eligible = np.where(sil < max_silence_sec)[0]
        if eligible.size < 50:
            raise RuntimeError(f"only {eligible.size} negatives in silences < {max_silence_sec}s")
        k = min(k, eligible.size)
        neg = np.sort(neg_all[rng.permutation(eligible)[:k]])
    else:
        neg = np.sort(rng.permutation(neg_all)[:k])
    pos = np.sort(rng.permutation(pos)[:k])

    times = np.concatenate([pos, neg])
    labels = np.concatenate([np.ones(k), np.zeros(k)])
    o = np.argsort(times)                       # temporal order, as every other arm uses
    sel_sil = np.array([silence_len(c) for c in neg])
    meta = {
        "neg_mode": neg_mode, "tile_sec": tile_sec, "win_sec": win_sec,
        "n_pos_available": int(pos.size), "n_neg_available": int(neg_all.size),
        "trigger_range": [t0, t1],
        "neg_in_silence_ge20s_frac": float(np.mean(sel_sil >= 20)) if sel_sil.size else None,
        "neg_in_silence_gt60s_frac": float(np.mean(sel_sil > 60)) if sel_sil.size else None,
        "neg_silence_len_median": float(np.median(sel_sil)) if sel_sil.size else None,
    }
    return times[o], labels[o], meta


# =========================================================================
# CORTEG feature transform (matches the Stanford pipeline)
# =========================================================================

# ---------------------------------------------------------------------------
NOTCH_FREQS = (60.0, 120.0, 180.0)
NOTCH_BW = 2.0          # band-stop half-width (Hz)
NOTCH_ORDER = 3         # 3rd-order Butterworth band-stop (Stanford recipe)
LO_FS = 128             # LOW-stream target rate (Stanford X_raw is 128 Hz)
HI_FS = 200             # HIGH-stream target rate (Stanford X_feat is 200 samples/sec)
# High-frequency-activity band. Default WIDENED to 70-200 Hz (broadband HFA) for
# speech/language sEEG — the old Stanford 60-124 Hz cap was a 1 kHz-Nyquist limit;
# BrainTreebank @ ~2048 Hz (Nyquist ~1024) has ample headroom. Start at 70 (excludes
# 60 Hz mains); the raw is notch-filtered (60/120/180) BEFORE the bandpass so the
# 120/180 Hz line harmonics inside the wider band are removed.
HGA_LOW, HGA_HIGH = 70.0, 200.0   # high-frequency-activity analytic band (override via CLI)
HGA_ORDER = 4           # 4th-order Butterworth band-pass


def _resample_factors(fs_in: int, fs_out: int):
    """Integer (up, down) for resample_poly so fs_in*up/down == fs_out."""
    from math import gcd
    g = gcd(int(fs_in), int(fs_out))
    up = int(fs_out) // g
    down = int(fs_in) // g
    return up, down


def corteg_features(x_raw: np.ndarray, fs: float,
                    hga_low: float = HGA_LOW, hga_high: float = HGA_HIGH):
    """CORTEG hi/lo transform. HFA band (hga_low..hga_high) is configurable.

    Args:
        x_raw: (N, C, T) raw BTB windows at ``fs`` Hz.
        fs:    sampling rate of x_raw (Hz).
        hga_low, hga_high: HIGH-stream band-pass edges (Hz); default 70-200
            (broadband HFA for speech/language; was 60-124 for Stanford motor).

    Returns:
        x_lo: (N, C, T_lo)  LOW stream  @ LO_FS (128 Hz)
        x_hi: (N, C, T_hi)  HIGH stream @ HI_FS (200 Hz), |hilbert envelope|

    The raw is notch-filtered (60/120/180 Hz) FIRST; BOTH streams derive from the
    notched signal, so the line harmonics inside the wider HFA band are removed.
    LOW : notch -> CAR (per-sample channel mean) -> polyphase resample to 128 Hz.
    HIGH: notch -> bandpass [hga_low,hga_high] -> hilbert -> abs -> resample 200 Hz.
    Filtering is applied along the time axis on the full (N, C, T) batch.
    """
    from scipy.signal import butter, filtfilt, hilbert, resample_poly

    fs = float(fs)
    nyq = fs / 2.0
    x = np.asarray(x_raw, dtype=np.float64)  # (N, C, T)

    # ---- notch line noise on the RAW (shared by both streams) ----
    xn = x
    for f0 in NOTCH_FREQS:
        if (f0 + NOTCH_BW) >= nyq:
            continue  # harmonic outside Nyquist
        b, a = butter(
            NOTCH_ORDER,
            [(f0 - NOTCH_BW) / nyq, (f0 + NOTCH_BW) / nyq],
            btype="bandstop",
        )
        xn = filtfilt(b, a, xn, axis=-1)

    # ---- LOW stream ----
    lo = xn - xn.mean(axis=1, keepdims=True)  # CAR
    up_lo, down_lo = _resample_factors(int(round(fs)), LO_FS)
    x_lo = resample_poly(lo, up_lo, down_lo, axis=-1)  # (N, C, T_lo @128Hz)

    # ---- HIGH stream (HFA analytic envelope) ----
    hi_hi = min(float(hga_high), nyq - 1.0)            # clamp below Nyquist
    b, a = butter(HGA_ORDER, [float(hga_low) / nyq, hi_hi / nyq], btype="bandpass")
    hi = filtfilt(b, a, xn, axis=-1)
    hi = np.abs(hilbert(hi, axis=-1))                  # envelope
    up_hi, down_hi = _resample_factors(int(round(fs)), HI_FS)
    x_hi = resample_poly(hi, up_hi, down_hi, axis=-1)  # (N, C, T_hi @200Hz)

    return x_lo.astype(np.float32), x_hi.astype(np.float32)



# ===========================================================================
# Leakage-free temporal splits
# ===========================================================================
# An arm reading [t + pre, t + pre + win] around event time t has footprint
# width `win`. Events at t_i < t_j overlap iff t_j - t_i < win -- the rule
# depends only on `win`, not `pre`, since shifting both windows equally cannot
# separate them.
#
# The embargo is sized once by the widest arm sharing the split, so that every
# arm sees the same train/test membership and the comparison stays fair:
#
#     Task A probes, CORTEG           1.5 s   [t, t+1.5]
#     Task B, CORTEG                  1.5 s   [c, c+1.5], c = window centre
#     Task B, upstream-style arms     5.0 s   [c-2.5, c+2.5]
#     Brant L=1                       6.11 s  6 s ending at the task window's right edge
#                                             ([t-4.5, t+1.5] Task A, [t-3.5, t+2.5] Task B)
#                                             + resample FIR edge
#     ----------------------------------------------------------------
#     EMBARGO_SEC = 7.0               widest (6.11) plus margin

# Widest shared-arm footprint (Brant L=1 = 6.11 s incl. resample FIR edge) plus margin.
EMBARGO_SEC = 7.0

# Per-arm footprints. Callers pass their own; this is the reference table.
ARM_FOOTPRINT_SEC = {
    # [t, t+1.5]. No pre-roll: one straddled the pause before sentence-initial
    # words and made Task A partly a pause detector.
    "sentence_onset": 1.5,
    "word_nonword_corteg": 1.5,     # [c, c+1.5], CORTEG's own window on Task B
    "word_nonword_upstream": 5.0,   # [c-2.5, c+2.5] -- upstream convention, for comparability
    # 6 s ending at the task window's right edge ([t-4.5, t+1.5] Task A, [t-3.5, t+2.5]
    # Task B) + resample FIR edge; sets the embargo
    "brant_L1": 6.11,
}


def _pairs_within(times, win_sec: float, a_idx, b_idx) -> int:
    """Count (i in a, j in b) pairs with |t_i - t_j| < win_sec. Sorted two-pointer sweep."""
    t = np.asarray(times, dtype=np.float64)
    ta = np.sort(t[np.asarray(a_idx, dtype=int)])
    tb = np.sort(t[np.asarray(b_idx, dtype=int)])
    if ta.size == 0 or tb.size == 0:
        return 0
    lo = np.searchsorted(tb, ta - win_sec, side="right")
    hi = np.searchsorted(tb, ta + win_sec, side="left")
    return int(np.sum(np.maximum(0, hi - lo)))


def assert_no_window_overlap(times, win_sec: float, a_idx, b_idx, label: str = "") -> int:
    """Raise unless ZERO cross-block pairs have overlapping windows. Returns the count (0).

    `win_sec` is the ARM's true footprint and must be supplied by the caller. It is deliberately not
    defaulted to the embargo: doing so would make this check a tautology.
    """
    if win_sec is None or not np.isfinite(win_sec) or win_sec <= 0:
        raise ValueError("win_sec must be the arm's true footprint in seconds, > 0")
    n = _pairs_within(times, win_sec, a_idx, b_idx)
    if n:
        raise AssertionError(
            f"{label or 'split'}: {n} cross-block pairs within {win_sec}s -- this split LEAKS. "
            f"Either the embargo is smaller than this arm's footprint, or the split was computed on "
            f"a different event list than the one being scored."
        )
    return 0


def _check_embargo(win_sec: float, embargo_sec: float) -> None:
    if win_sec > embargo_sec:
        raise ValueError(
            f"arm footprint {win_sec}s exceeds embargo {embargo_sec}s -- the split cannot be "
            f"leakage-free for this arm. Raise EMBARGO_SEC or exclude the arm from the shared split."
        )


def assert_valid_times(times, n_expected: int | None = None) -> np.ndarray:
    """Guard against a real bug: the split MUST be built on the events actually scored.

    Pass the event times AFTER the trigger-range / window-bounds mask, never the raw transcript list.
    """
    t = np.asarray(times, dtype=np.float64)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("times must be a non-empty 1-D array")
    if not np.all(np.isfinite(t)):
        raise ValueError("times contains non-finite values -- apply the validity mask first")
    if n_expected is not None and t.size != n_expected:
        raise ValueError(
            f"times has {t.size} events but {n_expected} will be scored -- split and scoring "
            f"disagree (a mismatch once cost sub_6 100% of its test block this way)"
        )
    return t


def forward_chaining_split(times, win_sec: float, n_folds: int = 4,
                           embargo_sec: float = EMBARGO_SEC, min_train_frac: float = 0.2,
                           val_frac: float = 0.0):
    """PRIMARY scheme: strictly causal forward chaining (expanding window).

    The session is cut into `n_folds + 1` contiguous blocks in time. Fold i trains on blocks
    0..i and tests on block i+1, with an embargo removed from the end of train. Every fold satisfies
    t[train].max() < t[test].min(), so no model ever sees signal recorded after its test data.

    Chosen over a single 60/15/25 cut because that gave only 450 test events per subject and dropped
    power for the headline FM contrast from 0.971 to 0.706, while shifting the estimand to
    "final-quarter discriminability". Forward chaining tests on ~75 % of events,
    recovers the session-average estimand to 0.011 MAE, and keeps power at 0.947.

    With `val_frac > 0` each fold returns (fit_idx, val_idx, test_idx) forming a strictly ordered
    fit | embargo | val | embargo | test chain -- a causal train/val/test split, and the only form
    safe for early stopping. With `val_frac = 0` it returns (train_idx, test_idx).
    """
    t = assert_valid_times(times)
    _check_embargo(win_sec, embargo_sec)
    order = np.argsort(t, kind="stable")
    ts = t[order]
    n = ts.size
    k = int(n_folds)
    if k < 1:
        raise ValueError("n_folds must be >= 1")
    bounds = np.linspace(0, n, k + 2).round().astype(int)
    out = []
    for i in range(k):
        tr_end, te_end = bounds[i + 1], bounds[i + 2]
        te_pos = np.arange(tr_end, te_end)
        if te_pos.size == 0:
            continue
        t_test0 = ts[tr_end]
        pool = np.arange(0, tr_end)
        pool = pool[ts[pool] < t_test0 - embargo_sec]         # embargo before test
        if pool.size < max(10, int(min_train_frac * n * (i + 1) / (k + 1))):
            continue                                          # too little history to train on

        if val_frac and val_frac > 0:
            # Causal val block: the TAIL of the history, embargoed from BOTH fit and test, giving
            # fit | embargo | val | embargo | test. A random val carve (e.g. rng.permutation)
            # would interleave val with fit, so early stopping would select on overlapping windows --
            # model-selection leakage even when train/test is clean.
            n_val = max(1, int(round(val_frac * pool.size)))
            va_pos = pool[-n_val:]
            t_val0 = ts[va_pos[0]]
            fit_pos = pool[:-n_val]
            fit_pos = fit_pos[ts[fit_pos] < t_val0 - embargo_sec]
            if fit_pos.size < 10:
                continue
            tr, va, te = order[fit_pos], order[va_pos], order[te_pos]
            assert_no_window_overlap(t, win_sec, tr, te, f"fold {i} fit/test")
            assert_no_window_overlap(t, win_sec, tr, va, f"fold {i} fit/val")
            assert_no_window_overlap(t, win_sec, va, te, f"fold {i} val/test")
            if not (t[tr].max() < t[va].min() < t[va].max() < t[te].min()):
                raise AssertionError(f"fold {i}: fit<val<test ordering violated")
            out.append((tr, va, te))
        else:
            tr, te = order[pool], order[te_pos]
            assert_no_window_overlap(t, win_sec, tr, te, f"forward-chain fold {i}")
            if not (t[tr].max() < t[te].min()):
                raise AssertionError(f"forward-chain fold {i}: causality violated")
            out.append((tr, te))
    if not out:
        raise ValueError(f"no usable folds for n={n} (too few events for a causal split)")
    return out


def assert_brant_fs_fixed(measured_fs: float, nominal_fs: float = 2048.0,
                          tol: float = 30.0) -> None:
    """Brant must resample from the MEASURED rate, never a hardcoded 2048 Hz.

    sub_9 records at ~1019 Hz. Treating it as 2048 makes Brant's 250 Hz stream
    124.4 Hz and its 1500-sample patch span 12.06 s rather than 6.0 -- nearly
    twice the embargo, and invisible to the overlap check, which is told the
    footprint is 6.0 s.
    """
    if abs(measured_fs - nominal_fs) > tol:
        raise AssertionError(
            f"measured fs {measured_fs:.0f} Hz != nominal {nominal_fs:.0f} Hz. Brant's "
            f"resample must use the measured rate or its true footprint becomes "
            f"{6.0 * nominal_fs / measured_fs:.2f}s, exceeding EMBARGO_SEC={EMBARGO_SEC}s.")


def split_report(times, win_sec: float, train_idx, test_idx, val_idx=None, scheme: str = "") -> dict:
    """Machine-readable provenance to embed in every results JSON."""
    t = np.asarray(times, dtype=np.float64)
    rep = {
        "scheme": scheme, "win_sec": float(win_sec), "embargo_sec": float(EMBARGO_SEC),
        "n_events_scored": int(t.size), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "overlapping_train_test_pairs": _pairs_within(t, win_sec, train_idx, test_idx),
        "train_time_range": [float(t[train_idx].min()), float(t[train_idx].max())] if len(train_idx) else None,
        "test_time_range": [float(t[test_idx].min()), float(t[test_idx].max())] if len(test_idx) else None,
    }
    if val_idx is not None and len(val_idx):
        rep["n_val"] = int(len(val_idx))
        rep["overlapping_train_val_pairs"] = _pairs_within(t, win_sec, train_idx, val_idx)
        rep["overlapping_val_test_pairs"] = _pairs_within(t, win_sec, val_idx, test_idx)
    rep["causal"] = bool(len(train_idx) and len(test_idx)
                         and t[train_idx].max() < t[test_idx].min())
    return rep


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    # 1. the check must be ABLE to fail (not a tautology)
    tt = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    try:
        assert_no_window_overlap(tt, 5.0, [0, 1, 2], [3, 4, 5], "deliberate")
        raise SystemExit("FAIL: overlap check did not fire on overlapping blocks")
    except AssertionError:
        pass
    # 2. footprint wider than the embargo must be refused, not silently passed
    try:
        forward_chaining_split(np.sort(rng.uniform(0, 500, 400)), win_sec=12.22, embargo_sec=7.0)
        raise SystemExit("FAIL: accepted an arm footprint wider than the embargo")
    except ValueError:
        pass
    # 3. non-finite / raw-list guards
    try:
        assert_valid_times(np.array([0.0, np.nan, 2.0]))
        raise SystemExit("FAIL: accepted non-finite times")
    except ValueError:
        pass
    # 4. randomised: causality + zero overlap at each arm's own footprint
    for _ in range(300):
        n = int(rng.integers(80, 900))
        tt = np.sort(rng.uniform(0, n * rng.uniform(0.5, 4.0), n))
        for w in (1.0, 2.0, 5.0, 6.11):
            for tr, te in forward_chaining_split(tt, w, 4):
                assert tt[tr].max() < tt[te].min()
                assert _pairs_within(tt, w, tr, te) == 0
    # 5. brute-force cross-check of the counter
    for _ in range(300):
        n = int(rng.integers(10, 120))
        tt = np.sort(rng.uniform(0, 50, n))
        a = rng.choice(n, size=n // 3, replace=False)
        b = np.setdiff1d(np.arange(n), a)
        w = float(rng.uniform(0.2, 8))
        assert _pairs_within(tt, w, a, b) == sum(1 for i in a for j in b if abs(tt[i] - tt[j]) < w)
    print("BrainTreebank split self-test PASSED "
          "(negative controls fire; 300 trials x 4 footprints; 300 brute-force checks)")
