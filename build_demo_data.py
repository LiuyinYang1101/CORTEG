"""Build the JSON files behind the docs/ interactive demo.

For every subject and model the demo plots a prediction trace against the
target and prints Pearson r on that subject's held-out test split. Each model
points at the run that produced the paper's number, and for those rows r is
read from that run's own results file, so the demo agrees with the per-subject
tables in the appendix to the printed precision. A model that is not a paper
row says so in its label.

Stanford finger flexion (public data, n=9):
  CORTEG pooled            Table 1, 0.554 (the exact cohort mean is 0.553487;
                           see paper_rounding).
  CORTEG per-subject       Table 1, 0.539 (the batch-size-16 per-subject run).
  Random init, full FT,    Table 2's random-init row (full fine-tuning, the
    seed 7                 table's star footnote). The paper's 0.510 is the
                           mean of seeds 7 (0.511, shown) and 123 (0.509); a
                           seed mean has no single set of predictions to plot.
                           A seed-42 run of the same configuration (seed 42
                           is what every other trained row uses) scored 0.554
                           and is not in the paper's mean; with it, the
                           three-seed mean is 0.525. The builder prints the
                           two-seed and the three-seed mean.
  HiLoFuseNet re-run       Not a paper value. Table 1's 0.534 is transcribed
                           from Sun et al.; this is our seed-42 re-run of the
                           architecture with predictions saved (0.529).
  Ridge_HGA, Ridge_LFS     Table 1, 0.336 and 0.181. Those runs saved no
                           predictions, so they are recomputed here from the
                           Stanford feature files with the paper's recipe
                           (ridge on the per-channel mean and std of the
                           z-scored stream). RidgeCV is closed-form, so this
                           reproduces the paper's per-subject r exactly.

Ghent speech envelope (private data, n=16, shown for visualisation only):
  CORTEG pooled 0.339, CORTEG per-subject 0.250 and Random init, full FT 0.156
  are the Table 1 / Table 2 runs. The HiLoFuseNet re-run, 0.249, is not a
  paper value. It is a seed-42 re-implementation with predictions saved, with
  the same data, model, optimiser and schedule as Table 1's runner but without
  its gradient clipping, its early-stopping min_delta or its cuDNN
  deterministic mode. Table 1's 0.259 comes from that runner (also seed 42),
  which saved no predictions. The two runs differ in recipe and the re-run is
  not bit-reproducible, so 0.249 is not a reproduction of 0.259, and a single
  re-run cannot say how much of the gap is the recipe and how much is
  run-to-run noise.

Not shown: PLS and the Ghent Ridge. Table 1's Stanford PLS is transcribed from
Sun et al. and its Ghent PLS run saved no predictions; the simplified local
re-runs an earlier version of this demo showed (0.276 / 0.108) matched no paper
row. Table 1's Ghent Ridge is scored continuously over every time point, not on
the windowed test split the demo plots.

Nothing is written unless every check passes. Every configured row must load
for every subject; a paper row's r must come from its run's single results
file and agree, per channel, with the r recomputed from its saved predictions;
every row's target must line up in time with the displayed one; and each paper
row's cohort mean must round to the paper's value. The output directories are
checked before the slow part of the build: the builder replaces only a
<dataset>/ directory it wrote itself (regular *.json files, manifest.json among
them) and refuses to touch anything else. On any failure the builder lists
every problem and exits with status 1, leaving the existing files as they were.
Every dataset is first written to a hidden sibling directory; only when all of
them are written are they swapped in, one after another. Each dataset is
swapped in whole, so no <dataset>/ directory ever holds a mix of old and new
files; a run stopped between two swaps leaves whole datasets from two builds,
and the next run replaces both.

This is an author-side script. The run outputs it reads are not part of the
release, so point CORTEG_OUTPUT_ROOT (or --output_root) at them; the Ridge rows
also need the Stanford feature files (CORTEG_DATA_ROOT or --data_root). With
--allow_missing, rows whose inputs are absent altogether are dropped instead of
failing the build; that is for previews written to --out_dir, and is refused
for the shipped docs/data.

    CORTEG_OUTPUT_ROOT=/path/to/run/outputs python build_demo_data.py
"""
import argparse
import json
import shutil
import tempfile
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from paths import get_data_root, get_output_root

REPO = Path(__file__).resolve().parent
DOCS_DATA = REPO / "docs" / "data"
N_DISPLAY = 1000            # points per trace in the shipped JSON

STANFORD_SUBJECTS = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]
GHENT_SUBJECTS = ["2018_001", "2019_001", "2019_003", "2019_004", "2019_007",
                  "2020_001", "2020_002", "2020_004", "2020_005", "2020_006",
                  "2021_001", "2021_002", "2021_005-1", "2021_006", "2021_007",
                  "2021_008-1"]

# Spacing of consecutive test windows, which places each run's windows in time.
# Stanford windows advance 40 ms (fs_dg = 25 Hz in data/stanford_preprocessing).
# The CORTEG Ghent loader asks for 50 ms but steps a whole number of 128 Hz
# samples, int(6.4) = 6, i.e. 46.875 ms. The Ghent HiLoFuseNet re-run steps
# 50 ms on its 200 Hz grid (its `step_s` below), so it has fewer windows (3580
# vs 3819) over the same 1-180 s of each test split. The page draws every row on
# the first row's time axis, so a row on another grid is shown at its window
# nearest in time to each displayed point: at most half its step (25 ms) off.
# Picking by index instead drifted by up to 62 ms. build() checks that each
# row's target still lines up with the displayed one.
STANFORD_STEP_S = 1 / 25
GHENT_STEP_S = 6 / 128

# `run` is relative to the output root and must contain predictions/<subject>/
# {y_true,y_pred}.npy plus one results*.json. `paper` is the value the paper
# prints for this row (None when the row is not a paper value); when the paper
# averages several seeds, `seed_mean_of` lists them so the check below can
# compare the paper against that mean rather than against the seed shown, and
# `seeds_not_in_paper` lists runs of the same configuration that the paper's
# mean leaves out; their scores are printed next to it but do not enter the
# comparison with the paper.
# `paper_rounding: "via 4 dp"` marks the one row whose paper value is reached
# only from a value first rounded to 4 decimals (see paper_rounding); every
# other row must round to the paper directly. `step_s` overrides the dataset's
# window step for a run on another grid.
#
# The page's badge shows a label up to its first "(", so what distinguishes a
# row (pooled or per-subject, seed, re-run) comes before any parenthesis.
STANFORD_MODELS = [
    {"label": "CORTEG pooled ⭐",
     "run": "stanford_best_lora_adapter",
     "paper": 0.554,
     "paper_rounding": "via 4 dp",
     "note": "Table 1, pooled CORTEG. The exact cohort mean 0.553487 rounds to "
             "0.553; the paper's 0.554 comes from an intermediate rounded value."},
    {"label": "CORTEG per-subject",
     "run": "stanford_persub_pretrained_lora_b16/f1.0",
     "paper": 0.539,
     "note": "Table 1, per-subject CORTEG."},
    # Table 2 prints the mean of seeds 7 and 123 and leaves out the seed-42 run
    # of the same configuration, while its pretrained full-FT row is seed 42
    # alone. If the paper moves to the three-seed mean (0.525), move that run
    # into `seed_mean_of`, set `paper` to 0.525, and show it here instead of
    # seed 7 so that this row uses seed 42 like the others.
    {"label": "Random init, full FT, seed 7",
     "run": "fair_controls_3seed/stanford_random_fullft_adapter_s7",
     "paper": 0.510,
     "seed_mean_of": ["fair_controls_3seed/stanford_random_fullft_adapter_s7",
                      "fair_controls_3seed/stanford_random_fullft_adapter_s123"],
     "seeds_not_in_paper": ["critical_baselines/stanford_fullft_random_adapter"],
     "note": "Table 2 random-init row (full fine-tuning). The paper's 0.510 is "
             "the mean of seeds 7 (shown, 0.511) and 123 (0.509). A seed-42 run "
             "of the same configuration scored 0.554 and is not in that mean; "
             "the mean of all three seeds is 0.525."},
    {"label": "HiLoFuseNet re-run (not a paper value)",
     "run": "stanford_lowdata_baselines/hilofusenet_f1.0_with_preds",
     "paper": None,
     "note": "Not a paper value: Table 1's 0.534 is transcribed from Sun et al. "
             "This is our seed-42 re-run of the architecture."},
    {"label": "Ridge_HGA",
     "ridge": "hi_gamma",
     "paper": 0.336,
     "note": "Table 1, recomputed: ridge on per-channel mean and std of the "
             "z-scored high-gamma stream."},
    {"label": "Ridge_LFS",
     "ridge": "lo_freq",
     "paper": 0.181,
     "note": "Table 1, recomputed: ridge on per-channel mean and std of the "
             "z-scored low-frequency stream."},
]

GHENT_MODELS = [
    {"label": "CORTEG pooled ⭐",
     "run": "ghent_mni_corrected/pretrained_lora_adapter",
     "paper": 0.339,
     "note": "Table 1, pooled CORTEG."},
    {"label": "CORTEG per-subject",
     "run": "ghent_mni_corrected/persub_pretrained",
     "paper": 0.250,
     "note": "Table 1, per-subject CORTEG."},
    {"label": "Random init, full FT",
     "run": "ghent_mni_corrected/random_fullft_adapter",
     "paper": 0.156,
     "note": "Table 2 random-init row (full fine-tuning), seed 42."},
    {"label": "HiLoFuseNet re-run (not a paper value)",
     "run": "ghent_mni_corrected/hilofusenet_persub_with_preds",
     "step_s": 0.05,
     "paper": None,
     "note": "Not a paper value: a seed-42 re-implementation with predictions "
             "saved, with the same data, model, optimiser and schedule as "
             "Table 1's runner but without its gradient clipping, "
             "early-stopping min_delta or cuDNN deterministic mode. Table 1's "
             "0.259 comes from that runner, which saved no predictions; the "
             "re-run is not a reproduction of it."},
]

DATASETS = {
    "stanford": {"subjects": STANFORD_SUBJECTS, "models": STANFORD_MODELS,
                 "step_s": STANFORD_STEP_S, "fingers": FINGERS},
    "ghent": {"subjects": GHENT_SUBJECTS, "models": GHENT_MODELS,
              "step_s": GHENT_STEP_S, "fingers": None},
}

# The r a run wrote to its results file and the r recomputed from its saved
# predictions differ slightly, because the predictions come from a second,
# mixed-precision inference pass after scoring. Over these runs the channel
# mean differs by at most 6.9e-4 and a single finger by at most 2.1e-3 (the
# Stanford pooled run), so the per-finger r printed under a trace can differ
# from the r of the saved predictions by that much. The page shows the stored
# value, the one the paper printed. A larger mean gap would mean the
# predictions are not from the model that was scored; a larger per-channel gap
# would also catch a results file that lists the fingers in another order.
STORED_VS_RECOMPUTED_TOL = 1e-3          # mean over channels
STORED_VS_RECOMPUTED_TOL_CHANNEL = 3e-3  # each channel
# Minimum correlation, per channel, between a row's target at its displayed
# samples and the displayed target. Rows on the first row's grid hold the same
# target up to z-scoring, so they must agree to within rounding: on every
# shipped row 1 - r is below 1e-12. A shift of a single window would not pass,
# since the Stanford targets are smooth: their lag-1 autocorrelation reaches
# 0.997 on one channel (Ghent: 0.71). A row on another grid only approximates
# the displayed samples; the Ghent HiLoFuseNet re-run, whose target comes from
# its own 200 Hz grid, correlates at 0.93, and a misaligned source would be far
# lower.
ALIGN_MIN_CORR_SAME_GRID = 0.9999
ALIGN_MIN_CORR_OTHER_GRID = 0.8
# The page prints r to 3 decimals. Stored at 4, a value such as 0.63354 would be
# rounded twice (0.6335 -> 0.633) and disagree with the paper's 0.634; at 6 the
# page's rounding reproduces every per-subject and per-finger cell of the
# appendix tables that these runs back.
R_DECIMALS = 6


def as_list(arr):
    # float64 first: rounding a float32 array and calling tolist() prints each
    # value's float32 representation error (0.5103999972343445), tripling size.
    return np.asarray(arr, dtype=np.float64).round(4).tolist()


def as_2d(a):
    a = np.asarray(a)
    return a[:, None] if a.ndim == 1 else a


def pearson_per_channel(yt, yp):
    yt, yp = as_2d(yt), as_2d(yp)
    return [float(np.corrcoef(yt[:, i], yp[:, i])[0, 1]) for i in range(yt.shape[1])]


def display_index(n_ref, step_ref, n, step):
    """Indices of the windows shown for a run of `n` windows `step` s apart.

    The first row sets the time axis: N_DISPLAY of its `n_ref` windows, evenly
    spaced by index. A run on the same grid shows the same windows; a run on
    another grid shows, for each displayed time, its window nearest in time.
    """
    idx_ref = np.linspace(0, n_ref - 1, min(n_ref, N_DISPLAY), dtype=int)
    if step == step_ref:
        return idx_ref
    return np.clip(np.rint(idx_ref * step_ref / step).astype(int), 0, n - 1)


def paper_rounding(value, paper):
    """How `value` reaches the paper's 3-decimal `paper`: "exact", "via 4 dp" or None.

    "exact": `value` rounds half up to `paper`, as the paper's numbers should.
    "via 4 dp": it does so only after first being rounded to 4 decimals. One
    published number needs this: Stanford pooled CORTEG's exact cohort mean
    0.553487 rounds to 0.553, and the paper's 0.554 is reached only from an
    intermediate rounded value (0.5535, or the mean of the appendix's 3-dp
    per-subject cells, 4.982 / 9 = 0.55356). A row may pass this way only if it
    says so (`paper_rounding` in its config; report() enforces it), so the
    tolerance of the other rows is not widened.
    """
    want = Decimal(f"{paper:.3f}")
    if round_half_up(value, 3) == want:
        return "exact"
    if round_half_up(round_half_up(value, 4), 3) == want:
        return "via 4 dp"
    return None


def round_half_up(value, places):
    """`value` rounded half up to `places` decimals, as a Decimal (no float error)."""
    return Decimal(value).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)


class Problems(list):
    """Every check that failed, printed as it is found and listed at the end."""

    def add(self, msg):
        print(f"  [FAIL] {msg}")
        self.append(msg)


def pred_dir(root, run, sub):
    return root / run / "predictions" / sub


def load_pred(root, run, sub):
    p = pred_dir(root, run, sub)
    return as_2d(np.load(p / "y_true.npy")), as_2d(np.load(p / "y_pred.npy"))


def results_files(root, run):
    return sorted((root / run).glob("results*.json"))


_RESULTS = {}


def run_results(root, run):
    """The run's own results*.json, or None unless there is exactly one."""
    key = str(root / run)
    if key not in _RESULTS:
        files = results_files(root, run)
        _RESULTS[key] = json.loads(files[0].read_text()) if len(files) == 1 else None
    return _RESULTS[key]


def stored_r(root, run, sub):
    """Per-channel r for `sub` as written by the run itself, or None."""
    res = run_results(root, run)
    if res is None or sub not in res.get("per_subject", {}):
        return None
    ps = res["per_subject"][sub]
    per_ch = ps.get("corr", ps.get("fingers"))
    if per_ch is None:                      # single-output runs store only "r"
        per_ch = [ps["r"]] if "r" in ps else None
    return None if per_ch is None else [float(v) for v in np.atleast_1d(per_ch)]


def missing_inputs(m, root, data_root, subjects):
    """Subjects for which row `m` has nothing to load."""
    if "ridge" in m:
        return [s for s in subjects if not (Path(data_root) / f"{s}_features.pkl").exists()]
    return [s for s in subjects
            if not all((pred_dir(root, m["run"], s) / f).exists()
                       for f in ("y_true.npy", "y_pred.npy"))]


def stanford_ridge(data_root, sub):
    """Ridge_HGA and Ridge_LFS predictions for one subject, as in the paper.

    Per-subject RidgeCV (alphas 0.1-1000) on the per-channel mean and std of
    each z-scored 1 s window, trained on the first 90% of the training windows
    (TailSplit 0.1, the split every Stanford run uses) and scored on the test
    windows. Features and targets are z-scored with training-portion
    statistics. Returns {mode: (y_true, y_pred)}.
    """
    from sklearn.linear_model import RidgeCV
    from data.io import load_subject
    from data.scalers import (apply_zscore_2d, apply_zscore_3d_per_channel,
                              fit_zscore_2d, fit_zscore_3d_per_channel)
    from data.splits import TailSplit

    sd = load_subject(data_root, sub, require_xyz=False)
    idx_tr, _ = TailSplit(0.1).split(sd.y_tr.shape[0])
    y_stats = fit_zscore_2d(sd.y_tr[idx_tr])
    y_tr = apply_zscore_2d(sd.y_tr[idx_tr], y_stats)
    y_te = apply_zscore_2d(sd.y_te, y_stats)
    out = {}
    for mode, stream in (("hi_gamma", 0), ("lo_freq", 1)):   # X_feat is [high, low]
        st = fit_zscore_3d_per_channel(sd.X_feat_tr[idx_tr][..., stream])
        tr = apply_zscore_3d_per_channel(sd.X_feat_tr[idx_tr][..., stream], st)
        te = apply_zscore_3d_per_channel(sd.X_feat_te[..., stream], st)
        f_tr = np.concatenate([tr.mean(axis=-1), tr.std(axis=-1)], axis=-1)
        f_te = np.concatenate([te.mean(axis=-1), te.std(axis=-1)], axis=-1)
        ridge = RidgeCV(alphas=(0.1, 1, 10, 100, 1000)).fit(f_tr, y_tr)
        out[mode] = (y_te, ridge.predict(f_te))
    return out


def usable_rows(name, cfg, root, data_root, allow_missing, problems):
    """The configured rows whose inputs are all present, after reporting the rest."""
    subjects = cfg["subjects"]
    rows = []
    for m in cfg["models"]:
        absent = missing_inputs(m, root, data_root, subjects)
        where = (f"Stanford features under {data_root}" if "ridge" in m
                 else f"predictions under {root / m['run']}")
        if absent and len(absent) == len(subjects) and allow_missing:
            print(f"  [DROPPED] {m['label']}: no {where} (--allow_missing)")
            continue
        if absent:
            shown = ", ".join(absent[:4]) + (", ..." if len(absent) > 4 else "")
            problems.add(f"{name}/{m['label']}: no {where} for "
                         f"{len(absent)}/{len(subjects)} subjects ({shown})")
            continue
        if "run" in m:
            n_res = len(results_files(root, m["run"]))
            if n_res > 1:
                problems.add(f"{name}/{m['label']}: {n_res} results*.json under "
                             f"{m['run']}, expected one")
                continue
            if n_res == 0 and m["paper"] is not None:
                problems.add(f"{name}/{m['label']}: no results*.json under {m['run']}, "
                             "so the r the paper printed cannot be read")
                continue
            if n_res == 0:
                print(f"  [NOTE] {m['label']}: no results file; r is recomputed "
                      "from the saved predictions")
        seed_runs = m.get("seed_mean_of", []) + m.get("seeds_not_in_paper", [])
        bad_seeds = [r for r in seed_runs if len(results_files(root, r)) != 1]
        if bad_seeds:
            problems.add(f"{name}/{m['label']}: the seed runs it reports need one "
                         f"results*.json in each of {', '.join(bad_seeds)}")
            continue
        rows.append(m)
    if not rows:
        problems.add(f"{name}: no row left to show")
    return rows


def load_row(name, m, sub, root, ridge, n_ch, problems, gaps):
    """(y_true, y_pred, per-channel r) of row `m` for `sub`, or None on a failed check."""
    tag = f"{name}/{m['label']}/{sub}"
    if "ridge" in m:
        yt, yp = (as_2d(a) for a in ridge[m["ridge"]])
    else:
        yt, yp = load_pred(root, m["run"], sub)
    if yt.shape != yp.shape or yt.shape[1] != n_ch:
        problems.add(f"{tag}: y_true {yt.shape} and y_pred {yp.shape}, expected "
                     f"{n_ch} channel(s) each")
        return None
    recomputed = pearson_per_channel(yt, yp)
    per_ch = recomputed if "ridge" in m else stored_r(root, m["run"], sub)
    if per_ch is None:
        if m["paper"] is not None:
            problems.add(f"{tag}: no stored r in {m['run']}'s results file")
            return None
        # Only non-paper rows may fall back; their r was never printed.
        if results_files(root, m["run"]):
            print(f"  [NOTE] {tag}: not in the results file; r is recomputed")
        per_ch = recomputed
    if len(per_ch) != n_ch:
        problems.add(f"{tag}: results file has {len(per_ch)} r value(s), expected {n_ch}")
        return None
    if not np.all(np.isfinite(per_ch)) or not np.all(np.isfinite(recomputed)):
        problems.add(f"{tag}: r is not finite (a constant or NaN prediction)")
        return None
    gap_mean = abs(float(np.mean(per_ch)) - float(np.mean(recomputed)))
    gap_ch = float(np.max(np.abs(np.subtract(per_ch, recomputed))))
    gaps[0], gaps[1] = max(gaps[0], gap_mean), max(gaps[1], gap_ch)
    if gap_mean > STORED_VS_RECOMPUTED_TOL or gap_ch > STORED_VS_RECOMPUTED_TOL_CHANNEL:
        problems.add(f"{tag}: stored r {np.round(per_ch, 4).tolist()} vs r from the "
                     f"saved predictions {np.round(recomputed, 4).tolist()}")
        return None
    return yt, yp, per_ch


def build(name, cfg, rows, root, data_root, problems):
    """One dataset's `rows`, entirely in memory: {file name: JSON text}.

    Nothing here writes; failed checks are added to `problems`.
    """
    print(f"\n=== {name} ===")
    subjects, fingers = cfg["subjects"], cfg["fingers"]
    multi = fingers is not None
    n_ch = len(fingers) if multi else 1
    step = {m["label"]: m.get("step_s", cfg["step_s"]) for m in rows}
    per_model = {m["label"]: [] for m in rows}
    gaps = {m["label"]: [0.0, 0.0] for m in rows}
    files, skipped = {}, 0
    for sub in subjects:
        ridge = stanford_ridge(data_root, sub) if any("ridge" in m for m in rows) else {}
        loaded = [(m, load_row(name, m, sub, root, ridge, n_ch, problems, gaps[m["label"]]))
                  for m in rows]
        if any(v is None for _, v in loaded):
            skipped += 1        # already reported; not held against the other rows
            continue
        ref_true = loaded[0][1][0]
        n_ref, step_ref = ref_true.shape[0], step[rows[0]["label"]]
        idx_ref = display_index(n_ref, step_ref, n_ref, step_ref)
        sub_data = {"subject": sub, "fingers": fingers} if multi else {"subject": sub}
        sub_data["models"] = {}
        for m, (yt, yp, per_ch) in loaded:
            tag = f"{name}/{m['label']}/{sub}"
            n, st = yt.shape[0], step[m["label"]]
            if st == step_ref and n != n_ref:
                problems.add(f"{tag}: {n} test windows on the same grid as the first "
                             f"row's {n_ref}")
                continue
            if abs((n - 1) * st - (n_ref - 1) * step_ref) > st:
                problems.add(f"{tag}: its windows span {(n - 1) * st:.2f} s, the first "
                             f"row's {(n_ref - 1) * step_ref:.2f} s")
                continue
            idx = display_index(n_ref, step_ref, n, st)
            align = min(np.corrcoef(ref_true[idx_ref, i], yt[idx, i])[0, 1]
                        for i in range(n_ch))
            min_align = (ALIGN_MIN_CORR_SAME_GRID if st == step_ref
                         else ALIGN_MIN_CORR_OTHER_GRID)
            if not align >= min_align:
                problems.add(f"{tag}: target does not line up with the displayed one "
                             f"(r={align:.6f}, needs {min_align})")
                continue
            entry = {"y_pred": as_list(yp[idx] if multi else yp[idx, 0])}
            if multi:
                entry["corr_per_finger"] = [round(c, R_DECIMALS) for c in per_ch]
                entry["corr_mean"] = round(float(np.mean(per_ch)), R_DECIMALS)
            else:
                entry["corr"] = round(per_ch[0], R_DECIMALS)
            sub_data["models"][m["label"]] = entry
            per_model[m["label"]].append(float(np.mean(per_ch)))
        sub_data["y_true"] = as_list(ref_true[idx_ref] if multi else ref_true[idx_ref, 0])
        # index.html draws point i at i / fs_hz seconds, so this is the display
        # rate after downsampling, not the recording rate.
        sub_data["fs_hz"] = round((len(idx_ref) - 1) / ((n_ref - 1) * step_ref), 4)
        sub_data["duration_s"] = round(n_ref * step_ref, 1)
        try:
            files[f"{sub}.json"] = json.dumps(sub_data, separators=(",", ":"), allow_nan=False)
        except ValueError:
            problems.add(f"{name}/{sub}: a prediction or target is NaN or infinite")
        print(f"  {sub}: {len(sub_data['models'])} models, "
              f"{len(files.get(f'{sub}.json', '')) / 1024:.0f} KB")

    report(name, rows, per_model, gaps, len(subjects) - skipped, root, problems)
    labels = [m["label"] for m in rows]
    if all(len(per_model[l]) == len(subjects) for l in labels):   # else nothing is written
        manifest = {"subjects": subjects}
        if multi:
            manifest["fingers"] = fingers
        manifest["models"] = labels
        manifest["notes"] = {m["label"]: m["note"] for m in rows}
        manifest["paper_r"] = {m["label"]: m["paper"] for m in rows}
        manifest["cohort_mean_r"] = {l: round(float(np.mean(per_model[l])), R_DECIMALS)
                                     for l in labels}
        files["manifest.json"] = json.dumps(manifest, indent=2)
    return files


def report(name, rows, per_model, gaps, n_subjects, root, problems):
    """Print each row's cohort mean next to the paper's value; fail any mismatch."""
    print(f"\n{name}: cohort mean r (mean over subjects of the per-subject r)")
    for m in rows:
        label, rs = m["label"], per_model[m["label"]]
        if not rs:                          # every subject failed; reasons listed above
            continue
        if len(rs) < n_subjects:
            problems.add(f"{name}/{label}: r for only {len(rs)} of the {n_subjects} "
                         "subjects that loaded")
            continue
        mean = float(np.mean(rs))
        extra = []
        if m["paper"] is None:
            status = "not a paper row"
        else:
            value, what = mean, ""
            if m.get("seed_mean_of"):
                runs = m["seed_mean_of"] + m.get("seeds_not_in_paper", [])
                scores = [(run_results(root, r) or {}).get("score") for r in runs]
                if None in scores:
                    bad = [r for r, s in zip(runs, scores) if s is None]
                    problems.add(f"{name}/{label}: no single results file with a score "
                                 f"for seed run(s) {', '.join(bad)}")
                    continue
                seeds = scores[:len(m["seed_mean_of"])]
                value = float(np.mean(seeds))
                what = (f"the mean of {len(seeds)} seeds "
                        f"({', '.join(f'{s:.4f}' for s in seeds)} -> {value:.4f}) ")
                if len(scores) > len(seeds):
                    left_out = scores[len(seeds):]
                    extra.append(f"not in the paper's mean: {len(left_out)} seed run(s) "
                                 f"({', '.join(f'{s:.4f}' for s in left_out)}); mean of "
                                 f"all {len(scores)}: {np.mean(scores):.4f}")
            how = paper_rounding(value, m["paper"])
            allowed = m.get("paper_rounding", "exact")
            if how is None:
                status = f"{what}DIFFERS from paper {m['paper']:.3f}"
                problems.add(f"{name}/{label}: {what}{value:.6f} does not round to "
                             f"the paper's {m['paper']:.3f}")
            elif how != allowed:
                status = f"{what}matches paper {m['paper']:.3f} {how}, row allows {allowed}"
                problems.add(f"{name}/{label}: {what}{value:.6f} reaches the paper's "
                             f"{m['paper']:.3f} {how}, but the row's paper_rounding is "
                             f"{allowed!r}")
            elif how == "exact":
                status = f"{what}matches paper {m['paper']:.3f}"
            else:
                status = (f"{what}matches paper {m['paper']:.3f} only via an intermediate "
                          f"rounding ({value:.6f} -> {value:.4f} -> {m['paper']:.3f}; "
                          f"the exact value rounds to {round_half_up(value, 3)})")
        print(f"  {label:40s} {mean:.4f}  {status}")
        if "run" in m:
            extra.append(f"stored vs recomputed r: mean {gaps[label][0]:.1e}, "
                         f"per channel {gaps[label][1]:.1e}")
        for line in extra:
            print(f"  {'':40s}         {line}")


def staging_dirs(target):
    """The hidden siblings a swap of `target` goes through: (new, old)."""
    return (target.with_name(f".{target.name}.new"),
            target.with_name(f".{target.name}.old"))


def not_ours(path, need_manifest):
    """Why this builder must not delete `path`, or None when it may.

    It may delete a directory that holds only regular *.json files, with a
    manifest.json among them when `need_manifest` (a finished build; a staging
    directory left by an interrupted run may lack it). --out_dir is
    user-supplied, so anything else under a dataset's name is left alone.
    """
    if not (path.exists() or path.is_symlink()):
        return None
    if path.is_symlink() or not path.is_dir():
        return f"{path} is not a directory"
    entries = list(path.iterdir())
    foreign = sorted(p.name for p in entries
                     if p.is_symlink() or not p.is_file() or p.suffix != ".json")
    if foreign:
        shown = ", ".join(foreign[:3]) + (", ..." if len(foreign) > 3 else "")
        return f"{path} holds {shown}, which this builder did not write"
    if need_manifest and entries and not (path / "manifest.json").is_file():
        return f"{path} has no manifest.json, so this builder did not write it"
    return None


def check_out_root(out_root, names, problems):
    """Refuse, before the slow build, output directories this builder did not write."""
    for name in names:
        target = out_root / name
        new, old = staging_dirs(target)
        for path, need_manifest in ((target, True), (old, True), (new, False)):
            why = not_ours(path, need_manifest)
            if why:
                problems.add(f"{why}; refusing to replace it")


def stage_dir(target, files):
    """Write `files` into the hidden `.new` sibling of `target`.

    A run stopped mid-swap is repaired first: its `.old` is put back if the
    target is missing, and dropped otherwise.
    """
    new, old = staging_dirs(target)
    if old.exists():
        if target.exists():
            shutil.rmtree(old)
        else:
            old.rename(target)
    if new.exists():
        shutil.rmtree(new)
    new.mkdir(parents=True)
    for fname, text in files.items():
        (new / fname).write_text(text, encoding="utf-8")


def swap_in(target):
    """Replace `target` by its staged `.new` sibling; one of the two is always in place."""
    new, old = staging_dirs(target)
    if target.exists():
        target.rename(old)
    new.rename(target)
    if old.exists():
        shutil.rmtree(old)


def build_all(datasets, root, data_root, out_root, allow_missing=False):
    """Build and check every dataset, then write them all; SystemExit on any failure."""
    _RESULTS.clear()
    runs = [m["run"] for cfg in datasets.values() for m in cfg["models"] if "run" in m]
    if not any((root / r).is_dir() for r in runs):
        raise SystemExit(f"none of the demo's {len(runs)} runs is under {root}; point "
                         "CORTEG_OUTPUT_ROOT or --output_root at the run outputs")
    problems = Problems()

    def stop_if_failed():
        if problems:
            raise SystemExit(f"\n{len(problems)} failed check(s); nothing was written "
                             f"to {out_root}:\n  " + "\n  ".join(problems))

    # Inputs and the output directory first, so that a missing run or an
    # unwritable destination fails before anything slow is loaded.
    print("\nChecking inputs")
    rows = {name: usable_rows(name, cfg, root, data_root, allow_missing, problems)
            for name, cfg in datasets.items()}
    check_out_root(out_root, datasets, problems)
    stop_if_failed()
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=out_root):
            pass
    except OSError as e:
        raise SystemExit(f"cannot write to {out_root}: {e}")
    built = {name: build(name, cfg, rows[name], root, data_root, problems)
             for name, cfg in datasets.items()}
    stop_if_failed()
    # Stage every dataset before swapping any in, so a failed write leaves all
    # of them as they were.
    targets = [out_root / name for name in built]
    try:
        for target, files in zip(targets, built.values()):
            stage_dir(target, files)
    except OSError as e:
        for target in targets:
            shutil.rmtree(staging_dirs(target)[0], ignore_errors=True)
        raise SystemExit(f"could not write under {out_root} ({e}); the existing "
                         "files are unchanged")
    for target in targets:
        swap_in(target)
    sizes = [f"{name} {sum(len(t) for t in files.values()) / 1024:.0f} KB"
             for name, files in built.items()]
    print(f"\nWrote {out_root}: " + " + ".join(sizes))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output_root", default="",
                    help="Root of the run outputs (default: CORTEG_OUTPUT_ROOT).")
    ap.add_argument("--data_root", default="",
                    help="Stanford *_features.pkl directory, for the Ridge rows "
                         "(default: CORTEG_DATA_ROOT).")
    ap.add_argument("--out_dir", default="",
                    help="Directory to write <dataset>/*.json into (default: docs/data). "
                         "Each <dataset>/ in it is replaced whole; the build refuses "
                         "one that holds anything but an earlier build's *.json files.")
    ap.add_argument("--allow_missing", action="store_true",
                    help="Drop rows whose inputs are absent for every subject instead "
                         "of failing. Needs --out_dir: a partial demo never replaces "
                         "docs/data.")
    args = ap.parse_args()
    root = Path(get_output_root(args.output_root))
    data_root = get_data_root(args.data_root)
    out_root = Path(args.out_dir).resolve() if args.out_dir else DOCS_DATA
    print(f"run outputs: {root}\nStanford features: {data_root}\nwriting to: {out_root}")
    if args.allow_missing and out_root == DOCS_DATA:
        raise SystemExit("--allow_missing builds a partial demo; write it to --out_dir, "
                         "not over docs/data")
    build_all(DATASETS, root, data_root, out_root, args.allow_missing)


if __name__ == "__main__":
    main()
