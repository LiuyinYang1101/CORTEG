"""How much of BrainTreebank Task A does the transcript alone give away?

The paper removes the pre-onset part of the Task A window because the pause before
a sentence-initial word already predicts the label. This script measures that claim
without neural data and without fitting anything. It is transcript arithmetic and runs
on CPU in well under a minute.

For each subject it rebuilds the exact Task A events that
``experiments/run_btb_classification.py`` scores: the same trial and film,
``load_events(max_per_class=900, seed=42)``, and the same trigger-range and
window-bounds mask as ``extract_windows`` for the [t, t+1.5] window. Each event is
scored by the silence before its word, and AUROC is reported with sentence-initial
as the positive class:

* over all scored events; and
* over the test blocks of the release's split (``forward_chaining_split``, 4 folds,
  7 s embargo, 1.5 s footprint, ``val_frac`` 0.15). ``test_fold_mean`` is the mean
  of the per-fold AUROCs. That is how the neural AUROCs in the paper are aggregated,
  so it is the like-for-like reference.

Pause definition (``gap``, the headline score). The gap is the word's start time
minus the end time of the previous timed word in transcript order. Untimed tokens
(clitics such as 'd and 's have no timestamps) are skipped. The gap is negative where
two speakers overlap. The first word of a film is measured from t = 0. Two variants
are also reported, so the result does not depend on one convention:

* ``word_diff``: the transcript's own column, i.e. start minus the end of the
  immediately previous row. It is NaN when that row is untimed, and those events
  are dropped for this variant only.
* ``onset_diff``: the transcript's onset-to-onset interval, which is the pause plus
  the previous word's duration. NaNs are handled the same way.

``all_words`` is the ``gap`` AUROC over every timed word in the film, not just the
balanced 900 + 900 draw. It is a property of the film, so the three subjects who
watched cars-2 (sub_3, sub_7, sub_10) share one value.

The events are checked against the release's own feature cache when that cache
exists, so the scored events cannot drift from the neural runs unnoticed.

Result at the paper's settings (max_per_class 900, seed 42, all 10 subjects; the
events match the release caches exactly). ``gap`` over the scored events ranges
0.852-0.934 (mean 0.898). The per-fold test mean ranges 0.854-0.941 (mean 0.902).
The submitted text's "0.87-0.93" is the ``all_words`` range over the 8 distinct films
(0.866-0.931). It was therefore measured on whole transcripts, not on the Task A
events. The difference comes from cars-2 (sub_3, sub_7 and sub_10), whose Task A draw
scores 0.852-0.859.

Example:

    python scripts/btb_pause_only_auroc.py
    python scripts/btb_pause_only_auroc.py --subjects sub_3 --out /path/to/pause.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from data.braintreebank import (  # noqa: E402
    EMBARGO_SEC,
    btb_output_root,
    btb_root,
    build_time_to_sample,
    estimate_fs,
    forward_chaining_split,
    load_events,
)
from experiments.run_btb_classification import movie_of, trial_of  # noqa: E402

# The Task A settings of run_btb_classification.py. Its cache name also uses them,
# which is how the events are cross-checked below.
WIN_SEC = 1.5
PRE_SEC = 0.0
HGA_LOW, HGA_HIGH = 70.0, 200.0
N_FOLDS = 4
VAL_FRAC = 0.15

SCORES = ("gap", "word_diff", "onset_diff")


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def transcript_rows(root: str, movie: str):
    """Per-row arrays for the rows load_events keeps, in the same order.

    load_events keeps a row iff ``start`` and ``is_onset`` parse and ``start`` is
    finite. The pause scores, however, are computed over *every* row, so an untimed
    or unparsable row between two kept words is skipped rather than treated as a
    word boundary.
    """
    starts, labels, gap, word_diff, onset_diff = [], [], [], [], []
    prev_end = 0.0                       # the first word of a film is timed from t = 0
    with open(f"{root}/transcripts/{movie}/features.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            s, e = _f(row.get("start")), _f(row.get("end"))
            try:                         # the same keep rule as load_events
                s_keep = float(row["start"])
                o = float(row["is_onset"])
                kept = bool(np.isfinite(s_keep))
            except (ValueError, KeyError):
                kept = False
            if kept:
                starts.append(s_keep)
                labels.append(1 if o >= 0.5 else 0)
                gap.append(s_keep - prev_end)
                word_diff.append(_f(row.get("word_diff")))
                onset_diff.append(_f(row.get("onset_diff")))
            if np.isfinite(s) and np.isfinite(e):
                prev_end = e
    return (np.asarray(starts), np.asarray(labels, dtype=int),
            {"gap": np.asarray(gap), "word_diff": np.asarray(word_diff),
             "onset_diff": np.asarray(onset_diff)})


def selected_rows(labels: np.ndarray, max_per_class: int, seed: int) -> np.ndarray:
    """Row indices load_events draws. It returns only (starts, labels), and the
    pause needs the row. The caller asserts that the two agree exactly."""
    rng = np.random.RandomState(seed)
    pos = np.where(labels == 1)[0]
    neg = np.where(labels == 0)[0]
    k = min(len(pos), len(neg), max_per_class)
    pos = rng.permutation(pos)[:k]
    neg = rng.permutation(neg)[:k]
    return np.sort(np.concatenate([pos, neg]))


def scored_mask(root: str, subj: str, trial: str, starts: np.ndarray) -> np.ndarray:
    """The ``valid`` mask of data.braintreebank.extract_windows at [t, t+1.5],
    computed without reading any signal: only the HDF5 length is opened."""
    import h5py

    fs = estimate_fs(root, subj, trial)
    t2s = build_time_to_sample(root, subj, trial)
    win = int(round(WIN_SEC * fs))
    pre = int(round(PRE_SEC * fs))
    centers_f = np.asarray(t2s(starts), dtype=np.float64)
    in_range = np.isfinite(centers_f)
    s0 = np.where(in_range, centers_f, 0).astype(np.int64) + pre
    with h5py.File(f"{root}/all_subject_data/{subj}_{trial}.h5", "r") as h5:
        t_total = h5["data/electrode_0"].shape[0]
    return in_range & (s0 >= 0) & (s0 + win <= t_total)


def release_cache(subj: str, trial: str, max_per_class: int, seed: int) -> str:
    """Path of run_btb_classification.py's Task A cache for this subject."""
    tag = (f"btb_{subj}_{trial}_win{WIN_SEC}_pre{PRE_SEC}"
           f"_hfa{int(HGA_LOW)}-{int(HGA_HIGH)}_n{max_per_class}_s{seed}.npz")
    return os.path.join(btb_output_root(), "cache", tag)


def auroc(y: np.ndarray, s: np.ndarray):
    """AUROC with NaN scores dropped; None if one class is missing."""
    from sklearn.metrics import roc_auc_score

    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    if y.size == 0 or np.unique(y).size < 2:
        return None, int(y.size)
    return float(roc_auc_score(y, s)), int(y.size)


def score_subject(root: str, subj: str, args) -> dict:
    trial = trial_of(root, subj)
    movie = movie_of(root, subj, trial)

    starts, labels, pause = transcript_rows(root, movie)
    keep = selected_rows(labels, args.max_per_class, args.seed)
    ev_starts, ev_y = load_events(root, movie, args.max_per_class, args.seed)
    if not (np.array_equal(starts[keep], ev_starts) and np.array_equal(labels[keep], ev_y)):
        raise AssertionError(f"{subj}: row selection disagrees with load_events")

    valid = scored_mask(root, subj, trial, ev_starts)
    t, y = ev_starts[valid], ev_y[valid]
    sc = {k: v[keep][valid] for k, v in pause.items()}

    cache = release_cache(subj, trial, args.max_per_class, args.seed)
    cache_match = None
    if os.path.exists(cache) and not args.skip_cache_check:
        with np.load(cache, allow_pickle=False) as d:
            cache_match = bool(np.array_equal(d["event_times"], t)
                               and np.array_equal(np.asarray(d["y"]).astype(int), y))
        if not cache_match:
            raise AssertionError(
                f"{subj}: rebuilt events differ from the release cache {os.path.basename(cache)}")

    folds = forward_chaining_split(t, win_sec=WIN_SEC, n_folds=N_FOLDS, val_frac=VAL_FRAC)
    if len(folds) != N_FOLDS:
        raise AssertionError(f"{subj}: {len(folds)} causal folds, the release uses {N_FOLDS}")
    test_idx = [np.asarray(f[-1]) for f in folds]
    test_all = np.concatenate(test_idx)

    out = {"trial": trial, "movie": movie, "n_events": int(t.size),
           "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
           "n_dropped_by_mask": int((~valid).sum()),
           "n_test_events": int(test_all.size),
           "events_match_release_cache": cache_match,
           "median_gap_sec": {"sentence_initial": float(np.median(sc["gap"][y == 1])),
                              "mid_sentence": float(np.median(sc["gap"][y == 0]))}}
    for name in SCORES:
        s = sc[name]
        a_all, n_all = auroc(y, s)
        per_fold = [auroc(y[i], s[i])[0] for i in test_idx]
        a_pool, n_pool = auroc(y[test_all], s[test_all])
        out[name] = {"all_events": a_all, "n_all_events": n_all,
                     "test_folds": per_fold,
                     "test_fold_mean": (float(np.mean(per_fold))
                                        if all(v is not None for v in per_fold) else None),
                     "test_pooled": a_pool, "n_test_pooled": n_pool}

    # Every timed word of the film, not the balanced draw.
    a_words, n_words = auroc(labels, pause["gap"])
    out["all_words"] = {"gap": a_words, "n_words": n_words}
    return out


def summarise(per_subject: dict) -> dict:
    def stats(vals):
        v = np.asarray([x for x in vals if x is not None], dtype=float)
        return {"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                "sd": float(v.std(ddof=1)) if v.size > 1 else 0.0, "n": int(v.size)}

    subs = list(per_subject.values())
    out = {}
    for name in SCORES:
        out[name] = {k: stats([r[name][k] for r in subs])
                     for k in ("all_events", "test_fold_mean", "test_pooled")}
    films = {}
    for r in subs:
        films.setdefault(r["movie"], r["all_words"]["gap"])
    out["all_words_gap_per_film"] = films
    out["all_words_gap"] = stats(list(films.values()))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+", default=[f"sub_{i}" for i in range(1, 11)])
    p.add_argument("--max_per_class", type=int, default=900)
    p.add_argument("--seed", type=int, default=42, help="event-draw seed (the paper's is 42)")
    p.add_argument("--skip_cache_check", action="store_true",
                   help="do not compare against the release's Task A feature cache")
    p.add_argument("--out", default="",
                   help="results JSON (default: <output root>/braintreebank/pause_only_auroc.json)")
    args = p.parse_args()

    root = btb_root()
    t0 = time.time()
    per_subject = {}
    for subj in args.subjects:
        r = score_subject(root, subj, args)
        per_subject[subj] = r
        g = r["gap"]
        print(f"[{subj}] {r['movie']:<28} n={r['n_events']:4d}  gap AUROC: "
              f"all={g['all_events']:.4f}  test-fold mean={g['test_fold_mean']:.4f}  "
              f"test pooled={g['test_pooled']:.4f}  (cache match: {r['events_match_release_cache']})",
              flush=True)

    summary = summarise(per_subject)
    out = {
        "what": "Task A AUROC of the pre-word pause alone (transcript only, no neural data, "
                "no fitting); positive class = sentence-initial word",
        "events": {"loader": "data.braintreebank.load_events", "max_per_class": args.max_per_class,
                   "seed": args.seed, "window": [PRE_SEC, PRE_SEC + WIN_SEC]},
        "split": {"scheme": "forward_chaining", "n_folds": N_FOLDS, "val_frac": VAL_FRAC,
                  "footprint_sec": WIN_SEC, "embargo_sec": EMBARGO_SEC},
        "scores": {
            "gap": "start - end of the previous timed word (transcript order; untimed tokens "
                   "skipped; first word timed from t=0)",
            "word_diff": "transcript column: start - end of the immediately previous row "
                         "(NaN, dropped, when that row is untimed)",
            "onset_diff": "transcript column: onset-to-onset interval (pause + previous word "
                          "duration; NaN dropped)",
        },
        "per_subject": per_subject,
        "summary": summary,
        "elapsed_s": time.time() - t0,
    }
    dest = args.out or os.path.join(btb_output_root(), "pause_only_auroc.json")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    for name in SCORES:
        s = summary[name]
        print(f"{name:<10} all events {s['all_events']['min']:.3f}-{s['all_events']['max']:.3f}"
              f"  |  test-fold mean {s['test_fold_mean']['min']:.3f}-{s['test_fold_mean']['max']:.3f}"
              f" (cohort {s['test_fold_mean']['mean']:.3f} +/- {s['test_fold_mean']['sd']:.3f})")
    s = summary["all_words_gap"]
    print(f"all words of each film (gap): {s['min']:.3f}-{s['max']:.3f} over {s['n']} films")
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
