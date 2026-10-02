#!/usr/bin/env python
"""Reprint the iEEG-FM regression tables (paper Table 1 FM rows, Tables 18 and 19).

    python scripts/aggregate_fm.py                     # the published cells, checked
    python scripts/aggregate_fm.py --cells DIR [DIR ...]    # fresh runs vs the paper

With the release scripts' default output folder:

    python scripts/aggregate_fm.py \\
        --cells "${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/ieeg_fm_regression"

With no arguments this reads ``paper_cells/fm_regression``, the result files
behind the published numbers (Stanford finger AND Ghent audio: the per-cell
metrics are published even though the Ghent data are not), rebuilds every
cell and checks it against the paper. It exits non-zero if any cell differs.

Aggregation rule, the one the paper uses:

* per-subject score = Pearson r on the test split, averaged over the output
  dimensions (``corr_mean`` in each file; 5 fingers, 1 audio envelope);
* a cell (adaptation x dataset x model x regime) averages its seeds WITHIN each
  subject first, then reports the mean over subjects +- sample SD (ddof=1)
  across subjects (n = 9 finger, 16 audio);
* Table 19 = those means, per regime (pooled and per-subject use seeds 42, 0, 1;
  LOO uses seed 42 only);
* Table 18 = for each model x adaptation x task, the regime with the highest
  mean, carrying that regime's SD; the paragraph mark (here "(3 seeds)") flags
  a winning regime that averaged three seeds. Only cells with the whole cohort
  compete. Brant's cell is its per-subject run, whose r App. A.11 quotes: a
  pooled head ran beside it (finger r = 0.001) and is not reported; it is
  printed under Table 18 but is not a candidate (T18_REGIMES);
* Table 1 FM rows = for each model and task, the Table 18 cell with the
  highest mean;
* rounding = once, from full precision, to 3 dp.

The temporal (BiLSTM) head and Brant ran pooled and per-subject only, never
LOO; a LOO run of either is listed under "Other cells". A LOO cell counts only
when its ``_loo_done.json`` exists; the per-fold ``ft_<subject>/results_persub.json``
files are never counted on their own, and a LOO directory with folds but no
``_loo_done.json`` is reported as incomplete.

Two Table 19 cells (Stanford pooled BrainBERT ft, Ghent LOO BrainBERT ft; see
NOT_IN_CELLS) come from runs that are incomplete in the archive, so paper_cells
cannot rebuild them: the value the paper prints is shown with "~" and not
checked, and a fresh run of either is shown with "*" but does not enter Table 18.
One cell, Ghent LOO PopT ft, is checked at its exact rounding (0.012) and the
paper's printed value (0.013, a second rounding of 0.0125) is noted beside it.

Fresh runs (``--cells DIR [DIR ...]``)
--------------------------------------
Every ``results_*.json`` and ``_loo_done.json`` under each DIR is classified by its
contents (``fm``, ``dataset``, ``train_mode``, ``mode``, ``head``, ``args``),
not its path, so any layout works; a DIR that is not a folder is an error.
``--tol`` applies to fresh runs only. The runners write, by default under
``$CORTEG_OUTPUT_ROOT/ieeg_fm_regression/<adaptation>/Stanford/<fm>/<regime>/seed<S>/``:

  experiments/run_ieeg_fm_regression.py   results_pooled.json, results_persub.json;
                                          LOO: ft_<subject>/results_persub.json per
                                          held-out subject + _loo_done.json
  experiments/run_brant_regression.py     results_pooled.json, results_persub.json

and paper_cells/fm_regression uses the same layout (see paper_cells/MANIFEST.md).
A setting outside the paper becomes its own adaptation and is listed under
"Other cells", never mixed into a paper cell. The runners' own folder tags
name the head and pooling (``--head mlp``, ``--bb_pool last10``, a causal
temporal head, another Brant stride, context length or patch pooling, e.g.
``probe_last10``); every other recipe value in RECIPE_FM (epochs, patience,
warm-up, learning rates, batch size, re-referencing, blocks unfrozen, temporal
sequence length and width) that differs from the paper runs' recorded value is
appended in brackets, and a ``--max_windows`` / ``--max_anchors`` cap adds
``smoke``: e.g. ``probe[epochs=1,smoke]``. A setting a file does not record is
taken to be the paper's. The same (cell, seed, subject) in two files is an
error. Ghent cells cannot be re-run from the public release and are reported
as not released.

What a fresh run is checked against:

1. A Table 19 cell is scored only when the fresh cell has the paper's seeds
   (42, 0, 1 for pooled and per-subject probe / fine-tune; 42 otherwise) and
   all 9 subjects. A Table 18 cell is scored only when every regime behind the
   paper's cell is there and scorable, a Table 1 cell only when every
   adaptation behind it is. Anything else is printed, marked "not scored", and
   left out of the exit status. When the best regime differs from the paper's
   a NOTE says so.
2. Every fresh seed that also exists in ``--reference`` (default
   paper_cells/fm_regression) is compared with that same seed, subject by
   subject: the cohort-mean difference (``--tol``, 0.005), the mean absolute
   per-subject difference (``--tol_subject``, 0.01) and the largest one
   (``--tol_subject_max``, 0.03). No run-to-run floor was measured for these
   cells; the defaults are loose heuristics, not a property of the paper runs.
   A run on another device class or precision than the paper's (a CPU run is
   fp32; the paper ran on GPUs with AMP) is expected to exceed them: a CPU run
   of the released code on the Stanford cohort was outside them on 4 of 7
   seeds (max |d| 0.038). The runners now record such a run as
   ``use_amp=False`` (AMP runs on CUDA only), so it is tagged
   ``[use_amp=False]`` and listed under "Other cells" instead of being
   compared as a paper cell. The check also presumes that the fresh run draws its
   random numbers as the paper run did; for the three-seed cells the reference's
   own seed-to-seed spread is printed beside it, and a failure within that
   spread is marked, since it is what another random stream would give.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from itertools import combinations

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CELLS = os.path.join(REPO, "paper_cells", "fm_regression")
DATASETS = ("Stanford", "Ghent")
RELEASED_DATASETS = ("Stanford",)
TASK = {"Stanford": "finger", "Ghent": "audio"}
N_SUBJECTS = {"Stanford": 9, "Ghent": 16}
REGIMES = ("pooled", "per_subject", "loo")
REGIME_LABEL = {"pooled": "pooled", "per_subject": "per-subject", "loo": "LOO"}
# The regimes each Table 18 adaptation ran in the paper (a run in another regime
# is an "other" cell).
ALLOWED_REGIMES = {"probe": REGIMES, "ft": REGIMES,
                   "temporal": ("pooled", "per_subject"), "brant": ("pooled", "per_subject")}
# The regimes a Table 18 cell is chosen from. Brant's cell is its per-subject run,
# whose r App. A.11 quotes; a pooled head ran beside it (r = 0.001 finger, far
# below) and is not reported. It is printed under Table 18 but not
# required, so the Stanford grid of scripts/table1_ieeg_fm_stanford.sh, which does
# not run it, scores every finger cell.
T18_REGIMES = {"probe": REGIMES, "ft": REGIMES,
               "temporal": ("pooled", "per_subject"), "brant": ("per_subject",)}
FM_LABEL = {"brainbert": "BrainBERT", "popt": "PopT", "brant": "Brant"}
ADAPT_LABEL = {"probe": "frozen linear probe", "ft": "last-N fine-tune",
               "temporal": "temporal head (BiLSTM)", "brant": "frozen probe (6 s context)"}
ADAPT_SHORT = {"probe": "probe", "ft": "last-N FT", "temporal": "temporal head",
               "brant": "6 s probe"}


def paper_seeds(key):
    """Seeds behind a paper cell (adaptation, dataset, fm, regime)."""
    ad, _, _, rg = key
    return (42, 0, 1) if ad in ("probe", "ft") and rg != "loo" else (42,)


# ─────────────────────────── the paper's values ─────────────────────────────
# Exact rounding of the artifacts in paper_cells/fm_regression, as the paper
# prints them. Table 19 columns: BrainBERT probe, PopT probe, BrainBERT ft, PopT ft.
T19_COLS = (("brainbert", "probe"), ("popt", "probe"), ("brainbert", "ft"), ("popt", "ft"))
EXPECTED_T19 = {
    ("Stanford", "pooled"):      ("0.030", "0.040", None, "0.041"),
    ("Stanford", "per_subject"): ("0.039", "0.049", "0.033", "0.037"),
    ("Stanford", "loo"):         ("0.044", "0.050", "0.047", "0.036"),
    ("Ghent", "pooled"):         ("0.031", "0.031", "0.028", "0.015"),
    ("Ghent", "per_subject"):    ("0.052", "0.038", "0.050", "0.021"),
    ("Ghent", "loo"):            ("0.050", "0.039", None, "0.012"),
}
# Cells the paper prints that paper_cells cannot rebuild (None above): the
# archived runs are incomplete. Value printed in the paper, and why.
NOT_IN_CELLS = {
    ("Stanford", "pooled", "brainbert", "ft"): ("0.035",
        "only seed 42 of the three seeds is archived (r = 0.039); "
        "the seed 0 and 1 outputs are not"),
    ("Ghent", "loo", "brainbert", "ft"): ("0.043",
        "only 3 of the 16 LOO folds are archived"),
}
# Cells the paper prints at a different rounding than the exact artifact value.
PRINTED_AS = {
    ("Ghent", "loo", "popt", "ft"): ("0.013", "exact 0.012479; the paper rounds 0.0125 again"),
}
# Table 18: (mean, sd, three-seed mark, winning regime) per dataset.
T18_ROWS = (("brainbert", "probe"), ("brainbert", "ft"), ("brainbert", "temporal"),
            ("popt", "probe"), ("popt", "ft"), ("popt", "temporal"), ("brant", "brant"))
EXPECTED_T18 = {
    ("brainbert", "probe"):    {"Stanford": ("0.044", "0.051", False, "loo"),
                                "Ghent": ("0.052", "0.067", True, "per_subject")},
    ("brainbert", "ft"):       {"Stanford": ("0.047", "0.038", False, "loo"),
                                "Ghent": ("0.050", "0.087", True, "per_subject")},
    ("brainbert", "temporal"): {"Stanford": ("0.053", "0.056", False, "per_subject"),
                                "Ghent": ("0.057", "0.066", False, "per_subject")},
    ("popt", "probe"):         {"Stanford": ("0.050", "0.045", False, "loo"),
                                "Ghent": ("0.039", "0.064", False, "loo")},
    ("popt", "ft"):            {"Stanford": ("0.041", "0.046", True, "pooled"),
                                "Ghent": ("0.021", "0.043", True, "per_subject")},
    ("popt", "temporal"):      {"Stanford": ("0.063", "0.046", False, "per_subject"),
                                "Ghent": ("0.028", "0.051", False, "per_subject")},
    ("brant", "brant"):        {"Stanford": ("0.028", "0.031", False, "per_subject"),
                                "Ghent": ("0.022", "0.035", False, "per_subject")},
}
# Table 18 caption: "In 11/14 cells, the cross-subject SD exceeds the mean, and
# no configuration in this table reaches r = 0.07".
EXPECTED_SD_GT_MEAN, CEILING = 11, 0.07
# Table 1 FM rows: (mean, sd, winning adaptation).
EXPECTED_T1 = {
    "brainbert": {"Stanford": ("0.053", "0.056", "temporal"), "Ghent": ("0.057", "0.066", "temporal")},
    "popt":      {"Stanford": ("0.063", "0.046", "temporal"), "Ghent": ("0.039", "0.064", "probe")},
    "brant":     {"Stanford": ("0.028", "0.031", "brant"), "Ghent": ("0.022", "0.035", "brant")},
}

# ─────────────────────────── the paper recipes ──────────────────────────────
# Recorded in every paper-run file (and the released runners' defaults). A
# result file that records a different value gets "<name>=<value>" as a tag.
RECIPE_FM_COMMON = (("batch_size", 256), ("lr", 1e-3), ("weight_decay", 1e-4),
                    ("min_lr", 1e-6), ("val_ratio", 0.1), ("reref", "laplacian_xyz"),
                    ("use_amp", True))
RECIPE_FM = {
    "probe": (("epochs", 60), ("early_stop_patience", 20), ("warmup_epochs", 10)),
    "ft": (("epochs", 60), ("early_stop_patience", 20), ("warmup_epochs", 5),
           ("ft_lr", 1e-4), ("unfreeze_last_n", 2)),
    "temporal": (("epochs", 100), ("early_stop_patience", 20), ("warmup_epochs", 10),
                 ("seq_len", 64), ("temporal_hidden", 128)),
    "brant": (("epochs", 200), ("early_stop_patience", 30), ("warmup_epochs", 10)),
}


def _same(got, want):
    if isinstance(want, (bool, str)) or want is None:
        return got == want
    try:
        return abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
    except (TypeError, ValueError):
        return False


def _recipe_tags(base, args):
    tags = [f"{k}={v:g}" if isinstance(v, float) else f"{k}={v}"
            for k, want in RECIPE_FM_COMMON + RECIPE_FM[base]
            for v in [args.get(k, want)] if not _same(v, want)]
    if (args.get("max_windows") or 0) > 0 or (args.get("max_anchors") or 0) > 0:
        tags.append("smoke")
    return tags


# ─────────────────────────── loading ────────────────────────────────────────
class Record:
    def __init__(self, adaptation, dataset, fm, regime, seed, scores, source):
        self.key = (adaptation, dataset, fm, regime)
        self.seed = int(seed)
        self.scores = {str(k): float(v) for k, v in scores.items()}
        self.source = source


def _adaptation(fm, mode, head, args):
    """Table 18 adaptation of a result file; a non-paper setting gets a suffix
    (the runners' own result-folder tags, then [recipe tags]) so it is never
    mixed into a paper cell."""
    if fm == "brant":
        tag = base = "brant"
        stride, L = args.get("stride_s"), args.get("context_patches")
        if stride not in (None, 0.5):
            tag += f"_s{stride:g}"
        if L not in (None, 1):
            tag += f"_L{L}"
            if args.get("patch_pool") not in (None, "mean"):
                tag += f"_{args['patch_pool']}"
        if head not in (None, "linear"):
            tag += f"_{head}"
    else:
        if mode == "finetune":
            # Full-backbone fine-tuning is a text-only control (App. A.11), not a table cell.
            if args.get("full_finetune"):
                return None
            base = "ft"
            tag = "ft" if head in (None, "linear") else f"ft_{head}"
        elif mode in (None, "probe"):
            base = tag = "temporal" if head == "temporal" else "probe"
            if head == "mlp":
                tag += "_mlp"
        else:
            return None
        if args.get("bb_pool") not in (None, "center10"):
            tag += f"_{args['bb_pool']}"
        if head == "temporal" and args.get("temporal_direction") not in (None, "bi"):
            tag += f"_{args['temporal_direction']}"
        if args.get("reseed_pooled_head"):   # the runner's probe_reseed folder
            tag += "_reseed"
    tags = _recipe_tags(base, args)
    return f"{tag}[{','.join(tags)}]" if tags else tag


def _seed_from_path(path):
    m = re.search(r"seed(\d+)", os.path.basename(os.path.dirname(path)))
    return int(m.group(1)) if m else 42


def _fold_args(loo_dir):
    """args of any per-fold file next to a _loo_done.json (it stores none itself)."""
    for p in sorted(glob.glob(os.path.join(loo_dir, "ft_*", "results_*.json"))):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh).get("args") or {}
        except (OSError, ValueError):
            continue
    return {}


def classify(path, d):
    if not isinstance(d, dict) or not {"fm", "dataset", "train_mode"} <= d.keys() \
            or not isinstance(d.get("per_subject"), dict):
        return None
    name = os.path.basename(path)
    if d["train_mode"] == "loo" and name != "_loo_done.json":
        return None                     # one LOO fold; the cell is _loo_done.json
    args = d.get("args") if isinstance(d.get("args"), dict) else {}
    if name == "_loo_done.json":
        args = {**_fold_args(os.path.dirname(path)), **args}
    head = d.get("head") or args.get("head")
    adaptation = _adaptation(d["fm"], d.get("mode", args.get("mode")), head, args)
    if adaptation is None:
        return None
    scores = {s: v["corr_mean"] for s, v in d["per_subject"].items()
              if isinstance(v, dict) and "corr_mean" in v}
    seed = args.get("seed", _seed_from_path(path))
    return Record(adaptation, d["dataset"], d["fm"], d["train_mode"], seed, scores, path)


def load_records(cells, verbose=False):
    """(records, incomplete-LOO notes) under `cells` (a folder or a list of them)."""
    roots = [cells] if isinstance(cells, str) else list(cells)
    files = sorted(p for r in roots
                   for p in glob.glob(os.path.join(r, "**", "*.json"), recursive=True))
    if not files:
        raise SystemExit(f"no JSON files under {cells}")
    recs, incomplete = [], []
    for p in files:
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            if verbose:
                print(f"[skip] {p} (unreadable)")
            continue
        r = classify(p, d)
        if r is not None:
            recs.append(r)
        elif verbose:
            print(f"[skip] {p}")
    # LOO folds without the completion marker: incomplete, excluded, reported.
    for fold_dir in sorted({os.path.dirname(os.path.dirname(p)) for p in files
                            if os.path.basename(os.path.dirname(p)).startswith("ft_")}):
        if not os.path.exists(os.path.join(fold_dir, "_loo_done.json")):
            n = len(glob.glob(os.path.join(fold_dir, "ft_*", "results_*.json")))
            incomplete.append(f"{fold_dir}: {n} LOO folds, "
                              "no _loo_done.json (incomplete; not counted)")
    return recs, incomplete


def aggregate(records):
    by = defaultdict(lambda: defaultdict(dict))         # key -> seed -> subj -> r
    origin = {}
    for r in records:
        for s, v in r.scores.items():
            k = (r.key, r.seed, s)
            if k in origin:
                raise SystemExit(f"{r.key} seed {r.seed} subject {s} is in two files:\n"
                                 f"  {origin[k]}\n  {r.source}\nPoint --cells at one set of runs.")
            origin[k] = r.source
            by[r.key][r.seed][s] = v
    cells = {}
    for key, seeds in by.items():
        seed_ids = sorted(seeds, key=lambda s: (s != 42, s))
        subs = sorted(set.intersection(*(set(seeds[sd]) for sd in seed_ids)))
        M = np.array([[seeds[sd][s] for s in subs] for sd in seed_ids], dtype=float)
        per = M.mean(axis=0)
        cells[key] = {
            "key": key, "seeds": seed_ids, "subjects": subs,
            "per_subject": dict(zip(subs, per.tolist())),
            "per_seed_subject": {sd: dict(seeds[sd]) for sd in seed_ids},
            "mean": float(per.mean()) if subs else float("nan"),
            "sd": float(per.std(ddof=1)) if len(subs) > 1 else float("nan"),
            "n_seeds": len(seed_ids),
            "ragged": any(set(seeds[sd]) != set(subs) for sd in seed_ids),
        }
    return cells


def complete(c):
    """The whole cohort, in every seed."""
    return len(c["subjects"]) == N_SUBJECTS.get(c["key"][1]) and not c["ragged"]


def not_comparable(c):
    """Why a fresh cell cannot be scored against its published value ([] if it can)."""
    why = []
    if sorted(c["seeds"]) != sorted(paper_seeds(c["key"])):
        why.append(f"{REGIME_LABEL.get(c['key'][3], c['key'][3])} seeds "
                   f"{','.join(map(str, c['seeds']))}, paper "
                   f"{','.join(map(str, paper_seeds(c['key'])))}")
    if not complete(c):
        why.append(f"{REGIME_LABEL.get(c['key'][3], c['key'][3])} has "
                   f"{len(c['subjects'])} of {N_SUBJECTS.get(c['key'][1])} subjects in every seed")
    return why


def paper_regimes(fm, adaptation, dataset):
    """The regimes the paper's Table 18 cell was chosen from."""
    return [rg for rg in T18_REGIMES[adaptation]
            if (dataset, rg, fm, adaptation) not in NOT_IN_CELLS]


def best_regime(cells, fm, adaptation, dataset):
    """Table 18: among the paper's regimes, the complete cell with the highest mean."""
    found = [cells[(adaptation, dataset, fm, rg)] for rg in paper_regimes(fm, adaptation, dataset)
             if (adaptation, dataset, fm, rg) in cells
             and complete(cells[(adaptation, dataset, fm, rg)])]
    return max(found, key=lambda c: c["mean"]) if found else None


def t18_not_comparable(cells, fm, adaptation, dataset):
    why = []
    for rg in paper_regimes(fm, adaptation, dataset):
        c = cells.get((adaptation, dataset, fm, rg))
        why += [f"no {REGIME_LABEL[rg]} run"] if c is None else not_comparable(c)
    return why


def fmt(x, dp=3):
    return "nan" if x is None or x != x else f"{x:.{dp}f}"


# ─────────────────────────── reporting ──────────────────────────────────────
class Checker:
    """Collects comparisons with the paper. tol=None means exact (string) match,
    used for paper_cells; otherwise a tolerance, used for fresh runs."""

    def __init__(self, tol=None):
        self.tol = tol
        self.n = 0              # published cells looked at
        self.compared = 0       # ... compared with the paper (fresh: scored)
        self.bad, self.missing, self.unreleased, self.unscored = [], [], [], []

    def check(self, where, got, want, scored=True, released=True):
        if want is None:
            return ""
        self.n += 1
        if got is None:
            if released or self.tol is None:
                self.missing.append(f"{where} (paper {want})")
            else:
                self.unreleased.append(where)
            return "  MISSING" if self.tol is None else ""
        if self.tol is not None and not scored:
            self.unscored.append(f"{where}: {fmt(got)} (paper {want})")
            return ""
        self.compared += 1
        ok = fmt(got) == want if self.tol is None else abs(got - float(want)) <= self.tol + 5e-4
        if not ok:
            self.bad.append(f"{where}: {fmt(got)} vs paper {want}")
            return f" <-- paper {want}"
        return ""


def _not_scored_mark(why):
    return f"  [not scored: {'; '.join(sorted(set(why)))}]" if why else ""


def print_table19(cells, ck, out):
    fresh = ck.tol is not None
    out("\nTable 19 - BrainBERT / PopT regression, mean Pearson r per training regime "
        "(pooled / per-subject: seeds 42, 0, 1 averaged within subject; LOO: seed 42).")
    out(f"{'dataset':<10}{'regime':<13}" + "".join(f"{FM_LABEL[f] + ' ' + a:>17}"
                                                   for f, a in T19_COLS))
    for (ds, rg), want in EXPECTED_T19.items():
        vals, marks = [], []
        for (fm, ad), w in zip(T19_COLS, want):
            c = cells.get((ad, ds, fm, rg))
            where = f"T19 {ds} {REGIME_LABEL[rg]} {FM_LABEL[fm]} {ad}"
            if (ds, rg, fm, ad) in NOT_IN_CELLS:
                printed = NOT_IN_CELLS[(ds, rg, fm, ad)][0]
                vals.append(f"{printed + '~' if c is None else fmt(c['mean']) + '*':>17}")
                if c is not None:
                    marks.append(f"{where}: fresh run; the paper prints {printed} "
                                 "from an incomplete archived run")
                continue
            vals.append(f"{fmt(c['mean']) if c else '--':>17}")
            why = not_comparable(c) if (fresh and c is not None) else []
            m = ck.check(where, c["mean"] if c else None, w, scored=not why,
                         released=ds in RELEASED_DATASETS)
            if m.strip() and c is not None:          # absent cells are counted, not listed
                marks.append(where + m)
            if why:
                marks.append(where + _not_scored_mark(why))
        out(f"{ds:<10}{REGIME_LABEL[rg]:<13}{''.join(vals)}")
        for m in marks:
            out(f"    {m}")
    for (ds, rg, fm, ad), (printed, why) in NOT_IN_CELLS.items():
        out(f"  ~ {ds} {REGIME_LABEL[rg]} {FM_LABEL[fm]} {ad}: printed in the paper as "
            f"{printed}, not checked: {why}.")
    for (ds, rg, fm, ad), (printed, why) in PRINTED_AS.items():
        out(f"  {ds} {REGIME_LABEL[rg]} {FM_LABEL[fm]} {ad}: printed in the paper as "
            f"{printed} ({why}).")


def print_table18(cells, ck, out):
    """Prints Table 18; returns {(fm, adaptation, dataset): (cell, reasons it is not scored)}."""
    fresh = ck.tol is not None
    out("\nTable 18 - iEEG-FM adaptation, best regime per cell. Mean Pearson r +- "
        "cross-subject SD (n=9 finger, 16 audio).")
    out(f"{'model':<11}{'adaptation':<28}  {'finger (Stanford)':<34}{'audio (Ghent)'}")
    best = {}
    for fm, ad in T18_ROWS:
        row, marks = [], []
        for ds in DATASETS:
            c = best_regime(cells, fm, ad, ds)
            want_m, want_s, want_p, want_rg = EXPECTED_T18[(fm, ad)][ds]
            where = f"T18 {FM_LABEL[fm]} {ad} {TASK[ds]}"
            if c is None:
                row.append(f"  {'--':<32}")
                ck.check(where, None, f"{want_m}+-{want_s}", released=ds in RELEASED_DATASETS)
                continue
            why = t18_not_comparable(cells, fm, ad, ds) if fresh else []
            best[(fm, ad, ds)] = (c, why)
            n = c["n_seeds"]
            txt = (f"{fmt(c['mean'])}+-{fmt(c['sd'])}{f' ({n} seeds)' if n > 1 else ''}"
                   f" [{REGIME_LABEL[c['key'][3]]}]")
            row.append(f"  {txt:<32}")
            for m in (ck.check(where + " mean", c["mean"], want_m, scored=not why),
                      ck.check(where + " SD", c["sd"], want_s, scored=not why)):
                if m.strip():
                    marks.append(where + m)
            if why:
                marks.append(where + _not_scored_mark(why))
            if fresh and c["key"][3] != want_rg:
                marks.append(f"NOTE {where}: best regime here {REGIME_LABEL[c['key'][3]]}, "
                             f"paper {REGIME_LABEL[want_rg]}")
            if not fresh:
                ck.n += 1
                if (n > 1) != want_p:
                    ck.bad.append(f"{where}: three-seed mark {n > 1} vs paper {want_p}")
                    marks.append(f"{where}: three-seed mark differs from the paper")
        out(f"{FM_LABEL[fm]:<11}{ADAPT_LABEL[ad]:<28}{''.join(row)}")
        for m in marks:
            out(f"    {m}")
        extra = [rg for rg in ALLOWED_REGIMES[ad] if rg not in T18_REGIMES[ad]]
        for rg in extra:                  # Brant's pooled head: shown, never chosen
            shown, notes = [], []
            for ds in DATASETS:
                c = cells.get((ad, ds, fm, rg))
                if c is None:
                    continue
                shown.append(f"{TASK[ds]} {fmt(c['mean'])}+-{fmt(c['sd'])}")
                chosen = best.get((fm, ad, ds))
                if fresh and chosen and complete(c) and c["mean"] > chosen[0]["mean"]:
                    notes.append(f"NOTE T18 {FM_LABEL[fm]} {ad} {TASK[ds]}: the {rg} head "
                                 f"({fmt(c['mean'])}) beats the "
                                 f"{REGIME_LABEL[chosen[0]['key'][3]]} cell here; the "
                                 "paper's cell is the per-subject run")
            if shown:
                out(f"    {FM_LABEL[fm]} {REGIME_LABEL[rg]} head (not reported in the paper; "
                    f"not a Table 18 candidate): {', '.join(shown)}")
            for m in notes:
                out(f"    {m}")
    if len(best) == 2 * len(T18_ROWS):
        n_gt = sum(c["sd"] > c["mean"] for c, _ in best.values())
        top = max((c for c, _ in best.values()), key=lambda c: c["mean"])
        out(f"  caption: SD > mean in {n_gt}/{len(best)} cells (paper {EXPECTED_SD_GT_MEAN}/14); "
            f"highest mean {fmt(top['mean'], 4)} (paper: none reaches r = {CEILING})")
        if not fresh:
            ck.n += 2
            if n_gt != EXPECTED_SD_GT_MEAN:
                ck.bad.append(f"T18 caption: SD > mean in {n_gt}/14, paper {EXPECTED_SD_GT_MEAN}/14")
            if top["mean"] >= CEILING:
                ck.bad.append(f"T18 caption: a cell reaches {top['mean']:.4f} >= {CEILING}")
    return best


def print_table1(best, ck, out):
    fresh = ck.tol is not None
    out("\nTable 1 - iEEG-FM rows: the best Table 18 adaptation per model and task.")
    out(f"{'model':<11}  {'finger (r, n=9)':<32}{'audio (r, n=16)'}")
    for fm in ("brainbert", "popt", "brant"):
        row, marks = [], []
        for ds in DATASETS:
            cand = [(ad, c, why) for (f, ad, d), (c, why) in best.items() if f == fm and d == ds]
            want_m, want_s, want_ad = EXPECTED_T1[fm][ds]
            where = f"T1 {FM_LABEL[fm]} {TASK[ds]}"
            if not cand:
                row.append(f"  {'--':<30}")
                ck.check(where, None, f"{want_m}+-{want_s}", released=ds in RELEASED_DATASETS)
                continue
            ad, c, _ = max(cand, key=lambda t: t[1]["mean"])
            behind = [a for f, a in T18_ROWS if f == fm]
            why = ([f"no {ADAPT_SHORT[a]} cell" for a in behind if a not in {t[0] for t in cand}]
                   + [f"{ADAPT_SHORT[a]} not scored" for a, _, w in cand if w]) if fresh else []
            row.append(f"  {fmt(c['mean']) + '+-' + fmt(c['sd']) + ' [' + ADAPT_SHORT[ad] + ']':<30}")
            for m in (ck.check(where + " mean", c["mean"], want_m, scored=not why),
                      ck.check(where + " SD", c["sd"], want_s, scored=not why)):
                if m.strip():
                    marks.append(where + m)
            if why:
                marks.append(where + _not_scored_mark(why))
            if fresh and ad != want_ad:
                marks.append(f"NOTE {where}: best adaptation here {ADAPT_SHORT[ad]}, "
                             f"paper {ADAPT_SHORT[want_ad]}")
        out(f"{FM_LABEL[fm]:<11}{''.join(row)}")
        for m in marks:
            out(f"    {m}")


def _seed_diff(got, want):
    """(subjects, cohort-mean d, mean |d|, max |d|) over the subjects both have."""
    common = sorted(set(got) & set(want))
    if not common:
        return common, 0.0, 0.0, 0.0
    d = np.array([got[s] - want[s] for s in common])
    dm = float(np.mean([got[s] for s in common]) - np.mean([want[s] for s in common]))
    return common, dm, float(np.mean(np.abs(d))), float(np.max(np.abs(d)))


def seed_spread(ref_cell):
    """The largest cohort |d|, mean |d| and max |d| between any two of the
    reference cell's own seeds (None for a one-seed cell)."""
    seeds = ref_cell["seeds"]
    if len(seeds) < 2:
        return None
    worst = [0.0, 0.0, 0.0]
    for a, b in combinations(seeds, 2):
        _, dm, mad, mx = _seed_diff(ref_cell["per_seed_subject"][a],
                                    ref_cell["per_seed_subject"][b])
        worst = [max(worst[0], abs(dm)), max(worst[1], mad), max(worst[2], mx)]
    return tuple(worst)


def seed_matched(cells, ref, tol, tol_subject, tol_subject_max, out):
    """Fresh vs reference, seed by seed: cohort-mean difference, mean absolute
    per-subject difference, and the largest per-subject difference. Returns
    (failures, number of seeds compared, failures within the reference's own
    seed-to-seed spread); a seed that lacks some of the reference's subjects is
    shown as partial and not scored. As in aggregate_btb, the check presumes
    the fresh run draws its random numbers as the paper run did; the
    reference's seed-to-seed spread is printed for its three-seed cells."""
    bad, n, any_, like_seed = [], 0, False, 0
    out(f"\nSeed-matched comparison with the reference cells (tolerances: cohort "
        f"{tol}, mean |d| per subject {tol_subject}, any subject {tol_subject_max}):")
    for key, c in sorted(cells.items()):
        r = ref.get(key)
        if r is None or not any(sd in r["seeds"] for sd in c["seeds"]):
            continue
        spread = seed_spread(r)
        if spread is not None:
            inside = (spread[0] <= tol and spread[1] <= tol_subject
                      and spread[2] <= tol_subject_max)
            out(f"  {'/'.join(key):<36} the reference seeds {'/'.join(map(str, r['seeds']))} "
                f"differ from each other by up to cohort {spread[0]:.4f}, mean|d| "
                f"{spread[1]:.4f}, any subject {spread[2]:.4f}"
                + ("  (inside the tolerances: this check cannot tell another seed "
                   "from the same one)" if inside else ""))
        for sd in c["seeds"]:
            if sd not in r["seeds"]:
                continue
            got, want = c["per_seed_subject"][sd], r["per_seed_subject"][sd]
            common, dm, mad, mx = _seed_diff(got, want)
            if not common:
                continue
            any_ = True
            d = np.array([got[s] - want[s] for s in common])
            worst = common[int(np.argmax(np.abs(d)))]
            line = (f"  {'/'.join(key):<36} seed {sd:<3} n={len(common):<3} cohort d={dm:+.4f}  "
                    f"mean|d|={mad:.4f}  worst {worst} d={d[common.index(worst)]:+.4f}")
            if len(common) < len(want):
                out(line + f"  (partial: {len(common)} of {len(want)} subjects; not scored)")
                continue
            n += 1
            ok = abs(dm) <= tol and mad <= tol_subject and mx <= tol_subject_max
            if ok:
                out(line)
                continue
            seedlike = (spread is not None and abs(dm) <= spread[0] and mad <= spread[1]
                        and mx <= spread[2])
            like_seed += seedlike
            out(line + "  <-- outside tolerance"
                + ("; within the reference's seed-to-seed spread" if seedlike else ""))
            bad.append(f"{'/'.join(key)} seed {sd}: seed-matched cohort d={dm:+.4f}, "
                       f"mean|d|={mad:.4f}, max|d|={mx:.4f}"
                       + (" (within the reference's seed-to-seed spread)" if seedlike else ""))
    if not any_:
        out("  (no seed in common with the reference)")
    return bad, n, like_seed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=[DEFAULT_CELLS],
                   help="one or more folders of result JSONs "
                        "(default: paper_cells/fm_regression)")
    p.add_argument("--reference", default=DEFAULT_CELLS,
                   help="per-seed reference for the seed-matched check of fresh runs "
                        "(default: paper_cells/fm_regression)")
    p.add_argument("--tol", type=float, default=None,
                   help="fresh runs: allowed |d| of a cohort value, default 0.005. "
                        "paper_cells are checked exactly")
    p.add_argument("--tol_subject", type=float, default=0.01,
                   help="seed-matched check: allowed mean |d| over subjects")
    p.add_argument("--tol_subject_max", type=float, default=0.03,
                   help="seed-matched check: allowed |d| of any one subject")
    p.add_argument("--verbose", action="store_true", help="list skipped files")
    a = p.parse_args(argv)
    for c in a.cells:
        if not os.path.isdir(c):
            p.error(f"--cells {c!r} is not a folder")
    if not os.path.isdir(a.reference):
        p.error(f"--reference {a.reference!r} is not a folder")

    fresh = [os.path.realpath(c) for c in a.cells] != [os.path.realpath(DEFAULT_CELLS)]
    if not fresh and a.tol is not None:
        p.error("--tol applies to fresh runs (--cells DIR); the paper cells are checked "
                "exactly against the printed values")
    where_cells = " ".join(a.cells)
    tol = 0.005 if fresh and a.tol is None else a.tol

    def out(line):
        print(line.rstrip())

    recs, incomplete = load_records(a.cells, a.verbose)
    cells = aggregate(recs)
    out(f"cells: {where_cells}  ({len(cells)} adaptation x dataset x model x regime cells)")
    paper_adaptations = {ad for _, ad in T18_ROWS}
    for key, c in cells.items():
        n_expected = N_SUBJECTS.get(key[1])
        if c["ragged"] or (n_expected and len(c["subjects"]) != n_expected):
            out(f"WARNING {'/'.join(key)}: {len(c['subjects'])} subjects common to all "
                f"seeds {c['seeds']} (the paper has {n_expected}); not used in Tables 18 and 1")
        if key[0] in paper_adaptations and key[3] in ALLOWED_REGIMES[key[0]] \
                and sorted(c["seeds"]) != sorted(paper_seeds(key)):
            out(f"NOTE {'/'.join(key)}: seeds {c['seeds']}; the paper cell uses seeds "
                f"{list(paper_seeds(key))}"
                + (": shown, not scored; the seed-matched comparison checks each seed"
                   if fresh else ""))
    for m in incomplete:
        out(f"NOTE {m}")

    ck = Checker(tol)
    print_table19(cells, ck, out)
    best = print_table18(cells, ck, out)
    print_table1(best, ck, out)
    other = sorted(k for k in cells
                   if k[0] not in paper_adaptations or k[3] not in ALLOWED_REGIMES[k[0]])
    if other:
        out("\nOther cells (settings outside the paper tables; not compared):")
        for k in other:
            c = cells[k]
            out(f"  {'/'.join(k):<40} {fmt(c['mean'], 4)}+-{fmt(c['sd'], 4)}  "
                f"n={len(c['subjects'])}  seeds {c['seeds']}")

    if not fresh:
        failed = ck.bad + ck.missing
        out(f"\n{ck.n} cells checked against the paper: "
            + ("all match." if not failed else f"{len(failed)} MISMATCH:"))
        for b in failed:
            out(f"  {b}")
        return 1 if failed else 0

    sm_bad, n_sm, n_seedlike = seed_matched(cells, aggregate(load_records(a.reference)[0]),
                                            tol, a.tol_subject, a.tol_subject_max, out)
    out(f"\n{ck.compared} of {ck.n} published cells compared with the paper (tol {tol}): "
        f"{len(ck.bad)} outside tolerance.")
    if ck.unscored:
        out(f"{len(ck.unscored)} shown but not scored: seeds, subjects or regimes differ "
            "from the paper cell's (see the [not scored] marks).")
    if ck.missing:
        out(f"{len(ck.missing)} not in {where_cells}.")
    if ck.unreleased:
        out(f"{len(ck.unreleased)} Ghent cells: the Ghent data are not released.")
    out(f"{n_sm} seeds compared with the same seed in {a.reference}: "
        f"{len(sm_bad)} outside tolerance.")
    if n_seedlike:
        verb = "is" if n_seedlike == 1 else "are"
        out(f"{n_seedlike} of them {verb} no further from their reference seed than "
            "the reference's own seeds are from each other: what a run that draws its "
            "random numbers in another order would give.")
    failed = ck.bad + sm_bad
    if ck.compared + n_sm == 0:
        out("\nNOTHING VERIFIED: no published cell had the paper's seeds and subjects, and "
            "no seed matched the reference with all its subjects.")
        return 1
    out("\nResult: " + (f"{len(failed)} FAILED:" if failed else "all compared cells and "
                        "seeds are within tolerance."))
    for b in failed:
        out(f"  {b}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
