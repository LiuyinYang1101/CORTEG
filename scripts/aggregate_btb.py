#!/usr/bin/env python
r"""Reprint the BrainTreebank tables (paper Tables 3, 9 and 20) from per-cell result files.

    python scripts/aggregate_btb.py                    # the published cells, checked
    python scripts/aggregate_btb.py --cells DIR [DIR ...]   # fresh runs vs the paper

With the release scripts' default output folders:

    R="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/braintreebank"
    python scripts/aggregate_btb.py --cells "$R/table3" "$R/fm_runs"

With no arguments this reads ``paper_cells/btb``, the result files behind the
published numbers, rebuilds every cell of Tables 3, 9 and 20 and checks it
against the value the paper prints, as well as the oracle null the Table 9
caption quotes ("≈0.53": both cohort means at 2 dp). It exits non-zero if any
cell differs.

Aggregation rule, the one the paper uses:

* per-subject score  = mean AUROC over the 4 causal folds (stored in each file);
* multi-seed arms    = the seeds are averaged within each subject first;
* cohort cell        = mean over the 10 subjects +- sample SD (ddof=1) across
                       subjects, computed on those seed-averaged scores;
* seed SD (Table 20) = sample SD (ddof=1) of the per-seed cohort means;
* rounding           = once, from full precision: 4 dp for Table 20, 3 dp for
                       Tables 3 and 9.

CORTEG, its random-init control, and the three scratch decoders use training
seeds 42, 1 and 2 on ONE event set, drawn with seed 42; every other arm is a
single seed-42 run or a deterministic frozen probe.

Fresh runs (``--cells DIR [DIR ...]``)
--------------------------------------
Every ``*.json`` under each DIR is read and classified by its contents, not its
name, so any layout works; a DIR that is not a folder is an error. Recognised
producers (and their file names):

  experiments/run_btb_classification.py   btb_<train_mode>_<merge>_<endpoint>[_tags]_seed<S>.json
                                          CORTEG gate, mean-pool, random init
  experiments/run_popt_finetune_btb.py    popt_<endpoint>_<mode>_seed<S>.json
                                          PopT LoRA, full fine-tune, head-only
  experiments/run_ieeg_fm_baselines.py    <fm>_<endpoint>_<arm>_seed<S>.json
                                          frozen BrainBERT / PopT / Brant arms
  and the paper-run formats in paper_cells/btb (see paper_cells/MANIFEST.md).

The raw spectral probes, the from-scratch decoders (HiLoFuseNet, CNN-LSTM,
LSTM), the BrainBERT and Brant adaptation arms (head-only, LoRA, full
fine-tune) and the oracle permutation null have no runner in this release;
their cells are reported as "not released", not as missing. Their paper-run
files in paper_cells/btb are still read and checked.

A run at a setting outside the paper becomes its own row, listed under "Other
cells" and never averaged into a published one. The settings checked are the
event set (event seed, events per class, Task B negatives), the training
arrangement (per-subject training, shared LoRA, per-subject model selection,
a loss other than BCE), the subjects of a CORTEG run and their training order (in pooled training the order is part of the run), a forced
--trial, the CORTEG warm-up, and every recipe value in RECIPE_* below (folds,
window, high-gamma band, backbone size, gate activation, epochs, learning
rates, gradient clipping, validation interval, LoRA rank, batch size, ...).
The recipe values are the ones the paper runs record; a setting a file does
not record is taken to be the paper's. As a last resort, a file whose runner
records its own non-paper settings (``nonpaper_settings``, ``name_tags``) is
tagged with them when none of these checks fired. For example a one-epoch
smoke run on 50 events per class becomes ``corteg_gate[n50,epochs=1]``. A
file with no event seed comes from the runner version whose --seed also drew
the events, so its training seed is taken as its event seed. Unrecognised JSON
(summaries, caches, logs) is skipped; ``--verbose`` lists it. The same (arm,
task, seed, subject) found in two files is an error rather than a silent
average: point --cells at one set of runs.

What a fresh run is checked against:

1. A published cell is scored only when the fresh cell has the paper row's
   seeds (42, 1, 2 for the three-seed rows, 42 otherwise), all 10 subjects,
   and one event set across its seeds. Anything else (typically a single
   seed-42 run of a three-seed row) is printed, marked "not scored", and left
   out of the exit status: one seed is not comparable with a three-seed mean.
   Seed SDs (an SD of three numbers) are printed but never scored.
2. Every fresh seed that also exists in ``--reference`` (default
   paper_cells/btb) is compared with that same seed, subject by subject. This
   is the check that a single-seed rerun can pass or fail.
3. Seeds of one row scored on different event sets fail.

The random stream. Check 2 and the tolerances below come from repeating the
paper code at a fixed seed, so they hold only if the released runner draws
its random numbers (initialisation, data order, dropout) in the order the
paper code did. A port that draws them in another order behaves like another
seed, and a correct port would then fail check 2. So for each multi-seed row
the reference's own seed-to-seed spread is printed beside the seed-matched
lines, and a failure that is no further from its reference seed than the
paper's seeds are from each other is marked as such: that is the signature of
another random stream, not necessarily of another recipe. Where that spread is
itself inside the tolerances (e.g. HiLoFuseNet, Task B), check 2 cannot tell
another seed from the same one, and the line says so.

Tolerances. The run-to-run floor was measured for the paper from three runs
of the identical Task B gate configuration at seed 42 (the paper's run and two
repeats; they differ only by GPU nondeterminism). The largest difference
between any two of the three is 0.0074 in the cohort mean, 0.0098 in the mean
absolute per-subject difference and 0.0284 for any one subject. Three runs
understate the spread, so each default is that largest difference times 1.5
(about two standard deviations of the difference between two runs, as far as
three runs can estimate it): ``--tol`` 0.0111 (cohort mean, also used for
published cohort cells), ``--tol_subject`` 0.0148 (mean |d| over subjects) and
``--tol_subject_max`` 0.0426 (any one subject, also used for Table 9 cells).
The floor was measured on that one arm and task only; it is applied to Task A
and to every other arm unmeasured. They are estimates, not constants: a seed
that lands just outside should be rerun before it is read as a failed
reproduction. ``--tol`` applies to fresh runs only; the paper cells are
always checked exactly.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
from itertools import combinations
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CELLS = os.path.join(REPO, "paper_cells", "btb")
SUBJECTS = [f"sub_{i}" for i in range(1, 11)]
TASKS = ("sentence_onset", "word_nonword")          # Task A, Task B
TASK_LABEL = {"sentence_onset": "A", "word_nonword": "B"}

# Table 20, in the paper's order: (key, label, group heading).
TABLE20 = [
    ("raw_hga_perchannel", "Raw high-gamma, per-channel", "Raw spectral features (linear probe)"),
    ("raw_bandpower_perchannel", "Raw band-power, per-channel", None),
    ("raw_hga_channelmean", "Raw high-gamma, channel-mean", None),
    ("corteg_gate", "CORTEG (layer-wise gate)", "CORTEG (pooled, per-subject LoRA)"),
    ("corteg_meanpool", "CORTEG (mean-pool fusion)", None),
    ("corteg_randinit", "CORTEG, random-init backbone", None),
    ("hilofusenet", "HiLoFuseNet", "Trained from scratch"),
    ("cnn_lstm", "CNN-LSTM", None),
    ("lstm", "LSTM", None),
    ("popt_lora", "PopT, LoRA", "PopT"),
    ("popt_full_ft", "PopT, full fine-tune", None),
    ("popt_head_only", "PopT, head-only", None),
    ("popt_frozen", "PopT, frozen probe", None),
    ("bb_single_max", "BrainBERT, single-elec. max (oracle)", "BrainBERT"),
    ("bb_pop_meanpool", "BrainBERT, population mean-pool", None),
    ("bb_full_ft", "BrainBERT, full fine-tune", None),
    ("bb_head_only", "BrainBERT, head-only", None),
    ("bb_lora", "BrainBERT, LoRA", None),
    ("bb_single_mean", "BrainBERT, single-elec. mean", None),
    ("brant_single_max", "Brant, single-elec. max (oracle)", "Brant (505M parameters)"),
    ("brant_head_only", "Brant, head-only", None),
    ("brant_lora", "Brant, LoRA", None),
    ("brant_full_ft", "Brant, full fine-tune", None),
    ("brant_single_mean", "Brant, single-elec. mean", None),
    ("brant_pop_meanpool", "Brant, population mean-pool", None),
]
# Table 3 is each method's best configuration; Table 9 is its per-subject breakdown.
TABLE3 = [
    ("bb_single_max", "BrainBERT (oracle)"),
    ("popt_lora", "PopT (LoRA)"),
    ("brant_single_max", "Brant (oracle)"),
    ("hilofusenet", "HiLoFuseNet"),
    ("corteg_randinit", "CORTEG, random init"),
    ("corteg_gate", "CORTEG (ours)"),
]
# Rows whose runner is not part of this release (their result files are in
# paper_cells/btb, so the published cells still recompute).
NOT_RELEASED = {"raw_hga_perchannel", "raw_bandpower_perchannel", "raw_hga_channelmean",
                "hilofusenet", "cnn_lstm", "lstm",
                "bb_full_ft", "bb_head_only", "bb_lora",
                "brant_head_only", "brant_lora", "brant_full_ft"}

# ─────────────────────────── the paper's values ─────────────────────────────
# Exact rounding of the artifacts in paper_cells/btb, as the paper prints them.
# (mean, sd) per task; seed SD (A, B) where the arm has three seeds.
EXPECTED_T20 = {
    "raw_hga_perchannel":       (("0.5846", "0.0296"), ("0.8155", "0.0623"), None),
    "raw_bandpower_perchannel": (("0.5796", "0.0344"), ("0.7978", "0.0728"), None),
    "raw_hga_channelmean":      (("0.5127", "0.0230"), ("0.5217", "0.0485"), None),
    "corteg_gate":              (("0.6376", "0.0783"), ("0.7492", "0.1345"), ("0.0135", "0.0146")),
    "corteg_meanpool":          (("0.6159", "0.0721"), ("0.7525", "0.1326"), ("0.0318", "0.0036")),
    "corteg_randinit":          (("0.5376", "0.0536"), ("0.5880", "0.1025"), ("0.0031", "0.0184")),
    "hilofusenet":              (("0.5330", "0.0220"), ("0.7208", "0.0886"), ("0.0217", "0.0018")),
    "cnn_lstm":                 (("0.5071", "0.0142"), ("0.5706", "0.0469"), ("0.0055", "0.0032")),
    "lstm":                     (("0.5073", "0.0136"), ("0.5618", "0.0429"), ("0.0036", "0.0006")),
    "popt_lora":                (("0.6003", "0.0845"), ("0.7790", "0.0991"), None),
    "popt_full_ft":             (("0.6001", "0.0726"), ("0.6979", "0.1228"), None),
    "popt_head_only":           (("0.5594", "0.0442"), ("0.6714", "0.0915"), None),
    "popt_frozen":              (("0.5697", "0.0358"), ("0.6698", "0.0814"), None),
    "bb_single_max":            (("0.6015", "0.0450"), ("0.6980", "0.0941"), None),
    "bb_pop_meanpool":          (("0.5294", "0.0280"), ("0.5788", "0.0664"), None),
    "bb_full_ft":               (("0.5480", "0.0343"), ("0.5682", "0.0775"), None),
    "bb_head_only":             (("0.5446", "0.0258"), ("0.5483", "0.0563"), None),
    "bb_lora":                  (("0.5463", "0.0243"), ("0.5412", "0.0534"), None),
    "bb_single_mean":           (("0.5085", "0.0063"), ("0.5236", "0.0166"), None),
    "brant_single_max":         (("0.5865", "0.0415"), ("0.5775", "0.0366"), None),
    "brant_head_only":          (("0.5245", "0.0168"), ("0.5423", "0.0183"), None),
    "brant_lora":               (("0.5174", "0.0147"), ("0.5322", "0.0268"), None),
    "brant_full_ft":            (("0.5310", "0.0377"), ("0.5283", "0.0314"), None),
    "brant_single_mean":        (("0.5216", "0.0114"), ("0.5261", "0.0107"), None),
    "brant_pop_meanpool":       (("0.5208", "0.0158"), ("0.5233", "0.0120"), None),
}
# Table 3 = the Mean column of Table 9: (mean, sd) for Task A, Task B.
EXPECTED_T3 = {
    "bb_single_max":    (("0.602", "0.045"), ("0.698", "0.094")),
    "popt_lora":        (("0.600", "0.084"), ("0.779", "0.099")),
    "brant_single_max": (("0.587", "0.041"), ("0.577", "0.037")),
    "hilofusenet":      (("0.533", "0.022"), ("0.721", "0.089")),
    "corteg_randinit":  (("0.538", "0.054"), ("0.588", "0.102")),
    "corteg_gate":      (("0.638", "0.078"), ("0.749", "0.134")),
}
# Table 9, S1..S10.
EXPECTED_T9 = {
    ("bb_single_max", "sentence_onset"):
        "0.577 0.578 0.616 0.641 0.680 0.634 0.546 0.614 0.532 0.598",
    ("popt_lora", "sentence_onset"):
        "0.626 0.644 0.613 0.683 0.729 0.513 0.502 0.513 0.510 0.671",
    ("brant_single_max", "sentence_onset"):
        "0.651 0.545 0.637 0.605 0.622 0.563 0.537 0.596 0.542 0.566",
    ("hilofusenet", "sentence_onset"):
        "0.541 0.521 0.540 0.558 0.561 0.530 0.519 0.555 0.493 0.514",
    ("corteg_randinit", "sentence_onset"):
        "0.556 0.538 0.682 0.510 0.540 0.517 0.507 0.503 0.514 0.509",
    ("corteg_gate", "sentence_onset"):
        "0.745 0.627 0.756 0.671 0.636 0.579 0.545 0.672 0.514 0.631",
    ("bb_single_max", "word_nonword"):
        "0.733 0.719 0.737 0.678 0.796 0.873 0.619 0.654 0.552 0.619",
    ("popt_lora", "word_nonword"):
        "0.829 0.844 0.869 0.776 0.865 0.901 0.671 0.737 0.604 0.693",
    ("brant_single_max", "word_nonword"):
        "0.657 0.556 0.599 0.559 0.620 0.544 0.561 0.574 0.550 0.555",
    ("hilofusenet", "word_nonword"):
        "0.756 0.718 0.863 0.746 0.821 0.721 0.712 0.693 0.549 0.629",
    ("corteg_randinit", "word_nonword"):
        "0.764 0.590 0.779 0.586 0.523 0.562 0.510 0.560 0.515 0.489",
    ("corteg_gate", "word_nonword"):
        "0.912 0.802 0.891 0.834 0.625 0.861 0.550 0.787 0.588 0.643",
}
# Tables 3 and 9 cells whose full-precision value sits just below a 3-dp
# rounding boundary (e.g. 0.134495): rounding Table 20's 4-dp value a second
# time would give 0.001 more, the value listed here. The paper and the
# expected values above round once; tests/test_paper_cells.py checks it.
DOUBLE_ROUNDING_TRAPS = {
    ("T3", "popt_lora", "sentence_onset", "sd"): "0.085",        # 0.084473
    ("T3", "brant_single_max", "sentence_onset", "sd"): "0.042",  # 0.041491
    ("T3", "brant_single_max", "word_nonword", "mean"): "0.578",  # 0.577465
    ("T3", "corteg_randinit", "word_nonword", "sd"): "0.103",     # 0.102461
    ("T3", "corteg_gate", "word_nonword", "sd"): "0.135",         # 0.134495
    ("T9", "brant_single_max", "sentence_onset", "sub_3"): "0.638",   # 0.637464
    ("T9", "corteg_randinit", "sentence_onset", "sub_4"): "0.511",    # 0.510495
    ("T9", "corteg_gate", "sentence_onset", "sub_2"): "0.628",        # 0.627483
    ("T9", "popt_lora", "word_nonword", "sub_9"): "0.605",            # 0.604489
    ("T9", "brant_single_max", "word_nonword", "sub_5"): "0.621",     # 0.620459
    ("T9", "hilofusenet", "word_nonword", "sub_10"): "0.630",         # 0.629466
    ("T9", "corteg_randinit", "word_nonword", "sub_3"): "0.780",      # 0.779499
    ("T9", "corteg_randinit", "word_nonword", "sub_4"): "0.587",      # 0.586452
    ("T9", "corteg_randinit", "word_nonword", "sub_7"): "0.511",      # 0.510457
}
# Seeds behind each published row; everything else is one seed-42 run.
PAPER_SEEDS = {k: (42, 1, 2) for k in
               ("corteg_gate", "corteg_meanpool", "corteg_randinit",
                "hilofusenet", "cnn_lstm", "lstm")}
# Oracle permutation null (Table 9 caption): "≈0.53". Measured on a 1-D high-gamma
# feature, so it is a lower bound for the fitted probes.
EXPECTED_NULL = "0.53"

# Run-to-run floor: the largest difference between any two of three runs of the
# identical Task B gate configuration at seed 42 (the paper's run and two
# repeats). The defaults are these times FLOOR_MARGIN, rounded up at 4 dp.
FLOOR = {"cohort": 0.007379, "mean_abs": 0.009842, "max_abs": 0.028356}
FLOOR_MARGIN = 1.5


def _floor_tol(name):
    return math.ceil(FLOOR[name] * FLOOR_MARGIN * 1e4) / 1e4


TOL, TOL_SUBJECT, TOL_SUBJECT_MAX = (_floor_tol("cohort"), _floor_tol("mean_abs"),
                                     _floor_tol("max_abs"))

# ─────────────────────────── the paper recipes ──────────────────────────────
# Settings recorded in the paper-run files (and the released runners' defaults).
# A result file that records a different value gets "<name>=<value>" as a tag.
# Each entry is (names, paper value); names lists the key's spellings, and a
# callable value accepts or rejects the recorded one. Besides these, classify()
# tags the training arrangement, the subjects and their order (_subject_tag), a
# forced --trial, the CORTEG warm-up (_warmup_tag), and, as a last resort, the
# non-paper settings a released runner records about itself (_runner_tags).
RECIPE_CORTEG = (
    (("n_folds", "folds"), 4), ("win_sec", 1.5), ("pre_sec", 0.0),
    ("hga_low", 70.0), ("hga_high", 200.0),
    (("steegformer_variant", "variant"), "small"),
    ("layerwise_gate_act", "tanh"), ("layerwise_gate_bottleneck", 16),
    ("epochs", 60), ("lr", 3e-4), (("weight_decay", "wd"), 0.005),
    ("min_lr", 1e-5), ("max_norm", 1.0),
    ("batch_size", 16), ("accum_iter", 4), ("patience", 15), ("eval_every", 2),
    ("val_frac", 0.15), ("lora_last_n", 4), ("lora_r", 4), ("lora_alpha", 16),
    ("lora_dropout", 0.2), ("head_dropout", 0.0),
    # the backbone config (and checkpoint) file; the runner requires one
    ("model_kwargs_json", lambda v: os.path.basename(str(v)) in ("", "steegformer_small.json")),
    ("full_finetune", False), ("use_amp", True),
)
RECIPE_DECODER = (
    (("n_folds", "folds"), 4), ("win_sec", 1.5), ("pre_sec", 0.0),
    ("hga_band", (70.0, 200.0)), ("val_frac", 0.15),
    ("epochs", 40), ("patience", 10), ("lr", 1e-3), ("weight_decay", 1e-4),
    ("batch_size", 64), ("hidden", 256), ("dropout", 0.5), ("D", 16),
)
RECIPE_POPT = (
    (("folds", "n_folds"), 4), ("val_frac", 0.15), ("epochs", 60), ("patience", 8),
    ("eval_every", 2), ("lr", 1e-4), ("head_lr", 1e-3), (("wd", "weight_decay"), 5e-3),
    ("max_norm", 1.0), ("batch_size", 32),
    ("lora_r", 4), ("lora_alpha", 16), ("lora_dropout", 0.2), ("lora_last_n", 4),
    ("use_amp", True),
)
RECIPE_FOLDS = ((("n_folds", "folds"), 4),)
RECIPE_FROZEN = ((("n_folds", "folds"), 4), ("val_frac", 0.0))

# Training order of the pooled CORTEG and scratch-decoder paper runs (recorded
# in the CORTEG files; the decoder files record none and used the same list).
# In pooled training a subject's position is its sid (its LoRA adapter, its
# slot in the batch interleave), so the same ten subjects in another order are
# another run.
PAPER_SUBJECT_ORDER = ["sub_1", "sub_2", "sub_3", "sub_4", "sub_6", "sub_7", "sub_10",
                       "sub_5", "sub_8", "sub_9"]

# ─────────────────────────── classification ─────────────────────────────────
FROZEN_ARMS = {                   # "arms" keys of the frozen-probe cell files
    "rawHGA_perchannel": "raw_hga_perchannel",
    "rawBandpower_perchannel": "raw_bandpower_perchannel",
    "rawHGA_popmean_legacy": "raw_hga_channelmean",
    "PopT_population": "popt_frozen",
    "BrainBERT_single_elec_max": "bb_single_max",
    "BrainBERT_single_elec_mean": "bb_single_mean",
    "BrainBERT_pop_meanpool": "bb_pop_meanpool",
    "Brant_single_elec_max": "brant_single_max",
    "Brant_single_elec_mean": "brant_single_mean",
    "Brant_population": "brant_pop_meanpool",
}
ADAPT_ARMS = {                    # per-subject FM adaptation runs, "arm" key
    "PopT_lora": "popt_lora", "PopT_full_ft": "popt_full_ft", "PopT_head_only": "popt_head_only",
    "BrainBERT_lora": "bb_lora", "BrainBERT_full_ft": "bb_full_ft",
    "BrainBERT_head_only": "bb_head_only",
    "Brant_lora": "brant_lora", "Brant_full_ft": "brant_full_ft", "Brant_head_only": "brant_head_only",
}
DECODERS = {"HiLoFuseNet": "hilofusenet", "CNN_LSTM": "cnn_lstm", "LSTM": "lstm"}
PUBLIC_FM_ARMS = {                # experiments/run_ieeg_fm_baselines.py (fm, arm)
    ("brainbert", "single_elec_max"): "bb_single_max",
    ("brainbert", "single_elec_mean"): "bb_single_mean",
    ("brainbert", "pop_meanpool"): "bb_pop_meanpool",
    ("popt", "pop_meanpool"): "popt_frozen",
    ("brant", "single_elec_max"): "brant_single_max",
    ("brant", "single_elec_mean"): "brant_single_mean",
    ("brant", "pop_meanpool"): "brant_pop_meanpool",
}


class Record:
    """Per-subject AUROCs of one arm, task and seed, from one file."""

    def __init__(self, row, task, seed, scores, source, signature=None):
        self.row, self.task, self.seed = row, task, int(seed)
        self.scores = {str(k): float(v) for k, v in scores.items()}
        self.source, self.signature = source, signature


def _split_signature(d):
    """What identifies the scored event set. Seeds of one published row must
    share it: the paper scores one seed-42 event draw under every training seed.

    The event seed when the file records one, plus the test blocks when the file
    stores its splits. A file with neither comes from a runner whose --seed also
    drew the events, so its training seed is its event seed."""
    a = d.get("args") or d.get("config") or {}
    parts = []
    ev = d.get("event_seed", a.get("event_seed"))
    if ev is not None:
        parts.append(f"event_seed={ev}")
    if isinstance(d.get("splits"), list) and d["splits"]:
        keys = [(s.get("subj", s.get("subject", s.get("sid"))), s.get("fold"),
                 s.get("n_test"), s.get("test_time_range")) for s in d["splits"]]
        parts.append("splits=" + hashlib.sha1(
            json.dumps(keys, sort_keys=True, default=str).encode()).hexdigest()[:12])
    if not parts:
        parts.append(f"event_seed={a.get('seed', d.get('seed'))} (drawn with --seed)")
    return ",".join(parts)


def _task(d, a):
    return d.get("endpoint") or a.get("endpoint") or "sentence_onset"


def _variant(row, tags):
    """A non-paper setting gets its own row, e.g. corteg_gate[sharedlora], so it is
    never averaged into a published one. Such rows are listed separately."""
    tags = [t for t in tags if t]
    return f"{row}[{','.join(tags)}]" if tags else row


def _same(got, want):
    if callable(want):
        return bool(want(got))
    if isinstance(want, (bool, str)) or want is None:
        return got == want
    if isinstance(want, (list, tuple)):
        return (isinstance(got, (list, tuple)) and len(got) == len(want)
                and all(_same(g, w) for g, w in zip(got, want)))
    try:
        return abs(float(got) - float(want)) <= 1e-9 * max(1.0, abs(float(want)))
    except (TypeError, ValueError):
        return False


def _short(v):
    if isinstance(v, (list, tuple)):
        return "-".join(_short(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}"
    return os.path.basename(str(v)) if isinstance(v, str) and "/" in v else str(v)


def _recipe_tags(src, recipe):
    """"name=value" for each recorded setting that differs from the paper recipe."""
    tags = []
    for names, want in recipe:
        names = (names,) if isinstance(names, str) else names
        name = next((n for n in names if n in src), None)
        if name is not None and not _same(src[name], want):
            tags.append(f"{names[0]}={_short(src[name])}")
    return tags


def _event_tags(d, a, default_event_seed=42):
    ev = d.get("event_seed", a.get("event_seed", default_event_seed))
    mpc = a.get("max_per_class")
    neg = d.get("neg_mode", a.get("neg_mode"))
    return [f"es{ev}" if ev not in (None, 42) else "",
            f"n{mpc}" if mpc not in (None, 900) else "",
            f"neg_{neg}" if neg not in (None, "upstream") else ""]


def _subject_tag(subjects, train_mode="pooled"):
    """'' for the paper's subjects in the paper's order, else e.g. 'subj1-2', as the
    CORTEG runner names its files. Pooled training depends on the
    order; per-subject training only on the set. None (not recorded) is the paper's."""
    if not isinstance(subjects, (list, tuple)):
        return ""
    subs = [str(s) for s in subjects]
    if train_mode == "per_subject":
        if sorted(subs) == sorted(PAPER_SUBJECT_ORDER):
            return ""
        subs = sorted(subs, key=_subject_order)
    elif subs == PAPER_SUBJECT_ORDER:
        return ""
    return "subj" + "-".join(s.replace("sub_", "") for s in subs)


def _trial_tag(src):
    """A forced --trial (the runners resolve each subject's trial when it is unset)."""
    t = src.get("trial")
    return f"trial={t}" if t not in (None, "") else ""


def _warmup_tag(src):
    """CORTEG warm-up: the paper runs used max(1, epochs // 10) epochs."""
    wu = src.get("warmup_epochs")
    if wu is None:
        return ""
    want = max(1, int(src.get("epochs", 60)) // 10)
    return "" if int(wu) == want else f"warmup_epochs={int(wu)}"


_SUBJECT_LIST_TAG = re.compile(r"^sub_\d+(\+sub_\d+)*$")


def _runner_tags(d, own_tags, key):
    """The non-paper settings a released runner records about itself (`key`:
    'nonpaper_settings' or 'name_tags'), used only when the aggregator's own
    checks found none, so a setting added to a runner later still cannot land in
    a published row. A subject list ('sub_1+sub_2') is dropped: the frozen probes
    and PopT fit each subject on its own, so subsets written side by side belong
    to one cell."""
    rec = d.get(key)
    if any(own_tags) or not isinstance(rec, list):
        return []
    return [str(t) for t in rec if t and not _SUBJECT_LIST_TAG.match(str(t))]


def classify(path, d):
    """Records carried by one JSON file (empty if it is not a result file)."""
    if not isinstance(d, dict):
        return []
    cfg = d.get("config") if isinstance(d.get("config"), dict) else {}
    args = d.get("args") if isinstance(d.get("args"), dict) else {}

    # Paper-run CORTEG (pooled model, per-subject LoRA).
    if d.get("regime") == "pooled" and "per_subject_auroc" in d and cfg:
        merge = cfg.get("merge_strategy") or "layerwise_gate"   # runs before the flag existed
        row = ("corteg_randinit" if cfg.get("random_backbone") else
               "corteg_gate" if merge == "layerwise_gate" else
               "corteg_meanpool" if merge == "average" else f"corteg_{merge}")
        tags = _event_tags(d, cfg) + [
            "" if cfg.get("per_subject_lora", True) else "sharedlora",
            "" if cfg.get("select_metric", "pooled") == "pooled" else "selpersubj",
            "" if cfg.get("loss", "bce") == "bce" else f"loss_{cfg.get('loss')}",
            _subject_tag(d.get("subjects")),
        ] + _recipe_tags(cfg, RECIPE_CORTEG)
        return [Record(_variant(row, tags), _task(d, cfg), d["seed"], d["per_subject_auroc"],
                       path, _split_signature(d))]

    # Scratch decoders (paper runs; their runner is not released). The files are
    # pooled runs over the paper's subjects in the paper's order.
    if d.get("decoder") in DECODERS and "per_subject_auroc" in d and "seed" in d:
        feats = d.get("features") if isinstance(d.get("features"), dict) else {}
        hp = d.get("hp") if isinstance(d.get("hp"), dict) else {}
        tags = _event_tags(d, {**feats, **args}) + [
            "" if d.get("loss", "bce") == "bce" else f"loss_{d.get('loss')}",
        ] + _recipe_tags({**d, **hp, **feats}, RECIPE_DECODER)
        return [Record(_variant(DECODERS[d["decoder"]], tags), _task(d, args), d["seed"],
                       d["per_subject_auroc"], path, _split_signature(d))]

    # Per-subject FM adaptation (head-only / LoRA / full fine-tune).
    if d.get("arm") in ADAPT_ARMS and isinstance(d.get("per_subject"), dict):
        scores = {s: v["mean_auroc"] for s, v in d["per_subject"].items()
                  if isinstance(v, dict) and "mean_auroc" in v}
        recipe = RECIPE_POPT if d["arm"].startswith("PopT") else RECIPE_FOLDS
        tags = _event_tags(d, cfg) + [
            "" if d.get("loss", "bce") == "bce" else f"loss_{d.get('loss')}",
            _trial_tag(cfg),
        ] + _recipe_tags({**d, **cfg}, recipe)
        tags += _runner_tags(d, tags, "name_tags")
        return [Record(_variant(ADAPT_ARMS[d["arm"]], tags), _task(d, cfg), d.get("seed", 42),
                       scores, path, _split_signature(d))]

    # Frozen probes of the paper runs, one subject per file, several arms per file.
    if isinstance(d.get("arms"), dict) and cfg.get("subj"):
        tags = _event_tags(d, cfg) + _recipe_tags(cfg, RECIPE_FOLDS)
        out = []
        for arm, v in d["arms"].items():
            if arm in FROZEN_ARMS and v is not None:
                out.append(Record(_variant(FROZEN_ARMS[arm], tags), _task(d, cfg),
                                  cfg.get("seed", 42), {cfg["subj"]: v}, path, None))
        return out

    # experiments/run_btb_classification.py
    if "cohort_mean_auroc" in d and isinstance(d.get("per_subject"), dict) and \
            "merge_strategy" in args:
        rand = args.get("no_pretrained") or args.get("random_backbone")
        merge = d.get("merge_strategy", args.get("merge_strategy"))
        row = ("corteg_randinit" if rand else
               "corteg_gate" if merge == "layerwise_gate" else
               "corteg_meanpool" if merge == "average" else f"corteg_{merge}")
        mode = d.get("train_mode", args.get("train_mode", "pooled"))
        psl = d.get("per_subject_lora", args.get("per_subject_lora"))
        sel = d.get("select_metric", args.get("select_metric", "pooled"))
        seed = args.get("seed", d.get("seed", 42))
        # No event seed recorded: the runner version whose --seed drew the events.
        tags = ([merge] if rand and merge != "layerwise_gate" else []) \
            + _event_tags(d, args, default_event_seed=seed) + [
                "" if mode == "pooled" else mode,
                # The runner names a --shared_lora run apart in both train modes.
                "sharedlora" if psl is False else "",
                "" if sel == "pooled" else "selpersubj",
                _subject_tag(d.get("subjects", args.get("subjects")), mode),
                _trial_tag(args),
                _warmup_tag(args),
            ] + _recipe_tags(args, RECIPE_CORTEG)
        scores = {s: v["mean"] for s, v in d["per_subject"].items()
                  if isinstance(v, dict) and "mean" in v}
        return [Record(_variant(row, tags), _task(d, args), seed, scores, path,
                       _split_signature(d))]

    # experiments/run_ieeg_fm_baselines.py
    if (d.get("fm"), d.get("arm")) in PUBLIC_FM_ARMS and isinstance(d.get("per_subject"), dict):
        scores = {s: v["auroc"] for s, v in d["per_subject"].items()
                  if isinstance(v, dict) and "auroc" in v}
        tags = _event_tags(d, args) + [_trial_tag(args)] + _recipe_tags(args, RECIPE_FROZEN)
        tags += _runner_tags(d, tags, "name_tags")
        return [Record(_variant(PUBLIC_FM_ARMS[(d["fm"], d["arm"])], tags),
                       _task(d, args), args.get("seed", 42), scores, path, None)]
    return []


def _json_files(cells):
    """Every *.json under one folder or a list of folders, in a stable order."""
    roots = [cells] if isinstance(cells, str) else list(cells)
    return sorted(p for r in roots
                  for p in glob.glob(os.path.join(r, "**", "*.json"), recursive=True))


def load_null(cells):
    """Cohort mean of the oracle permutation null per task, if the files exist."""
    out = {}
    for p in _json_files(cells):
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(d, dict) and "n_perm" in d and isinstance(d.get("per_subject"), dict):
            out[d.get("endpoint", "sentence_onset")] = float(np.mean(
                [v["null_mean"] for v in d["per_subject"].values()]))
    return out


def load_records(cells, verbose=False):
    """Records of every result file under `cells` (a folder or a list of them)."""
    files = _json_files(cells)
    if not files:
        raise SystemExit(f"no JSON files under {cells}")
    recs, skipped = [], []
    for p in files:
        try:
            with open(p, encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError) as e:
            skipped.append(f"{p} (unreadable: {e})")
            continue
        got = classify(p, d)
        if not got:
            skipped.append(p)
        recs.extend(got)
    if verbose:
        for s in skipped:
            print(f"[skip] {s}")
    return recs


def aggregate(records):
    """{(row, task): cell} with the paper's aggregation rule."""
    by = defaultdict(lambda: defaultdict(dict))        # (row, task) -> seed -> subj -> value
    origin = {}
    sigs = defaultdict(dict)                           # (row, task) -> seed -> signature
    for r in records:
        for s, v in r.scores.items():
            key = (r.row, r.task, r.seed, s)
            if key in origin:
                raise SystemExit(
                    f"{r.row} / {r.task} / seed {r.seed} / {s} is in two files:\n"
                    f"  {origin[key]}\n  {r.source}\n"
                    "Point --cells at one set of runs.")
            origin[key] = r.source
            by[(r.row, r.task)][r.seed][s] = v
        if r.signature is not None:
            sigs[(r.row, r.task)].setdefault(r.seed, r.signature)
    cells = {}
    for (row, task), seeds in by.items():
        seed_ids = sorted(seeds, key=lambda sd: (sd != 42, sd))
        subs = sorted(seeds[seed_ids[0]], key=_subject_order)
        warn = []
        for sd in seed_ids[1:]:
            if sorted(seeds[sd], key=_subject_order) != subs:
                warn.append(f"seed {sd} scored {sorted(seeds[sd])}, seed {seed_ids[0]} {subs}")
        subs = [s for s in subs if all(s in seeds[sd] for sd in seed_ids)]
        M = np.array([[seeds[sd][s] for s in subs] for sd in seed_ids], dtype=float)
        per_subject = M.mean(axis=0)
        per_seed = M.mean(axis=1)
        mixed = len(set(sigs[(row, task)].values())) > 1
        if mixed:
            warn.append("seeds were scored on different event sets / splits "
                        f"({sigs[(row, task)]}); the paper scores one seed-42 event set")
        cells[(row, task)] = {
            "row": row, "task": task, "seeds": seed_ids, "subjects": subs,
            "per_subject": dict(zip(subs, per_subject.tolist())),
            "per_seed_subject": {sd: dict(seeds[sd]) for sd in seed_ids},
            "per_seed_mean": dict(zip(seed_ids, per_seed.tolist())),
            "mean": float(per_subject.mean()),
            "sd": float(per_subject.std(ddof=1)) if len(subs) > 1 else float("nan"),
            "seed_sd": float(per_seed.std(ddof=1)) if len(seed_ids) > 1 else None,
            "mixed_events": mixed,
            "warnings": warn,
        }
    return cells


def _subject_order(s):
    try:
        return (0, int(str(s).split("_")[-1]))
    except ValueError:
        return (1, str(s))


def fmt(x, dp):
    return "nan" if x is None or x != x else f"{x:.{dp}f}"


def not_comparable(c):
    """Why a fresh cell cannot be scored against its published value ([] if it can)."""
    want = PAPER_SEEDS.get(c["row"], (42,))
    why = []
    if sorted(c["seeds"]) != sorted(want):
        why.append(f"seeds {','.join(map(str, c['seeds']))}, paper {','.join(map(str, want))}")
    if c["subjects"] != SUBJECTS:
        why.append(f"{len(c['subjects'])} of {len(SUBJECTS)} subjects")
    if c["mixed_events"]:
        why.append("seeds on different event sets")
    return why


# ─────────────────────────── reporting ──────────────────────────────────────
class Checker:
    """Collects comparisons with the paper. tol=None means exact (string) match,
    used for paper_cells; otherwise a tolerance, used for fresh runs."""

    def __init__(self, tol=None):
        self.tol = tol
        self.n = 0              # published cells looked at
        self.compared = 0       # ... compared with the paper (fresh: scored)
        self.bad, self.missing, self.unreleased, self.unscored = [], [], [], []

    def check(self, where, got, want, dp, tol=None, scored=True, released=True):
        """Returns the mark printed after a value. `tol` overrides the cohort
        tolerance (per-subject cells) unless the check is exact. A fresh cell
        that is not `scored` is shown but kept out of the exit status."""
        if want is None:
            return ""
        self.n += 1
        if got is None:                      # shown as "--"; counted, not flagged inline
            if released or self.tol is None:
                self.missing.append(f"{where}: missing (paper {want})")
            else:
                self.unreleased.append(where)
            return ""
        if self.tol is not None and not scored:
            self.unscored.append(f"{where}: {fmt(got, dp)} (paper {want})")
            return ""
        self.compared += 1
        if self.tol is None:
            ok = fmt(got, dp) == want
        else:
            t = self.tol if tol is None else tol
            ok = abs(got - float(want)) <= t + 0.5 * 10 ** -dp
        if not ok:
            self.bad.append(f"{where}: {fmt(got, dp)} vs paper {want}"
                            + ("" if self.tol is None else f" (|d|={abs(got - float(want)):.4f})"))
            return f"  <-- paper {want}"
        return ""


def _not_scored_mark(why, keys):
    reasons = sorted({w for k in keys for w in why.get(k, [])})
    return f"  [not scored: {'; '.join(reasons)}]" if reasons else ""


def print_table20(cells, ck, out, why):
    fresh = ck.tol is not None
    out("\nTable 20 - BrainTreebank, every arm. Mean AUROC +- cross-subject SD "
        "(n=10), 4 dp; seed SD = SD of the per-seed cohort means"
        + (" (shown, not scored, for fresh runs)." if fresh else "."))
    out(f"{'arm':<40}{'Task A':>17}{'Task B':>17}   {'seed SD A / B':<16}{'seeds'}")
    for key, label, group in TABLE20:
        if group:
            out(f"  [{group}]")
        exp = EXPECTED_T20[key]
        released = key not in NOT_RELEASED
        parts, marks = [], []
        for i, task in enumerate(TASKS):
            c = cells.get((key, task))
            where = f"T20 {label} Task {TASK_LABEL[task]}"
            if c is None:
                parts.append(f"{'--':>17}")
                marks.append(ck.check(where, None, "/".join(exp[i]), 4, released=released))
                continue
            ok = not why.get((key, task))
            m1 = ck.check(where + " mean", c["mean"], exp[i][0], 4, scored=ok)
            m2 = ck.check(where + " SD", c["sd"], exp[i][1], 4, scored=ok)
            parts.append(f"{fmt(c['mean'], 4) + '+-' + fmt(c['sd'], 4):>17}")
            marks += [m1, m2]
        ssd = []
        for i, task in enumerate(TASKS):
            c = cells.get((key, task))
            want = exp[2][i] if exp[2] else None
            if c is None or c["seed_sd"] is None:
                ssd.append("--")
                if want is not None and c is not None and not fresh:
                    marks.append(ck.check(f"T20 {label} seed SD {TASK_LABEL[task]}", None, want, 4))
                continue
            ssd.append(fmt(c["seed_sd"], 4))
            if want and not fresh:
                marks.append(ck.check(f"T20 {label} seed SD {TASK_LABEL[task]}",
                                      c["seed_sd"], want, 4))
        seeds = "/".join(",".join(map(str, cells[(key, t)]["seeds"])) if (key, t) in cells
                         else "-" for t in TASKS)
        tail = "".join(m for m in marks if m) + _not_scored_mark(why, [(key, t) for t in TASKS])
        if fresh and not released and not any((key, t) in cells for t in TASKS):
            tail += "  (runner not released)"
        out(f"{label:<40}{''.join(parts)}   {' / '.join(ssd):<16}{seeds}{tail}")


def print_table3_9(cells, ck, out, why, tol_subject=None):
    out("\nTable 3 - BrainTreebank, each method's best configuration. "
        "Mean AUROC +- cross-subject SD, 3 dp (= Table 9 Mean column).")
    out(f"{'method':<24}{'Task A':>15}{'Task B':>15}")
    for key, label in TABLE3:
        row, marks = [], []
        for i, task in enumerate(TASKS):
            c = cells.get((key, task))
            want_m, want_s = EXPECTED_T3[key][i]
            where = f"T3 {label} Task {TASK_LABEL[task]}"
            if c is None:
                row.append(f"{'--':>15}")
                marks.append(ck.check(where, None, f"{want_m}+-{want_s}", 3,
                                      released=key not in NOT_RELEASED))
                continue
            ok = not why.get((key, task))
            marks.append(ck.check(where + " mean", c["mean"], want_m, 3, scored=ok))
            marks.append(ck.check(where + " SD", c["sd"], want_s, 3, scored=ok))
            row.append(f"{fmt(c['mean'], 3) + '+-' + fmt(c['sd'], 3):>15}")
        out(f"{label:<24}{''.join(row)}" + "".join(m for m in marks if m)
            + _not_scored_mark(why, [(key, t) for t in TASKS]))

    for task in TASKS:
        out(f"\nTable 9 - per-subject AUROC, Task {TASK_LABEL[task]} ({task}); "
            "3-seed rows are averaged within subject first.")
        out(f"{'method':<24}" + "".join(f"{'S' + str(i):>7}" for i in range(1, 11))
            + f"{'Mean':>15}")
        for key, label in TABLE3:
            c = cells.get((key, task))
            want = EXPECTED_T9[(key, task)].split()
            if c is None:
                out(f"{label:<24}  --")
                ck.check(f"T9 {label} Task {TASK_LABEL[task]}", None, " ".join(want), 3,
                         released=key not in NOT_RELEASED)
                continue
            ok = not why.get((key, task))
            vals, marks = [], []
            for j, s in enumerate(SUBJECTS):
                v = c["per_subject"].get(s)
                vals.append(f"{fmt(v, 3) if v is not None else '--':>7}")
                m = ck.check(f"T9 {label} Task {TASK_LABEL[task]} S{j + 1}", v, want[j], 3,
                             tol=tol_subject, scored=ok)
                if m:
                    marks.append(f"S{j + 1} {m.strip().replace('<-- ', '')}")
            mean = f"{fmt(c['mean'], 3)}+-{fmt(c['sd'], 3)}"
            out(f"{label:<24}{''.join(vals)}{mean:>15}"
                + ("  <-- " + "; ".join(marks) if marks else "")
                + _not_scored_mark(why, [(key, task)]))


def _seed_diff(got, ref):
    """(subjects, cohort-mean d, mean |d|, max |d|) of two per-subject dicts,
    over the reference's subjects that `got` also has."""
    common = [s for s in sorted(ref, key=_subject_order) if s in got]
    if not common:
        return common, 0.0, 0.0, 0.0
    d = np.array([got[s] - ref[s] for s in common])
    dm = float(np.mean([got[s] for s in common]) - np.mean([ref[s] for s in common]))
    return common, dm, float(np.mean(np.abs(d))), float(np.max(np.abs(d)))


def seed_spread(ref_cell):
    """How far the reference row's own seeds lie from each other: the largest
    cohort |d|, mean |d| and max |d| over every pair of its seeds (None for a
    one-seed row). A fresh seed that fails the seed-matched check but is no
    further from its reference seed than this looks like another seed."""
    seeds = ref_cell["seeds"]
    if len(seeds) < 2:
        return None
    worst = [0.0, 0.0, 0.0]
    for a, b in combinations(seeds, 2):
        _, dm, mad, mx = _seed_diff(ref_cell["per_seed_subject"][a],
                                    ref_cell["per_seed_subject"][b])
        worst = [max(worst[0], abs(dm)), max(worst[1], mad), max(worst[2], mx)]
    return tuple(worst)


def seed_matched(cells, ref_cells, tol, tol_subject, tol_subject_max, out):
    """Fresh vs reference, seed by seed.

    Returns (failures, number of seeds compared, number of failures that are no
    further from the reference than the reference's own seeds are from each other).

    Three numbers per (arm, task, seed): the cohort-mean difference, the mean
    absolute per-subject difference, and the largest per-subject difference,
    each against its own floor (see --tol, --tol_subject, --tol_subject_max).
    A seed that lacks some of the reference's subjects is shown as partial and
    not scored. The check presumes the fresh run consumes the random stream as
    the reference run did for that seed; for a multi-seed row the reference's
    own seed-to-seed spread is printed beside it, so a failure that merely
    looks like another seed can be told from one that looks like another recipe."""
    bad, n, like_seed = [], 0, 0
    out(f"\nSeed-matched comparison with the reference cells (tolerances: cohort "
        f"{tol}, mean |d| per subject {tol_subject}, any subject {tol_subject_max}):")
    any_ = False
    for (row, task), c in sorted(cells.items()):
        r = ref_cells.get((row, task))
        if r is None or not any(sd in r["seeds"] for sd in c["seeds"]):
            continue
        spread = seed_spread(r)
        if spread is not None:
            inside = (spread[0] <= tol and spread[1] <= tol_subject
                      and spread[2] <= tol_subject_max)
            out(f"  {row:<20} Task {TASK_LABEL[task]}: the reference seeds "
                f"{'/'.join(map(str, r['seeds']))} differ from each other by up to cohort "
                f"{spread[0]:.4f}, mean|d| {spread[1]:.4f}, any subject {spread[2]:.4f}"
                + ("  (inside the tolerances: this check cannot tell another seed "
                   "from the same one)" if inside else ""))
        for sd in c["seeds"]:
            if sd not in r["seeds"]:
                continue
            got, ref = c["per_seed_subject"][sd], r["per_seed_subject"][sd]
            common, dm, mad, mx = _seed_diff(got, ref)
            if not common:
                continue
            any_ = True
            d = np.array([got[s] - ref[s] for s in common])
            worst = common[int(np.argmax(np.abs(d)))]
            line = (f"  {row:<20} Task {TASK_LABEL[task]} seed {sd:<3} n={len(common):<3} "
                    f"cohort d={dm:+.4f}  mean|d|={mad:.4f}  "
                    f"worst {worst} d={d[common.index(worst)]:+.4f}")
            if len(common) < len(ref):
                out(line + f"  (partial: {len(common)} of {len(ref)} subjects; not scored)")
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
            bad.append(f"{row} Task {TASK_LABEL[task]} seed {sd}: seed-matched "
                       f"cohort d={dm:+.4f}, mean|d|={mad:.4f}, max|d|={mx:.4f}"
                       + (" (within the reference's seed-to-seed spread)" if seedlike else ""))
    if not any_:
        out("  (no seed in common with the reference)")
    return bad, n, like_seed


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cells", nargs="+", default=[DEFAULT_CELLS],
                   help="one or more folders of result JSONs (default: paper_cells/btb)")
    p.add_argument("--reference", default=DEFAULT_CELLS,
                   help="per-seed reference for the seed-matched check of fresh runs "
                        "(default: paper_cells/btb)")
    p.add_argument("--tol", type=float, default=None,
                   help=f"fresh runs: allowed |d| of a cohort value, default {TOL} "
                        f"(1.5 x the run-to-run floor). paper_cells are checked exactly")
    p.add_argument("--tol_subject", type=float, default=TOL_SUBJECT,
                   help="seed-matched check: allowed mean |d| over subjects")
    p.add_argument("--tol_subject_max", type=float, default=TOL_SUBJECT_MAX,
                   help="allowed |d| of any one subject (seed-matched check and Table 9)")
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
    tol = TOL if fresh and a.tol is None else a.tol

    def out(line):
        print(line.rstrip())

    cells = aggregate(load_records(a.cells, a.verbose))
    out(f"cells: {where_cells}  ({len(cells)} arm x task cells)")
    why = {}
    for c in cells.values():
        where = f"{c['row']} Task {TASK_LABEL.get(c['task'], c['task'])}"
        for w in c["warnings"]:
            out(f"WARNING {where}: {w}")
        if c["row"] not in EXPECTED_T20:
            continue
        if sorted(c["seeds"]) != sorted(PAPER_SEEDS.get(c["row"], (42,))):
            out(f"NOTE {where}: seeds {c['seeds']}; the paper row uses seeds "
                f"{list(PAPER_SEEDS.get(c['row'], (42,)))}"
                + (": its published cells are shown, not scored; the seed-matched "
                   "comparison checks each seed" if fresh else ""))
        if c["subjects"] != SUBJECTS:
            out(f"NOTE {where}: {len(c['subjects'])} subjects; the paper cohort has 10")
        if fresh:
            why[(c["row"], c["task"])] = not_comparable(c)

    ck = Checker(tol)
    print_table20(cells, ck, out, why)
    print_table3_9(cells, ck, out, why, a.tol_subject_max)
    null = load_null(a.cells)
    if null or not fresh:
        # The paper quotes the null once, as "≈0.53": both cohort means at 2 dp.
        marks = [] if fresh else [
            ck.check(f"oracle null Task {TASK_LABEL[t]}", null.get(t), EXPECTED_NULL, 2)
            for t in TASKS]
        out("\nOracle permutation null (max over electrodes of a label-permuted 1-D "
            "high-gamma probe; a lower bound for the fitted probes), cohort mean: "
            + ", ".join(f"Task {TASK_LABEL[t]} {null[t]:.4f}" for t in TASKS if t in null)
            + f"  (paper: ≈{EXPECTED_NULL}, Table 9 caption; both at 2 dp)" + "".join(marks))
    other = sorted(k for k in cells if k[0] not in EXPECTED_T20)
    if other:
        out("\nOther cells (settings outside the paper tables; not compared):")
        for k in other:
            c = cells[k]
            out(f"  {c['row']:<40} Task {TASK_LABEL.get(c['task'], c['task'])}  "
                f"{fmt(c['mean'], 4)}+-{fmt(c['sd'], 4)}  n={len(c['subjects'])}  "
                f"seeds {c['seeds']}")

    if not fresh:
        failed = ck.bad + ck.missing
        out(f"\n{ck.n} cells checked against the paper: "
            + ("all match." if not failed else f"{len(failed)} MISMATCH:"))
        for b in failed:
            out(f"  {b}")
        return 1 if failed else 0

    sm_bad, n_sm, n_seedlike = seed_matched(cells, aggregate(load_records(a.reference)), tol,
                                            a.tol_subject, a.tol_subject_max, out)
    events = [f"{c['row']} Task {TASK_LABEL.get(c['task'], c['task'])}: seeds "
              f"{c['seeds']} were scored on different event sets"
              for c in cells.values() if c["row"] in EXPECTED_T20 and c["mixed_events"]]
    out(f"\n{ck.compared} of {ck.n} published cells compared with the paper (tol {tol}): "
        f"{len(ck.bad)} outside tolerance.")
    if ck.unscored:
        out(f"{len(ck.unscored)} shown but not scored: the run's seeds or subjects differ "
            "from the paper row's (see the NOTE lines and the seed-matched comparison).")
    if ck.missing:
        out(f"{len(ck.missing)} not in {where_cells}.")
    if ck.unreleased:
        out(f"{len(ck.unreleased)} from runners not in this release (raw spectral probes; "
            "from-scratch decoders; BrainBERT and Brant head-only / LoRA / full fine-tune).")
    out(f"{n_sm} seeds compared with the same seed in {a.reference}: "
        f"{len(sm_bad)} outside tolerance.")
    if n_seedlike:
        verb = "is" if n_seedlike == 1 else "are"
        out(f"{n_seedlike} of them {verb} no further from their reference seed than "
            "the reference's own seeds are from each other: what a run that draws its "
            "random numbers in another order would give. The seed-matched check "
            "presumes the random stream of the paper run.")
    failed = ck.bad + events + sm_bad
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
