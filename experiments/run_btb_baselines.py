"""From-scratch baselines on BrainTreebank: HiLoFuseNet, CNN-LSTM and LSTM.

These are the "trained from scratch" rows of the BrainTreebank tables: the
HiLoFuseNet row of Table 3, its per-subject rows in Table 9, and all three
decoders in Table 20. The model classes are ``models/baselines.py``.

Inputs are CORTEG's own features, obtained through
``run_btb_classification.extract_subject``: the same MNI electrode set, the same
[t, t+1.5] s window on both endpoints, and the same cache. LOW (128 Hz, 192
samples) is resampled onto the HIGH grid (200 Hz, 300 samples) and the two are
stacked as (N, C, 300, 2) in the order [HIGH, LOW].

The protocol is CORTEG's BrainTreebank protocol:

* **Pooled.** One model per fold is trained on every subject's fit block and
  scored on each subject's test block separately.
* **Causal folds.** ``forward_chaining_split`` with 4 folds, the shared 7 s
  embargo and a causal validation block (``val_frac`` 0.15) cut from the tail
  of each subject's history. Every fold's split report must show zero
  overlapping train/test pairs.
* **z-score** per channel and per stream, fitted on the fit block only.
* **Loss and optimiser.** BCE-with-logits and AdamW (lr 1e-3, weight decay
  1e-4), batch 64, at most 40 epochs.
* **Early stopping** with patience 10 on the mean of the per-subject
  validation AUROCs. Test is scored once, at the best-validation state.
* **Aggregation.** A subject's score is its mean AUROC over folds; the cohort
  score is the mean over subjects.

Two properties of the paper runs are kept as defaults, each with a flag for
the alternative:

* **Channels are not aligned across subjects.** These decoders take no
  electrode coordinates and have no subject-specific parameters. In the pooled
  model, subjects with fewer electrodes are zero-padded to the widest (191,
  sub_7), and input channel k is simply each subject's k-th electrode. The
  paper does not say that these baselines were pooled; they were.
  ``--train_mode per_subject`` trains one model per subject and fold instead,
  which needs no alignment. It was not run for the paper.
* **Task B negatives** are upstream's word-free tiles (``--neg_mode upstream``).
  Many of them fall in long silences such as the credits, so part of Task B
  can be solved by finding the quiet stretches of the film.
  ``--neg_mode short_silence`` draws negatives only from pauses shorter than
  10 s, the control the CORTEG runner also offers. It was not run for these
  baselines.

A run at any setting other than the paper's gets a tag in its result file name
for each departure (see ``setting_tags``), so a variant or smoke run never
overwrites a paper-setting file or is averaged with one.

Seeds. The paper averages ``--seed`` 42, 1 and 2, and all three runs score
the SAME events, drawn with seed 42. ``--seed`` therefore sets only the weight
initialisation and the batch order; ``--event_seed`` (default 42) picks the
events. Letting the training seed redraw the events would score a different
event set per seed, which is not what the published numbers average.

The published cells are the 3-seed mean per subject, then the mean and
cross-subject SD (ddof=1) over the ten subjects. ``--summarize`` computes them
from the three result files, together with the cross-seed SD of the cohort
mean. On CPU this runner reproduces the code behind the paper exactly: the
same seed, subjects and settings give the same AUROCs to the last digit. GPU
training is not bit-deterministic, so a GPU rerun agrees with the published
cells only to within run-to-run noise. Compare with ``scripts/aggregate_btb.py``,
whose default tolerances are 1.5 x the run-to-run floor measured on the Task B
gate arm at seed 42. The cross-seed SD is not a tolerance: several are below
that floor (Task B: 0.0006 to 0.0032).

Example:

    python -m experiments.run_btb_baselines --decoder HiLoFuseNet \\
        --endpoint sentence_onset --seed 42 --save_root $ROOT/base_HiLoFuseNet
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from data.braintreebank import (
    EMBARGO_SEC,
    HI_FS,
    assert_valid_times,
    btb_root,
    forward_chaining_split,
    load_events,
    split_report,
    word_nonword_events,
)

DECODERS = ("HiLoFuseNet", "CNN_LSTM", "LSTM")
ENDPOINTS = ("sentence_onset", "word_nonword")

# Subject order of the paper runs. It is not cosmetic: it sets the order in
# which the pooled training set is concatenated, and so the batches a given
# seed draws.
SUBS10 = ["sub_1", "sub_2", "sub_3", "sub_4", "sub_6", "sub_7", "sub_10",
          "sub_5", "sub_8", "sub_9"]

# Task B negative pools; "upstream" is the paper's (see the module docstring).
NEG_MODES = ("upstream", "short_silence")
TRAIN_MODES = ("pooled", "per_subject")

# The HIGH-frequency band of the cached features. It is part of the CORTEG
# cache key, so it is fixed here rather than exposed: another band would build
# a second cache and a different input.
HGA_BAND = (70.0, 200.0)

# The settings of the paper runs. The parser defaults are read from here, so
# the recipe is stated once. A run that departs from one of them carries the
# tag below in its file name (see setting_tags); the order is the tag order.
PAPER = {"max_per_class": 900, "n_folds": 4, "val_frac": 0.15, "win_sec": 1.5,
         "pre_sec": 0.0, "epochs": 40, "patience": 10, "lr": 1e-3,
         "weight_decay": 1e-4, "batch_size": 64, "hidden": 256, "dropout": 0.5,
         "D": 16}
TAG = {"max_per_class": "n", "n_folds": "f", "val_frac": "val", "win_sec": "win",
       "pre_sec": "pre", "epochs": "ep", "patience": "pat", "lr": "lr",
       "weight_decay": "wd", "batch_size": "bs", "hidden": "h", "dropout": "do",
       "D": "D"}
HP = ("lr", "hidden", "dropout", "D", "weight_decay", "epochs", "patience", "batch_size")
PAPER_SEEDS = [42, 1, 2]
PAPER_EVENT_SEED = 42

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ─────────────────────────────── features ───────────────────────────────────
def feature_args(a) -> argparse.Namespace:
    """Arguments for ``run_btb_classification.extract_subject``.

    The extractor draws events and keys its cache on ``event_seed``. ``seed``
    is set to the event seed too, so the training ``--seed`` cannot reach the
    event draw by any path.
    """
    return argparse.Namespace(
        trial=None, win_sec=a.win_sec, pre_sec=a.pre_sec,
        hga_low=HGA_BAND[0], hga_high=HGA_BAND[1],
        max_per_class=a.max_per_class, no_cache=a.no_cache,
        endpoint=a.endpoint, neg_mode=a.neg_mode,
        event_seed=a.event_seed, seed=a.event_seed)


def expected_events(subj: str, a):
    """(event_times, labels) that ``--endpoint`` draws for `subj`, unmasked.

    Reads only the transcript and the timings file, so it is cheap. Positions
    are what the CORTEG path stores as ``event_times``: the onset for Task A,
    and for Task B the window centre ``word_nonword_events`` returns, at which
    the [t, t+1.5] window is anchored.
    """
    from experiments.run_btb_classification import movie_of, trial_of
    root = btb_root()
    trial = trial_of(root, subj)
    movie = movie_of(root, subj, trial)
    if a.endpoint == "word_nonword":
        t, y, _ = word_nonword_events(root, subj, trial, movie, a.max_per_class,
                                      a.event_seed, neg_mode=a.neg_mode)
    else:
        t, y = load_events(root, movie, a.max_per_class, a.event_seed)
    return np.asarray(t, dtype=np.float64), np.asarray(y)


def check_events(subj: str, ev, y, want_t, want_y, what: str) -> None:
    """Refuse features whose events are not the ones this run asked for.

    The window-bounds mask may drop events, so the scored events must be a
    subset of the drawn ones, with the same labels. Anything else means the
    features belong to another endpoint or event seed -- a stale cache, or an
    extractor that ignored the endpoint -- and every number would be for the
    wrong task while the results file claimed the right one.
    """
    ev = np.asarray(ev, dtype=np.float64).tolist()
    y = np.asarray(y).astype(int).tolist()
    drawn = set(np.asarray(want_t, dtype=np.float64).tolist())
    missing = sum(t not in drawn for t in ev)
    if missing:
        raise SystemExit(
            f"{subj}: {missing}/{len(ev)} feature events are not {what} events. "
            "The features are for another endpoint or event seed: a stale cache, "
            "or an extractor that does not honour --endpoint/--event_seed. "
            "Rebuild with --no_cache.")
    pairs = set(zip(np.asarray(want_t, dtype=np.float64).tolist(),
                    np.asarray(want_y).astype(int).tolist()))
    wrong = sum((t, l) not in pairs for t, l in zip(ev, y))
    if wrong:
        raise SystemExit(f"{subj}: {wrong}/{len(ev)} feature labels disagree with "
                         f"the {what} labels")


def stack_hi_lo(x_lo: np.ndarray, x_hi: np.ndarray) -> np.ndarray:
    """(N, C, T_hi, 2) as [HIGH, LOW], LOW resampled onto HIGH's time grid.

    T is read from x_hi, never hardcoded: a literal length would silently
    resample every subject onto a window no cache holds.
    """
    from scipy.signal import resample_poly
    t_out = x_hi.shape[-1]
    lo = x_lo
    if lo.shape[-1] != t_out:                        # LOW 128 Hz -> HIGH's 200 Hz grid
        lo = resample_poly(lo.astype(np.float64), t_out, lo.shape[-1],
                           axis=-1).astype(np.float32)
    return np.stack([x_hi.astype(np.float32), lo], axis=-1)


def load_subject(subj: str, a):
    """(X (N, C, T, 2) [HIGH, LOW], y (N,) float32, event_times (N,))."""
    from experiments.run_btb_classification import extract_subject
    x_lo, x_hi, y, _xyz, ev = extract_subject(subj, feature_args(a))
    if not a.skip_event_check:
        want_t, want_y = expected_events(subj, a)
        check_events(subj, ev, y, want_t, want_y,
                     f"{a.endpoint} (event_seed={a.event_seed})")
    X = stack_hi_lo(x_lo, x_hi)
    return X, np.asarray(y).astype(np.float32), np.asarray(ev, dtype=np.float64)


# ─────────────────────────────── training ───────────────────────────────────
def zscore_fit_apply(Xfit, *rest):
    """Per-channel, per-stream z-score over (N, T), fitted on the fit block.

    A channel whose SD is below 1e-6 is divided by 1, not by a tiny number.
    """
    mu = Xfit.mean(axis=(0, 2), keepdims=True)
    sd = Xfit.std(axis=(0, 2), keepdims=True)
    sd[sd < 1e-6] = 1.0
    return [((x - mu) / sd).astype(np.float32) for x in (Xfit,) + rest]


def build(dec: str, C: int, F: int, dev, hidden: int = 256, dropout: float = 0.5,
          D: int = 16) -> nn.Module:
    """The decoder at the paper's settings. CNN-LSTM has no `hidden` knob."""
    from models.baselines import CNN_LSTM, LSTM, HiLoFuseNet
    if dec == "LSTM":
        return LSTM(input_size=C * F, hidden_size=hidden, output_size=1,
                    dropout_prob=dropout).to(dev)
    if dec == "CNN_LSTM":
        return CNN_LSTM(input_size=C, output_size=1, dropout_prob=dropout).to(dev)
    if dec == "HiLoFuseNet":
        return HiLoFuseNet(C=C, F=F, lstm_hidden=hidden, D=D, output_size=1,
                           dropout_prob=dropout).to(dev)
    raise ValueError(f"unknown decoder {dec!r}")


def make_folds(subs, data, times, n_folds: int, val_frac: float):
    """Causal folds per subject, and the split report of every fold.

    The footprint is measured from the features (T samples at 200 Hz), never
    taken from a literal. It only feeds the overlap assertions: fold membership
    depends on the event times and the shared embargo alone, so these folds are
    the ones every other arm on the same events receives.
    """
    if not val_frac > 0:
        raise ValueError("early stopping needs a validation block: val_frac must be > 0")
    footprint = float(data[subs[0]][0].shape[2] / HI_FS)
    if not all(abs(data[s][0].shape[2] / HI_FS - footprint) < 1e-9 for s in subs):
        raise RuntimeError("subjects disagree on window length; refusing to share one split")
    if footprint > EMBARGO_SEC:
        raise RuntimeError(f"window {footprint}s exceeds embargo {EMBARGO_SEC}s")
    folds, reports = {}, []
    for s in subs:
        tv = assert_valid_times(times[s], n_expected=len(data[s][1]))
        f = forward_chaining_split(tv, win_sec=footprint, n_folds=n_folds,
                                   embargo_sec=EMBARGO_SEC, val_frac=val_frac)
        if len(f) != n_folds:
            raise RuntimeError(f"{s}: {len(f)} causal folds, wanted {n_folds}")
        for fit, va, te in f:
            r = split_report(tv, footprint, fit, te, va, scheme="forward_chaining")
            if r["overlapping_train_test_pairs"] or not r["causal"]:
                raise AssertionError(f"{s}: split leaks -- {r}")
            reports.append({"subj": s, **r})
        folds[s] = f
    return folds, reports, footprint


def run_pooled(subs, data, folds, a, dev, log=print):
    """Train one model per fold on every subject; score each subject separately.

    Returns (per-subject list of fold AUROCs, per-fold training record).
    """
    from sklearn.metrics import roc_auc_score

    maxC = max(data[s][0].shape[1] for s in subs)
    F = data[subs[0]][0].shape[-1]

    def pad(X):
        if X.shape[1] == maxC:
            return X
        z = np.zeros((X.shape[0], maxC - X.shape[1]) + X.shape[2:], dtype=X.dtype)
        return np.concatenate([X, z], axis=1)

    per_fold = {s: [] for s in subs}
    record = []
    t0 = time.time()
    for f in range(a.n_folds):
        Xtr_l, ytr_l, va, te = [], [], {}, {}
        for s in subs:
            fit_i, va_i, te_i = folds[s][f]
            X, y = data[s]
            Xfit, Xva, Xte = zscore_fit_apply(X[fit_i], X[va_i], X[te_i])
            Xtr_l.append(pad(Xfit)); ytr_l.append(y[fit_i])
            va[s] = (pad(Xva), y[va_i]); te[s] = (pad(Xte), y[te_i])
        Xtr = np.concatenate(Xtr_l); ytr = np.concatenate(ytr_l)
        del Xtr_l, ytr_l
        model = build(a.decoder, maxC, F, dev, a.hidden, a.dropout, a.D)
        opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
        dl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
                        batch_size=a.batch_size, shuffle=True)
        best, bad, bsd, best_ep, ep = -9.0, 0, None, 0, 0
        for ep in range(1, a.epochs + 1):
            model.train()
            for xb, yb in dl:
                opt.zero_grad()
                # reshape, not squeeze: the decoders already return (B,), and a
                # second squeeze turns a last batch of one window into a 0-d
                # tensor that the loss rejects. For B >= 2 both are the same.
                out = model(xb.to(dev)).reshape(-1)
                loss = nn.functional.binary_cross_entropy_with_logits(out, yb.to(dev))
                loss.backward(); opt.step()
            # Early stopping reads VALIDATION only. Selecting the checkpoint on
            # the test block and then reporting that block inflates AUROC even
            # under a pure-noise null.
            model.eval(); aucs = []
            with torch.no_grad():
                for s in subs:
                    Xva, yva = va[s]
                    p = model(torch.from_numpy(Xva).to(dev)).reshape(-1).cpu().numpy()
                    if len(np.unique(yva)) > 1:
                        aucs.append(roc_auc_score(yva, p))
            m = float(np.mean(aucs)) if aucs else float("nan")
            if m > best:
                best, bad, best_ep = m, 0, ep
                bsd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= a.patience:
                break
        if bsd is None:
            raise RuntimeError(f"fold {f}: no epoch produced a finite validation AUROC")
        model.load_state_dict(bsd); model.eval()
        with torch.no_grad():
            for s in subs:
                Xte, yte = te[s]
                p = model(torch.from_numpy(Xte).to(dev)).reshape(-1).cpu().numpy()
                per_fold[s].append(float(roc_auc_score(yte, p))
                                   if len(np.unique(yte)) > 1 else float("nan"))
        record.append({"fold": f, "best_val_auroc": best, "best_epoch": best_ep,
                       "epochs_run": ep, "n_fit": int(len(ytr))})
        log(f"  fold {f+1}/{a.n_folds} best_VAL_auroc={best:.4f} at epoch {best_ep}"
            f"/{ep}  ({time.time()-t0:.0f}s)")
        del model, opt, dl, Xtr, ytr, va, te
    return per_fold, record


def train(subs, data, folds, a, dev, log=print):
    """``run_pooled`` over all subjects, or once per subject (``--train_mode``).

    In per-subject mode each subject gets its own model per fold, as wide as its
    own electrode set, early-stopped on its own validation AUROC. The seed is
    reset before each subject, so a subject's score does not depend on which
    subjects ran before it.
    """
    if a.train_mode == "pooled":
        return run_pooled(subs, data, folds, a, dev, log)
    per_fold, record = {}, []
    for s in subs:
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        log(f"  [{s}] per-subject training")
        pf, rec = run_pooled([s], data, folds, a, dev, log)
        per_fold[s] = pf[s]
        record += [{"subj": s, **r} for r in rec]
    return per_fold, record


# ─────────────────────────────── naming ─────────────────────────────────────
def _same(x, y) -> bool:
    """Equal settings; numbers compare by value, so 1e-3 == 0.001 and 4 == 4.0."""
    num = (int, float)
    if isinstance(x, num) and isinstance(y, num) and not isinstance(x, bool) \
            and not isinstance(y, bool):
        return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(y)))
    return x == y


def _subject_tag(a) -> str:
    """'' for the paper's subjects in the paper's order, else e.g. 'subj1-2'.

    Pooled training concatenates subjects in the given order, so the order is
    part of the run; per-subject training resets the seed per subject, so only
    the set is.
    """
    subs = list(a.subjects)
    if a.train_mode == "per_subject":
        if sorted(subs) == sorted(SUBS10):
            return ""
        subs = sorted(subs, key=lambda s: (len(s), s))
    elif subs == SUBS10:
        return ""
    return "subj" + "-".join(s.replace("sub_", "") for s in subs)


def setting_tags(a) -> list:
    """One short tag per setting in which this run departs from the paper runs.

    Empty at the paper's settings. Every setting that changes the estimand or
    the training recipe is covered: Task B negatives, subjects, event count,
    folds, validation fraction, window and every hyperparameter. The training
    seed and the event seed have their own fields in the file name.
    """
    tags = []
    if a.endpoint == "word_nonword" and a.neg_mode != "upstream":
        tags.append(a.neg_mode)
    sub = _subject_tag(a)
    if sub:
        tags.append(sub)
    for k, t in TAG.items():
        v = getattr(a, k)
        if not _same(v, PAPER[k]):
            tags.append(f"{t}{v:g}" if isinstance(v, float) else f"{t}{v}")
    return tags


def _suffix(a) -> str:
    ev = "" if int(a.event_seed) == PAPER_EVENT_SEED else f"_ev{a.event_seed}"
    return ev + "".join(f"_{t}" for t in setting_tags(a))


def result_path(save_root: str, a, seed=None) -> str:
    """Where a run writes: ``results_<train_mode>_<endpoint>_s<seed>[_ev<E>][_<tag>...].json``.

    The endpoint is in the name because both endpoints share a save_root. At the
    paper's settings the name is ``results_pooled_<endpoint>_s<seed>.json``, the
    name of the paper-run files in ``paper_cells/btb``.
    """
    seed = a.seed if seed is None else seed
    return os.path.join(save_root,
                        f"results_{a.train_mode}_{a.endpoint}_s{seed}{_suffix(a)}.json")


def summary_path(save_root: str, a) -> str:
    """``summary_<train_mode>_<endpoint>[_ev<E>][_<tag>...][_seeds<...>].json``."""
    seeds = "" if [int(s) for s in a.seeds] == PAPER_SEEDS else \
        "_seeds" + "-".join(str(s) for s in a.seeds)
    return os.path.join(save_root,
                        f"summary_{a.train_mode}_{a.endpoint}{_suffix(a)}{seeds}.json")


# ─────────────────────────────── summary ────────────────────────────────────
def recorded_settings(r: dict) -> dict:
    """The settings a result file records, under the parser's names.

    The paper-run files record fewer fields than this runner writes (no
    patience, val_frac or feature block). A field a file does not record is
    not compared.
    """
    hp = r.get("hp") if isinstance(r.get("hp"), dict) else {}
    ft = r.get("features") if isinstance(r.get("features"), dict) else {}
    got = {"decoder": r.get("decoder"), "endpoint": r.get("endpoint"),
           "event_seed": r.get("event_seed"), "train_mode": r.get("train_mode"),
           "neg_mode": ft.get("neg_mode"), "n_folds": r.get("folds"),
           "val_frac": r.get("val_frac"), "win_sec": ft.get("win_sec"),
           "pre_sec": ft.get("pre_sec"), "max_per_class": ft.get("max_per_class"),
           **{k: hp.get(k) for k in HP}}
    return {k: v for k, v in got.items() if v is not None}


def requested_settings(a) -> dict:
    """The settings `a` asks for, under the names ``recorded_settings`` uses."""
    want = {"endpoint": a.endpoint, "event_seed": a.event_seed,
            "train_mode": a.train_mode, **{k: getattr(a, k) for k in PAPER}}
    if a.endpoint == "word_nonword":
        want["neg_mode"] = a.neg_mode
    if a.decoder:
        want["decoder"] = a.decoder
    return want


def summarize(save_root: str, a) -> dict:
    """A published cell from its per-seed result files.

    Reads ``result_path(save_root, a, seed)`` for each of ``a.seeds``, so the
    settings in `a` pick the files. Per subject, the mean over seeds; then the
    mean and the cross-subject SD (ddof=1) over subjects. The cross-seed SD is
    the SD (ddof=1) of the seeds' cohort means -- the "Seed SD" column of
    Table 20.

    Refuses (ValueError) any file whose recorded settings differ from the ones
    requested, or whose subjects are not ``a.subjects``, and any set of files
    that disagree with one another (the decoder, when --decoder is not given).
    So seeds run at different folds, epochs, event counts or event draws are
    never averaged into one cell, whatever their file names say.
    """
    if len(set(a.seeds)) != len(a.seeds):
        raise ValueError(f"--seeds lists a seed more than once: {a.seeds}")
    want = requested_settings(a)
    runs = []
    for seed in a.seeds:
        path = result_path(save_root, a, seed)
        with open(path, encoding="utf-8") as fh:
            r = json.load(fh)
        if "seed" in r and int(r["seed"]) != int(seed):
            raise ValueError(f"{path} records seed {r['seed']}, not {seed}")
        got = recorded_settings(r)
        bad = [f"{k}={got[k]!r} (requested {want[k]!r})"
               for k in got if k in want and not _same(got[k], want[k])]
        if bad:
            raise ValueError(f"{path} was not run at the requested settings: "
                             + ", ".join(bad))
        if sorted(r["per_subject_auroc"]) != sorted(a.subjects):
            raise ValueError(f"{path} scored {sorted(r['per_subject_auroc'])}, "
                             f"not {sorted(a.subjects)}")
        runs.append((r, got))
    for k in sorted(set().union(*(g for _, g in runs))):
        vals = {json.dumps(g[k]) for _, g in runs if k in g}
        if len(vals) > 1:
            raise ValueError(f"the seeds' result files disagree on {k}: {sorted(vals)}")

    subs = list(a.subjects)
    M = np.array([[r["per_subject_auroc"][s] for s in subs] for r, _ in runs], dtype=float)
    per = M.mean(axis=0)
    cohort = M.mean(axis=1)
    return {
        "decoder": runs[0][0].get("decoder"), "endpoint": a.endpoint,
        "train_mode": a.train_mode, "seeds": [int(s) for s in a.seeds],
        "event_seed": a.event_seed, "nonpaper_settings": setting_tags(a),
        "per_subject_seed_mean": {s: float(v) for s, v in zip(subs, per)},
        "mean": float(per.mean()),
        "cross_subject_sd": float(per.std(ddof=1)) if len(subs) > 1 else float("nan"),
        "cohort_mean_per_seed": {str(r.get("seed")): float(c)
                                 for (r, _), c in zip(runs, cohort)},
        "cross_seed_sd": float(cohort.std(ddof=1)) if len(runs) > 1 else float("nan"),
    }


# ─────────────────────────────── entry point ────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--decoder", choices=list(DECODERS),
                   help="required unless --summarize")
    p.add_argument("--endpoint", default="sentence_onset", choices=list(ENDPOINTS),
                   help="sentence_onset = Task A, word_nonword = Task B")
    p.add_argument("--neg_mode", default="upstream", choices=list(NEG_MODES),
                   help="Task B negatives: upstream = any word-free tile (the paper); "
                        "short_silence = only pauses shorter than 10 s")
    p.add_argument("--train_mode", default="pooled", choices=list(TRAIN_MODES),
                   help="pooled = one model over all subjects, channels zero-padded "
                        "and not aligned (the paper); per_subject = one model per subject")
    p.add_argument("--subjects", nargs="+", default=list(SUBS10),
                   help="order matters for exact reproduction; see SUBS10")
    # 4, not 5: forward_chaining_split cuts the session into n_folds+1 blocks,
    # so the fold count changes train/test MEMBERSHIP. Every arm of the table
    # uses 4.
    p.add_argument("--n_folds", type=int, default=PAPER["n_folds"])
    p.add_argument("--val_frac", type=float, default=PAPER["val_frac"])
    p.add_argument("--epochs", type=int, default=PAPER["epochs"])
    p.add_argument("--patience", type=int, default=PAPER["patience"])
    p.add_argument("--lr", type=float, default=PAPER["lr"])
    p.add_argument("--weight_decay", type=float, default=PAPER["weight_decay"])
    p.add_argument("--batch_size", type=int, default=PAPER["batch_size"])
    p.add_argument("--hidden", type=int, default=PAPER["hidden"],
                   help="LSTM hidden size (HiLoFuseNet, LSTM)")
    p.add_argument("--dropout", type=float, default=PAPER["dropout"])
    p.add_argument("--D", type=int, default=PAPER["D"],
                   help="HiLoFuseNet depthwise multiplier")
    p.add_argument("--seed", type=int, default=42,
                   help="initialisation and batch order only; the paper uses 42, 1, 2")
    p.add_argument("--event_seed", type=int, default=PAPER_EVENT_SEED,
                   help="which events are drawn; 42 for every published seed")

    # These three select CORTEG's cache. The defaults are the paper's and make
    # the baselines read exactly the features the CORTEG runs read.
    p.add_argument("--win_sec", type=float, default=PAPER["win_sec"])
    p.add_argument("--pre_sec", type=float, default=PAPER["pre_sec"])
    p.add_argument("--max_per_class", type=int, default=PAPER["max_per_class"])
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--skip_event_check", action="store_true",
                   help="do not verify the features against the requested events")

    p.add_argument("--save_root", default="",
                   help="default: <output root>/braintreebank/base_<decoder>")
    p.add_argument("--summarize", action="store_true",
                   help="print the published cell from existing result files; no "
                        "training. The other flags select the files, as for a run")
    p.add_argument("--seeds", nargs="+", type=int, default=list(PAPER_SEEDS),
                   help="seeds read by --summarize")
    return p


def _inside_repo(path: str) -> bool:
    path = os.path.realpath(path)
    return os.path.commonpath([os.path.realpath(REPO), path]) == os.path.realpath(REPO)


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if not a.decoder and not (a.summarize and a.save_root):
        parser.error("--decoder is required (--summarize takes --save_root instead)")
    if a.endpoint == "sentence_onset" and a.neg_mode != "upstream":
        parser.error("--neg_mode applies to --endpoint word_nonword only: "
                     "Task A has no silence negatives")
    if len(set(a.subjects)) != len(a.subjects):
        # A repeated subject would be trained on twice per fold and scored twice.
        parser.error(f"--subjects lists a subject more than once: {a.subjects}")
    from data.braintreebank import btb_output_root
    # The same folder scripts/table3_btb_baselines.sh writes to.
    save_root = a.save_root or os.path.join(btb_output_root(), f"base_{a.decoder}")

    if a.summarize:
        s = summarize(save_root, a)
        print(f"[btb-base] {s['decoder']} {a.endpoint}: {s['mean']:.4f} "
              f"± {s['cross_subject_sd']:.4f} (cross-subject SD), "
              f"cross-seed SD {s['cross_seed_sd']:.4f}, seeds {s['seeds']}"
              + (f", not the paper's settings: {s['nonpaper_settings']}"
                 if s["nonpaper_settings"] else ""))
        for subj, v in s["per_subject_seed_mean"].items():
            print(f"    {subj:<8}{v:.4f}")
        dest = summary_path(save_root, a)
        if _inside_repo(save_root):
            # e.g. --save_root paper_cells/btb/base_LSTM: print, do not add files.
            print(f"[btb-base] {save_root} is inside the repository; summary not saved")
        else:
            with open(dest, "w", encoding="utf-8") as fh:
                json.dump(s, fh, indent=2)
            print(f"[btb-base] saved {dest}")
        return s

    tags = setting_tags(a)
    departures = tags + ([f"ev{a.event_seed}"] if a.event_seed != PAPER_EVENT_SEED else [])
    if departures:
        print(f"[btb-base] not the paper's settings ({', '.join(departures)}); "
              "the result file name carries them", flush=True)

    # Seeded before anything draws a random number, as in the paper runs.
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    subs = list(a.subjects)
    t0 = time.time()

    data, times = {}, {}
    for s in subs:
        X, y, ev = load_subject(s, a)
        data[s] = (X, y); times[s] = ev
        print(f"  [{s}] X{X.shape} pos={int(y.sum())} neg={int((1 - y).sum())}", flush=True)

    folds, reports, footprint = make_folds(subs, data, times, a.n_folds, a.val_frac)
    print(f"  splits: {a.n_folds} causal folds x {len(subs)} subjects, "
          f"footprint={footprint}s, embargo={EMBARGO_SEC}s, overlapping pairs=0", flush=True)

    per_fold, record = train(subs, data, folds, a, dev, log=lambda m: print(m, flush=True))

    per = {s: float(np.nanmean(per_fold[s])) for s in subs}
    score = float(np.mean(list(per.values())))
    print(f"\n[btb-base] {a.decoder} {a.endpoint} seed {a.seed} ({a.train_mode}): "
          f"mean AUROC over {len(subs)} subjects = {score:.4f}")
    for s in subs:
        print(f"    {s:<8}{per[s]:.4f}")

    neg_mode = a.neg_mode if a.endpoint == "word_nonword" else None
    os.makedirs(save_root, exist_ok=True)
    out = result_path(save_root, a)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({
            "mean_auroc": score, "score": score,
            "decoder": a.decoder, "endpoint": a.endpoint,
            "seed": a.seed, "event_seed": a.event_seed,
            "train_mode": a.train_mode, "neg_mode": neg_mode,
            "nonpaper_settings": tags,
            "subjects": subs, "n_subjects": len(subs),
            "per_subject_auroc": per,
            "per_subject_fold_auroc": {s: [float(x) for x in per_fold[s]] for s in subs},
            "folds": a.n_folds, "val_frac": a.val_frac,
            "win_sec": footprint, "embargo_sec": EMBARGO_SEC,
            "splits": reports, "early_stop_on": "val", "loss": "bce",
            "training": record,
            "hp": {"lr": a.lr, "hidden": a.hidden, "dropout": a.dropout, "D": a.D,
                   "weight_decay": a.weight_decay, "epochs": a.epochs,
                   "patience": a.patience, "batch_size": a.batch_size},
            "features": {"source": "run_btb_classification.extract_subject",
                         "win_sec": a.win_sec, "pre_sec": a.pre_sec,
                         "hga_band": list(HGA_BAND), "max_per_class": a.max_per_class,
                         "neg_mode": neg_mode,
                         "stacked": "[HIGH, LOW-resampled-to-HIGH]",
                         "event_check": not a.skip_event_check},
            "protocol": ("pooled over subjects, channels zero-padded and not aligned"
                         if a.train_mode == "pooled" else
                         "one model per subject and fold") +
                        "; per-subject causal forward chaining with embargo "
                        "(data.braintreebank.forward_chaining_split), val block carved "
                        "causally from the tail of history; early stopping on VAL",
            "device": str(dev), "elapsed_s": time.time() - t0,
        }, fh, indent=2)
    print(f"[btb-base] saved {out}")
    return out


if __name__ == "__main__":
    main()
