"""CORTEG on BrainTreebank: Task A (sentence-initial vs mid-sentence word) and
Task B (word vs non-word).

BrainTreebank is public sEEG from 10 patients watching films, with a word-level
transcript. The paper scores two binary tasks by per-subject AUROC:

* **Task A** (``--endpoint sentence_onset``, the default): is this word the first
  word of a sentence, or a word in the middle of one? Both classes are words, so
  this is NOT upstream BrainBERT/PopT's ``sentence_onset`` task, which contrasts
  sentence-initial words with non-word silence.
* **Task B** (``--endpoint word_nonword``): the BrainBERT/PopT word-vs-non-word
  benchmark. Positives are word onsets; negatives are the centres of 1 s
  word-free tiles inside the movie's trigger range.

CORTEG reads the post-event window [t, t+1.5] s on both tasks (``--pre_sec 0.0
--win_sec 1.5``); on Task B, t is the onset of a word or the centre of a silence
tile. Classes are balanced to at most ``--max_per_class`` 900 events each.

This runner reuses the Stanford model code unchanged --
``run_regression_hilo_clean.build_model``, ``configure_lora_lastn_probe`` and
``unfreeze_merge_params`` -- so the architecture here is the same CORTEG, with
``d_out=1`` and a BCE loss instead of the 5-D regression head. What differs is the
data path (``data.braintreebank``), the per-subject LoRA, and the evaluation
protocol.

The defaults are the recipe of the paper runs (Tables 3, 9 and 20): one pooled
model over all ten subjects, with a separate LoRA adapter (r=4, alpha=16, dropout
0.2) on qkv/proj/fc1/fc2 of the last 4 blocks for each subject, selected by
subject id; 60 epochs, AdamW lr 3e-4, weight decay 5e-3, batch 16 with 4-step
gradient accumulation, bf16 autocast on CUDA, cosine schedule with
max(1, epochs // 10) warmup epochs down to 1e-5, no head dropout, validation every
2 epochs with patience 15 evaluations, 4 causal folds with a 15 % causal
validation block. The random-init control changes only ``--no_pretrained``; it
keeps the backbone config (drop_path_rate 0.1) and every training setting. The
paper averages training seeds 42, 1 and 2, all on ONE event draw
(``--event_seed 42``).

Protocol, which is where the care goes:

* **Forward-chaining folds.** The session is cut into contiguous time blocks;
  fold i fits on blocks 0..i and tests on block i+1, so no model ever sees signal
  recorded after its test data. A single 60/15/25 cut was rejected: it left ~450
  test events per subject and dropped the power of the headline contrast from
  0.97 to 0.71, while changing the estimand to "final-quarter discriminability".
* **A causal validation block**, embargoed from both fit and test. Carving val by
  random permutation interleaves it with fit, so early stopping selects on
  overlapping windows -- model-selection leakage that survives any train/test check.
* **A 7 s embargo**, sized by the widest arm sharing the split, with every fold
  boundary asserted free of overlapping windows. Fold membership depends only on
  the event times and the embargo, so CORTEG's 1.5 s window on Task B gets the
  same folds as the 5 s foundation-model arms.

Notes vs the paper runs. The defaults reproduce what produced the published
numbers; where that behaviour is questionable, a flag gives the alternative.

* Early stopping selects on the POOLED validation AUROC, which ranks all subjects'
  windows together. Most of those pairs are between subjects, so it is largely
  blind to the per-subject AUROC that is reported. ``--select_metric
  per_subject`` selects on the mean per-subject validation AUROC instead.
* ``--patience`` counts evaluations, not epochs: with ``--eval_every 2``,
  patience 15 allows about 30 epochs without improvement.
* Task B's candidate events are filtered with ``word_nonword_events``' default
  5 s centred window, not CORTEG's 1.5 s one. That is what the paper runs did, and
  it keeps CORTEG on exactly the events the 5 s foundation-model arms score.
* Task B's negatives are drawn uniformly from all word-free tiles, so many sit in
  long silences (credits, scene breaks). ``--neg_mode short_silence`` draws them
  only from pauses shorter than 10 s.
* A Task B negative is a 1 s word-free tile, but CORTEG reads [c, c+1.5] from
  the tile centre c, so only the first 0.5 s of its window is guaranteed free of
  speech. In 8-18 % of each subject's 900 negatives (14 % on average) the window
  reaches the onset of the next word; the paper runs did the same. There is no
  flag for a stricter pool: upstream defines a non-word by the tile alone (the
  5 s foundation-model windows extend 2 s past it on both sides), and a stricter
  pool would no longer be the events those arms score.
* bf16 autocast is on for CUDA only; CPU runs are fp32. GPU runs are not
  bit-reproducible at a fixed seed (AMP and cuDNN kernels): three repeats of the
  seed-42 Task B gate run with the paper code spanned 0.0074 in cohort-mean AUROC.

Results go to ``--save_root`` (default ``<output root>/braintreebank/runs``) as
``btb_<train_mode>_<merge>_<endpoint>[_<tag>...]_seed<S>.json``. Every setting
that departs from the paper run adds a tag (see ``setting_tags``), so a variant
or smoke run never overwrites a paper-setting file.

Example (one subject, Task B):

    python -m experiments.run_btb_classification \\
        --subjects sub_3 --endpoint word_nonword --merge_strategy layerwise_gate \\
        --model_kwargs_json configs/steegformer_small.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data.braintreebank import (
    EMBARGO_SEC,
    HGA_HIGH,
    HGA_LOW,
    btb_output_root,
    btb_root,
    build_time_to_sample,
    corteg_features,
    estimate_fs,
    extract_windows,
    forward_chaining_split,
    load_electrode_map,
    load_events,
    load_localization_mni,
    split_report,
    word_nonword_events,
)

REPO = Path(__file__).resolve().parent.parent

# Architecture constants — these are the paper's BrainTreebank settings and are
# deliberately not exposed as flags; changing them changes the reported model.
HI_PATCH_SIZE = 25
HI_INJECT_LAST_N = 4
CHANNEL_ADAPTER = "knn_soft_fourier"
KNN_K = 8
LORA_TARGETS = "qkv,proj,fc1,fc2"

ENDPOINTS = ("sentence_onset", "word_nonword")
NEG_MODES = ("upstream", "short_silence")
PAPER_EVENT_SEED = 42
PAPER_MAX_PER_CLASS = 900

# Subject order of the paper runs: the seven Population Transformer test-trial
# subjects, then the three that complete the cohort. The order is not cosmetic.
# A subject's position is its sid, which picks its LoRA adapter and its slot in
# the round-robin batch interleave, so another order is another training run.
PAPER_SUBJECT_ORDER = ["sub_1", "sub_2", "sub_3", "sub_4", "sub_6", "sub_7",
                       "sub_10", "sub_5", "sub_8", "sub_9"]


def clean_electrodes(subj: str) -> list:
    """Electrode selection shared with the Population Transformer benchmark.

    The selection lives in the PopT repository and carries no licence permitting
    redistribution, so it is not vendored here. Clone it and point
    ``POPT_REPO`` at the checkout:

        git clone https://github.com/czlwang/PopulationTransformer
        export POPT_REPO=$PWD/PopulationTransformer
    """
    repo = os.environ.get("POPT_REPO", "")
    rel = "electrode_selections/clean_laplacian.json"
    path = os.path.join(repo, rel) if repo else ""
    if not path or not os.path.exists(path):
        raise SystemExit(
            "BrainTreebank needs the shared electrode selection, which this "
            "repository does not redistribute.\n"
            f"  Expected: $POPT_REPO/{rel}\n"
            "  Fix:  git clone https://github.com/czlwang/PopulationTransformer\n"
            "        export POPT_REPO=$PWD/PopulationTransformer\n"
            "  Using a different electrode set changes the numbers and makes the "
            "comparison with PopT/BrainBERT/Brant unfair."
        )
    with open(path, encoding="utf-8") as fh:
        sel = json.load(fh)
    if subj not in sel:
        raise SystemExit(f"{subj} not in {rel} (has {sorted(sel)[:4]}…)")
    return sel[subj]


# The trial scored for each subject, matching the Population Transformer test
# split so the CORTEG and iEEG-FM arms see identical data. Note sub_1, sub_2 and
# sub_6 are NOT trial000 -- defaulting to trial000 silently scores a different
# film for those three.
CANONICAL_TRIAL = {
    "sub_1": "trial001", "sub_2": "trial006", "sub_3": "trial000",
    "sub_4": "trial000", "sub_5": "trial000", "sub_6": "trial004",
    "sub_7": "trial000", "sub_8": "trial000", "sub_9": "trial000",
    "sub_10": "trial000",
}


def trial_of(root: str, subj: str) -> str:
    """The trial to score for `subj`, verified present on disk."""
    trial = CANONICAL_TRIAL.get(subj)
    if trial and os.path.exists(
            os.path.join(root, "all_subject_data", f"{subj}_{trial}.h5")):
        return trial
    have = sorted(f for f in os.listdir(os.path.join(root, "all_subject_data"))
                  if f.startswith(f"{subj}_trial") and f.endswith(".h5"))
    if not have:
        raise SystemExit(
            f"no .h5 for {subj} under {root}/all_subject_data"
            + (f" (expected {subj}_{trial}.h5)" if trial else ""))
    if trial:
        raise SystemExit(
            f"{subj}: the scored trial is {trial}, but only {have} is downloaded. "
            "Scoring a different trial means a different film and different events.")
    return have[0][len(subj) + 1:-3]


def movie_of(root: str, subj: str, trial: str) -> str:
    """Film name from the per-trial metadata."""
    meta_path = os.path.join(root, "subject_metadata", f"{subj}_{trial}_metadata.json")
    with open(meta_path, encoding="utf-8") as fh:
        movie = str(json.load(fh)["filename"]).strip()
    if not movie:
        raise SystemExit(f"empty 'filename' in {meta_path}")
    return movie


# =========================================================================
# Events and features
# =========================================================================

def event_spec(args):
    """(endpoint, neg_mode, event_seed) from `args`, validated.

    ``endpoint`` has no default on purpose: a caller that forgot it would
    otherwise get Task A features under a Task B request, with every
    downstream check passing.
    """
    endpoint = getattr(args, "endpoint", None)
    if endpoint not in ENDPOINTS:
        raise ValueError(f"args.endpoint must be one of {ENDPOINTS}, got {endpoint!r}")
    neg_mode = getattr(args, "neg_mode", "upstream") or "upstream"
    if neg_mode not in NEG_MODES:
        raise ValueError(f"args.neg_mode must be one of {NEG_MODES}, got {neg_mode!r}")
    if endpoint == "sentence_onset" and neg_mode != "upstream":
        raise ValueError("neg_mode applies to word_nonword only: Task A has no "
                         "silence negatives")
    event_seed = int(getattr(args, "event_seed", PAPER_EVENT_SEED))
    return endpoint, neg_mode, event_seed


def _band(args):
    return (float(getattr(args, "hga_low", HGA_LOW)),
            float(getattr(args, "hga_high", HGA_HIGH)))


def resolve_trial(subj: str, args) -> str:
    return getattr(args, "trial", None) or trial_of(btb_root(), subj)


def cache_tag(subj: str, trial: str, args) -> str:
    """Cache file name for one subject's features.

    Everything that selects WHICH events are drawn is in the key: the endpoint,
    Task B's negative pool, ``event_seed`` and ``max_per_class``. The training
    ``--seed`` is not: the paper trains seeds 42, 1 and 2 on one event draw, so
    all three must read the same cache.

    Task A with upstream negatives keeps the untagged name the release has always
    written, ``btb_<subj>_<trial>_win1.5_pre0.0_hfa70-200_n900_s42.npz``, so
    existing Task A caches stay valid; ``_s`` is the event seed. The endpoint and
    negative mode are also stored inside the file and checked on load.
    """
    endpoint, neg_mode, event_seed = event_spec(args)
    ep = "" if endpoint == "sentence_onset" else f"_{endpoint}"
    neg = "" if neg_mode == "upstream" else f"_{neg_mode}"
    lo, hi = _band(args)
    return (f"btb_{subj}_{trial}{ep}{neg}_win{float(args.win_sec)}_pre{float(args.pre_sec)}"
            f"_hfa{int(lo)}-{int(hi)}_n{int(args.max_per_class)}_s{event_seed}.npz")


def select_events(root: str, subj: str, trial: str, args):
    """The balanced event draw for ``args.endpoint``: (event_times, labels, meta).

    Times are movie seconds in temporal order, before the window-bounds mask.
    For Task A they are word onsets. For Task B they are window centres: the
    onset for a word, the tile centre for a silence (upstream centres its
    window); CORTEG's [t, t+1.5] window is anchored at them either way, so a
    silence window runs 1 s past its word-free tile (module notes).

    The draw depends on ``event_seed`` and ``max_per_class`` only, never on the
    training ``--seed``. Task B keeps ``word_nonword_events``' default 5 s
    window for filtering candidates against the trigger range; passing CORTEG's
    1.5 s here would admit events the paper runs and the FM arms never scored.
    """
    endpoint, neg_mode, event_seed = event_spec(args)
    movie = movie_of(root, subj, trial)
    if endpoint == "word_nonword":
        starts, labels, meta = word_nonword_events(
            root, subj, trial, movie, int(args.max_per_class), event_seed,
            neg_mode=neg_mode)
        return np.asarray(starts, dtype=np.float64), np.asarray(labels), meta
    starts, labels = load_events(root, movie, int(args.max_per_class), event_seed)
    return np.asarray(starts, dtype=np.float64), np.asarray(labels), None


def _check_cache(d, cache: str, want: dict) -> None:
    """Refuse a cache that holds other events or another window than requested.

    Caches written before the endpoint switch store only the window; everything
    else they hold is fixed by their name, and they hold Task A events drawn
    with the ``_s`` seed of that name. So they are accepted for exactly that
    request and nothing else.
    """
    for key, value in want.items():
        if key in d.files:
            got = d[key].item()
            same = (str(got) == value) if isinstance(value, str) \
                else abs(float(got) - float(value)) < 1e-9
            if not same:
                raise RuntimeError(f"{cache} holds {key}={got!r} but this run wants "
                                   f"{value!r}; rebuild it (--no_cache) rather than reuse")
        elif key in ("win_sec", "pre_sec"):
            raise RuntimeError(f"{cache} records no {key}; rebuild it (--no_cache)")
        elif key == "endpoint" and value != "sentence_onset":
            raise RuntimeError(f"{cache} predates the endpoint switch and holds Task A "
                               f"events; this run wants {value}")
        elif key == "neg_mode" and value != "upstream":
            raise RuntimeError(f"{cache} predates neg_mode and holds the upstream pool; "
                               f"this run wants {value}")
    for key in ("event_times", "y", "x_lo", "x_hi", "xyz"):
        if key not in d.files:
            raise RuntimeError(f"{cache} lacks {key}; rebuild it (--no_cache)")
    if len(d["event_times"]) != len(d["y"]):
        raise RuntimeError(f"{cache}: {len(d['event_times'])} event times for "
                           f"{len(d['y'])} labels")


def electrode_selection(root: str, subj: str):
    """``(names, h5 channel indices, MNI coordinates in metres)`` of the electrodes used.

    The shared PopT selection (``clean_electrodes``), restricted to electrodes
    that have a label in the recording and an MNI coordinate, in the
    selection's order. The channel order of the features follows it.
    """
    name2idx, _ = load_electrode_map(root, subj)
    loc = load_localization_mni(root, subj)          # shared MNI frame, mm
    use = [n for n in clean_electrodes(subj) if n in name2idx and n in loc]
    ch_idx = [name2idx[n] for n in use]
    xyz_m = (np.array([loc[n] for n in use], dtype=np.float64) / 1000.0).astype(np.float32)
    return use, ch_idx, xyz_m


_REBUILD_OVER_SELECTION = (
    "Point POPT_REPO at the checkout the cache was built with. --no_cache rebuilds "
    "it over the current selection instead, replacing this file, and the numbers "
    "then differ from the paper's.")


def _check_electrodes(d, cache: str, use: list, xyz_m: np.ndarray) -> None:
    """Refuse a cache built over another electrode set than the current selection.

    The selection is read from files outside this repository ($POPT_REPO's
    clean_laplacian.json, the dataset's electrode labels and localization), so
    another checkout can change it without changing the cache name or any of
    the metadata ``_check_cache`` compares. Caches that store the electrode
    names are compared by name, in order. Every cache stores the coordinates,
    which pin down the set and its order too, so older caches without the
    names are checked through those.
    """
    if "electrodes" in d.files:
        got = [str(n) for n in d["electrodes"]]
        if got != list(use):
            gone, new = sorted(set(got) - set(use)), sorted(set(use) - set(got))
            what = ("the same electrodes in another order" if not gone and not new else
                    f"only in the cache: {gone[:5]}; only in the selection: {new[:5]}")
            raise RuntimeError(
                f"{cache} was built over {len(got)} electrodes and the current "
                f"selection has {len(use)} ({what}). {_REBUILD_OVER_SELECTION}")
    xyz = np.asarray(d["xyz"])
    if xyz.shape != xyz_m.shape or not np.allclose(xyz, xyz_m, rtol=0.0, atol=1e-6):
        same_n = xyz.shape == xyz_m.shape
        raise RuntimeError(
            f"{cache} holds the coordinates of another electrode set than the current "
            f"selection ({xyz.shape[0]} cached, {xyz_m.shape[0]} selected"
            f"{', other positions or order' if same_n else ''}). {_REBUILD_OVER_SELECTION}")


def extract_subject(subj: str, args):
    """Features for one subject: ``(x_lo, x_hi, y, xyz_m, event_times)``, cached.

    The shared entry point of the BrainTreebank runners. It reads these
    attributes of ``args`` (an ``argparse.Namespace`` or anything like it):

      endpoint          'sentence_onset' (Task A) or 'word_nonword' (Task B);
                        required, there is no default
      neg_mode          'upstream' (default) or 'short_silence'; Task B only
      event_seed        seed of the balanced event draw (default 42, the paper's)
      max_per_class     events kept per class (the paper uses 900)
      win_sec, pre_sec  the window [t + pre, t + pre + win] (paper: 1.5, 0.0)
      hga_low, hga_high high-frequency-activity band (default 70-200 Hz)
      trial             optional; default is the subject's scored trial
      no_cache          optional; rebuild even when a cache exists
      event_chunk       optional; events per block in the feature transform
                        (default 200; bounds memory, the result is identical)

    Returns:
      x_lo         (N, C, T_lo) float32, 128 Hz low stream (T_lo = 192 at 1.5 s)
      x_hi         (N, C, T_hi) float32, 200 Hz high-frequency-activity envelope
                   (T_hi = 300 at 1.5 s)
      y            (N,) int64, 1 = sentence-initial word (Task A) / word (Task B)
      xyz_m        (C, 3) float32 MNI coordinates in metres (millimetres / 1000)
      event_times  (N,) float64 movie seconds of the scored events, in temporal
                   order. Build the causal split from these, never from the
                   transcript: events outside the trigger range are dropped.

    The cache name carries the window, band, endpoint, negative pool,
    max_per_class and event seed (see ``cache_tag``), and the file stores them
    too, so a cache built for another request is refused rather than reused.
    A cache hit also recomputes the electrode selection and refuses a cache
    built over another one (``_check_electrodes``), so it needs ``POPT_REPO``
    just as a build does.
    """
    endpoint, neg_mode, event_seed = event_spec(args)
    win, pre = float(args.win_sec), float(args.pre_sec)
    hga_low, hga_high = _band(args)
    root = btb_root()
    trial = resolve_trial(subj, args)
    tag = cache_tag(subj, trial, args)
    cache = os.path.join(btb_output_root(), "cache", tag)
    want = {"endpoint": endpoint, "neg_mode": neg_mode, "event_seed": event_seed,
            "max_per_class": int(args.max_per_class), "win_sec": win, "pre_sec": pre,
            "hga_low": hga_low, "hga_high": hga_high}
    use, ch_idx, xyz_m = electrode_selection(root, subj)
    if os.path.exists(cache) and not getattr(args, "no_cache", False):
        with np.load(cache, allow_pickle=False) as d:
            _check_cache(d, cache, want)
            _check_electrodes(d, cache, use, xyz_m)
            out = (d["x_lo"], d["x_hi"], np.asarray(d["y"]).astype(np.int64),
                   d["xyz"], np.asarray(d["event_times"], dtype=np.float64))
        if out[0].shape[1] != len(use) or out[1].shape[1] != len(use):
            raise RuntimeError(f"{cache}: features over {out[0].shape[1]} channels "
                               f"for {len(use)} electrodes; rebuild it (--no_cache)")
        print(f"[cache] {tag}", flush=True)
        return out

    fs = estimate_fs(root, subj, trial)              # measured, never assumed
    t2s = build_time_to_sample(root, subj, trial)
    starts, labels, meta = select_events(root, subj, trial, args)
    print(f"[{subj}] trial={trial} endpoint={endpoint} fs={fs:.1f}Hz "
          f"electrodes={len(use)} events={len(starts)}", flush=True)

    x_raw, valid = extract_windows(root, subj, trial, ch_idx, starts, t2s, fs, pre, win)
    y = labels[valid].astype(np.int64)
    ev_t = starts[valid]
    # Each event is transformed independently, so blocking is exact; it only
    # bounds peak memory, which otherwise reaches tens of GB for sub_7 (191
    # electrodes) because the transform holds several float64 copies.
    chunk = int(getattr(args, "event_chunk", 200) or 0)
    if chunk > 0 and len(x_raw) > chunk:
        parts = [corteg_features(x_raw[i:i + chunk], fs, hga_low, hga_high)
                 for i in range(0, len(x_raw), chunk)]
        x_lo = np.concatenate([p[0] for p in parts])
        x_hi = np.concatenate([p[1] for p in parts])
        del parts
    else:
        x_lo, x_hi = corteg_features(x_raw, fs, hga_low, hga_high)
    del x_raw

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    # A per-process temporary name: two processes building the same cache at
    # once (two runs of this runner, or a CPU pre-build) would otherwise
    # write one file, and the first os.replace would publish a half-written one.
    tmp = f"{cache[:-len('.npz')]}.{os.getpid()}.tmp.npz"
    try:
        np.savez_compressed(
            tmp, x_lo=x_lo, x_hi=x_hi, y=y, xyz=xyz_m, event_times=ev_t,
            win_sec=np.float64(win), pre_sec=np.float64(pre),
            endpoint=np.array(endpoint), neg_mode=np.array(neg_mode),
            event_seed=np.int64(event_seed), max_per_class=np.int64(args.max_per_class),
            hga_low=np.float64(hga_low), hga_high=np.float64(hga_high),
            fs_measured=np.float64(fs), subject=np.array(subj), trial=np.array(trial),
            electrodes=np.array(use), wnw_meta=np.array(json.dumps(meta) if meta else ""))
        os.replace(tmp, cache)                       # never leave a half-written cache
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    print(f"[cache] wrote {tag}", flush=True)
    return x_lo, x_hi, y, xyz_m, ev_t


# =========================================================================
# Model
# =========================================================================

def build_corteg(C: int, T_lo: int, xyz_m: np.ndarray, args):
    """The same CORTEG as Stanford, with a 1-D head.

    build_model wants coordinates in millimetres and divides by 1000 internally,
    so the metres-scale array is multiplied back up here.
    """
    from experiments.run_regression_hilo_clean import build_model, unfreeze_merge_params
    from models.steegformer.probe import configure_lora_lastn_probe

    a = SimpleNamespace(
        model_kwargs_json=args.model_kwargs_json, no_pretrained=args.no_pretrained,
        steegformer_variant=args.steegformer_variant,
        merge_strategy=args.merge_strategy, stream="both",
        hi_patch_size=HI_PATCH_SIZE, hi_inject_last_n=HI_INJECT_LAST_N,
        layerwise_gate_bottleneck=args.layerwise_gate_bottleneck,
        layerwise_gate_act=args.layerwise_gate_act,
        layerwise_gate_share_blocks=False,
        channel_adapter=CHANNEL_ADAPTER, knn_k=KNN_K,
        adapter_branch="both", xyz_mode="real",
        head_dropout=args.head_dropout,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_last_n=args.lora_last_n,
        lora_targets=LORA_TARGETS, full_finetune=False,
    )
    model = build_model(a, C_in=C, T_in=T_lo, ecog_xyz_m=xyz_m * 1000.0, d_out=1)
    configure_lora_lastn_probe(
        model, n_last=args.lora_last_n, r=args.lora_r, alpha=args.lora_alpha,
        dropout=args.lora_dropout, targets=tuple(LORA_TARGETS.split(",")))
    unfreeze_merge_params(model)
    # Build the readout NOW. It is otherwise created on the first forward, i.e.
    # after the optimizer has been built from model.parameters(), so it never
    # enters the optimizer and stays at its random init for the whole run.
    _embed = getattr(model.backbone, "embed_dim", None)
    if _embed is not None:
        model.head.materialize_head(embed_dim=int(_embed))
    return model


def add_per_subject_lora(model, n_subjects: int, lora_last_n: int, dropout: float) -> int:
    """Give each subject its own LoRA adapter in the last `lora_last_n` blocks.

    Swaps every LoRALinear that build_corteg injected for a TaskLoRALinear with
    one (A, B) pair per subject around the same frozen base weight. B starts at
    zero, so the swap leaves the model's output unchanged. Call it before the
    optimizer is built, or the new adapters are never updated, and select the
    adapter with ``set_active_task(model, sid)`` before every forward.

    Returns the number of layers swapped (16 for 4 blocks x qkv/proj/fc1/fc2).
    """
    from models.steegformer.lora import LoRALinear, TaskLoRALinear

    def _swap(mod):
        for name, child in list(mod.named_children()):
            if isinstance(child, LoRALinear):
                setattr(mod, name, TaskLoRALinear(
                    child.base, r=child.r, alpha=child.alpha,
                    dropout=float(dropout), n_tasks=int(n_subjects)))
            else:
                _swap(child)

    blocks = model.backbone.blocks
    for bi in range(max(0, len(blocks) - int(lora_last_n)), len(blocks)):
        _swap(blocks[bi])
    n = sum(isinstance(m, TaskLoRALinear) for m in model.modules())
    if n == 0:
        raise RuntimeError("per-subject LoRA: no LoRALinear to swap; build_corteg "
                           "must inject LoRA first")
    return n


def default_backbone_config(variant: str) -> str:
    """configs/steegformer_<variant>.json, for a --no_pretrained run given none.

    The config carries the backbone's regularisation (drop_path_rate 0.1) as
    well as the checkpoint path, and --no_pretrained skips only the checkpoint.
    Without it the random-init control would train with drop_path_rate 0.0,
    unlike the paper's control and the CORTEG arms it is compared against.
    """
    return str(REPO / "configs" / f"steegformer_{variant}.json")


def resolve_backbone_config(args) -> str:
    """Set and return ``args.model_kwargs_json``, the backbone config to build with.

    A run without a config stops unless it asks for random init: the backbone
    would otherwise stay randomly initialised under the CORTEG name. A
    ``--no_pretrained`` run without one falls back to
    ``default_backbone_config``, so the control keeps drop_path_rate 0.1.
    """
    if args.model_kwargs_json:
        return args.model_kwargs_json
    if not args.no_pretrained:
        raise SystemExit(
            "No pretrained backbone configured.\n"
            "  CORTEG loads ST-EEGFormer weights via --model_kwargs_json; without it\n"
            "  the backbone stays randomly initialised, which is the 'random init'\n"
            "  ablation rather than CORTEG.\n"
            f"  Fix:  --model_kwargs_json configs/steegformer_{args.steegformer_variant}.json\n"
            "  Or, to request random init deliberately:  --no_pretrained")
    args.model_kwargs_json = default_backbone_config(args.steegformer_variant)
    print(f"[random init] backbone architecture from {args.model_kwargs_json}; "
          "its checkpoint is not loaded", flush=True)
    return args.model_kwargs_json


def warmup_epochs_of(args) -> int:
    """--warmup_epochs, or max(1, epochs // 10) as in the paper runs."""
    if getattr(args, "warmup_epochs", None) is not None:
        return int(args.warmup_epochs)
    return max(1, int(args.epochs) // 10)


# =========================================================================
# Training
# =========================================================================

def run_pooled_fold(bundles, fold, args, device):
    """Train ONE model across all subjects on this fold; score each separately.

    This is the regime the paper reports. A pooled model sees every subject's
    windows, which is the whole point of the transfer claim -- a per-subject model
    is a different experiment and gets a different number.

    Variable electrode counts are handled the way the Stanford pooled path does
    it: one model built at ``max_C``, batches kept homogeneous per subject by
    SubjectInterleavedSampler, and the right coordinates attached per batch by
    make_collate_fn(SubjectXYZBank). Homogeneous batches are also what lets one
    per-subject LoRA adapter be selected per batch.

    Returns {sid: test AUROC} (NaN when a subject's test block has one class).
    """
    import torch.nn.functional as F
    from sklearn.metrics import roc_auc_score
    from torch.utils.data import ConcatDataset, DataLoader
    from data.collate import SubjectXYZBank, make_collate_fn
    from data.datasets import HiLoAddDataset
    from data.scalers import apply_zscore_3d_per_channel, fit_zscore_3d_per_channel
    from experiments.common import set_seed
    from models.steegformer.lora import set_active_task
    from train.earlystop import EarlyStopper
    from train.engine import EngineConfig, train_one_epoch
    from train.lr_schedule import WarmupCosineLR
    from train.sampling import SubjectInterleavedSampler

    # Reseed every fold, as the paper runs did: Python's `random` drives the
    # sampler's shuffle and is otherwise never seeded, and without a reseed
    # fold k>0 starts from whatever RNG state early stopping left behind.
    set_seed(args.seed)
    dev = torch.device(device)

    tr_sets, val_eval, te_eval, xyz_mm = [], [], [], []
    for sid, (x_lo, x_hi, y, xyz_m, _) in enumerate(bundles):
        fit_i, val_i, te_i = (np.asarray(a) for a in fold[sid])
        lo_st = fit_zscore_3d_per_channel(x_lo[fit_i])      # fit split only
        hi_st = fit_zscore_3d_per_channel(x_hi[fit_i])
        z = lambda a, st: apply_zscore_3d_per_channel(a, st)
        tr_sets.append(HiLoAddDataset(z(x_lo[fit_i], lo_st), z(x_hi[fit_i], hi_st),
                                      y[fit_i].astype(np.float32)[:, None], sid))
        val_eval.append((z(x_lo[val_i], lo_st), z(x_hi[val_i], hi_st), y[val_i], sid))
        te_eval.append((z(x_lo[te_i], lo_st), z(x_hi[te_i], hi_st), y[te_i], sid))
        xyz_mm.append(xyz_m * 1000.0)

    max_C = max(b[0].shape[1] for b in bundles)
    T_lo = bundles[0][0].shape[2]
    model = build_corteg(max_C, T_lo, bundles[0][3], args).to(dev)
    psl = bool(args.per_subject_lora)
    if psl:
        n = add_per_subject_lora(model, len(bundles), args.lora_last_n, args.lora_dropout)
        print(f"  [per-subject LoRA] {n} layers x {len(bundles)} subjects "
              f"in the last {args.lora_last_n} blocks", flush=True)

    bank = SubjectXYZBank.from_mm(xyz_mm)
    sampler = SubjectInterleavedSampler([len(d) for d in tr_sets],
                                        batch_size=args.batch_size, shuffle=True)
    tr_dl = DataLoader(ConcatDataset(tr_sets), batch_sampler=sampler, num_workers=0,
                       pin_memory=(dev.type == "cuda"), collate_fn=make_collate_fn(bank))

    # Built after the swap, so the per-subject adapters are in it.
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineLR(opt, warmup_epochs=warmup_epochs_of(args),
                           max_epochs=args.epochs, min_lr=args.min_lr)
    # bf16 needs no GradScaler; the engine only uses one for fp16.
    cfg = EngineConfig(use_amp=bool(args.use_amp), amp_dtype="bf16",
                       accum_iter=int(args.accum_iter), max_norm=float(args.max_norm))
    early = EarlyStopper(patience=args.patience)

    def forward(m, x_lo, x_hi, xyz, sid):
        if psl:
            set_active_task(m, int(sid))
        return m(x_lo, x_hi=x_hi, ecog_xyz=xyz)

    def step_fn(m, batch):
        sids = batch["sid"].reshape(-1)
        sid = int(sids[0])
        if psl and not bool((sids == sid).all()):
            raise RuntimeError("a training batch mixes subjects; per-subject LoRA "
                               "would train them all on one subject's adapter")
        y_hat = forward(m, batch["x_raw"], batch["x_hi"], batch["ecog_xyz"], sid)
        # Flatten both before the loss: (B, 1) against (B,) would broadcast to
        # (B, B) and silently train on pairwise differences.
        yh = y_hat.reshape(-1)
        yt = batch["y"].reshape(-1).to(yh.dtype)
        if yh.shape != yt.shape:
            raise RuntimeError(f"loss shape mismatch: {tuple(y_hat.shape)} vs "
                               f"{tuple(batch['y'].shape)}")
        return {"y_hat": y_hat, "loss": F.binary_cross_entropy_with_logits(yh, yt)}

    @torch.no_grad()
    def scores(split):
        """[(sid, labels, scores)] for each subject, in fp32, chunked."""
        model.eval()
        out = []
        for lo, hi, y, sid in split:
            if len(y) == 0:
                continue
            xyz = torch.from_numpy(np.asarray(bundles[sid][3], dtype=np.float32)).to(dev)
            s = []
            for i in range(0, len(lo), args.eval_batch_size):
                b_lo = torch.from_numpy(lo[i:i + args.eval_batch_size]).float().to(dev)
                b_hi = torch.from_numpy(hi[i:i + args.eval_batch_size]).float().to(dev)
                o = forward(model, b_lo, b_hi,
                            xyz.unsqueeze(0).expand(b_lo.shape[0], -1, -1), sid)
                s.append(np.atleast_1d(o.squeeze(-1).float().cpu().numpy()))
            out.append((sid, np.asarray(y), np.concatenate(s)))
        return out

    def per_subject(sc):
        return {sid: (float(roc_auc_score(y, s)) if len(np.unique(y)) > 1 else float("nan"))
                for sid, y, s in sc}

    def pooled(sc):
        y = np.concatenate([t[1] for t in sc]) if sc else np.array([])
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, np.concatenate([t[2] for t in sc])))

    eval_every = max(1, int(args.eval_every))
    for ep in range(args.epochs):
        tr = train_one_epoch(model, tr_dl, opt, dev, cfg=cfg, step_fn=step_fn)
        sched.step()
        if not np.isfinite(tr["loss"]):
            raise RuntimeError(f"[ep {ep + 1}] non-finite training loss {tr['loss']}")
        if (ep + 1) % eval_every and ep + 1 != args.epochs:
            continue                              # patience counts evaluations
        sc = scores(val_eval)
        if args.select_metric == "per_subject":
            v = [a for a in per_subject(sc).values() if np.isfinite(a)]
            val = float(np.mean(v)) if v else float("nan")
        else:
            val = pooled(sc)                      # what the paper runs selected on
        stop = early.step(val if np.isfinite(val) else -1e18, model)
        print(f"    [ep {ep + 1}] loss={tr['loss']:.4f} val_auroc({args.select_metric})"
              f"={val:.4f}", flush=True)
        if stop:
            break
    early.restore(model)
    return per_subject(scores(te_eval))


# =========================================================================
# CLI
# =========================================================================

def _subject_key(s: str):
    try:
        return (0, int(s.split("_")[1]))
    except (IndexError, ValueError):
        return (1, s)


# How each command-line setting enters the results file name. Every parser
# option is in exactly one group (tests/test_btb_corteg.py checks this, so a new
# flag cannot be added without deciding how it is named):
#   NAME_FIELDS      always in the name
#   NOT_IN_NAME      cannot change the numbers: the output folder, a forced cache
#                    rebuild, and the block sizes of the feature transform and of
#                    (fp32) scoring, which bound memory only
#   NAMED_SPECIALLY  a fixed word or a derived tag, see setting_tags
#   VALUE_TAGS       prefix + value, whenever the value is not the default
NAME_FIELDS = ("train_mode", "merge_strategy", "endpoint", "seed")
NOT_IN_NAME = ("save_root", "no_cache", "event_chunk", "eval_batch_size")
NAMED_SPECIALLY = ("neg_mode", "no_pretrained", "per_subject_lora", "select_metric",
                   "subjects", "use_amp", "model_kwargs_json", "warmup_epochs")
VALUE_TAGS = {
    "event_seed": "es", "max_per_class": "n", "trial": "", "win_sec": "win",
    "pre_sec": "pre", "hga_low": "hfalo", "hga_high": "hfahi", "n_folds": "f",
    "val_frac": "val", "steegformer_variant": "", "layerwise_gate_bottleneck": "gatebn",
    "layerwise_gate_act": "gate", "lora_last_n": "lastn", "lora_r": "r",
    "lora_alpha": "alpha", "lora_dropout": "lorado", "head_dropout": "headdo",
    "epochs": "ep", "batch_size": "bs", "accum_iter": "accum", "lr": "lr",
    "weight_decay": "wd", "min_lr": "minlr", "max_norm": "clip",
    "eval_every": "evalevery", "patience": "pat",
}


def _same(x, y) -> bool:
    """Equal settings; numbers compare by value, so 3e-4 == 0.0003 and 4 == 4.0."""
    num = (int, float)
    if isinstance(x, num) and isinstance(y, num) and not isinstance(x, bool) \
            and not isinstance(y, bool):
        return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(y)))
    return x == y


def subject_tag(args) -> str:
    """'' for the paper's subjects in the paper's order, else e.g. 'subj3-9'.

    In pooled training a subject's position is its sid, which picks its LoRA
    adapter and its slot in the batch interleave, so the order is part of the
    run: the ten subjects in another order are named by that order. Per-subject
    training trains each subject alone, where only the set matters.
    """
    subs = list(args.subjects)
    if args.train_mode == "per_subject":
        if sorted(subs) == sorted(PAPER_SUBJECT_ORDER):
            return ""
        subs = sorted(subs, key=_subject_key)
    elif subs == PAPER_SUBJECT_ORDER:
        return ""
    return "subj" + "-".join(s.replace("sub_", "") for s in subs)


def _config_tag(args) -> str:
    """'' when the backbone config is the variant's own, else 'cfg-<name>'.

    Every real run passes a config, so the flag's default ('') cannot be the
    reference; the shipped configs/steegformer_<variant>.json is. It is
    compared by content, so a relative and an absolute path to it agree.
    """
    s = (getattr(args, "model_kwargs_json", "") or "").strip()
    if not s:
        return ""
    from experiments.common import safe_parse_model_kwargs
    try:
        got = safe_parse_model_kwargs(s)
        with open(default_backbone_config(args.steegformer_variant), encoding="utf-8") as fh:
            if got == json.load(fh):
                return ""
    except (OSError, ValueError):
        pass
    name = Path(s).stem if s.endswith(".json") else hashlib.sha1(s.encode()).hexdigest()[:8]
    return f"cfg-{name}"


def setting_tags(args) -> list:
    """One short tag per setting in which this run departs from the paper run.

    Empty at the paper's settings, whatever the endpoint, merge strategy and
    seed (which are always in the name). The defaults compared against are the
    parser's, which are the paper recipe; ``--warmup_epochs`` is compared with
    max(1, epochs // 10), the value its default stands for.
    """
    defaults = vars(build_parser().parse_args([]))
    tags = []
    if args.neg_mode != "upstream":
        tags.append(args.neg_mode)
    if args.no_pretrained:
        tags.append("randinit")
    if not args.per_subject_lora:
        tags.append("sharedlora")
    if args.select_metric != "pooled":
        tags.append("selpersubj")
    for dest in ("event_seed", "max_per_class"):
        if not _same(getattr(args, dest), defaults[dest]):
            tags.append(f"{VALUE_TAGS[dest]}{getattr(args, dest)}")
    sub = subject_tag(args)
    if sub:
        tags.append(sub)
    for dest, prefix in VALUE_TAGS.items():
        v = getattr(args, dest)
        if dest in ("event_seed", "max_per_class") or _same(v, defaults[dest]):
            continue
        tags.append(f"{prefix}{v:g}" if isinstance(v, float) else f"{prefix}{v}")
    wu = getattr(args, "warmup_epochs", None)
    if wu is not None and int(wu) != max(1, int(args.epochs) // 10):
        tags.append(f"warmup{int(wu)}")
    if not args.use_amp:
        tags.append("noamp")
    cfg = _config_tag(args)
    if cfg:
        tags.append(cfg)
    return tags


def result_filename(args) -> str:
    """``btb_<train_mode>_<merge>_<endpoint>[_<tag>...]_seed<S>.json``.

    At the paper's settings the three Table-3 arms are
    ``btb_pooled_layerwise_gate_<ep>``, ``btb_pooled_average_<ep>`` and
    ``btb_pooled_layerwise_gate_<ep>_randinit``. Any other setting adds a tag
    (``setting_tags``), except the four in NOT_IN_NAME, which cannot change the
    numbers; so two runs share a name only when they are the same experiment.
    """
    head = [args.train_mode, args.merge_strategy, args.endpoint]
    return "btb_" + "_".join(head + setting_tags(args)) + f"_seed{args.seed}.json"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # ---- data and task ----
    p.add_argument("--subjects", nargs="+", default=list(PAPER_SUBJECT_ORDER),
                   help="training order = sid; the default is the paper's order")
    p.add_argument("--trial", default=None,
                   help="default: each subject's scored trial (CANONICAL_TRIAL)")
    p.add_argument("--endpoint", default="sentence_onset",
                   choices=["sentence_onset", "word_nonword"],
                   help="sentence_onset = Task A (sentence-initial vs mid-sentence "
                        "word); word_nonword = Task B (word vs non-word)")
    p.add_argument("--neg_mode", default="upstream",
                   choices=["upstream", "short_silence"],
                   help="Task B negatives: upstream = any word-free tile (the "
                        "paper); short_silence = only pauses shorter than 10 s")
    p.add_argument("--event_seed", type=int, default=PAPER_EVENT_SEED,
                   help="seed of the balanced event draw. Independent of --seed: "
                        "the paper trains seeds 42, 1 and 2 on the event_seed-42 draw")
    p.add_argument("--win_sec", type=float, default=1.5,
                   help="Window width; also the overlap footprint used by the split")
    p.add_argument("--pre_sec", type=float, default=0.0,
                   help="Window start relative to the event (0 = anchored at onset)")
    p.add_argument("--max_per_class", type=int, default=PAPER_MAX_PER_CLASS)
    p.add_argument("--hga_low", type=float, default=70.0)
    p.add_argument("--hga_high", type=float, default=200.0)
    p.add_argument("--train_mode", default="pooled", choices=["pooled", "per_subject"],
                   help="pooled = ONE model over all subjects (what the paper reports); "
                        "per_subject = an independent model each, a different experiment")
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--event_chunk", type=int, default=200,
                   help="events per block in the feature transform (memory only)")

    # ---- model ----
    p.add_argument("--model_kwargs_json", type=str, default="",
                   help="ST-EEGFormer config; required unless --no_pretrained, and "
                        "defaulted to configs/steegformer_<variant>.json with it")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Random-init backbone (the ablation, not CORTEG)")
    p.add_argument("--steegformer_variant", default="small",
                   choices=["small", "base", "large"])
    p.add_argument("--merge_strategy", default="layerwise_gate",
                   choices=["average", "layerwise_gate"],
                   help="layerwise_gate = gated fusion (the paper's CORTEG row); "
                        "average = mean-pool fusion (Table 20 'mean-pool fusion')")
    p.add_argument("--layerwise_gate_bottleneck", type=int, default=16)
    p.add_argument("--layerwise_gate_act", default="tanh",
                   choices=["tanh", "sigmoid", "none"])
    p.add_argument("--lora_last_n", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.0)
    psl = p.add_mutually_exclusive_group()
    psl.add_argument("--per_subject_lora", dest="per_subject_lora", action="store_true",
                     help="(default) one LoRA adapter per subject, selected by sid")
    psl.add_argument("--shared_lora", dest="per_subject_lora", action="store_false",
                     help="one LoRA adapter shared by all subjects (not the paper)")
    p.set_defaults(per_subject_lora=True)

    # ---- training ----
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=32)
    p.add_argument("--accum_iter", type=int, default=4,
                   help="gradient-accumulation steps (effective batch 64)")
    amp = p.add_mutually_exclusive_group()
    amp.add_argument("--use_amp", dest="use_amp", action="store_true",
                     help="(default) bf16 autocast on CUDA; ignored on CPU")
    amp.add_argument("--no_amp", dest="use_amp", action="store_false")
    p.set_defaults(use_amp=True)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--warmup_epochs", type=int, default=None,
                   help="default: max(1, epochs // 10)")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--eval_every", type=int, default=2,
                   help="validate every N epochs and on the last; patience counts these")
    p.add_argument("--patience", type=int, default=15,
                   help="evaluations without improvement before stopping")
    p.add_argument("--select_metric", default="pooled", choices=["pooled", "per_subject"],
                   help="early-stopping metric: pooled val AUROC (the paper runs) or "
                        "the mean per-subject val AUROC, which is what is reported")
    p.add_argument("--seed", type=int, default=42,
                   help="training seed; does not change the events (see --event_seed)")
    p.add_argument("--save_root", default="")
    return p


def main(argv=None):
    from experiments.common import set_seed

    p = build_parser()
    args = p.parse_args(argv)

    if args.endpoint == "sentence_onset" and args.neg_mode != "upstream":
        p.error("--neg_mode applies to --endpoint word_nonword only")
    if args.eval_every < 1:
        p.error("--eval_every must be >= 1")
    if args.epochs < 1:
        p.error("--epochs must be >= 1: with none, the untrained model is scored")
    if args.n_folds < 1:
        p.error("--n_folds must be >= 1")
    if not 0.0 < args.val_frac < 1.0:
        p.error("--val_frac must be in (0, 1): early stopping needs the causal "
                "validation block")
    if len(set(args.subjects)) != len(args.subjects):
        p.error("--subjects lists a subject more than once")
    resolve_backbone_config(args)
    args.warmup_epochs = warmup_epochs_of(args)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_root = args.save_root or os.path.join(btb_output_root(), "runs")
    os.makedirs(save_root, exist_ok=True)
    dest = os.path.join(save_root, result_filename(args))
    print(f"[results] {dest}", flush=True)

    t0 = time.time()
    trials = {s: resolve_trial(s, args) for s in args.subjects}
    bundles = [extract_subject(s, args) for s in args.subjects]

    # One causal split per subject, from the events actually scored. Every fold
    # of every subject is re-checked and recorded, so each results file carries
    # its own evidence that no test window overlaps a fit or validation window.
    per_subj, splits = [], []
    for sid, (subj, b) in enumerate(zip(args.subjects, bundles)):
        if len(b[4]) != len(b[2]):
            raise RuntimeError(f"{subj}: {len(b[4])} event times for {len(b[2])} labels")
        f = forward_chaining_split(b[4], win_sec=args.win_sec,
                                   n_folds=args.n_folds, val_frac=args.val_frac)
        if len(f) != args.n_folds:
            raise SystemExit(f"{subj}: {len(f)} causal folds, wanted {args.n_folds} "
                             "— too few events for this embargo")
        for fi, (fit, va, te) in enumerate(f):
            rep = split_report(b[4], args.win_sec, fit, te, va, scheme="forward_chaining")
            if (rep["overlapping_train_test_pairs"] or rep.get("overlapping_train_val_pairs")
                    or rep.get("overlapping_val_test_pairs") or not rep["causal"]):
                raise AssertionError(f"{subj} fold {fi}: split leaks — {rep}")
            splits.append({"subject": subj, "sid": sid, "fold": fi, **rep})
        per_subj.append(f)
    print(f"[folds] {args.n_folds} causal folds x {len(bundles)} subjects, "
          f"embargo {EMBARGO_SEC} s, overlapping pairs=0", flush=True)

    if args.train_mode == "per_subject":
        # The pooled trainer with one subject: same recipe, one model each.
        per_fold_by_subj = {}
        for sid, subj in enumerate(args.subjects):
            per_fold_by_subj[subj] = [
                run_pooled_fold([bundles[sid]], [per_subj[sid][fi]], args, device)
                .get(0, float("nan")) for fi in range(args.n_folds)]
    else:
        per_fold = []
        for fi in range(args.n_folds):
            fold = [per_subj[sid][fi] for sid in range(len(bundles))]
            scores = run_pooled_fold(bundles, fold, args, device)
            per_fold.append(scores)
            print(f"  fold {fi}: " + "  ".join(
                f"{args.subjects[s]}={v:.4f}" for s, v in sorted(scores.items())), flush=True)
        per_fold_by_subj = {subj: [float(pf.get(sid, float("nan"))) for pf in per_fold]
                            for sid, subj in enumerate(args.subjects)}

    results = {}
    for subj in sorted(args.subjects, key=_subject_key):
        folds = [float(v) for v in per_fold_by_subj[subj]]
        vals = [v for v in folds if np.isfinite(v)]
        results[subj] = {"folds": folds,
                         "mean": float(np.mean(vals)) if vals else float("nan")}
        print(f"[{subj}] mean AUROC = {results[subj]['mean']:.4f}", flush=True)

    means = [r["mean"] for r in results.values() if np.isfinite(r["mean"])]
    cohort = float(np.mean(means)) if means else float("nan")
    cohort_sd = float(np.std(means, ddof=1)) if len(means) > 1 else float("nan")
    out = {
        "cohort_mean_auroc": cohort,
        "cohort_sd": cohort_sd,                    # across subjects, ddof=1
        # per_subject[s]["folds"] has one entry per fold, NaN where the subject's
        # test block held one class (json writes it as NaN, which Python reads).
        "per_subject": results,
        # The same numbers under the key names and meaning of the paper-run
        # records, so one aggregator reads both. Those list only the finite
        # folds. (On the full cohort no fold was NaN, so the two agree there.)
        "mean_auroc": cohort,
        "per_subject_auroc": {s: r["mean"] for s, r in results.items()},
        "per_subject_per_fold": {s: [v for v in r["folds"] if np.isfinite(v)]
                                 for s, r in results.items()},
        "subjects": list(args.subjects),           # training order; index = sid
        "trials": trials,
        "n_per_subject": {s: int(len(b[2])) for s, b in zip(args.subjects, bundles)},
        "endpoint": args.endpoint,
        "neg_mode": args.neg_mode,
        "event_seed": int(args.event_seed),
        "seed": int(args.seed),
        "arm": "random_init" if args.no_pretrained else "corteg",
        "merge_strategy": args.merge_strategy,
        "train_mode": args.train_mode,
        "per_subject_lora": bool(args.per_subject_lora),
        "select_metric": args.select_metric,
        "embargo_sec": float(EMBARGO_SEC),
        "splits": splits,
        "args": vars(args),
        "elapsed_s": time.time() - t0,
    }
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"cohort mean AUROC over {len(means)} subjects = {cohort:.4f} "
          f"(SD {cohort_sd:.4f})")
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
