"""Intracranial foundation models on BrainTreebank: the frozen arms.

Scores BrainBERT, the Population Transformer (PopT) and Brant as FROZEN feature
extractors on both BrainTreebank endpoints -- Task A, sentence onset, and Task B,
word vs non-word -- the way the paper's FM rows were produced.

What is shared with CORTEG, and what is not. Shared: the event construction
(same transcripts, same class balancing, same fixed event seed), the
forward-chaining test folds and the 7 s embargo. Not shared:

  * Electrodes. PopT's clean_laplacian list, intersected with the per-subject
    voxel table (depth-wm.csv), which is what the published FM numbers used.
    CORTEG intersects the same list with the MNI table instead. The MNI set is
    a strict subset; the two differ on sub_3 (91 vs 82), sub_8 (121 vs 120),
    sub_9 (66 vs 64) and sub_10 (159 vs 157).
  * Readout. A StandardScaler + class-balanced logistic probe, fitted on all
    embargoed history (--val_frac 0). CORTEG carves a 15% causal validation
    block and trains its head.
  * Input. Each model gets its own native input, because a foundation-model
    comparison means nothing otherwise; that preprocessing lives in ieeg_fm.py.
    BrainBERT and PopT read the task window as 2048 Hz spectrograms: [t, t+1.5]
    on Task A, [t-2.5, t+2.5] on Task B. Brant's unit is a fixed 6 s patch at
    250 Hz, longer than either window, so it reads the 6 s ending at the task
    window's right edge: [t-4.5, t+1.5] on Task A and [t-3.5, t+2.5] on Task B.
    On Task A that is an exception to the post-onset rule every other arm
    follows: Brant sees 4.5 s of PRE-ONSET signal, including the pause before a
    sentence. A pre-onset-only Brant patch, [t-6, t], scores 0.540 +/- 0.009 on
    single_elec_max -- close to that arm's null (at least ~0.53, see below) and
    well below the full patch's 0.587 -- so the pause contributes at most a
    small part of Brant's Task A number.

Arms (``--arm``), matching the names in the paper's arm matrix:

  single_elec_max   fit an independent probe per electrode, average its AUROC
                    over the folds, then take the MAXIMUM over electrodes.
                    This is the paper's BrainBERT-dagger and Brant-dagger row.
                    It is an ORACLE: the winning electrode is chosen on the very
                    folds that score it, so its null is not 0.50. Permuting the
                    labels of a 1-D high-gamma feature through the same
                    max-over-electrodes search gives a null of ~0.53, and that
                    is a LOWER BOUND: these 768-d (BrainBERT) and 2048-d
                    (Brant) probes select more optimistically, and their own
                    null was not measured. It grows with electrode count.
                    Reported as published, with the caveat, rather than dropped.
  single_elec_mean  the same probes averaged instead of maximised -- no oracle.
  pop_meanpool      average embeddings over electrodes first, then one probe.
                    PopT is a population model and has only this arm.

``--arm`` takes several arms at once, and each is written to its own result
file. They share one embedding load, and single_elec_max and single_elec_mean
share one set of per-electrode probes -- C electrodes x 4 folds logistic fits,
by far the most expensive CPU step -- so all three arms together cost about
what single_elec_max alone does. The numbers are those of separate calls.

Paper rows (the FM rows are one seed, 42):

  Table 3 and 9, BrainBERT-dagger / Brant-dagger    --arm single_elec_max
  Table 20, single-elec. mean / population mean-pool  single_elec_mean / pop_meanpool
  Table 20, "PopT, frozen probe"                    --fm popt
  Table 3 and 9, PopT (0.600 / 0.779)               NOT here: that row is the LoRA
      fine-tune in experiments/run_popt_finetune_btb.py, which also gives
      Table 20's PopT full fine-tune and head-only rows.
  Table 20, BrainBERT and Brant head-only / LoRA / full fine-tune: not released.

Caches. Embeddings are cached per subject under
``<output root>/braintreebank/cache/`` as
``fm_{fm}_{endpoint}_{subj}_{trial}_win{w}_pre{p}_n{max_per_class}_s{event_seed}.npz``
(keys: emb, y, event_times, electrodes, fs, endpoint, win_sec, pre_sec,
event_seed). The BrainBERT cache holds per-electrode (N, C, 768) embeddings and
is the input of both ``--fm popt`` and run_popt_finetune_btb.py, exactly as the
paper's PopT arms consumed the frozen BrainBERT arm's embeddings. Brant's
resampled 250 Hz recordings are cached under ``<output root>/braintreebank/
brant_stream_cache/``.

Memory and exactness. BrainBERT is run in event chunks (--event_chunk 200, as
the paper runs were) and ieeg_fm.brainbert_embeddings pools each forward batch
as it returns; both are numerically exact. The largest subject on Task B
(sub_7, 191 electrodes) then needs about 15 GB, almost all of it the raw
windows; unchunked and unpooled it needed several hundred. Brant resamples
each whole recording once at the exact measured rate and cuts its patches from
that stream, as the paper runs did; resampling each window separately
(ieeg_fm.brant_embeddings) only approximates it.

GPU and CPU stages. Only the embedding pass needs a model (and a GPU); the
probes are CPU work. ``--embed_only`` builds the caches and fits nothing;
``--from_cache`` fits the probes from existing caches, stops with an error if
one is missing, and never loads a model. scripts/table3_ieeg_fm_braintreebank.sh
uses the two to keep the per-electrode probes off the GPU.

Result files. ``<fm>_<endpoint>_<arm>_seed<seed>.json`` under --save_root
(default ``<output root>/braintreebank/fm_runs``). A setting that differs from
the paper run's default adds a tag, for example
``brainbert_sentence_onset_single_elec_max_seed42_n50_sub_9.json`` for
``--max_per_class 50 --subjects sub_9``, so a smoke or partial run never
overwrites a full one.

Seeds. --event_seed (default 42) draws the events and keys the cache; --seed
only seeds the probe. The paper drew its events once, with seed 42.

Third-party code and weights are not redistributed -- see ieeg_fm.py for the
environment variables and download sources.

Example:

    python -m experiments.run_ieeg_fm_baselines \\
        --fm brainbert --endpoint sentence_onset --arm single_elec_max
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from types import SimpleNamespace

import numpy as np

import ieeg_fm
from data.braintreebank import (
    EMBARGO_SEC,
    assert_valid_times,
    btb_output_root,
    btb_root,
    build_time_to_sample,
    estimate_fs,
    extract_windows,
    forward_chaining_split,
    load_electrode_map,
    load_events,
    load_localization,
    split_report,
    word_nonword_events,
)
from experiments.run_btb_classification import clean_electrodes, movie_of, trial_of

FMS = ("brainbert", "popt", "brant")

# Task A anchors the window at the event; Task B centres a 5 s window, which is
# also why its overlap footprint is wider.
ENDPOINTS = {
    "sentence_onset": {"win_sec": 1.5, "pre_sec": 0.0},
    "word_nonword": {"win_sec": 5.0, "pre_sec": -2.5},
}

# Brant's atomic unit is a 6 s patch at 250 Hz, longer than either task window.
# It is given its native context instead: the 6 s ENDING AT THE TASK WINDOW'S
# RIGHT EDGE, so that no arm reads further into the future than any other:
# [t-4.5, t+1.5] on Task A and [t-3.5, t+2.5] on Task B. These are the windows
# the paper's Brant numbers were computed on. Its overlap footprint is the patch
# plus the resampler's filter edge, 6.11 s, and that is what sizes the 7 s
# embargo every arm shares.
BRANT_PATCH_SEC = ieeg_fm.BRANT_PATCH_LEN / ieeg_fm.BRANT_FS          # 6.0
BRANT_WINDOW = {
    ep: {"win_sec": BRANT_PATCH_SEC,
         "pre_sec": w["pre_sec"] + w["win_sec"] - BRANT_PATCH_SEC}
    for ep, w in ENDPOINTS.items()
}
BRANT_FOOTPRINT_SEC = BRANT_PATCH_SEC + 0.11     # + resample FIR edge, measured

# Forward batch sizes of the paper runs: BrainBERT 256 spectrograms, Brant 16
# windows (--batch_size overrides these two), PopT 64 windows (fixed). Batching
# does not change the maths, but on a GPU it can move the last bits, so the
# defaults match.
DEFAULT_BATCH = {"brainbert": 256, "brant": 16}
POPT_BATCH = 64

PER_ELECTRODE_ARMS = ("single_elec_max", "single_elec_mean")
ARMS = PER_ELECTRODE_ARMS + ("pop_meanpool",)


def window_for(fm: str, endpoint: str) -> dict:
    """The window an arm reads: its start (pre_sec) and length (win_sec)."""
    return BRANT_WINDOW[endpoint] if fm == "brant" else ENDPOINTS[endpoint]


def footprint_for(fm: str, endpoint: str) -> float:
    """The arm's overlap footprint in seconds, which its split is checked at."""
    return BRANT_FOOTPRINT_SEC if fm == "brant" else ENDPOINTS[endpoint]["win_sec"]


def cache_path(subj: str, trial: str, args) -> str:
    """Where one subject's embeddings for `args.fm` / `args.endpoint` live."""
    ep = window_for(args.fm, args.endpoint)
    # event_seed and max_per_class choose WHICH events are drawn, and the window
    # (start and length) what is read; all of them key the cache. The probe
    # --seed changes neither, so it is deliberately absent: the paper scores
    # every seed on one event draw.
    tag = (f"fm_{args.fm}_{args.endpoint}_{subj}_{trial}"
           f"_win{ep['win_sec']}_pre{ep['pre_sec']}"
           f"_n{args.max_per_class}_s{args.event_seed}.npz")
    return os.path.join(btb_output_root(), "cache", tag)


def _events(root, subj, trial, movie, args):
    """(event_times, labels) for the endpoint, drawn with the fixed event seed."""
    if args.endpoint == "sentence_onset":
        return load_events(root, movie, args.max_per_class, args.event_seed)
    starts, y, _ = word_nonword_events(root, subj, trial, movie,
                                       args.max_per_class, args.event_seed)
    return starts, y


def _device(requested: str) -> str:
    import torch
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("[device] CUDA is not available; running on CPU", flush=True)
        return "cpu"
    return requested


def _chunks(n: int, step: int):
    step = step if step and step > 0 else n
    return [(c0, min(c0 + step, n)) for c0 in range(0, n, step)]


def brant_stream(root, subj, trial, ch_idx, fs):
    """The whole recording at 250 Hz, (T250, C), cached on disk.

    Resampling a recording at its exact rate is deterministic but not cheap, and
    every Brant run on a subject reads the same stream, so it is computed once.
    The key covers everything that changes the output: subject, trial, measured
    rate and the exact electrode list (the columns are in ch_idx order). Set
    BRANT_STREAM_CACHE=0 to recompute without reading or writing the cache.
    """
    import h5py

    cdir = os.path.join(btb_output_root(), "brant_stream_cache")
    key = hashlib.sha1(f"{subj}|{trial}|{float(fs):.6f}|{','.join(map(str, ch_idx))}"
                       .encode()).hexdigest()[:16]
    cpath = os.path.join(cdir, f"stream250_{subj}_{trial}_C{len(ch_idx)}_{key}.npy")
    use_cache = os.environ.get("BRANT_STREAM_CACHE", "1") == "1"
    if use_cache and os.path.exists(cpath):
        arr = np.load(cpath)
        if arr.ndim == 2 and arr.shape[1] == len(ch_idx) and arr.dtype == np.float32:
            print(f"[cache] {os.path.basename(cpath)} {arr.shape}", flush=True)
            return arr
        print(f"[cache] {os.path.basename(cpath)} has shape {arr.shape}; recomputing",
              flush=True)
    t0 = time.time()
    with h5py.File(f"{root}/all_subject_data/{subj}_{trial}.h5", "r") as h5:
        n_samples = h5["data/electrode_0"].shape[0]
        stream = ieeg_fm.brant_stream_at_250(
            lambda i: h5[f"data/electrode_{ch_idx[i]}"][:], len(ch_idx), n_samples, fs)
    print(f"[{subj}] 250 Hz stream {stream.shape} in {time.time() - t0:.0f}s", flush=True)
    if use_cache:
        os.makedirs(cdir, exist_ok=True)
        tmp = cpath + f".tmp{os.getpid()}.npy"   # must end in .npy, or np.save renames it
        np.save(tmp, stream)
        os.replace(tmp, cpath)                  # atomic: no reader sees a partial file
    return stream


def load_subject(subj: str, args) -> dict:
    """One subject's embeddings for `args.fm`, built or read from the cache.

    Returns a dict with emb ((N, C, D) for BrainBERT and Brant, (N, 512) for
    PopT), y, event_times, electrodes (names, in column order), fs and trial.

    `args` needs fm, endpoint, trial, max_per_class, event_seed, event_chunk,
    batch_size, device and no_cache; `from_cache` is optional (default False).
    """
    root = btb_root()
    trial = args.trial or trial_of(root, subj)
    cache = cache_path(subj, trial, args)
    if os.path.exists(cache) and not args.no_cache:
        d = np.load(cache, allow_pickle=False)
        print(f"[cache] {os.path.basename(cache)}", flush=True)
        return {"emb": d["emb"], "y": d["y"], "event_times": d["event_times"],
                "electrodes": [str(e) for e in d["electrodes"]],
                "fs": float(d["fs"]), "trial": trial}
    if getattr(args, "from_cache", False):
        # The CPU probe stage must never fall back to running a model: on a
        # machine without the GPU that would take hours and give embeddings
        # that differ in the last bits from the ones the GPU stage writes.
        raise FileNotFoundError(
            f"{subj}: --from_cache, but there is no {args.fm} embedding cache at "
            f"{cache}. Build it first: the same call with --embed_only instead.")

    movie = movie_of(root, subj, trial)
    name2idx, _ = load_electrode_map(root, subj)
    # The FM arms select and re-reference in the per-subject voxel frame, which
    # is what the published numbers used; CORTEG uses MNI. The sets differ.
    loc = load_localization(root, subj)
    use = [n for n in clean_electrodes(subj) if n in name2idx and n in loc]
    ch_idx = [name2idx[n] for n in use]
    # PopT's native integer L/I/P voxel coordinates. BrainBERT uses them only to
    # find each electrode's nearest neighbours for its Laplacian re-reference.
    lip = np.array([loc[n] for n in use], dtype=np.float64)

    fs = estimate_fs(root, subj, trial)          # measured, never assumed
    t2s = build_time_to_sample(root, subj, trial)
    starts, y = _events(root, subj, trial, movie, args)
    w = ENDPOINTS[args.endpoint]
    batch = args.batch_size or DEFAULT_BATCH.get(args.fm)
    device = _device(args.device)
    print(f"[{subj}] {trial} {movie} fs={fs:.1f}Hz electrodes={len(use)} "
          f"events={len(starts)}", flush=True)

    if args.fm == "brainbert":
        x_raw, valid = extract_windows(root, subj, trial, ch_idx, starts, t2s, fs,
                                       w["pre_sec"], w["win_sec"])
        model = ieeg_fm.load_brainbert(device=device)
        parts = []
        for c0, c1 in _chunks(len(x_raw), args.event_chunk):
            parts.append(ieeg_fm.brainbert_embeddings(
                x_raw[c0:c1], fs=fs, xyz_mm=lip, device=device, reref="laplacian_xyz",
                pool="default", batch_size=batch, model=model).astype(np.float32))
        del x_raw
        emb = np.concatenate(parts, axis=0)
    elif args.fm == "popt":
        # PopT is defined over frozen BrainBERT embeddings: read (or build) the
        # BrainBERT arm's cache rather than running BrainBERT a second time. The
        # paper's PopT arm fed PopT the BrainBERT arm's embeddings in the same way.
        # batch_size=None: the BrainBERT cache is shared by every PopT arm and its
        # key does not include the batch, so it is always built at BrainBERT's
        # own default, whatever --batch_size says.
        bb = load_subject(subj, SimpleNamespace(**{**vars(args), "fm": "brainbert",
                                                   "batch_size": None}))
        if bb["electrodes"] != use:
            raise RuntimeError(f"{subj}: the BrainBERT cache has a different electrode "
                               f"list than the one PopT's coordinates are built for")
        model = ieeg_fm.load_popt_model(device=device)
        parts = []
        for c0, c1 in _chunks(len(bb["emb"]), args.event_chunk):
            parts.append(ieeg_fm.popt_embeddings(
                None, lip, coords_are_mni_mm=False, device=device, batch_size=POPT_BATCH,
                per_electrode_embeddings=bb["emb"][c0:c1], popt_model=model))
        emb = np.concatenate(parts, axis=0)
        y, ev_t = bb["y"], bb["event_times"]
    else:
        # The paper's Brant arm: validity is decided on the TASK window, as for
        # every other arm, so Brant scores the same events; its patch is then
        # cut from the whole-recording 250 Hz stream, ending at the window's
        # right edge.
        import h5py
        c = np.asarray(t2s(starts), dtype=np.float64)
        in_range = np.isfinite(c)                # mask BEFORE casting NaN to int
        centers = np.where(in_range, c, 0.0).astype(np.int64)
        s0 = centers + int(round(w["pre_sec"] * fs))
        with h5py.File(f"{root}/all_subject_data/{subj}_{trial}.h5", "r") as h5:
            n_samples = h5["data/electrode_0"].shape[0]
        valid = in_range & (s0 >= 0) & (s0 + int(round(w["win_sec"] * fs)) <= n_samples)
        right = centers + int(round((w["pre_sec"] + w["win_sec"]) * fs))
        stream = brant_stream(root, subj, trial, ch_idx, fs)
        ends = np.clip(ieeg_fm.brant_right_edges_250(right, fs)[valid], 1, stream.shape[0])
        emb = ieeg_fm.brant_stream_embeddings(stream, ends, device=device,
                                              batch_size=batch, n_patches=1)
        del stream

    if args.fm != "popt":
        y, ev_t = y[valid], starts[valid]
    emb = np.asarray(emb, dtype=np.float32)
    ep = window_for(args.fm, args.endpoint)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    tmp = cache + f".tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, emb=emb, y=y, event_times=ev_t,
                        electrodes=np.asarray(use, dtype=str), fs=np.float64(fs),
                        endpoint=np.str_(args.endpoint), win_sec=np.float64(ep["win_sec"]),
                        pre_sec=np.float64(ep["pre_sec"]),
                        event_seed=np.int64(args.event_seed))
    os.replace(tmp, cache)
    print(f"[cache] wrote {os.path.basename(cache)}  {emb.shape}", flush=True)
    return {"emb": emb, "y": y, "event_times": ev_t, "electrodes": list(use),
            "fs": float(fs), "trial": trial}


def probe_auroc(x_tr, y_tr, x_te, y_te, seed):
    """One logistic probe, as the published arms used.

    Scored on predict_proba like the paper runs. The decision function ranks
    identically in exact arithmetic, but the sigmoid saturates to 1.0 in
    floating point for large margins, which can create ties the decision
    function does not have.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")
    sc = StandardScaler().fit(x_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                             random_state=seed)
    clf.fit(sc.transform(x_tr), y_tr)
    return float(roc_auc_score(y_te, clf.predict_proba(sc.transform(x_te))[:, 1]))


def score_arms(emb, y, folds, arms, seed, n_jobs: int = 1) -> dict:
    """{arm: (AUROC, per-electrode AUROCs or None)} for one subject.

    single_elec_max and single_elec_mean are two reductions of the SAME
    per-electrode probes: C electrodes x len(folds) logistic fits, the most
    expensive CPU step of this runner (the paper's Brant arm records 62 min for
    C=91 serially, on 2048-d features). They are fitted once, however many of
    the two arms are asked for, and each arm's result is what scoring it alone
    gives.
    """
    def fold_mean(feats):
        vals = [probe_auroc(feats[tr], y[tr], feats[te], y[te], seed)
                for tr, _, te in folds]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    out = {}
    if "pop_meanpool" in arms:
        # PopT already returns one population vector per window, (N, D); the
        # per-electrode models return (N, C, D) and need pooling first.
        feats = emb if emb.ndim == 2 else emb.mean(axis=1)
        out["pop_meanpool"] = (fold_mean(feats), None)
    wanted = [a for a in arms if a in PER_ELECTRODE_ARMS]
    if wanted:
        if emb.ndim != 3:
            raise ValueError(f"{wanted[0]} needs per-electrode embeddings (N, C, D); "
                             f"got {emb.shape}")
        if n_jobs == 1:
            per_elec = [fold_mean(emb[:, c, :]) for c in range(emb.shape[1])]
        else:
            # The electrodes are independent, so this only changes the wall time.
            from joblib import Parallel, delayed
            per_elec = Parallel(n_jobs=n_jobs)(
                delayed(fold_mean)(emb[:, c, :]) for c in range(emb.shape[1]))
        per_elec = np.asarray(per_elec, dtype=np.float64)
        reduce = {"single_elec_max": np.nanmax, "single_elec_mean": np.nanmean}
        for a in wanted:
            out[a] = (float(reduce[a](per_elec)), per_elec)
    unknown = [a for a in arms if a not in out]
    if unknown:
        raise ValueError(f"unknown arm(s) {unknown}; expected some of {ARMS}")
    return out


def score_subject(emb, y, folds, arm, seed, n_jobs: int = 1):
    """AUROC for one subject under one arm, and the per-electrode AUROCs if any."""
    return score_arms(emb, y, folds, [arm], seed, n_jobs=n_jobs)[arm]


# Settings that change neither which events are scored nor how they are scored
# (or that are already in the file name). Every other setting that differs from
# its default is tagged onto the result file name.
NAME_NEUTRAL = {"fm", "endpoint", "arm", "mode", "seed", "device", "save_root",
                "batch_size", "eval_batch_size", "event_chunk", "n_jobs", "no_cache",
                "embed_only", "from_cache"}


def result_tags(args, parser, neutral=NAME_NEUTRAL) -> list:
    """Short tags for every setting that differs from the paper run's default.

    Without them a smoke or partial run (--subjects sub_9, --max_per_class 50,
    --n_folds 2) writes the full run's file name and silently overwrites it.
    The paper settings give no tag, so a paper-setting run keeps the plain name.
    The subject list is order-free; any subset is spelled out, so per-subject
    runs can be written side by side and read together by the aggregator.
    """
    def natural(s):
        return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]

    tags = []
    for k, v in sorted(vars(args).items()):
        if k in neutral:
            continue
        default = parser.get_default(k)
        if k == "subjects":
            if sorted(v) != sorted(default):
                tags.append("+".join(sorted(v, key=natural)))
        elif v == default:
            continue
        elif k == "max_per_class":
            tags.append(f"n{v}")
        elif k == "event_seed":
            tags.append(f"es{v}")
        elif k == "trial":
            tags.append(str(v))
        elif isinstance(v, bool):
            tags.append(k if v else f"no_{k}")
        else:
            tags.append(f"{k}{v}")
    return [re.sub(r"[^A-Za-z0-9_.+-]", "-", t) for t in tags]


def result_name(stem: str, seed: int, tags) -> str:
    """``<stem>_seed<seed>[_<tag>...].json``."""
    return f"{stem}_seed{seed}" + "".join(f"_{t}" for t in tags) + ".json"


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fm", required=True, choices=["brainbert", "popt", "brant"])
    p.add_argument("--endpoint", default="sentence_onset",
                   choices=["sentence_onset", "word_nonword"])
    p.add_argument("--arm", nargs="+", default=None, metavar="ARM",
                   choices=["single_elec_max", "single_elec_mean", "pop_meanpool"],
                   help="one or more of single_elec_max, single_elec_mean, "
                        "pop_meanpool; one result file each, and the per-electrode "
                        "probes are fitted once for both single_elec arms. Default: "
                        "pop_meanpool for popt, single_elec_max otherwise")
    p.add_argument("--subjects", nargs="+",
                   default=[f"sub_{i}" for i in range(1, 11)])
    p.add_argument("--trial", default=None)
    p.add_argument("--max_per_class", type=int, default=900)
    p.add_argument("--event_seed", type=int, default=42,
                   help="draws the events and keys the cache; the paper used 42 "
                        "for every run. --seed does not change the events")
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="Frozen probes need no validation block; 0 keeps all "
                        "history for fitting")
    p.add_argument("--batch_size", type=int, default=None,
                   help="BrainBERT and Brant forward batch; default the paper "
                        "runs' (BrainBERT 256 spectrograms, Brant 16 windows). "
                        "PopT always runs 64 windows, and the BrainBERT cache "
                        "PopT reads is always built at 256")
    p.add_argument("--event_chunk", type=int, default=200,
                   help="run BrainBERT and PopT this many events at a time; bounds "
                        "memory and is numerically exact. 0 = all at once")
    p.add_argument("--n_jobs", type=int, default=1,
                   help="parallel per-electrode probes (joblib); numbers unchanged")
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_cache", action="store_true",
                   help="recompute and overwrite the embedding caches this run "
                        "reads (for popt, the BrainBERT cache too)")
    p.add_argument("--embed_only", action="store_true",
                   help="build any missing embedding caches, then stop: no probe "
                        "is fitted and no result file is written (the GPU stage)")
    p.add_argument("--from_cache", action="store_true",
                   help="fit the probes from existing embedding caches only; a "
                        "missing cache is an error, and no model or GPU is used "
                        "(the CPU stage)")
    p.add_argument("--seed", type=int, default=42,
                   help="the probe's random_state; does not change the events")
    p.add_argument("--save_root", default="")
    args = p.parse_args()
    default_arm = "pop_meanpool" if args.fm == "popt" else "single_elec_max"
    arms = list(dict.fromkeys(args.arm or [default_arm]))      # ordered, no repeats
    if args.fm == "popt" and arms != ["pop_meanpool"]:
        p.error("PopT is a population model: it has one vector per window, so "
                "only --arm pop_meanpool applies")
    if args.from_cache and (args.no_cache or args.embed_only):
        p.error("--from_cache reads the caches; it cannot be combined with "
                "--no_cache or --embed_only")
    args.arm = arms

    if args.embed_only:
        for subj in args.subjects:
            trial = args.trial or trial_of(btb_root(), subj)
            cache = cache_path(subj, trial, args)
            if os.path.exists(cache) and not args.no_cache:
                print(f"[cache] {os.path.basename(cache)} exists", flush=True)
                continue
            d = load_subject(subj, args)
            print(f"[{subj}] {args.fm} embeddings {tuple(np.shape(d['emb']))} cached",
                  flush=True)
            del d
        print("--embed_only: embeddings cached; no probe fitted, no result written")
        return

    footprint = footprint_for(args.fm, args.endpoint)
    save_root = args.save_root or os.path.join(btb_output_root(), "fm_runs")
    os.makedirs(save_root, exist_ok=True)

    results, t0 = {a: {} for a in arms}, time.time()
    for subj in args.subjects:
        d = load_subject(subj, args)
        emb, y = d["emb"], d["y"]
        ev_t = assert_valid_times(d["event_times"], n_expected=len(y))
        folds = forward_chaining_split(ev_t, win_sec=footprint, n_folds=args.n_folds,
                                       embargo_sec=EMBARGO_SEC, val_frac=args.val_frac)
        folds = [(f[0], None, f[-1]) for f in folds]        # (train, _, test)
        splits = []
        for fi, (tr, _, te) in enumerate(folds):
            rep = split_report(ev_t, footprint, tr, te, scheme="forward_chaining")
            if rep["overlapping_train_test_pairs"] or not rep["causal"]:
                raise AssertionError(f"{subj} fold {fi}: split leaks — {rep}")
            splits.append(rep)
        scored = score_arms(emb, y, folds, arms, args.seed, n_jobs=args.n_jobs)
        for arm in arms:
            score, per_elec = scored[arm]
            results[arm][subj] = {"auroc": score, "trial": d["trial"],
                                  "n_electrodes": len(d["electrodes"]),
                                  "n_events": int(len(y)), "splits": splits}
            if per_elec is not None:
                results[arm][subj]["per_electrode_auroc"] = [float(v) for v in per_elec]
            print(f"[{subj}] {arm} AUROC = {score:.4f} "
                  f"({len(d['electrodes'])} electrodes)", flush=True)
        del d, emb

    tags = result_tags(args, p)
    elapsed = time.time() - t0
    for arm in arms:
        vals = [r["auroc"] for r in results[arm].values() if not np.isnan(r["auroc"])]
        # Each file reads as a single-arm run: "arm" and args["arm"] name this arm.
        out = {"fm": args.fm, "endpoint": args.endpoint, "arm": arm,
               "window": window_for(args.fm, args.endpoint), "footprint_sec": footprint,
               "cohort_mean_auroc": float(np.mean(vals)) if vals else None,
               # An SD needs two subjects; one is recorded as null, not as 0.
               "cohort_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
               "per_subject": results[arm], "args": {**vars(args), "arm": arm},
               "name_tags": tags, "elapsed_s": elapsed}
        if arm == "single_elec_max":
            out["caveat"] = (
                "single_elec_max selects the best electrode on the same folds that "
                "score it, so its null is not 0.50: it is at least ~0.53, a lower "
                "bound measured by permuting the labels of a 1-D high-gamma feature "
                "through the same max-over-electrodes search. The null of these "
                "higher-dimensional probes is higher and was not measured, and it "
                "rises with electrode count. Compare against single_elec_mean.")
        dest = os.path.join(
            save_root, result_name(f"{args.fm}_{args.endpoint}_{arm}", args.seed, tags))
        with open(dest, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        sd = "n/a" if out["cohort_sd"] is None else f"{out['cohort_sd']:.4f}"
        mean = "n/a" if out["cohort_mean_auroc"] is None else f"{out['cohort_mean_auroc']:.4f}"
        print(f"\n[{arm}] cohort mean AUROC = {mean} ± {sd} over {len(vals)} subjects")
        print(f"written: {dest}")


if __name__ == "__main__":
    main()
