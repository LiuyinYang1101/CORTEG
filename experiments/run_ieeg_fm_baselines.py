"""Intracranial foundation models on the BrainTreebank benchmark.

Scores BrainBERT, the Population Transformer and Brant as FROZEN feature
extractors under the same causal protocol as CORTEG: the same events, the same
forward-chaining folds, the same 7 s embargo. Only the encoder changes.

Each model is fed its own native input, because a foundation-model comparison
means nothing otherwise -- BrainBERT and PopT take 2048 Hz spectrograms, Brant
takes 250 Hz patches. That preprocessing lives in ieeg_fm.py.

Arms (``--arm``), matching the names in the paper's arm matrix:

  single_elec_max   fit an independent probe per electrode, average its AUROC
                    over the folds, then take the MAXIMUM over electrodes.
                    This is the paper's headline row for BrainBERT and Brant.
                    It is an ORACLE: the winning electrode is chosen on the very
                    folds that score it, so its null is ~0.53, not 0.50, and the
                    inflation grows with electrode count. Reported as published,
                    with the caveat, rather than quietly dropped.
  single_elec_mean  the same probes averaged instead of maximised -- no oracle.
  pop_meanpool      average embeddings over electrodes first, then one probe.

The published PopT row (Task B, 0.779) came from a LoRA fine-tune, not a frozen
probe; this runner covers the frozen arms only. See README.

Third-party code and weights are not redistributed -- see ieeg_fm.py for the
environment variables and download sources.

Example:

    python -m experiments.run_ieeg_fm_baselines \\
        --fm brainbert --endpoint sentence_onset --arm single_elec_max
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

import ieeg_fm
import paths
from data.braintreebank import (
    assert_brant_fs_fixed,
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

# Task A anchors the window at the event; Task B centres a 5 s window, which is
# also why its overlap footprint is wider.
ENDPOINTS = {
    "sentence_onset": {"win_sec": 1.5, "pre_sec": 0.0},
    "word_nonword": {"win_sec": 5.0, "pre_sec": -2.5},
}

# Brant's atomic unit is a 6 s patch at 250 Hz, longer than either task window, so
# it is given its native context instead: [t-4.5, t+1.5]. That is the paper's
# configuration, and its 6.11 s footprint (including the resampler's filter edge)
# is what sizes the 7 s embargo everything else also uses.
BRANT_WINDOW = {"win_sec": 6.0, "pre_sec": -4.5}


def window_for(fm: str, endpoint: str) -> dict:
    """The window an arm reads, and therefore its overlap footprint."""
    return BRANT_WINDOW if fm == "brant" else ENDPOINTS[endpoint]


def subject_embeddings(subj: str, args):
    """(embeddings, labels, event_times) for one subject, cached as an npz.

    Embeddings are (N, C, D) for the per-electrode models. The cache name carries
    the model, endpoint and window, because a cache built under different
    settings is silently wrong for every number downstream.
    """
    ep = window_for(args.fm, args.endpoint)
    root = btb_root()
    trial = args.trial or trial_of(root, subj)
    tag = f"fm_{args.fm}_{args.endpoint}_{subj}_{trial}_win{ep['win_sec']}.npz"
    cache = os.path.join(btb_output_root(), "cache", tag)
    if os.path.exists(cache) and not args.no_cache:
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] {tag}", flush=True)
        return d["emb"], d["y"], d["event_times"]

    movie = movie_of(root, subj, trial)
    name2idx, _ = load_electrode_map(root, subj)
    # The FM arms select and re-reference in the per-subject voxel frame, which
    # is what the published numbers used; CORTEG uses MNI. The sets differ.
    loc = load_localization(root, subj)
    use = [n for n in clean_electrodes(subj) if n in name2idx and n in loc]
    ch_idx = [name2idx[n] for n in use]
    xyz = np.array([loc[n] for n in use], dtype=np.float64)

    fs = estimate_fs(root, subj, trial)          # measured, never assumed
    if args.fm == "brant":
        # sub_9 runs at ~1019 Hz; assuming 2048 doubles Brant's true footprint
        # past the embargo without tripping the overlap check.
        assert_brant_fs_fixed(fs)
    t2s = build_time_to_sample(root, subj, trial)

    if args.endpoint == "sentence_onset":
        starts, y = load_events(root, movie, args.max_per_class, args.seed)
    else:
        starts, y, _ = word_nonword_events(root, subj, trial, movie,
                                           args.max_per_class, args.seed)
    print(f"[{subj}] {trial} {movie} fs={fs:.1f}Hz electrodes={len(use)} "
          f"events={len(starts)}", flush=True)

    x_raw, valid = extract_windows(root, subj, trial, ch_idx, starts, t2s, fs,
                                   ep["pre_sec"], ep["win_sec"])
    y, ev_t = y[valid], starts[valid]

    if args.fm == "brainbert":
        emb = ieeg_fm.brainbert_embeddings(
            x_raw, fs=fs, xyz_mm=xyz, device=args.device,
            reref="laplacian_xyz", pool="default", batch_size=args.batch_size)
    elif args.fm == "brant":
        emb = ieeg_fm.brant_embeddings(x_raw, fs=fs, device=args.device,
                                       batch_size=args.batch_size)
    else:
        emb = ieeg_fm.popt_embeddings(x_raw, xyz, fs=fs,
                                      device=args.device, batch_size=args.batch_size)
    emb = np.asarray(emb, dtype=np.float32)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, emb=emb, y=y, event_times=ev_t)
    print(f"[cache] wrote {tag}  {emb.shape}", flush=True)
    return emb, y, ev_t


def probe_auroc(x_tr, y_tr, x_te, y_te, seed):
    """One logistic probe, as the published arms used."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
        return float("nan")
    sc = StandardScaler().fit(x_tr)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000,
                             random_state=seed)
    clf.fit(sc.transform(x_tr), y_tr)
    return float(roc_auc_score(y_te, clf.decision_function(sc.transform(x_te))))


def score_subject(emb, y, folds, arm, seed):
    """AUROC for one subject under one arm."""
    def fold_mean(feats):
        vals = [probe_auroc(feats[tr], y[tr], feats[te], y[te], seed)
                for tr, _, te in folds]
        vals = [v for v in vals if not np.isnan(v)]
        return float(np.mean(vals)) if vals else float("nan")

    if arm == "pop_meanpool":
        return fold_mean(emb.mean(axis=1)), None
    per_elec = [fold_mean(emb[:, c, :]) for c in range(emb.shape[1])]
    per_elec = np.asarray(per_elec, dtype=np.float64)
    if arm == "single_elec_max":
        return float(np.nanmax(per_elec)), per_elec
    return float(np.nanmean(per_elec)), per_elec


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fm", required=True, choices=["brainbert", "popt", "brant"])
    p.add_argument("--endpoint", default="sentence_onset",
                   choices=["sentence_onset", "word_nonword"])
    p.add_argument("--arm", default="single_elec_max",
                   choices=["single_elec_max", "single_elec_mean", "pop_meanpool"])
    p.add_argument("--subjects", nargs="+",
                   default=[f"sub_{i}" for i in range(1, 11)])
    p.add_argument("--trial", default=None)
    p.add_argument("--max_per_class", type=int, default=900)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.0,
                   help="Frozen probes need no validation block; 0 keeps all "
                        "history for fitting")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no_cache", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_root", default="")
    args = p.parse_args()

    ep = window_for(args.fm, args.endpoint)
    save_root = args.save_root or os.path.join(btb_output_root(), "fm_runs")
    os.makedirs(save_root, exist_ok=True)

    results, t0 = {}, time.time()
    for subj in args.subjects:
        emb, y, ev_t = subject_embeddings(subj, args)
        folds = forward_chaining_split(ev_t, win_sec=ep["win_sec"],
                                       n_folds=args.n_folds, val_frac=args.val_frac)
        folds = [(f[0], None, f[-1]) for f in folds]        # (train, _, test)
        rep = split_report(ev_t, ep["win_sec"], folds[0][0], folds[0][2],
                           scheme="forward_chaining")
        if rep["overlapping_train_test_pairs"] or not rep["causal"]:
            raise AssertionError(f"{subj}: split leaks — {rep}")
        score, per_elec = score_subject(emb, y, folds, args.arm, args.seed)
        results[subj] = {"auroc": score,
                         "n_electrodes": int(emb.shape[1]),
                         "n_events": int(len(y))}
        print(f"[{subj}] {args.arm} AUROC = {score:.4f} "
              f"({emb.shape[1]} electrodes)", flush=True)

    vals = [r["auroc"] for r in results.values() if not np.isnan(r["auroc"])]
    out = {"fm": args.fm, "endpoint": args.endpoint, "arm": args.arm,
           "cohort_mean_auroc": float(np.mean(vals)),
           "cohort_sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
           "per_subject": results, "args": vars(args),
           "elapsed_s": time.time() - t0}
    if args.arm == "single_elec_max":
        out["caveat"] = (
            "single_elec_max selects the best electrode on the same folds that "
            "score it. Its null is ~0.53, not 0.50, and rises with electrode "
            "count. Compare against single_elec_mean.")
    dest = os.path.join(
        save_root, f"{args.fm}_{args.endpoint}_{args.arm}_seed{args.seed}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\ncohort mean AUROC = {out['cohort_mean_auroc']:.4f} "
          f"± {out['cohort_sd']:.4f} over {len(vals)} subjects")
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
