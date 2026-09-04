"""CORTEG on the BrainTreebank sentence-onset benchmark.

BrainTreebank is public sEEG from 10 patients watching films. The task is binary:
given a window of sEEG anchored at a transcript event, did a sentence begin?
Metric is AUROC; the paper reports per-subject AUROC over 10 subjects.

This runner reuses the Stanford model code unchanged --
``run_regression_hilo_clean.build_model``, ``configure_lora_lastn_probe`` and
``unfreeze_merge_params`` -- so the architecture here is the same CORTEG, with
``d_out=1`` and a BCE loss instead of the 5-D regression head. What differs is the
data path (``data.braintreebank``) and the evaluation protocol.

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
  boundary asserted free of overlapping windows.

Example:

    python -m experiments.run_btb_classification \\
        --subjects sub_3 --merge_strategy layerwise_gate \\
        --model_kwargs_json configs/steegformer_small.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import paths
from data.braintreebank import (
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
)

# Architecture constants — these are the paper's BrainTreebank settings and are
# deliberately not exposed as flags; changing them changes the reported model.
HI_PATCH_SIZE = 25
HI_INJECT_LAST_N = 4
CHANNEL_ADAPTER = "knn_soft_fourier"
KNN_K = 8
LORA_TARGETS = "qkv,proj,fc1,fc2"


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


def extract_subject(subj: str, args):
    """(x_lo, x_hi, y, xyz_m, event_times), cached as an npz.

    The cache name encodes the window and band, because a cache built at a
    different window is silently wrong for every downstream number.
    """
    root = btb_root()
    trial = args.trial or trial_of(root, subj)
    tag = (f"btb_{subj}_{trial}_win{args.win_sec}_pre{args.pre_sec}"
           f"_hfa{int(args.hga_low)}-{int(args.hga_high)}.npz")
    cache = os.path.join(btb_output_root(), "cache", tag)
    if os.path.exists(cache) and not args.no_cache:
        d = np.load(cache, allow_pickle=True)
        print(f"[cache] {tag}", flush=True)
        return d["x_lo"], d["x_hi"], d["y"], d["xyz"], d["event_times"]

    name2idx, _ = load_electrode_map(root, subj)
    loc = load_localization_mni(root, subj)          # shared MNI frame, mm
    use = [n for n in clean_electrodes(subj) if n in name2idx and n in loc]
    ch_idx = [name2idx[n] for n in use]
    xyz_m = (np.array([loc[n] for n in use], dtype=np.float64) / 1000.0).astype(np.float32)

    fs = estimate_fs(root, subj, trial)              # measured, never assumed
    t2s = build_time_to_sample(root, subj, trial)
    starts, y = load_events(root, movie_of(root, subj, trial), args.max_per_class, args.seed)
    print(f"[{subj}] trial={trial} fs={fs:.1f}Hz electrodes={len(use)} events={len(starts)}",
          flush=True)

    x_raw, valid = extract_windows(root, subj, trial, ch_idx, starts, t2s, fs,
                                   args.pre_sec, args.win_sec)
    y, ev_t = y[valid], starts[valid]
    x_lo, x_hi = corteg_features(x_raw, fs, args.hga_low, args.hga_high)
    del x_raw

    os.makedirs(os.path.dirname(cache), exist_ok=True)
    np.savez_compressed(cache, x_lo=x_lo, x_hi=x_hi, y=y, xyz=xyz_m,
                        event_times=ev_t, win_sec=args.win_sec, pre_sec=args.pre_sec)
    print(f"[cache] wrote {tag}", flush=True)
    return x_lo, x_hi, y, xyz_m, ev_t


def zscore_fit_apply(fit, *others, eps: float = 1e-6):
    """Per-channel z-score over (N, C, T), fitted on the fit split only."""
    mu = fit.mean(axis=(0, 2), keepdims=True)
    sd = fit.std(axis=(0, 2), keepdims=True) + eps
    return [((a - mu) / sd).astype(np.float32) for a in (fit,) + others]


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
        channel_adapter=CHANNEL_ADAPTER, knn_k=KNN_K, knn_sigma=0.0,
        adapter_branch="both", xyz_mode="real",
        use_ecog_fuser=False, M_EEG=145, fuser_hidden=128,
        head_dropout=args.head_dropout, head_hidden=0,
        lora_r=args.lora_r, lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout, lora_last_n=args.lora_last_n,
        lora_targets=LORA_TARGETS, full_finetune=False, use_full_codebook=False,
    )
    model = build_model(a, C_in=C, T_in=T_lo, ecog_xyz_m=xyz_m * 1000.0, d_out=1)
    configure_lora_lastn_probe(
        model, n_last=args.lora_last_n, r=args.lora_r, alpha=args.lora_alpha,
        dropout=args.lora_dropout, targets=tuple(LORA_TARGETS.split(",")))
    unfreeze_merge_params(model)
    return model


def run_fold(x_lo, x_hi, y, xyz_m, fold, args, device):
    """Fit on one causal fold; return test AUROC."""
    from sklearn.metrics import roc_auc_score
    from train.earlystop import EarlyStopper
    from train.lr_schedule import WarmupCosineLR

    fit_i, val_i, te_i = (np.asarray(a) for a in fold)
    lo_f, lo_v, lo_t = zscore_fit_apply(x_lo[fit_i], x_lo[val_i], x_lo[te_i])
    hi_f, hi_v, hi_t = zscore_fit_apply(x_hi[fit_i], x_hi[val_i], x_hi[te_i])

    model = build_corteg(x_lo.shape[1], x_lo.shape[2], xyz_m, args).to(device)
    xyz_t = torch.from_numpy(xyz_m).float().to(device)

    def loader(lo, hi, yy, shuffle):
        return DataLoader(TensorDataset(torch.from_numpy(lo), torch.from_numpy(hi),
                                        torch.from_numpy(yy.astype(np.float32))),
                          batch_size=args.batch_size, shuffle=shuffle)

    tr_dl = loader(lo_f, hi_f, y[fit_i], True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                           max_epochs=args.epochs, min_lr=args.min_lr)
    lossf = nn.BCEWithLogitsLoss()
    early = EarlyStopper(patience=args.patience)

    @torch.no_grad()
    def scores(lo, hi):
        model.eval()
        out = []
        for i in range(0, len(lo), args.batch_size):
            b_lo = torch.from_numpy(lo[i:i + args.batch_size]).to(device)
            b_hi = torch.from_numpy(hi[i:i + args.batch_size]).to(device)
            o = model(b_lo, x_hi=b_hi,
                      ecog_xyz=xyz_t.unsqueeze(0).expand(b_lo.shape[0], -1, -1))
            out.append(o.squeeze(-1).float().cpu().numpy())
        return np.concatenate(out)

    for ep in range(args.epochs):
        model.train()
        for b_lo, b_hi, b_y in tr_dl:
            b_lo, b_hi, b_y = b_lo.to(device), b_hi.to(device), b_y.to(device)
            out = model(b_lo, x_hi=b_hi,
                        ecog_xyz=xyz_t.unsqueeze(0).expand(b_lo.shape[0], -1, -1))
            loss = lossf(out.squeeze(-1), b_y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, args.max_norm)
            opt.step()
        sched.step()
        val_auc = float(roc_auc_score(y[val_i], scores(lo_v, hi_v)))
        if ep % 10 == 0:
            print(f"    [ep {ep+1}] loss={loss.item():.4f} val_auroc={val_auc:.4f}", flush=True)
        if early.step(val_auc, model):
            break
    early.restore(model)
    return float(roc_auc_score(y[te_i], scores(lo_t, hi_t)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subjects", nargs="+",
                   default=[f"sub_{i}" for i in range(1, 11)])
    p.add_argument("--trial", default=None, help="default: the subject's only trial")
    p.add_argument("--win_sec", type=float, default=1.5,
                   help="Window width; also the overlap footprint used by the split")
    p.add_argument("--pre_sec", type=float, default=0.0,
                   help="Window start relative to the event (0 = anchored at onset)")
    p.add_argument("--max_per_class", type=int, default=900)
    p.add_argument("--hga_low", type=float, default=70.0)
    p.add_argument("--hga_high", type=float, default=200.0)
    p.add_argument("--n_folds", type=int, default=4)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--no_cache", action="store_true")

    p.add_argument("--model_kwargs_json", type=str, default="",
                   help="ST-EEGFormer config; required unless --no_pretrained")
    p.add_argument("--no_pretrained", action="store_true",
                   help="Random-init backbone (the ablation, not CORTEG)")
    p.add_argument("--steegformer_variant", default="small",
                   choices=["small", "base", "large"])
    p.add_argument("--merge_strategy", default="layerwise_gate",
                   choices=["average", "layerwise_gate"])
    p.add_argument("--layerwise_gate_bottleneck", type=int, default=16)
    p.add_argument("--layerwise_gate_act", default="tanh",
                   choices=["tanh", "sigmoid", "none"])
    p.add_argument("--lora_last_n", type=int, default=4)
    p.add_argument("--lora_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--lora_dropout", type=float, default=0.2)
    p.add_argument("--head_dropout", type=float, default=0.1)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.005)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--max_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_root", default="")
    args = p.parse_args()

    if not args.model_kwargs_json and not args.no_pretrained:
        raise SystemExit(
            "No pretrained backbone configured.\n"
            "  CORTEG loads ST-EEGFormer weights via --model_kwargs_json; without it\n"
            "  the backbone stays randomly initialised, which is the 'random init'\n"
            "  ablation rather than CORTEG.\n"
            f"  Fix:  --model_kwargs_json configs/steegformer_{args.steegformer_variant}.json\n"
            "  Or, to request random init deliberately:  --no_pretrained")

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_root = args.save_root or os.path.join(btb_output_root(), "runs")
    os.makedirs(save_root, exist_ok=True)

    results, t0 = {}, time.time()
    for subj in args.subjects:
        x_lo, x_hi, y, xyz_m, ev_t = extract_subject(subj, args)
        folds = forward_chaining_split(ev_t, win_sec=args.win_sec,
                                       n_folds=args.n_folds, val_frac=args.val_frac)
        rep = split_report(ev_t, args.win_sec, folds[0][0], folds[0][2],
                           folds[0][1], scheme="forward_chaining")
        print(f"[{subj}] {len(folds)} causal folds | {rep}", flush=True)

        aucs = []
        for fi, fold in enumerate(folds):
            auc = run_fold(x_lo, x_hi, y, xyz_m, fold, args, device)
            aucs.append(auc)
            print(f"  [{subj}] fold {fi}: test AUROC = {auc:.4f}", flush=True)
        results[subj] = {"folds": aucs, "mean": float(np.mean(aucs))}
        print(f"[{subj}] mean AUROC = {np.mean(aucs):.4f}\n", flush=True)

    cohort = float(np.mean([r["mean"] for r in results.values()]))
    out = {"cohort_mean_auroc": cohort, "per_subject": results,
           "args": vars(args), "elapsed_s": time.time() - t0}
    dest = os.path.join(save_root, f"btb_{args.merge_strategy}_seed{args.seed}.json")
    with open(dest, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"cohort mean AUROC over {len(results)} subjects = {cohort:.4f}")
    print(f"written: {dest}")


if __name__ == "__main__":
    main()
