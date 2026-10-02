"""Brant on Stanford finger-flexion regression, with its own 6 s context.

This produces the Brant row of Table 1 (finger 0.028 +/- 0.031) and of
Table 18, and the 0.1 s-grid variant (0.025). That variant is quoted in the
App. A.11 text only, and there is no archived result file for it in
paper_cells/ to compare a re-run with, seed by seed or subject by subject.

Why a separate runner. Brant's atomic input is a 6 s patch at 250 Hz (1500
samples); CORTEG and the other foundation models read a 1 s window. Handing
Brant the 1 s window would mean zero-padding five sixths of every patch, so
instead each target gets the real 6 s of signal that precedes it:

  1. The whole raw 1 kHz stream of a subject (`data/stanford_native.py`) is
     resampled once to 250 Hz with a polyphase filter. Resampling the stream,
     not each window, avoids edge artefacts at every window boundary.
  2. A 6 s embedding changes slowly next to the 40 ms target grid, so targets
     are binned on a coarse anchor grid (--stride_s, 0.5 s). Each bin gets ONE
     embedding, whose context ends at the EARLIEST target in the bin, and every
     window in the bin reuses it: no window sees signal after its own target
     (beyond the resampler's anti-alias filter, whose taps reach up to 40 ms
     ahead), and the embedding is at most stride_s stale.
  3. The context is the --context_patches x 1500 samples ending at the anchor.
     The rare anchor with less history than that (the first 6 s of a stream) is
     left-filled by tiling its real history, never with zeros.
  4. Brant's own 8-band log-PSD per patch, the official TimeEncoder +
     ChannelEncoder (strictly loaded, see `ieeg_fm.load_brant`), and a mean over
     channels and patches give one 2048-d vector per anchor.
  5. A linear MSE head on those vectors, trained like the other FM heads:
     targets and features z-scored with training statistics, early stopping on
     the last 10% of the training windows, Pearson r per finger on the test split.

Recorded settings (the defaults): --context_patches 1, --stride_s 0.5, linear
head, 200 epochs, patience 30, batch 256, lr 1e-3, weight decay 1e-4, warmup
10, min_lr 1e-6, AMP on a GPU, --extract_batch 16, --train_mode per_subject,
seed 42.
The paper's cell is per-subject; a pooled head scores 0.001 and is not reported.

What "native context" means here, precisely. Brant was pretrained on
SEQUENCES of up to 15 consecutive 6 s patches (90 s), with a (15, 2048)
positional encoding. The paper runs used ONE patch (--context_patches 1): the
positional encoding is sliced to its first row, which is the documented way to
run fewer than 15 patches, and App. A.11 says so. "6 s native context" thus
describes Brant's atomic patch, not the sequence length it was trained on.
--context_patches 15 restores the full 90 s sequence, and it scores higher.
App. A.11 reports this as a separate control: with a per-electrode ridge
readout, r rises from 0.059 at one patch to 0.107 with 15 patches (the newest
patch's output). That control is not this runner's protocol. It extracted
each window independently, on a subsampled split (1500 training and 1000
test windows per subject), and used n = 8 subjects, dropping mv, which lacks
the 90 s of history. This runner instead uses the anchor grid on the full
split and tiles short history (step 3). The same App. A.11 control also scored a ridge
readout of the electrode-averaged embedding, which is the input this runner's
head gets. That readout rose from 0.028 at one patch to 0.079 with 15 patches
when the newest patch's output is used (--patch_pool last). It reached only
0.032 when all 15 patches are averaged (--patch_pool mean, the default, which
is identical to last at one patch). With 15 patches, the tiling of step 3 affects the first 90 s of a
stream rather than the first 6 s.

Two smaller things worth knowing: the input is in the recording's ADC counts,
as it was for the paper (Brant's log-PSD is scale-sensitive, unlike the
z-scored BrainBERT input); and for the first test windows the 6 s context
reaches back across the split boundary into training-time signal -- inputs
only, never targets.

The Ghent (audio) Brant cell, 0.022 +/- 0.035, is not reproducible here: the
Ghent recordings are private.

Third-party code and weights are not redistributed: set BRANT_SRC (Brant_src/
from huggingface.co/Daoze/Brant) and BRANT_WEIGHTS (the folder holding
time_encoder.pt and channel_encoder.pt); see ieeg_fm.py.

Result folders follow experiments/run_ieeg_fm_regression.py. The default is
$CORTEG_OUTPUT_ROOT/ieeg_fm_regression/brant/Stanford/brant/<train_mode>/seed<S>/.
A Brant variant (stride, context, patch pooling, head) or any other value of a
recorded setting is named in the folder (e.g. brant_s0.1, brant_epochs=5).
So is a CPU run: it records the device it used and AMP as off (AMP applies on
CUDA only), and lands in brant_use_amp=False; its fp32 numbers differ from the
paper's GPU run. Subject subsets and --max_windows / --max_anchors runs go
under ieeg_fm_regression_smoke/, in a folder that names the subset and caps.
The same caps are named in the embedding cache.

Example (the paper cell, and the 0.1 s-grid variant):

    python -m experiments.run_brant_regression
    python -m experiments.run_brant_regression --stride_s 0.1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import ieeg_fm
from ieeg_fm import (BRANT_D_MODEL, BRANT_FS, BRANT_MAX_PATCHES, BRANT_PATCH_LEN,
                     _context_patches, compute_power, get_emb, load_brant)
from data.stanford_native import NativeSubject, native_root as resolve_native_root
from experiments.common import STANFORD_SUBJECTS, set_seed
from experiments.run_ieeg_fm_regression import (
    MLP_RECIPE, NOT_SETTINGS, check_cache_rows, departures, expected_windows,
    finished_result, json_args, resolve_device, result_dir, run_settings)
from paths import get_data_root
from train.earlystop import EarlyStopper
from train.lr_schedule import WarmupCosineLR
from train.metrics import corr_per_dim

DATASET = "Stanford"
D_OUT = 5

# The recorded value of every head setting that changes a number (the Brant
# variant itself -- stride, context, patch pooling, head -- is named separately
# in adaptation_tag).
RECIPE = {"epochs": 200, "early_stop_patience": 30, "warmup_epochs": 10, "batch_size": 256,
          "lr": 1e-3, "weight_decay": 1e-4, "min_lr": 1e-6, "val_ratio": 0.1,
          "use_amp": True}
# Also no effect on a number: where Brant is read from, and the extraction
# batch (a memory knob; embeddings are cached per anchor).
BRANT_NOT_SETTINGS = NOT_SETTINGS | {"brant_src", "weights_dir", "extract_batch"}


# ===========================================================================
# Anchor-grid extraction
# ===========================================================================

def resample_stream(stream_tc: np.ndarray, fs: float) -> np.ndarray:
    """(T, C) at fs -> (T', C) at 250 Hz, polyphase, over the whole stream.

    resample_poly rather than an FFT resample: the stream is long and not
    periodic, and an FFT resample would wrap its end onto its start.
    """
    if int(round(fs)) == BRANT_FS:
        return np.ascontiguousarray(stream_tc, dtype=np.float32)
    from scipy.signal import resample_poly
    g = math.gcd(BRANT_FS, int(round(fs)))
    out = resample_poly(stream_tc.astype(np.float64), BRANT_FS // g, int(round(fs)) // g, axis=0)
    return np.ascontiguousarray(out, dtype=np.float32)


def anchor_plan(right_edges: np.ndarray, fs: float, stride_s: float, n250: int,
                max_anchors: int = 0) -> Tuple[np.ndarray, np.ndarray, Dict[int, int]]:
    """Map every window to an anchor bin and give each bin its context end.

    Returns (end250, bin_id, anchor_end): end250[n] is window n's target time as
    an exclusive 250 Hz sample index, bin_id[n] its bin, and anchor_end[b] the
    EARLIEST end250 in bin b -- the causal end of that bin's context.
    """
    end = np.round(np.asarray(right_edges, dtype=np.float64) * (BRANT_FS / float(fs)))
    end = np.clip(end.astype(np.int64), 1, n250)
    bin_sz = max(1, int(round(stride_s * BRANT_FS)))
    bin_id = end // bin_sz
    bins = np.unique(bin_id)
    if max_anchors and max_anchors > 0:
        bins = bins[:max_anchors]
    return end, bin_id, {int(b): int(end[bin_id == b].min()) for b in bins}


def extract_split(et, ec, stream_tc: np.ndarray, fs: float, right_edges: np.ndarray,
                  args, device: str, emb_fn=get_emb) -> Tuple[np.ndarray, Dict]:
    """(N, 2048) per-window embeddings for one split, via the anchor grid."""
    L = int(args.context_patches)
    stream250 = resample_stream(stream_tc, fs)
    _, bin_id, anchor_end = anchor_plan(right_edges, fs, args.stride_s,
                                        stream250.shape[0], args.max_anchors)
    keys = list(anchor_end)
    emb_by_bin: Dict[int, np.ndarray] = {}
    n_padded_anchors = 0
    for i in range(0, len(keys), args.extract_batch):
        chunk = keys[i:i + args.extract_batch]
        pats, flags = zip(*[_context_patches(stream250, anchor_end[b], L) for b in chunk])
        x = np.stack(pats, axis=0).astype(np.float32)                 # (B, C, L, 1500)
        pw = compute_power(x, BRANT_FS).astype(np.float32)            # (B, C, L, 8)
        z = emb_fn(torch.from_numpy(x).to(device), torch.from_numpy(pw).to(device), et, ec)
        if args.patch_pool == "mean":                                 # (B, C, L, 2048)
            pooled = z.mean(dim=(1, 2))
        else:
            pooled = z[:, :, -1].mean(dim=1)
        pooled = pooled.float().cpu().numpy()
        for j, b in enumerate(chunk):
            emb_by_bin[b] = pooled[j]
            n_padded_anchors += int(flags[j])

    # Broadcast to windows. A window whose bin was never extracted exists only
    # under the --max_anchors debug cap; it borrows the nearest extracted bin.
    have = np.array(sorted(emb_by_bin), dtype=np.int64)
    ctx = L * BRANT_PATCH_LEN
    emb = np.zeros((len(right_edges), BRANT_D_MODEL), dtype=np.float32)
    n_padded_windows = 0
    for n in range(len(right_edges)):
        b = int(bin_id[n])
        if b not in emb_by_bin:
            b = int(have[np.argmin(np.abs(have - b))])
        emb[n] = emb_by_bin[b]
        n_padded_windows += int(anchor_end[b] < ctx)
    meta = {"n_windows": int(len(right_edges)), "n_anchors": int(len(emb_by_bin)),
            "n_patches": L, "context_s": L * 6, "stride_s": float(args.stride_s),
            "n_padded_anchors": int(n_padded_anchors),
            "n_padded_windows": int(n_padded_windows),
            "frac_padded_windows": float(n_padded_windows) / max(1, len(right_edges))}
    return emb, meta


def _cache_path(emb_cache: str, sub: str, split: str, args) -> str:
    """The name the paper runs wrote, for the paper's settings.

    Both debug caps (--max_windows as _w<N>, --max_anchors as _a<N>) and
    --patch_pool last are named in it, so a capped array is never read by an
    uncapped run. (The paper runs' own script named only the anchor cap.)
    """
    cap = f"_w{int(args.max_windows)}" if args.max_windows and args.max_windows > 0 else ""
    cap += f"_a{int(args.max_anchors)}" if args.max_anchors and args.max_anchors > 0 else ""
    pool = "_plast" if args.patch_pool == "last" and args.context_patches > 1 else ""
    return os.path.join(
        emb_cache, f"brant_fair_{DATASET}_{sub}_{split}_L{int(args.context_patches)}"
                   f"_s{args.stride_s:g}{pool}{cap}.npz")


def extract_subject(get_model, ns: NativeSubject, args, device: str):
    """(emb_tr, y_tr, emb_te, y_te, meta) for one subject, through the cache.

    `get_model` returns (time_encoder, channel_encoder); on a full cache hit
    nothing is resampled and the pickle is not read.
    """
    out, metas = {}, {}
    for i, split in enumerate(("train", "test")):
        cpath = _cache_path(args.emb_cache, ns.sub, split, args) if args.emb_cache else None
        if cpath and os.path.exists(cpath):
            d = np.load(cpath, allow_pickle=False)
            emb, y = d["emb"].astype(np.float32), d["y"].astype(np.float32)
            check_cache_rows(cpath, {"emb": emb, "y": y},
                             expected_windows(ns, split, args.max_windows))
            out[split] = (emb, y)
            metas[split] = json.loads(str(d["meta"]))
            print(f"  [cache hit] {os.path.basename(cpath)} emb={emb.shape}")
            continue
        re = ns.right_edges(split)
        y = ns.targets()[i]
        if args.max_windows and args.max_windows > 0:
            re, y = re[: args.max_windows], y[: args.max_windows]
        et, ec = get_model()
        emb, meta = extract_split(et, ec, ns.stream(), ns.fs, re, args, device)
        y = np.asarray(y, dtype=np.float32)
        out[split], metas[split] = (emb, y), meta
        print(f"  [extract] {ns.sub}/{split}: emb={emb.shape} anchors={meta['n_anchors']} "
              f"tiled windows={meta['n_padded_windows']}/{meta['n_windows']}")
        if cpath:
            os.makedirs(os.path.dirname(cpath), exist_ok=True)
            tmp = f"{cpath}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmp, emb=emb, y=y, meta=json.dumps(meta))
            os.replace(tmp, cpath)
    return (*out["train"], *out["test"], metas)


# ===========================================================================
# Head (the same protocol as experiments/run_ieeg_fm_regression.py)
# ===========================================================================

def fit_zscore(x):
    mu, sd = x.mean(0, keepdims=True), x.std(0, keepdims=True)
    return mu.astype(np.float32), np.where(sd < 1e-8, 1.0, sd).astype(np.float32)


def apply_zscore(x, mu, sd):
    return ((x - mu) / sd).astype(np.float32)


def build_head(head: str, d_in: int, d_out: int, hidden: int, dropout: float) -> nn.Module:
    if head == "linear":
        return nn.Sequential(nn.Linear(d_in, d_out))
    if head == "mlp":
        return nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(hidden, d_out))
    raise ValueError(head)


def make_subject_dict(sub, emb_tr, y_tr, emb_te, y_te, val_ratio) -> Dict:
    """Tail validation split and training-only target z-score."""
    n = emb_tr.shape[0]
    n_val = min(max(1, int(round(n * val_ratio))), n - 1)
    tr, va = slice(0, n - n_val), slice(n - n_val, n)
    ymu, ysd = fit_zscore(y_tr[tr])
    y_tr_z, y_te_z = apply_zscore(y_tr, ymu, ysd), apply_zscore(y_te, ymu, ysd)
    return {"sub": sub, "emb_tr": emb_tr[tr], "y_tr": y_tr_z[tr],
            "emb_va": emb_tr[va], "y_va": y_tr_z[va], "emb_te": emb_te, "y_te": y_te_z}


def train_head(emb_tr, y_tr, emb_va, y_va, d_out, args, device) -> nn.Module:
    fmu, fsd = fit_zscore(emb_tr)
    Xtr, Xva = apply_zscore(emb_tr, fmu, fsd), apply_zscore(emb_va, fmu, fsd)
    model = build_head(args.head, Xtr.shape[1], d_out, args.mlp_hidden,
                       args.mlp_dropout).to(device)
    model._feat_mu, model._feat_sd = fmu, fsd
    Xtr_t = torch.from_numpy(Xtr).to(device)
    ytr_t = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    Xva_t = torch.from_numpy(Xva).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = WarmupCosineLR(opt, warmup_epochs=args.warmup_epochs,
                           max_epochs=args.epochs, min_lr=args.min_lr)
    early = EarlyStopper(patience=args.early_stop_patience)
    use_amp = bool(args.use_amp) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    n, bs = Xtr_t.shape[0], args.batch_size
    for _ in range(args.epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = torch.mean((model(Xtr_t[idx]) - ytr_t[idx]) ** 2)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        model.eval()
        with torch.no_grad():
            pv = model(Xva_t).float().cpu().numpy()
        if early.step(float(np.nanmean(corr_per_dim(pv, y_va))), model):
            break
    early.restore(model)
    return model


def predict(model, emb, device) -> np.ndarray:
    model.eval()
    X = (emb - model._feat_mu) / model._feat_sd
    with torch.no_grad():
        return model(torch.from_numpy(X.astype(np.float32)).to(device)).float().cpu().numpy()


def score_subject(pred, y_true) -> Dict:
    corr = corr_per_dim(pred, y_true)
    return {"corr_mean": float(np.nanmean(corr)), "corr": [float(c) for c in corr],
            "mse": float(np.mean((pred - y_true) ** 2)), "n": int(y_true.shape[0])}


def write_results(path, args, train_mode, per_subject, extra) -> float:
    vals = [r["corr_mean"] for r in per_subject.values()]
    score = float(np.nanmean(vals))
    sd = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
    os.makedirs(os.path.dirname(path), exist_ok=True)
    obj = {"fm": "brant", "dataset": DATASET, "train_mode": train_mode,
           "mode": f"probe_fairctx{args.context_patches * 6}s", "head": args.head,
           "score": score, "score_sd": sd, "n_subjects": len(vals),
           "score_mse": float(np.nanmean([r["mse"] for r in per_subject.values()])),
           "per_subject": per_subject, "brant_fair": extra, "args": json_args(args)}
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"\nResults saved: {path}  (score={score:.4f}"
          + ("" if sd is None else f" +/- {sd:.4f}") + ")")
    return score


def run_pooled(subjects, d_out, args, device, out_dir, extra):
    """One head for every subject. Not reseeded after the model load, as in the
    run that produced the (unreported) pooled number."""
    cat = lambda k: np.concatenate([s[k] for s in subjects], 0)  # noqa: E731
    model = train_head(cat("emb_tr"), cat("y_tr"), cat("emb_va"), cat("y_va"),
                       d_out, args, device)
    per_subject = {s["sub"]: score_subject(predict(model, s["emb_te"], device), s["y_te"])
                   for s in subjects}
    path = os.path.join(out_dir, "results_pooled.json")
    return path, write_results(path, args, "pooled", per_subject, extra)


def run_per_subject(subjects, d_out, args, device, out_dir, extra):
    per_subject = {}
    for s in subjects:
        set_seed(args.seed)
        model = train_head(s["emb_tr"], s["y_tr"], s["emb_va"], s["y_va"], d_out, args, device)
        per_subject[s["sub"]] = rec = score_subject(predict(model, s["emb_te"], device), s["y_te"])
        print(f"  {s['sub']}: r={rec['corr_mean']:.4f}")
    path = os.path.join(out_dir, "results_persub.json")
    return path, write_results(path, args, "per_subject", per_subject, extra)


# ===========================================================================
# CLI
# ===========================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Brant on Stanford finger-flexion regression (6 s context, anchor grid).")
    p.add_argument("--train_mode", default="per_subject", choices=["per_subject", "pooled"],
                   help="per_subject is the paper's cell")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--context_patches", type=int, default=1,
                   help="6 s patches per context. 1 is the paper's cell (positional encoding "
                        "sliced to one row); 15 is Brant's full 90 s pretraining sequence, "
                        "which scores higher (App. A.11's separate control; see the module "
                        "docstring)")
    p.add_argument("--stride_s", type=float, default=0.5,
                   help="anchor grid in seconds: 0.5 is the paper's cell, 0.1 its variant")
    p.add_argument("--patch_pool", default="mean", choices=["mean", "last"],
                   help="with several patches: average all (the paper's rule) or keep the "
                        "newest. Identical at --context_patches 1")
    p.add_argument("--extract_batch", type=int, default=16, help="anchors per Brant forward")
    p.add_argument("--max_anchors", type=int, default=0,
                   help="smoke tests only: first N anchors per split (0 = all)")
    p.add_argument("--max_windows", type=int, default=0,
                   help="smoke tests only: first N windows per split (0 = all)")

    p.add_argument("--brant_src", default="", help="default $BRANT_SRC")
    p.add_argument("--weights_dir", default="", help="default $BRANT_WEIGHTS")
    p.add_argument("--native_root", default="",
                   help="built <sub>_native1k.npz files (default $CORTEG_NATIVE1K_ROOT, else "
                        "<data_root>/native_1k/built with --data_root if given)")
    p.add_argument("--data_root", default="", help="CORTEG pickles, for the targets")
    p.add_argument("--emb_cache", default="", help="folder for cached anchor embeddings")
    p.add_argument("--save_root", default="",
                   help="default $CORTEG_OUTPUT_ROOT/ieeg_fm_regression/brant[...]/"
                        "Stanford/brant/<train_mode>/seed<seed>; subject subsets and capped "
                        "runs go under ieeg_fm_regression_smoke/")
    p.add_argument("--skip_if_done", action="store_true",
                   help="exit at once if the result folder already holds this exact run "
                        "(same subjects, same value of every setting)")
    p.add_argument("--subjects", default="", help="comma-separated subset")
    p.add_argument("--skip_subjects", default="")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                   help="the paper ran on a GPU with AMP; a CPU run (fp32) is recorded with "
                        "use_amp False and named <adaptation>_use_amp=False")

    p.add_argument("--head", default="linear", choices=["linear", "mlp"])
    p.add_argument("--mlp_hidden", type=int, default=256)
    p.add_argument("--mlp_dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--early_stop_patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--use_amp", dest="use_amp", action="store_true", default=True,
                   help="mixed precision on CUDA (default; the paper run used it)")
    p.add_argument("--no_amp", dest="use_amp", action="store_false")
    return p


def adaptation_tag(args) -> str:
    """Result sub-folder: 'brant' for the paper's settings; the variant, and any
    other recorded setting as _<name>=<value>, named otherwise."""
    tag = "brant"
    if args.stride_s != 0.5:
        tag += f"_s{args.stride_s:g}"
    if args.context_patches != 1:
        tag += f"_L{args.context_patches}"
        if args.patch_pool != "mean":
            tag += f"_{args.patch_pool}"
    if args.head != "linear":
        tag += f"_{args.head}"
        tag += departures(args, MLP_RECIPE)
    return tag + departures(args, RECIPE)


def resolve_save_root(args, subjects: Optional[List[str]] = None) -> str:
    """--save_root, or the default folder (see run_ieeg_fm_regression.result_dir)."""
    subs = resolve_subjects(args) if subjects is None else subjects
    return result_dir(args, subs, adaptation_tag(args), "brant")


def resolve_subjects(args) -> List[str]:
    subs = list(STANFORD_SUBJECTS)
    if args.subjects:
        want = [s.strip() for s in args.subjects.split(",") if s.strip()]
        unknown = [s for s in want if s not in subs]
        if unknown:
            raise SystemExit(f"unknown Stanford subjects: {unknown}")
        subs = [s for s in subs if s in want]
    if args.skip_subjects:
        skip = {s.strip() for s in args.skip_subjects.split(",") if s.strip()}
        subs = [s for s in subs if s not in skip]
    return subs


def main(argv=None):
    args = build_argparser().parse_args(argv)
    if not 1 <= args.context_patches <= BRANT_MAX_PATCHES:
        raise SystemExit(f"--context_patches must be in [1, {BRANT_MAX_PATCHES}]")
    set_seed(args.seed)
    # Before anything is named or compared: records the device and AMP as used,
    # so a CPU run is brant_use_amp=False, never the paper's GPU cell.
    device = resolve_device(args)
    subjects = resolve_subjects(args)
    if not subjects:
        raise SystemExit("no subjects selected")
    out_dir = resolve_save_root(args, subjects)
    if args.skip_if_done:
        done = finished_result(out_dir, args.train_mode,
                               run_settings(args, BRANT_NOT_SETTINGS), subjects)
        if done:
            print(f"[done] {done} already records this run; skipping")
            with open(done) as f:
                return json.load(f).get("score")
    os.makedirs(out_dir, exist_ok=True)
    if args.emb_cache:
        os.makedirs(args.emb_cache, exist_ok=True)
    data_root = args.data_root or get_data_root()
    nroot = resolve_native_root(args.native_root, args.data_root)

    print("=" * 68)
    print(f"Brant regression: train_mode={args.train_mode} head={args.head} seed={args.seed}")
    print(f"  device={device} amp={args.use_amp} L={args.context_patches} "
          f"({args.context_patches * 6} s) "
          f"stride_s={args.stride_s} patch_pool={args.patch_pool}")
    print(f"  subjects ({len(subjects)}): {subjects}")
    print(f"  native_root={nroot}\n  save_root={out_dir}\n  emb_cache={args.emb_cache or None}")
    print("=" * 68)

    # The model is loaded up front, as in the paper run: its construction draws
    # from the RNG, which the (unreported) pooled head inherits.
    et, ec = load_brant(args.brant_src or ieeg_fm.brant_src_dir(),
                        args.weights_dir or ieeg_fm.brant_weights_dir(),
                        str(device), n_patches=args.context_patches)

    t0 = time.time()
    subjects_emb, metas = [], {}
    for sub in subjects:
        print(f"\n[load+extract] {sub}")
        ns = NativeSubject(sub, nroot, data_root)
        emb_tr, y_tr, emb_te, y_te, meta = extract_subject(lambda: (et, ec), ns, args, str(device))
        subjects_emb.append(make_subject_dict(sub, emb_tr, y_tr, emb_te, y_te, args.val_ratio))
        metas[sub] = meta
        del ns

    tot = sum(m["test"]["n_windows"] for m in metas.values())
    pad = sum(m["test"]["n_padded_windows"] for m in metas.values())
    extra = {"stride_s": args.stride_s, "n_patches": args.context_patches,
             "context_s": args.context_patches * 6, "patch_pool": args.patch_pool,
             "resample_hz": BRANT_FS, "pad_mode": "wrap (tiles real history)",
             "per_subject_meta": metas, "test_padded_windows": int(pad),
             "test_windows": int(tot), "test_frac_padded": float(pad) / max(1, tot)}
    print(f"\n[extract] {time.time() - t0:.0f}s; test windows with tiled context: {pad}/{tot}")

    run = run_pooled if args.train_mode == "pooled" else run_per_subject
    path, score = run(subjects_emb, D_OUT, args, device, out_dir, extra)
    with open(os.path.join(out_dir, "brant_extraction_meta.json"), "w") as f:
        json.dump(extra, f, indent=2)
    print(f"\n{'=' * 68}\nDONE brant/{args.train_mode} score={score:.4f}\n{path}\n{'=' * 68}")
    return score


if __name__ == "__main__":
    main()
