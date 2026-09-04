# CORTEG

**Foundation Models Enable Cross-Modality Representation Transfer from Scalp to Intracranial Brain Recordings**

> Adapt a frozen scalp-EEG foundation model to intracranial ECoG decoding —
> calibrate to a new patient in **10–30 minutes** on a single GPU.

[![Paper](https://img.shields.io/badge/arXiv-2605.10337-b31b1b.svg)](https://arxiv.org/abs/2605.10337)
[![Demo](https://img.shields.io/badge/demo-live-2ea44f.svg)](https://liuyinyang1101.github.io/CORTEG/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

![CORTEG overview](docs/assets/figure1_overview_scheme.png)

## Motivation

Intracranial ECoG offers high-SNR access to cortical activity, but each
recording is short, patient-specific, and electrode layouts vary, so most
ECoG decoders are trained per-subject and ignore information shared across
patients. Scalp-EEG foundation models, by contrast, are pretrained on
millions of recordings — but they have never been systematically adapted
*down into the skull*. The question CORTEG asks is whether the
representations learned from extracranial EEG are useful for intracranial
ECoG, and whether a population model can be deployed to a new patient
without retraining from scratch.

## What CORTEG is

A small set of trainable components wrapped around a frozen EEG foundation
model:

- **Frozen [ST-EEGFormer](https://github.com/LiuyinYang1101/STEEGFormer) backbone** —
  pretrained on 128-Hz scalp EEG with an EEG channel-embedding codebook.
- **KNNSoftFourier spatial adapter** — maps each ECoG electrode's 3-D MNI
  coordinate into the pretrained EEG embedding space (soft k-NN over the
  codebook + a learnable Fourier residual).
- **Dual-stream tokenization** — low-frequency (1–64 Hz) + high-gamma
  (70–200 Hz) tokens merged before the last transformer blocks.
- **LoRA on the last 4 blocks**, regression head, and the spatial adapter
  are the *only* trainable parameters (≈297K total).
- **Two-stage LOO-FT** — pool on N − 1 patients, then fine-tune the
  adapter + LoRA + head on the held-out patient in minutes.

## Key findings

- **Cross-modality pretraining transfers.** Pooled CORTEG reaches the
  highest mean correlation among compared methods on both tasks: r=0.554
  on Stanford finger (n=9) and r=0.339 on Ghent audio (n=16), beating the
  strongest task-specific deep baselines.
- **LOO-FT matches pooled training.** A new patient calibrates in 10–30 min
  on a single GPU and lands within ties of the pooled upper bound
  (Wilcoxon p=0.65 finger, p=0.82 audio).
- **The signal is the EEG pretraining, not the architecture.** Replacing
  ST-EEGFormer with a random init drops r by 0.044 (finger) / 0.183 (audio);
  swapping in LaBraM, CBraMod, or MantisV2 backbones drops r by 0.18–0.36
  on finger.

## Headline results — Stanford finger (n=9) and Ghent audio (n=16)

Pearson r (± cross-subject SD). Best per column in **bold**, second-best in *italic*.

| Method | Trainable params | Finger (n=9) | Audio (n=16) |
| --- | ---: | ---: | ---: |
| Ridge (HGA) | — | 0.336 ± 0.095 | 0.175 |
| HiLoFuseNet | 334 K | 0.534 ± 0.138 | *0.259 ± 0.203* |
| CNN-LSTM | 557 K | 0.411 ± 0.124 | 0.261 ± 0.195 |
| DeepFingerNet | 1.16 M | *0.542 ± 0.129* | 0.085 ± 0.144 |
| **CORTEG (pooled)** | **297 K** | **0.554 ± 0.154** | **0.339 ± 0.170** |
| CORTEG (LOO-FT) | 297 K | 0.551 ± 0.147 | 0.331 ± 0.184 |

Full table, ablations, and per-subject paired tests: see the paper.

## Repository scope

This release trains and evaluates **CORTEG on the public Stanford fingerflex
dataset**, in both fusion variants, and ships the trained adapter.

The **Ghent** speech-envelope dataset is private, is **not** redistributed, and
no Ghent training code is included. The live demo shows per-subject Ghent
prediction traces (and the corresponding target envelope segment) for
visualisation only.

```
load_corteg.py        load the released adapter and run it
paths.py              environment-variable path resolution
models/steegformer/   ST-EEGFormer backbone, KNNSoftFourier adapter, LoRA
models/               LaBraM / CBraMod foundation-model baselines
data/                 Stanford loader, splits, z-score scalers, collate
train/                training engine, early stopping, LR schedule
experiments/          CORTEG runner + baseline runners
configs/              ST-EEGFormer Small/Base/Large JSONs
scripts/              paper-aligned reproduction scripts
notebooks/            quickstart.ipynb — start here
checkpoints/          released CORTEG adapter + manifest
docs/                 live interactive demo (Three.js + Plotly, no backend)
```

| Paper artefact | Script |
| --- | --- |
| Table 1, CORTEG pooled (Stanford) | `scripts/table1_corteg_pooled_stanford.sh` |
| Table 1, CORTEG LOO-FT | `scripts/table1_corteg_loo_ft.sh <subject>` |

Not covered by this release: the intracranial-FM comparison (BrainBERT / PopT /
Brant), the BrainTreebank benchmark, and the REVE backbone port. Tracked for a
follow-up.

## Install

```bash
git clone https://github.com/LiuyinYang1101/CORTEG.git
cd CORTEG
conda create -n corteg python=3.11 -y && conda activate corteg
pip install -e .
```

Tested on Ubuntu 24.04 with PyTorch 2.11 / CUDA 12.8 on an RTX 5090.

## Quick start

**Start with [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).** It goes
from a fresh clone to reproducing a published per-subject number, and explains
the one thing that is easy to get wrong (CORTEG takes *two* input streams).

## Data

### Stanford `fingerflex` — public

5-finger flexion regression from ECoG on 9 epilepsy patients (`bp, cc, ht, jc,
jp, mv, wc, wm, zt`).

**Source.** Miller K. J., *A library of human electrocorticographic data and
analyses*, Nat. Hum. Behav. 2019,
[10.1038/s41562-019-0678-3](https://doi.org/10.1038/s41562-019-0678-3). Hosted on the
[Stanford Digital Repository](https://exhibits.stanford.edu/data/catalog/zk881ps0522).

**Expected layout** under `$CORTEG_DATA_ROOT`:

```
bp_features.pkl      # X_feat_tr (N,C,200,2)  X_feat_te  X_raw_tr (N,C,128)  X_raw_te  y_tr (N,5)  y_te (N,5)
bp_electrode_loc.mat # (C, 3) electrode coordinates in mm
...                  # one pair per subject
```

The pipeline that produces these files — MATLAB cleaning plus Python feature
extraction, band-pass 70–200 Hz with Hilbert envelope for the high-gamma stream
and 1–64 Hz for the low-frequency stream — is documented step by step in
[`data/stanford_preprocessing/tutorial_Stanford.md`](data/stanford_preprocessing/tutorial_Stanford.md).

### Ghent speech-envelope — private

Continuous audio-envelope regression on 16 patients at Ghent University
Hospital, held under their epilepsy-monitoring data-use agreement. **Not part of
this release**: no data, no checkpoints, no training code. Contact the
corresponding author to pursue replication.

### Path resolution

| Variable | Used by | Default |
| --- | --- | --- |
| `CORTEG_DATA_ROOT` / `ECOG_DATA_ROOT` | Stanford loader | `~/workspace/datasets/stanford_ecog` |
| `CORTEG_PRETRAINED_ROOT` / `ECOG_PRETRAINED_ROOT` | backbone checkpoints | `~/workspace/datasets/pretrained_eeg_mae` |
| `CORTEG_OUTPUT_ROOT` / `ECOG_OUTPUT_ROOT` | where runs write results | `~/workspace/outputs/corteg` |

CLI flags (`--data_root`, `--save_root`) take precedence over the environment.

## Checkpoints

CORTEG is **two files**: a large frozen backbone you download once, and a small
trained adapter that ships here.

### 1. ST-EEGFormer backbone (third-party, required)

Released with the ST-EEGFormer paper. Place under `$CORTEG_PRETRAINED_ROOT`
matching the paths in `configs/steegformer_*.json`:

```
experiment3_small/checkpoint-300.pth      # Small  (D=512,  L=8,  25.6 M)  ← all main results
experiment4_base/checkpoint-288.pth       # Base   (D=768,  L=12, 85.6 M)
experiment5_large/checkpoint-196.pth      # Large  (D=1024, L=24, 303 M)
```

The Small backbone is a direct download (376 MB):

```bash
mkdir -p "$CORTEG_PRETRAINED_ROOT/experiment3_small"
curl -L -o "$CORTEG_PRETRAINED_ROOT/experiment3_small/checkpoint-300.pth" \
  https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-small/checkpoint-300.pth
```

The original paper's redistribution license applies — these weights are not
re-hosted here.

### 2. CORTEG adapter (this paper, included)

```
checkpoints/corteg_stanford_pooled.pt     # 1.2 MB, 297,236 params — Table 1 finger row
checkpoints/corteg_stanford_pooled.json   # manifest: sha256, paper number, exact build args
```

It holds only the trainable parameters — LoRA A/B on blocks 4–7, the
KNNSoftFourier adapter, LayerNorms and the head — so it must be loaded together
with the backbone above.

```python
from load_corteg import load_corteg, predict

model = load_corteg(C_in=46, T_in=128, ecog_xyz_mm=xyz, d_out=5, device="cuda")
y = predict(model, x_lo, x_hi)      # (N,46,128) and (N,46,200) -> (N,5)
```

Prefer this over `--finetune_from`, which loads with `strict=False`: correct for
cross-task fine-tuning, but against a mismatched architecture it loads almost
nothing, reports success, and yields an untrained model. `load_corteg` raises
instead.

**Verified.** This checkpoint reproduces the published per-subject scores on all
9 Stanford subjects: cohort mean 0.5537 against 0.5535 in the paper, largest
per-subject deviation 0.0016 (from fitting z-score statistics on the full
training split rather than the exact 90 % subset).

### 3. Baseline FM checkpoints (optional)

| Model | Path | Source |
| --- | --- | --- |
| LaBraM-base | `$CORTEG_PRETRAINED_ROOT/../pretrained_eeg_fms/labram/labram-base.pth` | [LaBraM](https://github.com/935963004/LaBraM) |
| CBraMod | `$CORTEG_PRETRAINED_ROOT/../pretrained_eeg_fms/cbramod/pretrained_weights.pth` | [CBraMod](https://github.com/wjq-learning/CBraMod) |
| MantisV2 | `$CORTEG_PRETRAINED_ROOT/../pretrained_tsfm/mantis_v2` | [Mantis-TS](https://github.com/Mantis-TS/MantisV2) |

Each runner takes `--pretrained_path` if you keep them elsewhere. These weights
are not re-hosted here; each carries its own license.

## Reproducing the paper

```bash
export CORTEG_DATA_ROOT=$HOME/workspace/datasets/stanford_ecog
export CORTEG_PRETRAINED_ROOT=$HOME/workspace/datasets/pretrained_eeg_mae
export CORTEG_OUTPUT_ROOT=$HOME/workspace/outputs/corteg
```

**Table 1, CORTEG pooled** (≈6–12 GPU-hours on one RTX 5090):

```bash
bash scripts/table1_corteg_pooled_stanford.sh
```

**Table 1, CORTEG LOO-FT** (10–30 min per held-out subject after stage 1):

```bash
for s in bp cc ht jc jp mv wc wm zt; do bash scripts/table1_corteg_loo_ft.sh $s; done
```

**The two fusion variants.** `--merge_strategy average` is fixed fusion at layer
*k* (what the released checkpoint uses); `--merge_strategy layerwise_gate` is
gated fusion, where a small network emits one scalar per block and each block
receives `+ g_l · hi`. With `tanh` the gates start at exactly 0, so training
begins from the low-frequency-only baseline.

**Foundation-model baselines:**

```bash
python -m experiments.run_labram_baseline   --dataset Stanford --train_mode pooled --seed 42
python -m experiments.run_cbramod_baseline  --dataset Stanford --train_mode pooled --seed 42
python -m experiments.run_mantis_baseline   --dataset Stanford --train_mode pooled --seed 42
```

**Numbers taken from their original papers**, not retrained here: DeepFingerNet
(finger r = 0.542, its Table II) and HiLoFuseNet (r = 0.534, its Table V).

All scripts pin `--seed 42`. Per-subject Pearson r fluctuates by ≈±0.005 across
seeds.

## Tests

```bash
python -m unittest discover -s tests
```

The suite checks that every script loads a pretrained backbone, that no script
passes a flag the runner does not have, and that the released checkpoint still
rebuilds and reproduces its paper number.

## Interactive demo

`docs/` is a static, single-page site (Three.js + Plotly, no backend) that
visualises ground-truth vs. predicted trajectories across 9 Stanford subjects
× 16 Ghent subjects × a curated set of models matched to the paper's
Table 1 + key ablation rows. Live at
[liuyinyang1101.github.io/CORTEG](https://liuyinyang1101.github.io/CORTEG/),
or locally:

```bash
cd docs && python -m http.server 8000   # open http://localhost:8000
```

## License

Code: MIT (see [LICENSE](LICENSE)). The pretrained ST-EEGFormer backbone is
distributed under its own license — see **Checkpoints** above.
Ghent dataset: not included.

## Citation

```bibtex
@misc{corteg2026,
  title  = {CORTEG: Foundation Models Enable Cross-Modality Representation Transfer
            from Scalp to Intracranial Brain Recordings},
  author = {Liuyin Yang and Qiang Sun and Bob Van Dyck and Eva Calvo Merino and Marc M. Van Hulle},
  year   = {2026},
  eprint = {2605.10337},
  archivePrefix = {arXiv}
}
```
