# CORTEG

**Foundation Models Enable Cross-Modality Representation Transfer from Scalp to Intracranial Brain Recordings**

[![Paper](https://img.shields.io/badge/arXiv-TODO-b31b1b.svg)](https://arxiv.org/abs/TODO)
[![Demo](https://img.shields.io/badge/demo-live-2ea44f.svg)](https://liuyinyang1101.github.io/CORTEG/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

CORTEG adapts a pretrained scalp-EEG foundation model to intracranial ECoG
decoding. A new patient can be calibrated from a pooled population model in
**10–30 minutes on a single GPU**, matching or surpassing task-specific deep
baselines on continuous finger-trajectory and audio-envelope regression.

> 🌐 **[Live interactive demo →](https://liuyinyang1101.github.io/CORTEG/)** —
> side-by-side ground-truth vs. predicted hand animation (Stanford) and speech-envelope
> traces (Ghent), with model selector and trajectory smoothing.

## Highlights

- **Stanford finger movement** (n=9, public): pooled CORTEG reaches r=0.554,
  above the strongest prior decoders (DeepFingerNet 0.542, HiLoFuseNet 0.534).
- **Ghent audio envelope** (n=16, private dataset): pooled CORTEG reaches
  r=0.339 vs. 0.261 (CNN-LSTM) — gap significant under Bonferroni-corrected
  paired Wilcoxon.
- **Leave-one-subject-out fine-tuning** (LOO-FT) recovers pooled-level
  performance while updating only ≈297K parameters (LoRA + spatial adapter).

## Repository scope

This release reproduces all paper results on the **public Stanford
fingerflex dataset**. The Ghent speech-envelope dataset is private and is
**not** redistributed; its predictions are included in the live demo for
visualization only.

```
models/steegformer/   ST-EEGFormer backbone, KNNSoftFourier spatial adapter, LoRA
models/               LaBraM / CBraMod / classical baselines
data/                 Stanford loader, splits, z-score scalers, collate
train/                Training engine, early-stopping, LR schedule
experiments/          Runner scripts (CORTEG + every baseline)
configs/              ST-EEGFormer Small/Base/Large JSONs
scripts/              Paper-aligned reproduction shell scripts (one per table/figure)
docs/                 Live interactive demo (Three.js + Plotly, no backend)
```

| Paper artefact | Script |
| --- | --- |
| Table 1, CORTEG pooled (Stanford) | `scripts/table1_corteg_pooled_stanford.sh` |
| Table 1, CORTEG LOO-FT | `scripts/table1_corteg_loo_ft.sh <subject>` |
| Table 1, classical baselines | `scripts/table1_classical_baselines.sh` |
| Table 2, ablations | `scripts/table2_ablations.sh` |
| Fig 2(b,d), scaling + low-data | `scripts/fig2_scaling_and_lowdata.sh` |

Full reproduction recipe: [`REPRODUCE.md`](REPRODUCE.md).

## Install

```bash
git clone https://github.com/LiuyinYang1101/CORTEG.git
cd CORTEG
conda create -n corteg python=3.11 -y && conda activate corteg
pip install -e .
```

Tested on Ubuntu 24.04 with PyTorch 2.11 / CUDA 12.8 on an RTX 5090.

## Quick start

1. **Data.** See [`DATASETS.md`](DATASETS.md) for the Stanford fingerflex
   download instructions.
2. **Pretrained weights.** See [`CHECKPOINTS.md`](CHECKPOINTS.md) for both the
   ST-EEGFormer EEG-FM backbone and the released CORTEG adapter.
3. **Set environment.**
   ```bash
   export CORTEG_DATA_ROOT=$HOME/workspace/datasets/stanford_ecog
   export ECOG_PRETRAINED_ROOT=$HOME/workspace/datasets/pretrained_eeg_mae
   export CORTEG_OUTPUT_ROOT=$HOME/workspace/outputs/corteg
   ```
4. **Reproduce Table 1, CORTEG pooled:**
   ```bash
   bash scripts/table1_corteg_pooled_stanford.sh
   ```
   ~6–12 GPU-hours on a single RTX 5090.

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
distributed under its own license — see [CHECKPOINTS.md](CHECKPOINTS.md).
Ghent dataset: not included.

## Citation

```bibtex
@misc{corteg2026,
  title  = {CORTEG: Foundation Models Enable Cross-Modality Representation Transfer
            from Scalp to Intracranial Brain Recordings},
  author = {TODO: replace with arXiv author list},
  year   = {2026},
  eprint = {TODO},
  archivePrefix = {arXiv}
}
```
