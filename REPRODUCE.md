# Reproducing CORTEG

This document maps each table and figure in the paper to the script that
reproduces it on the **public Stanford fingerflex dataset**. Ghent
audio-envelope numbers in the paper come from a private dataset and are not
publicly reproducible; the Ghent panels of the demo are included only for
visualization.

## Prerequisites

- Stanford `fingerflex` data (see [DATASETS.md](DATASETS.md)).
- ST-EEGFormer pretrained checkpoint (see [CHECKPOINTS.md](CHECKPOINTS.md)).
- Environment variables:

```bash
export CORTEG_DATA_ROOT=$HOME/workspace/datasets/stanford_ecog
export ECOG_PRETRAINED_ROOT=$HOME/workspace/datasets/pretrained_eeg_mae
export CORTEG_OUTPUT_ROOT=$HOME/workspace/outputs/corteg
```

Compute: one RTX 5090 (or H100) suffices. Each pooled CORTEG run takes
≈6–12 GPU-hours; each LOO-FT Stage-2 call is 10–30 minutes per held-out
subject. The full Table 2 ablation sweep is ≈70 GPU-hours.

## Table 1 — Main results (Stanford finger, n=9)

| Row | Command |
| --- | --- |
| CORTEG pooled | `bash scripts/table1_corteg_pooled_stanford.sh` |
| CORTEG LOO-FT | `for s in bp cc ht jc jp mv wc wm zt; do bash scripts/table1_corteg_loo_ft.sh $s; done` |
| CORTEG per-subject | `bash scripts/table1_corteg_pooled_stanford.sh --train_mode per_subject` |
| Classical baselines | *Not in this release.* `experiments/run_ecog_baselines.py` implements PLS, HOPLS, LSTM and HiLoFuseNet, but is Ghent-only (no `--dataset` argument, Ghent loader hardcoded); Ridge is not implemented at all. |
| DeepFingerNet | Transcribed from its original paper (Table II). |

## Baseline comparisons

The foundation-model baselines use dedicated runners:

```bash
python -m experiments.run_labram_baseline   --dataset Stanford --train_mode pooled --seed 42
python -m experiments.run_cbramod_baseline  --dataset Stanford --train_mode pooled --seed 42
python -m experiments.run_mantis_baseline   --dataset Stanford --train_mode pooled --seed 42
```

See [CHECKPOINTS.md](CHECKPOINTS.md) for their pretrained weights.


## Figure 3 — Neural manifold & electrode importance

These analyses operate on a saved CORTEG-S checkpoint. After running
`scripts/table1_corteg_pooled_stanford.sh`, the latent-extraction and brain
plotting are documented in the analysis section of the paper's appendix
(scripts not part of this release; will be added if there is interest —
open an issue).

## Numerical reproducibility

All scripts pin `--seed 42`. Per-subject Pearson r fluctuates by ≈±0.005
across seeds on Stanford.
