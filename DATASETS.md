# Datasets

CORTEG is evaluated on two ECoG regression tasks. Only the Stanford dataset
is publicly reproducible from this repo.

## Stanford `fingerflex` — public

5-finger flexion regression from ECoG on 9 epilepsy patients (subjects
`bp, cc, ht, jc, jp, mv, wc, wm, zt`).

**Source.** Miller K. J. *A library of human electrocorticographic data and
analyses* (Nat. Hum. Behav. 2019). DOI:
[10.1038/s41562-019-0678-3](https://doi.org/10.1038/s41562-019-0678-3).
Hosted on the [Stanford Digital Repository](https://exhibits.stanford.edu/data/catalog/zk881ps0522).

**Expected layout** under `CORTEG_DATA_ROOT`:

```
$CORTEG_DATA_ROOT/
  bp_features.pkl    # X_feat_tr (N,C,200,2)  X_feat_te  X_raw_tr (N,C,128)  X_raw_te  y_tr (N,5)  y_te (N,5)
  bp_electrode_loc.mat
  cc_features.pkl
  cc_electrode_loc.mat
  ... (one pair per subject)
```

The preprocessing pipeline that produces these `*_features.pkl` files
follows DeepFingerNet (Petrosyan et al., 2022) — band-pass 70–200 Hz with
Hilbert envelope for the HGA stream, band-pass 1–64 Hz for the LFS stream,
both downsampled to 200 Hz / 128 Hz respectively, then synchronised with the
finger trajectories at 25 Hz. The full recipe (MATLAB cleaning + Python
feature extraction) lives in
[`data/stanford_preprocessing/`](data/stanford_preprocessing/tutorial_Stanford.md).

## Ghent speech-envelope — private (not redistributed)

Continuous broadband audio-envelope regression from ECoG on 16 epilepsy
patients listening to a Dutch audiobook, recorded at Ghent University
Hospital. The dataset is held under their epilepsy-monitoring data-use
agreement and is **not part of this release**.

The live demo includes per-subject prediction traces from this dataset
purely for visualization. The data, the trained Ghent checkpoints, and the
Ghent training pipeline are not redistributed. To pursue replication, contact
the corresponding author of the paper.

## Path resolution

The library reads roots from environment variables, with a CLI override
available on every runner:

| Variable | Used by | Default |
| --- | --- | --- |
| `CORTEG_DATA_ROOT` / `ECOG_DATA_ROOT` | Stanford loader | `~/workspace/datasets/stanford_ecog` |
| `ECOG_PRETRAINED_ROOT` | Backbone checkpoints | `~/workspace/datasets/pretrained_eeg_mae` |
| `CORTEG_OUTPUT_ROOT` | Where runs write results | `~/workspace/outputs/corteg` |

CLI flags (`--data_root`, `--save_root`, etc.) take precedence over env vars.
