# Pretrained Weights

CORTEG combines a frozen ST-EEGFormer EEG-FM backbone with a small set of
trainable parameters (LoRA + KNNSoftFourier spatial adapter + regression head).
Two artifact families are needed.

## 1. ST-EEGFormer EEG-FM backbone (third-party)

Released with the ST-EEGFormer paper (winner of the NeurIPS 2025 EEG
Foundation Challenge regression task). Place under `$ECOG_PRETRAINED_ROOT`
matching the paths in `configs/steegformer_*.json`:

```
$ECOG_PRETRAINED_ROOT/
  experiment3_small/checkpoint-300.pth      # Small  (D=512, L=8,  25.6 M params)
  experiment4_base/checkpoint-288.pth       # Base   (D=768, L=12, 85.6 M params)
  experiment5_large/checkpoint-196.pth      # Large  (D=1024, L=24, 303 M params)
```

**Source.** TODO — link to the ST-EEGFormer release once that project is
public. The original paper's redistribution license applies; we do not
re-host these weights here.

The Small backbone is sufficient for all main paper numbers (Table 1, Table 2).

## 2. CORTEG trained adapter (this paper)

We release the trained Stage-1 pooled checkpoint that produced the Table 1
finger row. The checkpoint contains:

- LoRA A/B matrices for the last 4 transformer blocks
- KNNSoftFourier adapter weights
- Regression head

It does **not** contain the frozen ST-EEGFormer backbone — load it together
with the backbone above.

```
checkpoints/
  corteg_small_stanford_pooled_seed42.pth   # Table 1 finger row, r = 0.554
```

**Host.** TODO — HuggingFace Hub release at
`https://huggingface.co/<org>/CORTEG`. Download with:

```bash
huggingface-cli download <org>/CORTEG --local-dir checkpoints/
```

To eval a released checkpoint without retraining:

```bash
python -m experiments.run_regression_hilo_clean \
    --dataset Stanford --train_mode per_subject \
    --finetune_from checkpoints/corteg_small_stanford_pooled_seed42.pth \
    --epochs 0   # eval-only
```

The Ghent checkpoint is not released (private dataset; see [DATASETS.md](DATASETS.md)).

## 3. Baseline FM checkpoints (optional, Table 2 ablation rows)

| Model | Path | Source |
| --- | --- | --- |
| LaBraM-base | `~/workspace/datasets/pretrained_eeg_fms/labram/labram-base.pth` | [LaBraM repo](https://github.com/935963004/LaBraM) |
| CBraMod | `~/workspace/datasets/pretrained_eeg_fms/cbramod/pretrained_weights.pth` | [CBraMod repo](https://github.com/wjq-learning/CBraMod) |
| MantisV2 | `~/workspace/datasets/pretrained_tsfm/mantis_v2` | [Mantis-TS repo](https://github.com/Mantis-TS/MantisV2) |

Each runner takes a `--pretrained_path` flag if you prefer a different
location.
