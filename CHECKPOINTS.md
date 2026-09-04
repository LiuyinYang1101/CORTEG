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

**Source.** Released by the [ST-EEGFormer repo](https://github.com/LiuyinYang1101/STEEGFormer).
The Small backbone -- the only one needed for Table 1 and Table 2 -- is a direct
download (394 MB):

```bash
mkdir -p "$ECOG_PRETRAINED_ROOT/experiment3_small"
curl -L -o "$ECOG_PRETRAINED_ROOT/experiment3_small/checkpoint-300.pth" \
  https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-small/checkpoint-300.pth
```

The layout above is what `configs/steegformer_*.json` expects, so rename release
assets accordingly if they differ. The original paper's redistribution
license applies — we do not re-host these weights here.

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
  corteg_stanford_pooled.pt      # 1.2 MB, 297,236 params — Table 1 finger row, r = 0.554
  corteg_stanford_pooled.json    # manifest: sha256, paper number, exact build args
```

**Host.** Shipped in this repository — at 1.2 MB the adapter needs no external
host. Only the 376 MB ST-EEGFormer backbone above is a separate download.

## 3. Using the released adapter

`load_corteg.py` rebuilds the exact architecture from the manifest and loads
every tensor, raising if anything fails to match:

```python
from load_corteg import load_corteg, predict

model = load_corteg(C_in=46, T_in=128, ecog_xyz_mm=xyz, d_out=5, device="cuda")
y = predict(model, x_lo, x_hi)      # (N, 46, 128) and (N, 46, 200) -> (N, 5)
```

CORTEG is dual-stream: `x_lo` is the broadband signal at 128 Hz and `x_hi` the
high-gamma envelope feature, and the patch sizes are chosen so both give 8
tokens per electrode (128/16 == 200/25). `predict` refuses a missing `x_hi`
rather than silently changing the model.

Prefer this over `--finetune_from`, which loads with `strict=False`: that is
correct for cross-task fine-tuning, but against a mismatched architecture it
loads almost nothing, reports success, and yields an untrained model.

**Verified.** Loading this checkpoint and running it over the Stanford test
split reproduces the published per-subject scores on all 9 subjects (cohort
mean 0.5537 vs 0.5535 in the paper; largest per-subject deviation 0.0016, from
fitting the z-score statistics on the full training split rather than the exact
90 % training subset). `notebooks/quickstart.ipynb` walks through this.

The Ghent checkpoint is not released (private dataset; see [DATASETS.md](DATASETS.md)).

## 3. Baseline FM checkpoints (optional, Table 2 ablation rows)

| Model | Path | Source |
| --- | --- | --- |
| LaBraM-base | `~/workspace/datasets/pretrained_eeg_fms/labram/labram-base.pth` | [LaBraM repo](https://github.com/935963004/LaBraM) |
| CBraMod | `~/workspace/datasets/pretrained_eeg_fms/cbramod/pretrained_weights.pth` | [CBraMod repo](https://github.com/wjq-learning/CBraMod) |
| MantisV2 | `~/workspace/datasets/pretrained_tsfm/mantis_v2` | [Mantis-TS repo](https://github.com/Mantis-TS/MantisV2) |

Each runner takes a `--pretrained_path` flag if you prefer a different
location.
