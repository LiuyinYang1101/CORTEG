#!/usr/bin/env bash
# Table 3 — CORTEG on the BrainTreebank sentence-onset benchmark.
#
# Prerequisites
#   BTB_DATA_ROOT  the BrainTreebank download (~52 GB), from https://braintreebank.dev/
#   POPT_REPO      a checkout of https://github.com/czlwang/PopulationTransformer,
#                  for the shared electrode selection (not redistributed here)
#
# Cost: the first run builds a per-subject cache by reading whole electrode
# signals out of the 52 GB tree — CPU-bound, a few minutes per subject, ~260 MB
# of cache each. Later runs reuse it. Training is 4 causal folds per subject.
set -euo pipefail

ROOT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/braintreebank"
SEED="${SEED:-42}"

COMMON=(
    --model_kwargs_json configs/steegformer_small.json
    --steegformer_variant small
    --win_sec 1.5 --pre_sec 0.0
    --n_folds 4 --val_frac 0.15
    --epochs 100 --patience 20 --batch_size 32
    --lr 3e-4 --weight_decay 0.005 --warmup_epochs 5
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2
    --seed "$SEED"
)

# Gated fusion is the BrainTreebank configuration reported in the paper.
echo "=== CORTEG (gated fusion) ==="
python -m experiments.run_btb_classification \
    "${COMMON[@]}" \
    --merge_strategy layerwise_gate \
    --save_root "$ROOT/corteg_gated_seed$SEED"

# Fixed fusion at layer k, for comparison.
echo "=== CORTEG (fixed fusion at layer k) ==="
python -m experiments.run_btb_classification \
    "${COMMON[@]}" \
    --merge_strategy average \
    --save_root "$ROOT/corteg_average_seed$SEED"

# Random-init control: the same architecture without the pretrained backbone.
echo "=== random init (control) ==="
python -m experiments.run_btb_classification \
    --no_pretrained \
    --steegformer_variant small \
    --win_sec 1.5 --pre_sec 0.0 --n_folds 4 --val_frac 0.15 \
    --epochs 100 --patience 20 --batch_size 32 \
    --lr 3e-4 --weight_decay 0.005 --warmup_epochs 5 \
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2 \
    --seed "$SEED" --merge_strategy layerwise_gate \
    --save_root "$ROOT/randinit_seed$SEED"
