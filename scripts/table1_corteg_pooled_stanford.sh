#!/usr/bin/env bash
# Table 1, "CORTEG (ours) Pooled" row, Finger (n=9).
# Reproduces r = 0.554 ± 0.154 on the Stanford fingerflex dataset.
#
# Hyperparameters: §4 of the paper.
set -euo pipefail

OUT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/table1/stanford_corteg_pooled"
mkdir -p "$OUT"

python -m experiments.run_regression_hilo_clean \
    --dataset Stanford \
    --train_mode pooled \
    --steegformer_variant small \
    --model_kwargs_json configs/steegformer_small.json \
    --epochs 100 \
    --early_stop_patience 90 \
    --batch_size 64 \
    --lr 3e-3 \
    --weight_decay 0.005 \
    --sched cosine \
    --warmup_epochs 10 \
    --min_lr 1e-5 \
    --use_amp \
    --lora_last_n 4 \
    --lora_r 4 \
    --lora_alpha 16 \
    --lora_dropout 0.2 \
    --lora_targets "qkv,proj,fc1,fc2" \
    --channel_adapter knn_soft_fourier \
    --knn_k 8 \
    --use_ecog_fuser \
    --hi_patch_size 25 \
    --hi_inject_last_n 4 \
    --merge_strategy average \
    --stream both \
    --xyz_mode real \
    --adapter_branch both \
    --seed 42 \
    --save_root "$OUT" \
    "$@"
