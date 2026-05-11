#!/usr/bin/env bash
# Table 1, "CORTEG (ours) LOO-FT" row, Stanford finger.
# Two-stage protocol (§3.3): Stage 1 trains pooled on N-1 subjects;
# Stage 2 fine-tunes the spatial adapter, LoRA, and regression head on the
# held-out subject.
#
# LR schedule for Stage 2 (paper §3.3):
#   - adapter LR = 10x base LR at recording fraction f >= 0.25
#   - adapter LR = 2x  base LR at f = 0.1
# This script assumes f = 1.0 and uses the 10x multiplier.
#
# Usage:
#   bash scripts/table1_corteg_loo_ft.sh bp     # holds out subject bp
set -euo pipefail

HELDOUT="${1:?usage: $0 <held_out_subject>}"
OUT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/table1/loo_ft_stanford/${HELDOUT}"
mkdir -p "$OUT"

# --- Stage 1: pooled training on N-1 subjects ---
python -m experiments.run_regression_hilo_clean \
    --dataset Stanford \
    --train_mode pooled \
    --exclude_subjects "$HELDOUT" \
    --steegformer_variant small \
    --epochs 100 \
    --early_stop_patience 90 \
    --batch_size 64 \
    --lr 3e-3 \
    --weight_decay 0.01 \
    --sched cosine --warmup_epochs 10 --min_lr 1e-5 --use_amp \
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2 \
    --lora_targets "qkv,proj,fc1,fc2" \
    --channel_adapter knn_soft_fourier --knn_k 8 \
    --use_ecog_fuser --hi_patch_size 25 --hi_inject_last_n 4 \
    --merge_strategy average --stream both \
    --xyz_mode real --adapter_branch both \
    --seed 42 \
    --save_root "$OUT/stage1" \
    --save_ckpt_path "$OUT/stage1/checkpoint.pth"

# --- Stage 2: per-subject calibration on the held-out subject (f = 1.0) ---
python -m experiments.run_regression_hilo_clean \
    --dataset Stanford \
    --train_mode per_subject \
    --finetune_subjects "$HELDOUT" \
    --finetune_from "$OUT/stage1/checkpoint.pth" \
    --finetune_modules "head,lora,adapter" \
    --finetune_lr_lora 3e-3 \
    --finetune_lr_adapter 3e-2 \
    --epochs 30 \
    --early_stop_patience 30 \
    --batch_size 64 \
    --weight_decay 0.01 \
    --sched cosine --warmup_epochs 2 --min_lr 1e-5 --use_amp \
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 \
    --lora_targets "qkv,proj,fc1,fc2" \
    --channel_adapter knn_soft_fourier --knn_k 8 \
    --use_ecog_fuser --hi_patch_size 25 --hi_inject_last_n 4 \
    --merge_strategy average --stream both \
    --xyz_mode real --adapter_branch both \
    --seed 42 \
    --save_root "$OUT/stage2"
