#!/usr/bin/env bash
# Fig 2(b) — backbone scaling (Small / Base / Large) on Stanford finger
# Fig 2(d)  — low-data sweep (train_fraction in {0.1, 0.25, 0.5, 1.0})
set -euo pipefail

ROOT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/fig2_stanford"
mkdir -p "$ROOT"

BASE=(
    --dataset Stanford --train_mode pooled
    --epochs 100 --early_stop_patience 90 --batch_size 64
    --lr 3e-3 --weight_decay 0.005
    --sched cosine --warmup_epochs 10 --min_lr 1e-5 --use_amp
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2
    --lora_targets "qkv,proj,fc1,fc2"
    --channel_adapter knn_soft_fourier --knn_k 8
    --use_ecog_fuser --hi_patch_size 25 --hi_inject_last_n 4
    --merge_strategy average --stream both
    --xyz_mode real --adapter_branch both
    --seed 42
)

# --- Fig 2(b): backbone scaling ---
for VARIANT in small base large; do
    python -m experiments.run_regression_hilo_clean \
        "${BASE[@]}" --steegformer_variant "$VARIANT" \
        --model_kwargs_json "configs/steegformer_${VARIANT}.json" \
        --save_root "$ROOT/scaling/${VARIANT}"
done

# --- Fig 2(d): low-data sweep ---
for FRAC in 0.1 0.25 0.5 1.0; do
    python -m experiments.run_regression_hilo_clean \
        "${BASE[@]}" --steegformer_variant small \
        --model_kwargs_json configs/steegformer_small.json \
        --train_fraction "$FRAC" \
        --save_root "$ROOT/lowdata/frac_${FRAC}"
done
