#!/usr/bin/env bash
# Table 2 — Ablation study. Each row modifies one component of full CORTEG-S
# on the Stanford finger-flexion task.
#
# Argparse uses last-wins for duplicate flags, so per-row overrides at the end
# correctly replace the defaults set in $COMMON.
set -euo pipefail

ROOT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/table2_stanford"
mkdir -p "$ROOT"

COMMON_BASE=(
    --dataset Stanford --train_mode pooled --steegformer_variant small
    --model_kwargs_json configs/steegformer_small.json
    --epochs 100 --early_stop_patience 90 --batch_size 64
    --lr 3e-3 --weight_decay 0.005
    --sched cosine --warmup_epochs 10 --min_lr 1e-5 --use_amp
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2
    --lora_targets "qkv,proj,fc1,fc2"
    --channel_adapter knn_soft_fourier --knn_k 8
    --hi_patch_size 25 --hi_inject_last_n 4
    --merge_strategy average --stream both
    --xyz_mode real --adapter_branch both
    --seed 42
)

# --use_ecog_fuser is store_true, so a row that needs it OFF cannot override it
# later; it must simply not be passed. Keep one source of truth for the rest.
COMMON=( "${COMMON_BASE[@]}" --use_ecog_fuser )

run() {
    local name="$1"; shift
    echo "=== $name ==="
    python -m experiments.run_regression_hilo_clean \
        "${COMMON[@]}" "$@" \
        --save_root "$ROOT/$name"
}

run_no_fuser() {
    local name="$1"; shift
    echo "=== $name ==="
    python -m experiments.run_regression_hilo_clean \
        "${COMMON_BASE[@]}" "$@" \
        --save_root "$ROOT/$name"
}

# Full model (Table 2 top row)
run "00_full"

# Foundation-model ablation
run "01_random_init"        --no_pretrained --full_finetune
# LaBraM, CBraMod, MantisV2 use dedicated runners (see experiments/run_{labram,cbramod,mantis}_baseline.py).

# Adaptation strategy
run "10_full_ft_no_lora"    --full_finetune
# Row 11 needs the fuser off; the published 0.529 run also used lr 1e-3.
run_no_fuser "11_lora_no_adapter"  --channel_adapter none --lr 1e-3

# Training regime
run "20_per_subject"        --train_mode per_subject
# LOO-FT row: use scripts/table1_corteg_loo_ft.sh per subject.

# Input stream
run "30_high_gamma_only"    --stream hi_only
run "31_low_freq_only"      --stream lo_only

# Electrode geometry
run "40_random_xyz"         --xyz_mode random
run "41_zero_xyz"            --xyz_mode zero

# Adapter branch
run "50_fourier_only"        --adapter_branch fourier_only
run "51_soft_only"           --adapter_branch soft_only
