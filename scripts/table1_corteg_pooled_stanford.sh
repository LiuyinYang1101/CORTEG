#!/usr/bin/env bash
# Table 1, "CORTEG (ours) Pooled" row, Finger (n=9): paper r = 0.554 ± 0.154.
# checkpoints/corteg_stanford_pooled.pt is the adapter this recipe produced for the
# paper, and it reproduces 0.554 at inference (see README.md).
#
# Hyperparameters: §4 of the paper.
#
# --freeze_readout reproduces how the paper run was trained: its linear readout
# was built lazily on the first forward pass, after the optimizer, so it stayed at
# its random initialisation (the released checkpoint's readout still is). With the
# flag and --seed 42, on a GPU and with the tested versions (torch 2.11, timm
# 1.0.26), the readout starts from values bit-identical to that checkpoint's;
# other versions may draw other values. This costs little, since a random 512->5
# projection has full rank and the adapted backbone can still produce any output
# it needs. With the flag the startup log reports 294,671 trainable parameters;
# the saved checkpoint still holds 297,236, because the untouched readout is saved
# with it.
#
# To train the readout too, as the method section describes (and as the
# BrainTreebank and per-subject paths do by default), pass --train_readout:
#   bash scripts/table1_corteg_pooled_stanford.sh --train_readout
# That variant has not been retrained end to end. Refitting a ridge readout on the
# released checkpoint's own features lifts cohort r from 0.5537 to 0.5643 (+0.011),
# which is about the cross-seed SD of this configuration (~0.01). Either way a
# retrain is single-seed and GPU-nondeterministic: expect a cohort r close to
# 0.554, not identical.
#
# Usage (from the repository root, with the environment that has the dependencies
# active; the script calls `python`):
#   bash scripts/table1_corteg_pooled_stanford.sh [extra runner flags]
set -euo pipefail

OUT="${CORTEG_OUTPUT_ROOT:-${ECOG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}}/table1/stanford_corteg_pooled"
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
    --freeze_readout \
    --seed 42 \
    --save_root "$OUT" \
    "$@"
