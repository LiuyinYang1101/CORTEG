#!/usr/bin/env bash
# Table 1, "CORTEG (ours) LOO-FT" row, Stanford finger: paper r = 0.551 ± 0.147.
# Two-stage protocol, the App. A.1 recipe:
#   Stage 1 trains pooled on the N-1 other subjects (same recipe as the pooled row).
#   Stage 2 fine-tunes the spatial adapter, LoRA and the regression head on the
#   held-out subject's training split: AdamW, base LR 1e-3 (head, LayerNorms and
#   LoRA), adapter LR 10x the base (1e-2), batch 16, 100 epochs, patience 30,
#   weight decay 5e-3, cosine schedule with 10 warmup epochs down to 1e-6.
# These are the arguments of the runs behind the paper's numbers; their logs print
# "Differential LR: head/LN=1.0e-03, LoRA=1.0e-03, adapter=1.0e-02".
#
# Notes against the paper text:
#   - This covers the full recording (f = 1.0), where the adapter multiplier is 10x.
#     The paper's 2x multiplier at f = 0.1 belongs to the low-data sweep, which
#     needs a training-fraction option this runner does not have.
#   - Stage 1 uses weight decay 0.005 (App. A.1), as the paper's Stage-1 runs did.
#   - Stage 1 passes --freeze_readout, like the pooled script: the paper's Stage-1
#     runs built their readout lazily, after the optimizer, so it stayed at its
#     initialisation (see scripts/table1_corteg_pooled_stanford.sh). Stage 2 loads
#     that readout from Stage 1 and does train it, as in the paper. Pass
#     --train_readout to train the Stage-1 readout as well (safe for both stages).
#   - Stage 1 selects its epoch on the mean validation r over all nine subjects,
#     including the held-out subject's validation split (the last 10% of its
#     training recording; never its test split), as the paper runs did. The App.
#     zero-shot table (Stage 1 alone) was read from checkpoints selected this way,
#     so "without ever having seen the held-out subject during Stage 1" holds for
#     the gradient updates but not for the epoch choice. Stage 2 trains on that
#     subject's training split anyway, so the Table 1 LOO-FT row is unaffected.
#     The "score" in Stage 1's results_pooled.json likewise averages the eight
#     trained subjects with the held-out subject's zero-shot r.
#   - Stage 2 starts from Stage 1's trainable_weights.pt, as the paper runs did.
#
# Usage (from the repository root, with the environment that has the dependencies
# active; the script calls `python`):
#   bash scripts/table1_corteg_loo_ft.sh bp                 # holds out subject bp
#   bash scripts/table1_corteg_loo_ft.sh bp --data_root DIR # extra flags go to both stages
# Extra flags must suit both stages: never pass --save_root (Stage 2 reads Stage 1's
# checkpoint from under this script's output directory) or --freeze_readout (Stage 2
# is a finetune run, which rejects it).
#
# Table 1 figure, once all nine subjects have run (mean ± sample SD, as the paper
# reports; the paper runs give 9 0.551 0.147):
#   python -c 'import glob,json,os,statistics as s; r=os.environ.get("CORTEG_OUTPUT_ROOT") or os.environ.get("ECOG_OUTPUT_ROOT") or os.path.expanduser("~/workspace/outputs/corteg"); v=[json.load(open(f))["score"] for f in glob.glob(r+"/table1/loo_ft_stanford/*/stage2/results_persub_*.json")]; print(len(v), round(s.mean(v),3), round(s.stdev(v),3))'
set -euo pipefail

HELDOUT="${1:?usage: $0 <held_out_subject> [extra runner flags]}"
shift
# An unknown name would exclude nobody: Stage 1 would train on all nine subjects
# for about two GPU-hours before Stage 2 failed on the name.
case "$HELDOUT" in
    bp|cc|ht|jc|jp|mv|wc|wm|zt) ;;
    *) echo "unknown Stanford subject: $HELDOUT (expected one of bp cc ht jc jp mv wc wm zt)" >&2
       exit 2 ;;
esac

OUT="${CORTEG_OUTPUT_ROOT:-${ECOG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}}/table1/loo_ft_stanford/${HELDOUT}"
mkdir -p "$OUT"

# --- Stage 1: pooled training on N-1 subjects ---
python -m experiments.run_regression_hilo_clean \
    --dataset Stanford \
    --train_mode pooled \
    --exclude_subjects "$HELDOUT" \
    --steegformer_variant small \
    --model_kwargs_json configs/steegformer_small.json \
    --epochs 100 \
    --early_stop_patience 90 \
    --batch_size 64 \
    --lr 3e-3 \
    --weight_decay 0.005 \
    --sched cosine --warmup_epochs 10 --min_lr 1e-5 --use_amp \
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2 \
    --lora_targets "qkv,proj,fc1,fc2" \
    --channel_adapter knn_soft_fourier --knn_k 8 \
    --hi_patch_size 25 --hi_inject_last_n 4 \
    --merge_strategy average --stream both \
    --xyz_mode real --adapter_branch both \
    --freeze_readout \
    --seed 42 \
    --save_root "$OUT/stage1" \
    "$@"

# --- Stage 2: per-subject calibration on the held-out subject (f = 1.0) ---
# No --finetune_lr_lora: LoRA falls back to the base --lr, as in the paper runs.
python -m experiments.run_regression_hilo_clean \
    --dataset Stanford \
    --train_mode finetune \
    --finetune_from "$OUT/stage1/checkpoints/trainable_weights.pt" \
    --finetune_subjects "$HELDOUT" \
    --finetune_modules "head,lora,adapter" \
    --steegformer_variant small \
    --model_kwargs_json configs/steegformer_small.json \
    --lr 1e-3 \
    --finetune_lr_adapter 1e-2 \
    --epochs 100 \
    --early_stop_patience 30 \
    --batch_size 16 \
    --weight_decay 0.005 \
    --sched cosine --warmup_epochs 10 --min_lr 1e-6 --use_amp \
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2 \
    --lora_targets "qkv,proj,fc1,fc2" \
    --channel_adapter knn_soft_fourier --knn_k 8 \
    --hi_patch_size 25 --hi_inject_last_n 4 \
    --merge_strategy average --stream both \
    --xyz_mode real --adapter_branch both \
    --seed 42 \
    --save_root "$OUT/stage2" \
    "$@"
