#!/usr/bin/env bash
# Tables 3, 9 and 20 — CORTEG on BrainTreebank, Task A and Task B.
#
# Task A (--endpoint sentence_onset): sentence-initial vs mid-sentence word.
# Task B (--endpoint word_nonword):   word vs non-word, the BrainBERT/PopT benchmark.
# CORTEG reads the window [t, t+1.5] s on both tasks. On Task B a non-word
# event t is the centre of a 1 s word-free tile, so only [t, t+0.5] of its
# window is guaranteed free of speech: in 8-18 % of each subject's negatives
# (14 % on average) the window reaches the next word's onset. The paper runs
# did the same; the events are the ones the 5 s foundation-model arms score.
#
# Three arms, each on both tasks and training seeds 42, 1 and 2:
#   gate      gated fusion, --merge_strategy layerwise_gate: the paper's CORTEG row
#   average   mean-pool fusion, --merge_strategy average: Table 20 "CORTEG (mean-pool fusion)"
#   randinit  the random-init control: the gate arm with --no_pretrained, keeping the
#             backbone config (drop_path_rate 0.1) and every training setting
# All seeds score ONE event draw (--event_seed 42), so the seed SD measures
# initialisation only. The paper averages the three seeds within each subject,
# then reports mean ± SD (ddof=1) over the 10 subjects.
#
# The recipe below is the runner's default, spelled out so this file documents
# it: one pooled model over the 10 subjects with a per-subject LoRA adapter
# (r 4, alpha 16, dropout 0.2, last 4 blocks), 60 epochs, AdamW lr 3e-4 and
# weight decay 5e-3, batch 16 with 4-step accumulation, bf16 autocast, cosine
# schedule with max(1, epochs/10) warmup epochs down to 1e-5, validation every
# 2 epochs with patience 15 evaluations, early stopping on the pooled
# validation AUROC, 4 causal folds with a 15 % causal validation block.
#
# Prerequisites
#   BTB_DATA_ROOT  the BrainTreebank download (~52 GB), from https://braintreebank.dev/
#   POPT_REPO      a checkout of https://github.com/czlwang/PopulationTransformer,
#                  for the shared electrode selection (not redistributed here)
#
# Usage: the script takes no arguments. SEEDS, ENDPOINTS and ARMS choose what
# runs, e.g. one job per cell:
#   SEEDS=42 ENDPOINTS=word_nonword ARMS=gate bash scripts/table3_corteg_braintreebank.sh
# Any argument (--help, a typo) prints this header and exits without running.
#
# Output root: CORTEG_OUTPUT_ROOT, else ECOG_OUTPUT_ROOT, else
# ~/workspace/outputs/corteg -- the order paths.get_output_root() uses, so the
# results land next to the caches the runner reads ($root/braintreebank/cache).
# Output: $root/braintreebank/table3/btb_pooled_{layerwise_gate,average}_
#         {sentence_onset,word_nonword}[_randinit]_seed{42,1,2}.json, each with
#         per-subject, per-fold AUROCs and the split report of every fold.
#
# Cost: the first run of each task builds a per-subject cache by reading whole
# electrode signals out of the 52 GB tree -- CPU-bound, a few minutes per
# subject, about 0.2-0.6 GB of cache each (for all 10 subjects about 3.4 GB on
# Task A and 3.8 GB on Task B). Every seed and arm reuses it. Each run trains
# 4 folds; the full table is 18 runs.
set -euo pipefail

if [ $# -gt 0 ]; then
    awk 'NR > 1 { if (!/^#/) exit; sub(/^# ?/, ""); print }' "$0"
    case "$1" in
        -h|*-help) exit 0 ;;
        *) echo "unexpected argument '$1': this script takes none" >&2; exit 2 ;;
    esac
fi
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${CORTEG_OUTPUT_ROOT:-${ECOG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}}/braintreebank/table3"
SEEDS="${SEEDS:-${SEED:-42 1 2}}"
ENDPOINTS="${ENDPOINTS:-sentence_onset word_nonword}"
ARMS="${ARMS:-gate average randinit}"

# Check every choice before the first run, not when the loop reaches it hours in.
for ARM in $ARMS; do
    case "$ARM" in
        gate|average|randinit) ;;
        *) echo "unknown arm '$ARM' (expected gate, average or randinit)" >&2; exit 2 ;;
    esac
done
for EP in $ENDPOINTS; do
    case "$EP" in
        sentence_onset|word_nonword) ;;
        *) echo "unknown endpoint '$EP' (expected sentence_onset or word_nonword)" >&2; exit 2 ;;
    esac
done
for SEED in $SEEDS; do
    case "$SEED" in
        ''|*[!0-9]*) echo "seed '$SEED' is not a non-negative integer" >&2; exit 2 ;;
    esac
done

COMMON=(
    --model_kwargs_json configs/steegformer_small.json
    --steegformer_variant small
    --win_sec 1.5 --pre_sec 0.0 --max_per_class 900 --event_seed 42
    --n_folds 4 --val_frac 0.15
    --epochs 60 --patience 15 --eval_every 2 --select_metric pooled
    --batch_size 16 --eval_batch_size 32 --accum_iter 4 --use_amp
    --lr 3e-4 --weight_decay 0.005 --min_lr 1e-5 --head_dropout 0.0
    --lora_last_n 4 --lora_r 4 --lora_alpha 16 --lora_dropout 0.2 --per_subject_lora
    --save_root "$ROOT"
)

for EP in $ENDPOINTS; do
  for SEED in $SEEDS; do
    for ARM in $ARMS; do
      echo "=== $ARM / $EP / seed $SEED ==="
      case "$ARM" in
        gate)
          python -m experiments.run_btb_classification \
              "${COMMON[@]}" --endpoint "$EP" --seed "$SEED" \
              --merge_strategy layerwise_gate
          ;;

        average)
          python -m experiments.run_btb_classification \
              "${COMMON[@]}" --endpoint "$EP" --seed "$SEED" \
              --merge_strategy average
          ;;

        randinit)
          python -m experiments.run_btb_classification \
              "${COMMON[@]}" --endpoint "$EP" --seed "$SEED" \
              --merge_strategy layerwise_gate --no_pretrained
          ;;

        *)
          echo "unknown arm '$ARM' (expected gate, average or randinit)" >&2
          exit 2
          ;;
      esac
    done
  done
done
