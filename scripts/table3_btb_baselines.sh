#!/usr/bin/env bash
# Tables 3, 9 and 20 — the from-scratch BrainTreebank baselines (HiLoFuseNet,
# CNN-LSTM, LSTM) on both endpoints.
#
# The recipe is the runner's default, spelled out below so this file documents
# it: one pooled model per fold over the ten subjects, 4 causal folds with a
# 15 % causal validation block, BCE-with-logits, AdamW lr 1e-3 and weight decay
# 1e-4, batch 64, at most 40 epochs with patience 10 on the mean per-subject
# validation AUROC. The decoders take no electrode coordinates: each subject's
# electrodes are zero-padded to the widest subject and channel k is simply its
# k-th electrode, not aligned across subjects. That is how the published
# numbers were produced; --train_mode per_subject is the alternative.
#
# Prerequisites: the same as scripts/table3_corteg_braintreebank.sh
#   BTB_DATA_ROOT  the BrainTreebank download (~52 GB), from https://braintreebank.dev/
#   POPT_REPO      a checkout of https://github.com/czlwang/PopulationTransformer,
#                  for the shared electrode selection (not redistributed here)
#
# The baselines read CORTEG's own features and cache, so after the CORTEG
# script has run nothing is re-extracted. Each published cell averages seeds
# 42, 1 and 2, and all three score one event set drawn with seed 42: --seed
# changes only the initialisation and batch order, --event_seed picks the
# events. GPU training is not bit-deterministic, so a rerun agrees with the
# paper to within run-to-run noise, not digit for digit; check it with
# scripts/aggregate_btb.py, whose default tolerances are 1.5 x the run-to-run
# floor measured on the Task B gate arm at seed 42. A single-seed rerun is
# checked seed by seed against paper_cells/btb.
#
# Subsets: SEEDS, ENDPOINTS and DECODERS choose what runs, e.g. one job per cell:
#   SEEDS=42 ENDPOINTS=word_nonword DECODERS=LSTM bash scripts/table3_btb_baselines.sh
#
# Output: $ROOT/base_{HiLoFuseNet,CNN_LSTM,LSTM}/
#           results_pooled_{sentence_onset,word_nonword}_s{42,1,2}.json, each with
#           per-subject and per-fold AUROCs and the split report of every fold;
#           summary_pooled_<endpoint>.json, the published cell of those seeds
#           (with a _seeds<...> suffix when SEEDS is not "42 1 2").
#
# Cost: once CORTEG's cache exists, each run reads the ten cached subjects and
# trains 4 folds of at most 40 epochs; the full set is 18 runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

ROOT="${CORTEG_OUTPUT_ROOT:-${ECOG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}}/braintreebank"
SEEDS="${SEEDS:-${SEED:-42 1 2}}"
ENDPOINTS="${ENDPOINTS:-sentence_onset word_nonword}"
DECODERS="${DECODERS:-HiLoFuseNet CNN_LSTM LSTM}"

for DEC in $DECODERS; do
  for EP in $ENDPOINTS; do
    for SEED in $SEEDS; do
      echo "=== $DEC / $EP / seed $SEED ==="
      python -m experiments.run_btb_baselines \
          --decoder "$DEC" --endpoint "$EP" \
          --seed "$SEED" --event_seed 42 \
          --train_mode pooled --neg_mode upstream \
          --n_folds 4 --val_frac 0.15 \
          --epochs 40 --patience 10 --batch_size 64 \
          --lr 1e-3 --weight_decay 1e-4 \
          --hidden 256 --dropout 0.5 --D 16 \
          --save_root "$ROOT/base_$DEC"
    done
    # The published cell: 3-seed mean per subject, then mean ± cross-subject
    # SD over the ten subjects, plus the cross-seed SD of the cohort mean.
    python -m experiments.run_btb_baselines \
        --summarize --decoder "$DEC" --endpoint "$EP" --seeds $SEEDS \
        --save_root "$ROOT/base_$DEC"
  done
done
