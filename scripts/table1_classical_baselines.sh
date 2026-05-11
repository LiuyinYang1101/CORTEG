#!/usr/bin/env bash
# Table 1, classical-baseline rows on Stanford finger:
# Ridge_LFS, Ridge_HGA, PLS, HOPLS, LSTM_LFS, LSTM_HGA, CNN-LSTM, HiLoFuseNet.
#
# DeepFingerNet numbers are transcribed from its original paper (Table II);
# we do not retrain it here. See baselines/THIRDPARTY.md.
set -euo pipefail

OUT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/table1/stanford_classical"
mkdir -p "$OUT"

for MODEL in Ridge_LFS Ridge_HGA PLS HOPLS LSTM_LFS LSTM_HGA CNN_LSTM HiLoFuseNet; do
    python -m experiments.run_ecog_baselines \
        --dataset Stanford \
        --model "$MODEL" \
        --train_mode per_subject \
        --epochs 100 \
        --batch_size 64 \
        --use_amp \
        --seed 42 \
        --save_root "$OUT/$MODEL"
done
