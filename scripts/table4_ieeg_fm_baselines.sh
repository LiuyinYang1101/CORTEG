#!/usr/bin/env bash
# Intracranial foundation models on BrainTreebank, under CORTEG's protocol.
#
# Third-party code and weights are not redistributed. Point these at your own
# copies (download sources are in README.md):
#   BRAINBERT_REPO  BRAINBERT_WEIGHTS
#   POPT_REPO       POPT_WEIGHTS
#   BRANT_SRC       BRANT_WEIGHTS
# BTB_DATA_ROOT must also be set. POPT_REPO is required even for BrainBERT and
# Brant: the shared electrode selection lives there.
set -euo pipefail

ROOT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}/braintreebank/fm_runs"
SEED="${SEED:-42}"

# The headline rows for BrainBERT and Brant are single_elec_max. That arm picks
# the best electrode on the same folds that score it, so single_elec_mean is run
# alongside it: without the non-oracle number the max is not interpretable.
for FM in brainbert brant; do
  for ARM in single_elec_max single_elec_mean; do
    echo "=== $FM / $ARM / sentence_onset ==="
    python -m experiments.run_ieeg_fm_baselines \
        --fm "$FM" --endpoint sentence_onset --arm "$ARM" \
        --seed "$SEED" --save_root "$ROOT"

    echo "=== $FM / $ARM / word_nonword ==="
    python -m experiments.run_ieeg_fm_baselines \
        --fm "$FM" --endpoint word_nonword --arm "$ARM" \
        --seed "$SEED" --save_root "$ROOT"
  done
done

# PopT is a population model: it pools over electrodes by construction, so the
# per-electrode arms do not apply.
for EP in sentence_onset word_nonword; do
  echo "=== popt / pop_meanpool / $EP ==="
  python -m experiments.run_ieeg_fm_baselines \
      --fm popt --endpoint "$EP" --arm pop_meanpool \
      --seed "$SEED" --save_root "$ROOT"
done
