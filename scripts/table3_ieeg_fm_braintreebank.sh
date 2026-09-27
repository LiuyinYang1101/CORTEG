#!/usr/bin/env bash
# Intracranial foundation models on BrainTreebank, both endpoints (Task A,
# sentence onset; Task B, word vs non-word): the iEEG-FM rows of Tables 3 and 9
# and of App. Table 20.
#
#   frozen arms, experiments/run_ieeg_fm_baselines.py
#     BrainBERT / Brant  single_elec_max    Tables 3, 9: BrainBERT-dagger, Brant-dagger
#                        single_elec_mean   Table 20
#                        pop_meanpool       Table 20, "population mean-pool"
#     PopT               pop_meanpool       Table 20, "PopT, frozen probe"
#   trained PopT arms, experiments/run_popt_finetune_btb.py
#     lora                                  Tables 3, 9: PopT; Table 20, "PopT, LoRA"
#     full_ft                               Table 20, "PopT, full fine-tune"
#     head_only                             Table 20, "PopT, head-only"
#
# Not released: the BrainBERT and Brant head-only / LoRA / full fine-tune rows of
# Table 20.
#
# The FM rows are one seed. SEED seeds the probes and the PopT training only;
# every arm scores the same events, drawn once with event seed 42.
#
# Third-party code and weights are not redistributed. Point these at your own
# copies (download sources are in README.md):
#   BRAINBERT_REPO  BRAINBERT_WEIGHTS
#   POPT_REPO       POPT_WEIGHTS (optional: unset, PopT's checkpoint is
#                   downloaded from Hugging Face)
#   BRANT_SRC       BRANT_WEIGHTS
# BTB_DATA_ROOT must also be set. POPT_REPO is required even for BrainBERT and
# Brant: the shared electrode selection lives there. The BrainBERT and PopT
# checkpoints need omegaconf, and the PopT download needs huggingface_hub
# (both: pip install -e .[ieeg-fm]).
#
# Stages. Only the embedding passes and PopT need a model and a GPU. The
# per-electrode probes, C electrodes x 4 folds logistic fits per FM and task and
# most of the CPU time, read the cached embeddings and use no GPU. STAGE picks
# what runs:
#   STAGE=all     (default) both stages, in order
#   STAGE=gpu     BrainBERT embeddings, PopT frozen and trained, Brant embeddings
#   STAGE=probe   the BrainBERT and Brant probes, from the caches the gpu stage
#                 wrote (a missing cache is an error, never a silent rebuild)
# On a shared GPU, run STAGE=gpu on it and then STAGE=probe on CPU alone.
#
# Order. BrainBERT runs first and caches its per-electrode embeddings, which
# every PopT arm (frozen and trained) reads instead of recomputing. Brant runs
# last: it is the costliest step and the one most likely to fail, and the PopT
# rows do not depend on it.
#
# Cost. BrainBERT reads its windows per subject (~15 GB for the largest subject
# on Task B) and runs in chunks of 200 events. Brant first resamples each whole
# recording to 250 Hz (CPU, roughly 45 s per channel at 2 kHz, so up to about an
# hour per subject) and caches it under <output root>/braintreebank/
# brant_stream_cache/, where every later Brant run finds it. One probe call
# scores all three arms, and single_elec_max and single_elec_mean share one set
# of per-electrode fits. N_JOBS runs those fits in parallel; the default,
# min(12, cores - 4), is what the paper's Brant arm used, and the numbers do
# not change with it.
#
# Failures. Each call runs on its own: a failed call is reported, the rest
# still run, and the script exits non-zero at the end, listing what failed.
set -uo pipefail

OUT="${CORTEG_OUTPUT_ROOT:-${ECOG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}}"
ROOT="$OUT/braintreebank/fm_runs"
SEED="${SEED:-42}"
STAGE="${STAGE:-all}"
if [[ -z "${N_JOBS:-}" ]]; then
  N_JOBS=$(( $(nproc) - 4 ))
  if (( N_JOBS > 12 )); then N_JOBS=12; fi
  if (( N_JOBS < 1 )); then N_JOBS=1; fi
fi
case "$STAGE" in
  all|gpu|probe) ;;
  *) echo "STAGE must be all, gpu or probe, not '$STAGE'" >&2; exit 2 ;;
esac

FAILED=()
run() {                     # one call; a failure is recorded, not fatal
  echo "=== $* ==="
  "$@"
  local rc=$?
  if (( rc != 0 )); then
    echo "!!! FAILED (exit $rc): $*" >&2
    FAILED+=("$*")
  fi
}

if [[ "$STAGE" != probe ]]; then
  # BrainBERT per-electrode embeddings. PopT reads this cache.
  for EP in sentence_onset word_nonword; do
    run python -m experiments.run_ieeg_fm_baselines \
        --fm brainbert --endpoint "$EP" --embed_only
  done

  # PopT is a population model: it pools over electrodes by construction, so
  # the per-electrode arms do not apply. Its frozen probe is Table 20's frozen
  # row; it is one cheap probe per fold, so it runs here, next to its GPU pass.
  for EP in sentence_onset word_nonword; do
    run python -m experiments.run_ieeg_fm_baselines \
        --fm popt --endpoint "$EP" --arm pop_meanpool \
        --seed "$SEED" --save_root "$ROOT"
  done

  # PopT trained per subject on the cached BrainBERT embeddings. lora is the PopT
  # row of Tables 3 and 9 (0.600 / 0.779).
  for EP in sentence_onset word_nonword; do
    for MODE in lora full_ft head_only; do
      run python -m experiments.run_popt_finetune_btb \
          --endpoint "$EP" --mode "$MODE" \
          --seed "$SEED" --save_root "$ROOT"
    done
  done

  # Brant embeddings, from the 250 Hz stream (built on first use).
  for EP in sentence_onset word_nonword; do
    run python -m experiments.run_ieeg_fm_baselines \
        --fm brant --endpoint "$EP" --embed_only
  done
fi

if [[ "$STAGE" != gpu ]]; then
  # BrainBERT and Brant probes, CPU only. The headline rows are single_elec_max.
  # That arm picks the best electrode on the same folds that score it, so its
  # null is at least ~0.53 rather than 0.50; single_elec_mean comes from the
  # same fits, and without that non-oracle number the max is not interpretable.
  # pop_meanpool gives Table 20's population rows.
  for FM in brainbert brant; do
    for EP in sentence_onset word_nonword; do
      run python -m experiments.run_ieeg_fm_baselines \
          --fm "$FM" --endpoint "$EP" \
          --arm single_elec_max single_elec_mean pop_meanpool \
          --from_cache --device cpu --n_jobs "$N_JOBS" \
          --seed "$SEED" --save_root "$ROOT"
    done
  done
fi

if (( ${#FAILED[@]} )); then
  echo "${#FAILED[@]} call(s) failed:" >&2
  printf '  %s\n' "${FAILED[@]}" >&2
  exit 1
fi
