#!/usr/bin/env bash
# Intracranial foundation models on Stanford finger flexion: the BrainBERT, PopT
# and Brant rows of Table 1 (finger column) and the Stanford cells of Tables
# 18-19 (App. A.11). Each Table 1 row is the model's best adaptation; this runs
# every adaptation the paper evaluated, so the choice can be checked.
#
# Needs, before the first run:
#   * the native 1 kHz windows: python -m data.stanford_native (reads the raw
#     <sub>/<sub>_fingerflex.mat download; see data/stanford_native.py; give it
#     the same --data_root as this script);
#   * your own copies of the third-party models, which are not redistributed
#     (download sources in ieeg_fm.py):
#       BRAINBERT_REPO  BRAINBERT_WEIGHTS  POPT_REPO  [POPT_WEIGHTS]  BRANT_SRC  BRANT_WEIGHTS
#
# ONLY_TABLE1=1 runs just the three Table 1 cells (BrainBERT and PopT temporal
# head per subject, Brant per subject). RUN_BB_FT_POOLED=1 adds pooled
# BrainBERT fine-tuning, a cell Table 19 does not report (see its block below);
# it needs more than 32 GB of GPU memory.
#
# Arguments. With none, or only --device cuda|auto and --data_root DIR, this is
# the paper run, on a GPU with AMP as the paper's were. Results go to
# $CORTEG_OUTPUT_ROOT/ieeg_fm_regression/. A cell is skipped when its result
# file already records this exact run: all nine subjects, the same device type
# and the same value of every setting (the runners' --skip_if_done). The flags
# that narrow or change a run (--subjects, --skip_subjects, --max_windows,
# --epochs, --no_amp) are also passed to every run. With any of them, and on a
# CPU (--device cpu, or no CUDA device visible to torch: fp32, whose numbers
# differ from the paper's), results AND embedding caches go to
# $CORTEG_OUTPUT_ROOT/ieeg_fm_regression_smoke/ instead, so a smoke pass can
# neither stand in for a paper cell nor hand one its cache. A smoke pass with
# fewer than two subjects skips the four leave-one-subject-out cells. Any
# other flag is refused, because the two runners do not share it; call a
# runner directly for, e.g., --reref or --bb_pool. The script can be started
# from any directory; a relative --data_root is taken from where it is started.
#
# Values next to each block are the paper's (mean +/- SD over 9 subjects)
# unless marked otherwise. The three-seed cells average each subject over
# seeds first; each run prints its own seed's score. To compare a finished
# grid with the paper:
#   python scripts/aggregate_fm.py --cells "$CORTEG_OUTPUT_ROOT/ieeg_fm_regression"
set -euo pipefail

usage() {
  echo "allowed arguments: --device DEV, --data_root DIR (paper run on a GPU);" >&2
  echo "  --subjects A,B, --skip_subjects A,B, --max_windows N, --epochs N, --no_amp" >&2
  echo "  (smoke pass, written to ieeg_fm_regression_smoke/). Other flags: call the runner." >&2
  exit 2
}

# The runners are started as `python -m experiments.<runner>` from the
# repository root (below), so a path relative to the caller is made absolute.
abspath() { case "$1" in /*) printf '%s\n' "$1" ;; *) printf '%s\n' "$PWD/$1" ;; esac; }

ALL_SUBJECTS="bp cc ht jc jp mv wc wm zt"
EXTRA=("$@")
SMOKE=0
DATA_ROOT_ARG=""
DEVICE_ARG=auto
SUBJECTS_ARG=""
SKIP_ARG=""
i=0
while (( i < ${#EXTRA[@]} )); do
  a="${EXTRA[i]}"
  case "$a" in
    --device|--data_root|--subjects|--skip_subjects|--max_windows|--epochs)
      if (( i + 1 >= ${#EXTRA[@]} )); then echo "$a needs a value" >&2; usage; fi
      i=$((i + 1))
      case "$a" in
        --data_root) EXTRA[i]="$(abspath "${EXTRA[i]}")"; DATA_ROOT_ARG="${EXTRA[i]}" ;;
        --device) DEVICE_ARG="${EXTRA[i]}" ;;
        --subjects) SUBJECTS_ARG="${EXTRA[i]}"; SMOKE=1 ;;
        --skip_subjects) SKIP_ARG="${EXTRA[i]}"; SMOKE=1 ;;
        *) SMOKE=1 ;;
      esac ;;
    --device=*) DEVICE_ARG="${a#--device=}" ;;
    --data_root=*) DATA_ROOT_ARG="$(abspath "${a#--data_root=}")"
                   EXTRA[i]="--data_root=$DATA_ROOT_ARG" ;;
    --subjects=*) SUBJECTS_ARG="${a#--subjects=}"; SMOKE=1 ;;
    --skip_subjects=*) SKIP_ARG="${a#--skip_subjects=}"; SMOKE=1 ;;
    --max_windows=*|--epochs=*|--no_amp) SMOKE=1 ;;
    *) echo "unsupported argument: $a" >&2; usage ;;
  esac
  i=$((i + 1))
done
case "$DEVICE_ARG" in
  auto|cpu|cuda) ;;
  *) echo "--device must be auto, cpu or cuda, not '$DEVICE_ARG'" >&2; usage ;;
esac

# Subjects this pass runs on (the runners reject an unknown name themselves).
N_SUBJ=0
for S in $ALL_SUBJECTS; do
  if [[ -n "$SUBJECTS_ARG" && ",$SUBJECTS_ARG," != *",$S,"* ]]; then continue; fi
  if [[ ",$SKIP_ARG," == *",$S,"* ]]; then continue; fi
  N_SUBJ=$((N_SUBJ + 1))
done

ROOT="$(abspath "${CORTEG_OUTPUT_ROOT:-${ECOG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}}")"
DATA="$(abspath "${DATA_ROOT_ARG:-${CORTEG_DATA_ROOT:-${ECOG_DATA_ROOT:-$HOME/workspace/datasets/stanford_ecog}}}")"
NATIVE="$(abspath "${CORTEG_NATIVE1K_ROOT:-$DATA/native_1k/built}")"
# The runners read the pickles from $CORTEG_DATA_ROOT when --data_root is not
# given; pass the same (now absolute) folder the native files were found by.
export CORTEG_DATA_ROOT="$DATA"
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# The paper cells ran on GPUs with AMP. A CPU run is fp32 and gives other
# numbers, so it is a smoke pass: it must not fill the paper folders or caches.
if (( ! SMOKE )); then
  if [[ "$DEVICE_ARG" == cpu ]]; then
    echo "--device cpu: running as a smoke pass (the paper ran on GPUs with AMP)"
    SMOKE=1
  elif [[ "$DEVICE_ARG" == auto ]] &&
       ! python -c 'import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "no CUDA device visible to torch: running as a smoke pass on the CPU" \
         "(the paper ran on GPUs with AMP)"
    SMOKE=1
  fi
fi

if (( SMOKE )); then
  OUT="$ROOT/ieeg_fm_regression_smoke"
  echo "smoke pass: everything goes to $OUT"
else
  OUT="$ROOT/ieeg_fm_regression"
fi
CACHE="$OUT/fm_emb_cache"

# A smoke pass may use fewer subjects; the runner names any file it lacks.
if (( ! SMOKE )); then
  for S in $ALL_SUBJECTS; do
    if [[ ! -f "$NATIVE/${S}_native1k.npz" ]]; then
      echo "missing $NATIVE/${S}_native1k.npz -- build it with:" \
           "python -m data.stanford_native --data_root $DATA" >&2
      exit 1
    fi
  done
fi

# reg <save_dir> <flags...>: BrainBERT / PopT. The runner itself skips a cell
# whose folder already records this exact run.
reg() {
  local save="$1"; shift
  echo "=== $save"
  if (( N_SUBJ < 2 )) && [[ " $* " == *" --train_mode loo "* ]]; then
    echo "    skipped: leave-one-subject-out needs at least two subjects"
    return 0
  fi
  python -m experiments.run_ieeg_fm_regression "$@" --skip_if_done \
      --native_root "$NATIVE" --emb_cache "$CACHE" --save_root "$save" ${EXTRA[@]+"${EXTRA[@]}"}
}

# brant <save_dir> <flags...>
brant() {
  local save="$1"; shift
  echo "=== $save"
  python -m experiments.run_brant_regression "$@" --skip_if_done \
      --native_root "$NATIVE" --emb_cache "$CACHE" --save_root "$save" ${EXTRA[@]+"${EXTRA[@]}"}
}

# ---- Table 1 cells -----------------------------------------------------------
# Temporal head (2-layer BiLSTM over 64-window sequences of frozen embeddings),
# per subject: BrainBERT 0.053 +/- 0.056, PopT 0.063 +/- 0.046.
for FM in brainbert popt; do
  reg "$OUT/temporal/Stanford/$FM/per_subject/seed42" \
      --fm "$FM" --mode probe --head temporal --train_mode per_subject --seed 42 \
      --seq_len 64 --temporal_hidden 128 --epochs 100 --early_stop_patience 20 \
      --lr 1e-3 --use_amp
done

# Brant, 6 s context on a 0.5 s anchor grid, per subject: 0.028 +/- 0.031.
brant "$OUT/brant/Stanford/brant/per_subject/seed42" \
    --train_mode per_subject --context_patches 1 --stride_s 0.5 --head linear \
    --extract_batch 16 --epochs 200 --early_stop_patience 30 --use_amp --seed 42

if [[ "${ONLY_TABLE1:-0}" == 1 ]]; then
  exit 0
fi

# ---- Brant on a 0.1 s anchor grid (App. A.11 text): 0.025 ---------------------
# Quoted in the App. A.11 text only: paper_cells/ holds no result file for it,
# so a re-run can be compared with the mean, not seed by seed or per subject.
brant "$OUT/brant_s0.1/Stanford/brant/per_subject/seed42" \
    --train_mode per_subject --context_patches 1 --stride_s 0.1 --head linear \
    --extract_batch 16 --epochs 200 --early_stop_patience 30 --use_amp --seed 42

# ---- Temporal head, pooled (per-subject readouts) ------------------------------
# Archived seed-42 runs, not printed in the paper (Table 18 shows the better
# per-subject regime): BrainBERT 0.043, PopT 0.055.
for FM in brainbert popt; do
  reg "$OUT/temporal/Stanford/$FM/pooled/seed42" \
      --fm "$FM" --mode probe --head temporal --train_mode pooled --seed 42 \
      --seq_len 64 --temporal_hidden 128 --epochs 100 --early_stop_patience 20 \
      --lr 1e-3 --use_amp
done

# ---- Frozen linear probe --------------------------------------------------------
# pooled (3 seeds):      BrainBERT 0.030, PopT 0.040
# per subject (3 seeds): BrainBERT 0.039, PopT 0.049
# LOO (Table 18 rows):   BrainBERT 0.044 +/- 0.051, PopT 0.050 +/- 0.045
# The pooled cells run on a warm embedding cache (the temporal cells above
# filled it). Pooled probe training is not reseeded after extraction, so a
# cold-cache run of the same seed gives another number; whether the paper's
# pooled runs were warm or cold is not recorded (see the runner's docstring).
for FM in brainbert popt; do
  for REGIME in pooled per_subject; do
    for SEED in 42 0 1; do
      reg "$OUT/probe/Stanford/$FM/$REGIME/seed$SEED" \
          --fm "$FM" --mode probe --head linear --train_mode "$REGIME" --seed "$SEED" \
          --epochs 60 --early_stop_patience 20 --use_amp
    done
  done
  reg "$OUT/probe/Stanford/$FM/loo/seed42" \
      --fm "$FM" --mode probe --head linear --train_mode loo --seed 42 \
      --epochs 60 --early_stop_patience 20 --use_amp
done

# ---- Last-2-block fine-tuning ------------------------------------------------------
# per subject (3 seeds): BrainBERT 0.033, PopT 0.037
# pooled (3 seeds):      PopT 0.041 +/- 0.046 (its Table 18 row); BrainBERT below
# LOO:                   BrainBERT 0.047 +/- 0.038 (its Table 18 row), PopT 0.036
for FM in brainbert popt; do
  for SEED in 42 0 1; do
    reg "$OUT/ft/Stanford/$FM/per_subject/seed$SEED" \
        --fm "$FM" --mode finetune --unfreeze_last_n 2 --ft_lr 1e-4 --warmup_epochs 5 \
        --train_mode per_subject --seed "$SEED" --epochs 60 --early_stop_patience 20 --use_amp
  done
  reg "$OUT/ft/Stanford/$FM/loo/seed42" \
      --fm "$FM" --mode finetune --unfreeze_last_n 2 --ft_lr 1e-4 --warmup_epochs 5 \
      --train_mode loo --seed 42 --epochs 60 --early_stop_patience 20 --use_amp
done
for SEED in 42 0 1; do
  reg "$OUT/ft/Stanford/popt/pooled/seed$SEED" \
      --fm popt --mode finetune --unfreeze_last_n 2 --ft_lr 1e-4 --warmup_epochs 5 \
      --train_mode pooled --seed "$SEED" --epochs 60 --early_stop_patience 20 --use_amp
done

# Pooled BrainBERT fine-tuning: Table 19 prints "---" here, because the run is
# incomplete. Only the seed-42 run (r = 0.039) was archived; the authors'
# revision notes give seeds 0 and 1 as 0.030 and 0.036, from cluster runs
# whose outputs were never retrieved. RUN_BB_FT_POOLED=1 re-runs all three
# seeds, should the cell be restored. Needs more than 32 GB of GPU memory.
if [[ "${RUN_BB_FT_POOLED:-0}" == 1 ]]; then
  for SEED in 42 0 1; do
    reg "$OUT/ft/Stanford/brainbert/pooled/seed$SEED" \
        --fm brainbert --mode finetune --unfreeze_last_n 2 --ft_lr 1e-4 --warmup_epochs 5 \
        --train_mode pooled --seed "$SEED" --epochs 60 --early_stop_patience 20 --use_amp
  done
fi
