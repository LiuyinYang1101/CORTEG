# Reproducing CORTEG

The detailed companion to [README.md](README.md): data preparation, checkpoints, the
intracranial foundation-model setup, the commands behind each table, the published values
to compare against, and where the released code differs from the paper text. Table,
section and appendix numbers refer to the NeurIPS 2026 camera-ready.

## Scripts and scope

| Paper artefact | Script |
| --- | --- |
| Table 1, CORTEG pooled (finger) | `scripts/table1_corteg_pooled_stanford.sh` |
| Table 1, CORTEG LOO-FT (finger) | `scripts/table1_corteg_loo_ft.sh <subject>` |
| Table 1 iEEG-FM rows (finger); Tables 18–19, Stanford cells | `scripts/table1_ieeg_fm_stanford.sh` |
| Tables 3, 9, 20: CORTEG gate, mean-pool and random-init rows | `scripts/table3_corteg_braintreebank.sh` |
| Tables 3, 9, 20: BrainBERT†, Brant†, PopT (LoRA) and the other PopT, BrainBERT and Brant arms listed below | `scripts/table3_ieeg_fm_braintreebank.sh` |
| Reprint Tables 3, 9, 20 / Table 1 FM rows, 18, 19 from `paper_cells/` | `scripts/aggregate_btb.py`, `scripts/aggregate_fm.py` |

The FM script runs, per task: BrainBERT and Brant single-electrode max, single-electrode
mean and population mean-pool; PopT frozen probe, LoRA, full fine-tune and head-only.

**Not released.**
- Anything on Ghent (the audio column of every table): the data are private.
- Table 2 and Table 4 / App. Table 17 have no script. The pooled Stanford runner has the
  switches the ablations and the fusion study use (`--no_pretrained --full_finetune`,
  `--xyz_mode`, `--stream`, `--adapter_branch`, `--merge_strategy layerwise_gate`,
  `--steegformer_variant`), but none of these has been re-run against the published
  values (Table 4 uses a multi-seed HPC protocol). The backbone swaps of Table 2
  (LaBraM, CBraMod, MantisV2, REVE) are not included.
- The LOO-FT low-data sweep (Fig. 2d,e; the 2× adapter multiplier at f = 0.1).
- BrainTreebank: the from-scratch decoders (HiLoFuseNet, CNN-LSTM, LSTM), the raw
  spectral linear probes, the BrainBERT and Brant head-only / LoRA / full fine-tune rows
  of Table 20, the oracle permutation null, and the pause-only AUROC of App. A.2.3. Their
  cells are in `paper_cells/btb` and are reprinted by `scripts/aggregate_btb.py`.
- App. A.11 adaptations described only in the text: per-electrode readouts, full-backbone
  FM fine-tuning, joint FM + temporal fine-tuning, and the representation analysis.
- Two cells of Table 19 come from runs that are incomplete in the archive, so
  `paper_cells/` cannot rebuild them: Stanford pooled BrainBERT fine-tune (printed 0.035;
  only seed 42 is archived, r = 0.039; `RUN_BB_FT_POOLED=1` re-runs all three seeds) and
  Ghent LOO BrainBERT fine-tune (printed 0.043; 3 of 16 folds archived).
- The run outputs the demo is built from (`build_demo_data.py` is an author-side script).


## Data

### Stanford `fingerflex` — public

5-finger flexion regression from ECoG on 9 epilepsy patients (`bp, cc, ht, jc,
jp, mv, wc, wm, zt`).

**Source.** Miller K. J., *A library of human electrocorticographic data and
analyses*, Nat. Hum. Behav. 2019,
[10.1038/s41562-019-0678-3](https://doi.org/10.1038/s41562-019-0678-3). Hosted on the
[Stanford Digital Repository](https://exhibits.stanford.edu/data/catalog/zk881ps0522).

**Expected layout** under `$CORTEG_DATA_ROOT`:

```
bp_features.pkl      # X_feat_tr (N,C,200,2)  X_feat_te  X_raw_tr (N,C,128)  X_raw_te  y_tr (N,5)  y_te (N,5)
bp_electrode_loc.mat # (C, 3) electrode coordinates in mm
...                  # one pair per subject
```

The pipeline that produces these files — MATLAB cleaning plus Python feature extraction, band-pass 70–200 Hz with Hilbert envelope for the high-gamma stream and 1–64 Hz for the low-frequency stream — is documented step by step in
[`data/stanford_preprocessing/tutorial_Stanford.md`](data/stanford_preprocessing/tutorial_Stanford.md).

**Native 1 kHz windows (iEEG-FM comparison only).** BrainBERT, PopT and Brant are
evaluated on the raw 1 kHz recordings, not the 128 Hz stream (App. A.11). Build them
once from the raw `<sub>/<sub>_fingerflex.mat` download plus the pickles above:

```bash
python -m data.stanford_native --data_root "$CORTEG_DATA_ROOT"
```

The raw download is read from `<data_root>/raw/Stanford` (or `--raw_root`,
`$CORTEG_STANFORD_RAW_ROOT`) and `<sub>_native1k.npz` is written to
`<data_root>/native_1k/built` (or `--out_root`, `$CORTEG_NATIVE1K_ROOT`), where the FM
runners look. A file is written only if it aligns with the CORTEG windows (median
correlation 0.998–0.999 on all nine subjects); window *n* of the native file pairs with
target *n* of the pickle.

### BrainTreebank — public

sEEG from 10 patients watching films (Wang et al., 2024), with two binary
word-level tasks scored by AUROC:

- **Task A** (`--endpoint sentence_onset`): sentence-initial vs mid-sentence word. Both
  classes are words. This is not upstream BrainBERT's `sentence_onset` task, whose
  negatives are non-words.
- **Task B** (`--endpoint word_nonword`): word vs non-word, the BrainBERT/PopT benchmark.
  Positives are word onsets; negatives are centres of 1 s word-free tiles inside the
  movie trigger range.

Classes are balanced to at most 900 events per class, drawn once with `--event_seed 42`.
CORTEG reads `[t, t+1.5]` s on both tasks. Download (~52 GB)
from [braintreebank.dev](https://braintreebank.dev/) and point `BTB_DATA_ROOT` at the
directory holding `all_subject_data/`, `electrode_labels/`, `localization/`,
`subject_metadata/`, `subject_timings/` and `transcripts/`. `sub_5`, `sub_8` and `sub_9`
are in the BrainBERT and PopT pretraining data (paper Table 6).

Electrodes come from PopT's `clean_laplacian` list, which is not redistributed here:

```bash
git clone https://github.com/czlwang/PopulationTransformer
export POPT_REPO=$PWD/PopulationTransformer
```

CORTEG keeps the electrodes that have MNI coordinates; the
iEEG-FM arms keep those in the per-subject voxel table, as in the paper runs. The MNI
set is a strict subset, smaller on 4 of 10 subjects (`sub_3` 82 vs 91, `sub_8` 120 vs
121, `sub_9` 64 vs 66, `sub_10` 157 vs 159). `POPT_REPO` is needed even when a feature
cache exists: a cache hit recomputes the selection and refuses a cache built over
another one.

No preprocessing is needed beyond the download: `data/braintreebank.py` reads the raw
HDF5 and produces the same two streams as Stanford. After a 60/120/180 Hz notch, the
128 Hz low stream is common-average referenced and the 70–200 Hz envelope (200 Hz) is
computed on the monopolar signal; the 1.5 s window gives 12 tokens per electrode per
stream (192/16 low, 300/25 high). The first run of each task builds a feature cache of
0.2–0.6 GB per subject (about 3.4 GB for Task A and 3.8 GB for Task B), which every
seed and arm then reuses. Notice:

- **Sampling rate is not fixed for all subjects.** Nine subjects run at a nominal 2048 Hz (measured 2038–2050 Hz); `sub_9` runs at about 1019 Hz. The loader measures the rate from the triggers; hardcoding 2048 mis-scales every window and frequency band for `sub_9`.
- **The scored trial is not always `trial000`.** `sub_1` is `trial001`, `sub_2` is `trial006`, `sub_6` is `trial004` — a different trial is a different film.
- **Events outside the trigger range are dropped.** `np.interp` clamps out-of-range inputs to the end value, collapsing 47 % of `sub_6`'s events onto one identical window, with both labels.

We use strictly causal forward-chaining splits: 4 folds, each with a causal
validation block (15 % of the pre-test data), and a 7 s gap between fit, validation and
test, wider than any arm's input window (Brant's 6 s patch with its resampling filter
edge is 6.11 s). We explicitly verify that no windows overlap across any fold boundary,
and every result file records the split of each fold. Randomly selected validation
samples could still leak through early stopping even when the train/test split itself
is clean. The frozen FM probes need no early stopping and fit on all pre-test data.

### Path resolution

| Variable | Used by | Default |
| --- | --- | --- |
| `CORTEG_DATA_ROOT` / `ECOG_DATA_ROOT` | Stanford loader | `~/workspace/datasets/stanford_ecog` |
| `CORTEG_PRETRAINED_ROOT` / `ECOG_PRETRAINED_ROOT` | backbone checkpoints | `~/workspace/datasets/pretrained_eeg_mae` |
| `CORTEG_OUTPUT_ROOT` / `ECOG_OUTPUT_ROOT` | where runs write results and caches | `~/workspace/outputs/corteg` |
| `CORTEG_NATIVE1K_ROOT` | native 1 kHz Stanford windows | `<data_root>/native_1k/built` |
| `CORTEG_STANFORD_RAW_ROOT` | raw Stanford download, for the native build | `<data_root>/raw/Stanford` |
| `BTB_DATA_ROOT` | BrainTreebank loader | `~/workspace/datasets/braintreebank` |
| `POPT_REPO` | BrainTreebank electrode selection; PopT | *(unset — see above)* |
| `BRAINBERT_REPO`, `BRAINBERT_WEIGHTS`, `POPT_WEIGHTS`, `BRANT_SRC`, `BRANT_WEIGHTS` | intracranial FMs | *(see below)* |

CLI flags (`--data_root`, `--save_root`) take precedence over the environment.

## Checkpoints

CORTEG is **two files**: a large frozen backbone you download once, and a small trained adapter that ships here.

### 1. ST-EEGFormer backbone (third-party, required)

Released with the ST-EEGFormer paper. Place under `$CORTEG_PRETRAINED_ROOT` matching the paths in `configs/steegformer_*.json`:

```
experiment3_small/checkpoint-300.pth      # Small  (D=512,  L=8,  25.6 M)  ← all main results
experiment4_base/checkpoint-288.pth       # Base   (D=768,  L=12, 85.6 M)
experiment5_large/checkpoint-196.pth      # Large  (D=1024, L=24, 303 M)
```

The Small backbone is a direct download (376 MB):

```bash
mkdir -p "$CORTEG_PRETRAINED_ROOT/experiment3_small"
curl -L -o "$CORTEG_PRETRAINED_ROOT/experiment3_small/checkpoint-300.pth" \
  https://github.com/LiuyinYang1101/STEEGFormer/releases/download/ST-EEGFormer-small/checkpoint-300.pth
```

The original paper's redistribution license applies — these weights are not re-hosted here.

### 2. CORTEG adapter (this paper, included)

```
checkpoints/corteg_stanford_pooled.pt     # 1.2 MB, 297,236 params — Table 1 finger row
checkpoints/corteg_stanford_pooled.json   # manifest: sha256, paper number, exact build args
```

It holds the KNNSoftFourier adapter (142,095), LoRA A/B on blocks 4–7 (131,072), the warm-started high-gamma patch embed (13,312), LayerNorms (8,192) and the readout (2,565) — so it must be loaded together with the backbone above.

```python
from load_corteg import load_corteg, predict

model = load_corteg(C_in=46, T_in=128, ecog_xyz_mm=xyz, d_out=5, device="cuda")
y = predict(model, x_lo, x_hi)      # (N,46,128) and (N,46,200) -> (N,5)
```

**Verified.** This checkpoint reproduces the published per-subject scores on all 9
Stanford subjects: cohort mean 0.5537 against 0.554 in the paper (Table 1; the
per-subject values in App. Table 7 average to 0.5536), largest per-subject deviation
0.0016 from the run's unrounded values. Those figures fit the z-score statistics on
each subject's whole training split; with the runner's 90 % split, subject `bp` gives
0.5783, the run's own value (notebook §3).

Its readout is at its random initialisation: the paper run built the readout lazily,
after the optimizer, so it was never trained. A random 512→5 projection has full rank,
so 0.554 reproduces from this file regardless. `scripts/table1_corteg_pooled_stanford.sh`
reproduces that run with `--freeze_readout`; see [Notes vs the paper](#notes-vs-the-paper).

## Intracranial foundation models

The paper compares CORTEG with BrainBERT, the Population Transformer (PopT) and Brant
on the regression tasks (Table 1; App. A.11, Tables 18–19) and on BrainTreebank (Table 3;
App. Table 20). `ieeg_fm.py` wraps all three. This release runs the Stanford half of the
regression comparison and the whole BrainTreebank comparison except the BrainBERT and
Brant fine-tune arms. Each model gets its own native input: BrainBERT and PopT 2048 Hz
spectrograms (PopT also takes electrode coordinates), Brant one 6 s, 250 Hz patch.

Neither the code nor the weights are redistributed here. Point these at your own copies:

| Model | Variable | Source |
| --- | --- | --- |
| BrainBERT | `BRAINBERT_REPO` | [github.com/czlwang/BrainBERT](https://github.com/czlwang/BrainBERT) |
| | `BRAINBERT_WEIGHTS` | `stft_large_pretrained.pth`, from the Google Drive link in that repo's README |
| PopT | `POPT_REPO` | [github.com/czlwang/PopulationTransformer](https://github.com/czlwang/PopulationTransformer) |
| | `POPT_WEIGHTS` | optional: a local `pretrained_popt_brainbert_stft.pth`; unset, the same file is downloaded from [huggingface.co/PopulationTransformer/popt_brainbert_stft](https://huggingface.co/PopulationTransformer/popt_brainbert_stft) |
| Brant | `BRANT_SRC` | `Brant_src/` from [huggingface.co/Daoze/Brant](https://huggingface.co/Daoze/Brant) |
| | `BRANT_WEIGHTS` | the **directory** holding `time_encoder.pt` and `channel_encoder.pt` — the released weights are two files, not one state_dict |

`POPT_REPO` is needed even for BrainBERT and Brant, because the shared electrode
selection lives there; PopT also needs `BRAINBERT_REPO`, since it runs on frozen
BrainBERT embeddings. BrainBERT and PopT need the `ieeg-fm` extra.

**Stanford regression** (`experiments/run_ieeg_fm_regression.py` for BrainBERT and PopT:
frozen linear probe and last-2-block fine-tuning, each pooled, per-subject or
leave-one-subject-out, and a BiLSTM temporal head, pooled or per-subject; `experiments/run_brant_regression.py` for Brant).
The defaults are the recorded settings of the paper runs, which are the FMs' own
budgets, not CORTEG's (App. A.11). All BrainBERT/PopT numbers use the task's 1 s causal
window with BrainBERT's centre-10-frame pooling (`--bb_pool last10` pools the frames
nearest the target instead).

**BrainTreebank** (`experiments/run_ieeg_fm_baselines.py` for the frozen arms,
`experiments/run_popt_finetune_btb.py` for PopT LoRA / full fine-tune / head-only). The
FM arms share CORTEG's events, test folds and 7 s embargo, but not its electrode set
(voxel, above), its readout or its window: the frozen arms fit a StandardScaler +
class-balanced logistic probe on all pre-test data; PopT fine-tuning carves a 15 %
validation block like CORTEG; BrainBERT and PopT read `[t, t+1.5]` s on Task A and
`[t-2.5, t+2.5]` s on Task B; Brant reads the 6 s ending at the task window's right edge,
`[t-4.5, t+1.5]` on Task A (which includes 4.5 s before onset) and `[t-3.5, t+2.5]` on
Task B. Caveats, as in the paper:

- The BrainBERT† and Brant† rows of Table 3 are `single_elec_max`, an oracle that picks
  the best electrode on the folds that score it. Its permutation null is at least 0.53
  (a lower bound, measured on a 1-D high-gamma feature), so read it next to
  `single_elec_mean`, which the script also runs.
- The PopT row of Table 3 (0.600 / 0.779) is the per-subject LoRA fine-tune, one seed.
- Task A windows are post-onset only, because the pause before a sentence-initial word
  alone predicts the label (0.85–0.93 AUROC from the transcripts, App. A.2.3). Brant's
  fixed 6 s patch is the exception: it includes 4.5 s before onset.

## Reproducing the paper

```bash
export CORTEG_DATA_ROOT=$HOME/workspace/datasets/stanford_ecog
export CORTEG_PRETRAINED_ROOT=$HOME/workspace/datasets/pretrained_eeg_mae
export CORTEG_OUTPUT_ROOT=$HOME/workspace/outputs/corteg
export BTB_DATA_ROOT=$HOME/workspace/datasets/braintreebank
export POPT_REPO=/path/to/PopulationTransformer
```

**Seeds.** Table 1's CORTEG and iEEG-FM rows are single runs at seed 42. On
BrainTreebank (Tables 3, 9, 20) the CORTEG, random-init and scratch-baseline rows
average training seeds 42, 1 and 2 within each subject, all on one event set drawn with
`--event_seed 42` (`--seed` changes only initialisation and batch order); the trained FM
arms are one seed-42 run and the frozen probes are deterministic. The
pooled and per-subject cells of Table 19 (and the ¶ cells of Table 18) average seeds
42, 0 and 1; the LOO cells are seed 42. Every script's defaults follow this.

**Table 1, CORTEG pooled** (≈2 GPU-hours on one RTX 5090):

```bash
bash scripts/table1_corteg_pooled_stanford.sh
```

Expect a cohort r close to 0.554 (the paper run's unrounded value is 0.553487), not
identical: GPU training is not deterministic. The script passes `--freeze_readout`,
which keeps the readout at its initialisation as in the paper run; with seed 42 on a
GPU and the tested versions, that initial readout is bit-identical to the released
checkpoint's. `--train_readout` trains it instead.

**Table 1, CORTEG LOO-FT** (per held-out subject: Stage 1 is a pooled run on the other
eight, ≈2 GPU-hours; Stage 2 takes 10–30 min):

```bash
for s in bp cc ht jc jp mv wc wm zt; do bash scripts/table1_corteg_loo_ft.sh $s; done
```

The script's header has a one-line summary of the nine Stage-2 results; the paper runs
give `9 0.551 0.147` (n, mean, sample SD). Stage 1 also scores the held-out subject
zero-shot (App. Table 11, finger column).

**Table 1 iEEG-FM rows and the Stanford cells of Tables 18–19** (needs the native
1 kHz windows and the `ieeg-fm` extra):

```bash
ONLY_TABLE1=1 bash scripts/table1_ieeg_fm_stanford.sh   # the three Table 1 cells
bash scripts/table1_ieeg_fm_stanford.sh                 # the full grid, 31 cells
```

Finished cells are skipped. `--device cpu`, a missing GPU or a narrowing flag
(`--subjects`, `--max_windows`, `--epochs`, ...) turns the pass into a smoke run written
to `ieeg_fm_regression_smoke/`, so it can never stand in for a paper cell.

**Tables 3, 9, 20, BrainTreebank** (10 subjects × 4 causal folds):

```bash
bash scripts/table3_corteg_braintreebank.sh     # CORTEG gate, mean-pool, random init
bash scripts/table3_ieeg_fm_braintreebank.sh    # BrainBERT, PopT, Brant
```

- The CORTEG script runs 3 arms × 2 tasks × seeds 42, 1, 2 (18 runs): gated fusion
  (the paper's CORTEG row), mean-pool fusion, and the random-init control (the gate arm
  with `--no_pretrained` and the same backbone config and recipe). The recipe is the
  runner's default: one pooled model over the ten subjects with a per-subject LoRA
  adapter, 60 epochs, AdamW lr 3e-4, batch 16 × 4 accumulation steps, bf16 on CUDA,
  validation every 2 epochs, patience 15 evaluations. `SEEDS`, `ENDPOINTS` and `ARMS`
  choose a subset, e.g. `SEEDS=42 ENDPOINTS=word_nonword ARMS=gate`. The first run of
  each task builds the feature cache, CPU-bound and reading the full 52 GB tree.
- The FM script needs the FM variables above. `STAGE=gpu` runs the embedding passes and
  PopT; `STAGE=probe` then fits the BrainBERT and Brant probes on CPU from the cached
  embeddings. Brant first resamples each whole recording to 250 Hz (CPU, up to about an
  hour per subject; cached).

Compare finished runs with the paper as shown under [Expected results](#expected-results).

**Fusion variants.** The two fusion variants of the Stanford runner are
`--merge_strategy average` (mean-pool fusion at block *L*−*K*, which the released
checkpoint uses) and `--merge_strategy layerwise_gate`, where a small network emits one
scalar per block and each block receives `+ g_l · hi`; with `tanh` the gates start at
exactly 0, so training begins from the low-frequency-only model (Table 4; App. A.10).

## Expected results

The published values, as `scripts/aggregate_btb.py` and `scripts/aggregate_fm.py`
recompute them from `paper_cells/`.

**BrainTreebank** (App. Table 20; mean AUROC ± SD over 10 subjects, 4 dp). "Seed 42" is
the paper's own seed-42 cohort mean, the value a single-seed rerun should land near.

| Arm | Task A | Task B | Seed 42 (A / B) |
| --- | ---: | ---: | ---: |
| CORTEG, layer-wise gate | 0.6376 ± 0.0783 | 0.7492 ± 0.1345 | 0.6499 / 0.7591 |
| CORTEG, mean-pool fusion | 0.6159 ± 0.0721 | 0.7525 ± 0.1326 | 0.6500 / 0.7558 |
| CORTEG, random-init backbone | 0.5376 ± 0.0536 | 0.5880 ± 0.1025 | 0.5396 / 0.5994 |
| HiLoFuseNet° | 0.5330 ± 0.0220 | 0.7208 ± 0.0886 | 0.5217 / 0.7216 |
| CNN-LSTM° | 0.5071 ± 0.0142 | 0.5706 ± 0.0469 | 0.5096 / 0.5723 |
| LSTM° | 0.5073 ± 0.0136 | 0.5618 ± 0.0429 | 0.5058 / 0.5613 |
| PopT, LoRA | 0.6003 ± 0.0845 | 0.7790 ± 0.0991 | one seed |
| PopT, full fine-tune | 0.6001 ± 0.0726 | 0.6979 ± 0.1228 | one seed |
| PopT, head-only | 0.5594 ± 0.0442 | 0.6714 ± 0.0915 | one seed |
| PopT, frozen probe | 0.5697 ± 0.0358 | 0.6698 ± 0.0814 | deterministic |
| BrainBERT, single-electrode max† | 0.6015 ± 0.0450 | 0.6980 ± 0.0941 | deterministic |
| BrainBERT, single-electrode mean | 0.5085 ± 0.0063 | 0.5236 ± 0.0166 | deterministic |
| BrainBERT, population mean-pool | 0.5294 ± 0.0280 | 0.5788 ± 0.0664 | deterministic |
| Brant, single-electrode max† | 0.5865 ± 0.0415 | 0.5775 ± 0.0366 | deterministic |
| Brant, single-electrode mean | 0.5216 ± 0.0114 | 0.5261 ± 0.0107 | deterministic |
| Brant, population mean-pool | 0.5208 ± 0.0158 | 0.5233 ± 0.0120 | deterministic |

° Published cells only; these decoders have no runner in this release.

At 3 dp these are Table 3: CORTEG 0.638 ± 0.078 / 0.749 ± 0.134, random init
0.538 ± 0.054 / 0.588 ± 0.102, HiLoFuseNet 0.533 ± 0.022 / 0.721 ± 0.089, PopT
0.600 ± 0.084 / 0.779 ± 0.099, BrainBERT† 0.602 ± 0.045 / 0.698 ± 0.094, Brant†
0.587 ± 0.041 / 0.577 ± 0.037.

**Stanford iEEG FMs** (finger; mean r ± SD over 9 subjects; the Table 1 cells are the
best adaptation per model):

| Model | Frozen linear probe | Last-2 fine-tune | Temporal head (BiLSTM) |
| --- | ---: | ---: | ---: |
| BrainBERT | 0.044 ± 0.051 (LOO) | 0.047 ± 0.038 (LOO) | **0.053 ± 0.056** (per-subject) |
| PopT | 0.050 ± 0.045 (LOO) | 0.041 ± 0.046 (pooled, 3 seeds) | **0.063 ± 0.046** (per-subject) |
| Brant | **0.028 ± 0.031** (6 s context, per-subject) | — | — |

Seed-42 cohort means of the paper runs, for single-seed checks: temporal head
per-subject BrainBERT 0.0526, PopT 0.0626; Brant per-subject 0.0284; probe LOO
BrainBERT 0.0437, PopT 0.0497; last-2 fine-tune LOO BrainBERT 0.0469, PopT 0.0360.

**Checking a rerun.** Point the aggregators at the run folders (quote the paths):

```bash
OUT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}"
python scripts/aggregate_btb.py --cells "$OUT/braintreebank/table3" "$OUT/braintreebank/fm_runs"
python scripts/aggregate_fm.py --cells "$OUT/ieeg_fm_regression"
```

A published cell is scored only when the rerun has its seeds and all subjects; every
fresh seed is also compared, subject by subject, with the same seed in `paper_cells/`.
A run at any non-paper setting (another event seed, fewer epochs, a subject subset, a
CPU run of the FM regression, ...) becomes its own row under "Other cells". GPU
training is not bit-reproducible: three identical seed-42 Task B gate runs of the paper
code spanned 0.0074 in cohort AUROC, and the BrainTreebank tolerances are 1.5 × that
floor (0.0111 cohort, 0.0148 mean |Δ| per subject, 0.0426 any subject). No floor was
measured for the FM regression cells, whose tolerances (0.005 / 0.01 / 0.03) are
heuristics; a CPU run is fp32 and exceeds them.

**Verification status.** On CPU, the released BrainTreebank code builds the same events,
features (bit-identical) and fold splits as the paper runs; the Stanford FM runners
reproduce the paper code bit for bit on reduced configurations; the BrainBERT/PopT
embeddings match the paper's caches to ≈1e-6 and the Brant 250 Hz streams are
bit-identical. On GPU (RTX 5090), seed-42 runs of the released code land on the paper's
seed-42 values:

| Cell | Released code | Paper run (seed 42) |
| --- | ---: | ---: |
| BrainTreebank CORTEG gate, Task A / Task B | 0.6585 / 0.7641 | 0.6499 / 0.7591 |
| BrainTreebank PopT LoRA, Task B | 0.7790 | 0.7790 |
| Stanford temporal head, BrainBERT / PopT | 0.0551 / 0.0627 | 0.0526 / 0.0626 |
| Stanford Brant, per-subject | 0.0284 | 0.0284 |

All are within the tolerances above. The CORTEG Stanford runs of Table 1 have not been
repeated with the released code; the released checkpoint reproduces 0.554 at inference
(see Checkpoints).

## Reprinting the tables without a GPU

`paper_cells/` holds the 244 result files behind Tables 3, 9, 20 and the FM rows of
Tables 1, 18 and 19, including the Ghent cells, with only absolute paths replaced by
placeholders ([`paper_cells/MANIFEST.md`](paper_cells/MANIFEST.md) maps every cell to its
files).

```bash
python scripts/aggregate_btb.py    # Tables 3, 9, 20 + oracle null -> "258 cells checked against the paper: all match."
python scripts/aggregate_fm.py     # Table 1 FM rows, Tables 18, 19 -> "78 cells checked against the paper: all match."
```

Seeds are averaged within subject, then the cell is the mean ± sample SD (ddof=1) over
subjects, rounded once from full precision; the Table 20 seed SD is the SD of the
per-seed cohort means. Either script exits non-zero if a cell differs from the paper.

## Notes vs the paper

The scripts reproduce the paper runs as they were produced, including the behaviours
below, which the paper does not state. Where there is an alternative it is a flag, off
by default; none of the alternatives was run for the paper.

- **Table 1 readout.** The method section lists the regression readout as trainable, but
  the pooled Table 1 run and the LOO-FT Stage-1 runs built it after the optimizer, so it
  stayed at its initialisation. Both scripts pass `--freeze_readout`; `--train_readout`
  trains it. Not retrained end to end; a ridge readout refit on the released checkpoint's
  features scores 0.5643 against 0.5537. Stage 2 trains the readout, as described.
- **Table 1 rounding.** The pooled run's cohort mean is 0.553487, which rounds to 0.553;
  the paper prints 0.554, the rounding of 0.5535 (or of the mean of the 3-dp per-subject
  values, 0.5536). Likewise LOO-FT audio is 0.330477 and printed as 0.331.
- **Low-frequency band.** The paper and Fig. 1 describe the low-frequency stream as
  1–64 Hz. The released Stanford pipeline applies 60/120/180 Hz notches and resamples to
  128 Hz, which low-passes at 64 Hz; there is no 1 Hz high-pass (the recordings were
  acquired band-passed at 0.3–200 Hz).
- **LOO-FT Stage 1** selects its epoch on the validation r of all nine subjects,
  including the held-out subject's validation split (the last 10 % of its training
  recording, never its test split). The zero-shot table was read from checkpoints
  chosen this way, and Stage 1's reported score averages in the held-out subject's
  zero-shot r. The Table 1 LOO-FT row is unaffected, since Stage 2 trains on that
  subject's training split. There is no flag for this.
- **BrainTreebank CORTEG.** Early stopping selects on the pooled validation AUROC,
  which mixes between-subject pairs (`--select_metric per_subject` selects on the mean
  per-subject AUROC), and patience counts evaluations, every 2 epochs. Task B candidates
  are filtered with the 5 s window, so CORTEG scores exactly the FM arms' events. Many
  Task B negatives sit in long silences (`--neg_mode short_silence` restricts them to
  pauses under 10 s), and in 8–18 % of each subject's negatives (14 % on average) the
  `[t, t+1.5]` window reaches the next word's onset, since only the tile's 1 s is
  word-free.
- **FM regression.** Pooled probes are not reseeded after embedding extraction, so a
  cold and a warm embedding cache give different numbers for the same seed
  (`--reseed_pooled_head`); the script runs them warm, and the paper runs' cache state
  is not recorded. Brant's input is in raw ADC counts, its resampling filter reaches at
  most 40 ms past a target, and its context for the first test windows starts before the
  split boundary (inputs only, never targets). Three further defaults have flags for
  the alternative: BrainBERT's centre-10-frame pooling, about 0.5 s before the target
  (`--bb_pool last10`); the bidirectional temporal head, non-causal within 64-window
  chunks (`--temporal_direction uni`); and Brant's input, one 6 s patch with its
  released (15, 2048) positional encoding sliced to the first row
  (`--context_patches 15` with `--patch_pool last` feeds the 15-patch, 90 s sequence it
  was pretrained on).

## Tests

```bash
python -m unittest discover -s tests
```

The suite runs on CPU without any dataset. `tests/test_release_integrity.py` checks
that every script calls an existing runner with flags that runner has and loads the
pretrained backbone where it should, that the Table 1 recipes match the paper runs,
that the documented paper numbers match the aggregators, and that the released checkpoint
matches its manifest hash, rebuilds from its recorded build_args and loads every tensor
(these checkpoint tests, 7 in all, need the ST-EEGFormer backbone and skip without it). The other files test
each area on synthetic data (`test_btb_corteg`, `test_btb_fm`,
`test_fm_regression`), the per-cell files (`test_paper_cells`) and the demo data
(`test_demo_data`). Reproducing r itself needs the Stanford data (see the notebook).

## Interactive demo

`docs/` is a static, single-page site (Three.js + Plotly, no backend) that plots
ground-truth vs predicted trajectories for 9 Stanford and 16 Ghent subjects. The rows
are CORTEG pooled and per-subject (Table 1), the random-init full fine-tuning control
(Table 2), Ridge_HGA and Ridge_LFS on Stanford (Table 1, recomputed with the paper's
recipe, since those runs saved no predictions), and a labelled HiLoFuseNet re-run that is
not a paper value. The CORTEG and ridge rows reproduce the paper's per-subject values.
On Stanford the random-init row shows seed 7 (0.511): Table 2's 0.510 is the mean of
seeds 7 and 123, and a seed-42 run of the same configuration (0.554) is not in that mean.
The r shown for each subject is computed on the full test split; the plotted traces
are downsampled to 1,000 points, so their own r can differ by a few hundredths.
`build_demo_data.py` documents the source of every trace. Live at
[liuyinyang1101.github.io/CORTEG](https://liuyinyang1101.github.io/CORTEG/),
or locally:

```bash
cd docs && python -m http.server 8000   # open http://localhost:8000
```
