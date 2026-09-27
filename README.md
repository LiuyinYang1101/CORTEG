# CORTEG

**Foundation Models Enable Cross-Modality Representation Transfer from Scalp to Intracranial Brain Recordings**

> Adapt a frozen scalp-EEG foundation model to intracranial ECoG decoding —
> calibrate to a new patient in **10–30 minutes** on a single GPU.

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS-2026-4b44ce.svg)](#citation)
[![Paper](https://img.shields.io/badge/arXiv-2605.10337-b31b1b.svg)](https://arxiv.org/abs/2605.10337)
[![Demo](https://img.shields.io/badge/demo-live-2ea44f.svg)](https://liuyinyang1101.github.io/CORTEG/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Table, section and appendix numbers in this README refer to the NeurIPS 2026
camera-ready. arXiv v1 (2605.10337v1) is the submission: it predates the
BrainTreebank and intracranial-FM experiments (Tables 3, 9, 18–20; §5.5; App.
A.2.3, A.11) and numbers its tables differently.

![CORTEG overview](docs/assets/figure1_overview_scheme.png)

## Motivation

Intracranial ECoG offers high-SNR access to cortical activity, but each recording is short, patient-specific, and electrode layouts vary, so most ECoG decoders are trained per-subject and ignore information shared across
patients. Scalp-EEG foundation models, by contrast, are pretrained on far larger, fast-growing scalp-EEG corpora — but they have never been systematically adapted *down into the skull*. The question CORTEG asks is whether the
representations learned from extracranial EEG are useful for intracranial ECoG, and whether a population model can be deployed to a new patient without retraining from scratch.

## What CORTEG is

A small set of trainable components wrapped around a frozen EEG foundation model:

- **Frozen [ST-EEGFormer](https://github.com/LiuyinYang1101/STEEGFormer) backbone** —
  pretrained on 128-Hz scalp EEG with an EEG channel-embedding codebook.
- **KNNSoftFourier spatial adapter** — maps each ECoG electrode's 3-D MNI coordinate into the pretrained EEG embedding space (soft k-NN over the codebook + a learnable Fourier residual).
- **Dual-stream tokenization** — low-frequency (1–64 Hz, 128 Hz) and high-gamma (70–200 Hz envelope, 200 Hz) tokens, averaged before the last *K* = 4 transformer blocks (mean-pool fusion; a layer-wise gate is the alternative).
- **LoRA on the last 4 blocks** (QKV, attention output, fc1, fc2), the LayerNorms of those blocks, the high-gamma patch embedding, the spatial adapter and a linear readout are the *only* trainable parameters: ≈297K on the regression tasks. On BrainTreebank each subject gets its own LoRA adapter, ≈1.49M for the ten subjects.
- **Two-stage LOO-FT** — pool on N − 1 patients, then fine-tune the adapter + LoRA + readout on the held-out patient in minutes.

## Key findings

- **Cross-modality pretraining transfers.** Pooled CORTEG reaches the highest mean correlation among compared methods on both tasks: r=0.554 on Stanford finger (n=9) and r=0.339 on Ghent audio (n=16). On finger it is statistically comparable to the strongest task-specific decoders (DeepFingerNet 0.542, HiLoFuseNet 0.534; not significant at n=9); on audio the gain is significant (Bonferroni-corrected p<0.01).
- **LOO-FT matches pooled training.** A new patient calibrates in 10–30 min on a single GPU, with no significant difference from pooled training (paired Wilcoxon p=0.65 finger, p=0.82 audio).
- **The signal is the EEG pretraining, not the architecture.** Replacing ST-EEGFormer with a random init drops r by 0.044 (finger) / 0.183 (audio); swapping in LaBraM, CBraMod, or MantisV2 backbones drops r by 0.18–0.36 on finger.
- **Intracranial FMs do not transfer to continuous regression.** BrainBERT, PopT and Brant stay at r ≤ 0.063 (finger) and ≤ 0.057 (audio) in their best adaptation. On BrainTreebank classification they are much stronger: PopT reaches AUROC 0.779 on word vs non-word, CORTEG 0.749, and CORTEG is best on the sentence-position task (0.638).

## Headline results — Stanford finger (n=9) and Ghent audio (n=16)

Pearson r (± cross-subject SD), paper Table 1. Best per column in **bold**, second-best in *italic*.

| Method | Trainable params | Finger (n=9) | Audio (n=16) |
| --- | ---: | ---: | ---: |
| Ridge (HGA) | — | 0.336 ± 0.095 | 0.175◇ |
| HiLoFuseNet | 334 K | 0.534† ± 0.138 | 0.259 ± 0.203 |
| CNN-LSTM | 557 K | 0.411† ± 0.124 | 0.261 ± 0.195 |
| DeepFingerNet | 1.16 M | 0.542‡ ± 0.129 | 0.085 ± 0.144 |
| BrainBERT§ | varies | 0.053 ± 0.056 | 0.057 ± 0.066 |
| PopT§ | varies | 0.063 ± 0.046 | 0.039 ± 0.064 |
| Brant§ | varies | 0.028 ± 0.031 | 0.022 ± 0.035 |
| CORTEG (per-subject) | 297 K | 0.539 ± 0.140 | 0.250 ± 0.226 |
| CORTEG (LOO-FT) | 297 K | *0.551 ± 0.147* | *0.331* ± 0.184¶ |
| **CORTEG (pooled)** | **297 K** | **0.554 ± 0.154** | **0.339 ± 0.170** |

† finger value transcribed from Sun et al. 2025, ‡ from Tao et al. 2025; the other
baselines were trained by the authors. The Table 1 baselines are not part of this
release, and the audio column uses the private Ghent data. ◇ SD omitted (Ridge audio is
evaluated continuously). § Best iEEG-FM adaptation of App. Table 18: the BiLSTM
temporal head for BrainBERT (both tasks) and PopT (finger), PopT's frozen linear probe
with leave-one-subject-out training (audio), and Brant's frozen probe with its 6 s
context. ¶ 0.330477 unrounded, printed as in the paper (see Notes vs the paper). The
per-subject CORTEG row has no script in this release; its recipe is in App. A.4. The full
table, ablations and per-subject paired tests are in the paper.

The CORTEG values of Table 1 are single runs (seed 42) at a fixed local budget, so they
are a lower bound: under the paper's larger-scale training, CORTEG reaches 0.577 (Small)
and 0.583 (Base) on finger (App. Table 10).

## Repository scope

This release trains and evaluates CORTEG on the two public datasets — Stanford
fingerflex (5-finger regression) and BrainTreebank (two binary language tasks: Task A,
sentence-initial vs mid-sentence word; Task B, word vs non-word) — together with most
of the public-data comparison arms of Tables 1, 3, 9 and 18–20 (the rest are listed
under "Not released" below). It also ships the trained Stanford adapter and the
per-cell result files behind the published tables.

The **Ghent** speech-envelope dataset is private, is **not** redistributed, and no Ghent training code is included. The live demo shows per-subject Ghent prediction traces (and the corresponding target envelope segment) for visualisation only.

```
load_corteg.py        load the released adapter and run it
paths.py              environment-variable path resolution
ieeg_fm.py            BrainBERT / PopT / Brant adapters (third-party code and weights not included)
models/steegformer/   ST-EEGFormer backbone, KNNSoftFourier adapter, LoRA
models/               LaBraM / CBraMod backbones; HiLoFuseNet, CNN-LSTM, LSTM (baselines.py)
data/                 Stanford + BrainTreebank loaders, splits, scalers, collate;
                      stanford_native.py builds the native 1 kHz Stanford windows
train/                training engine, early stopping, LR schedule
experiments/          CORTEG runners (Stanford, BrainTreebank) and the baseline runners
configs/              ST-EEGFormer Small/Base/Large JSONs
scripts/              paper-aligned reproduction scripts and the table aggregators
paper_cells/          the result files behind Tables 1 (FM rows), 3, 9, 18, 19, 20
notebooks/            quickstart.ipynb — start here
checkpoints/          released CORTEG adapter + manifest
docs/                 live interactive demo (Three.js + Plotly, no backend)
```

| Paper artefact | Script |
| --- | --- |
| Table 1, CORTEG pooled (finger) | `scripts/table1_corteg_pooled_stanford.sh` |
| Table 1, CORTEG LOO-FT (finger) | `scripts/table1_corteg_loo_ft.sh <subject>` |
| Table 1 iEEG-FM rows (finger); Tables 18–19, Stanford cells | `scripts/table1_ieeg_fm_stanford.sh` |
| Tables 3, 9, 20: CORTEG gate, mean-pool and random-init rows | `scripts/table3_corteg_braintreebank.sh` |
| Tables 3, 9, 20: HiLoFuseNet, CNN-LSTM, LSTM (trained from scratch) | `scripts/table3_btb_baselines.sh` |
| Tables 3, 9, 20: BrainBERT†, Brant†, PopT (LoRA) and the other PopT, BrainBERT and Brant arms listed below | `scripts/table3_ieeg_fm_braintreebank.sh` |
| App. A.2.3, pause-only AUROC of the Task A events (0.85–0.93) | `scripts/btb_pause_only_auroc.py` |
| Reprint Tables 3, 9, 20 / Table 1 FM rows, 18, 19 from `paper_cells/` | `scripts/aggregate_btb.py`, `scripts/aggregate_fm.py` |

The FM script runs, per task: BrainBERT and Brant single-electrode max, single-electrode
mean and population mean-pool; PopT frozen probe, LoRA, full fine-tune and head-only.

**Not released.**
- Anything on Ghent (the audio column of every table): the data are private.
- Table 2 and Table 4 / App. Table 17 have no script. The pooled Stanford runner has the
  switches the ablations and the fusion study use (`--no_pretrained --full_finetune`,
  `--xyz_mode`, `--stream`, `--adapter_branch`, `--merge_strategy layerwise_gate`,
  `--steegformer_variant`), and `experiments/run_{labram,cbramod,mantis}_baseline.py`
  swap the backbone, but none of these has been re-run against the published values
  (Table 4 uses a multi-seed HPC protocol, and the LaBraM and CBraMod cells of Table 2
  average three seeds). The REVE port is not included.
- The LOO-FT low-data sweep (Fig. 2d,e; the 2× adapter multiplier at f = 0.1).
- BrainTreebank: the raw spectral linear probes and the BrainBERT and Brant head-only /
  LoRA / full fine-tune rows of Table 20, and the oracle permutation null. Their cells
  are in `paper_cells/btb` and are reprinted by `scripts/aggregate_btb.py`.
- App. A.11 controls quoted only in the text: full-backbone FM fine-tuning, joint FM +
  temporal fine-tuning, the extraction check (per-electrode ridge readouts), Brant's
  15-patch control, the representation analysis, the omni_ieeg seizure-onset-zone
  control and the centre-pooling proxy. `paper_cells/MANIFEST.md` lists these numbers.
- The Stanford pooled BrainBERT fine-tune cell of Table 19, which the paper does not
  report (the run is incomplete); `RUN_BB_FT_POOLED=1` re-runs it.
- The run outputs the demo is built from (`build_demo_data.py` is an author-side script).

## Install

```bash
git clone https://github.com/LiuyinYang1101/CORTEG.git
cd CORTEG
conda create -n corteg python=3.11 -y && conda activate corteg
pip install -e .
pip install -e ".[ieeg-fm]"   # BrainBERT / PopT baselines: omegaconf, huggingface_hub
pip install -e ".[mantis]"    # optional: the MantisV2 backbone swap
pip install -e ".[mne]"       # optional: Stanford preprocessing, LaBraM montage
```

Tested on Ubuntu 24.04 with PyTorch 2.11 / CUDA 12.8 on an RTX 5090. The scripts call
`python`, so run them with this environment active.

## Quick start

**Start with [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb).** It goes from a fresh clone to reproducing a published per-subject number, and explains the one thing that is easy to get wrong (CORTEG takes *two* input streams).

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
CORTEG and the scratch baselines read `[t, t+1.5]` s on both tasks. Download (~52 GB)
from [braintreebank.dev](https://braintreebank.dev/) and point `BTB_DATA_ROOT` at the
directory holding `all_subject_data/`, `electrode_labels/`, `localization/`,
`subject_metadata/`, `subject_timings/` and `transcripts/`. `sub_5`, `sub_8` and `sub_9`
are in the BrainBERT and PopT pretraining data (paper Table 6).

Electrodes come from PopT's `clean_laplacian` list, which is not redistributed here:

```bash
git clone https://github.com/czlwang/PopulationTransformer
export POPT_REPO=$PWD/PopulationTransformer
```

CORTEG and the scratch baselines keep the electrodes that have MNI coordinates; the
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
seed, arm and scratch baseline then reuses. Notice:

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

### 3. Baseline FM checkpoints (optional)

| Model | Path | Source |
| --- | --- | --- |
| LaBraM-base | `$CORTEG_PRETRAINED_ROOT/../pretrained_eeg_fms/labram/labram-base.pth` | [LaBraM](https://github.com/935963004/LaBraM) |
| CBraMod | `$CORTEG_PRETRAINED_ROOT/../pretrained_eeg_fms/cbramod/pretrained_weights.pth` | [CBraMod](https://github.com/wjq-learning/CBraMod) |
| MantisV2 | `$CORTEG_PRETRAINED_ROOT/../pretrained_tsfm/mantis_v2` | [Mantis-TS](https://github.com/Mantis-TS/MantisV2) |

Each runner takes `--pretrained_path` if you keep them elsewhere. These weights are not re-hosted here; each carries its own license.

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
window with BrainBERT's centre-10-frame pooling; App. A.11's extraction check shows that
longer windows and per-electrode readouts score higher, though still below raw
high-gamma ridge.

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
- Task A windows are post-onset only because the pause before a sentence-initial word
  alone predicts the label: `scripts/btb_pause_only_auroc.py` measures 0.85–0.93 AUROC
  (mean 0.90) from the transcripts. Brant's patch is the exception; a patch ending at
  onset scores 0.540 ± 0.009, near the oracle's null (App. A.2.3).

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
bash scripts/table3_btb_baselines.sh            # HiLoFuseNet, CNN-LSTM, LSTM
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
- The baselines script runs 3 decoders × 2 tasks × 3 seeds on CORTEG's cached features
  (`DECODERS`, `ENDPOINTS`, `SEEDS` choose a subset) and prints each published cell.
- The FM script needs the FM variables above. `STAGE=gpu` runs the embedding passes and
  PopT; `STAGE=probe` then fits the BrainBERT and Brant probes on CPU from the cached
  embeddings. Brant first resamples each whole recording to 250 Hz (CPU, up to about an
  hour per subject; cached).
- `python scripts/btb_pause_only_auroc.py` (CPU, transcripts only) gives the pause-only
  AUROCs of App. A.2.3.

Compare finished runs with the paper as shown under [Expected results](#expected-results).

**Other runners.** `experiments/run_{labram,cbramod,mantis}_baseline.py` swap the backbone
(Table 2, see "Not released" above). The two fusion variants of the Stanford runner are
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
| HiLoFuseNet | 0.5330 ± 0.0220 | 0.7208 ± 0.0886 | 0.5217 / 0.7216 |
| CNN-LSTM | 0.5071 ± 0.0142 | 0.5706 ± 0.0469 | 0.5096 / 0.5723 |
| LSTM | 0.5073 ± 0.0136 | 0.5618 ± 0.0429 | 0.5058 / 0.5613 |
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
python scripts/aggregate_btb.py --cells "$OUT/braintreebank/table3" \
    "$OUT/braintreebank/base_HiLoFuseNet" "$OUT/braintreebank/base_CNN_LSTM" \
    "$OUT/braintreebank/base_LSTM" "$OUT/braintreebank/fm_runs"
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
features (bit-identical) and fold splits as the paper runs; the scratch baselines and
the Stanford FM runners reproduce the paper code bit for bit on reduced configurations;
the BrainBERT/PopT embeddings match the paper's caches to ≈1e-6 and the Brant 250 Hz
streams are bit-identical. The full training runs of the released code have not yet
been repeated on a GPU, so no BrainTreebank or FM cell is verified end to end.

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
- **BrainTreebank scratch baselines** are one model pooled over the ten subjects, with no
  electrode coordinates: channel *k* is each subject's *k*-th electrode, zero-padded to
  the widest subject (191). `--train_mode per_subject` trains one model per subject.
- **FM regression.** Pooled probes are not reseeded after embedding extraction, so a
  cold and a warm embedding cache give different numbers for the same seed
  (`--reseed_pooled_head`); the script runs them warm, and the paper runs' cache state
  is not recorded. Brant's input is in raw ADC counts, its resampling filter reaches at
  most 40 ms past a target, and its context for the first test windows starts before the
  split boundary (inputs only, never targets). App. A.11 describes three further
  defaults, whose alternatives are flags: BrainBERT's centre-10-frame pooling, about
  0.5 s before the target (`--bb_pool last10`); the bidirectional temporal head,
  non-causal within 64-window chunks (`--temporal_direction uni`); and Brant's single
  patch with its positional encoding sliced to one row (`--context_patches 15` with
  `--patch_pool last` restores the 15-patch sequence).

## Tests

```bash
python -m unittest discover -s tests
```

The suite runs on CPU without any dataset. `tests/test_release_integrity.py` checks
that every script calls an existing runner with flags that runner has and loads the
pretrained backbone where it should, that the Table 1 recipes match the paper runs,
that the README's paper numbers match the aggregators, and that the released checkpoint
matches its manifest hash, rebuilds from its recorded build_args and loads every tensor
(these checkpoint tests, 7 in all, need the ST-EEGFormer backbone and skip without it). The other files test
each area on synthetic data (`test_btb_corteg`, `test_btb_baselines`, `test_btb_fm`,
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
`build_demo_data.py` documents the source of every trace. Live at
[liuyinyang1101.github.io/CORTEG](https://liuyinyang1101.github.io/CORTEG/),
or locally:

```bash
cd docs && python -m http.server 8000   # open http://localhost:8000
```

## License

Code: MIT (see [LICENSE](LICENSE)). The pretrained ST-EEGFormer backbone is distributed under its own license — see **Checkpoints** above.
Ghent dataset: not included.

## Citation

```bibtex
@inproceedings{yang2026corteg,
  title     = {{CORTEG}: Foundation Models Enable Cross-Modality Representation Transfer
               from Scalp to Intracranial Brain Recordings},
  author    = {Yang, Liuyin and Sun, Qiang and Van Dyck, Bob and Calvo Merino, Eva and
               Van Hulle, Marc M.},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}

@misc{corteg2026,
  title  = {CORTEG: Foundation Models Enable Cross-Modality Representation Transfer
            from Scalp to Intracranial Brain Recordings},
  author = {Liuyin Yang and Qiang Sun and Bob Van Dyck and Eva Calvo Merino and Marc M. Van Hulle},
  year   = {2026},
  eprint = {2605.10337},
  archivePrefix = {arXiv}
}
```

Liuyin Yang and Qiang Sun contributed equally.
