# paper_cells — the result files behind the published tables

This folder holds the per-cell result files that produced the paper's
BrainTreebank tables (Tables 3, 9 and 20) and its intracranial
foundation-model regression tables (the FM rows of Table 1, Tables 18 and 19).
With them every one of those numbers can be recomputed on a CPU in a second,
including the Ghent audio cells, whose data are not released.

```bash
python scripts/aggregate_btb.py      # Tables 3, 9, 20 + oracle null -> "258 cells checked against the paper: all match."
python scripts/aggregate_fm.py       # Tables 1 (FM rows), 18, 19 -> "78 cells checked against the paper: all match."
python -m unittest tests.test_paper_cells
```

Both scripts exit non-zero if any printed cell differs from the paper. They
also aggregate fresh runs from the released runners (`--cells <dir>`, below).
The BrainTreebank count is the 256 table cells plus the oracle null, which the
paper quotes as "at least 0.53": its two cohort means, 0.5297 (Task A) and
0.5284 (Task B), are checked at 2 dp.

| Folder | Files | Size | Tables |
| --- | ---: | ---: | --- |
| `btb/` | 96 | 1.45 MB | 3, 9, 20 (and the oracle null quoted in their footnotes) |
| `fm_regression/` | 148 | 0.37 MB | 1 (FM rows), 18, 19 |

`SHA256SUMS` lists every file; `tests/test_paper_cells.py` fails if a file
changes without it.

## What these files are, and what was changed

Each file is a result JSON written by the run that produced the paper number,
copied from the private experiment repository's output tree. Only one thing
was changed: absolute paths in `args` / `config` fields (and the `save_root` /
`out_root` values) were replaced by placeholders:

| Placeholder | Stood for |
| --- | --- |
| `<OUTPUT_ROOT>` | the run's output root (a workstation or cluster scratch path) |
| `<BTB_DATA_ROOT>` | the BrainTreebank download |
| `<STANFORD_NATIVE_1K>` | the native 1 kHz Stanford windows (`data/stanford_native.py`) |
| `<GHENT_ROOT>` | the private Ghent data |
| `<BRANT_SRC>`, `<BRANT_WEIGHTS>` | Brant's code and weights |

Every number, hyperparameter, split record and note is byte-identical to the
original. What is left behind the placeholders (for example
`<OUTPUT_ROOT>/btb_rebench/bce_tune/lr3e-4` or `<OUTPUT_ROOT>/P7FM/Stanford/...`)
is the private run folder, kept so a file can be traced to the run that wrote it.

Not included: feature caches and embeddings (`.npz` / `.npy`), predictions,
logs, launch scripts, superseded runs (the MSE-loss, lr 1e-3, dropout-contaminated
and mean-pool-only Brant generations kept beside the final ones in the private
tree), controls quoted only in commented-out text, and the two cells the paper
does not report (below).

Numbers the paper quotes only in its prose, not in a table, have no per-cell
file here and are not recomputed by the two aggregators:

| Where | Numbers | Control |
| --- | --- | --- |
| App. A.11, protocol paragraph (BrainBERT centre pooling) | ≈34 % lower r | model-free proxy: ridge on raw high-gamma over the same 1 s windows, three Stanford subjects, centre vs full-window pooling |
| App. A.11, adaptation paragraph | BrainBERT 0.040, PopT 0.053; +0.055 (PopT, `bp`); `cc` 0.150 / 0.136 | full-backbone fine-tuning, 9 Stanford subjects (the frozen-probe comparison values, 0.163 and 0.149, are `probe/Stanford/*/per_subject` here) |
| App. A.11, adaptation paragraph | pooled r near zero | joint fine-tuning of the last two FM blocks and the temporal head |
| App. A.11, extraction check | 0.094 ± 0.059, 0.107, 0.166 ± 0.064, t = 5.57, 0.179, 0.062, 0.374, 0.337, 0.247, 0.326 | per-electrode ridge readouts of re-extracted embeddings |
| App. A.11, Brant paragraph | 0.025 | Brant on a 0.1 s anchor grid |
| App. A.11, Brant paragraph | 0.059 → 0.107 (n = 8) | Brant with its 15-patch context |
| App. A.11, representation analysis | 0.181 and the reconstruction results | frozen-embedding representation analysis |
| App. A.11, seizure-onset zone | 0.654, 0.538 | omni_ieeg seizure-onset-zone control |
| App. A.2, BrainTreebank windows | 0.85–0.93 AUROC (mean 0.90) | the pause before each Task A word alone, no neural data |
| App. A.2, BrainTreebank windows | 0.540 ± 0.009 | Brant patch ending at onset |

The pause-only AUROC is recomputed by `scripts/btb_pause_only_auroc.py`
(transcripts only, CPU), as the paper says. The 0.1 s Brant grid is run by
`scripts/table1_ieeg_fm_stanford.sh` (`brant_s0.1/`; `aggregate_fm.py` lists it
under "Other cells"). The code for the other controls is not part of this
release.

Known stale fields, left as recorded:

- `btb/fm_*/results_sub_*.json` say `"cv_folds": 5`. That is a leftover literal;
  the runs used 4 folds, as `splits` in the same file shows.
- The CORTEG files in `btb/corteg_*` have no `merge_strategy` field except the
  mean-pool runs (`..._average...json`, `"average"`). The others predate the flag
  and used the layer-wise gate; the aggregator reads a missing field as the gate.
- The gate runs' `config.save_root` points at a learning-rate sweep folder
  (`bce_tune/lr3e-4`, `bce_tune/lr3e-4_taskB`): the lr 3e-4 runs were copied into
  `corteg_<task>/` after the sweep chose that rate.
- The `note` of `btb/max_electrode_null/*.json` calls the FM probes
  "higher-variance". The paper's wording is the accurate one: their scores are
  less correlated across electrodes.

## Aggregation rule

The same for every table, and the only rule the scripts implement:

1. **Per-subject score** — the value stored per subject in each file: mean AUROC
   over the 4 causal forward-chaining folds (BrainTreebank), or Pearson r on the
   test split averaged over output dimensions, `corr_mean` (regression).
2. **Seeds** — a multi-seed cell averages its seeds *within each subject* first.
3. **Cell** — mean over subjects ± sample SD (ddof=1) across subjects, on the
   seed-averaged scores (n = 10 BrainTreebank, 9 Stanford finger, 16 Ghent audio).
4. **Seed SD** (Table 20 only) — sample SD (ddof=1) of the per-seed cohort means.
5. **Rounding** — once, from full precision: 4 dp for Table 20, 3 dp elsewhere.

Rounding once matters for fifteen cells whose full-precision value sits just
below a 3-dp boundary (e.g. CORTEG, Task B SD 0.134495, or Ghent LOO PopT
fine-tune 0.012479): rounding a 4-dp value again (Table 20's 0.1345 for the
first; 0.0125, the 4-dp summary of the second) would print them 0.001 high.
The paper and the scripts round once; the fourteen BrainTreebank cells are
listed in `DOUBLE_ROUNDING_TRAPS` in `scripts/aggregate_btb.py`, and
`tests/test_paper_cells.py` checks all fifteen.

## BrainTreebank: `btb/`

`<task>` is `sentence_onset` (Task A, sentence-initial vs mid-sentence word) or
`word_nonword` (Task B, word vs non-word). All arms share the same test folds.
"Released runner" names the script in this release that regenerates the row;
the rows marked "not released" can be recomputed from these files but not
re-run.

| Paper row (Table 20; Table 3 / 9 rows marked T3) | Files | Per-subject value | Seeds | Released runner |
| --- | --- | --- | --- | --- |
| Raw high-gamma, per-channel | `fm_<task>/results_sub_<s>_<trial>.json` | `arms.rawHGA_perchannel` | deterministic | not released |
| Raw band-power, per-channel | same | `arms.rawBandpower_perchannel` | deterministic | not released |
| Raw high-gamma, channel-mean | same | `arms.rawHGA_popmean_legacy` | deterministic | not released |
| CORTEG (layer-wise gate) — T3 "CORTEG (ours)" | `corteg_<task>/results_pooled_small_s<S>_lora4_mni_hfa70-200[_word_nonword].json` | `per_subject_auroc` | 42, 1, 2 | `experiments/run_btb_classification.py` (`scripts/table3_corteg_braintreebank.sh`) |
| CORTEG (mean-pool fusion) | `corteg_<task>/..._hfa70-200_average[_word_nonword].json` | `per_subject_auroc` | 42, 1, 2 | same, `--merge_strategy average` |
| CORTEG, random-init backbone — T3 | `corteg_<task>_random/..._randinit[_word_nonword].json` | `per_subject_auroc` | 42, 1, 2 | same, `--no_pretrained` |
| HiLoFuseNet — T3; CNN-LSTM; LSTM | `base_HiLoFuseNet/`, `base_CNN_LSTM/`, `base_LSTM/`: `results_pooled_<task>_s<S>.json` | `per_subject_auroc` | 42, 1, 2 | `experiments/run_btb_baselines.py` (`scripts/table3_btb_baselines.sh`) |
| PopT, LoRA — T3 "PopT"; full fine-tune; head-only | `popt_finetune_<task>/results_popt_{lora,full_ft,head_only}_<task>_seed42.json` | `per_subject.<s>.mean_auroc` | 42 | `experiments/run_popt_finetune_btb.py` (`scripts/table3_ieeg_fm_braintreebank.sh`) |
| PopT, frozen probe | `fm_<task>/results_sub_*.json` | `arms.PopT_population` | deterministic | `experiments/run_ieeg_fm_baselines.py` (same script) |
| BrainBERT, single-elec. max — T3 "BrainBERT†" | `fm_<task>/results_sub_*.json` | `arms.BrainBERT_single_elec_max` | deterministic | `experiments/run_ieeg_fm_baselines.py` |
| BrainBERT, population mean-pool / single-elec. mean | same | `arms.BrainBERT_pop_meanpool` / `arms.BrainBERT_single_elec_mean` | deterministic | `experiments/run_ieeg_fm_baselines.py` |
| BrainBERT, full fine-tune / head-only / LoRA | `brainbert_finetune_<task>/results_bb_{full_ft,head_only,lora}_<task>_seed42.json` | `per_subject.<s>.mean_auroc` | 42 | not released |
| Brant, single-elec. max — T3 "Brant†" | `brant_<task>/brant_sub_<s>_<trial>__<task>.json` | `arms.Brant_single_elec_max` | deterministic | `experiments/run_ieeg_fm_baselines.py` |
| Brant, single-elec. mean / population mean-pool | same | `arms.Brant_single_elec_mean` / `arms.Brant_population` | deterministic | `experiments/run_ieeg_fm_baselines.py` |
| Brant, head-only / LoRA / full fine-tune | `brant_finetune_<task>/results_brant_{head_only,lora,full_ft}_<task>_seed42.json` | `per_subject.<s>.mean_auroc` | 42 | not released |
| Oracle permutation null, "at least 0.53" (Table 20 footnote, Table 9 caption) | `max_electrode_null/max_null_<task>.json` | mean of `per_subject.<s>.null_mean` | 60 permutations | not released |

Table 3 = the six T3 rows at 3 dp. Table 9 = their per-subject scores (after
the within-subject seed average) at 3 dp; its Mean column is Table 3. The three
CORTEG seeds and the three scratch-decoder seeds score one event set, drawn once
with seed 42: their `splits` records are identical, which the aggregator checks.
The oracle null (cohort means 0.5297 and 0.5284; the paper's "0.53" is both
at 2 dp, which the aggregator checks) is a lower bound: it was measured by
permuting labels through the same max-over-electrodes search on a
one-dimensional high-gamma feature, not on the fitted FM probes.

## Regression FMs: `fm_regression/`

Layout: `<adaptation>/<dataset>/<fm>/<regime>/seed<S>/results_{pooled,persub}.json`,
and for LOO `<adaptation>/<dataset>/<fm>/loo/seed42/_loo_done.json` plus one
`ft_<subject>/results_persub.json` per held-out subject. This is the layout
`experiments/run_ieeg_fm_regression.py` and `experiments/run_brant_regression.py`
write under `$CORTEG_OUTPUT_ROOT/ieeg_fm_regression/`.

| Folder | Adaptation (Table 18 row) | Regimes, seeds | Private run |
| --- | --- | --- | --- |
| `probe/` | frozen linear probe | pooled, per-subject: 42, 0, 1; LOO: 42 | P7FM (Stanford), P6FM (Ghent) |
| `ft/` | last-N fine-tune (last 2 blocks) | pooled, per-subject: 42, 0, 1; LOO: 42 | P9FM (Stanford), P8FM (Ghent) |
| `temporal/` | temporal head (BiLSTM) | pooled, per-subject: 42 | the temporal-head run |
| `brant/` | Brant frozen probe, 6 s context | per-subject: 42 (and an unreported pooled head, 42) | P6BRANT |

`<dataset>` is `Stanford` (finger, n = 9) or `Ghent` (audio, n = 16);
`<fm>` is `brainbert`, `popt` or `brant`. Per-subject value: `per_subject.<s>.corr_mean`.
Every Stanford cell here except Brant's pooled head is regenerated by
`scripts/table1_ieeg_fm_stanford.sh` (`experiments/run_ieeg_fm_regression.py`,
`experiments/run_brant_regression.py`); the Ghent cells cannot be, since the
Ghent data are not released.

- **Table 19** — the mean of each BrainBERT / PopT probe and fine-tune cell,
  per dataset and regime (pooled / per-subject / LOO).
- **Table 18** — per model × adaptation × task, the regime with the highest
  mean, with that regime's SD; ¶ marks a winner that averaged three seeds.
  Brant's cell is its per-subject run (the r App. A.11 quotes). A pooled
  head ran beside it (finger 0.001 ± 0.018, audio 0.014 ± 0.035) and is not
  reported; `aggregate_fm.py` prints it under Table 18 but never picks it, so
  a fresh Brant cell does not need a pooled run. This gives 11/14 cells with
  SD > mean and a maximum of 0.0626 (< 0.07), as the caption says.
- **Table 1, FM rows** — per model and task, the Table 18 cell with the highest mean.

A LOO cell counts only through its `_loo_done.json`. Each `ft_<subject>/`
beside it holds the run for one held-out subject (trained on the others, then
trained further on that subject's training split); those files are never
counted on their own, and a LOO folder without `_loo_done.json` is reported as
incomplete.

## Cells the paper does not report

Two Table 19 cells have no complete run on record. The paper prints "---"
for them, with a footnote; their files are not here and the aggregator prints `--`:

| Cell | What exists |
| --- | --- |
| Stanford, pooled, BrainBERT ft | seed 42 only (r = 0.039); the seed 0 and 1 outputs were never retrieved from the cluster |
| Ghent, LOO, BrainBERT ft | 3 of 16 LOO folds |

Neither would be the best regime of its row, so Tables 18 and 1 do not depend
on them.

## Verification status

- **These files reproduce the paper — verified.** Every printed cell of
  Tables 3, 9, 20, 18, 19 and the Table 1 FM rows, and the oracle null, is
  recomputed exactly from this folder by the two scripts (CPU only);
  `tests/test_paper_cells.py` keeps it that way.
- **The released code reproduces these files — not yet verified** on the
  paper's hardware (GPU, mixed precision) for any BrainTreebank or FM cell.
  What exists so far:
  - BrainTreebank: the only earlier public-code run (CORTEG gate, Task A,
    seed 42, cohort 0.6364) predates the readout fix and used a different
    training recipe (100 epochs, batch 32, patience 20, head dropout 0.1,
    minimum lr 1e-6, 5 warm-up epochs, subjects in natural order); the
    aggregator lists it under "Other cells", and it is not evidence either way.
  - FM regression, Stanford: one run of the released code on CPU (fp32, since
    AMP is off on CPU; seed 42 only, all nine subjects). Of the 7 seeds it
    shares with this folder, 3 are within the seed-matched tolerances and 4 are
    outside: probe PopT per-subject (cohort d −0.0069, max |d| 0.033, within
    the paper's own seed-to-seed spread for that cell), probe BrainBERT LOO
    (mean |d| 0.011), temporal head BrainBERT per-subject (mean |d| 0.013) and
    temporal head PopT per-subject (mean |d| 0.013, max |d| 0.038). The 6
    published values it can score (the two LOO probe cells of Table 19, and
    the mean and SD of Brant's Table 18 and Table 1 finger cells) are within
    tolerance. A CPU run is not like-for-like with the paper's GPU runs; the
    GPU reruns are pending.

  Check new runs seed by seed, as below, and record the outcome in the README.
- **Ghent cells** cannot be re-run from this release; these files are the
  only public record of them.

## Aggregating fresh runs

With the release scripts' default output folders
(`scripts/table3_corteg_braintreebank.sh` -> `table3/`,
`scripts/table3_btb_baselines.sh` -> `base_<decoder>/`,
`scripts/table3_ieeg_fm_braintreebank.sh` -> `fm_runs/`,
`scripts/table1_ieeg_fm_stanford.sh` -> `ieeg_fm_regression/`):

```bash
OUT="${CORTEG_OUTPUT_ROOT:-$HOME/workspace/outputs/corteg}"
python scripts/aggregate_btb.py --cells "$OUT/braintreebank/table3" \
    "$OUT/braintreebank/base_HiLoFuseNet" "$OUT/braintreebank/base_CNN_LSTM" \
    "$OUT/braintreebank/base_LSTM" "$OUT/braintreebank/fm_runs"
python scripts/aggregate_fm.py  --cells "$OUT/ieeg_fm_regression"
```

`--cells` takes any number of folders; give it the folders of one set of runs.
A `--cells` argument that is not a folder is an error, so a mistyped folder
cannot pass unnoticed.

Files are classified by their contents, so any folder layout works, and
everything that is not a result file (caches, summaries, logs) is skipped
(`--verbose` lists it). Recognised:

| Runner | Result file | Cells |
| --- | --- | --- |
| `experiments/run_btb_classification.py` | `btb_<train_mode>_<merge>_<endpoint>[tags]_seed<S>.json` | CORTEG gate / mean-pool / random init |
| `experiments/run_btb_baselines.py` | `results_<train_mode>_<endpoint>_s<S>[_ev<E>][_<tag>...].json` | HiLoFuseNet, CNN-LSTM, LSTM |
| `experiments/run_popt_finetune_btb.py` | `popt_<endpoint>_<mode>_seed<S>.json` | PopT LoRA / full fine-tune / head-only |
| `experiments/run_ieeg_fm_baselines.py` | `<fm>_<endpoint>_<arm>_seed<S>.json` | frozen BrainBERT / PopT / Brant arms |
| `experiments/run_ieeg_fm_regression.py` | `<adaptation>/Stanford/<fm>/<regime>/seed<S>/results_{pooled,persub}.json`, `_loo_done.json` | BrainBERT / PopT probe, last-N ft, temporal head |
| `experiments/run_brant_regression.py` | `brant/Stanford/brant/<regime>/seed<S>/results_{pooled,persub}.json` | Brant |

A run at a non-paper setting is listed under "Other cells" and never averaged
into a published row. That covers the event set (another event seed or event
count, other Task B negatives), the training arrangement (per-subject
training, e.g. `hilofusenet[per_subject]`; shared LoRA; per-subject model
selection), the subjects of a CORTEG or decoder run and, when it is pooled,
their training order (e.g. `corteg_gate[subj9-2]`), a forced `--trial`, the runners' own
folder tags (`--bb_pool last10`, a causal temporal head, more Brant patches,
...), and the recipe values in the scripts' `RECIPE_*` tables, which are the
values the paper runs record (folds, window, high-gamma band, backbone size,
gate activation, epochs, learning rates, minimum lr and warm-up, gradient
clipping, validation interval, batch size, LoRA rank, re-referencing, blocks
unfrozen, ...), as well as truncated runs (`--max_windows`, `--max_anchors`).
As a last resort, a BrainTreebank file whose runner records its own non-paper
settings (`nonpaper_settings`, `name_tags`) gets those as its tag when none of
the checks above fired. The same cell and seed found in two files is an error.

How fresh cells are checked:

1. **Published cells** are scored only when the fresh cell has the paper's
   seeds and whole cohort (and, for BrainTreebank, one event set across its
   seeds). A seed-42-only run of a three-seed row is shown and marked
   "not scored": one seed is not comparable with a three-seed mean. A seed-42
   CORTEG gate run should land near 0.6499 on Task A and 0.7591 on Task B, not
   near the three-seed 0.6376 / 0.7492. A Table 18 cell is scored only when
   every regime it is chosen from is there (for Brant, the per-subject run
   alone), a Table 1 cell only when every adaptation behind it is. Seed SDs
   are shown, never scored.
2. **Seed by seed**: each fresh seed that also exists here is compared with the
   same seed in this folder, subject by subject. This is the check a
   single-seed rerun passes or fails.
3. BrainTreebank seeds of one row scored on different event sets fail.
4. A run in which nothing could be compared (only smoke or non-paper runs, or
   no result files) exits non-zero with "NOTHING VERIFIED". A `--reference`
   that is not a folder is an error, and `--tol` is refused without `--cells`:
   this folder is always checked exactly.

The seed-by-seed check presumes that the released runner draws its random
numbers (initialisation, data order, dropout) in the order the paper code did
for that seed. A port that draws them in another order behaves like another
seed, and would fail the check although its recipe is right. So for each
multi-seed row the scripts print how far the paper's own seeds lie from each
other, and mark a failure that is no further from its paper seed than that
("within the reference's seed-to-seed spread"): that pattern points at the
random stream rather than the recipe. For some rows (BrainTreebank LSTM,
HiLoFuseNet and mean-pool CORTEG on Task B) the paper's seeds already agree
within the tolerances, so there the check cannot tell another seed from the
same one; the output says so.

Tolerances. The BrainTreebank floor was measured for the paper from three runs
of the identical Task B gate configuration at seed 42 (the paper's run and two
repeats; they differ only by GPU nondeterminism). The largest difference
between any two of them is 0.0074 in the cohort mean, 0.0098 in the mean
|difference| over subjects and 0.0284 for any one subject. Three runs
understate the spread, so the defaults are 1.5 times those: 0.0111, 0.0148
and 0.0426 (`--tol`, `--tol_subject`, `--tol_subject_max`). The floor was
measured on that one arm and task only and is applied, unmeasured, to Task A
and to every other arm (decoders, PopT, frozen probes). Different seeds of one
row differ by far more (on Task A the gate's per-seed cohort means are 0.650,
0.640 and 0.623, mean-pool's 0.650, 0.611 and 0.587), which is why only the
same seed is compared. The tolerances are estimates, not
constants; rerun a seed that lands just outside before reading it as a failed
reproduction. No floor was measured for the regression cells; the FM defaults
(0.005, 0.01, 0.03) are heuristics, and a CPU rerun (fp32 instead of the
paper's GPU mixed precision) is expected to exceed them, as the CPU run above
did on 4 of 7 seeds.
