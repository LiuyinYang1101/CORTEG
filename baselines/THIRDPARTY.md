# Third-party baselines

CORTEG is compared against several external decoders. We do **not** vendor
these into this repository; reproduce by cloning each project at the
indicated commit and following its own README.

## Deep, single-subject decoders

### DeepFingerNet
- Repo: <https://github.com/MakhanovAA/DeepFingerNet>
- Paper: Petrosyan et al., *DeepFingerNet predicts finger trajectory from ECoG
  measurements*, J. Neural Eng. 2022.
- Our Table 1 number for DeepFingerNet (Finger r = 0.542) is transcribed
  directly from its Table II — we did not retrain it.

### HiLoFuseNet
- Repo: ask the corresponding author of Liu et al., *HiLoFuseNet: Dual-stream
  high/low-frequency fusion for ECoG decoding*, NER 2023.
- Our Finger r = 0.534 is transcribed from its Table V; we re-evaluated under
  the shared protocol of §4 for the audio task.

## Foundation-model controls

| Model | Repo | Used as |
| --- | --- | --- |
| LaBraM | <https://github.com/935963004/LaBraM> | Drop-in EEG FM (Table 2 ablation) |
| CBraMod | <https://github.com/wjq-learning/CBraMod> | Drop-in EEG FM (Table 2 ablation) |
| MantisV2 | <https://github.com/Mantis-TS/MantisV2> | Generic time-series FM control |

Each of these is loaded by a dedicated runner in `experiments/` (e.g.
`run_labram_baseline.py`) that wraps the encoder under our shared training
loop, so the per-subject preprocessing, splits, and early-stopping criteria
are identical across models.

## Classical baselines

Ridge, PLS, HOPLS, LSTM, and CNN-LSTM baselines are implemented inline in
`models/baselines.py` and driven by `experiments/run_ecog_baselines.py`. They
do not require external code.
