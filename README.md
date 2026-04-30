# CORTEG — Cross-Modal Representation Transfer to ECoG Decoding

> Foundation-model transfer from scalp EEG to intracranial ECoG decoding.

🌐 **[Live interactive demo →](https://liuyinyang1101.github.io/CORTEG/)**

Side-by-side ground-truth vs predicted hand animation (Stanford finger movement) and
speech envelope traces (Ghent), with model selector and trajectory smoothing.

## Highlights

- **Stanford finger movement** (n=9): LOO-finetune matches population training
  (Wilcoxon p=0.65 vs Pool, n=9)
- **Ghent speech envelope** (n=16): LOO-finetune matches population training
  (Wilcoxon p=0.82 vs Pool, n=16) — significantly beats LaBraM, CBraMod, HiLoFuseNet,
  CNN-LSTM, and per-subject baselines (all p<0.01)
- The same recipe **transfers across tasks** without subject-specific re-pretraining

## Demo features

- 🎚️ **Dataset toggle**: Stanford finger movement ↔ Ghent speech envelope
- 👆 **Subject selector** (9 Stanford / 16 Ghent)
- 🤖 **Model selector** with multiple variants (CORTEG-Small/Large, ablations,
  per-subject baselines, classical methods)
- 📈 **Live time-series plot** with all 5 fingers (Stanford) or 1D envelope (Ghent),
  ground-truth vs prediction overlay
- ✋ **3D hand animation** (Stanford only) — side-by-side ground-truth vs predicted
  hand, both driven by the same time slider
- 🌊 **Trajectory smoothing** (off / 5 / 11 / 21-sample moving average)
- 🎯 **Per-finger correlation cards** updated live with the selected model

The demo is fully **static** — runs entirely in the browser via Three.js + Plotly.
No backend required.

## Project structure

```
docs/
├── index.html         ← single-page interactive demo
└── data/
    ├── stanford/      ← finger movement predictions (9 subjects × 8 models)
    └── ghent/         ← speech envelope predictions (16 subjects × 8 models)
build_demo_data.py     ← rebuild the data/ JSONs from raw prediction outputs
```

## Local preview

```bash
git clone https://github.com/LiuyinYang1101/CORTEG.git
cd CORTEG/docs
python -m http.server 8000
# open http://localhost:8000
```

## Reproduce the demo data

`build_demo_data.py` extracts predictions from the research codebase's output
directory and writes the compact JSON files used by `index.html`. Edit the
`STANFORD_MODELS` / `GHENT_MODELS` dicts at the top of the script to add or
remove models.

## License

Demo code: MIT.
Research code and trained models: see the main research repository (private).

## Citation

```
[BibTeX coming soon — paper under review]
```
