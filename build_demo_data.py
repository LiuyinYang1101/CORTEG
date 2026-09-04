"""Build compact JSON files for the docs/index.html interactive demo.

Extracts predictions for selected models and subjects (Stanford finger movement
+ Ghent speech envelope), downsamples to keep the static site small, and writes
per-subject JSON manifests.

Model labels here are paper-aligned. The Stanford set covers Table 1 main rows
plus the key "random init" ablation. The Ghent set is included for
visualization only (the dataset itself is private).
"""
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get(
    "CORTEG_OUTPUT_ROOT",
    os.path.expanduser("~/workspace/outputs/corteg"),
))
DOCS_DATA = Path(__file__).resolve().parent / "docs" / "data"

# ─── Stanford (finger movement, 5 outputs) ───
STANFORD_OUT = DOCS_DATA / "stanford"
STANFORD_OUT.mkdir(parents=True, exist_ok=True)

STANFORD_SUBJECTS = ["bp", "cc", "ht", "jc", "jp", "mv", "wc", "wm", "zt"]
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# Paper-aligned: 6 models — Table 1 main rows + one ablation
STANFORD_MODELS = {
    "CORTEG (pooled) ⭐":         "stanford_best_lora_adapter",
    "Random init (no pretrain)":     "stanford_random_fullft_adapter_rerun_v2",
    "CORTEG (per-subject)":          "stanford_persub_pretrained_lora/f1.0",
    "HiLoFuseNet":                   "stanford_lowdata_baselines/hilofusenet_f1.0_with_preds",
    "Ridge":                         "stanford_lowdata_baselines/ridge_f1.0_with_preds",
    "PLS":                           "stanford_lowdata_baselines/pls_f1.0_with_preds",
}

# ─── Ghent (speech envelope, 1 output) ───
GHENT_OUT = DOCS_DATA / "ghent"
GHENT_OUT.mkdir(parents=True, exist_ok=True)

GHENT_SUBJECTS = ["2018_001", "2019_001", "2019_003", "2019_004", "2019_007",
                  "2020_001", "2020_002", "2020_004", "2020_005", "2020_006",
                  "2021_001", "2021_002", "2021_005-1", "2021_006", "2021_007",
                  "2021_008-1"]

# Paper-aligned: 6 models — same scheme as Stanford, no CNN-LSTM (not stored on disk)
GHENT_MODELS = {
    "CORTEG (pooled) ⭐":         "ghent_mni_corrected/pretrained_lora_adapter",
    "Random init (no pretrain)":     "ghent_mni_corrected/random_lora_adapter",
    "CORTEG (per-subject)":          "ghent_mni_corrected/persub_pretrained",
    "HiLoFuseNet":                   "ghent_mni_corrected/hilofusenet_persub_with_preds",
    "Ridge":                         "ghent_mni_corrected/ridge_persub_with_preds",
    "PLS":                           "ghent_mni_corrected/pls_persub_with_preds",
}

TARGET_LENGTH = 1000


def load_pred(model_path: str, sub: str):
    p = ROOT / model_path / "predictions" / sub
    yt = p / "y_true.npy"
    yp = p / "y_pred.npy"
    if not yt.exists() or not yp.exists():
        return None
    return np.load(yt), np.load(yp)


def downsample(arr, n_target):
    n = arr.shape[0]
    if n <= n_target:
        return arr
    idx = np.linspace(0, n - 1, n_target, dtype=int)
    return arr[idx]


def pearson_per_finger(yt, yp):
    return [float(np.corrcoef(yt[:, i], yp[:, i])[0, 1]) for i in range(5)]


# ─── Stanford build ───
print("\n=== Stanford ===")
manifest = {"subjects": STANFORD_SUBJECTS, "fingers": FINGERS, "models": list(STANFORD_MODELS.keys())}
with open(STANFORD_OUT / "manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(f"Wrote {STANFORD_OUT / 'manifest.json'}")

for sub in STANFORD_SUBJECTS:
    sub_data = {"subject": sub, "fingers": FINGERS, "models": {}}
    y_true_ref = None
    for model_name, model_path in STANFORD_MODELS.items():
        loaded = load_pred(model_path, sub)
        if loaded is None:
            print(f"  [SKIP] {model_name}/{sub}")
            continue
        yt, yp = loaded
        if y_true_ref is None:
            y_true_ref = yt
        sub_data["models"][model_name] = {
            "y_pred": downsample(yp, TARGET_LENGTH).round(4).tolist(),
            "corr_per_finger": [round(c, 4) for c in pearson_per_finger(yt, yp)],
            "corr_mean": round(float(np.mean(pearson_per_finger(yt, yp))), 4),
        }
    if y_true_ref is not None:
        sub_data["y_true"] = downsample(y_true_ref, TARGET_LENGTH).round(4).tolist()
        sub_data["fs_hz"] = 25
        sub_data["duration_s"] = round(len(y_true_ref) / 200.0, 1)
    out_path = STANFORD_OUT / f"{sub}.json"
    with open(out_path, "w") as f:
        json.dump(sub_data, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"  Wrote {out_path.name}  ({len(sub_data['models'])} models, {size_kb:.1f} KB)")

# ─── Ghent build (speech envelope, 1D target) ───
print("\n=== Ghent ===")
g_manifest = {"subjects": GHENT_SUBJECTS, "models": list(GHENT_MODELS.keys())}
with open(GHENT_OUT / "manifest.json", "w") as f:
    json.dump(g_manifest, f, indent=2)
print(f"Wrote {GHENT_OUT / 'manifest.json'}")

for sub in GHENT_SUBJECTS:
    sub_data = {"subject": sub, "models": {}}
    y_true_ref = None
    for model_name, model_path in GHENT_MODELS.items():
        loaded = load_pred(model_path, sub)
        if loaded is None:
            print(f"  [SKIP] {model_name}/{sub}")
            continue
        yt, yp = loaded
        # Squeeze (T, 1) → (T,) for envelope
        yt = yt.squeeze() if yt.ndim > 1 else yt
        yp = yp.squeeze() if yp.ndim > 1 else yp
        if y_true_ref is None:
            y_true_ref = yt
        r = float(np.corrcoef(yt, yp)[0, 1])
        sub_data["models"][model_name] = {
            "y_pred": downsample(yp, TARGET_LENGTH).round(4).tolist(),
            "corr": round(r, 4),
        }
    if y_true_ref is not None:
        sub_data["y_true"] = downsample(y_true_ref, TARGET_LENGTH).round(4).tolist()
        # Ghent target is at 200Hz; predictions made every 50ms → ~20 Hz effective
        sub_data["fs_hz"] = 20
        sub_data["duration_s"] = round(len(y_true_ref) / 20.0, 1)
    out_path = GHENT_OUT / f"{sub}.json"
    with open(out_path, "w") as f:
        json.dump(sub_data, f, separators=(",", ":"))
    size_kb = out_path.stat().st_size / 1024
    print(f"  Wrote {out_path.name}  ({len(sub_data['models'])} models, {size_kb:.1f} KB)")

stanford_total = sum(p.stat().st_size for p in STANFORD_OUT.glob('*.json')) / 1024
ghent_total = sum(p.stat().st_size for p in GHENT_OUT.glob('*.json')) / 1024
print(f"\nTotal demo data size: Stanford {stanford_total:.1f} KB + Ghent {ghent_total:.1f} KB")
