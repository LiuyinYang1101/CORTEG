"""
Per-subject baseline experiments on Ghent speech dataset.

Runs HiLoFuseNet and DeepFingerNet on each Ghent subject individually.
Uses our standard training framework (MSE loss, Pearson r, early stopping).

HiLoFuseNet input: (B, C, T=200, F=2) — HGA + LFS stacked
DeepFingerNet input: (B, C*n_wavelets, T) — Morlet wavelet spectrograms

Usage:
    python -m experiments.run_ecog_baselines --model HiLoFuseNet
    python -m experiments.run_ecog_baselines --model DeepFingerNet
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from experiments.common import set_seed
from train.earlystop import EarlyStopper
from train.metrics import corr_per_dim


# ─── Ghent data loading ───

def load_ghent_subject(h5_path: str):
    """Load raw ECoG streams and envelope from a Ghent HDF5 file.

    Returns per-split arrays: train/val/test for each stream.
    """
    import h5py
    h5 = h5py.File(h5_path, "r")
    n_ch = int(h5.attrs["n_channels_good"])
    splits = {}
    for s in ["train", "val", "test"]:
        g = h5[s]
        splits[s] = {
            "ecog_256": np.array(g["ecog_256hz"], dtype=np.float32),
            "ecog_128": np.array(g["ecog_128hz"], dtype=np.float32),
            "hga_200": np.array(g["hga_200hz"], dtype=np.float32),
            "envelope_128": np.array(g["envelope_128hz"], dtype=np.float32),
            "envelope_256": np.array(g["envelope_256hz"], dtype=np.float32),
            "envelope_200": np.array(g["envelope_200hz"], dtype=np.float32),
        }
    h5.close()
    return {"splits": splits, "n_ch": n_ch}


# ─── Dataset for HiLoFuseNet ───

class GhentHiLoDataset(Dataset):
    """Ghent windowed dataset returning (x_hilo, y) for HiLoFuseNet.

    x_hilo: (C, T=200, F=2) where F[0]=HGA, F[1]=LFS (both at 200 Hz)
    y: scalar (speech envelope, endpoint of window)
    """
    def __init__(self, hga_200: np.ndarray, lfs_200: np.ndarray,
                 envelope_200: np.ndarray, window_samples: int = 200,
                 step: int = 1):
        # hga_200, lfs_200: (N, C), envelope_200: (N,)
        self.hga = hga_200
        self.lfs = lfs_200
        self.env = envelope_200
        self.win = window_samples
        self.step = step
        self.n_windows = (len(envelope_200) - window_samples) // step

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.step
        hga_win = self.hga[start:start+self.win]  # (T, C)
        lfs_win = self.lfs[start:start+self.win]  # (T, C)
        # Stack: (C, T, 2)
        x = np.stack([hga_win.T, lfs_win.T], axis=-1)  # (C, T, 2)
        y = self.env[start + self.win - 1]  # endpoint
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ─── Dataset for DeepFingerNet ───

class GhentWaveletDataset(Dataset):
    """Ghent windowed dataset returning (x_wavelet, y) for DeepFingerNet.

    x_wavelet: (C * n_freqs, T) — flattened wavelet spectrogram
    y: scalar (speech envelope, endpoint of window)
    """
    def __init__(self, spectrogram: np.ndarray, envelope: np.ndarray,
                 window_samples: int = 256, step: int = 1):
        # spectrogram: (C, n_freqs, N_total) — precomputed
        # envelope: (N_total,)
        self.spec = spectrogram
        self.env = envelope
        self.win = window_samples
        self.step = step
        C, F, N = spectrogram.shape
        self.n_features = C * F
        self.n_windows = (N - window_samples) // step

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.step
        spec_win = self.spec[:, :, start:start+self.win]  # (C, F, T)
        x = spec_win.reshape(self.n_features, self.win)  # (C*F, T)
        y = self.env[start + self.win - 1]  # endpoint
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ─── Dataset for LSTM single-stream ───

class GhentSingleStreamDataset(Dataset):
    """Ghent windowed dataset for single-stream models (LSTM_HGA, LSTM_LFS).

    x: (C, T) — single frequency stream
    y: scalar (speech envelope, endpoint of window)
    """
    def __init__(self, signal: np.ndarray, envelope: np.ndarray,
                 window_samples: int = 200, step: int = 1):
        # signal: (N, C), envelope: (N,)
        self.signal = signal
        self.env = envelope
        self.win = window_samples
        self.step = step
        self.n_windows = (len(envelope) - window_samples) // step

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        start = idx * self.step
        x = self.signal[start:start+self.win].T  # (C, T)
        y = self.env[start + self.win - 1]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)


# ─── Wavelet preprocessing ───

def compute_morlet_spectrograms(ecog_256: np.ndarray, fs: int = 256,
                                 n_wavelets: int = 20,
                                 f_low: float = 1.0, f_high: float = 120.0):
    """Compute Morlet wavelet power spectrograms for each channel.

    Args:
        ecog_256: (N, C) raw ECoG at 256 Hz
        fs: sampling rate
        n_wavelets: number of log-spaced frequencies
        f_low, f_high: frequency range (must be < Nyquist)

    Returns:
        spectrograms: (C, n_wavelets, N) power values
    """
    import mne
    freqs = np.logspace(np.log10(f_low), np.log10(f_high), n_wavelets)
    # mne expects (n_epochs, n_channels, n_times)
    data = ecog_256.T[np.newaxis, :, :]  # (1, C, N)
    power = mne.time_frequency.tfr_array_morlet(
        data, sfreq=fs, freqs=freqs, output='power',
        verbose=False, n_jobs=4,
    )[0]  # (C, n_freqs, N)
    return power.astype(np.float32), freqs


def resample_envelope(envelope_256: np.ndarray, target_rate: int = 200,
                      source_rate: int = 256):
    """Resample envelope from 256 Hz to target rate."""
    from scipy.signal import resample_poly
    from math import gcd
    g = gcd(target_rate, source_rate)
    return resample_poly(envelope_256, up=target_rate // g,
                         down=source_rate // g, axis=0).astype(np.float32)


# ─── Training loop ───

def train_one_subject(model, train_loader, val_loader, test_loader,
                      device, args):
    """Standard per-subject training loop."""
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    early = EarlyStopper(patience=args.patience)
    loss_fn = nn.MSELoss()

    for ep in range(args.epochs):
        # Train
        model.train()
        losses = []
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            pred = model(x).squeeze(-1)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())

        # Validate
        val_r = evaluate(model, val_loader, device)
        if ep % 10 == 0:
            print(f"  [ep {ep+1:3d}] loss={np.mean(losses):.4f} val_r={val_r:.4f}",
                  flush=True)
        if early.step(val_r, model):
            print(f"  Early stop at epoch {ep+1}", flush=True)
            break

    early.restore(model)
    test_r = evaluate(model, test_loader, device)
    return test_r


def evaluate(model, loader, device):
    """Evaluate and return mean Pearson r."""
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).squeeze(-1)
            preds.append(pred.cpu().numpy())
            targets.append(y.numpy())
    preds = np.concatenate(preds)
    targets = np.concatenate(targets)
    if preds.ndim == 1:
        preds = preds[:, None]
        targets = targets[:, None]
    r = float(corr_per_dim(preds, targets).mean())
    return r


# ─── Main ───

def main():
    p = argparse.ArgumentParser("ECoG Baselines on Ghent")
    p.add_argument("--model", type=str, required=True,
                   choices=["HiLoFuseNet", "DeepFingerNet",
                            "LSTM_HGA", "LSTM_LFS", "HOPLS",
                            "CNN_LSTM", "PLS"])
    p.add_argument("--ghent_root", type=str,
                   default=os.path.expanduser("~/workspace/datasets/Ghent_story/processed"))
    p.add_argument("--save_root", type=str, default="")
    p.add_argument("--subjects", type=str, default="",
                   help="Comma-separated subjects (empty=all)")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_wavelets", type=int, default=20,
                   help="Number of Morlet wavelet frequencies (DeepFingerNet)")
    p.add_argument("--window_samples", type=int, default=0,
                   help="Window size in samples (0=auto: 200 for HiLo, 256 for DFN)")
    p.add_argument("--eval_step_ms", type=int, default=50,
                   help="Evaluation stride in ms (match CORTEG's 50ms for fair comparison)")
    p.add_argument("--train_step_ms", type=int, default=0,
                   help="Training stride in ms (0=auto: 40ms for DFN, 5ms for others)")
    p.add_argument("--dropout", type=float, default=0.0,
                   help="Model dropout override (0=use model default)")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_root = args.save_root or os.path.expanduser(
        f"~/workspace/outputs/ECoG_EEGFM/ghent_mni_corrected/{args.model.lower()}_persub")
    os.makedirs(save_root, exist_ok=True)

    # Get subjects
    if args.subjects:
        subjects = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects = sorted([f.replace(".h5", "") for f in os.listdir(args.ghent_root)
                           if f.endswith(".h5")])

    print(f"ECoG Baseline: {args.model} on Ghent ({len(subjects)} subjects)")
    print(f"  device={device}, lr={args.lr}, epochs={args.epochs}, seed={args.seed}")

    all_results = {}

    for si, sub in enumerate(subjects):
        set_seed(args.seed)
        h5_path = os.path.join(args.ghent_root, f"{sub}.h5")
        print(f"\n===== [{si+1}/{len(subjects)}] {sub} =====", flush=True)
        t0 = time.time()

        data = load_ghent_subject(h5_path)
        C = data["n_ch"]
        sp = data["splits"]

        # Compute eval step in samples for the model's native rate
        eval_step_ms = args.eval_step_ms

        if args.model == "HiLoFuseNet":
            win = args.window_samples or 200
            native_fs = 200  # HiLoFuseNet operates at 200 Hz
            eval_step = max(1, int(eval_step_ms * native_fs / 1000))
            train_step = 1  # dense windows for training

            # Use pre-split streams: HGA at 200Hz, LFS from 128Hz upsampled to 200Hz
            # Using ecog_128hz (not ecog_256hz) ensures LFS bandwidth ≤ 64 Hz,
            # matching CORTEG's low-freq stream for a fair comparison.
            def prep_hilo_split(split_data):
                hga = split_data["hga_200"]
                # Upsample 128Hz → 200Hz (preserves 64Hz bandwidth, no extra info)
                from scipy.signal import resample_poly
                lfs = resample_poly(split_data["ecog_128"], up=25, down=16,
                                    axis=0).astype(np.float32)
                env = split_data["envelope_200"]
                N = min(len(hga), len(lfs), len(env))
                return hga[:N], lfs[:N], env[:N]

            hga_tr, lfs_tr, env_tr = prep_hilo_split(sp["train"])
            hga_va, lfs_va, env_va = prep_hilo_split(sp["val"])
            hga_te, lfs_te, env_te = prep_hilo_split(sp["test"])

            # Z-score per channel (fit on train only)
            for arrays in [(hga_tr, hga_va, hga_te), (lfs_tr, lfs_va, lfs_te)]:
                mu = arrays[0].mean(axis=0)
                sd = arrays[0].std(axis=0) + 1e-8
                for arr in arrays:
                    arr -= mu
                    arr /= sd
            env_mu, env_sd = env_tr.mean(), env_tr.std() + 1e-8
            env_tr = (env_tr - env_mu) / env_sd
            env_va = (env_va - env_mu) / env_sd
            env_te = (env_te - env_mu) / env_sd

            ds_tr = GhentHiLoDataset(hga_tr, lfs_tr, env_tr, win, step=train_step)
            ds_va = GhentHiLoDataset(hga_va, lfs_va, env_va, win, step=eval_step)
            ds_te = GhentHiLoDataset(hga_te, lfs_te, env_te, win, step=eval_step)

            from models.baselines import HiLoFuseNet
            model = HiLoFuseNet(C=C, F=2, lstm_hidden=256, output_size=1,
                                D=16, dropout_prob=0.2).to(device)

        elif args.model == "DeepFingerNet":
            # Original paper uses 256 samples at 100Hz = 2.56s
            # At 256Hz, 2.56s = 655 samples. Use 512 (~2s) as practical default.
            win = args.window_samples or 512
            n_wav = args.n_wavelets
            native_fs = 256  # DeepFingerNet operates at 256 Hz
            eval_step = max(1, int(eval_step_ms * native_fs / 1000))
            # Original uses stride=1 at 100Hz. At 256Hz stride=1 is too dense.
            # 40ms stride at 256Hz = 10 samples (matches original's ~40ms at 100Hz)
            train_step_ms = args.train_step_ms or 40
            train_step = max(1, int(train_step_ms * native_fs / 1000))

            # Compute Morlet wavelets per split
            def prep_wavelet_split(split_data):
                print(f"    wavelet on {split_data['ecog_256'].shape[0]} samples...",
                      end="", flush=True)
                spec, freqs = compute_morlet_spectrograms(
                    split_data["ecog_256"], fs=256, n_wavelets=n_wav,
                    f_low=1.0, f_high=120.0)
                print(f" done ({spec.shape})", flush=True)
                return spec, split_data["envelope_256"]

            print(f"  Computing {n_wav} Morlet wavelets (1-120 Hz)...", flush=True)
            spec_tr, env_tr = prep_wavelet_split(sp["train"])
            spec_va, env_va = prep_wavelet_split(sp["val"])
            spec_te, env_te = prep_wavelet_split(sp["test"])
            # Free raw ECoG after wavelet computation
            del sp

            # Z-score per channel×freq (fit on train)
            mu = spec_tr.mean(axis=2, keepdims=True)
            sd = spec_tr.std(axis=2, keepdims=True) + 1e-8
            spec_tr = (spec_tr - mu) / sd
            spec_va = (spec_va - mu) / sd
            spec_te = (spec_te - mu) / sd

            env_mu, env_sd = env_tr.mean(), env_tr.std() + 1e-8
            env_tr = (env_tr - env_mu) / env_sd
            env_va = (env_va - env_mu) / env_sd
            env_te = (env_te - env_mu) / env_sd

            ds_tr = GhentWaveletDataset(spec_tr, env_tr, win, step=train_step)
            ds_va = GhentWaveletDataset(spec_va, env_va, win, step=eval_step)
            ds_te = GhentWaveletDataset(spec_te, env_te, win, step=eval_step)

            from models.baselines import DeepFingerNet
            dfn_dropout = args.dropout if args.dropout > 0 else 0.3
            model = DeepFingerNet(n_input_features=C * n_wav, d_out=1,
                                  dropout=dfn_dropout).to(device)

        elif args.model in ("LSTM_HGA", "LSTM_LFS"):
            is_hga = args.model == "LSTM_HGA"
            if is_hga:
                # HGA at 200 Hz, decimate by 10 → 20 time steps (matching paper)
                native_fs = 200
                decimate = 10
                win = args.window_samples or 200
                stream_key = "hga_200"
                env_key = "envelope_200"
            else:
                # LFS at 128 Hz, decimate by 6 → ~21 steps (comparable to LSTM_HGA's 20)
                native_fs = 128
                decimate = 6
                win = args.window_samples or 128
                stream_key = "ecog_128"
                env_key = "envelope_128"

            eval_step = max(1, int(eval_step_ms * native_fs / 1000))

            def prep_lstm_split(split_data):
                sig = split_data[stream_key]
                env = split_data[env_key]
                N = min(len(sig), len(env))
                return sig[:N], env[:N]

            sig_tr, env_tr = prep_lstm_split(sp["train"])
            sig_va, env_va = prep_lstm_split(sp["val"])
            sig_te, env_te = prep_lstm_split(sp["test"])

            # Z-score (fit on train)
            mu_s, sd_s = sig_tr.mean(axis=0), sig_tr.std(axis=0) + 1e-8
            for arr in [sig_tr, sig_va, sig_te]:
                arr -= mu_s
                arr /= sd_s
            env_mu, env_sd = env_tr.mean(), env_tr.std() + 1e-8
            env_tr = (env_tr - env_mu) / env_sd
            env_va = (env_va - env_mu) / env_sd
            env_te = (env_te - env_mu) / env_sd

            ds_tr = GhentSingleStreamDataset(sig_tr, env_tr, win, step=1)
            ds_va = GhentSingleStreamDataset(sig_va, env_va, win, step=eval_step)
            ds_te = GhentSingleStreamDataset(sig_te, env_te, win, step=eval_step)

            from models.baselines import LSTMSingleStream
            model = LSTMSingleStream(n_channels=C, d_out=1, hidden_size=256,
                                      decimate=decimate, dropout=0.2).to(device)

        elif args.model == "HOPLS":
            # CP-PLSR: tensor PLS on wavelet features (CP decomposition)
            # Approximates HOPLS (Tucker-based) using tensorly's CP_PLSR
            # Input: (N, C, n_wavelets) — decimated wavelet features
            native_fs = 256
            n_wav = args.n_wavelets

            print(f"  Computing {n_wav} Morlet wavelets for HOPLS...", flush=True)

            def prep_hopls_split(split_data):
                spec, _ = compute_morlet_spectrograms(
                    split_data["ecog_256"], fs=256, n_wavelets=n_wav,
                    f_low=1.0, f_high=120.0)
                # spec: (C, n_wav, N) → decimate temporal by 10 via avg pool
                # Then reshape to (N_dec, C, n_wav) for tensor PLS
                N = spec.shape[2]
                # Decimate: average non-overlapping windows of 10
                n_dec = N // 10
                spec_dec = spec[:, :, :n_dec*10].reshape(C, n_wav, n_dec, 10).mean(axis=3)
                # spec_dec: (C, n_wav, n_dec) → transpose to (n_dec, C, n_wav)
                X = spec_dec.transpose(2, 0, 1)  # (n_dec, C, n_wav)
                # Average envelope in matching windows (not subsampling)
                env = split_data["envelope_256"][:n_dec*10].reshape(n_dec, 10).mean(axis=1)
                return X.astype(np.float64), env.astype(np.float64)

            X_tr, y_tr = prep_hopls_split(sp["train"])
            X_va, y_va = prep_hopls_split(sp["val"])
            X_te, y_te = prep_hopls_split(sp["test"])
            del sp

            # Z-score
            X_mu = X_tr.mean(axis=0, keepdims=True)
            X_sd = X_tr.std(axis=0, keepdims=True) + 1e-8
            X_tr = (X_tr - X_mu) / X_sd
            X_va = (X_va - X_mu) / X_sd
            X_te = (X_te - X_mu) / X_sd
            y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-8
            y_tr = (y_tr - y_mu) / y_sd
            y_va = (y_va - y_mu) / y_sd
            y_te = (y_te - y_mu) / y_sd

            # Fit CP_PLSR with cross-validation on n_components
            from tensorly.regression import CP_PLSR
            best_r, best_k = -1, 5
            for k in [3, 5, 10, 15, 20]:
                pls = CP_PLSR(n_components=k)
                pls.fit(X_tr, y_tr[:, None])
                pred_va = pls.predict(X_va).flatten()
                r_va = float(np.corrcoef(pred_va, y_va)[0, 1])
                print(f"    K={k}: val_r={r_va:.4f}", flush=True)
                if r_va > best_r:
                    best_r, best_k = r_va, k

            print(f"  Best K={best_k} (val_r={best_r:.4f})", flush=True)
            pls_final = CP_PLSR(n_components=best_k)
            pls_final.fit(X_tr, y_tr[:, None])
            pred_te = pls_final.predict(X_te).flatten()
            test_r = float(np.corrcoef(pred_te, y_te)[0, 1])
            elapsed = time.time() - t0
            print(f"  SCORE = {test_r:.4f} ({elapsed:.0f}s)", flush=True)
            all_results[sub] = {"r": round(test_r, 4), "elapsed": round(elapsed, 1),
                                "best_k": best_k}
            continue  # skip the neural network training loop

        elif args.model == "CNN_LSTM":
            # CNN-LSTM (Lin et al.): same input as HiLoFuseNet (B, C, T, F=2)
            win = args.window_samples or 200
            native_fs = 200
            eval_step = max(1, int(eval_step_ms * native_fs / 1000))

            def prep_cnnlstm_split(split_data):
                hga = split_data["hga_200"]
                from scipy.signal import resample_poly
                lfs = resample_poly(split_data["ecog_128"], up=25, down=16,
                                    axis=0).astype(np.float32)
                env = split_data["envelope_200"]
                N = min(len(hga), len(lfs), len(env))
                return hga[:N], lfs[:N], env[:N]

            hga_tr, lfs_tr, env_tr = prep_cnnlstm_split(sp["train"])
            hga_va, lfs_va, env_va = prep_cnnlstm_split(sp["val"])
            hga_te, lfs_te, env_te = prep_cnnlstm_split(sp["test"])

            for arrays in [(hga_tr, hga_va, hga_te), (lfs_tr, lfs_va, lfs_te)]:
                mu = arrays[0].mean(axis=0)
                sd = arrays[0].std(axis=0) + 1e-8
                for arr in arrays:
                    arr -= mu
                    arr /= sd
            env_mu, env_sd = env_tr.mean(), env_tr.std() + 1e-8
            env_tr = (env_tr - env_mu) / env_sd
            env_va = (env_va - env_mu) / env_sd
            env_te = (env_te - env_mu) / env_sd

            ds_tr = GhentHiLoDataset(hga_tr, lfs_tr, env_tr, win, step=1)
            ds_va = GhentHiLoDataset(hga_va, lfs_va, env_va, win, step=eval_step)
            ds_te = GhentHiLoDataset(hga_te, lfs_te, env_te, win, step=eval_step)

            from models.baselines import CNN_LSTM
            model = CNN_LSTM(input_size=C, output_size=1, dropout_prob=0.2).to(device)

        elif args.model == "PLS":
            # PLS: vectorized wavelet features → sklearn PLSRegression
            # Following HiLoFuseNet paper: Morlet wavelets → vectorize → PLS
            n_wav = args.n_wavelets

            print(f"  Computing {n_wav} Morlet wavelets for PLS...", flush=True)

            def prep_pls_split(split_data):
                spec, _ = compute_morlet_spectrograms(
                    split_data["ecog_256"], fs=256, n_wavelets=n_wav,
                    f_low=1.0, f_high=120.0)
                N = spec.shape[2]
                # Decimate by 10 (avg pool) then flatten C*F per time point
                n_dec = N // 10
                spec_dec = spec[:, :, :n_dec*10].reshape(C, n_wav, n_dec, 10).mean(axis=3)
                # (C, n_wav, n_dec) → (n_dec, C*n_wav) — vectorized
                X = spec_dec.reshape(C * n_wav, n_dec).T
                env = split_data["envelope_256"][:n_dec*10].reshape(n_dec, 10).mean(axis=1)
                return X.astype(np.float64), env.astype(np.float64)

            X_tr, y_tr = prep_pls_split(sp["train"])
            X_va, y_va = prep_pls_split(sp["val"])
            X_te, y_te = prep_pls_split(sp["test"])
            del sp

            # Z-score
            X_mu = X_tr.mean(axis=0, keepdims=True)
            X_sd = X_tr.std(axis=0, keepdims=True) + 1e-8
            X_tr = (X_tr - X_mu) / X_sd
            X_va = (X_va - X_mu) / X_sd
            X_te = (X_te - X_mu) / X_sd
            y_mu, y_sd = y_tr.mean(), y_tr.std() + 1e-8
            y_tr = (y_tr - y_mu) / y_sd
            y_va = (y_va - y_mu) / y_sd
            y_te = (y_te - y_mu) / y_sd

            from sklearn.cross_decomposition import PLSRegression
            best_r, best_k = -1, 5
            for k in [3, 5, 10, 15, 20, 30, 50]:
                pls = PLSRegression(n_components=min(k, X_tr.shape[1]))
                pls.fit(X_tr, y_tr)
                pred_va = pls.predict(X_va).flatten()
                r_va = float(np.corrcoef(pred_va, y_va)[0, 1])
                print(f"    K={k}: val_r={r_va:.4f}", flush=True)
                if r_va > best_r:
                    best_r, best_k = r_va, k

            print(f"  Best K={best_k} (val_r={best_r:.4f})", flush=True)
            pls_final = PLSRegression(n_components=min(best_k, X_tr.shape[1]))
            pls_final.fit(X_tr, y_tr)
            pred_te = pls_final.predict(X_te).flatten()
            test_r = float(np.corrcoef(pred_te, y_te)[0, 1])
            elapsed = time.time() - t0
            print(f"  SCORE = {test_r:.4f} ({elapsed:.0f}s)", flush=True)
            all_results[sub] = {"r": round(test_r, 4), "elapsed": round(elapsed, 1),
                                "best_k": best_k}
            continue  # skip neural network training loop

        print(f"  C={C}, train={len(ds_tr)}, val={len(ds_va)}, test={len(ds_te)}")
        print(f"  params: {sum(p.numel() for p in model.parameters()):,}")

        tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                               num_workers=0)
        va_loader = DataLoader(ds_va, batch_size=args.batch_size)
        te_loader = DataLoader(ds_te, batch_size=args.batch_size)

        test_r = train_one_subject(model, tr_loader, va_loader, te_loader,
                                    device, args)
        elapsed = time.time() - t0
        print(f"  SCORE = {test_r:.4f} ({elapsed:.0f}s)", flush=True)
        all_results[sub] = {"r": round(test_r, 4), "elapsed": round(elapsed, 1)}

    # Save results
    mean_r = np.mean([v["r"] for v in all_results.values()])
    results = {
        "model": args.model,
        "score": round(float(mean_r), 4),
        "per_subject": all_results,
        "args": vars(args),
    }
    out_path = os.path.join(save_root, "results_persub.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {out_path}")
    print(f"Mean r = {mean_r:.4f}")


if __name__ == "__main__":
    main()
