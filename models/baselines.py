"""Task-specific decoders trained from scratch.

LSTM, CNN_LSTM and HiLoFuseNet are the "trained from scratch" BrainTreebank
baselines of Tables 3, 9 and 20; ``experiments/run_btb_baselines.py`` drives
them. HiLoFuseNet and the LSTM follow Sun et al. (2025), CNN-LSTM follows Lin
et al. (2025). All three take x as (B, C, T, F) and return one value per
window. They were written for 5-finger regression (``output_size=5``); the
BrainTreebank runner builds them with ``output_size=1`` and reads that value
as a logit.

Only this docstring differs from the file that produced the published
BrainTreebank numbers; every class below is unchanged. At C=64 and F=2,
Table 1's five finger outputs give HiLoFuseNet 334,341 parameters and
CNN-LSTM 557,477, its 334 K and 557 K; the one-logit heads the BrainTreebank
runner builds have 333,825 and 556,961.

No released runner uses the remaining classes (the two HiLoFuseNet ablations,
LSTMSingleStream and DeepFingerNet). They are kept so the module stays the one
the experiments imported.
"""

import torch
import torch.nn as nn
import torch.nn.init as init

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class LSTM(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=1, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),  # e.g., 256 -> 128
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.lstm(x)

        x = x[:, -1, :]
        x = self.mlp(x)  # shape: (batch, seq_len, output_size)
        return x.squeeze(-1)


class CNN_LSTM(nn.Module):
    def __init__(self, input_size, output_size=1, dropout_prob=0.2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(input_size, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.lstm1 = nn.LSTM(128, 128, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        x = self.cnn(x)  # (batch, 128, T, F)
        x = torch.mean(x, dim=3)  # get (batch, 128, T)
        x = x.permute(0, 2, 1)  # (batch, T, 128)

        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = x[:, -1, :]
        x = self.mlp(x)
        return x.squeeze(-1)


class HiLoFuseNet(nn.Module):
    def __init__(self, C, F, lstm_hidden=256, output_size=1, D=16, dropout_prob=0.2):
        super().__init__()
        self.D = D
        self.spatialConv = nn.Sequential(
            Conv2dWithConstraint(F, self.D * F, (C, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=F),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(self.D * F, self.D * F, (1, 20),
                      padding=(0, 20 // 2), bias=False,
                      groups=self.D * F),
            nn.Conv2d(self.D * F, self.D * F, (1, 1), bias=False),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=10),
            nn.Dropout(dropout_prob),
        )

        self.lstm1 = nn.LSTM(self.D * F, lstm_hidden, batch_first=True, bidirectional=False)

        mlp_in = lstm_hidden  # lstm_hidden + D * F
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        in_cnn = x  # self.pool(x)
        in_cnn = in_cnn.permute(0, 3, 1, 2)
        out_cnn = self.spatialConv(in_cnn)

        in_lstm = out_cnn.squeeze(2)
        in_lstm = in_lstm.transpose(1, 2)
        x_lstm, _ = self.lstm1(in_lstm)
        out_lstm = x_lstm[:, -1, :]  # (B, lstm_hidden)

        out = self.mlp(out_lstm)  # (B, num_classes)

        return out.squeeze(-1)


# models modified for ablation study
class HiLoFuseNet_woDSConv(nn.Module):
    def __init__(self, input_size=100, hidden_size=256, output_size=1, num_layers=1, dropout_prob=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)

        mlp_in = hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.size(0), x.size(1), -1)
        out_lstm, _ = self.lstm(x)

        out_lstm = out_lstm[:, -1, :]
        out = self.mlp(out_lstm)  # (B, num_classes)

        return out.squeeze(-1)


class HiLoFuseNet_woLSTM(nn.Module):
    def __init__(self, C, F, output_size=1, D=16, dropout_prob=0.2):
        super().__init__()
        self.D = D
        self.spatialConv = nn.Sequential(
            Conv2dWithConstraint(F, self.D * F, (C, 1),
                                 padding=0, bias=False, max_norm=1,
                                 groups=F),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.Dropout(dropout_prob),

            nn.Conv2d(self.D * F, self.D * F, (1, 20),
                      padding=(0, 20 // 2), bias=False,
                      groups=self.D * F),
            nn.Conv2d(self.D * F, self.D * F, (1, 1), bias=False),
            nn.BatchNorm2d(self.D * F),
            nn.ELU(),
            nn.AvgPool2d((1, 10), stride=10),
            nn.Dropout(dropout_prob),
        )

        mlp_in = self.D * F * 20
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, mlp_in // 2),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(mlp_in // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, C, T, F)
        in_cnn = x  # self.pool(x)
        in_cnn = in_cnn.permute(0, 3, 1, 2)
        out_cnn = self.spatialConv(in_cnn)
        out_cnn = out_cnn.reshape(out_cnn.size(0), -1)

        out = self.mlp(out_cnn)  # (B, num_classes)

        return out.squeeze(-1)


# ─────────────────────────────────────────────────────
# LSTM_HGA / LSTM_LFS (Costello et al., adapted by Sun et al.)
# Single-stream LSTM for ECoG decoding
# Input: (B, C, T) — single stream (HGA or LFS)
# ─────────────────────────────────────────────────────

class LSTMSingleStream(nn.Module):
    """Single-stream LSTM for ECoG decoding.

    Following the LSTM_HGA protocol from Sun et al. (HiLoFuseNet paper):
    - Input HGA (or LFS) is decimated by a factor (default 10) via avg pooling
    - Fed to single-layer LSTM as (B, T_decimated, C)
    - Last hidden state → 2-layer MLP → output

    Input:  (B, C, T) — single frequency stream
    Output: (B, d_out)
    """
    def __init__(self, n_channels: int, d_out: int = 1,
                 hidden_size: int = 256, decimate: int = 10,
                 dropout: float = 0.2):
        super().__init__()
        self.decimate = decimate
        self.pool = nn.AvgPool1d(decimate, decimate) if decimate > 1 else nn.Identity()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=hidden_size,
                            num_layers=1, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, d_out),
        )

    def forward(self, x):
        # x: (B, C, T)
        x = self.pool(x)          # (B, C, T//decimate)
        x = x.transpose(1, 2)     # (B, T//decimate, C)
        _, (h_n, _) = self.lstm(x) # h_n: (1, B, hidden)
        h = h_n.squeeze(0)         # (B, hidden)
        return self.mlp(h)         # (B, d_out)


# ─────────────────────────────────────────────────────
# DeepFingerNet (Tao et al., IEEE TIM 2025)
# UNet++ with 1D convolutions for ECoG → finger/speech decoding
# Adapted from https://github.com/UM-Tao/DeepFingerNet
# ─────────────────────────────────────────────────────

class _ConvBlock(nn.Module):
    """Conv1d → LayerNorm → GELU → Dropout → MaxPool."""
    def __init__(self, in_ch, out_ch, kernel_size=7, stride=1, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', bias=False)
        self.norm = nn.LayerNorm(out_ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(stride, stride) if stride > 1 else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.drop(x)
        x = self.pool(x)
        return x


class _ConvBlockUp(nn.Module):
    """LayerNorm → GELU → Conv1d → Dropout → LayerNorm → GELU → Conv1d."""
    def __init__(self, in_ch, out_ch, kernel_size=7, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding='same', bias=False)
        self.norm2 = nn.LayerNorm(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding='same', bias=False)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm1(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.conv1(x)
        x = self.drop(x)
        x = self.norm2(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.act(x)
        x = self.conv2(x)
        return x


class _UpConvBlock(nn.Module):
    """ConvBlockUp + linear upsample."""
    def __init__(self, scale, in_ch, out_ch, kernel_size=7, dropout=0.1):
        super().__init__()
        self.conv = _ConvBlockUp(in_ch, out_ch, kernel_size, dropout)
        self.up = nn.Upsample(scale_factor=scale, mode='linear', align_corners=False)

    def forward(self, x):
        return self.up(self.conv(x))


class DeepFingerNet(nn.Module):
    """UNet++ for ECoG decoding (Tao et al., IEEE TIM 2025).

    Input:  (B, C*F, T) — flattened channels×frequencies, T time steps
    Output: (B, d_out)

    Args:
        n_input_features: C * F (channels × wavelet frequencies)
        d_out: output dimension (5 for fingers, 1 for speech)
    """
    def __init__(self, n_input_features: int, d_out: int = 5,
                 filters=(32, 64, 128, 256), kernel_size: int = 7,
                 dropout: float = 0.1):
        super().__init__()
        f0, f1, f2, f3 = filters

        # Spatial reduction: flatten C*F → f0
        self.spatial_reduce = _ConvBlock(n_input_features, f0, kernel_size=3, stride=1)

        # Encoder
        self.stage_1 = _ConvBlock(f0, f1, kernel_size, stride=2, dropout=dropout)
        self.stage_2 = _ConvBlock(f1, f2, kernel_size, stride=2, dropout=dropout)
        self.stage_3 = _ConvBlock(f2, f3, kernel_size, stride=2, dropout=dropout)

        # Decoder (UNet++ skip paths)
        self.up_2_1 = _UpConvBlock(2, f3, f2, kernel_size, dropout)
        self.up_1_1 = _UpConvBlock(2, f2, f1, kernel_size, dropout)
        self.up_1_2 = _UpConvBlock(2, f2, f1, kernel_size, dropout)
        self.up_0_1 = _UpConvBlock(2, f1, f0, kernel_size, dropout)
        self.up_0_2 = _UpConvBlock(2, f1, f0, kernel_size, dropout)
        self.up_0_3 = _UpConvBlock(2, f1, f0, kernel_size, dropout)

        # Final merge + output
        self.final_conv = _ConvBlockUp(f0 * 4, f0, kernel_size, dropout)
        self.output_proj = nn.Conv1d(f0, d_out, kernel_size=1, padding='same')

    def forward(self, x):
        # x: (B, C*F, T)
        if x.dim() == 4:
            B, C, F, T = x.shape
            x = x.reshape(B, C * F, T)

        # Encoder
        x_0_0 = self.spatial_reduce(x)
        x_1_0 = self.stage_1(x_0_0)
        x_2_0 = self.stage_2(x_1_0)
        x_3_0 = self.stage_3(x_2_0)

        # Decoder with skip connections
        x_0_1 = self.up_0_1(x_1_0)
        x_1_1 = self.up_1_1(x_2_0)
        x_2_1 = self.up_2_1(x_3_0)
        x_1_2 = self.up_1_2(x_2_1)
        x_0_2 = self.up_0_2(x_1_1)
        x_0_3 = self.up_0_3(x_1_2)

        # Merge all decoder outputs at scale 0
        T_min = min(x_0_0.shape[-1], x_0_1.shape[-1],
                    x_0_2.shape[-1], x_0_3.shape[-1])
        merged = torch.cat([
            x_0_0[..., :T_min], x_0_1[..., :T_min],
            x_0_2[..., :T_min], x_0_3[..., :T_min],
        ], dim=1)

        out = self.final_conv(merged)
        out = self.output_proj(out)  # (B, d_out, T)
        return out.mean(dim=-1)  # (B, d_out) — mean-pool over time
