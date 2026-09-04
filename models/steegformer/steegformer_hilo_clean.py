# models/steegformer/steegformer_hilo_clean.py
"""Clean Hi-Lo backbone with optional SPVAE latent router.

Merge strategies:
  - average:        fixed 0.5 lo + 0.5 hi  (clean baseline)
  - hi_lora:        hi tokens pass through early blocks with DualLoRA adapters,
                    then fixed 0.5 merge (hi gets real transformer processing)
  - learned_router: per-token gating network predicts lo/hi blend weight
  - cross_attn:     lo attends to hi via cross-attention + residual
  - spvae_router:   SPVAE precision-based per-channel gate (from private latent variances)
  - hi_lora_router: hi_lora early processing + learned_router merge (combined)
"""
from __future__ import annotations

import math
from functools import partial
from typing import Optional, Literal, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer

from .lora import set_block_lora_enabled


# ============================================================
# Channel Adaptation Methods
# ============================================================

CHANNEL_ADAPTERS = (
    "none", "original", "zero_mlp", "additive_mlp",
    "fourier_add", "direct_fourier", "subspace", "soft_lookup_add", "coord_pe",
    # Spatial KNN family — uses known EEG electrode positions
    "knn_hard", "knn_fourier", "knn_soft", "knn_soft_fourier",
    # Gaussian Process family — proper GP posterior over pretrained embeddings
    "gp_hard", "gp_fourier",
)


def _fourier_features(xyz: torch.Tensor, n_freq: int = 32) -> torch.Tensor:
    """Sinusoidal Fourier features of xyz coordinates.

    xyz: (..., 3)  →  (..., 6*n_freq)
    Uses log-spaced frequencies like NeRF positional encoding.
    """
    freq = torch.logspace(0, math.log10(50.0), n_freq, device=xyz.device, dtype=xyz.dtype)  # (F,)
    # xyz (..., 3) @ freq (F,) → (..., 3, F)
    angles = xyz.unsqueeze(-1) * freq  # (..., 3, F)
    angles = angles.flatten(-2)        # (..., 3F)
    return torch.cat([angles.sin(), angles.cos()], dim=-1)  # (..., 6F)


class ChannelAdapterBase(nn.Module):
    """Base class for channel adapters. All adapters:
    - Take ecog_xyz (B, C, 3) or (C, 3)
    - Return (B, C, D) embedding to ADD to enc_channel output
    """
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ZeroMLPAdapter(ChannelAdapterBase):
    """MLP(xyz→D), zero-init last layer. Used in 'replace' or 'additive' mode."""
    def __init__(self, D: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, D, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        return self.net(xyz)  # (B, C, D)


class FourierMLPAdapter(ChannelAdapterBase):
    """Fourier(xyz) → MLP → D, zero-init. Multi-scale spatial features."""
    def __init__(self, D: int, hidden: int = 128, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        self.net = nn.Sequential(
            nn.Linear(6 * n_freq, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, D, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        ff = _fourier_features(xyz, self.n_freq)  # (B, C, 6F)
        return self.net(ff)  # (B, C, D)


class SubspaceAdapter(ChannelAdapterBase):
    """Fourier(xyz) → MLP → k coefficients → V_k^T → D.

    Projects into the principal subspace of pretrained channel embeddings,
    guaranteeing outputs are in-distribution for frozen transformer blocks.
    """
    def __init__(self, eeg_emb: torch.Tensor, hidden: int = 128, n_freq: int = 32, k: int = 48):
        super().__init__()
        M, D = eeg_emb.shape
        k = min(k, M, D)
        self.n_freq = n_freq

        # Compute principal directions of pretrained embedding space
        with torch.no_grad():
            # Center the embeddings for better SVD
            mean = eeg_emb.mean(dim=0, keepdim=True)
            _, _, V = torch.linalg.svd(eeg_emb - mean, full_matrices=False)
            V_k = V[:k, :]  # (k, D) — top-k right singular vectors
        self.register_buffer("V_k", V_k)
        self.register_buffer("emb_mean", mean.squeeze(0))  # (D,)

        self.net = nn.Sequential(
            nn.Linear(6 * n_freq, hidden),
            nn.GELU(),
            nn.Linear(hidden, k, bias=False),
        )
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        ff = _fourier_features(xyz, self.n_freq)    # (B, C, 6F)
        coeff = self.net(ff)                         # (B, C, k)
        return torch.matmul(coeff, self.V_k)         # (B, C, D) — always in subspace


class SoftLookupAdditiveAdapter(ChannelAdapterBase):
    """softmax(MLP(xyz)/τ) @ E, with learnable scale starting at 0.

    Like the original fuser but:
    1. ADDITIVE (not replacing enc_channel)
    2. Proper softmax normalization
    3. Scale parameter starts at 0 → no-op at init
    """
    def __init__(self, eeg_emb: torch.Tensor, hidden: int = 128, tau: float = 1.0):
        super().__init__()
        M, D = eeg_emb.shape
        self.register_buffer("E", eeg_emb.to(torch.float32))
        self.tau = tau
        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, M),
        )
        self.scale = nn.Parameter(torch.zeros(1))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        logits = self.net(xyz)                       # (B, C, M)
        w = F.softmax(logits / self.tau, dim=-1)     # (B, C, M)
        fused = torch.matmul(w, self.E)              # (B, C, D)
        return self.scale * fused


class CoordPEAdapter(ChannelAdapterBase):
    """Fixed sinusoidal positional encoding of xyz — no learnable parameters.

    Encodes spatial position like a transformer PE but in 3D.
    """
    def __init__(self, D: int, n_freq: int = 32):
        super().__init__()
        self.n_freq = n_freq
        # Fixed projection from Fourier features to D
        # Use a deterministic projection (no learning)
        proj = torch.zeros(6 * n_freq, D)
        # Fill with scaled random orthogonal-ish directions (fixed at init)
        torch.manual_seed(42)
        nn.init.orthogonal_(proj[:D, :])  # first D rows orthogonal
        if 6 * n_freq > D:
            nn.init.orthogonal_(proj[D:min(2*D, 6*n_freq), :])
        proj *= 0.02  # small scale so it doesn't dominate
        self.register_buffer("proj", proj)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        ff = _fourier_features(xyz, self.n_freq)     # (B, C, 6F)
        return torch.matmul(ff, self.proj)            # (B, C, D)


# ============================================================
# EEG electrode positions (meters) for the 142-channel montage
# used in STEEGFormer pretraining.  Index = embedding table row.
# Generated from MNE standard_1005 montage.
# ============================================================
# fmt: off
_EEG_XYZ_142 = [
    [-0.036158,-0.009984, 0.089752],[ 0.000325,-0.081115, 0.082615],[ 0.067118,-0.010900, 0.063580],
    [ 0.067914, 0.049830, 0.016367],[ 0.083888, 0.001946, 0.008501],[ 0.000108,-0.114892, 0.014657],
    [-0.029437, 0.083917,-0.006990],[-0.074692, 0.004303, 0.045307],[ 0.078520,-0.060432, 0.012902],
    [ 0.071096,-0.062624, 0.047328],[ 0.037672,-0.009624, 0.088412],[ 0.051836, 0.054305, 0.040814],
    [ 0.015095,-0.118018,-0.006933],[ 0.035712, 0.077726, 0.021956],[ 0.000376, 0.027390, 0.088668],
    [ 0.079006,-0.028986, 0.049628],[ 0.085549,-0.045545,-0.007130],[ 0.043895,-0.109127,-0.013170],
    [-0.034062, 0.026011, 0.079987],[ 0.079534, 0.019936, 0.024438],[-0.080280,-0.013760, 0.029160],
    [ 0.073056,-0.073068,-0.002540],[ 0.081815, 0.015417,-0.011330],[ 0.067888,-0.075904, 0.028091],
    [-0.073009,-0.073766,-0.040998],[ 0.000312, 0.058512, 0.066462],[-0.023582, 0.069917, 0.047293],
    [ 0.078903,-0.060955,-0.023805],[ 0.025558, 0.070556, 0.047827],[ 0.073895,-0.074390,-0.041220],
    [ 0.019420,-0.065595, 0.092405],[-0.086076,-0.044990,-0.067986],[ 0.077002, 0.005336, 0.045350],
    [-0.082355, 0.000826, 0.008579],[ 0.034784, 0.026438, 0.078808],[ 0.026437,-0.092295, 0.063199],
    [-0.017195, 0.084849, 0.010027],[-0.054840, 0.068572,-0.010590],[ 0.054988,-0.098091,-0.035541],
    [ 0.055743, 0.069657,-0.010755],[-0.015820,-0.065600, 0.091164],[-0.072434,-0.073453,-0.002487],
    [-0.027496, 0.056931, 0.060342],[ 0.018923, 0.085597, 0.011443],[-0.054910,-0.098045,-0.035465],
    [-0.084076, 0.014567,-0.050429],[ 0.038384,-0.047073, 0.090695],[ 0.000004,-0.118565,-0.023078],
    [-0.018219, 0.009094, 0.092529],[-0.077215, 0.018643, 0.024460],[-0.072434,-0.073453,-0.002487],
    [-0.079592,-0.046551, 0.030949],[ 0.083322,-0.046101, 0.031206],[-0.065238, 0.036428, 0.036144],
    [ 0.029514, 0.057602, 0.059540],[ 0.085794,-0.045009,-0.068031],[-0.042862,-0.108073,-0.013151],
    [-0.050800, 0.064041, 0.023089],[ 0.036782,-0.100849, 0.036397],[-0.019862,-0.108942, 0.029760],
    [ 0.029872, 0.084896,-0.007080],[-0.084161,-0.016019,-0.009346],[ 0.066612,-0.046637, 0.065580],
    [ 0.000216,-0.102178, 0.050608],[-0.085565,-0.030629, 0.011153],[-0.084161,-0.016019,-0.009346],
    [ 0.085794,-0.025009,-0.068031],[ 0.055114,-0.028386, 0.080474],[ 0.085080,-0.015020,-0.009490],
    [ 0.065014,-0.087806,-0.018952],[-0.060182, 0.022716, 0.055544],[-0.050244, 0.053111, 0.042192],
    [-0.064466, 0.048035, 0.016921],[-0.086076,-0.024990,-0.067986],[-0.053007,-0.078788, 0.055940],
    [ 0.062293, 0.023723, 0.055630],[ 0.018787, 0.009248, 0.091562],[ 0.067128, 0.037800, 0.035296],
    [ 0.078053, 0.032982, 0.004483],[ 0.020220,-0.028148, 0.098172],[ 0.050674,-0.064482, 0.076130],
    [ 0.073056,-0.073068,-0.002540],[-0.084125,-0.001847,-0.029794],[ 0.054609,-0.089640, 0.037035],
    [-0.063556,-0.047009, 0.065624],[-0.035513,-0.047292, 0.091315],[-0.033701, 0.076837, 0.021227],
    [ 0.084113, 0.014365,-0.050538],[-0.014850,-0.117987,-0.006920],[-0.078160,-0.060757,-0.023824],
    [-0.067272,-0.076291, 0.028382],[ 0.029742,-0.114260,-0.029256],[-0.018354,-0.028322, 0.098220],
    [ 0.085080,-0.015020,-0.009490],[-0.052928,-0.028906, 0.080304],[-0.029413,-0.112449, 0.008839],
    [-0.048424,-0.099341, 0.021599],[-0.064597,-0.087656,-0.019014],[-0.054010,-0.089899, 0.037332],
    [-0.028620,-0.080525, 0.075436],[ 0.000231, 0.080771, 0.035417],[ 0.049820,-0.099446, 0.021727],
    [-0.036511,-0.100853, 0.037167],[ 0.029843,-0.112156, 0.008800],[-0.068115,-0.062975, 0.047252],
    [-0.015424, 0.043660, 0.077682],[ 0.051885, 0.007798, 0.073507],[-0.074500, 0.031300, 0.004846],
    [ 0.017592, 0.044054, 0.077788],[ 0.045853, 0.041623, 0.060647],[ 0.000401,-0.009167, 0.100244],
    [-0.084830,-0.046022,-0.007056],[ 0.000112, 0.088247,-0.001713],[ 0.084123,-0.001808,-0.029638],
    [-0.054840,-0.097528, 0.002792],[-0.046914,-0.064691, 0.075296],[ 0.055667,-0.078560, 0.056561],
    [ 0.031920,-0.080487, 0.076716],[ 0.073043, 0.044422,-0.012000],[ 0.000386,-0.047318, 0.099432],
    [-0.051051, 0.007177, 0.074377],[-0.044410, 0.040762, 0.061690],[-0.080775, 0.014120,-0.011135],
    [-0.029818,-0.114570,-0.029216],[ 0.086000,-0.029820, 0.011248],[ 0.052397, 0.065071, 0.022862],
    [-0.076407,-0.029731, 0.049217],[ 0.083456,-0.012776, 0.029208],[-0.024648,-0.092292, 0.062076],
    [ 0.055667,-0.097625, 0.002730],[-0.065358,-0.011632, 0.064358],[ 0.020294,-0.108914, 0.028944],
    [-0.076680,-0.060832, 0.012880],[-0.070263, 0.042474,-0.011420],[-0.085894,-0.015829,-0.048283],
    [-0.085619,-0.046515,-0.045707],[ 0.085560,-0.016361,-0.048271],[ 0.086162,-0.047035,-0.045869],
    [-0.013664,-0.109266, 0.032856],[ 0.013651,-0.109106, 0.030936],[-0.012047,-0.092607, 0.065508],
    [ 0.013923,-0.092694, 0.066958],
]
# fmt: on

_N_EEG = 142  # number of positioned EEG channels in the pretrained table


def _get_eeg_xyz_tensor(device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """Return (142, 3) float32 tensor of EEG electrode positions in meters."""
    return torch.tensor(_EEG_XYZ_142, dtype=torch.float32, device=device)


def _knn_weights(
    ecog_xyz: torch.Tensor,
    eeg_xyz: torch.Tensor,
    k: int = 8,
    sigma: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KNN inverse-distance weights from ECoG electrodes to EEG electrodes.

    Args:
        ecog_xyz: (C_ecog, 3) ECoG electrode positions (meters)
        eeg_xyz:  (M, 3) EEG electrode positions (meters)
        k: number of nearest neighbours
        sigma: Gaussian bandwidth; if None, uses median NN distance

    Returns:
        weights: (C_ecog, M) sparse-ish weight matrix (only k nonzero per row), sums to 1
        nn_idx:  (C_ecog, k) indices of nearest EEG channels
    """
    # (C, M) pairwise distances
    dist = torch.cdist(ecog_xyz.unsqueeze(0), eeg_xyz.unsqueeze(0)).squeeze(0)  # (C, M)
    topk_dist, topk_idx = dist.topk(k, dim=-1, largest=False)  # (C, k)

    if sigma is None:
        sigma = topk_dist.median().clamp(min=1e-6).item()

    # Gaussian kernel weights
    w = torch.exp(-0.5 * (topk_dist / sigma) ** 2)  # (C, k)
    w = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-8)  # normalize

    # Scatter into full (C, M) matrix
    C = ecog_xyz.shape[0]
    M = eeg_xyz.shape[0]
    weights = torch.zeros(C, M, device=ecog_xyz.device, dtype=ecog_xyz.dtype)
    weights.scatter_(1, topk_idx, w)

    return weights, topk_idx


class KNNHardAdapter(ChannelAdapterBase):
    """Pure spatial KNN interpolation from pretrained EEG embeddings — no learnable params.

    For each ECoG electrode, computes a Gaussian-weighted average of the K nearest
    EEG channel embeddings based on 3D distance. This gives a physically meaningful
    channel embedding that the frozen transformer blocks understand.

    Computes at runtime from xyz so it works with pooled training
    (different subjects have different electrode positions per batch).
    """
    def __init__(self, eeg_emb: torch.Tensor, ecog_xyz_m: torch.Tensor,
                 k: int = 8, sigma: Optional[float] = None):
        super().__init__()
        self.k = k
        # Store sigma from initial subject for consistent bandwidth
        eeg_xyz = _get_eeg_xyz_tensor(ecog_xyz_m.device)
        _, topk_idx = _knn_weights(ecog_xyz_m, eeg_xyz, k=k, sigma=sigma)
        # We don't precompute embeddings — just store EEG positions & embeddings
        self.register_buffer("eeg_xyz", eeg_xyz)
        self.register_buffer("eeg_emb", eeg_emb[:_N_EEG].clone().detach())

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        # xyz: (B, C, 3) in meters
        B, C, _ = xyz.shape
        # Compute per-sample KNN weights
        # Flatten batch for efficiency: (B*C, 3)
        flat_xyz = xyz.reshape(-1, 3)
        weights, _ = _knn_weights(flat_xyz, self.eeg_xyz, k=self.k)  # (B*C, M)
        emb = weights @ self.eeg_emb  # (B*C, D)
        return emb.reshape(B, C, -1)


class KNNFourierAdapter(ChannelAdapterBase):
    """KNN spatial interpolation + learnable Fourier residual.

    Base: Gaussian-KNN weighted average of pretrained EEG embeddings (computed at runtime).
    Residual: FourierMLPAdapter (zero-init) learns corrections.
    """
    def __init__(self, eeg_emb: torch.Tensor, ecog_xyz_m: torch.Tensor,
                 k: int = 8, sigma: Optional[float] = None,
                 hidden: int = 128, n_freq: int = 32):
        super().__init__()
        D = eeg_emb.shape[1]
        self.k = k
        self.register_buffer("eeg_xyz", _get_eeg_xyz_tensor(ecog_xyz_m.device))
        self.register_buffer("eeg_emb", eeg_emb[:_N_EEG].clone().detach())
        self.residual = FourierMLPAdapter(D, hidden=hidden, n_freq=n_freq)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        B, C, _ = xyz.shape
        flat_xyz = xyz.reshape(-1, 3)
        weights, _ = _knn_weights(flat_xyz, self.eeg_xyz, k=self.k)
        base = (weights @ self.eeg_emb).reshape(B, C, -1)
        return base + self.residual(xyz)


class KNNSoftAdapter(ChannelAdapterBase):
    """Learnable soft attention over EEG embeddings, initialized from KNN weights.

    Instead of fixed KNN weights, an MLP predicts attention logits over all M
    EEG embeddings. Initialized so that initial logits reproduce the KNN solution
    for the first _N_EEG slots (which have known 10-10 XYZ positions). Slots
    beyond _N_EEG (e.g. HBN-specific channels, 142-255) get neutral zero logits
    and are learned from scratch during training.
    """
    def __init__(self, eeg_emb: torch.Tensor, ecog_xyz_m: torch.Tensor,
                 k: int = 8, sigma: Optional[float] = None, hidden: int = 128,
                 use_full_table: bool = False):
        super().__init__()
        M, D = eeg_emb.shape[0], eeg_emb.shape[1]
        # If use_full_table=True, use the full embedding table (e.g. 256 HBN slots).
        # Otherwise, cap at _N_EEG=142 (the 10-10 positioned subset).
        M_pos = M if use_full_table else min(M, _N_EEG)
        self.register_buffer("E", eeg_emb[:M_pos].to(torch.float32))

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, M_pos),
        )
        self.scale = nn.Parameter(torch.ones(1))

        # Initialize: KNN prior on slots 0.._N_EEG-1, zeros on slots _N_EEG..M_pos-1
        with torch.no_grad():
            eeg_xyz = _get_eeg_xyz_tensor(ecog_xyz_m.device)
            weights, _ = _knn_weights(ecog_xyz_m, eeg_xyz, k=k, sigma=sigma)  # (C, 142)
            n_known = min(M_pos, _N_EEG)
            target_logits = torch.zeros(ecog_xyz_m.shape[0], M_pos, device=weights.device)
            target_logits[:, :n_known] = torch.log(weights[:, :n_known] + 1e-8)
            self.net[-1].bias.data.copy_(target_logits.mean(dim=0))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        logits = self.net(xyz)                        # (B, C, M_pos)
        w = F.softmax(logits, dim=-1)                 # (B, C, M_pos)
        return self.scale * torch.matmul(w, self.E)   # (B, C, D)


class KNNSoftFourierAdapter(ChannelAdapterBase):
    """Full: learnable soft attention (KNN-init) + Fourier residual."""
    def __init__(self, eeg_emb: torch.Tensor, ecog_xyz_m: torch.Tensor,
                 k: int = 8, sigma: Optional[float] = None,
                 hidden: int = 128, n_freq: int = 32,
                 use_full_table: bool = False):
        super().__init__()
        D = eeg_emb.shape[1]
        self.soft = KNNSoftAdapter(eeg_emb, ecog_xyz_m, k=k, sigma=sigma, hidden=hidden,
                                    use_full_table=use_full_table)
        self.residual = FourierMLPAdapter(D, hidden=hidden, n_freq=n_freq)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.soft(xyz) + self.residual(xyz)


# ============================================================
# Gaussian Process spatial adapters
# ============================================================

def _se_kernel(X1: torch.Tensor, X2: torch.Tensor, lengthscale: float) -> torch.Tensor:
    """Squared-exponential (RBF) kernel: k(x,x') = exp(-||x-x'||^2 / 2l^2)."""
    dist_sq = torch.cdist(X1.unsqueeze(0), X2.unsqueeze(0)).squeeze(0).pow(2)
    return torch.exp(-0.5 * dist_sq / (lengthscale ** 2))


class GPHardAdapter(ChannelAdapterBase):
    """GP posterior mean over pretrained EEG embeddings — no learnable params.

    Given EEG positions P with embeddings E, the GP posterior mean at ECoG
    position q is:
        μ(q) = k(q, P) K(P,P)^{-1} E

    where k is a squared-exponential kernel with auto-tuned lengthscale.
    Unlike KNN, the GP properly accounts for correlations between EEG channels
    (nearby EEG electrodes carry redundant information → avoids double-counting).

    Also computes posterior variance σ²(q) which is stored for optional use
    as per-channel confidence weights.
    """
    def __init__(self, eeg_emb: torch.Tensor, ecog_xyz_m: torch.Tensor,
                 lengthscale: Optional[float] = None, noise: float = 0.5):
        super().__init__()
        M = min(eeg_emb.shape[0], _N_EEG)
        eeg_xyz = _get_eeg_xyz_tensor(ecog_xyz_m.device)

        # Auto-tune lengthscale: 2x median nearest-neighbour distance.
        # Using median of ALL pairwise distances gives ~head diameter (13 cm),
        # making the kernel nearly flat and K(P,P) ill-conditioned (cond ~1M).
        # 2x median NN distance (~4.4 cm) gives a local kernel that respects
        # spatial structure while remaining smooth.
        if lengthscale is None:
            dists = torch.cdist(eeg_xyz.unsqueeze(0), eeg_xyz.unsqueeze(0)).squeeze(0)
            nn_dists = dists.topk(2, dim=-1, largest=False).values[:, 1]  # nearest neighbour
            lengthscale = (nn_dists.median() * 2.0).clamp(min=1e-6).item()

        # K(P,P) + noise*I — (M, M)
        # noise=0.5 gives smooth interpolation (eff_N~8, matching KNN).
        # noise=1e-4 (old default) forces near-exact interpolation through 142
        # dense points, causing wild oscillations (abs weight sum ~42, 65 negatives).
        K_PP = _se_kernel(eeg_xyz, eeg_xyz, lengthscale)
        K_PP += noise * torch.eye(M, device=K_PP.device, dtype=K_PP.dtype)

        # Precompute alpha = K(P,P)^{-1} @ E via Cholesky — (M, D)
        E = eeg_emb[:M].clone().detach().float()
        L = torch.linalg.cholesky(K_PP)
        alpha = torch.cholesky_solve(E, L)  # (M, D)

        # Precompute L_inv for variance: σ²(q) = k(q,q) - ||L^{-1} k(P,q)||²
        L_inv = torch.linalg.solve_triangular(L, torch.eye(M, device=L.device), upper=False)

        self.register_buffer("eeg_xyz", eeg_xyz)
        self.register_buffer("alpha", alpha)
        self.register_buffer("L_inv", L_inv)
        self.lengthscale = lengthscale

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        B, C, _ = xyz.shape
        flat_xyz = xyz.reshape(-1, 3)

        # k(q, P): (B*C, M)
        K_qP = _se_kernel(flat_xyz, self.eeg_xyz, self.lengthscale)

        # μ(q) = K(q,P) @ alpha: (B*C, D)
        emb = K_qP @ self.alpha
        return emb.reshape(B, C, -1)

    def posterior_variance(self, xyz: torch.Tensor) -> torch.Tensor:
        """Compute GP posterior variance at query positions.

        Returns (B, C) variance values — lower = more confident mapping.
        """
        if xyz.ndim == 2:
            xyz = xyz.unsqueeze(0)
        B, C, _ = xyz.shape
        flat_xyz = xyz.reshape(-1, 3)

        K_qP = _se_kernel(flat_xyz, self.eeg_xyz, self.lengthscale)
        # v = L^{-1} k(P, q)  →  (M, B*C)
        v = self.L_inv @ K_qP.T
        # σ²(q) = k(q,q) - ||v||² = 1.0 - ||v||²  (since k(q,q)=1 for SE kernel)
        var = (1.0 - (v * v).sum(dim=0)).clamp(min=0)
        return var.reshape(B, C)


class GPFourierAdapter(ChannelAdapterBase):
    """GP posterior mean + learnable Fourier residual.

    Combines the principled GP spatial mapping with a learnable correction
    that can capture patterns the GP kernel misses.
    """
    def __init__(self, eeg_emb: torch.Tensor, ecog_xyz_m: torch.Tensor,
                 lengthscale: Optional[float] = None, noise: float = 0.5,
                 hidden: int = 128, n_freq: int = 32):
        super().__init__()
        D = eeg_emb.shape[1]
        self.gp = GPHardAdapter(eeg_emb, ecog_xyz_m, lengthscale=lengthscale, noise=noise)
        self.residual = FourierMLPAdapter(D, hidden=hidden, n_freq=n_freq)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        return self.gp(xyz) + self.residual(xyz)


# ============================================================
# Building blocks (self-contained, no V3 dependency)
# ============================================================

class ECoGChannelAdapter(nn.Module):
    """Map ECoG electrode xyz → channel embeddings via a learned weighted sum
    over a fixed pretrained EEG embedding table E (M, D), plus a residual adapter.

    forward(pos_ecog_xyz):
      - pos_ecog_xyz: (C, 3) or (B, C, 3)
      - returns (fused, weights):  (B, C, D), (B, C, M)
    """
    def __init__(self, eeg_emb_fixed: torch.Tensor, hidden: int = 128):
        super().__init__()
        if eeg_emb_fixed.ndim != 2:
            raise ValueError(f"eeg_emb_fixed must be (M, D), got {tuple(eeg_emb_fixed.shape)}")
        M, D = eeg_emb_fixed.shape
        self.M, self.D = int(M), int(D)

        self.register_buffer("E", eeg_emb_fixed.to(torch.float32), persistent=True)

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
        )
        self.eeg_proj = nn.Linear(hidden, self.M, bias=False)
        self.residual_proj = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, D, bias=False),
        )
        nn.init.zeros_(self.residual_proj[-1].weight)
        self.layerNorm = nn.LayerNorm(D)

    def forward(self, pos_ecog_xyz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        squeeze_b = False
        if pos_ecog_xyz.ndim == 2:
            pos_ecog_xyz = pos_ecog_xyz.unsqueeze(0)
            squeeze_b = True

        latent = self.net(pos_ecog_xyz)         # (B, C, hidden)
        w = self.eeg_proj(latent)               # (B, C, M)
        fused = torch.matmul(w, self.E)         # (B, C, D)
        fused = self.layerNorm(fused + self.residual_proj(latent))

        if squeeze_b:
            fused = fused.squeeze(0)
            w = w.squeeze(0)
        return fused, w

class PatchEmbedEEG(nn.Module):
    """Tokenize low-freq EEG: unfold + Linear(patch_size, D)."""
    def __init__(self, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.unfold = nn.Unfold(kernel_size=(1, self.patch_size), stride=self.patch_size)
        self.proj = nn.Linear(self.patch_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] -> [B, Seq, C, D]
        b, c, t = x.shape
        x = x.unsqueeze(2)                       # [B, C, 1, T]
        u = self.unfold(x)                        # [B, C*patch, Seq]
        _, _, seq = u.shape
        u = u.view(b, c, self.patch_size, seq)
        tok = u.permute(0, 3, 1, 2).contiguous()  # [B, Seq, C, patch]
        return self.proj(tok)                      # [B, Seq, C, D]


class PatchEmbed1D(nn.Module):
    """Tokenize hi-freq signal: unfold + Linear(patch_size, D)."""
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)
        self.unfold = nn.Unfold(kernel_size=(1, self.patch_size), stride=self.patch_size)
        self.proj = nn.Linear(self.patch_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        x2 = x.unsqueeze(2)
        u = self.unfold(x2)
        _, _, seq = u.shape
        u = u.view(b, c, self.patch_size, seq)
        tok = u.permute(0, 3, 1, 2).contiguous()
        return self.proj(tok)


class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim: int, max_ch_idx: int = 145):
        super().__init__()
        self.emb = nn.Embedding(max_ch_idx, embedding_dim)
        nn.init.zeros_(self.emb.weight)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        return self.emb(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float()
        )
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe)

    def cls_token_pe(self) -> torch.Tensor:
        return self.pe[0, 0, :]

    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        b, n = seq_indices.shape
        return self.pe[0, seq_indices.reshape(-1)].view(b, n, -1)


# ============================================================
# SPVAE helpers — latent router from private variance
# ============================================================

def _gauss_kl_std_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q(z|x) || N(0,I)), averaged over batch."""
    return 0.5 * torch.mean(torch.sum(torch.exp(logvar) + mu**2 - 1.0 - logvar, dim=-1))


def _reparam(mu: torch.Tensor, logvar: torch.Tensor, training: bool) -> torch.Tensor:
    if not training:
        return mu
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)


def _poe_diag_gaussians(mu1, logvar1, mu2, logvar2, eps=1e-8):
    """Product-of-experts aggregation of two diagonal Gaussians."""
    var1 = torch.exp(logvar1).clamp_min(eps)
    var2 = torch.exp(logvar2).clamp_min(eps)
    prec = (1.0 / var1 + 1.0 / var2).clamp_min(eps)
    var = 1.0 / prec
    mu = var * (mu1 / var1 + mu2 / var2)
    return mu, torch.log(var.clamp_min(eps))


class MLPDiagGaussian(nn.Module):
    """MLP encoder → (mu, logvar) for diagonal Gaussian."""
    def __init__(self, D_in: int, D_h: int, D_z: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_in, D_h),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
        )
        self.mu = nn.Linear(D_h, D_z)
        self.logvar = nn.Linear(D_h, D_z)
        nn.init.zeros_(self.logvar.weight)
        nn.init.zeros_(self.logvar.bias)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h)


# ============================================================
# Merge modules
# ============================================================

class LearnedRouter(nn.Module):
    """Per-token gating: examines both lo and hi to predict blend weight.

    gate = sigmoid(MLP(cat(lo, hi)))   ∈ (0,1) per token
    merged = (1 - gate) * lo + gate * hi

    Uses a narrow bottleneck (default 16) to keep params minimal (~4K for D=512).
    Zero-init last layer → gate starts at 0.5 (matches average baseline).
    """
    def __init__(self, embed_dim: int, bottleneck: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, 1),
        )
        # Zero-init last linear so gate ≈ 0.5 at start
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, lo: torch.Tensor, hi: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (merged, gate) where gate is [B, N, 1]."""
        gate = torch.sigmoid(self.net(torch.cat([lo, hi], dim=-1)))
        merged = (1 - gate) * lo + gate * hi
        return merged, gate


class LayerwiseHiLoGate(nn.Module):
    """Per-layer gated residual fusion of hi tokens into the lo stream.

    g = act(MLP(LN[ mean_N(lo0), mean_N(hi0) ]))   # [B, depth], one scalar/block
    at block l:   tok <- block_l(tok);   tok[:, patches] += g_l * hi

    Zero-init last linear: with gate_act="tanh", g==0 at start so the model
    begins exactly as the lo-only baseline then learns where to open hi
    (Flamingo-style); tanh also lets it suppress hi (negative g). GMU-style
    input-dependent gate, generalised from LearnedRouter to one scalar/layer.
    Params (D=512, bottleneck=16, depth=12): ~16.6K, no extra block params.
    """

    def __init__(
        self,
        embed_dim: int,
        depth: int,
        bottleneck: int = 16,
        gate_act: str = "tanh",
    ):
        super().__init__()
        if gate_act not in ("tanh", "sigmoid", "none"):
            raise ValueError(f"gate_act must be tanh|sigmoid|none, got {gate_act!r}")
        self.gate_act = gate_act
        self.net = nn.Sequential(
            nn.LayerNorm(2 * embed_dim),
            nn.Linear(2 * embed_dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, int(depth)),
        )
        # Zero-init last linear -> raw == 0 at start.
        #   tanh(0) = 0     -> exactly the lo-only baseline (Flamingo-style)
        #   sigmoid(0)=0.5  -> matches the 0.5 'average' baseline magnitude
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def compute_gates(self, lo: torch.Tensor, hi: torch.Tensor) -> torch.Tensor:
        """lo, hi: [B, N, D]  ->  g: [B, depth]."""
        s = torch.cat([lo.mean(dim=1), hi.mean(dim=1)], dim=-1)  # [B, 2D]
        raw = self.net(s)                                        # [B, depth]
        if self.gate_act == "tanh":
            return torch.tanh(raw)
        if self.gate_act == "sigmoid":
            return torch.sigmoid(raw)
        return raw


# ============================================================
# TokenRegressor (copied from V3 for self-containment)
# ============================================================

TokenMode = Literal["flatten", "mean", "cls"]


class TokenRegressor(nn.Module):
    def __init__(
        self,
        d_out: int,
        token_mode: TokenMode = "flatten",
        include_cls: bool = True,
        dropout: float = 0.0,
        head_hidden: int = 0,
    ):
        super().__init__()
        self.d_out = int(d_out)
        self.token_mode = token_mode
        self.include_cls = bool(include_cls)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.head_hidden = int(head_hidden)
        self.head: Optional[nn.Module] = None

    def _build_head(self, dim: int, device: torch.device) -> nn.Module:
        if self.head_hidden > 0:
            return nn.Sequential(
                nn.Linear(dim, self.head_hidden),
                nn.GELU(),
                nn.Dropout(self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0),
                nn.Linear(self.head_hidden, self.d_out),
            ).to(device)
        return nn.Linear(dim, self.d_out).to(device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.include_cls:
            feat = tokens
        else:
            feat = tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

        if self.head is None:
            if self.token_mode in ("cls", "mean"):
                dim = feat.shape[2]
            else:
                dim = feat.shape[1] * feat.shape[2]
            self.head = self._build_head(dim, tokens.device)

        if self.token_mode == "cls":
            h = feat[:, 0, :]
        elif self.token_mode == "mean":
            h = feat.mean(dim=1)
        else:  # flatten
            h = feat.flatten(1)
        return self.head(self.dropout(h))


# ============================================================
# FC Graph Adapter (GAT with functional connectivity adjacency)
# ============================================================

class FCGraphAdapter(nn.Module):
    """Graph Attention Network adapter using functional connectivity adjacency.

    Inserted after channel embedding: h' = h + alpha * GAT(h, A_s)
    where alpha is a learnable scalar initialized to 0 (zero-init residual).
    """

    def __init__(self, embed_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)

        # Zero-init residual gate
        self.alpha = nn.Parameter(torch.zeros(1))

        # Zero-init output projection for stable start
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, D) channel embeddings
            adj: (C, C) adjacency matrix (sparse OK, will be used as attention mask)
        Returns:
            (B, C, D) updated embeddings with graph-aware residual
        """
        B, C, D = x.shape
        H = self.num_heads

        q = self.q_proj(x).reshape(B, C, H, self.head_dim).permute(0, 2, 1, 3)  # (B,H,C,hd)
        k = self.k_proj(x).reshape(B, C, H, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, C, H, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B,H,C,C)

        # Mask attention by adjacency: -inf where adj == 0
        mask = (adj == 0).unsqueeze(0).unsqueeze(0)  # (1,1,C,C)
        attn = attn.masked_fill(mask, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # handle rows with all -inf
        attn = self.attn_drop(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, C, D)  # (B,C,D)
        out = self.out_proj(out)

        return x + self.alpha * out


# ============================================================
# Clean Hi-Lo Backbone
# ============================================================

MERGE_STRATEGIES = ("average", "hi_lora", "learned_router", "cross_attn", "spvae_router", "hi_lora_router", "layerwise_gate")


class HiLoCleanBackbone(timm.models.vision_transformer.VisionTransformer):
    """ViT backbone with clean hi-freq merge — no SPVAE/router/CID."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        max_len: int = 512,
        chan_idx: Optional[list[int]] = None,
        expect_num_chans: int = 62,
        max_ch_idx: int = 145,
        # Hi-Lo merge config
        merge_strategy: str = "average",
        hi_inject_last_n: int = 4,
        hi_patch_size: int = 25,
        # layerwise_gate config
        layerwise_gate_bottleneck: int = 16,
        layerwise_gate_act: str = "tanh",
        layerwise_gate_share_blocks: bool = False,
    ):
        super().__init__(
            img_size=224, patch_size=patch_size, in_chans=3, num_classes=0,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )
        # Remove timm's pos_embed (we use our own)
        if hasattr(self, "pos_embed"):
            delattr(self, "pos_embed")

        # Patch embeddings
        self.patch_embed = PatchEmbedEEG(patch_size=patch_size, embed_dim=embed_dim)
        self.patch_embed_hi = PatchEmbed1D(patch_size=hi_patch_size, embed_dim=embed_dim)

        # Positional encodings
        self.enc_channel = ChannelPositionalEmbed(embed_dim, max_ch_idx=max_ch_idx)
        self.enc_time = TemporalPositionalEncoding(embed_dim, max_len=max_len)

        self.expect_num_chans = int(expect_num_chans)
        if chan_idx is None:
            chan_idx = list(range(self.expect_num_chans))
        self.register_buffer(
            "default_chan_idx", torch.tensor(chan_idx, dtype=torch.long), persistent=False
        )

        # Merge config
        if merge_strategy not in MERGE_STRATEGIES:
            raise ValueError(
                f"Unknown merge_strategy={merge_strategy!r}. "
                f"Choose from {MERGE_STRATEGIES}"
            )
        self.merge_strategy = merge_strategy
        self.merge_block_idx = max(0, depth - int(hi_inject_last_n))

        # Strategy-specific modules (only create what's needed)
        # hi_lora: DualLoRA is injected into early blocks externally (by the runner);
        #          no extra modules needed here, just the flag.
        self.learned_router: Optional[LearnedRouter] = None
        self.cross_attn_layer: Optional[nn.MultiheadAttention] = None
        self.layerwise_gate: Optional[LayerwiseHiLoGate] = None
        self.layerwise_gate_share_blocks: bool = bool(layerwise_gate_share_blocks)
        if merge_strategy in ("learned_router", "hi_lora_router"):
            self.learned_router = LearnedRouter(embed_dim)
        if merge_strategy == "layerwise_gate":
            self.layerwise_gate = LayerwiseHiLoGate(
                embed_dim,
                depth=depth,
                bottleneck=int(layerwise_gate_bottleneck),
                gate_act=str(layerwise_gate_act),
            )
        if merge_strategy == "cross_attn":
            self.cross_attn_layer = nn.MultiheadAttention(
                embed_dim, num_heads=min(4, num_heads), batch_first=False,
            )
            # Zero-init cross-attn output projection for stable start
            nn.init.zeros_(self.cross_attn_layer.out_proj.weight)
            nn.init.zeros_(self.cross_attn_layer.out_proj.bias)

        # ECoG channel adapter (attached after pretrained load)
        self.ecog_fuser: Optional[ECoGChannelAdapter] = None
        # New modular adapter system (additive to enc_channel by default)
        self.channel_adapter: Optional[ChannelAdapterBase] = None
        self._channel_adapter_mode: str = "additive"  # "additive" or "replace"

        # SPVAE latent router (attached via attach_spvae)
        self.use_spvae = False
        self.spvae_beta_s: float = 0.0
        self.spvae_beta_p: float = 0.0
        self.spvae_lambda_agree: float = 0.0
        self.enc_s_lo: Optional[MLPDiagGaussian] = None
        self.enc_s_hi: Optional[MLPDiagGaussian] = None
        self.enc_p_lo: Optional[MLPDiagGaussian] = None
        self.enc_p_hi: Optional[MLPDiagGaussian] = None
        self._spvae_alpha_per_channel: Optional[torch.Tensor] = None
        self.last_aux: Dict[str, torch.Tensor] = {}

        # For logging: store last router gate mean
        self._last_gate_mean: Optional[float] = None

        # --- Subject prompt tokens (attached via attach_subject_prompts) ---
        self.subject_prompts: Optional[nn.Embedding] = None
        self.num_prompt_tokens: int = 0
        self._prompt_dim: int = embed_dim

        # --- FC Graph Adapter (attached via attach_fc_graph_adapter) ---
        self.fc_graph_adapter: Optional[FCGraphAdapter] = None
        self._fc_adj: Optional[Dict[int, torch.Tensor]] = None  # per-subject adjacency

    def attach_ecog_fuser_from_channel_embed(self, M: Optional[int] = None, hidden: int = 128):
        """Create and attach an ECoGChannelAdapter using the pretrained EEG embedding table."""
        dev = self.enc_channel.emb.weight.device
        with torch.no_grad():
            E = self.enc_channel.emb.weight.detach().clone()
            if M is not None:
                E = E[:int(M)]
            E = E.to(device=dev, dtype=torch.float32)
        self.ecog_fuser = ECoGChannelAdapter(E, hidden=hidden).to(dev)

    def attach_spvae(
        self, *,
        z_shared_dim: int = 128,
        z_private_dim: int = 128,
        hidden: int = 256,
        beta_s: float = 1e-4,
        beta_p: float = 1e-3,
        lambda_agree: float = 1e-3,
    ):
        """Attach SPVAE encoders for precision-based latent routing.

        Four VAE encoders map time-pooled lo/hi patch embeddings to
        shared + private latent spaces.  The ratio of private-latent
        precisions gives a per-channel merge gate alpha:

            prec_hi = exp(-logvar_p_hi)
            alpha   = prec_hi / (prec_lo + prec_hi)
            merged  = (1-alpha)*lo + alpha*hi

        Losses (added to self.last_aux):
            loss_spvae = beta_s * KL_shared + beta_p * KL_private
                       + lambda_agree * ||mu_s_lo - mu_s_hi||^2
        """
        D = self.enc_channel.emb.weight.shape[1]
        zs, zp = int(z_shared_dim), int(z_private_dim)
        self.use_spvae = True
        self.spvae_beta_s = float(beta_s)
        self.spvae_beta_p = float(beta_p)
        self.spvae_lambda_agree = float(lambda_agree)
        self.enc_s_lo = MLPDiagGaussian(D, hidden, zs)
        self.enc_s_hi = MLPDiagGaussian(D, hidden, zs)
        self.enc_p_lo = MLPDiagGaussian(D, hidden, zp)
        self.enc_p_hi = MLPDiagGaussian(D, hidden, zp)

    def attach_subject_prompts(self, num_subjects: int, num_tokens: int = 4, init_std: float = 0.02):
        """Attach per-subject learnable prompt tokens.

        Tokens are prepended after CLS in forward_tokens.
        """
        D = self.enc_channel.emb.weight.shape[1]
        self.num_prompt_tokens = num_tokens
        # Embedding: num_subjects x (num_tokens * D), reshaped to (K, D) per subject
        self.subject_prompts = nn.Embedding(num_subjects, num_tokens * D)
        nn.init.normal_(self.subject_prompts.weight, std=init_std)
        self._prompt_dim = D
        print(f"  Subject prompts attached: {num_subjects} subjects x {num_tokens} tokens "
              f"({self.subject_prompts.weight.numel()} params)", flush=True)

    def attach_fc_graph_adapter(self, num_heads: int = 4, dropout: float = 0.1):
        """Attach FC Graph Adapter (GAT with functional connectivity adjacency).

        Adjacency matrices are set per-subject via set_fc_adjacency().
        """
        D = self.enc_channel.emb.weight.shape[1]
        self.fc_graph_adapter = FCGraphAdapter(D, num_heads=num_heads, dropout=dropout)
        self._fc_adj = {}
        n_params = sum(p.numel() for p in self.fc_graph_adapter.parameters())
        print(f"  FC Graph Adapter attached: {num_heads} heads, {n_params} params", flush=True)

    def set_fc_adjacency(self, sid: int, adj: torch.Tensor):
        """Register a pre-computed adjacency matrix for a subject."""
        if self._fc_adj is None:
            self._fc_adj = {}
        dev = self.enc_channel.emb.weight.device
        self._fc_adj[sid] = adj.to(dev)

    def attach_channel_adapter(
        self, adapter_type: str, *,
        M: Optional[int] = None,
        hidden: int = 128,
        ecog_xyz_m: Optional[torch.Tensor] = None,
        knn_k: int = 8,
        knn_sigma: Optional[float] = None,
        use_full_table: bool = False,
    ):
        """Attach a channel adapter from the CHANNEL_ADAPTERS registry.

        Args:
            adapter_type: one of CHANNEL_ADAPTERS
            M: number of EEG positions to use from embedding table
            hidden: hidden dim for MLPs
            ecog_xyz_m: (C, 3) ECoG electrode positions in meters.
                        Required for knn_* adapters.
            knn_k: number of nearest EEG neighbours for KNN adapters
            knn_sigma: Gaussian bandwidth for KNN; None = auto (median NN dist)
        """
        if adapter_type not in CHANNEL_ADAPTERS:
            raise ValueError(f"Unknown adapter: {adapter_type!r}. Choose from {CHANNEL_ADAPTERS}")

        dev = self.enc_channel.emb.weight.device
        D = self.enc_channel.emb.weight.shape[1]

        if adapter_type == "none":
            self.channel_adapter = None
            return

        # Get EEG embedding table (used by some adapters)
        with torch.no_grad():
            E = self.enc_channel.emb.weight.detach().clone()
            if M is not None:
                E = E[:int(M)]
            E = E.to(device=dev, dtype=torch.float32)

        if adapter_type == "original":
            # Use the old ECoGChannelAdapter via ecog_fuser path (replaces enc_channel)
            self.ecog_fuser = ECoGChannelAdapter(E, hidden=hidden).to(dev)
            return

        # KNN and GP families require ecog_xyz_m
        _spatial_types = ("knn_hard", "knn_fourier", "knn_soft", "knn_soft_fourier",
                          "gp_hard", "gp_fourier")
        if adapter_type in _spatial_types and ecog_xyz_m is None:
            raise ValueError(f"{adapter_type} requires ecog_xyz_m (ECoG electrode positions in meters)")
        if ecog_xyz_m is not None:
            ecog_xyz_m = ecog_xyz_m.to(device=dev, dtype=torch.float32)

        if adapter_type == "zero_mlp":
            self.channel_adapter = ZeroMLPAdapter(D, hidden=hidden).to(dev)
            self._channel_adapter_mode = "replace"
        elif adapter_type == "additive_mlp":
            self.channel_adapter = ZeroMLPAdapter(D, hidden=hidden).to(dev)
            self._channel_adapter_mode = "additive"
        elif adapter_type == "fourier_add":
            self.channel_adapter = FourierMLPAdapter(D, hidden=hidden).to(dev)
            self._channel_adapter_mode = "additive"
        elif adapter_type == "direct_fourier":
            self.channel_adapter = FourierMLPAdapter(D, hidden=hidden).to(dev)
            self._channel_adapter_mode = "replace"
        elif adapter_type == "subspace":
            self.channel_adapter = SubspaceAdapter(E, hidden=hidden).to(dev)
            self._channel_adapter_mode = "additive"
        elif adapter_type == "soft_lookup_add":
            self.channel_adapter = SoftLookupAdditiveAdapter(E, hidden=hidden).to(dev)
            self._channel_adapter_mode = "additive"
        elif adapter_type == "coord_pe":
            self.channel_adapter = CoordPEAdapter(D).to(dev)
            self._channel_adapter_mode = "additive"
        # --- KNN family: REPLACE enc_channel entirely with spatial interpolation ---
        elif adapter_type == "knn_hard":
            self.channel_adapter = KNNHardAdapter(E, ecog_xyz_m, k=knn_k, sigma=knn_sigma).to(dev)
            self._channel_adapter_mode = "replace"
        elif adapter_type == "knn_fourier":
            self.channel_adapter = KNNFourierAdapter(E, ecog_xyz_m, k=knn_k, sigma=knn_sigma, hidden=hidden).to(dev)
            self._channel_adapter_mode = "replace"
        elif adapter_type == "knn_soft":
            self.channel_adapter = KNNSoftAdapter(E, ecog_xyz_m, k=knn_k, sigma=knn_sigma, hidden=hidden).to(dev)
            self._channel_adapter_mode = "replace"
        elif adapter_type == "knn_soft_fourier":
            self.channel_adapter = KNNSoftFourierAdapter(E, ecog_xyz_m, k=knn_k, sigma=knn_sigma, hidden=hidden,
                                                          use_full_table=use_full_table).to(dev)
            self._channel_adapter_mode = "replace"
        # --- GP family: REPLACE enc_channel with GP posterior over pretrained embeddings ---
        elif adapter_type == "gp_hard":
            self.channel_adapter = GPHardAdapter(E, ecog_xyz_m).to(dev)
            self._channel_adapter_mode = "replace"
        elif adapter_type == "gp_fourier":
            self.channel_adapter = GPFourierAdapter(E, ecog_xyz_m, hidden=hidden).to(dev)
            self._channel_adapter_mode = "replace"

    # -------------------------
    # Forward
    # -------------------------
    def forward_tokens(
        self,
        x: torch.Tensor,
        *,
        x_hi: Optional[torch.Tensor] = None,
        ecog_xyz: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
        b, c, _ = x.shape

        # 1. Tokenize lo
        tok4 = self.patch_embed(x)               # [B, Seq, C, D]
        b, seq, c_out, d = tok4.shape
        n = seq * c_out
        tok_lo = tok4.reshape(b, n, d)            # [B, N, D]

        # 2. Tokenize hi
        tok_hi = None
        tok_hi4 = None
        if x_hi is not None:
            tok_hi4 = self.patch_embed_hi(x_hi)   # [B, Seq_hi, C, D]
            seq_hi = tok_hi4.shape[1]
            if seq_hi != seq:
                # Pool to match lo sequence length
                tok_hi_ = tok_hi4.permute(0, 2, 3, 1).contiguous().view(b * c_out, d, seq_hi)
                tok_hi_ = F.adaptive_avg_pool1d(tok_hi_, output_size=seq)
                tok_hi4 = tok_hi_.view(b, c_out, d, seq).permute(0, 3, 1, 2).contiguous()
            tok_hi = tok_hi4.reshape(b, n, d)      # [B, N, D]

        # 3. Positional embeddings (time + channel) to both
        if c_out == self.expect_num_chans:
            chan_idx = self.default_chan_idx.to(x.device)
        else:
            chan_idx = torch.arange(c_out, device=x.device, dtype=torch.long)

        chan_idx_bn = chan_idx.unsqueeze(0).unsqueeze(1).repeat(b, seq, 1).reshape(b, n)
        t_idx = (
            torch.arange(1, seq + 1, device=x.device)
            .unsqueeze(0).unsqueeze(-1)
            .repeat(b, 1, c_out)
            .reshape(b, n)
        )

        # Channel embeddings
        if self.ecog_fuser is not None:
            # Legacy path: original fuser REPLACES enc_channel entirely
            if ecog_xyz is None:
                raise ValueError("ecog_fuser is active but no ecog_xyz provided.")
            fused, _ = self.ecog_fuser(ecog_xyz)
            if fused.ndim == 2:
                ch_emb = fused.unsqueeze(0).unsqueeze(1).expand(b, seq, -1, -1).reshape(b, n, d)
            else:
                ch_emb = fused.unsqueeze(1).expand(-1, seq, -1, -1).reshape(b, n, d)
        elif self.channel_adapter is not None:
            if ecog_xyz is None:
                raise ValueError("channel_adapter is active but no ecog_xyz provided.")
            adapter_out = self.channel_adapter(ecog_xyz)  # (B, C, D)
            if adapter_out.ndim == 2:
                adapter_out = adapter_out.unsqueeze(0)
            # Expand (B, C, D) → (B, N, D) by repeating over time
            adapter_emb = adapter_out.unsqueeze(1).expand(-1, seq, -1, -1).reshape(b, n, d)
            if self._channel_adapter_mode == "replace":
                ch_emb = adapter_emb
            else:  # additive
                ch_emb = self.enc_channel(chan_idx_bn) + adapter_emb
        else:
            ch_emb = self.enc_channel(chan_idx_bn)
        t_emb = self.enc_time(t_idx)

        tok_lo = tok_lo + t_emb + ch_emb
        if tok_hi is not None:
            tok_hi = tok_hi + t_emb + ch_emb

        # 3a. FC Graph Adapter: apply GAT over channel dimension
        if self.fc_graph_adapter is not None and sid is not None and self._fc_adj:
            assert (sid == sid[0]).all(), "FC graph adapter requires homogeneous batches (single subject per batch)"
            tok_lo_4d = tok_lo.reshape(b, seq, c_out, d)
            ch_repr = tok_lo_4d.mean(dim=1)  # (B, C, D)
            first_sid = sid[0].item()
            if first_sid in self._fc_adj:
                adj = self._fc_adj[first_sid]  # already on model device
                if adj.shape[0] == c_out:
                    ch_updated = self.fc_graph_adapter(ch_repr, adj)  # (B, C, D)
                    delta = (ch_updated - ch_repr).unsqueeze(1).expand(-1, seq, -1, -1)
                    tok_lo = tok_lo + delta.reshape(b, n, d)
                    if tok_hi is not None:
                        tok_hi = tok_hi + delta.reshape(b, n, d)

        # 3b. SPVAE latent router: compute per-channel precision gate
        self.last_aux = {}
        self._spvae_alpha_per_channel = None
        if self.use_spvae and tok_hi is not None:
            # Pool across time per channel: (B, Seq, C, D) → (B*C, D)
            g_lo = tok4.mean(dim=1).reshape(b * c_out, d)
            g_hi = tok_hi4.mean(dim=1).reshape(b * c_out, d)

            # Encode shared + private latents
            mu_s_lo, lv_s_lo = self.enc_s_lo(g_lo)
            mu_s_hi, lv_s_hi = self.enc_s_hi(g_hi)
            mu_p_lo, lv_p_lo = self.enc_p_lo(g_lo)
            mu_p_hi, lv_p_hi = self.enc_p_hi(g_hi)

            # PoE aggregation of shared latents
            mu_s, lv_s = _poe_diag_gaussians(mu_s_lo, lv_s_lo, mu_s_hi, lv_s_hi)

            # KL losses
            kl_shared = (_gauss_kl_std_normal(mu_s_lo, lv_s_lo)
                         + _gauss_kl_std_normal(mu_s_hi, lv_s_hi)
                         + _gauss_kl_std_normal(mu_s, lv_s))
            kl_private = (_gauss_kl_std_normal(mu_p_lo, lv_p_lo)
                          + _gauss_kl_std_normal(mu_p_hi, lv_p_hi))
            loss_agree_raw = (mu_s_lo - mu_s_hi).pow(2).mean()

            loss_spvae = (self.spvae_beta_s * kl_shared
                          + self.spvae_beta_p * kl_private
                          + self.spvae_lambda_agree * loss_agree_raw)
            self.last_aux["loss_spvae"] = loss_spvae
            self.last_aux["agree_raw"] = loss_agree_raw.detach()

            # Precision gate from private latent variances: (B, C)
            zp = lv_p_lo.shape[-1]
            prec_lo = torch.exp(-lv_p_lo.view(b, c_out, zp)).mean(dim=-1)
            prec_hi = torch.exp(-lv_p_hi.view(b, c_out, zp)).mean(dim=-1)
            alpha_BC = prec_hi / (prec_lo + prec_hi + 1e-8)  # (B, C)
            self._spvae_alpha_per_channel = alpha_BC.detach()
            self.last_aux["router_alpha"] = alpha_BC.mean().detach()

        # 4. CLS + lo assembly
        cls = (self.cls_token + self.enc_time.cls_token_pe().to(x.device)).expand(b, -1, -1)
        tok = torch.cat([cls, tok_lo], dim=1)     # [B, 1+N, D]
        tok = self.pos_drop(tok)

        # 4a. Subject prompt tokens: prepend after CLS
        _n_prompts = 0
        if self.subject_prompts is not None and sid is not None:
            K = self.num_prompt_tokens
            prompt_flat = self.subject_prompts(sid)  # (B, K*D)
            prompt_tokens = prompt_flat.reshape(b, K, self._prompt_dim)  # (B, K, D)
            tok = torch.cat([tok[:, :1, :], prompt_tokens, tok[:, 1:, :]], dim=1)
            _n_prompts = K
        _prefix_len = 1 + _n_prompts  # CLS + prompts (used by merge step)

        # layerwise_gate: per-layer gated residual injection of hi. One tiny
        # gate net (sees pooled initial lo+hi) emits `depth` scalars; at every
        # block the lo stream gets  + g_l * hi.
        if self.merge_strategy == "layerwise_gate":
            if tok_hi is None:
                for blk in self.blocks:
                    tok = blk(tok)
                return self.norm(tok)
            H = tok_hi                                   # [B, N, D] hi tokens
            g = self.layerwise_gate.compute_gates(tok_lo, H)  # [B, depth]
            self._last_gate_mean = g.mean().item()
            self.last_aux["layerwise_gate_per_layer"] = g.mean(dim=0).detach()
            share = self.layerwise_gate_share_blocks
            # hi only ever lands in the body region (after CLS+prompts), so
            # zero-pad the prefix rows once -> each block is a single
            # out-of-place add instead of slice+slice+cat (autograd-safe).
            H_pad = F.pad(H, (0, 0, _prefix_len, 0))     # [B, _prefix_len+N, D]
            for li, blk in enumerate(self.blocks):
                tok = blk(tok)
                if share:
                    H = blk(H)
                    H_pad = F.pad(H, (0, 0, _prefix_len, 0))
                tok = tok + g[:, li].view(b, 1, 1) * H_pad
            return self.norm(tok)

        merge_k = self.merge_block_idx

        # 5. Early blocks
        #    For hi_lora: early blocks have SwitchableLoRA injected.
        #    Lo pass: LoRA disabled (pure frozen weights).
        #    Hi pass: LoRA enabled (frozen + adapter).
        if self.merge_strategy in ("hi_lora", "hi_lora_router") and tok_hi is not None:
            tok_hi_ctx = torch.cat([cls, tok_hi], dim=1)
            for blk in self.blocks[:merge_k]:
                set_block_lora_enabled(blk, False)
                tok = blk(tok)
                set_block_lora_enabled(blk, True)
                tok_hi_ctx = blk(tok_hi_ctx)
            hi_processed = tok_hi_ctx[:, 1:, :]    # drop CLS
        else:
            for blk in self.blocks[:merge_k]:
                tok = blk(tok)
            hi_processed = tok_hi  # raw embeddings (may be None)

        # 6. Merge hi into lo
        if hi_processed is not None:
            lo_part = tok[:, _prefix_len:, :]      # [B, N, D] (skip CLS + prompts)
            hi_part = hi_processed                 # [B, N, D]

            if self.merge_strategy == "spvae_router" and self._spvae_alpha_per_channel is not None:
                # Per-channel precision gate: expand (B,C) → (B,N)
                alpha = self._spvae_alpha_per_channel.to(lo_part.dtype)
                # alpha is (B, C), expand to (B, seq*C) = (B, N)
                alpha_BN = alpha.unsqueeze(1).expand(b, seq, c_out).reshape(b, n, 1)
                merged = (1.0 - alpha_BN) * lo_part + alpha_BN * hi_part
                self._last_gate_mean = alpha.mean().item()

            elif self.merge_strategy in ("learned_router", "hi_lora_router"):
                merged, gate = self.learned_router(lo_part, hi_part)
                self._last_gate_mean = gate.mean().item()

            elif self.merge_strategy == "cross_attn":
                # lo attends to hi, with residual
                attn_out, _ = self.cross_attn_layer(
                    lo_part.transpose(0, 1),
                    hi_part.transpose(0, 1),
                    hi_part.transpose(0, 1),
                )
                merged = lo_part + attn_out.transpose(0, 1)

            else:
                # "average" and "hi_lora": fixed 0.5/0.5
                merged = 0.5 * lo_part + 0.5 * hi_part

            tok = torch.cat([tok[:, :_prefix_len, :], merged], dim=1)

        # 7. Late blocks (LoRA)
        for blk in self.blocks[merge_k:]:
            tok = blk(tok)

        return self.norm(tok)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)

    def get_router_gate_mean(self) -> Optional[float]:
        """Return the last mean gate value (for logging). None if not router strategy."""
        return self._last_gate_mean


# ============================================================
# Regressor wrapper
# ============================================================

class HiLoCleanRegressor(nn.Module):
    def __init__(
        self,
        backbone: HiLoCleanBackbone,
        d_out: int = 5,
        token_mode: TokenMode = "mean",
        include_cls: bool = True,
        head_dropout: float = 0.0,
        head_hidden: int = 0,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = TokenRegressor(
            d_out=d_out,
            token_mode=token_mode,
            include_cls=include_cls,
            dropout=head_dropout,
            head_hidden=head_hidden,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        x_hi: Optional[torch.Tensor] = None,
        ecog_xyz: Optional[torch.Tensor] = None,
        sid: Optional[torch.Tensor] = None,
        return_losses: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        tokens = self.backbone.forward_tokens(x, x_hi=x_hi, ecog_xyz=ecog_xyz, sid=sid)
        y_hat = self.head(tokens)
        if not return_losses:
            return y_hat
        return {"y_hat": y_hat}


# ============================================================
# Factory functions
# ============================================================

def hilo_clean_small(**kwargs) -> HiLoCleanBackbone:
    return HiLoCleanBackbone(
        patch_size=16, embed_dim=512, depth=8, num_heads=8, **kwargs
    )

def hilo_clean_base(**kwargs) -> HiLoCleanBackbone:
    return HiLoCleanBackbone(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs
    )

def hilo_clean_large(**kwargs) -> HiLoCleanBackbone:
    return HiLoCleanBackbone(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs
    )
