
# ECoG_Finger/models/steegformer.py
# Clean regression-focused STEEGFormer model (Backbone + Wrapper)
#   - input x: [B, C, T]
#   - channel ordering corresponds to chan_idx (len=C) OR identity if not provided
#   - optionally condition channel embeddings on ECoG electrode xyz via a differentiable "soft top-k" fusion
#
# This file is self-contained except for timm.

from __future__ import annotations

import math
from functools import partial
from typing import Optional, Literal, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm.models.vision_transformer

# -------------------------
# AMP autocast compatibility
# -------------------------
try:
    from torch.amp import autocast as _autocast_new
    autocast_amp = partial(_autocast_new, device_type="cuda")
except Exception:  # older torch
    from torch.cuda.amp import autocast as autocast_amp
    
# =========================
# ECoG -> EEG embedding fusion (soft top-k)
# =========================
#Bottleneck version
class ECoG2EEGEmbeddingSoftTopK(nn.Module):
    """
    Adds a cheap bottleneck adapter after fusion:
      LN(D) + Linear(D->r) + GELU + Linear(r->D)
    Params ~ 2*D*r (vs D^2).
    """
    def __init__(
        self,
        eeg_emb_fixed: torch.Tensor,  # (M,D)
        hidden: int = 64,
        bottleneck: int = 64,         # r (try 8/16/32)
        tau: float = 0.3,
        power: float = 2.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        if eeg_emb_fixed.ndim != 2:
            raise ValueError(f"eeg_emb_fixed must be (M,D), got {tuple(eeg_emb_fixed.shape)}")
        M, D = eeg_emb_fixed.shape
        self.M, self.D = int(M), int(D)
        self.eps = float(eps)

        self.register_buffer("tau", torch.tensor(float(tau)), persistent=True)
        self.register_buffer("power", torch.tensor(float(power)), persistent=True)
        self.register_buffer("E", eeg_emb_fixed.to(torch.float32), persistent=True)

        self.net = nn.Sequential(
            nn.Linear(3, hidden),
            nn.GELU(),
            nn.Linear(hidden, self.M),
        )

        r = int(bottleneck)
        self.out_proj = nn.Sequential(
            nn.LayerNorm(self.D),
            nn.Linear(self.D, r, bias=False),
            nn.GELU(),
            nn.Linear(r, self.D, bias=False),
        )

        # start close to identity-ish: make last layer small
        nn.init.zeros_(self.out_proj[-1].weight)

    def forward(self, pos_ecog_xyz: torch.Tensor, *, tau=None, power=None, return_weights=True):
        squeeze_b = False
        if pos_ecog_xyz.ndim == 2:
            pos_ecog_xyz = pos_ecog_xyz.unsqueeze(0)
            squeeze_b = True

        logits = self.net(pos_ecog_xyz)  # (B,C,M)

        t = float(self.tau.item()) if tau is None else float(tau)
        p = float(self.power.item()) if power is None else float(power)
        t = max(t, 1e-6)
        p = max(p, 1.0)

        w = F.softmax(logits / t, dim=-1)
        if p != 1.0:
            w = w.clamp_min(self.eps).pow(p)
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(self.eps)

        fused = torch.matmul(w, self.E)      # (B,C,D)
        fused = fused + self.out_proj(fused) # residual adapter

        if squeeze_b:
            fused = fused.squeeze(0)
            w = w.squeeze(0)
        return (fused, w) if return_weights else fused


# =========================
# Patch embedding for EEG/ECoG
# =========================

class PatchEmbed1D(nn.Module):
    """
    Patchify along time for 1D signal:
      input:  [B, C, T]
      output: [B, Seq, C, D]
    """
    def __init__(self, patch_size: int, embed_dim: int):
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        self.unfold = torch.nn.Unfold(kernel_size=(1, self.patch_size), stride=self.patch_size)
        self.proj = nn.Linear(self.patch_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        x = x.unsqueeze(2)                            # [B,C,1,T]
        u = self.unfold(x)                            # [B,C*P,Seq]
        _, _, seq = u.shape
        u = u.view(b, c, self.patch_size, seq)        # [B,C,P,Seq]
        tok = u.permute(0, 3, 1, 2).contiguous()      # [B,Seq,C,P]
        return self.proj(tok)                         # [B,Seq,C,D]
    
    
class PatchEmbedEEG(nn.Module):
    """
    Patchify along time:
      input:  [B, C, T]
      output: [B, Seq, C, D]
    """
    def __init__(self, patch_size: int = 16, embed_dim: int = 768):
        super().__init__()
        self.patch_size = int(patch_size)
        self.embed_dim = int(embed_dim)

        self.unfold = torch.nn.Unfold(kernel_size=(1, self.patch_size), stride=self.patch_size)
        self.proj = nn.Linear(self.patch_size, self.embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        x = x.unsqueeze(2)                            # [B, C, 1, T]
        u = self.unfold(x)                            # [B, C*P, Seq]
        _, _, seq = u.shape
        u = u.view(b, c, self.patch_size, seq)        # [B, C, P, Seq]
        tok = u.permute(0, 3, 1, 2).contiguous()      # [B, Seq, C, P]
        return self.proj(tok)                         # [B, Seq, C, D]


class ChannelPositionalEmbed(nn.Module):
    def __init__(self, embedding_dim: int, max_ch_idx: int = 145):
        super().__init__()
        self.emb = nn.Embedding(max_ch_idx, embedding_dim)
        nn.init.zeros_(self.emb.weight)

    def forward(self, channel_indices: torch.Tensor) -> torch.Tensor:
        # channel_indices: [B, N]
        return self.emb(channel_indices)


class TemporalPositionalEncoding(nn.Module):
    """
    Sin/cos positional encoding lookup by integer indices.
    """
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).float())
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position.float() * div_term)
        pe[0, :, 1::2] = torch.cos(position.float() * div_term)
        self.register_buffer("pe", pe)

    def cls_token_pe(self) -> torch.Tensor:
        return self.pe[0, 0, :]  # [D]

    def forward(self, seq_indices: torch.Tensor) -> torch.Tensor:
        # seq_indices: [B, N]
        b, n = seq_indices.shape
        return self.pe[0, seq_indices.reshape(-1)].view(b, n, -1)


# =========================
# Backbone: returns tokens [B, 1+N, D]
# =========================
class STEEGFormerBackbone(timm.models.vision_transformer.VisionTransformer):
    """
    ViT backbone for EEG/ECoG:
      - patchify along time
      - add temporal + (optionally fused) channel embeddings
      - returns tokens [B, 1+N, D] (CLS + patch tokens)

    ECoG conditioning:
      If you call `attach_ecog_fuser_from_channel_embed(...)` and set per-subject xyz
      using `set_ecog_xyz(...)`, the channel embedding term becomes:
          ch_emb[c] = soft_topk(xyz[c]) @ E   (E from the backbone channel embedding table)
      Then it is broadcast across time patches.
    """
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
    ):
        super().__init__(
            img_size=224,       # unused
            patch_size=patch_size,
            in_chans=3,         # unused
            num_classes=0,      # no classifier
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            norm_layer=norm_layer,
        )

        # Remove timm's image pos_embed (we use our own)
        if hasattr(self, "pos_embed"):
            delattr(self, "pos_embed")
            
        self.patch_embed = PatchEmbedEEG(patch_size=patch_size, embed_dim=embed_dim)
        self.enc_channel = ChannelPositionalEmbed(embed_dim, max_ch_idx=max_ch_idx)
        self.enc_time = TemporalPositionalEncoding(embed_dim, max_len=max_len)

        if chan_idx is None:
            chan_idx = list(range(int(expect_num_chans)))
        if len(chan_idx) != int(expect_num_chans):
            raise ValueError(f"chan_idx must have length {expect_num_chans}, got {len(chan_idx)}")

        self.expect_num_chans = int(expect_num_chans)
        self.register_buffer("default_chan_idx", torch.tensor(chan_idx, dtype=torch.long), persistent=False)

        # ECoG conditioning
        self.ecog_fuser: Optional[ECoG2EEGEmbeddingSoftTopK] = None
        self.register_buffer("ecog_xyz", torch.empty(0, 3), persistent=False)  # (C,3) set later

    # ---- public helpers ----
    def set_ecog_xyz(self, ecog_xyz_m: torch.Tensor):
        """
        ecog_xyz_m: (C,3) tensor in meters (or any consistent unit)
        """
        if ecog_xyz_m.ndim != 2 or ecog_xyz_m.shape[1] != 3:
            raise ValueError(f"ecog_xyz must be (C,3), got {tuple(ecog_xyz_m.shape)}")
        # store on same device as module buffers (will be moved by .to(device))
        self.ecog_xyz = ecog_xyz_m.detach().to(dtype=torch.float32)

    def attach_ecog_fuser_from_channel_embed(
        self,
        *,
        M: Optional[int] = None,
        hidden: int = 128,
        tau: float = 0.3,
        power: float = 2.0,
        out_proj: bool = True,
    ):
        """
        Use the backbone's own channel embedding table as the fixed EEG embedding bank.
        Call this AFTER loading the pretrained checkpoint (so the table is meaningful).

        M: number of EEG embeddings to use from the table. If None, use all rows.
        """
        # Use the current device of the backbone parameters as the target device
        dev = self.enc_channel.emb.weight.device

        with torch.no_grad():
            # IMPORTANT: clone() to avoid any weird ties to the original Parameter storage
            E = self.enc_channel.emb.weight.detach().clone()
            if M is not None:
                E = E[: int(M)]
            # keep it on the same device
            E = E.to(device=dev, dtype=torch.float32)

        fuser = ECoG2EEGEmbeddingSoftTopK(
            eeg_emb_fixed=E,
            hidden=hidden,
            tau=tau,
            power=power,
        )

        # CRITICAL: if this module is attached after model.to(device),
        # we must explicitly move it.
        fuser = fuser.to(dev)

        self.ecog_fuser = fuser


    # ---- forward ----
    def forward_tokens(self, x: torch.Tensor, *, ecog_xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: [B, C, T]
        ecog_xyz: optional (C,3) or (B,C,3). If None, will use self.ecog_xyz if present.
        """
        if x.dim() != 3:
            raise ValueError(f"Expected x [B,C,T], got {tuple(x.shape)}")
        b, c, _ = x.shape
        if c != self.expect_num_chans:
            raise ValueError(f"Expected C={self.expect_num_chans}, got C={c}")

        # patchify along time: [B, Seq, C, D]
        tok = self.patch_embed(x)
        b, seq, ch, d = tok.shape
        #print("tokens:",tok.shape)
        n = seq * ch
        tok = tok.view(b, n, d)  # [B, N, D]

        # channel indices repeated across time patches
        chan_idx = self.default_chan_idx.to(x.device)                     # [C]
        chan_idx_bc = chan_idx.unsqueeze(0).unsqueeze(1).repeat(b, seq, 1)  # [B,Seq,C]
        chan_idx_bn = chan_idx_bc.reshape(b, n)                           # [B,N]

        # time indices (1..Seq); 0 reserved for CLS PE
        t_idx = torch.arange(1, seq + 1, device=x.device)                 # [Seq]
        t_idx = t_idx.unsqueeze(0).unsqueeze(-1).repeat(b, 1, ch)         # [B,Seq,C]
        t_idx = t_idx.reshape(b, n)                                       # [B,N]

        # channel embedding term
        if self.ecog_fuser is not None:
            xyz = ecog_xyz
            if xyz is None and self.ecog_xyz.numel() > 0:
                xyz = self.ecog_xyz
            if xyz is None:
                raise ValueError("ECoG fuser is attached, but no ecog_xyz provided and self.ecog_xyz not set.")
            fused, _w = self.ecog_fuser(xyz, return_weights=True)  # (C,D) or (B,C,D)
            if fused.ndim == 2:
                # (C,D) -> (B,Seq,C,D)
                ch_emb = fused.unsqueeze(0).unsqueeze(1).expand(b, seq, -1, -1).reshape(b, n, d)
            else:
                # (B,C,D) -> (B,Seq,C,D)
                ch_emb = fused.unsqueeze(1).expand(-1, seq, -1, -1).reshape(b, n, d)
        else:
            ch_emb = self.enc_channel(chan_idx_bn)  # [B,N,D]

        tok = tok + self.enc_time(t_idx) + ch_emb

        # CLS token with its positional encoding
        cls_pe = self.enc_time.cls_token_pe().to(tok.device).view(1, 1, -1)
        cls = self.cls_token + cls_pe
        cls = cls.expand(b, -1, -1)                 # [B,1,D]
        tok = torch.cat([cls, tok], dim=1)          # [B,1+N,D]

        tok = self.pos_drop(tok)
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.norm(tok)
        return tok

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_tokens(x)


# =========================
# Token -> regression head
# =========================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Literal

TokenMode = Literal["flatten", "mean", "cls", "cnn"]


class CNNTokenHead(nn.Module):
    """
    Input:  tokens (B, 1 + T*C, D)  [CLS + flattened (T,C)]
    Output: (B, d_out) by default, or (B, d_out, 1) if return_5x1=True

    Pipeline:
      1) drop CLS
      2) reshape to (B, C, T, D) (correct for STEEGFormerBackbone flattening: (T,C) then flatten)
      3) depthwise Conv2d kernel (T,1): (B,C,T,D)->(B,C,1,D)
      4) 1x1 Conv2d: C->H, then H->d_out: (B,d_out,1,D)
      5) projection over D: Linear(D->1) => (B,d_out,1,1)
    """
    def __init__(
        self,
        n_channels: int,
        t_per_channel: int,
        embed_dim: int,
        hidden_channels: int = 16,
        d_out: int = 5,
        token_merge_bias: bool = False,
        chan_bias: bool = True,
        out_proj_bias: bool = True,
        return_5x1: bool = False,
    ):
        super().__init__()
        C = int(n_channels)
        T = int(t_per_channel)
        D = int(embed_dim)
        H = int(hidden_channels)
        O = int(d_out)

        self.C, self.T, self.D, self.H, self.O = C, T, D, H, O
        self.return_5x1 = bool(return_5x1)

        # (3) depthwise token merge over T: kernel (T,1) collapses T -> 1
        self.token_merge = nn.Conv2d(
            in_channels=C,
            out_channels=C,
            kernel_size=(T, 1),
            groups=C,
            bias=token_merge_bias,
        )

        # (4) channel mixing via 1x1 convs: C -> H -> O
        self.chan_mix = nn.Sequential(
            nn.Conv2d(C, H, kernel_size=(1, 1), bias=chan_bias),
            nn.GELU(),
            nn.Conv2d(H, O, kernel_size=(1, 1), bias=chan_bias),
        )

        # (5) projection over D: (B,O,1,D)->(B,O,1,1)
        self.out_proj = nn.Linear(D, 1, bias=out_proj_bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens (B,L,D), got {tuple(tokens.shape)}")
        B, L, D = tokens.shape
        if D != self.D:
            raise ValueError(f"Expected embed_dim D={self.D}, got D={D}")
        if L != 1 + self.T * self.C:
            raise ValueError(f"Expected L == 1 + T*C == {1 + self.T*self.C}, got L={L}")

        # 1) drop CLS
        x = tokens[:, 1:, :]  # (B, T*C, D)

        # 2) correct unflatten: (B, T, C, D) then to (B, C, T, D)
        x = x.reshape(B, self.T, self.C, self.D).permute(0, 2, 1, 3)  # (B, C, T, D)

        # 3) token merge: (B,C,T,D)->(B,C,1,D)
        x = self.token_merge(x)

        # 4) channel mixing: (B,C,1,D)->(B,O,1,D)
        x = self.chan_mix(x)

        # 5) project over D: (B,O,1,D)->(B,O,1,1)
        x = self.out_proj(x)

        if self.return_5x1:
            return x.squeeze(-1)          # (B,O,1)

        return x.squeeze(-1).squeeze(-1)  # (B,O)


class TokenRegressor(nn.Module):
    """
    Probing head.
    - token_mode="flatten": flatten tokens (optionally incl. CLS)
    - token_mode="mean": mean over tokens (optionally incl. CLS)
    - token_mode="cls": use CLS only
    - token_mode="cnn":
        Uses CNNTokenHead:
          CLS dropped, reshape to (B,C,T,D),
          depthwise (T,1) conv for token merge,
          1x1 convs for channel mixing C->H->d_out,
          then Linear(D->1) projection.
    """
    def __init__(
        self,
        d_out: int,
        token_mode: TokenMode = "flatten",
        include_cls: bool = True,   # ignored in cnn mode
        dropout: float = 0.0,

        # --- cnn args ---
        n_channels: Optional[int] = 62,
        t_per_channel: Optional[int] = None,   # if None infer from (L-1)/C
        hidden_channels: int = 16,
        token_merge_bias: bool = False,
        chan_bias: bool = True,
        out_proj_bias: bool = True,
        return_5x1: bool = False,
    ):
        super().__init__()
        self.d_out = int(d_out)
        self.token_mode = token_mode
        self.include_cls = bool(include_cls)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # non-cnn head (lazy)
        self.head: Optional[nn.Linear] = None

        # cnn config
        self.n_channels = n_channels
        self.t_per_channel = t_per_channel
        self.hidden_channels = int(hidden_channels)
        self.token_merge_bias = bool(token_merge_bias)
        self.chan_bias = bool(chan_bias)
        self.out_proj_bias = bool(out_proj_bias)
        self.return_5x1 = bool(return_5x1)

        # lazy cnn module
        self.cnn_head: Optional[CNNTokenHead] = None

    def _select_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.include_cls:
            return tokens
        return tokens[:, 1:, :] if tokens.shape[1] > 1 else tokens

    def _ensure_head(self, tokens: torch.Tensor):
        if self.head is not None:
            return
        _, L, D = tokens.shape
        if self.token_mode in ("cls", "mean"):
            in_dim = D
        elif self.token_mode == "flatten":
            in_dim = L * D
        else:
            raise ValueError(f"_ensure_head called for token_mode={self.token_mode}")
        self.head = nn.Linear(in_dim, self.d_out).to(tokens.device)

    def _ensure_cnn(self, tokens: torch.Tensor):
        if self.cnn_head is not None:
            return

        if self.n_channels is None or int(self.n_channels) <= 0:
            raise ValueError("token_mode='cnn' requires n_channels.")
        C = int(self.n_channels)

        if tokens.ndim != 3:
            raise ValueError(f"Expected tokens (B,L,D), got {tuple(tokens.shape)}")
        _, L, D = tokens.shape
        if L < 2:
            raise ValueError("cnn mode expects CLS + at least 1 non-CLS token.")

        L_no_cls = L - 1
        if self.t_per_channel is None:
            if L_no_cls % C != 0:
                raise ValueError(f"Cannot infer t_per_channel: (L-1)={L_no_cls} not divisible by C={C}.")
            T = L_no_cls // C
        else:
            T = int(self.t_per_channel)
            if C * T != L_no_cls:
                raise ValueError(f"cnn reshape requires (L-1)==C*T, got (L-1)={L_no_cls}, C={C}, T={T}.")

        self.cnn_head = CNNTokenHead(
            n_channels=C,
            t_per_channel=T,
            embed_dim=D,
            hidden_channels=self.hidden_channels,
            d_out=self.d_out,
            token_merge_bias=self.token_merge_bias,
            chan_bias=self.chan_bias,
            out_proj_bias=self.out_proj_bias,
            return_5x1=self.return_5x1,
        ).to(tokens.device)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.token_mode == "cnn":
            self._ensure_cnn(tokens)
            out = self.cnn_head(tokens)          # (B,d_out) or (B,d_out,1)
            # optional dropout on the (B,d_out) representation
            if out.ndim == 2:
                out = self.dropout(out)
            elif out.ndim == 3:
                out = self.dropout(out.squeeze(-1)).unsqueeze(-1)
            return out

        # ---------- non-cnn modes ----------
        tokens = self._select_tokens(tokens)
        self._ensure_head(tokens)

        if self.token_mode == "cls":
            feat = tokens[:, 0, :]
        elif self.token_mode == "mean":
            feat = tokens.mean(dim=1)
        elif self.token_mode == "flatten":
            feat = tokens.flatten(1)
        else:
            raise ValueError(f"Unknown token_mode={self.token_mode}")

        feat = self.dropout(feat)
        return self.head(feat)




class STEEGFormerRegressor(nn.Module):
    """
    Wrapper: y_hat = head(backbone.forward_tokens(x))
    """
    def __init__(
        self,
        backbone: STEEGFormerBackbone,
        d_out: int,
        token_mode: TokenMode = "flatten",
        include_cls: bool = True,
        head_dropout: float = 0.0,
        c_in: int = None,
        seq_l: int = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = TokenRegressor(d_out=d_out, token_mode=token_mode, include_cls=include_cls, dropout=head_dropout, n_channels=c_in, t_per_channel=seq_l)

    def forward(self, x: torch.Tensor, *, ecog_xyz: Optional[torch.Tensor] = None) -> torch.Tensor:
        #print("input:", x.shape)
        tokens = self.backbone.forward_tokens(x, ecog_xyz=ecog_xyz)
        return self.head(tokens)


# =========================
# Factories
# =========================
def steegformer_small(**kwargs) -> STEEGFormerBackbone:
    return STEEGFormerBackbone(patch_size=16, embed_dim=512, depth=8, num_heads=8, mlp_ratio=4.0, qkv_bias=True, **kwargs)

def steegformer_base(**kwargs) -> STEEGFormerBackbone:
    return STEEGFormerBackbone(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True, **kwargs)

def steegformer_large(**kwargs) -> STEEGFormerBackbone:
    return STEEGFormerBackbone(patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4.0, qkv_bias=True, **kwargs)
