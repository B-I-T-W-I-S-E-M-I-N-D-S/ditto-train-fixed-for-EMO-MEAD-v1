"""
emotion_encoder.py
------------------
Temporal emotion feature extractor inspired by MEMO's emotion-aware audio module.

Takes frame-level HuBERT audio features [B, T, hubert_dim] and produces a
temporally-aligned emotion embedding  e(t): [B, T, emo_dim]  that is used to:
  1. Enrich cross-attention keys/values in each DiT block (MultiModalAttention).
  2. Drive Emotion-Adaptive LayerNorm (EmoAdaLN) after cross-attention.

Architecture
------------
  - Lightweight input projection (Linear)
  - Stack of Conv1D residual blocks for local temporal context
  - Optional 1–2 Transformer encoder layers for global context
  - Output projection to emo_dim
  - Optional EMA smoothing for temporal stability (inference only)

Design choices
--------------
  - No external pretrained emotion model required at runtime – the encoder is
    trained end-to-end with the rest of Stage-2/3.
  - HuBERT dim is always the *first* hubert_dim channels of cond_embed
    (default 1024), so this module slices that from the shared conditioning.
  - The module is small (~1-2M params) to preserve real-time performance.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _ConvResBlock(nn.Module):
    """1-D causal-ish Conv residual block (non-causal for training speed)."""

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        residual = x
        x = F.gelu(self.norm1(self.conv1(x)))
        x = self.drop(self.norm2(self.conv2(x)))
        return x + residual


class _TransformerEncoderBlock(nn.Module):
    """Lightweight single Transformer encoder layer (Pre-LN)."""

    def __init__(self, d_model: int, nhead: int = 4, ff_mult: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                           batch_first=True)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        _x = self.norm1(x)
        _x, _ = self.attn(_x, _x, _x, need_weights=False)
        x = x + _x
        x = x + self.ff(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# EmotionEncoder
# ---------------------------------------------------------------------------

class EmotionEncoder(nn.Module):
    """
    Frame-level emotion encoder for DITTO's emotion-aware DiT.

    Parameters
    ----------
    hubert_dim : int
        Dimensionality of the HuBERT slice from cond_embed (default 1024).
    emo_dim : int
        Output emotion embedding dimension (default 128).
    hidden_dim : int
        Internal channel width (default 256).
    num_conv_layers : int
        Number of Conv1D residual blocks (default 3).
    num_transformer_layers : int
        Number of Transformer encoder layers after convs (default 1).
    dropout : float
        Dropout probability (default 0.1).
    ema_alpha : float
        EMA smoothing factor for inference (0 = no smoothing, 1 = identity).
        Applied only when `smooth_inference=True` is passed to forward.
    """

    def __init__(
        self,
        hubert_dim: int = 1024,
        emo_dim: int = 128,
        hidden_dim: int = 256,
        num_conv_layers: int = 3,
        num_transformer_layers: int = 1,
        dropout: float = 0.1,
        ema_alpha: float = 0.9,
    ):
        super().__init__()

        self.hubert_dim = hubert_dim
        self.emo_dim    = emo_dim
        self.ema_alpha  = ema_alpha

        # Input projection: hubert_dim → hidden_dim
        self.input_proj = nn.Sequential(
            nn.LayerNorm(hubert_dim),
            nn.Linear(hubert_dim, hidden_dim),
            nn.GELU(),
        )

        # Conv1D temporal residual stack (operates on [B, C, T])
        self.conv_blocks = nn.ModuleList([
            _ConvResBlock(hidden_dim, kernel_size=5, dropout=dropout)
            for _ in range(num_conv_layers)
        ])

        # Optional Transformer layers for longer-range context
        self.transformer_blocks = nn.ModuleList([
            _TransformerEncoderBlock(hidden_dim, nhead=4, ff_mult=2,
                                     dropout=dropout)
            for _ in range(num_transformer_layers)
        ])

        # Output projection: hidden_dim → emo_dim
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, emo_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        hubert_feats: torch.Tensor,
        smooth_inference: bool = False,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        hubert_feats : Tensor [B, T, hubert_dim]
            Frame-level HuBERT features (first `hubert_dim` channels of aud cond).
        smooth_inference : bool
            If True, apply EMA smoothing along the time axis (inference only).

        Returns
        -------
        e : Tensor [B, T, emo_dim]
            Temporally-aligned emotion embeddings.
        """
        # 1. Input projection: [B, T, hubert_dim] → [B, T, hidden_dim]
        x = self.input_proj(hubert_feats)

        # 2. Conv residual blocks: need [B, C, T]
        x = rearrange(x, 'b t c -> b c t')
        for block in self.conv_blocks:
            x = block(x)
        x = rearrange(x, 'b c t -> b t c')

        # 3. Transformer encoder layers: [B, T, hidden_dim]
        for block in self.transformer_blocks:
            x = block(x)

        # 4. Output projection: [B, T, hidden_dim] → [B, T, emo_dim]
        e = self.output_proj(x)

        # 5. Optional EMA smoothing along time (inference only, no grad needed)
        if smooth_inference and not self.training:
            e = self._ema_smooth(e)

        return e  # [B, T, emo_dim]

    @torch.no_grad()
    def _ema_smooth(self, e: torch.Tensor) -> torch.Tensor:
        """Apply exponential moving average along the time dimension."""
        alpha = self.ema_alpha
        smoothed = [e[:, 0:1]]
        for t in range(1, e.size(1)):
            s = alpha * smoothed[-1] + (1.0 - alpha) * e[:, t : t + 1]
            smoothed.append(s)
        return torch.cat(smoothed, dim=1)
