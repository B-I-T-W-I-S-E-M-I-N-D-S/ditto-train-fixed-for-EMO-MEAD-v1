"""
model.py  —  Emotion-aware MotionDecoder for DITTO
====================================================

Changes vs. original (MEMO-inspired):
  1. FiLMTransformerDecoderLayer gains:
       • Separate K/V projections for emotion tokens (MultiModalAttention – Option B).
       • EmoAdaLN applied after the cross-attention + FiLM step.
  2. DecoderLayerStack.forward accepts an optional  emo_embed  [B,T,emo_dim].
  3. MotionDecoder gains:
       • EmotionEncoder submodule (runs on HuBERT slice of cond_embed).
       • Separate emotion token pathway through cond stack.
       • Null emotion embedding for classifier-free guidance.
       • `use_emotion` / `emo_dim` flags – backward-compatible (default: False).
"""

from typing import Callable, Optional, Union
import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import Tensor
from torch.nn import functional as F

from .rotary_embedding_torch import RotaryEmbedding
from .utils import PositionalEncoding, SinusoidalPosEmb, prob_mask_like
from .emotion_encoder import EmotionEncoder
from .emo_adaln import EmoAdaLN, EmoPooler


# ---------------------------------------------------------------------------
# FiLM helpers (unchanged)
# ---------------------------------------------------------------------------

class DenseFiLM(nn.Module):
    """Feature-wise linear modulation (FiLM) generator."""

    def __init__(self, embed_channels):
        super().__init__()
        self.embed_channels = embed_channels
        self.block = nn.Sequential(
            nn.Mish(), nn.Linear(embed_channels, embed_channels * 2)
        )

    def forward(self, position):
        pos_encoding = self.block(position)
        pos_encoding = rearrange(pos_encoding, "b c -> b 1 c")
        scale_shift = pos_encoding.chunk(2, dim=-1)
        return scale_shift


def featurewise_affine(x, scale_shift):
    scale, shift = scale_shift
    return (scale + 1) * x + shift


# ---------------------------------------------------------------------------
# Encoder (unchanged)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = True,
        device=None,
        dtype=None,
        rotary=None,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = activation

        self.rotary = rotary
        self.use_rotary = rotary is not None

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))
        return x

    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk, qk, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


# ---------------------------------------------------------------------------
# Emotion-aware FiLM Decoder Layer
# ---------------------------------------------------------------------------

class FiLMTransformerDecoderLayer(nn.Module):
    """
    DiT decoder block with optional emotion-aware extensions:
      • MultiModalAttention  –  separate K/V projections for emotion tokens.
      • EmoAdaLN             –  emotion-adaptive layer norm after cross-attn.

    When  emo_dim=0  (default) the block is identical to the original.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward=2048,
        dropout=0.1,
        activation=F.relu,
        layer_norm_eps=1e-5,
        batch_first=False,
        norm_first=True,
        device=None,
        dtype=None,
        rotary=None,
        emo_dim: int = 0,           # <- NEW: 0 = emotion disabled
    ):
        super().__init__()
        self.emo_dim    = emo_dim
        self.use_emotion = emo_dim > 0

        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first
        )

        # ── MEMO MultiModalAttention: separate K/V projections for emotion ──
        if self.use_emotion:
            # Projects emotion tokens into the *same* d_model space as audio K/V
            self.emo_k_proj = nn.Linear(emo_dim, d_model)
            self.emo_v_proj = nn.Linear(emo_dim, d_model)

        # Feedforward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout  = nn.Dropout(dropout)
        self.linear2  = nn.Linear(dim_feedforward, d_model)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = activation

        self.film1 = DenseFiLM(d_model)
        self.film2 = DenseFiLM(d_model)
        self.film3 = DenseFiLM(d_model)

        # ── MEMO EmoAdaLN: modulate hidden states with emotion embedding ─────
        if self.use_emotion:
            self.emo_adaln = EmoAdaLN(d_model, emo_dim)

        self.rotary     = rotary
        self.use_rotary = rotary is not None

    # ------------------------------------------------------------------
    def forward(
        self,
        tgt,
        memory,
        t,
        emo_embed: Optional[Tensor] = None,   # <- NEW: [B, T_m, emo_dim] or None
        tgt_mask=None,
        memory_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        x = tgt
        if self.norm_first:
            # 1. Self-attention → FiLM → residual
            x_1 = self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
            x   = x + featurewise_affine(x_1, self.film1(t))

            # 2. Cross-attention (MultiModal) → FiLM → residual
            x_2 = self._mha_block(
                self.norm2(x), memory, attn_mask=memory_mask,
                key_padding_mask=memory_key_padding_mask,
                emo_embed=emo_embed,
            )
            x = x + featurewise_affine(x_2, self.film2(t))

            # 3. EmoAdaLN modulation (NEW – skipped if emotion disabled)
            if self.use_emotion and emo_embed is not None:
                x = self.emo_adaln(x, emo_embed)

            # 4. Feedforward → FiLM → residual
            x_3 = self._ff_block(self.norm3(x))
            x   = x + featurewise_affine(x_3, self.film3(t))
        else:
            x = self.norm1(
                x + featurewise_affine(
                    self._sa_block(x, tgt_mask, tgt_key_padding_mask), self.film1(t)
                )
            )
            x = self.norm2(
                x + featurewise_affine(
                    self._mha_block(x, memory, memory_mask, memory_key_padding_mask,
                                    emo_embed=emo_embed),
                    self.film2(t),
                )
            )
            if self.use_emotion and emo_embed is not None:
                x = self.emo_adaln(x, emo_embed)
            x = self.norm3(x + featurewise_affine(self._ff_block(x), self.film3(t)))
        return x

    # ------------------------------------------------------------------
    def _sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.use_rotary else x
        x = self.self_attn(
            qk, qk, x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )[0]
        return self.dropout1(x)

    def _mha_block(self, x, mem, attn_mask, key_padding_mask,
                   emo_embed: Optional[Tensor] = None):
        """
        MultiModalAttention (MEMO Option B):
          K = concat(K_audio, K_emotion)
          V = concat(V_audio, V_emotion)
        If emotion is disabled or emo_embed is None, behaves exactly as before.
        """
        q = self.rotary.rotate_queries_or_keys(x)   if self.use_rotary else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.use_rotary else mem

        if self.use_emotion and emo_embed is not None:
            # emo_embed: [B, T_e, emo_dim]  →  K_e, V_e: [B, T_e, d_model]
            k_e = self.emo_k_proj(emo_embed)
            v_e = self.emo_v_proj(emo_embed)
            # Concatenate along the sequence (key/value) dimension
            k_mm = torch.cat([k,   k_e], dim=1)   # [B, T_mem + T_e, d_model]
            v_mm = torch.cat([mem, v_e], dim=1)   # [B, T_mem + T_e, d_model]
        else:
            k_mm = k
            v_mm = mem

        x = self.multihead_attn(
            q, k_mm, v_mm,
            attn_mask=attn_mask,
            key_padding_mask=None,   # padding mask shape would change – skip for now
            need_weights=False,
        )[0]
        return self.dropout2(x)

    def _ff_block(self, x):
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# ---------------------------------------------------------------------------
# Decoder stack (emotion-aware)
# ---------------------------------------------------------------------------

class DecoderLayerStack(nn.Module):
    def __init__(self, stack):
        super().__init__()
        self.stack = stack

    def forward(self, x, cond, t, emo_embed: Optional[Tensor] = None):
        for layer in self.stack:
            x = layer(x, cond, t, emo_embed=emo_embed)
        return x


# ---------------------------------------------------------------------------
# MotionDecoder (emotion-aware)
# ---------------------------------------------------------------------------

class MotionDecoder(nn.Module):
    def __init__(
        self,
        nfeats: int,
        seq_len: int = 100,
        latent_dim: int = 256,
        ff_size: int = 1024,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        cond_feature_dim: int = 4800,
        activation: Callable[[Tensor], Tensor] = F.gelu,
        use_rotary=True,
        # ── NEW emotion options ───────────────────────────────────────────
        use_emotion: bool = False,
        emo_dim: int = 128,
        hubert_dim: int = 1024,
        **kwargs
    ) -> None:

        super().__init__()

        self.use_emotion = use_emotion
        self.emo_dim     = emo_dim if use_emotion else 0
        self.hubert_dim  = hubert_dim

        output_feats = nfeats

        # positional embeddings
        self.rotary = None
        self.abs_pos_encoding = nn.Identity()
        if use_rotary:
            self.rotary = RotaryEmbedding(dim=latent_dim)
        else:
            self.abs_pos_encoding = PositionalEncoding(
                latent_dim, dropout, batch_first=True
            )

        # time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.Mish(),
        )
        self.to_time_cond   = nn.Sequential(nn.Linear(latent_dim * 4, latent_dim))
        self.to_time_tokens = nn.Sequential(
            nn.Linear(latent_dim * 4, latent_dim * 2),
            Rearrange("b (r d) -> b r d", r=2),
        )

        # null embeddings for guidance dropout
        self.null_cond_embed  = nn.Parameter(torch.randn(1, seq_len, latent_dim))
        self.null_cond_hidden = nn.Parameter(torch.randn(1, latent_dim))

        # ── NEW: null emotion embed for guidance dropout ──────────────────
        if use_emotion:
            self.null_emo_embed = nn.Parameter(torch.zeros(1, seq_len, emo_dim))

        self.norm_cond = nn.LayerNorm(latent_dim)

        # input projection
        self.input_projection = nn.Linear(nfeats * 2, latent_dim)
        self.cond_encoder = nn.Sequential()
        for _ in range(2):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    d_model=latent_dim,
                    nhead=num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                )
            )

        # conditional projection
        self.cond_projection = nn.Linear(cond_feature_dim, latent_dim)
        self.non_attn_cond_projection = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )

        # ── NEW: EmotionEncoder (runs on HuBERT slice of cond_embed) ──────
        if use_emotion:
            self.emotion_encoder = EmotionEncoder(
                hubert_dim=hubert_dim,
                emo_dim=emo_dim,
                hidden_dim=min(256, emo_dim * 2),
                num_conv_layers=3,
                num_transformer_layers=1,
                dropout=dropout,
            )

        # decoder
        decoderstack = nn.ModuleList([])
        for _ in range(num_layers):
            decoderstack.append(
                FiLMTransformerDecoderLayer(
                    latent_dim,
                    num_heads,
                    dim_feedforward=ff_size,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                    rotary=self.rotary,
                    emo_dim=self.emo_dim,       # <- pass emo_dim to each block
                )
            )

        self.seqTransDecoder = DecoderLayerStack(decoderstack)
        self.final_layer     = nn.Linear(latent_dim, output_feats)
        self.epsilon         = 0.00001

    # ------------------------------------------------------------------
    def guided_forward(self, x, cond_frame, cond_embed, times, guidance_weight):
        unc        = self.forward(x, cond_frame, cond_embed, times, cond_drop_prob=1)
        conditioned = self.forward(x, cond_frame, cond_embed, times, cond_drop_prob=0)
        return unc + (conditioned - unc) * guidance_weight

    # ------------------------------------------------------------------
    def forward(
        self,
        x: Tensor,
        cond_frame: Tensor,
        cond_embed: Tensor,
        times: Tensor,
        cond_drop_prob: float = 0.0,
    ):
        batch_size, device = x.shape[0], x.device

        # concat last frame, project to latent space
        x = torch.cat([x, cond_frame.unsqueeze(1).repeat(1, x.shape[1], 1)], dim=-1)
        x = self.input_projection(x)
        x = self.abs_pos_encoding(x)

        # ── Classifier-free guidance mask ────────────────────────────────
        keep_mask        = prob_mask_like((batch_size,), 1 - cond_drop_prob, device=device)
        keep_mask_embed  = rearrange(keep_mask, "b -> b 1 1")
        keep_mask_hidden = rearrange(keep_mask, "b -> b 1")

        # ── Audio conditioning pathway ────────────────────────────────────
        cond_embed = cond_embed.to(self.cond_projection.weight.dtype)
        cond_tokens = self.cond_projection(cond_embed)       # [B, T, latent_dim]
        cond_tokens = self.abs_pos_encoding(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        null_cond_embed = self.null_cond_embed.to(cond_tokens.dtype)
        cond_tokens = torch.where(keep_mask_embed, cond_tokens, null_cond_embed)

        mean_pooled_cond_tokens = cond_tokens.mean(dim=-2)
        cond_hidden = self.non_attn_cond_projection(mean_pooled_cond_tokens)

        # ── Emotion pathway (NEW) ─────────────────────────────────────────
        emo_embed = None
        if self.use_emotion:
            # Slice HuBERT features (always the first `hubert_dim` channels)
            hubert_feats = cond_embed[:, :, : self.hubert_dim]   # [B, T, hubert_dim]
            emo_embed    = self.emotion_encoder(hubert_feats)      # [B, T, emo_dim]

            # Apply same guidance dropout mask
            null_emo = self.null_emo_embed.expand(batch_size, -1, -1).to(emo_embed.dtype)
            emo_embed = torch.where(keep_mask_embed.expand_as(emo_embed),
                                    emo_embed, null_emo)

        # ── Diffusion timestep embedding ──────────────────────────────────
        t_hidden  = self.time_mlp(times)
        t         = self.to_time_cond(t_hidden)
        t_tokens  = self.to_time_tokens(t_hidden)

        null_cond_hidden = self.null_cond_hidden.to(t.dtype)
        cond_hidden = torch.where(keep_mask_hidden, cond_hidden, null_cond_hidden)
        t += cond_hidden

        # Cross-attention conditioning tokens = [audio_tokens, time_tokens]
        c          = torch.cat((cond_tokens, t_tokens), dim=-2)
        cond_tokens = self.norm_cond(c)

        # ── Transformer decoder (emotion injected per-block) ──────────────
        output = self.seqTransDecoder(x, cond_tokens, t, emo_embed=emo_embed)

        output = self.final_layer(output)
        return output