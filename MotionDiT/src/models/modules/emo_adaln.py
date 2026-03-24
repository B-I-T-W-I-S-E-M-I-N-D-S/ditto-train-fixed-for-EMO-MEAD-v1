# emo_adaln.py — fix: move Optional import to the top
from typing import Optional

import torch
import torch.nn as nn


class EmoAdaLN(nn.Module):
    """
    Emotion-Adaptive Layer Normalisation.

    Parameters
    ----------
    hidden_dim : int
        Dimensionality of the hidden states  (d_model of the DiT).
    emo_dim : int
        Dimensionality of the emotion embedding from EmotionEncoder.
    mlp_hidden : int
        Hidden width of the scale/shift MLPs (default: 2 × emo_dim).
    eps : float
        LayerNorm epsilon.
    """

    def __init__(
        self,
        hidden_dim: int,
        emo_dim: int,
        mlp_hidden: Optional[int] = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        if mlp_hidden is None:
            mlp_hidden = emo_dim * 2

        self.norm = nn.LayerNorm(hidden_dim, eps=eps)

        # MLP_gamma  predicts per-element scale  (γ)
        self.mlp_gamma = nn.Sequential(
            nn.Linear(emo_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # MLP_beta   predicts per-element shift  (β)
        self.mlp_beta = nn.Sequential(
            nn.Linear(emo_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, hidden_dim),
        )

        # Initialise to near-identity (γ≈0, β≈0) for stable training start
        nn.init.zeros_(self.mlp_gamma[-1].weight)
        nn.init.zeros_(self.mlp_gamma[-1].bias)
        nn.init.zeros_(self.mlp_beta[-1].weight)
        nn.init.zeros_(self.mlp_beta[-1].bias)

    def forward(
        self,
        x: torch.Tensor,
        e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, T, hidden_dim]
            Hidden states to be modulated.
        e : Tensor  [B, T, emo_dim]  or  [B, emo_dim]
            Emotion embedding.  If 2-D it is broadcast over all T frames.

        Returns
        -------
        out : Tensor [B, T, hidden_dim]
        """
        if e.dim() == 2:
            # Clip-level pooled emotion: [B, emo_dim] → [B, 1, emo_dim]
            e = e.unsqueeze(1)

        gamma = self.mlp_gamma(e)   # [B, T_e, hidden_dim]  (T_e=1 or T)
        beta  = self.mlp_beta(e)    # [B, T_e, hidden_dim]

        return self.norm(x) * (1.0 + gamma) + beta


# ---------------------------------------------------------------------------
# Small helper: pools frame-level e(t) to a per-clip vector
# ---------------------------------------------------------------------------

class EmoPooler(nn.Module):
    """
    Reduces  e(t) [B, T, emo_dim]  to a single clip-level vector [B, emo_dim]
    via attention-weighted mean pooling.  Used when a single emotion token
    is preferred over per-frame modulation.
    """

    def __init__(self, emo_dim: int):
        super().__init__()
        self.attn_score = nn.Linear(emo_dim, 1)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e: [B, T, emo_dim] → pooled: [B, emo_dim]"""
        w = torch.softmax(self.attn_score(e), dim=1)   # [B, T, 1]
        return (w * e).sum(dim=1)                       # [B, emo_dim]
