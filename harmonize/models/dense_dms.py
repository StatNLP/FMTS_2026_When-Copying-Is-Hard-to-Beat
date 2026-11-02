"""
harmonize/models/dense_dms.py

Direct Multi-Step (DMS) decoder — Paper-1 comparison baseline.

Same dense encoder as DenseAutoregressive (concat [x_in ‖ mask_in ‖ static],
linear project, sinusoidal PE, TransformerEncoder). The decoder replaces the
iterative autoregressive loop with T_out learnable query embeddings that
cross-attend to the full encoder memory in one shot.

Key properties:
  - No autoregression: all T_out output steps produced in parallel.
  - No teacher forcing (teacher_forcing_ratio argument is silently ignored).
  - Each output position has a dedicated learnable "query" that attends to the
    encoder's full context; the model must learn to route temporal information
    to the correct output timestep via cross-attention.

Reference: Staniek et al. (MLHC 2024) — DMS variant used as ablation baseline.
"T randomly initialized self-attention modules predict all timesteps
independently." Our implementation uses T_out learnable query embeddings
cross-attending to encoder memory, which is equivalent in spirit.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import ForecastModel


class _PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DenseDirectDecoder(ForecastModel):
    """Dense encoder + Direct Multi-Step (DMS) decoder.

    Hyperparameters (same as DenseAutoregressive for fair comparison):
        d_model:       hidden dim (default 128)
        n_heads:       attention heads (default 4)
        d_ff:          feed-forward dim (default 256)
        n_layers_enc:  encoder depth (default 2)
        n_layers_dec:  decoder depth — cross-attention layers (default 1)
        dropout:       dropout (default 0.1)
        static_to_encoder: broadcast static features into encoder input (True)
    """

    def __init__(
        self,
        n_features_in: int,
        n_features_target: int,
        n_features_static: int,
        window_in: int,
        window_out: int,
        target_idx_in_input: list[int] | None = None,
        *,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        n_layers_enc: int = 2,
        n_layers_dec: int = 1,
        dropout: float = 0.1,
        static_to_encoder: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.static_to_encoder = static_to_encoder

        # Encoder (identical to DenseAutoregressive)
        in_dim = 2 * n_features_in + (n_features_static if static_to_encoder else 0)
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, max_len=window_in + window_out + 8)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # DMS decoder: T_out learnable query embeddings, one per output step
        self.query_embed = nn.Embedding(window_out, d_model)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers_dec)

        self.out_proj = nn.Linear(d_model, n_features_target)

    # ------------------------------------------------------------------
    def _encode(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        feats = [x_in, mask_in.float()]
        if self.static_to_encoder:
            feats.append(static.unsqueeze(1).expand(-1, x_in.size(1), -1))
        x = self.input_proj(torch.cat(feats, dim=-1))
        return self.encoder(self.pos_enc(x))  # (B, T_in, d_model)

    def forward(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target=None,          # unused — DMS is non-autoregressive
        teacher_forcing_ratio: float = 0.0,  # unused
    ) -> torch.Tensor:
        B = x_in.size(0)
        memory = self._encode(x_in, mask_in, static)  # (B, T_in, d_model)

        # T_out fixed query vectors (shared across batch)
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, T_out, d_model)

        # Cross-attend to encoder memory (no causal mask — all steps in parallel)
        dec_out = self.decoder(tgt, memory)    # (B, T_out, d_model)
        return self.out_proj(dec_out)          # (B, T_out, F_target)
