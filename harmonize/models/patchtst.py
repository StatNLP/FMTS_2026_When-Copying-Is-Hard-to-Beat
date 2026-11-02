"""
harmonize/models/patchtst.py
PatchTST (Nie et al., ICLR 2023, "A Time Series is Worth 64 Words") — a
channel-independent patching Transformer, added as a Paper-1 forecast baseline.

Design (matched to this project's ForecastModel interface):
  * **Channel-independent**: each of the F_target channels is treated as a univariate
    series and run through *shared* weights (B*F batched), as in the paper.
  * **Patching**: the T_in history is split into overlapping patches (patch_len, stride),
    each linearly embedded to d_model + a learned positional embedding, then a Transformer
    encoder; a flatten-linear head predicts the whole T_out horizon (direct multi-step).
  * **RevIN-lite**: per-window, per-channel instance normalisation (subtract mean / divide
    std over the input window, predict, denormalise) — the instance-norm PatchTST relies on,
    applied on top of the global Stage-2 z-score.
  * **Non-autoregressive**: like DLinear / the DMS decoder, `x_target` and
    `teacher_forcing_ratio` are ignored.

It consumes the z-scored, 0-filled `x_in` (same as the other grid models); the observed
mask is not used (PatchTST presupposes a regular grid).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import ForecastModel


class PatchTSTForecast(ForecastModel):
    def __init__(
        self,
        n_features_in: int,
        n_features_target: int,
        n_features_static: int,
        window_in: int,
        window_out: int,
        target_idx_in_input: list[int] | None = None,
        d_model: int = 128,
        n_heads: int = 8,
        dropout: float = 0.1,
        patch_len: int = 8,
        stride: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(n_features_in, n_features_target, n_features_static,
                         window_in, window_out, target_idx_in_input)
        n_layers = int(kwargs.get("n_layers_enc", kwargs.get("n_layers", 3)))
        d_ff = int(kwargs.get("d_ff") or 4 * d_model)
        patch_len = min(patch_len, window_in)
        self.patch_len, self.stride = patch_len, stride
        # ceil, not floor: with floor, (n_patch-1)*stride + patch_len <= window_in always, so
        # `self.pad` below was provably 0 and the trailing timesteps were silently dropped —
        # e.g. patch_len=8/stride=5 discards hour 23, the most recent observation and the
        # strongest predictor for a 24 h forecast. Latent for the evenly-dividing pairs used so
        # far; corrupts any new sweep point.
        self.n_patch = max(1, -(-(window_in - patch_len) // stride) + 1)
        # right-pad (repeat last step) so the patches exactly cover window_in
        self.pad = max(0, (self.n_patch - 1) * stride + patch_len - window_in)

        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, self.n_patch, d_model) * 0.02)
        self.drop = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(self.n_patch * d_model, window_out)

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N, T_in) -> (N, n_patch, patch_len)
        if self.pad:
            x = torch.cat([x, x[:, -1:].expand(-1, self.pad)], dim=1)
        return x.unfold(dimension=1, size=self.patch_len, step=self.stride)

    def forward(self, x_in, mask_in, static, *, x_target=None, teacher_forcing_ratio: float = 0.0):
        B = x_in.shape[0]
        xc = x_in[..., self.target_idx]                       # (B, T_in, F_tgt)
        F = xc.shape[-1]
        xc = xc.permute(0, 2, 1).reshape(B * F, self.window_in)   # (B*F, T_in)
        # RevIN-lite (per-instance over the input window)
        mu = xc.mean(dim=1, keepdim=True)
        sd = xc.std(dim=1, keepdim=True) + 1e-5
        xn = (xc - mu) / sd
        h = self.drop(self.embed(self._patch(xn)) + self.pos)    # (B*F, n_patch, d_model)
        h = self.encoder(h)
        yn = self.head(h.reshape(B * F, -1))                     # (B*F, T_out)
        y = yn * sd + mu                                         # denormalise with input stats
        return y.reshape(B, F, self.window_out).permute(0, 2, 1)  # (B, T_out, F_tgt)
