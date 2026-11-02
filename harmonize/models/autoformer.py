"""
harmonize/models/autoformer.py

Autoformer — Paper-1 comparison baseline.

Wraps the Autoformer architecture (Wu et al., NeurIPS 2021) in the ForecastModel
interface. Key architectural features:
  - Series decomposition: moving-average trend + residual seasonal component
  - AutoCorrelation replaces self-attention (FFT-based time-delay aggregation)
  - Encoder: decomposition blocks with AutoCorrelation
  - Decoder: cross-AutoCorrelation + series decomposition projection

Encoder input: concat [x_in ‖ mask_in ‖ static_broadcast] → linear → d_model.
This matches the DenseAutoregressive encoder convention (mask as explicit input).

Decoder initialization (Autoformer convention):
  - "Observed" part: last label_len steps of x_in projected to F_target
  - "Target" part: zeros for T_out steps
  x_dec = cat([x_in[:, -label_len:, :F_target], zeros(B, T_out, F_target)], 1)
  If F_target > F_in, the extra target features are zero-initialized.
  If F_target < F_in, only the first F_target input features are used.

No temporal marks (x_mark_enc / x_mark_dec) — zeros passed to archive code.

Reference:
  Wu, H. et al. "Autoformer: Decomposition Transformers with Auto-Correlation
  for Long-Term Series Forecasting." NeurIPS 2021.
  Original code: 10_scripts/archive/.../models/Autoformer.py (adapted).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import ForecastModel
from .layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from .layers.Autoformer_EncDec import (
    Encoder, EncoderLayer,
    Decoder, DecoderLayer,
    my_Layernorm, series_decomp,
)
from .layers.Embed import DataEmbedding_wo_pos_temp as _Embed


class AutoformerForecast(ForecastModel):
    """Autoformer time-series forecasting model.

    Hyperparameters:
        d_model:       hidden dim (default 128)
        n_heads:       attention heads (default 4)
        d_ff:          feed-forward dim (default 256)
        n_layers_enc:  encoder depth (default 2)
        n_layers_dec:  decoder depth (default 1)
        dropout:       dropout (default 0.1)
        moving_avg:    moving-average kernel for series decomp (odd, clamped to <= window_in;
                       default = largest odd value <= window_in // 2)
        factor:        AutoCorrelation top-k factor (default 1)
        static_to_encoder: broadcast static into encoder input (default True)
        label_len:     number of input steps fed as decoder "observed" context
                       (default 12 = half of window_in=24)
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
        moving_avg: int | None = None,
        factor: int = 1,
        static_to_encoder: bool = True,
        label_len: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.static_to_encoder = static_to_encoder
        self.label_len  = label_len if label_len is not None else window_in // 2
        self.window_out = window_out

        # Encoder input dimension: [x_in ‖ mask_in ‖ static] (concat)
        enc_in = 2 * n_features_in + (n_features_static if static_to_encoder else 0)
        dec_in = n_features_target   # decoder processes target-space features

        # DataEmbedding_wo_pos: Linear(in, d_model) + dropout (no position, no time mark)
        self.enc_embedding = _Embed(enc_in,  d_model, dropout=dropout)
        self.dec_embedding = _Embed(dec_in,  d_model, dropout=dropout)

        # Decomposition kernel — must be odd for symmetric padding, and must FIT the sequence.
        # The upstream default is 25, but window_in here is 24: a 25-wide moving average over a
        # 24-step series makes the "trend" the whole-window mean and the seasonal residual
        # ~nothing, collapsing the decomposition that is Autoformer's entire point. Default to
        # the largest odd kernel <= half the window, and clamp any user value the same way.
        if moving_avg is None:
            moving_avg = max(3, window_in // 2)
        kernel = moving_avg if moving_avg % 2 == 1 else moving_avg + 1
        max_kernel = window_in if window_in % 2 == 1 else window_in - 1
        kernel = max(3, min(kernel, max_kernel))
        self.moving_avg_kernel = kernel
        # decomposition used for the decoder init (upstream: self.decomp)
        self.decomp = series_decomp(kernel)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, d_ff,
                    moving_avg=kernel, dropout=dropout, activation="gelu",
                )
                for _ in range(n_layers_enc)
            ],
            norm_layer=my_Layernorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(
                        AutoCorrelation(True, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads,
                    ),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, factor, attention_dropout=dropout,
                                        output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, n_features_target, d_ff,
                    moving_avg=kernel, dropout=dropout, activation="gelu",
                )
                for _ in range(n_layers_dec)
            ],
            norm_layer=my_Layernorm(d_model),
            projection=nn.Linear(d_model, n_features_target, bias=True),
        )

    def forward(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target=None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B, T_in, F_in = x_in.shape
        T_out = self.window_out
        F_tgt = self.n_features_target
        device = x_in.device

        # Build encoder input
        feats = [x_in, mask_in.float()]
        if self.static_to_encoder:
            feats.append(static.unsqueeze(1).expand(-1, T_in, -1))
        x_enc = torch.cat(feats, dim=-1)  # (B, T_in, enc_in)

        label_len = min(self.label_len, T_in)      # guard: label_len may exceed the history

        # Null temporal marks (Autoformer accepts them but we don't use them)
        mark_zeros_enc = torch.zeros(B, T_in,  1, device=device)
        mark_zeros_dec = torch.zeros(B, label_len + T_out, 1, device=device)

        # ------------------------------------------------------------------ decoder init
        # Autoformer's decoder is initialised from the DECOMPOSITION of the observed window:
        # the last `label_len` steps supply a real seasonal residual and a real trend, and only
        # the forecast horizon is filled with placeholders (zeros for seasonal, the window mean
        # for trend). Feeding an all-zero seasonal_init instead makes the decoder input
        # identically zero for every sample — and because `_Embed` is a bias-free Conv1d with no
        # positional term, exact zeros propagate through, so AutoCorrelation resolves the same
        # lags with the same weights for every patient and the decoder collapses to an
        # input-independent smoothing of the encoder output. That was the bug: the trained model
        # correlated 0.93 with simply broadcasting the input-window mean.
        # target-space view of the input (use the real target indices, not the first F_tgt cols)
        if F_in >= F_tgt:
            x_tgt = x_in.index_select(-1, self.target_idx)          # (B, T_in, F_tgt)
        else:
            x_tgt = torch.cat([x_in, torch.zeros(B, T_in, F_tgt - F_in, device=device)], -1)

        # mask-aware window mean: this data is 81-89 % imputed, so an unmasked mean is dominated
        # by fill values. Fall back to the plain mean where a channel is never observed.
        m_tgt = (mask_in.index_select(-1, self.target_idx).float() if F_in >= F_tgt
                 else torch.cat([mask_in.float(),
                                 torch.zeros(B, T_in, F_tgt - F_in, device=device)], -1))
        obs_n = m_tgt.sum(dim=1, keepdim=True)
        mean_obs = (x_tgt * m_tgt).sum(dim=1, keepdim=True) / obs_n.clamp(min=1.0)
        mean_all = x_tgt.mean(dim=1, keepdim=True)
        mean = torch.where(obs_n > 0, mean_obs, mean_all).expand(-1, T_out, -1)

        seasonal_obs, trend_obs = self.decomp(x_tgt[:, -label_len:, :])
        seasonal_init = torch.cat(
            [seasonal_obs, torch.zeros(B, T_out, F_tgt, device=device)], dim=1)
        trend_init = torch.cat([trend_obs, mean], dim=1)   # (B, label+T_out, F_tgt)

        # Encode
        enc_out = self.enc_embedding(x_enc, mark_zeros_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        # Decode
        dec_out = self.dec_embedding(seasonal_init, mark_zeros_dec)
        seasonal_part, trend_part = self.decoder(
            dec_out, enc_out, x_mask=None, cross_mask=None, trend=trend_init
        )
        out = trend_part + seasonal_part   # (B, label_len + T_out, F_tgt)
        return out[:, -T_out:, :]          # (B, T_out, F_tgt)
