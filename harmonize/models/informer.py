"""
harmonize/models/informer.py

Informer — Paper-1 comparison baseline.

Wraps the Informer architecture (Zhou et al., AAAI 2021) in the ForecastModel
interface. Key features:
  - ProbSparse self-attention: O(L log L) encoder attention
  - Distillation: ConvLayer halves sequence length between encoder layers
  - Decoder: half-observed initialization (last label_len encoder steps) + zeros
    for prediction steps; standard autoregressive Informer decoder

For our clinical time-series windows (T_in = T_out = 24), ProbSparse attention
and regular full attention are equivalent — the sequence is too short for the
O(L log L) advantage to matter. The architecture is nonetheless faithful to
the paper's baseline.

Encoder input convention: same as DenseAutoregressive — concat [x_in ‖ mask_in
‖ static_broadcast] → embed to d_model.

Decoder initialization:
  x_dec = cat([x_in[:, -label_len:, :F_target], zeros(B, T_out, F_target)], 1)

Reference:
  Zhou, H. et al. "Informer: Beyond Efficient Transformer for Long Sequence
  Time-Series Forecasting." AAAI 2021.
  Original code: 10_scripts/archive/.../models/Informer.py (adapted).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ForecastModel
from .layers.SelfAttention_Family import ProbAttention, FullAttention, AttentionLayer
from .layers.Transformer_EncDec import (
    Encoder, EncoderLayer, ConvLayer,
    Decoder, DecoderLayer,
)
# DataEmbedding_wo_temp = value + POSITIONAL embedding (no temporal marks, which this project
# does not have). NOT `_wo_pos_temp`: that drops the positional term too, and since the decoder
# is seeded with zeros over the forecast horizon and TokenEmbedding is a bias-free Conv1d, 22 of
# the 24 forecast tokens then become byte-identical zero vectors — the decoder cannot tell hour 1
# from hour 22. (Autoformer legitimately uses the no-positional variant: AutoCorrelation is
# position-agnostic by design. Informer's reference implementation uses positional encoding.)
from .layers.Embed import DataEmbedding_wo_temp as _Embed


class InformerForecast(ForecastModel):
    """Informer time-series forecasting model.

    Hyperparameters:
        d_model:       hidden dim (default 128)
        n_heads:       attention heads (default 4)
        d_ff:          feed-forward dim (default 256)
        n_layers_enc:  encoder depth (default 2)
        n_layers_dec:  decoder depth (default 1)
        dropout:       dropout (default 0.1)
        factor:        ProbSparse attention top-k factor (default 5)
        distil:        use distillation (ConvLayer halving) between enc layers (True)
        static_to_encoder: broadcast static into encoder input (True)
        label_len:     observed context fed to decoder (default = window_in // 2)
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
        factor: int = 5,
        distil: bool = True,
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

        enc_in = 2 * n_features_in + (n_features_static if static_to_encoder else 0)
        dec_in = n_features_target

        self.enc_embedding = _Embed(enc_in, d_model, dropout=dropout)
        self.dec_embedding = _Embed(dec_in, d_model, dropout=dropout)

        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        ProbAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, dropout=dropout, activation="gelu",
                )
                for _ in range(n_layers_enc)
            ],
            [ConvLayer(d_model) for _ in range(n_layers_enc - 1)] if distil else None,
            norm_layer=nn.LayerNorm(d_model),
        )

        self.decoder = Decoder(
            [
                DecoderLayer(
                    AttentionLayer(
                        FullAttention(True, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    AttentionLayer(
                        FullAttention(False, factor, attention_dropout=dropout,
                                      output_attention=False),
                        d_model, n_heads,
                    ),
                    d_model, d_ff, dropout=dropout, activation="gelu",
                )
                for _ in range(n_layers_dec)
            ],
            norm_layer=nn.LayerNorm(d_model),
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
        T_out  = self.window_out
        F_tgt  = self.n_features_target
        device = x_in.device

        # Encoder input
        feats = [x_in, mask_in.float()]
        if self.static_to_encoder:
            feats.append(static.unsqueeze(1).expand(-1, T_in, -1))
        x_enc = torch.cat(feats, dim=-1)

        mark_enc = torch.zeros(B, T_in,                     1, device=device)
        mark_dec = torch.zeros(B, self.label_len + T_out,   1, device=device)

        # Decoder init: last label_len observed input steps (target features) + zeros
        obs_steps = min(self.label_len, T_in)
        if F_in >= F_tgt:
            # select the REAL target columns, not the first F_tgt (identical while
            # target == input, wrong the moment targets are a subset/reordering, e.g.
            # stage3_make_dataset.py --predict-interventions)
            dec_obs = x_in[:, -obs_steps:, :].index_select(-1, self.target_idx)
        else:
            pad = torch.zeros(B, obs_steps, F_tgt - F_in, device=device)
            dec_obs = torch.cat([x_in[:, -obs_steps:, :], pad], dim=-1)
        dec_zeros = torch.zeros(B, T_out, F_tgt, device=device)
        x_dec = torch.cat([dec_obs, dec_zeros], dim=1)   # (B, label_len + T_out, F_tgt)

        enc_out = self.enc_embedding(x_enc, mark_enc)
        enc_out, _ = self.encoder(enc_out, attn_mask=None)

        dec_out = self.dec_embedding(x_dec, mark_dec)
        dec_out = self.decoder(dec_out, enc_out, x_mask=None, cross_mask=None)

        return dec_out[:, -T_out:, :]   # (B, T_out, F_tgt)
