"""
harmonize/models/strats.py

STraTS-style set-function encoder + iterative autoregressive (IMS) decoder —
the "triplet" arm of Paper-1, ported into the unified ForecastModel interface.

Instead of the dense hourly grid, the input window is treated as an unordered
SET of observed measurements, each encoded as a triplet (time, variable, value):

    token_i = CVE_value(value_i) + CVE_time(time_i) + Embedding_var(variable_i)

where CVE (Continuous Value Embedding, Tipirneni & Reddy 2022 / STraTS) is a tiny
1 -> hidden -> d_model MLP with a tanh nonlinearity that embeds a single scalar
without binning. A learnable static/demographic token is prepended. A standard
Transformer encoder then attends over the set (padding-masked), producing a set
memory the decoder cross-attends to.

The triplets are reconstructed on the fly from the dense Stage-3 tensors: a
position (t, f) is a genuine observation wherever `mask_in[t, f]` is True (the
mask marks pre-imputation observations), with value `x_in[t, f]`. This keeps the
same HDF5 datasets, training loop and masked-loss as every other model — only the
encoder differs.

Decoder: the SAME iterative AR decoder as DenseAutoregressive (one step at a time,
own-prediction or teacher feedback), so student/teacher forcing schedules apply.
In Paper-1 terms this is the triplet IMS + student-forcing + backprop-through-
predictions variant — the strongest set-function configuration.

Reference:
   "Early Prediction of Causes (not Effects) ...", MLHC 2024 (triplet models).
   Tipirneni & Reddy, "Self-supervised Transformer for Sparse and Irregularly
   Sampled Multivariate Clinical Time-Series" (STraTS), ACM TKDD 2022.
   Original code:
    10_scripts/archive/mlhc_2024_prediction_of_causes-main/
        eicu_experiments/model_files/mod02_triplet_IMS_TF.py (CVE/Attention)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ForecastModel
from .dense_autoregressive import _PositionalEncoding


class _CVE(nn.Module):
    """Continuous Value Embedding: embed one scalar -> d_model via a 1->h->d MLP.

    x: (..., ) scalars  ->  (..., d_model).
    """

    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(1, hidden)
        self.w2 = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(torch.tanh(self.w1(x.unsqueeze(-1))))


class StratsForecast(ForecastModel):
    """Set-function (triplet) encoder + iterative AR decoder.

    Hyperparameters:
        d_model:      hidden dim (default 128)
        n_heads:      attention heads (default 4)
        d_ff:         feed-forward dim (default 256)
        n_layers_enc / n_layers_dec: depths (default 2 / 2)
        dropout:      dropout (default 0.1)
        cve_hidden:   hidden units of the CVE scalar embedders (default 32)
        max_tokens:   cap on observed triplets per window (default 768). On the
                      rare overflow the MOST RECENT observations are kept.
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
        n_layers_dec: int = 2,
        dropout: float = 0.1,
        cve_hidden: int = 32,
        max_tokens: int = 768,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.max_tokens = int(max_tokens)

        # --- set-function input embedding ---
        self.value_cve = _CVE(d_model, cve_hidden)
        self.time_cve  = _CVE(d_model, cve_hidden)
        self.var_embed = nn.Embedding(n_features_in, d_model)
        self.static_proj = nn.Linear(n_features_static, d_model)
        # learnable marker so the static/demographic token is distinguishable
        self.static_type = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.static_type, std=0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # --- iterative AR decoder (same as DenseAutoregressive) ---
        self.pos_enc   = _PositionalEncoding(d_model, max_len=window_in + window_out + 8)
        self.tgt_embed = nn.Linear(n_features_target, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder  = nn.TransformerDecoder(dec_layer, num_layers=n_layers_dec)
        self.out_proj = nn.Linear(d_model, n_features_target)

        self.start_token = nn.Parameter(torch.zeros(1, 1, n_features_target))
        nn.init.normal_(self.start_token, std=0.02)

    # ------------------------------------------------------------------
    def _encode_set(
        self,
        x_in: torch.Tensor,      # (B, T_in, F_in)
        mask_in: torch.Tensor,   # (B, T_in, F_in) bool
        static: torch.Tensor,    # (B, F_static)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pack observed (t, f, value) triplets into a padded set, then encode.

        Returns (memory, mem_pad) where memory is (B, 1+S, d_model) and mem_pad
        is the key-padding mask (B, 1+S), True at padded positions.
        """
        B, T, F = x_in.shape
        device  = x_in.device
        L = T * F

        flat_val = x_in.reshape(B, L)
        flat_obs = mask_in.reshape(B, L).bool()

        # time / variable index of every flat position (row-major: pos = t*F + f)
        pos   = torch.arange(L, device=device)
        t_idx = (pos // F).to(x_in.dtype)        # (L,) hour 0..T-1
        v_idx = (pos % F).long()                 # (L,)

        # Priority for packing: observed first, and within observed the most
        # recent (largest t) first — so a max_tokens cap drops the oldest, not
        # the freshest, measurements. unobserved get strictly lower priority.
        prio = flat_obs.to(x_in.dtype) * (T + 1.0) + t_idx.unsqueeze(0)   # (B, L)
        order = torch.argsort(prio, dim=1, descending=True)              # (B, L)

        counts  = flat_obs.sum(1)                                        # (B,)
        max_obs = int(counts.max().clamp(min=1).item())
        if self.max_tokens > 0:
            max_obs = min(max_obs, self.max_tokens)
        sel = order[:, :max_obs]                                         # (B, S)

        val_sel = torch.gather(flat_val, 1, sel)                         # (B, S)
        valid   = torch.gather(flat_obs, 1, sel)                         # (B, S) bool
        t_sel   = t_idx[sel]                                             # (B, S)
        v_sel   = v_idx[sel]                                             # (B, S)

        return self._encode_tokens(val_sel, t_sel, v_sel, ~valid, static)

    # ------------------------------------------------------------------
    def _encode_tokens(
        self,
        val: torch.Tensor,     # (B, S)  z-scored values
        time: torch.Tensor,    # (B, S)  Δt
        var: torch.Tensor,     # (B, S)  long variable indices
        pad_in: torch.Tensor,  # (B, S)  bool, True = padded/ignore
        static: torch.Tensor,  # (B, F_static)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Embed a (possibly padded) set of triplets + a static token, then encode.

        Shared by the dense-reconstruction (`_encode_set`) and the raw-triplet
        (`StratsRawForecast`) front-ends. Returns (memory, mem_pad).
        """
        B = val.size(0)
        device = val.device
        tok = self.value_cve(val) + self.time_cve(time) + self.var_embed(var)  # (B,S,d)
        stat_tok = self.static_proj(static).unsqueeze(1) + self.static_type    # (B,1,d)
        tokens = torch.cat([stat_tok, tok], dim=1)                             # (B,1+S,d)
        pad = torch.cat(
            [torch.zeros(B, 1, dtype=torch.bool, device=device), pad_in], dim=1
        )                                                                     # static valid
        memory = self.encoder(tokens, src_key_padding_mask=pad)
        return memory, pad

    # ------------------------------------------------------------------
    def forward(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        memory, mem_pad = self._encode_set(x_in, mask_in, static)
        return self._decode(memory, mem_pad,
                            x_target=x_target,
                            teacher_forcing_ratio=teacher_forcing_ratio)

    # ------------------------------------------------------------------
    def _decode(
        self,
        memory: torch.Tensor,
        mem_pad: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device

        outputs: list[torch.Tensor] = []
        prefix = self.start_token.expand(B, -1, -1)   # (B, 1, F_target)
        use_teacher = x_target is not None and teacher_forcing_ratio > 0.0

        for t in range(self.window_out):
            tgt = self.tgt_embed(prefix) + self.pos_enc.pe[:, : prefix.size(1)]
            T_dec = tgt.size(1)
            causal = nn.Transformer.generate_square_subsequent_mask(T_dec).to(device)
            dec_out = self.decoder(tgt, memory, tgt_mask=causal,
                                   memory_key_padding_mask=mem_pad)
            step_pred = self.out_proj(dec_out[:, -1:, :])  # (B, 1, F_target)
            outputs.append(step_pred)

            if use_teacher and (
                torch.rand((), device=device).item() < teacher_forcing_ratio
            ):
                next_in = x_target[:, t : t + 1, :]
                nan_mask = torch.isnan(next_in)
                if nan_mask.any():
                    next_in = torch.where(nan_mask, step_pred, next_in)
            else:
                next_in = step_pred
            prefix = torch.cat([prefix, next_in], dim=1)

        return torch.cat(outputs, dim=1)  # (B, T_out, F_target)
