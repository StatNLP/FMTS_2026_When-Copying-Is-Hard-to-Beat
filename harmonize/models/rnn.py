"""
harmonize/models/rnn.py

Recurrent (GRU / LSTM) sequence-to-sequence forecaster — the recurrent
architecture family for the model comparison (§9.2/R1), the canonical
pre-transformer clinical time-series baseline that the zoo was missing.

Encoder: a GRU/LSTM over `[x_in ‖ mask_in ‖ static_broadcast]` produces a final
hidden state that summarises the 24 h input window. Decoder: the same recurrent
cell, initialised from that hidden state, rolls out the 24 h forecast one step at
a time (IMS), feeding back its own prediction (student) or the ground truth
(teacher) — the *same* teacher-forcing schedule and exogenous-medication handling
as `DenseAutoregressive`, so it slots into the identical training loop and Stage-5
eval. Only the encoder/decoder cells differ (recurrent vs. attention).

`--model gru` / `--model lstm`.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ForecastModel


class RNNForecast(ForecastModel):
    """GRU/LSTM encoder–decoder forecaster with an iterative AR (IMS) rollout.

    Hyperparameters:
        rnn_type:  "gru" or "lstm"
        d_model:   hidden size (default 128)
        n_layers_enc: recurrent depth — shared by encoder and decoder so the
                   encoder's final hidden state initialises the decoder (default 2)
        dropout:   inter-layer dropout (only active for depth > 1)
        static_to_encoder: broadcast static features into the encoder input (default True)
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
        rnn_type: str = "gru",
        d_model: int = 128,
        n_layers_enc: int = 2,
        dropout: float = 0.1,
        static_to_encoder: bool = True,
        exogenous_idx: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in ("gru", "lstm"):
            raise ValueError(f"rnn_type must be 'gru' or 'lstm', got {rnn_type!r}")
        Cell = nn.GRU if self.rnn_type == "gru" else nn.LSTM
        self.static_to_encoder = static_to_encoder
        self.n_layers = n_layers_enc
        in_dim = 2 * n_features_in + (n_features_static if static_to_encoder else 0)
        rnn_dropout = dropout if n_layers_enc > 1 else 0.0

        # Encoder and decoder share depth + hidden size so the encoder's final
        # hidden state directly seeds the decoder.
        self.encoder = Cell(in_dim, d_model, num_layers=n_layers_enc,
                            batch_first=True, dropout=rnn_dropout)
        self.decoder = Cell(n_features_target, d_model, num_layers=n_layers_enc,
                            batch_first=True, dropout=rnn_dropout)
        self.out_proj = nn.Linear(d_model, n_features_target)

        self.start_token = nn.Parameter(torch.zeros(1, 1, n_features_target))
        nn.init.normal_(self.start_token, std=0.02)

        # Partial teacher forcing for exogenous (medication) target channels — fed
        # from ground truth during the rollout, like DenseAutoregressive. Empty by
        # default ⇒ standard autoregressive behaviour.
        self.register_buffer(
            "exogenous_idx",
            torch.tensor(exogenous_idx or [], dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B, T_in, _ = x_in.shape
        device = x_in.device

        feats = [x_in, mask_in.float()]
        if self.static_to_encoder:
            feats.append(static.unsqueeze(1).expand(-1, T_in, -1))
        enc_in = torch.cat(feats, dim=-1)          # (B, T_in, in_dim)
        _, hidden = self.encoder(enc_in)           # GRU: h_n; LSTM: (h_n, c_n)

        use_teacher = x_target is not None and teacher_forcing_ratio > 0.0
        prev = self.start_token.expand(B, -1, -1)  # (B, 1, F_target)
        outputs: list[torch.Tensor] = []
        for t in range(self.window_out):
            out, hidden = self.decoder(prev, hidden)        # (B, 1, d_model)
            step_pred = self.out_proj(out)                  # (B, 1, F_target)
            outputs.append(step_pred)

            if use_teacher and torch.rand((), device=device).item() < teacher_forcing_ratio:
                next_in = x_target[:, t:t + 1, :]
                nan_mask = torch.isnan(next_in)
                if nan_mask.any():
                    next_in = torch.where(nan_mask, step_pred, next_in)
            else:
                next_in = step_pred

            # exogenous (medication) channels fed from ground truth (z=0 for not-given)
            if self.exogenous_idx.numel() and x_target is not None:
                exo = self.exogenous_idx
                true_exo = torch.nan_to_num(x_target[:, t:t + 1, :].index_select(-1, exo), nan=0.0)
                next_in = next_in.clone()
                next_in[:, :, exo] = true_exo

            prev = next_in

        return torch.cat(outputs, dim=1)           # (B, T_out, F_target)


class GRUForecast(RNNForecast):
    """GRU encoder–decoder forecaster (`--model gru`)."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["rnn_type"] = "gru"
        super().__init__(*args, **kwargs)


class LSTMForecast(RNNForecast):
    """LSTM encoder–decoder forecaster (`--model lstm`)."""

    def __init__(self, *args, **kwargs) -> None:
        kwargs["rnn_type"] = "lstm"
        super().__init__(*args, **kwargs)
