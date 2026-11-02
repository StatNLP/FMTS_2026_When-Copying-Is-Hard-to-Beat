"""
harmonize/models/dense_autoregressive.py

Paper-1 main model — dense encoder over the input window + iterative
autoregressive Transformer decoder. The decoder produces one timestep at a
time and (optionally) feeds its own predictions back as the next input.

Reference:
   "Early Prediction of Causes (not Effects) in Healthcare by Long-Term
    Clinical Time Series Forecasting", MLHC 2024.
   Original code:
    10_scripts/archive/mlhc_2024_prediction_of_causes-main/
        mimic_experiments/models/DenseAutoregressiveFull.py

Key differences vs. the original:
- Uniform ForecastModel interface (no `configs` namespace).
- Decoder cross-attention works in the d_model space (instead of c_out as in
  the original), with a final projection to F_target. Cleaner gradient flow.
- Student / teacher forcing is controlled by `teacher_forcing_ratio` in the
  forward, not by a separate `trainn` flag.
- Static features are concatenated to the encoder input as a per-step bias
  (broadcast along time). Set `static_to_encoder=False` to skip.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import ForecastModel


class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 256) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() *
                        (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class DenseAutoregressive(ForecastModel):
    """Dense Transformer encoder + iterative AR decoder.

    Hyperparameters:
        d_model:  hidden dim (default 128)
        n_heads:  attention heads (default 4)
        d_ff:     feed-forward dim (default 256)
        n_layers_enc / n_layers_dec: transformer depths (default 2/2)
        dropout:  dropout (default 0.1)
        static_to_encoder: broadcast static features into the encoder input
                          (default True)
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
        static_to_encoder: bool = True,
        exogenous_idx: list[int] | None = None,
        detach_rollout: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        # If True, student-forced predictions are fed back .detach()'d → no gradient
        # flows back through the AR rollout (the Staniek "backprop through predictions"
        # ablation: off = train each step independently, on = the default IMS+BP).
        self.detach_rollout = bool(detach_rollout)

        # Input embedding: concat [x_in, mask_in] over feature dim → Linear
        in_dim = 2 * n_features_in + (n_features_static if static_to_encoder else 0)
        self.input_proj = nn.Linear(in_dim, d_model)
        self.pos_enc    = _PositionalEncoding(d_model, max_len=window_in + window_out + 8)
        self.static_to_encoder = static_to_encoder

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers_enc)

        # Decoder: embed F_target → d_model, then stacked TransformerDecoderLayers
        self.tgt_embed = nn.Linear(n_features_target, d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers_dec)

        self.out_proj = nn.Linear(d_model, n_features_target)

        # Learnable "start token" for the AR decoder
        self.start_token = nn.Parameter(torch.zeros(1, 1, n_features_target))
        nn.init.normal_(self.start_token, std=0.02)

        # Partial teacher forcing: indices of EXOGENOUS target channels (e.g. medication
        # rates = physician decisions). During the AR rollout these are fed back from the
        # ground truth instead of the model's own prediction — the model forecasts the
        # physiological response to a *known* treatment, rather than the treatment itself.
        # Empty (default) ⇒ standard autoregressive model, no behaviour change.
        self.register_buffer(
            "exogenous_idx",
            torch.tensor(exogenous_idx or [], dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------
    # Embedding-space DA split (Paper-2 MIA defenses). See base.py.
    #   embed_inputs:          input_proj + positional encoding  → emb
    #   forward_from_embedding: transformer encoder + AR decoder  → prediction
    # The frozen embedding layer (input_proj) is bypassed when a precomputed
    # `emb` is supplied, so privacy fine-tuning only updates the encoder/decoder.
    supports_embedding_da = True

    def embed_inputs(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """Input embedding (B, T_in, d_model): the layer bypassed during privacy DA."""
        B, T_in, _ = x_in.shape
        feats = [x_in, mask_in.float()]
        if self.static_to_encoder:
            feats.append(static.unsqueeze(1).expand(-1, T_in, -1))
        x = torch.cat(feats, dim=-1)  # (B, T_in, 2*F_in [+F_static])
        return self.pos_enc(self.input_proj(x))

    def encode(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
    ) -> torch.Tensor:
        """Returns encoder memory (B, T_in, d_model)."""
        return self.encoder(self.embed_inputs(x_in, mask_in, static))

    # ------------------------------------------------------------------
    def forward_from_embedding(
        self,
        emb: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        tf_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run transformer encoder + AR decoder on a precomputed input embedding.

        `tf_target` (B, T_out, F_target): fully teacher-forced **parallel** decode in one
        pass — mathematically identical to the sequential AR rollout under
        teacher_forcing_ratio=1.0 (causal mask), ~T× faster. Used for fast training /
        DP-SGD. NaN target inputs are read as z=0 (the "not-given" convention), so it is
        not bit-identical to the sequential rollout (which fills NaN with its own pred)."""
        memory = self.encoder(emb)  # (B, T_in, d_model)
        if tf_target is not None:
            return self._decode_tf_parallel(memory, tf_target)
        return self._decode(memory, x_target=x_target,
                            teacher_forcing_ratio=teacher_forcing_ratio)

    def forward(
        self,
        x_in: torch.Tensor | None = None,
        mask_in: torch.Tensor | None = None,
        static: torch.Tensor | None = None,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
        emb: torch.Tensor | None = None,
        tf_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # `emb` routes directly to forward_from_embedding — lets functional_call
        # (torch.func) reach the embedding path for vectorised per-sample grads
        # (privacy/dpsgd.py). Otherwise: standard embed → encoder → AR decode.
        if emb is None:
            emb = self.embed_inputs(x_in, mask_in, static)
        return self.forward_from_embedding(
            emb, x_target=x_target, teacher_forcing_ratio=teacher_forcing_ratio,
            tf_target=tf_target,
        )

    # ------------------------------------------------------------------
    def _decode_tf_parallel(self, memory: torch.Tensor,
                            x_target: torch.Tensor) -> torch.Tensor:
        """Single-pass teacher-forced decode (B, T_out, F_target). Feeds
        [start_token, x_target[:, :-1]] through the decoder once with a causal mask —
        position i predicts from [start, x_0..x_{i-1}], exactly the sequential rollout
        under full teacher forcing but T× cheaper."""
        B = memory.size(0)
        tgt_in = torch.nan_to_num(x_target, nan=0.0)            # (B, T_out, F_target)
        seq = torch.cat([self.start_token.expand(B, -1, -1),
                         tgt_in[:, :-1, :]], dim=1)             # (B, T_out, F_target)
        tgt = self.tgt_embed(seq) + self.pos_enc.pe[:, : seq.size(1)]
        causal = nn.Transformer.generate_square_subsequent_mask(seq.size(1)).to(memory.device)
        dec_out = self.decoder(tgt, memory, tgt_mask=causal)
        return self.out_proj(dec_out)                          # (B, T_out, F_target)

    # ------------------------------------------------------------------
    def _decode(
        self,
        memory: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B = memory.size(0)
        device = memory.device

        outputs: list[torch.Tensor] = []
        # AR sequence prefix in F_target space — starts with the learnable token
        prefix = self.start_token.expand(B, -1, -1)   # (B, 1, F_target)

        use_teacher = (
            x_target is not None and teacher_forcing_ratio > 0.0
        )
        # Pre-compute the mask for the decoder (causal)
        # We embed sequentially, so each step we feed prefix-so-far.
        for t in range(self.window_out):
            tgt = self.tgt_embed(prefix) + self.pos_enc.pe[:, : prefix.size(1)]
            T_dec = tgt.size(1)
            causal = nn.Transformer.generate_square_subsequent_mask(T_dec).to(device)
            dec_out = self.decoder(tgt, memory, tgt_mask=causal)
            step_pred = self.out_proj(dec_out[:, -1:, :])  # (B, 1, F_target)
            outputs.append(step_pred)

            # Decide what to feed next: teacher (ground truth) or student (own pred)
            if use_teacher and (
                torch.rand((), device=device).item() < teacher_forcing_ratio
            ):
                next_in = x_target[:, t : t + 1, :]
                # Replace NaN target with own prediction (NaNs would corrupt the AR)
                nan_mask = torch.isnan(next_in)
                if nan_mask.any():
                    next_in = torch.where(nan_mask, step_pred, next_in)
            else:
                next_in = step_pred

            # Partial teacher forcing: overwrite exogenous channels (e.g. medications)
            # with their true values — physician decisions are known, not forecast.
            # "Not administered" arrives as NaN in x_target, but the model was trained
            # to read it as 0 in z-space (x_in fills not-given meds with z=0, ~99% of
            # med positions). Feed 0 there — NOT the model's own (untrained, since meds
            # are excluded from the loss) medication prediction, which corrupted the
            # rollout for the ~96% of med-hours that are unobserved.
            if self.exogenous_idx.numel() and x_target is not None:
                exo = self.exogenous_idx
                true_exo = x_target[:, t : t + 1, :].index_select(-1, exo)
                true_exo = torch.nan_to_num(true_exo, nan=0.0)
                next_in = next_in.clone()
                next_in[:, :, exo] = true_exo

            if self.detach_rollout:
                next_in = next_in.detach()   # cut the gradient through the AR rollout
            prefix = torch.cat([prefix, next_in], dim=1)

        return torch.cat(outputs, dim=1)  # (B, T_out, F_target)
