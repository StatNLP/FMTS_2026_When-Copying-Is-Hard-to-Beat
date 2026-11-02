"""
harmonize/models/mamba.py

Selective state-space (Mamba) sequence-to-sequence forecaster — the
*dynamic / state-space* architecture family for the model comparison (§9.2/R1).

A Mamba block is a learnable linear dynamical system

    h_t = A(Δ_t) · h_{t-1} + B(x_t) · x_t        (state recurrence)
    y_t = C(x_t) · h_t      + D · x_t            (readout)

where the parameters Δ, B, C are *input-dependent* (the "selective" part), so
the model can modulate its own dynamics over time — the natural recurrent
counterpart to the GRU/LSTM cell, with linear-time inference.

To slot into the identical training loop and Stage-5 eval as `RNNForecast` and
`DenseAutoregressive`, the model keeps the *same* encoder-decoder / IMS contract:

  Encoder: the Mamba stack runs over `[x_in ‖ mask_in ‖ static_broadcast]` and
           retains its final per-layer SSM state (and causal-conv cache).
  Decoder: the *same* Mamba stack, seeded from that state, rolls out the 24 h
           forecast one step at a time (IMS), feeding back its own prediction
           (student) or the ground truth (teacher) — same teacher-forcing
           schedule and exogenous-medication handling as the other AR models.

The selective scan is implemented as an explicit per-timestep recurrence in pure
PyTorch (no custom CUDA kernels / no `mamba-ssm` dependency). With windows of
~24 steps the Python-level loop is negligible, and the same `step()` primitive
serves both the parallel encoder pass and the recurrent decoder rollout.

`--model mamba`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ForecastModel


class _RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


class _MambaBlock(nn.Module):
    """A single selective-SSM block (Mamba-1 style) with a unified per-step API.

    State carried between timesteps is ``(conv_cache, ssm_state)``:
        conv_cache: (B, d_inner, d_conv-1)  recent inputs for the causal conv1d
        ssm_state:  (B, d_inner, d_state)   the latent dynamical state h_t
    """

    def __init__(
        self,
        d_model: int,
        *,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_inner = expand * d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.dt_rank = dt_rank or max(1, math.ceil(d_model / 16))

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # depthwise causal conv over the inner channels
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            groups=self.d_inner, padding=d_conv - 1, bias=True,
        )
        # input-dependent (Δ, B, C)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # Canonical Mamba Δ initialisation: scale dt_proj's weights and set its
        # bias so that softplus(bias) ≈ a small timestep drawn from
        # [dt_min, dt_max]. Without this the default zero bias gives an initial
        # Δ ≈ softplus(0) ≈ 0.69 — far too large, which makes the SSM recurrence
        # unstable early in training (loss bumps / divergence at higher lr).
        dt_min, dt_max, dt_init_floor = 1e-3, 1e-1, 1e-4
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp_min(dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))   # inverse softplus
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)

        # A is parameterised in log-space and kept negative (stable decay).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def init_state(self, batch: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
        conv_cache = torch.zeros(batch, self.d_inner, self.d_conv - 1, device=device, dtype=dtype)
        ssm_state = torch.zeros(batch, self.d_inner, self.d_state, device=device, dtype=dtype)
        return conv_cache, ssm_state

    def step(
        self,
        x_t: torch.Tensor,                       # (B, d_model)
        state: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        conv_cache, ssm_state = state
        xz = self.in_proj(x_t)                    # (B, 2*d_inner)
        x, z = xz.chunk(2, dim=-1)                # each (B, d_inner)

        # causal depthwise conv via the rolling cache, then SiLU
        conv_in = torch.cat([conv_cache, x.unsqueeze(-1)], dim=-1)   # (B, d_inner, d_conv)
        x = (conv_in * self.conv1d.weight.squeeze(1)).sum(-1) + self.conv1d.bias
        new_conv_cache = conv_in[..., 1:]
        x = F.silu(x)

        # input-dependent Δ, B, C
        dbl = self.x_proj(x)                      # (B, dt_rank + 2*d_state)
        dt, B_t, C_t = torch.split(dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))      # (B, d_inner)

        A = -torch.exp(self.A_log.float())        # (d_inner, d_state)
        dA = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))          # (B, d_inner, d_state)
        dB = delta.unsqueeze(-1) * B_t.unsqueeze(1)                   # (B, d_inner, d_state)
        new_ssm_state = dA * ssm_state + dB * x.unsqueeze(-1)
        y = (new_ssm_state * C_t.unsqueeze(1)).sum(-1)               # (B, d_inner)
        y = y + self.D * x
        y = y * F.silu(z)
        return self.out_proj(y), (new_conv_cache, new_ssm_state)


class MambaForecast(ForecastModel):
    """Mamba (selective state-space) encoder–decoder forecaster (`--model mamba`).

    Hyperparameters:
        d_model:      block width (default 128)
        n_layers_enc: depth of the shared Mamba stack (default 2)
        dropout:      residual dropout between blocks
        d_state:      SSM latent size N (default 16)
        d_conv:       causal-conv kernel (default 4)
        expand:       inner-width expansion factor (default 2)
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
        d_model: int = 128,
        n_layers_enc: int = 2,
        dropout: float = 0.1,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        static_to_encoder: bool = True,
        exogenous_idx: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.static_to_encoder = static_to_encoder
        self.n_layers = n_layers_enc
        enc_in_dim = 2 * n_features_in + (n_features_static if static_to_encoder else 0)

        # Two embeddings (encoder reads inputs+mask+static, decoder reads its own
        # F_target prediction) feed a *shared* Mamba stack whose state carries
        # across the encode → decode boundary — the SSM analogue of the RNN's
        # shared hidden state.
        self.enc_embed = nn.Linear(enc_in_dim, d_model)
        self.dec_embed = nn.Linear(n_features_target, d_model)
        self.blocks = nn.ModuleList(
            _MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand)
            for _ in range(n_layers_enc)
        )
        self.norms = nn.ModuleList(_RMSNorm(d_model) for _ in range(n_layers_enc))
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, n_features_target)

        self.start_token = nn.Parameter(torch.zeros(1, 1, n_features_target))
        nn.init.normal_(self.start_token, std=0.02)

        self.register_buffer(
            "exogenous_idx",
            torch.tensor(exogenous_idx or [], dtype=torch.long),
            persistent=False,
        )

    def _run_step(
        self,
        h: torch.Tensor,                              # (B, d_model)  embedded input
        states: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        """One timestep through the stacked blocks (pre-norm residual), in place
        on `states`. Returns the top-of-stack representation (B, d_model)."""
        for i, (block, norm) in enumerate(zip(self.blocks, self.norms)):
            out, states[i] = block.step(norm(h), states[i])
            h = h + self.dropout(out)
        return h

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
        device, dtype = x_in.device, x_in.dtype

        states = [blk.init_state(B, device, dtype) for blk in self.blocks]

        # --- encoder: consume the input window, retaining the carried state ---
        feats = [x_in, mask_in.float()]
        if self.static_to_encoder:
            feats.append(static.unsqueeze(1).expand(-1, T_in, -1))
        enc_in = self.enc_embed(torch.cat(feats, dim=-1))   # (B, T_in, d_model)
        for t in range(T_in):
            self._run_step(enc_in[:, t, :], states)

        # --- decoder: IMS rollout, carrying the SSM state forward ---
        use_teacher = x_target is not None and teacher_forcing_ratio > 0.0
        prev = self.start_token.expand(B, -1, -1).squeeze(1)   # (B, F_target)
        outputs: list[torch.Tensor] = []
        for t in range(self.window_out):
            h = self._run_step(self.dec_embed(prev), states)
            step_pred = self.out_proj(h).unsqueeze(1)          # (B, 1, F_target)
            outputs.append(step_pred)

            if use_teacher and torch.rand((), device=device).item() < teacher_forcing_ratio:
                next_in = x_target[:, t:t + 1, :]
                nan_mask = torch.isnan(next_in)
                if nan_mask.any():
                    next_in = torch.where(nan_mask, step_pred, next_in)
            else:
                next_in = step_pred

            # exogenous (medication) channels fed from ground truth (z=0 if not given)
            if self.exogenous_idx.numel() and x_target is not None:
                exo = self.exogenous_idx
                true_exo = torch.nan_to_num(
                    x_target[:, t:t + 1, :].index_select(-1, exo), nan=0.0)
                next_in = next_in.clone()
                next_in[:, :, exo] = true_exo

            prev = next_in.squeeze(1)

        return torch.cat(outputs, dim=1)                       # (B, T_out, F_target)