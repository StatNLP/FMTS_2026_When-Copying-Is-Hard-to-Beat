"""
harmonize/models/chronos.py

Chronos / Chronos-Bolt pretrained time-series foundation model as a forecast
baseline (Ansari et al., 2024). Wraps `amazon/chronos-bolt-*` behind the uniform
`ForecastModel` interface so it slots into Stage-5 eval next to the small
dataset-specific models.

Design (see docs §9.6/§24 + project_fm_baselines handover)
----------------------------------------------------------
- **Univariate.** Chronos forecasts one series at a time, so we restrict to the
  target feature subset and run each of the F_target channels independently:
  the (B, T_in, F) target slice is reshaped to (B*F, T_in), pushed through the
  pipeline in one batched `predict_quantiles` call, and the (B*F, T_out) result
  is folded back to (B, T_out, F).
- **Scale.** Chronos-Bolt applies an internal per-series instance norm, so it is
  affine-equivariant: feeding the z-scored Stage-3 context yields a forecast in
  the same z-score space. The wrapper therefore stays in z-score units like every
  other model, and `eval_forecast` de-normalises downstream — no `norm_stats`
  coupling needed here.
- **Missingness.** The H5 windows arrive 0-filled at unobserved steps
  (nan_to_zero_in). We re-insert NaN at the `mask_in`=False positions so Chronos's
  scaler sees the true observed series (it ignores NaN); fully-missing channels
  fall back to the z-score mean (0) → climatological forecast.
- **Zero-shot.** No trainable weights here (a frozen pretrained pipeline + a dummy
  parameter to keep the generic training loop happy). Fine-tuning is a separate
  follow-up; Stage-5 eval builds this model directly via the `--fm` path (no
  checkpoint).
- Static features and teacher forcing are ignored (Chronos has no side inputs and
  is non-autoregressive over our horizon).
"""

from __future__ import annotations

import torch

from .base import ForecastModel


class ChronosForecast(ForecastModel):
    """Zero-shot Chronos / Chronos-Bolt forecaster (univariate, per-channel).

    Hyperparameters:
        chronos_id: HF model id, e.g. ``amazon/chronos-bolt-{tiny,mini,small,base}``
            (Chronos-Bolt, default ``base`` ≈ 205M) or ``amazon/chronos-t5-*``.
        point:      point-forecast reduction over the predictive distribution —
            ``"median"`` (0.5 quantile, MAE-optimal, default) or ``"mean"``
            (MSE-optimal).
        dtype:      pipeline compute dtype (``"float32"`` default; ``"bfloat16"``
            for faster GPU inference).
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
        chronos_id: str = "amazon/chronos-bolt-base",
        point: str = "median",
        dtype: str = "float32",
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        if point not in ("median", "mean"):
            raise ValueError(f"point must be 'median' or 'mean', got {point!r}")
        self.chronos_id = chronos_id
        self.point = point
        self.dtype = dtype
        # Keep the generic training loop / optimiser happy without exposing any
        # Chronos weight (this is a zero-shot baseline).
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        # Lazily loaded pipeline (NOT an nn.Module → not registered as a child,
        # so it stays out of state_dict / .to() / the optimiser). Boxed in a list
        # so nn.Module.__setattr__ never tries to register it.
        self._pipe_box: list = []

    # ------------------------------------------------------------------
    def _pipeline(self, device: torch.device):
        if not self._pipe_box:
            from chronos import BaseChronosPipeline
            dtype = getattr(torch, self.dtype)
            dev = "cuda" if device.type == "cuda" else "cpu"
            self._pipe_box.append(
                BaseChronosPipeline.from_pretrained(
                    self.chronos_id, device_map=dev, dtype=dtype,
                )
            )
        return self._pipe_box[0]

    # ------------------------------------------------------------------
    # Representation extraction for frozen-backbone classification heads
    # (§9.7/§24.5): expose the pretrained encoder hidden state so a small head
    # can be trained on the FM's representation. d_model = inner T5 width.
    def repr_dim(self, device: torch.device) -> int:
        return int(self._pipeline(device).model.config.d_model)

    @torch.no_grad()
    def encode(self, x_in, mask_in, static):
        """(B, T_in, F_in) → (B, F, d_model): per-channel pretrained-encoder
        representation (mean-pooled over patches), one token per target channel.
        FrozenBackboneHead then mean-pools over the channel tokens."""
        B = x_in.shape[0]; F = int(self.target_idx.numel())
        ctx = x_in.index_select(dim=-1, index=self.target_idx)
        m   = mask_in.index_select(dim=-1, index=self.target_idx).bool()
        ctx = torch.where(m, ctx, torch.full_like(ctx, float("nan")))
        ctx = ctx.permute(0, 2, 1).reshape(B * F, self.window_in)
        all_missing = torch.isnan(ctx).all(dim=1)
        if all_missing.any():
            ctx[all_missing] = 0.0
        inner = self._pipeline(x_in.device).model
        # inner.encode does NOT move inputs to the model's device (unlike the
        # high-level predict API) → feed on x_in.device to avoid a cpu/cuda mismatch.
        hid, *_ = inner.encode(context=ctx.float().to(x_in.device))  # (B*F, n_patch, d_model)
        z = hid.mean(dim=1).reshape(B, F, -1)                        # (B, F, d_model)
        return z.to(device=x_in.device, dtype=x_in.dtype)

    @torch.no_grad()
    def forward(
        self,
        x_in: torch.Tensor,         # (B, T_in, F_in)  z-scored, 0-filled
        mask_in: torch.Tensor,      # (B, T_in, F_in)  bool — observed positions
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        B = x_in.shape[0]
        F = int(self.target_idx.numel())

        # target-channel context, with true gaps restored as NaN for the scaler
        ctx = x_in.index_select(dim=-1, index=self.target_idx)          # (B,T_in,F)
        m   = mask_in.index_select(dim=-1, index=self.target_idx).bool()
        ctx = torch.where(m, ctx, torch.full_like(ctx, float("nan")))
        ctx = ctx.permute(0, 2, 1).reshape(B * F, self.window_in)        # (B*F, T_in)

        # fully-missing series → z-score mean (0) so Chronos returns a flat baseline
        all_missing = torch.isnan(ctx).all(dim=1)
        if all_missing.any():
            ctx[all_missing] = 0.0

        pipe = self._pipeline(x_in.device)
        q, mean = pipe.predict_quantiles(
            ctx.float().cpu(), prediction_length=self.window_out,
            quantile_levels=[0.5],
        )
        pred = mean if self.point == "mean" else q[..., 0]               # (B*F, T_out)

        pred = pred.reshape(B, F, self.window_out).permute(0, 2, 1)      # (B,T_out,F)
        pred = torch.nan_to_num(pred, nan=0.0)
        return pred.to(device=x_in.device, dtype=x_in.dtype)
