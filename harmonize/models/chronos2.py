"""
harmonize/models/chronos2.py

Chronos-2 (`amazon/chronos-2`) pretrained time-series foundation model as a
**natively multivariate** forecast baseline. Unlike Chronos-Bolt (univariate,
run per channel — `chronos.py`), Chronos-2 shares information across the target
channels of each window in a single forward, so it is the *fair* multivariate
comparator (the role originally intended for Moirai; Chronos-2 supersedes it and,
unlike uni2ts, runs in the py313_tsf env).

Same conventions as `chronos.py`: zero-shot, z-score in / z-score out (Chronos-2
applies its own per-series instance norm), NaN re-inserted at mask=False so the
scaler sees true gaps, fully-missing channels fall back to the z-score mean (0).
Static features and teacher forcing are ignored. The pipeline can additionally
consume past/future covariates — not used here (kept a like-for-like forecast).
"""

from __future__ import annotations

import torch

from .base import ForecastModel


class Chronos2Forecast(ForecastModel):
    """Zero-shot Chronos-2 forecaster (natively multivariate over target channels).

    Hyperparameters:
        chronos_id: HF model id (default ``amazon/chronos-2``).
        point:      ``"median"`` (0.5 quantile, default) or ``"mean"``.
        dtype:      pipeline compute dtype (``"float32"`` default).
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
        chronos_id: str = "amazon/chronos-2",
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
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._pipe_box: list = []

    # ------------------------------------------------------------------
    def _pipeline(self, device: torch.device):
        if not self._pipe_box:
            from chronos import Chronos2Pipeline
            dtype = getattr(torch, self.dtype)
            dev = "cuda" if device.type == "cuda" else "cpu"
            self._pipe_box.append(
                Chronos2Pipeline.from_pretrained(
                    self.chronos_id, device_map=dev, dtype=dtype,
                )
            )
        return self._pipe_box[0]

    # ------------------------------------------------------------------
    # Representation extraction for frozen-backbone classification heads (§9.7).
    def repr_dim(self, device: torch.device) -> int:
        return int(self._pipeline(device).model.config.d_model)

    @torch.no_grad()
    def encode(self, x_in, mask_in, static):
        """(B, T_in, F_in) → (B, F, d_model): Chronos-2's multivariate encoder
        representation, one mean-pooled token per target channel (channels of a
        window grouped via group_ids so cross-channel info enters the encoding)."""
        import math
        B = x_in.shape[0]; F = int(self.target_idx.numel())
        ctx = x_in.index_select(dim=-1, index=self.target_idx)
        m   = mask_in.index_select(dim=-1, index=self.target_idx).bool()
        ctx = torch.where(m, ctx, torch.full_like(ctx, float("nan"))).permute(0, 2, 1).contiguous()
        ctx = ctx.reshape(B * F, self.window_in)
        cm  = m.permute(0, 2, 1).contiguous().reshape(B * F, self.window_in)
        gid = torch.arange(B).repeat_interleave(F)
        all_missing = torch.isnan(ctx).all(dim=1)
        if all_missing.any():
            ctx[all_missing] = 0.0
        inner = self._pipeline(x_in.device).model
        nop = math.ceil(self.window_out / inner.chronos_config.output_patch_size)
        # inner.encode does NOT move inputs to the model's device → feed all on x_in.device.
        dev = x_in.device
        out = inner.encode(context=ctx.float().to(dev), context_mask=cm.float().to(dev),
                           group_ids=gid.to(dev), num_output_patches=nop)
        enc = out[0] if isinstance(out, (tuple, list)) else out
        hid = getattr(enc, "last_hidden_state", enc)                  # (B*F, n_patch, d_model)
        z = hid.reshape(B * F, -1, hid.shape[-1]).mean(dim=1).reshape(B, F, -1)   # (B,F,d_model)
        return z.to(device=x_in.device, dtype=x_in.dtype)

    @torch.no_grad()
    def forward(
        self,
        x_in: torch.Tensor,         # (B, T_in, F_in)  z-scored, 0-filled
        mask_in: torch.Tensor,      # (B, T_in, F_in)  bool
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        F = int(self.target_idx.numel())

        # (B, F, T_in) multivariate context, true gaps restored as NaN
        ctx = x_in.index_select(dim=-1, index=self.target_idx)          # (B,T_in,F)
        m   = mask_in.index_select(dim=-1, index=self.target_idx).bool()
        ctx = torch.where(m, ctx, torch.full_like(ctx, float("nan")))
        ctx = ctx.permute(0, 2, 1).contiguous()                         # (B, F, T_in)

        # fully-missing channels → z-score mean (0)
        all_missing = torch.isnan(ctx).all(dim=2)                        # (B, F)
        if all_missing.any():
            ctx[all_missing] = 0.0

        pipe = self._pipeline(x_in.device)
        q, mean = pipe.predict_quantiles(
            ctx.float().cpu(), prediction_length=self.window_out,
            quantile_levels=[0.5],
        )
        # q: list[B] of (F, T_out, 1); mean: list[B] of (F, T_out)
        if self.point == "mean":
            pred = torch.stack(list(mean), dim=0)                        # (B, F, T_out)
        else:
            pred = torch.stack([qi[..., 0] for qi in q], dim=0)          # (B, F, T_out)

        pred = pred.permute(0, 2, 1)                                     # (B, T_out, F)
        pred = torch.nan_to_num(pred, nan=0.0)
        return pred.to(device=x_in.device, dtype=x_in.dtype)
