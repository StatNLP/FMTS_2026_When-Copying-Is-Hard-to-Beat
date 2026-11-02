"""
harmonize/models/ets.py

Exponential-smoothing baseline (damped Holt linear trend) per target channel.

A classical, parameter-free forecaster that sits between persistence ("repeat the
last value") and a learned model: it carries a *level* and a *trend* through the input
window and extrapolates the (damped) trend across the forecast horizon. Fully vectorised
over batch and features — no per-window/per-channel loop — so it is as fast as persistence.

Smoothing constants are fixed (not fitted): alpha (level), beta (trend), phi (trend
damping). Damping (phi < 1) keeps the 24-step extrapolation from running away on z-scored
data. This is the standard "ETS/Holt" non-neural reference.
"""

from __future__ import annotations

import torch

from .base import ForecastModel


class ETSForecast(ForecastModel):
    """Damped Holt's linear exponential smoothing (parameter-free)."""

    def __init__(
        self,
        n_features_in: int,
        n_features_target: int,
        n_features_static: int,
        window_in: int,
        window_out: int,
        target_idx_in_input: list[int] | None = None,
        *,
        alpha: float = 0.5,
        beta: float = 0.1,
        phi: float = 0.9,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.phi = float(phi)
        # Keep one (frozen) parameter so the generic training loop stays happy.
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        x_in: torch.Tensor,         # (B, T_in, F_in)
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        x = x_in.index_select(dim=-1, index=self.target_idx)   # (B, T_in, F)
        x = torch.nan_to_num(x, nan=0.0)
        B, T, F = x.shape
        a, b, phi = self.alpha, self.beta, self.phi

        level = x[:, 0, :].clone()
        trend = (x[:, 1, :] - x[:, 0, :]) if T > 1 else torch.zeros_like(level)
        for t in range(1, T):
            prev = level
            level = a * x[:, t, :] + (1.0 - a) * (level + phi * trend)
            trend = b * (level - prev) + (1.0 - b) * phi * trend

        # Damped-trend forecast: ŷ_h = level + (Σ_{i=1..h} phi^i) · trend
        h = torch.arange(1, self.window_out + 1, device=x.device, dtype=x.dtype)
        damp = (phi * (1.0 - phi ** h) / (1.0 - phi)) if phi < 1.0 else h
        out = level.unsqueeze(1) + damp.view(1, -1, 1) * trend.unsqueeze(1)
        return out                                              # (B, T_out, F)
