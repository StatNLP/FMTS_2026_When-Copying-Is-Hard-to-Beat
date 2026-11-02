"""
harmonize/models/dlinear.py

Decomposition-Linear forecast baseline (Zeng et al., AAAI 2023).
Per-channel linear projection of the input horizon to the output horizon,
optionally after subtracting a moving-average trend.

Adapted from
10_scripts/archive/mlhc_2024_prediction_of_causes-main/.../models/DLinear.py
to the uniform `ForecastModel` interface.

Notes
-----
- DLinear ignores cross-channel structure, so we restrict it to the target
  feature subset (`features_target`) of the input. The non-target input
  channels (interventions) are dropped at the model input.
- DLinear is non-autoregressive, so `x_target` and `teacher_forcing_ratio`
  are ignored.
- Static features are ignored (DLinear has no notion of side inputs).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import ForecastModel


class _MovingAvg(nn.Module):
    """Moving-average pooling with edge-replication padding."""

    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        pad = (self.kernel_size - 1) // 2
        front = x[:, :1, :].repeat(1, pad, 1)
        end   = x[:, -1:, :].repeat(1, pad, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.transpose(1, 2)).transpose(1, 2)
        return x


class _SeriesDecomp(nn.Module):
    def __init__(self, kernel_size: int) -> None:
        super().__init__()
        self.moving_avg = _MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        trend = self.moving_avg(x)
        season = x - trend
        return season, trend


class DLinear(ForecastModel):
    """Decomposition-Linear forecaster.

    Hyperparameters:
        kernel_size: Moving-average kernel (odd, default 25 — paper's value;
            for short 24 h windows we recommend 5–9 instead).
        individual:  If True, use one linear layer per channel; if False,
            share weights across channels.
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
        kernel_size: int = 5,
        individual: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        if kernel_size % 2 == 0:
            kernel_size += 1  # force odd for symmetric padding
        self.decomp = _SeriesDecomp(kernel_size)
        self.individual = individual
        C = self.n_features_target

        if individual:
            self.lin_season = nn.ModuleList(
                [nn.Linear(window_in, window_out) for _ in range(C)]
            )
            self.lin_trend = nn.ModuleList(
                [nn.Linear(window_in, window_out) for _ in range(C)]
            )
        else:
            self.lin_season = nn.Linear(window_in, window_out)
            self.lin_trend  = nn.Linear(window_in, window_out)

    def forward(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        # Restrict to target channels (drop interventions etc.)
        x = x_in.index_select(dim=-1, index=self.target_idx)  # (B, T_in, F_target)
        season, trend = self.decomp(x)
        # Move to (B, C, T_in) for the per-channel linear
        season = season.transpose(1, 2)
        trend  = trend.transpose(1, 2)
        if self.individual:
            out_season = torch.stack(
                [self.lin_season[c](season[:, c, :]) for c in range(season.size(1))],
                dim=1,
            )
            out_trend = torch.stack(
                [self.lin_trend[c](trend[:, c, :]) for c in range(trend.size(1))],
                dim=1,
            )
        else:
            out_season = self.lin_season(season)
            out_trend  = self.lin_trend(trend)
        # (B, C, T_out) → (B, T_out, C)
        return (out_season + out_trend).transpose(1, 2)
