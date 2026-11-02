"""
harmonize/models/null.py

Null / climatological baseline: predict the per-feature training mean for every
forecast step, irrespective of the input window.

Targets are z-scored per feature on the TRAINING split (Stage-2), so the training
mean is 0 in normalised units — the null forecast is therefore simply all-zeros.
Its per-variable MSE equals the mean normalised target variance (≈ 1.0 when the
validation distribution matches training), so it is the natural skill reference:
any model with mse_per_var < null's has learned something beyond the mean.

No learnable parameters (the training loop detects this and skips the optimizer).
"""

from __future__ import annotations

import torch

from .base import ForecastModel


class NullMean(ForecastModel):
    """Predict the (normalised) training mean = zeros across the whole horizon."""

    def __init__(
        self,
        n_features_in: int,
        n_features_target: int,
        n_features_static: int,
        window_in: int,
        window_out: int,
        target_idx_in_input: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
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
        B = x_in.size(0)
        return x_in.new_zeros(B, self.window_out, self.n_features_target)
