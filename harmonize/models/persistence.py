"""
harmonize/models/persistence.py

Last-value persistence baseline: ŷ_t = x_in[-1, target] for every t in T_out.
No learnable parameters. Used as the trivial reference number.
"""

from __future__ import annotations

import torch

from .base import ForecastModel


class Persistence(ForecastModel):
    """Repeat the last observed value of each target feature across T_out."""

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
        # Need at least one parameter to keep the training loop generic.
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)

    def forward(
        self,
        x_in: torch.Tensor,
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        last = x_in.index_select(dim=-1, index=self.target_idx)[:, -1:, :]
        return last.expand(-1, self.window_out, -1).clone()
