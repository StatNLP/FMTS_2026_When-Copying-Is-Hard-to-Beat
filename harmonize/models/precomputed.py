"""
harmonize/models/precomputed.py

A ForecastModel that returns **precomputed** forecasts loaded from a .npy file,
in DataLoader order. Used to evaluate models whose inference runs in a different
environment (e.g. Moirai/uni2ts in py311_moirai → `harmonize/infer_moirai.py`
writes z-scored predictions in H5 row order; the py313 Stage-5 eval then consumes
them here and runs all the usual clinical metrics).

The eval DataLoader is `shuffle=False`, so windows arrive in H5 row order; this
module hands out consecutive slices of the prediction array via an internal
pointer. It therefore assumes a single ordered pass over the matching test H5.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import ForecastModel


class PrecomputedForecast(ForecastModel):
    def __init__(
        self,
        n_features_in: int,
        n_features_target: int,
        n_features_static: int,
        window_in: int,
        window_out: int,
        target_idx_in_input: list[int] | None = None,
        *,
        pred_file: str,
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        arr = np.load(pred_file)                       # (N, T_out, F_target) z-scored
        if arr.ndim != 3 or arr.shape[1] != window_out or arr.shape[2] != n_features_target:
            raise ValueError(
                f"pred file {pred_file} has shape {arr.shape}, expected "
                f"(N, {window_out}, {n_features_target})")
        # N is validated by the CALLER against len(dataset): an over-long .npy (e.g. left over
        # from a different split) otherwise passes silently, the eval consumes the first
        # len(ds) rows, and every clinical metric is computed on mis-paired windows.
        self.n_rows = int(arr.shape[0])
        self.register_buffer("_preds", torch.from_numpy(arr.astype(np.float32)))
        self._ptr = 0
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
        b = x_in.shape[0]
        if self._ptr + b > self._preds.shape[0]:
            raise RuntimeError(
                f"precomputed predictions exhausted: ptr={self._ptr} + {b} > "
                f"{self._preds.shape[0]} (pred file and test H5 row counts must match)")
        out = self._preds[self._ptr:self._ptr + b]
        self._ptr += b
        return out.to(device=x_in.device, dtype=x_in.dtype)
