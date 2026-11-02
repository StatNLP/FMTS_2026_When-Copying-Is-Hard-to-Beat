"""
harmonize/models/arima.py

Per-channel ARIMA baseline (classical TSF reference for Paper 1).

For every (window, target channel) the input series is fit with a small ARIMA model
(`statsmodels`) and forecast `T_out` steps ahead. This is **parameter-free** (no gradient
training) but **slow** — one model fit per window × channel — so evaluate it on a
*subsample* (`--max-steps-per-epoch` small / `eval --limit`), not the full set.

Robustness: degenerate series (constant / non-finite) and fit failures fall back to a
last-value + linear-drift forecast (≈ ARIMA(0,1,0) with drift); if `statsmodels` is not
installed the whole model uses that fallback, so it always returns a valid forecast.

Default order (2,1,0) = AR(2) on first differences: fast and stable (no slow/finicky MA
term). The order is fixed here; expose a CLI flag if a sweep over (p,d,q) is wanted.
"""

from __future__ import annotations

import logging
import warnings

import numpy as np
import torch

from .base import ForecastModel

log = logging.getLogger(__name__)


def _drift_forecast(s: np.ndarray, t_out: int) -> np.ndarray:
    """Last value + linear drift over the input window."""
    last = s[-1]
    drift = (s[-1] - s[0]) / max(len(s) - 1, 1)
    return (last + drift * np.arange(1, t_out + 1)).astype(np.float32)


def _ar_cls_forecast(x: np.ndarray, p: int, d: int, t_out: int) -> np.ndarray:
    """ARIMA(p, d, 0) by **conditional least squares**, vectorised over batch × features.

    Fits AR(p) (intercept + p lags, ridge-stabilised) on the d-times-differenced series
    via the normal equations — no per-window MLE, so this is ~1000× faster than the
    statsmodels state-space fit and runs on the full validation set. Supports d ∈ {0, 1}.
    Returns (B, t_out, F).
    """
    B, T, F = x.shape
    base_last = x[:, -1, :]                                  # (B, F) — for d=1 integration
    series = np.nan_to_num(x.astype(np.float64), nan=0.0)
    for _ in range(d):
        series = np.diff(series, axis=1)                     # (B, m, F)
    m = series.shape[1]
    p = max(1, min(p, m - 1))
    Y = series[:, p:, :]                                     # (B, n, F)
    n = Y.shape[1]
    cols = [series[:, p - j:m - j, :] for j in range(1, p + 1)]   # AR lags only (canonical no-drift)
    Xr = np.moveaxis(np.stack(cols, axis=2), 3, 1).reshape(B * F, n, p)
    Yr = np.moveaxis(Y, 2, 1).reshape(B * F, n)
    XtX = np.einsum("bnk,bnl->bkl", Xr, Xr) + 1e-6 * np.eye(p)
    Xty = np.einsum("bnk,bn->bk", Xr, Yr)
    phi = np.linalg.solve(XtX, Xty[..., None])[..., 0].reshape(B, F, p)   # (B, F, p)
    lags = [series[:, -j, :] for j in range(1, p + 1)]       # lag1 … lagp (newest first)
    preds = []
    for _ in range(t_out):
        nxt = sum(phi[..., j] * lags[j] for j in range(p))
        preds.append(nxt)
        lags = [nxt] + lags[:-1]
    pred = np.stack(preds, axis=1)                           # (B, t_out, F) on the diff scale
    if d == 1:                                               # integrate back to levels
        pred = base_last[:, None, :] + np.cumsum(pred, axis=1)
    pred = np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    # Guard non-stationary AR (|φ|>1) blow-up over the horizon: data are z-scored, so a
    # legitimate value essentially never exceeds ±10 σ. Without this the val MSE goes to inf.
    return np.clip(pred, -10.0, 10.0).astype(np.float32)


class ARIMAForecast(ForecastModel):
    """ARIMA(p,d,q) per target channel (parameter-free; slow → subsample eval)."""

    def __init__(
        self,
        n_features_in: int,
        n_features_target: int,
        n_features_static: int,
        window_in: int,
        window_out: int,
        target_idx_in_input: list[int] | None = None,
        *,
        order: tuple[int, int, int] = (1, 1, 0),
        **kwargs,
    ) -> None:
        super().__init__(
            n_features_in, n_features_target, n_features_static,
            window_in, window_out, target_idx_in_input,
        )
        self.order = tuple(order)
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        try:
            from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
            self._have_sm = True
        except Exception:
            self._have_sm = False
            log.warning("statsmodels not available — ARIMAForecast falls back to drift.")

    def forward(
        self,
        x_in: torch.Tensor,         # (B, T_in, F_in)
        mask_in: torch.Tensor,
        static: torch.Tensor,
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        x = x_in.index_select(dim=-1, index=self.target_idx).detach().cpu().numpy()
        B, T, F = x.shape
        t_out = self.window_out
        p, d, q = self.order

        # Fast path: AR(p, d, 0) via conditional least squares — vectorised, no MLE.
        if q == 0 and d <= 1:
            out = _ar_cls_forecast(x, p, d, t_out)
            return torch.from_numpy(out).to(x_in.device, x_in.dtype)

        # Slow path (MA term q>0 / d>1): per-window statsmodels MLE, drift fallback.
        out = np.empty((B, t_out, F), dtype=np.float32)
        SM = None
        if self._have_sm:
            from statsmodels.tsa.arima.model import ARIMA as SM
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")     # convergence noise; we have a drift fallback
            for bi in range(B):
                for fi in range(F):
                    s = np.asarray(x[bi, :, fi], dtype=np.float64)
                    fc = None
                    if SM is not None and np.isfinite(s).all() and s.std() > 1e-6:
                        try:
                            res = SM(s, order=self.order,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False).fit(
                                method_kwargs={"maxiter": 25, "disp": 0})
                            fc = np.asarray(res.forecast(t_out), dtype=np.float32)
                        except Exception:
                            fc = None
                    if fc is None or not np.isfinite(fc).all():
                        fc = _drift_forecast(s, t_out)
                    out[bi, :, fi] = fc
        return torch.from_numpy(out).to(x_in.device, x_in.dtype)
