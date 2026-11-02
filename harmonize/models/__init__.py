"""
harmonize/models/__init__.py

Model registry for Paper-1 reproduction + extensions.
See docs/2605_comprehensive_documentation.md for the design.

Use `build_model(name, **kwargs)` to instantiate a model by string id.
"""

from __future__ import annotations

from typing import Any

from .autoformer import AutoformerForecast
from .base import ForecastModel
from .chronos import ChronosForecast
from .chronos2 import Chronos2Forecast
from .dense_autoregressive import DenseAutoregressive
from .arima import ARIMAForecast
from .dense_dms import DenseDirectDecoder
from .dlinear import DLinear
from .ets import ETSForecast
from .informer import InformerForecast
from .patchtst import PatchTSTForecast
from .mamba import MambaForecast
from .null import NullMean
from .outcome_heads import AKIHead, DSOFA2Head, DSOFA3Head, LOSHead, MortalityHead
from .persistence import Persistence
from .rnn import GRUForecast, LSTMForecast
from .strats import StratsForecast
from .strats_raw import StratsRawForecast

REGISTRY: dict[str, type[ForecastModel]] = {
    "zeroforecast":         NullMean,   # climatological "predict z-scored mean = 0" baseline
    "null":                 NullMean,   # back-compat alias (pre-rename checkpoints store model="null")
    "persistence":          Persistence,
    "dlinear":              DLinear,
    "ets":                  ETSForecast,    # damped Holt exponential smoothing
    "arima":                ARIMAForecast,  # per-channel ARIMA (slow → subsample eval)
    "dense_autoregressive": DenseAutoregressive,
    "dense_pteacher":       DenseAutoregressive,  # partial TF: exogenous meds fed from GT
    "dense_dms":            DenseDirectDecoder,
    "gru":                  GRUForecast,    # recurrent seq2seq (GRU encoder–decoder, IMS)
    "lstm":                 LSTMForecast,   # recurrent seq2seq (LSTM encoder–decoder, IMS)
    "mamba":                MambaForecast,  # selective state-space (dynamic) seq2seq, IMS
    "informer":             InformerForecast,
    "patchtst":             PatchTSTForecast,   # channel-independent patching Transformer (Nie 2023, DMS)
    "autoformer":           AutoformerForecast,
    "strats":               StratsForecast,
    "strats_raw":           StratsRawForecast,
    "chronos":              ChronosForecast,   # pretrained FM baseline (zero-shot, univariate)
    "chronos2":             Chronos2Forecast,  # pretrained FM baseline (zero-shot, multivariate)
}

HEAD_REGISTRY: dict[str, type] = {
    "mortality": MortalityHead,
    # horizon mortality — same binary architecture, different label horizon; the label itself is
    # built per stay from the cohort (harmonize/scores/mortality.py) because the H5's
    # mortality_hours is los_hours-for-the-deceased, not a time-to-death (§13.4a).
    "mortality7":  MortalityHead,
    "mortality30": MortalityHead,
    "los":       LOSHead,
    # binary LoS at the conventional 3/7-day thresholds -- same binary architecture as
    # mortality. Only meaningful with --first-day-only: across all windows a stay must
    # already be long to contribute a late window, so "LoS > 3d" is 91-98 % positive.
    "los3":      MortalityHead,
    "los7":      MortalityHead,
    "aki":       AKIHead,   # new AKI (KDIGO>=1, creatinine-only) in the horizon
    "aki_uo":    AKIHead,   # new AKI incl. KDIGO oliguria (urine, dense-charting gated)
    "dsofa2":    DSOFA2Head, # ΔSOFA_total >= 2 deterioration in the horizon
    "dsofa3":    DSOFA3Head, # 3-class signed ΔSOFA: stable / deteriorate / improve
    "sepsis":    DSOFA2Head, # Sepsis-3 = ΔSOFA>=2 AND suspected infection (label gated in train_head)
    "sofa":      LOSHead,     # SOFA total (worst-of-window) regression — discriminative head vs forecast read-off
    "saps":      LOSHead,     # SAPS-II (worst-of-window) regression — discriminative head vs forecast read-off
}


def build_model(name: str, **kwargs: Any) -> ForecastModel:
    if name not in REGISTRY:
        raise ValueError(
            f"unknown model {name!r}. Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](**kwargs)


def build_head(name: str, **kwargs: Any):
    if name not in HEAD_REGISTRY:
        raise ValueError(
            f"unknown head {name!r}. Available: {sorted(HEAD_REGISTRY)}"
        )
    return HEAD_REGISTRY[name](**kwargs)


__all__ = [
    "ForecastModel",
    "ChronosForecast",
    "Chronos2Forecast",
    "DenseAutoregressive",
    "DenseDirectDecoder",
    "DLinear",
    "ETSForecast",
    "ARIMAForecast",
    "InformerForecast",
    "AutoformerForecast",
    "MambaForecast",
    "StratsForecast",
    "StratsRawForecast",
    "NullMean",
    "Persistence",
    "LOSHead",
    "MortalityHead",
    "REGISTRY",
    "HEAD_REGISTRY",
    "build_model",
    "build_head",
]
