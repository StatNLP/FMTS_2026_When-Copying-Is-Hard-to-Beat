"""
harmonize/models/outcome_heads.py
Lightweight outcome-prediction heads (mortality / LOS) that take the same
input as the forecast models — `(x_in, mask_in, static)` — and emit a single
scalar per stay.

Two variants:
    - MortalityHead  — binary logit (BCE loss)
    - LOSHead        — log-LOS regression (MSE on log-scaled hours)

Both share an encoder (per-feature MLP over the 24 h window + static side
input). When a trained forecast checkpoint is available, the encoder can be
initialised from it via `load_forecast_encoder(path)` (planned, Phase E.3).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class _WindowEncoder(nn.Module):
    """Compact encoder: flatten (T, F) + mask + static through an MLP.
    No attention, no AR — these heads only need a representation, not a
    forecast. Inputs are already imputed by Stage-3.
    """

    def __init__(self, n_features_in: int, n_features_static: int,
                 window_in: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        flat_in = window_in * (2 * n_features_in) + n_features_static
        self.net = nn.Sequential(
            nn.Linear(flat_in, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        # x_in:    (B, T, F)
        # mask_in: (B, T, F) bool
        # static:  (B, F_static)
        B = x_in.size(0)
        x = torch.cat([x_in, mask_in.float()], dim=-1)        # (B, T, 2F)
        x = x.reshape(B, -1)                                   # (B, T*2F)
        x = torch.cat([x, static], dim=-1)                     # (B, T*2F + F_static)
        return self.net(x)                                     # (B, d_model)


class MortalityHead(nn.Module):
    """Binary classifier for ICU mortality from first-24h input window.

    forward(...) returns logits of shape (B,). Pair with `BCEWithLogitsLoss`.
    """

    def __init__(self, n_features_in: int, n_features_static: int,
                 window_in: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = _WindowEncoder(n_features_in, n_features_static,
                                      window_in, d_model, dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_in, mask_in, static)
        return self.classifier(h).squeeze(-1)


class AKIHead(MortalityHead):
    """Binary classifier for new AKI (KDIGO stage >= 1) within the forecast
    horizon, from the first-24h input window. Same architecture as MortalityHead;
    the label is derived from the ground-truth target-window creatinine trajectory
    (see harmonize.train_head.labels_from_batch). Pair with BCEWithLogitsLoss."""
    pass


class DSOFA2Head(MortalityHead):
    """Binary classifier for clinically-significant deterioration — ΔSOFA_total ≥ 2
    within the forecast horizon — from the first-24h input window. Same architecture
    as MortalityHead; the label is derived from the ground-truth target-window SOFA
    total (see harmonize.train_head.labels_from_batch). The discriminative counterpart
    to the forecast-implicit ΔSOFA≥2 read-off, which sits near chance (the core
    validity test). Pair with BCEWithLogitsLoss."""
    pass


class DSOFA3Head(nn.Module):
    """3-class trajectory classifier — does the patient deteriorate / improve / stay
    over the forecast horizon? Label = signed daily ΔSOFA binned at ±1 (0=stable,
    1=deteriorate ≥+1, 2=improve ≤−1; see harmonize.train_head.labels_from_batch and
    10_scripts/recompute_sofa_scores.py). Same _WindowEncoder as the binary heads but a
    3-way head. Pair with CrossEntropyLoss (reduction='none')."""

    def __init__(self, n_features_in: int, n_features_static: int,
                 window_in: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = _WindowEncoder(n_features_in, n_features_static,
                                      window_in, d_model, dropout)
        self.classifier = nn.Linear(d_model, 3)

    def forward(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_in, mask_in, static)
        return self.classifier(h)          # (B, 3) logits


class LOSHead(nn.Module):
    """Regress log(LOS_hours) from first-24h input window.

    Use log-LOS for stability: LOS is right-skewed (median ~70h, p99 ~ 2 weeks).
    Train loss = MSE on log-scaled label; report MAE in physical (hours) space.
    """

    def __init__(self, n_features_in: int, n_features_static: int,
                 window_in: int, d_model: int = 128, dropout: float = 0.1):
        super().__init__()
        self.encoder = _WindowEncoder(n_features_in, n_features_static,
                                      window_in, d_model, dropout)
        self.regressor = nn.Linear(d_model, 1)

    def forward(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x_in, mask_in, static)
        return self.regressor(h).squeeze(-1)


class FrozenBackboneHead(nn.Module):
    """Binary classifier on a FROZEN pretrained forecast encoder. Pools the forecast
    model's encoder hidden state (`backbone.encode(...)` → (B, T_in, d_model)) over time
    and classifies with a small MLP. Only the classifier trains.

    Unlike the standalone heads (which train their own input MLP from scratch), this probes
    whether the *forecast's learned representation* encodes the label — the clean test for
    "is the deterioration signal in the forecast backbone, or only in the raw input?".
    Pair with BCEWithLogitsLoss. `backbone` must share the head data's feature space."""

    def __init__(self, backbone: nn.Module, d_model: int, n_static: int = 0,
                 pool: str = "mean", dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.pool = pool
        # n_static > 0: also concatenate the raw static (demographics) to the pooled
        # representation before the classifier — the "representation + demographics"
        # variant, comparable to the from-scratch heads (which see static). n_static=0
        # is the pure "does the representation alone encode it?" probe (the R2 default).
        self.n_static = n_static
        self.classifier = nn.Sequential(
            nn.Linear(d_model + n_static, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            h = self.backbone.encode(x_in, mask_in, static)   # (B, T_in, d_model)
        z = h.mean(dim=1) if self.pool == "mean" else h[:, -1]
        if self.n_static > 0:
            z = torch.cat([z, torch.nan_to_num(static)], dim=-1)
        return self.classifier(z).squeeze(-1)


class ForecastOutputHead(nn.Module):
    """Two-stage *forecast → detect* classifier: run a FROZEN forecaster's full student
    rollout to get the GENERATED (B, T_out, F_target) trajectory, then train a from-scratch
    `_WindowEncoder` + linear classifier on that generated trajectory.

    This is the third probe of the R2 dissociation, distinct from both existing ones:
      • the hand-coded read-off  reads the forecast OUTPUT with a *fixed* (Vincent/KDIGO)
        rule  → ≈0.6 (near chance);
      • `FrozenBackboneHead`     reads the forecast ENCODER REPRESENTATION with a learned
        MLP                       → ≈0.7;
      • this head               reads the forecast OUTPUT TRAJECTORY with a *learned* MLP.
    It answers the sharp question the other two leave open: **is the deterioration signal
    actually present in the generated trajectory (recoverable by a learned read-off that
    the hand-coded SOFA/KDIGO extraction throws away), or is it genuinely lost in the AR
    rollout?** If this head ≈ the representation head (0.7) ≫ the hand-coded read-off (0.6),
    the trajectory carries the signal and the fix is a learned read-off, not a new forecast
    objective; if it stays ≈0.6, the rollout truly destroys it → a deterioration-aware
    forecast objective is required. The forecaster is frozen; only the classifier trains.
    Pair with BCEWithLogitsLoss. `forecaster` must share the head data's feature space."""

    def __init__(self, forecaster: nn.Module, n_features_target: int, n_features_static: int,
                 window_out: int, d_model: int = 128, n_static: int = 0, dropout: float = 0.1):
        super().__init__()
        self.forecaster = forecaster
        for p in self.forecaster.parameters():
            p.requires_grad = False
        # Encode the generated trajectory exactly like a from-scratch head encodes a raw
        # window — same _WindowEncoder, but over (T_out, F_target) instead of (T_in, F_in),
        # with an all-ones mask (the generated trajectory is dense). n_static is folded into
        # the encoder's flat input (it concatenates static internally), so we pass the raw
        # F_static when static is wanted and 0 otherwise.
        self.n_static = n_static
        self.encoder = _WindowEncoder(n_features_target, n_static, window_out, d_model, dropout)
        self.classifier = nn.Linear(d_model, 1)

    def forward(self, x_in: torch.Tensor, mask_in: torch.Tensor,
                static: torch.Tensor) -> torch.Tensor:
        self.forecaster.eval()
        with torch.no_grad():
            out = self.forecaster(x_in, mask_in, static)      # (B, T_out, F_target), z-score
        out = torch.nan_to_num(out, nan=0.0)
        mask = torch.ones_like(out)                            # generated trajectory is dense
        st = torch.nan_to_num(static) if self.n_static > 0 else static[:, :0]
        h = self.encoder(out, mask, st)
        return self.classifier(h).squeeze(-1)
