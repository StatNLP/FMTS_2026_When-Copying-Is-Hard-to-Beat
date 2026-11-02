"""
harmonize/models/base.py

Uniform forecast-model interface for Paper-1 reproduction + extensions.
See docs/2605_comprehensive_documentation.md

All concrete models should subclass ForecastModel and implement `_forward`,
which receives the encoder representation + decoder targets and must return
predictions of shape (B, T_out, F_target).

The base class itself only forwards the canonical kwargs. Student-forcing
schedules live in the training loop, not the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class ForecastModel(nn.Module, ABC):
    """Common interface for all forecast models in this project.

    Args:
        n_features_in:     number of input features per timestep
        n_features_target: number of target (predicted) features per timestep
        n_features_static: number of static / per-stay features (e.g. age, sex)
        window_in:         input horizon T_in (typically 24 h)
        window_out:        forecast horizon T_out (typically 24 h)
        target_idx_in_input: indices of the target features within features_in
                             (the canonical embedding of target ⊆ input).
                             Used by autoregressive models to feed predictions
                             back into the input space at the next step.
    """

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
        super().__init__()
        self.n_features_in     = n_features_in
        self.n_features_target = n_features_target
        self.n_features_static = n_features_static
        self.window_in         = window_in
        self.window_out        = window_out
        # If not given, assume target == input (Paper-1 default).
        if target_idx_in_input is None:
            if n_features_target != n_features_in:
                raise ValueError(
                    "target_idx_in_input required when "
                    "n_features_target != n_features_in"
                )
            target_idx_in_input = list(range(n_features_in))
        self.register_buffer(
            "target_idx", torch.tensor(target_idx_in_input, dtype=torch.long)
        )

    @abstractmethod
    def forward(
        self,
        x_in: torch.Tensor,         # (B, T_in,  F_in)   float32
        mask_in: torch.Tensor,      # (B, T_in,  F_in)   bool
        static: torch.Tensor,       # (B, F_static)      float32
        *,
        x_target: torch.Tensor | None = None,    # (B, T_out, F_target) — for teacher forcing
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Return predictions of shape (B, T_out, F_target)."""

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ------------------------------------------------------------------
    # Embedding-space data-augmentation hooks (Paper-2 MIA defenses).
    #
    # Models that set `supports_embedding_da = True` split their forward into
    # two halves at the (frozen) input-embedding layer:
    #
    #   emb  = embed_inputs(x_in, mask_in, static)          # (B, T_in, d_model)
    #   pred = forward_from_embedding(emb, ...)              # (B, T_out, F_target)
    #
    # Composing the two must reproduce `forward(...)` exactly. The split lets the
    # privacy stage (harmonize/privacy/) (a) precompute + perturb the embedding
    # and (b) fine-tune *only* the weights downstream of the embedding — the
    # embedding layer itself is simply not on the `forward_from_embedding` path,
    # so it receives no gradient and stays fixed.
    # ------------------------------------------------------------------
    supports_embedding_da: bool = False

    def embed_inputs(
        self,
        x_in: torch.Tensor,         # (B, T_in, F_in)
        mask_in: torch.Tensor,      # (B, T_in, F_in)
        static: torch.Tensor,       # (B, F_static)
    ) -> torch.Tensor:
        """Return the input embedding (B, T_in, d_model) fed downstream."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support embedding-space DA "
            "(embed_inputs). Set supports_embedding_da=True and implement it."
        )

    def forward_from_embedding(
        self,
        emb: torch.Tensor,          # (B, T_in, d_model)
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        """Run everything downstream of `embed_inputs`; return (B, T_out, F_target)."""
        raise NotImplementedError(
            f"{type(self).__name__} does not support embedding-space DA "
            "(forward_from_embedding). Set supports_embedding_da=True and implement it."
        )
