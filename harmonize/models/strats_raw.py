"""
harmonize/models/strats_raw.py

True STraTS — set-function encoder over RAW (non-aggregated) observations.

Identical architecture to `StratsForecast` (CVE value+time embeddings + per-variable
embedding + set-Transformer + iterative AR decoder), but the input is the genuine
irregular set of measurements produced by `harmonize.stage3_make_triplets` (no
per-hour densification). The decoder and target are unchanged, so it is directly
comparable to every other model; only the input representation differs.

Consumes the padded batch from `harmonize.triplet_dataset.triplet_collate`:
    tri_val (B,M) z-scored values · tri_time (B,M) Δt · tri_var (B,M) var idx ·
    tri_pad (B,M) bool (True = padded).
"""

from __future__ import annotations

import torch

from .strats import StratsForecast


class StratsRawForecast(StratsForecast):
    """STraTS on raw triplets. Inherits all submodules + the AR decoder from
    StratsForecast; only the forward front-end changes (no dense reconstruction
    and no max_tokens cap — the per-batch collate already bounds the set length)."""

    def forward(
        self,
        tri_val: torch.Tensor,    # (B, M)
        tri_time: torch.Tensor,   # (B, M)
        tri_var: torch.Tensor,    # (B, M) long
        tri_pad: torch.Tensor,    # (B, M) bool, True = padded
        static: torch.Tensor,     # (B, F_static)
        *,
        x_target: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> torch.Tensor:
        memory, mem_pad = self._encode_tokens(
            tri_val, tri_time, tri_var.long(), tri_pad, static
        )
        return self._decode(memory, mem_pad,
                            x_target=x_target,
                            teacher_forcing_ratio=teacher_forcing_ratio)
