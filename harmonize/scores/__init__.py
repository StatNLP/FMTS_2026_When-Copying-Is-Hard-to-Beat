"""
harmonize/scores/
Derived clinical scores computed *from* a (T, F) feature matrix — used to
turn forecast outputs into interpretable labels (SOFA, Sepsis-3, SAPS-II, …).

Each score module exposes a `score_from_matrix(x, feature_names, ...)` that
takes a physical-unit feature matrix (T, F) and returns the score in the
shape expected by Stage-4 eval.
"""

from .sapsii import (
    SAPSIIComponents,
    mortality_probability,
    sapsii_batch,
    sapsii_components,
    sapsii_from_matrix,
)
from .sepsis3 import (
    organ_dysfunction,
    sepsis3_at_t,
    sepsis3_per_stay,
    sepsis3_per_stay_umm,
    susinf_from_matrix,
    susinf_per_stay,
    susinf_per_stay_umm,
)
from .sofa import sofa_components, sofa_from_matrix

__all__ = [
    # SOFA
    "sofa_from_matrix", "sofa_components",
    # SAPS-II
    "sapsii_from_matrix", "sapsii_components", "sapsii_batch",
    "SAPSIIComponents", "mortality_probability",
    # Sepsis-3
    "sepsis3_at_t", "sepsis3_per_stay",
    "organ_dysfunction", "susinf_from_matrix",
    "susinf_per_stay", "susinf_per_stay_umm", "sepsis3_per_stay_umm",
]
