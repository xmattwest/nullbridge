"""Nullbridge: Zero-FLOP null-space alignment inside recurrent transformer loop bridges."""

from .algebra import (
    ConceptBasis,
    FoldedBridgeWeights,
    ProcrustesFit,
    audit_algebraic_properties,
    bootstrap_mean_ci,
    build_orthonormal_basis,
    exact_or_mc_signflip_p,
    extract_controlled_basis,
    fold_linear_writer,
    fold_message_bank_projection,
    fold_procrustes_bridge,
    holm_adjust,
    sample_random_orthonormal_basis,
    solve_orthogonal_procrustes,
)
from .benchmarks import build_nullbridge_benchmarks, audit_nullbridge_benchmarks

__version__ = "0.1.0"

__all__ = [
    "ConceptBasis",
    "FoldedBridgeWeights",
    "ProcrustesFit",
    "__version__",
    "audit_algebraic_properties",
    "audit_nullbridge_benchmarks",
    "bootstrap_mean_ci",
    "build_nullbridge_benchmarks",
    "build_orthonormal_basis",
    "exact_or_mc_signflip_p",
    "extract_controlled_basis",
    "fold_linear_writer",
    "fold_message_bank_projection",
    "fold_procrustes_bridge",
    "holm_adjust",
    "sample_random_orthonormal_basis",
    "solve_orthogonal_procrustes",
]
