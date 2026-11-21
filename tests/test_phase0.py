"""Comprehensive Phase 0 unit tests for algebra, disjoint benchmarks, synthetic loops, and repo hygiene."""

from __future__ import annotations

from pathlib import Path
import numpy as np

from nullbridge.algebra import (
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
from nullbridge.benchmarks import audit_nullbridge_benchmarks, build_nullbridge_benchmarks
from nullbridge.cli import BLOCKED_PATTERNS, run_phase0_audit
from nullbridge.mock_loop import run_synthetic_loop_verification


def test_algebraic_nullspace_and_isometry() -> None:
    res = audit_algebraic_properties(d_model=128, rank=4, lora_rank=16, r_msg=64)
    assert res["valid"] is True
    assert res["max_null_leakage_r"] < 1e-6
    assert res["max_null_leakage_a"] < 1e-6
    assert res["max_complement_isometry_err"] < 1e-6
    assert res["n_unit_singular_values"] == 124
    assert res["n_zero_singular_values"] == 4
    assert res["extra_parameters"] == 0
    assert res["extra_runtime_flops"] == 0


def test_linear_writer_and_message_output_fold() -> None:
    rng = np.random.default_rng(1234)
    w_writer = rng.standard_normal((96, 64))
    bias = rng.standard_normal(96)
    neutral = rng.standard_normal(96)
    dirs = rng.standard_normal((3, 96))
    q, mu_proj = build_orthonormal_basis(dirs, neutral_mean=neutral)

    w_safe, b_safe, leak = fold_linear_writer(w_writer, q, bias=bias, neutral_mean=neutral)
    assert leak < 1e-6
    assert b_safe is not None

    x = rng.standard_normal((16, 64))
    y = x @ w_safe.astype(np.float64).T + b_safe.astype(np.float64)
    proj_on_q = (y @ q.astype(np.float64)) @ q.astype(np.float64).T
    assert np.allclose(proj_on_q, mu_proj.astype(np.float64), atol=1e-5)

    w_out, out_leak = fold_message_bank_projection(w_writer, q, side="output")
    assert out_leak < 1e-6
    assert w_out.shape == w_writer.shape


def test_controlled_basis_orthogonalization_and_stats() -> None:
    rng = np.random.default_rng(77)
    n_pairs, n_layers, d = 16, 4, 48
    ctrl_dir = rng.standard_normal(d)
    ctrl_dir /= np.linalg.norm(ctrl_dir)
    core_dir = rng.standard_normal(d)
    core_dir = core_dir - (core_dir @ ctrl_dir) * ctrl_dir
    core_dir /= np.linalg.norm(core_dir)

    pos_ctrl = rng.standard_normal((n_pairs, n_layers, d)) * 0.1 + ctrl_dir
    neg_ctrl = rng.standard_normal((n_pairs, n_layers, d)) * 0.1 - ctrl_dir
    # Mix 0.6 * ctrl_dir into core positive pairs to verify QR control removal recovers pure core_dir!
    pos_core = rng.standard_normal((n_pairs, n_layers, d)) * 0.1 + (core_dir + 0.6 * ctrl_dir)
    neg_core = rng.standard_normal((n_pairs, n_layers, d)) * 0.1 - (core_dir + 0.6 * ctrl_dir)

    basis = extract_controlled_basis(
        pos_by_concept={"syc": pos_core, "ctrl": pos_ctrl},
        neg_by_concept={"syc": neg_core, "ctrl": neg_ctrl},
        concepts=("syc", "ctrl"),
        controls=("ctrl",),
    )
    for ell in range(n_layers):
        dot_ctrl = abs(float(basis.controlled_dirs["syc"][ell] @ basis.raw_dirs["ctrl"][ell]))
        assert dot_ctrl < 1e-5
        assert float(basis.lopo_accuracy["syc"][ell]) > 0.95

    p_strong = exact_or_mc_signflip_p(np.ones(12) * 1.5)
    assert p_strong < 0.001
    adj = holm_adjust([0.001, 0.02, 0.50])
    assert list(adj) == sorted(adj)
    lo, hi = bootstrap_mean_ci(np.arange(10, dtype=float))
    assert lo < 4.5 < hi


def test_disjoint_benchmarks_zero_leakage() -> None:
    aud = audit_nullbridge_benchmarks()
    assert aud["valid"] is True
    assert aud["overlaps"]["probe_vs_train_sha256"] == 0
    assert aud["overlaps"]["probe_vs_eval_sha256"] == 0
    assert aud["overlaps"]["train_vs_eval_sha256"] == 0
    assert aud["overlaps"]["kl_passage_overlap"] == 0
    assert aud["overlaps"]["leaked_people_into_eval"] == []
    assert aud["overlaps"]["leaked_orgs_into_eval"] == []

    bm = build_nullbridge_benchmarks()
    assert len(bm["eval_covert_sycophancy"]) == 64
    assert len(bm["eval_verifier"]) == 96
    assert len(bm["eval_alias_guardrail"]) == 48


def test_synthetic_loop_end_to_end() -> None:
    syn = run_synthetic_loop_verification()
    assert syn["valid"] is True
    assert syn["conditions"]["dual_nullbridge"]["by_k"]["k8"]["covert_numeric_bias"] < 1e-5
    assert syn["conditions"]["unclamped_loop"]["by_k"]["k8"]["multihop_acc"] > 0.90


def test_guard_catches_ndif_and_sas_patterns() -> None:
    samples = {
        "ndif_key_assignment": "NDIF_API" + "_KEY=" + ("a9f8" * 6),
        "azure_sas_signature": "https://acct.blob.core.windows.net/c/b?sv=2023-11-03&si" + "g=" + ("AbCd" * 7),
    }
    patterns = dict(BLOCKED_PATTERNS)
    for key, text in samples.items():
        assert patterns[key].search(text) is not None


def test_phase0_cli_audit(tmp_path: Path) -> None:
    out_json = tmp_path / "audit.json"
    res = run_phase0_audit(out_path=out_json)
    assert res["all_valid"] is True
    assert out_json.is_file()
