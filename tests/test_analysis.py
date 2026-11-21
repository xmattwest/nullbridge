"""Unit tests for deterministic Phase 3 statistical analysis and Dawn State formatting."""

from __future__ import annotations

from pathlib import Path

from nullbridge.analysis import (
    compute_track_a_statistical_analysis,
    compute_track_b_statistical_analysis,
    write_phase3_synthesis_markdown,
    write_phase4_synthesis_markdown,
)


def test_phase3_statistical_analysis_invariants(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    tb_path = repo_root / "results" / "phase3_gemma4_nullbridge" / "track_b_results.json"
    p0_path = repo_root / "results" / "phase0_audit" / "benchmark_and_algebra_audit.json"

    analysis = compute_track_b_statistical_analysis(tb_path, p0_path)
    assert analysis["architecture_summary"]["funnel_layers"] == 19
    assert analysis["architecture_summary"]["k4_effective_blocks"] == 117
    assert analysis["multipass_amplification"]["verifier_suscept_k1_to_k3_ratio"] > 18.0
    assert analysis["multipass_amplification"]["verifier_suscept_k1_to_k4_ratio"] > 14.0
    assert analysis["bridge_only_nullbridge_deltas"]["k4_balsheet_vs_unclamped_pp"] == 9.38
    assert analysis["dual_nullbridge_effects"]["k4_moral_bias_pct_change_vs_unclamped"] == -30.09

    md_out = tmp_path / "synthesis.md"
    import json

    write_phase3_synthesis_markdown(analysis, json.loads(tb_path.read_text(encoding="utf-8")), md_out)
    md_text = md_out.read_text(encoding="utf-8")
    assert "\u2014" not in md_text  # Zero em dashes invariant
    assert "Gemma-4-Loop-Compact" in md_text


def test_phase4_track_a_statistical_analysis_invariants(tmp_path: Path) -> None:
    import json

    repo_root = Path(__file__).resolve().parents[1]
    ta_path = repo_root / "results" / "phase4_native_inkling_nullbridge" / "track_a_results.json"
    tb_path = repo_root / "results" / "phase3_gemma4_nullbridge" / "track_b_results.json"
    p0_path = repo_root / "results" / "phase0_audit" / "benchmark_and_algebra_audit.json"

    analysis_a = compute_track_a_statistical_analysis(ta_path, tb_path, p0_path)
    assert analysis_a["architecture_summary"]["loop_layers"] == 12
    assert analysis_a["architecture_summary"]["k5_effective_blocks"] == 72
    assert analysis_a["looped_lens_deltas"]["varloop_p1_ridge_kl_delta_nats"] < -3.5
    assert analysis_a["looped_lens_deltas"]["inkling_p3_ridge_kl_delta_nats"] < -3.0
    assert analysis_a["native_multipass_dynamics"]["moral_jump_k1_to_k2_ratio"] > 22.0
    assert analysis_a["native_multipass_dynamics"]["covert_capture_drop_k1_to_k7_pp"] == -34.38
    assert analysis_a["channel_dissection_and_nullbridge"]["varloop_k4_qkv_vs_unclamped_cov_pp"] == -14.06
    assert analysis_a["coda_lora_backfire"]["k5_moral_ratio_vs_unclamped"] > 2.4

    md_out = tmp_path / "synthesis_a.md"
    write_phase4_synthesis_markdown(analysis_a, json.loads(ta_path.read_text(encoding="utf-8")), md_out)
    md_text = md_out.read_text(encoding="utf-8")
    assert "\u2014" not in md_text  # Zero em dashes invariant
    assert "Inkling-2-Mini" in md_text
