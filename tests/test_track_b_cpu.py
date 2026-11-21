"""CPU tests for Track B integer parsing, Dawn State SVG/HTML rendering, and module imports."""

from __future__ import annotations

import json
from pathlib import Path

from nullbridge.plotting import render_track_b_bundle
from nullbridge.track_b import _parse_int_pred


def test_parse_int_pred_truncates_turn_markers() -> None:
    assert _parse_int_pred("FINAL_INTEGER: -42<turn|>extra 999") == -42
    assert _parse_int_pred("<|channel>FINAL_INTEGER: 17<end_of_turn>88") == 17
    assert _parse_int_pred("  +105 ") == 105


def test_render_track_b_dawn_state_bundle(tmp_path: Path) -> None:
    def _mk_cell(k: int, overt: float, pull: float, hack_acc: float, bs_acc: float) -> dict:
        return {
            "n_loops": k,
            "overt_sycophancy": {"endorse_wrong_rate": overt, "mean_bias_log_odds": -1.2},
            "covert_sycophancy": {"sycophantic_pull_delta_lo": pull},
            "verifier_0cot": {"hacked_acc": hack_acc, "plain_acc": 0.78},
            "alias_guardrail": {"overall_acc": 0.85},
            "capabilities_and_safety": {
                "kl_eval_nats": 0.51,
                "multihop": {"overall_acc": 0.25},
                "balsheet": {"overall_acc": bs_acc},
                "cashflow_sanity_acc": 1.0,
                "moral_valuation": {"combined_bias_delta_lo": 0.12},
            },
        }

    conds = {}
    for idx, c in enumerate((
        "unclamped", "coda_lora", "full_lora", "bridge_only_nullbridge",
        "dual_nullbridge", "full_funnel_clamp", "random_subspace_control",
    )):
        conds[c] = {f"k{k}": _mk_cell(k, 0.30 - 0.03 * idx, 0.8 - 0.12 * idx, 0.55 + 0.05 * idx, 0.90) for k in (1, 2, 3, 4)}

    payload = {
        "metadata": {
            "base_model_id": "google/gemma-4-31B-it",
            "compact_storage": {"total_gb": 23.08},
        },
        "block1_subspace_geometry": {
            "bridge_fold_audit": {"max_null_leakage_r": 1.4e-8},
        },
        "block3_head_to_head_matrix": {"conditions": conds},
    }
    (tmp_path / "track_b_results.json").write_text(json.dumps(payload), encoding="utf-8")
    rendered = render_track_b_bundle(tmp_path)
    assert Path(rendered["synthesis_md"]).is_file()
    assert Path(rendered["index_html"]).is_file()
    assert "\u2014" not in Path(rendered["synthesis_md"]).read_text(encoding="utf-8")
    assert "\u2014" not in Path(rendered["index_html"]).read_text(encoding="utf-8")
