"""Dawn State / Warm Precision SVG chart, Markdown synthesis, and HTML report renderer for Track B."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_wp_figures_and_theme() -> tuple[Any, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    tokens_path = repo_root / "design-system" / "profiles" / "dawn-state" / "tokens.json"
    fig_script = repo_root / "design-system" / "scripts" / "figures.py"
    spec = importlib.util.spec_from_file_location("wp_figures", fig_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Warm Precision figures helper at {fig_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    profile = json.loads(tokens_path.read_text(encoding="utf-8"))
    return mod, mod.Theme(profile)


CONDITION_LABELS: dict[str, str] = {
    "unclamped": "Unclamped Loop",
    "coda_lora": "Coda LoRA (L43..L59)",
    "full_lora": "Full LoRA (L24..L59)",
    "bridge_only_nullbridge": "Bridge-Only R_safe (0 FLOPs)",
    "dual_nullbridge": "Dual Nullbridge (Boundary)",
    "full_funnel_clamp": "19-Layer Funnel Clamp",
    "random_subspace_control": "Random 4D Control",
}


def render_track_b_bundle(run_dir: Path) -> dict[str, str]:
    """Render Dawn State SVG charts, `synthesis.md`, and `index.html` from `track_b_results.json`."""
    res_path = run_dir / "track_b_results.json"
    data = json.loads(res_path.read_text(encoding="utf-8"))
    wp, theme = _load_wp_figures_and_theme()

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    b3 = data["block3_head_to_head_matrix"]["conditions"]
    k_focus = "k2"

    # 1. Overt vs. Covert Sycophancy Dissociation at K = 2
    overt_rows = [
        (CONDITION_LABELS[c], round(100.0 * b3[c][k_focus]["overt_sycophancy"]["endorse_wrong_rate"], 1))
        for c in ("unclamped", "coda_lora", "full_lora", "bridge_only_nullbridge", "dual_nullbridge", "random_subspace_control")
    ]
    svg_overt = wp.bar_chart(
        theme,
        overt_rows,
        id="overt-syc-k2",
        focus="Dual Nullbridge (Boundary)",
        sort=False,
        decimals=1,
        unit="%",
        kind="percent",
        title="Overt sycophancy endorsement rate at K = 2",
    )
    (fig_dir / "01_overt_sycophancy_k2.svg").write_text(svg_overt, encoding="utf-8")

    # 2. Multi-pass Covert Sycophantic Log-Odds Pull across K = 1..4
    x_k = ["K = 1", "K = 2", "K = 3", "K = 4"]
    cov_series = {
        CONDITION_LABELS[c]: [
            b3[c][f"k{k}"]["covert_sycophancy"]["sycophantic_pull_delta_lo"] for k in (1, 2, 3, 4)
        ]
        for c in ("unclamped", "coda_lora", "full_lora", "bridge_only_nullbridge", "dual_nullbridge")
    }
    svg_cov = wp.line_chart(
        theme,
        x_k,
        cov_series,
        id="covert-pull-k",
        focus="Dual Nullbridge (Boundary)",
        decimals=2,
        unit=" LO",
        title="Covert numeric sycophantic pull across K = 1..4",
    )
    (fig_dir / "02_covert_sycophancy_across_k.svg").write_text(svg_cov, encoding="utf-8")

    # 3. Deliberative 0-CoT Verifier Accuracy under Reward-Hacked Gloss across K = 1..4
    ver_series = {
        CONDITION_LABELS[c]: [
            round(100.0 * b3[c][f"k{k}"]["verifier_0cot"]["hacked_acc"], 1) for k in (1, 2, 3, 4)
        ]
        for c in ("unclamped", "coda_lora", "full_lora", "bridge_only_nullbridge", "dual_nullbridge")
    }
    svg_ver = wp.line_chart(
        theme,
        x_k,
        ver_series,
        id="verifier-hacked-k",
        focus="Dual Nullbridge (Boundary)",
        decimals=1,
        unit="%",
        kind="percent",
        title="Single-pass 0-CoT verifier accuracy on reward-hacked rollouts across K = 1..4",
    )
    (fig_dir / "03_verifier_reward_hack_across_k.svg").write_text(svg_ver, encoding="utf-8")

    # 4. Capability Preservation (Balance-Sheet & Multi-Hop at K = 2): Boundary Nullbridge vs. 19-Layer Clamp
    cap_rows = [
        (CONDITION_LABELS[c], round(100.0 * b3[c]["k2"]["capabilities_and_safety"]["balsheet"]["overall_acc"], 1))
        for c in ("unclamped", "coda_lora", "full_lora", "bridge_only_nullbridge", "dual_nullbridge", "full_funnel_clamp")
    ]
    svg_cap = wp.bar_chart(
        theme,
        cap_rows,
        id="balsheet-k2",
        focus="Dual Nullbridge (Boundary)",
        sort=False,
        decimals=1,
        unit="%",
        kind="percent",
        title="Corporate 0%-loan balance-sheet accuracy at K = 2",
    )
    (fig_dir / "04_balsheet_capability_k2.svg").write_text(svg_cap, encoding="utf-8")

    # Build concise Warm Precision Markdown synthesis (zero em dashes!)
    md_lines = [
        "# Track B Synthesis: Behavioral LoRA vs. Algebraic Nullbridge on Gemma-4-Loop-Compact",
        "",
        f"- **Model:** `{data['metadata']['base_model_id']}` (`{data['metadata']['compact_storage']['total_gb']} GB` Hourglass INT4/INT8 + Procrustes `L24..L42` recurrent funnel)",
        f"- **Subspace Removed:** `4` controlled bias directions (`sycophancy`, `authority_framing`, `confident_formatting`, `moral_spite`), QR-orthogonalized against `epistemic_evidence` and `affect_valence` (`4 / 5376 = 0.074%` of residual dimensions; `99.926%` retained)",
        f"- **Bridge Fold Leakage:** `||R_safe Q||_F = {data['block1_subspace_geometry']['bridge_fold_audit']['max_null_leakage_r']:.2e}` (`0` extra parameters, `0` extra runtime FLOPs)",
        "",
        "## Head-to-Head Scorecard at `K = 2` and `K = 4` (`0` CoT Tokens)",
        "",
        "| Condition | Loops | Overt Sycophancy | Covert Pull (LO) | Verifier (Hacked) | Alias Guardrail | Multi-Hop | Balance-Sheet | Cashflow Sanity | Moral Bias (LO) | KL (nats) |",
        "| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for c_key in ("unclamped", "coda_lora", "full_lora", "bridge_only_nullbridge", "dual_nullbridge", "full_funnel_clamp", "random_subspace_control"):
        for k_num in (1, 2, 4):
            cell = b3[c_key][f"k{k_num}"]
            cap = cell["capabilities_and_safety"]
            md_lines.append(
                f"| {CONDITION_LABELS[c_key]} | `{k_num}` | "
                f"`{cell['overt_sycophancy']['endorse_wrong_rate']*100:.1f}%` | "
                f"`{cell['covert_sycophancy']['sycophantic_pull_delta_lo']:+.3f}` | "
                f"`{cell['verifier_0cot']['hacked_acc']*100:.1f}%` | "
                f"`{cell['alias_guardrail']['overall_acc']*100:.1f}%` | "
                f"`{cap['multihop']['overall_acc']*100:.1f}%` | "
                f"`{cap['balsheet']['overall_acc']*100:.1f}%` | "
                f"`{cap['cashflow_sanity_acc']*100:.1f}%` | "
                f"`{cap['moral_valuation']['combined_bias_delta_lo']:+.3f}` | "
                f"`{cap['kl_eval_nats']:.4f}` |"
            )

    md_path = run_dir / "synthesis.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    # Standalone Dawn State HTML report
    fig1_html = wp.figure(
        "Behavioral LoRAs and boundary Nullbridge both suppress explicit verbal sycophancy.",
        "Wrong-claim endorsement rate at K = 2 loops, %, n = 48 counterbalanced A/B items",
        svg_overt,
        "Source: Nullbridge Track B held-out overt sycophancy suite on Gemma-4-Loop-Compact.",
    )
    fig2_html = wp.figure(
        "Covert numeric sycophantic pull across K = 1..4 loops.",
        "Teacher-forced log-odds shift toward authoritative wrong integer (pressured minus neutral), log-odds, n = 64",
        svg_cov,
        "Source: Nullbridge Track B held-out covert numeric sycophancy suite.",
    )
    fig3_html = wp.figure(
        "Single-forward-pass 0-CoT verifier accuracy on confident-gloss reward-hacked rollouts.",
        "Counterbalanced accuracy selecting valid code/ledger/trace over confident buggy decoy, %, n = 96",
        svg_ver,
        "Source: Nullbridge Track B held-out deliberative 0-CoT verifier suite.",
    )
    fig4_html = wp.figure(
        "Boundary Nullbridge preserves corporate balance-sheet reasoning where 19-layer clamping degrades it.",
        "Exact-match 0-CoT accuracy on 0%-loan balance-sheet net-worth at K = 2, %, n = 32",
        svg_cap,
        "Source: Nullbridge Track B held-out corporate balance-sheet suite.",
    )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Nullbridge Track B: Gemma-4-Loop-Compact Report</title>
<style>
  :root {{
    --canvas: #FFFFF0; --well: #F5EEDC; --ink: #0F0F0F; --slate: #2C3E50; --deep: #1E3447;
  }}
  body {{
    background: var(--canvas); color: var(--ink); font-family: Inter, Arial, sans-serif;
    font-size: 18px; line-height: 1.65; max-width: 1000px; margin: 0 auto; padding: 48px 24px;
  }}
  h1 {{ font-family: Newsreader, Georgia, serif; font-size: 40px; letter-spacing: -0.04em; margin-bottom: 8px; }}
  .lede {{ color: var(--deep); font-size: 20px; margin-top: 0; margin-bottom: 36px; }}
  .wp-figure {{ background: var(--canvas); border-top: 2px solid var(--slate); padding: 20px 0 28px; margin: 32px 0; }}
  .wp-figure-title {{ font-weight: 600; font-size: 20px; margin: 0 0 4px; }}
  .wp-figure-subtitle {{ color: var(--deep); font-size: 16px; margin: 0 0 16px; }}
  figcaption {{ color: var(--deep); font-size: 14px; margin-top: 12px; }}
</style>
</head>
<body>
  <h1>Nullbridge Track B: Behavioral LoRA vs. Algebraic Bridge Null-Space</h1>
  <p class="lede">Head-to-head evaluation on Gemma-4-Loop-Compact (23.1 GB, K = 1..4 silent recurrent passes) across overt verbal sycophancy, covert numeric anchoring, single-pass 0-CoT verifiers, and multi-hop capabilities.</p>
  {fig1_html}
  {fig2_html}
  {fig3_html}
  {fig4_html}
</body>
</html>
"""
    html_path = run_dir / "index.html"
    html_path.write_text(html_doc, encoding="utf-8")
    return {"synthesis_md": str(md_path), "index_html": str(html_path), "figures_dir": str(fig_dir)}
