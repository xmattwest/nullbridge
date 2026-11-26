"""Dawn State SVG & 2x Retina PNG Figure Suite for Phase 7 (`phase7_biophysics`).

Renders 6 publication-grade figures via `design-system/scripts/figures.py` and
`design-system/profiles/dawn-state/tokens.json`, exporting `.svg`, 2x `.png`,
and an inspectable `index.html` gallery into `figures/phase7_biophysics/`.
Zero em dashes anywhere.
"""

from __future__ import annotations

import html
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Any


EM_DASH = "\u2014"
_INNER_SVG_RE = re.compile(r"<svg[^>]*>(.*)</svg>", re.DOTALL)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_wp_figures() -> tuple[Any, dict[str, Any], Any]:
    root = _repo_root()
    tokens_path = root / "design-system" / "profiles" / "dawn-state" / "tokens.json"
    fig_script = root / "design-system" / "scripts" / "figures.py"
    profile = json.loads(tokens_path.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("wp_figures_p7", fig_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {fig_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod, profile, mod.Theme(profile)


def _strip_outer_svg(svg_str: str) -> str:
    m = _INNER_SVG_RE.search(svg_str)
    return m.group(1) if m is not None else svg_str


def _two_panel_svg(
    theme: Any,
    fig_id: str,
    title_desc: str,
    panel_a_title: str,
    svg_a: str,
    panel_b_title: str,
    svg_b: str,
    panel_w: int = 420,
    panel_h: int = 330,
    gap: int = 28,
) -> str:
    """Compose two `figures.py` SVG charts side-by-side in one accessible Dawn State SVG."""
    total_w = panel_w * 2 + gap
    top_y = 28
    total_h = panel_h + top_y + 8
    font = html.escape(theme.body)
    esc = html.escape
    inner_a = _strip_outer_svg(svg_a)
    inner_b = _strip_outer_svg(svg_b)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" class="wp-chart" viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}" role="img" aria-labelledby="{fig_id}-t {fig_id}-d">'
        f'<title id="{fig_id}-t">{esc(title_desc)}</title>'
        f'<desc id="{fig_id}-d">{esc(panel_a_title)}; {esc(panel_b_title)}</desc>'
        f'<text x="0" y="18" fill="{theme["text-primary"]}" font-family="{font}" font-size="14.5" font-weight="600">'
        f'{esc(panel_a_title)}</text>'
        f'<g transform="translate(0, {top_y})">{inner_a}</g>'
        f'<text x="{panel_w + gap}" y="18" fill="{theme["text-primary"]}" font-family="{font}" font-size="14.5" font-weight="600">'
        f'{esc(panel_b_title)}</text>'
        f'<g transform="translate({panel_w + gap}, {top_y})">{inner_b}</g>'
        f'</svg>'
    )


def _find_chrome_bin() -> str | None:
    env_bin = os.environ.get("CHROME_BIN")
    if env_bin and Path(env_bin).exists():
        return env_bin
    for cand in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ):
        if cand and Path(cand).exists():
            return cand
    return None


def _rasterize_figure_card_png(
    chrome_bin: str,
    theme: Any,
    title: str,
    subtitle: str,
    svg_markup: str,
    source_line: str,
    png_path: Path,
    width: int = 920,
    height: int = 660,
) -> bool:
    """Render a self-contained Warm Precision figure card (title + subtitle + SVG + source) to 2x PNG."""
    card_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: {width}px;
    background: {theme["canvas"]};
    color: {theme["text-primary"]};
    font-family: {theme.body};
    -webkit-font-smoothing: antialiased;
  }}
  .card {{
    padding: 28px 32px 24px;
    background: {theme["canvas"]};
  }}
  .title {{
    font-size: 20px;
    font-weight: 600;
    line-height: 1.3;
    letter-spacing: -0.015em;
    color: {theme["text-primary"]};
    margin-bottom: 6px;
  }}
  .subtitle {{
    font-size: 15px;
    line-height: 1.45;
    color: {theme["text-secondary"]};
    margin-bottom: 18px;
  }}
  .plot svg {{
    width: 100%;
    height: auto;
    display: block;
  }}
  .source {{
    margin-top: 14px;
    padding-top: 10px;
    border-top: 1px solid rgba(44, 62, 80, 0.2);
    font-size: 13px;
    color: {theme["text-secondary"]};
  }}
</style>
</head>
<body>
  <div class="card" id="card">
    <p class="title">{html.escape(title)}</p>
    <p class="subtitle">{html.escape(subtitle)}</p>
    <div class="plot">{svg_markup}</div>
    <p class="source">{html.escape(source_line)}</p>
  </div>
</body>
</html>
"""
    with tempfile.TemporaryDirectory() as td:
        html_p = Path(td) / "card.html"
        raw_png = Path(td) / "raw.png"
        html_p.write_text(card_html, encoding="utf-8")
        proc = subprocess.run(
            [
                chrome_bin,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--hide-scrollbars",
                f"--window-size={width},{height}",
                "--force-device-scale-factor=2",
                f"--screenshot={raw_png}",
                html_p.as_uri(),
            ],
            capture_output=True,
            timeout=45,
            check=False,
        )
        if proc.returncode != 0 or not raw_png.is_file():
            return False

        try:
            import numpy as np
            from PIL import Image

            im = Image.open(raw_png).convert("RGB")
            arr = np.asarray(im)
            bg = arr[4, 4]
            diff = abs(arr.astype(int) - bg.astype(int)).max(axis=2) > 6
            ys = [idx for idx, v in enumerate(diff.any(axis=1)) if v]
            if ys:
                bottom = min(im.height, max(ys) + 48)
                im = im.crop((0, 0, im.width, bottom))
            png_path.parent.mkdir(parents=True, exist_ok=True)
            im.save(png_path, format="PNG", optimize=True)
            return True
        except Exception:
            shutil.copy2(raw_png, png_path)
            return True


def render_phase7_figure_suite(
    results_json_path: Path = Path("results/phase7_biophysics/phase7_biophysics_results.json"),
    phase6_results_json_path: Path = Path("results/phase6_inkling_biophysical_loop/phase6_biophysical_results.json"),
    out_dir: Path = Path("figures/phase7_biophysics"),
) -> dict[str, Any]:
    """Render 6 Dawn State `.svg` + 2x `.png` figures and `index.html` for Phase 7."""
    mod, _profile, theme = _load_wp_figures()
    p7 = json.loads(Path(results_json_path).read_text(encoding="utf-8"))
    p6 = json.loads(Path(phase6_results_json_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrome_bin = _find_chrome_bin()

    ink_a = p7["inkling_block_a"]
    var_b1 = p7["varloop_block_b1"]
    hug_b2 = p7["huginn_block_b2"]

    svg_files: list[str] = []
    png_files: list[str] = []
    figure_blocks_html: list[str] = []

    def _emit(stem: str, title: str, subtitle: str, svg_str: str, source: str, height_px: int = 640) -> None:
        for txt in (title, subtitle, svg_str, source):
            if EM_DASH in txt:
                raise ValueError(f"Em dash found in {stem}")
        svg_name = f"{stem}.svg"
        (out_dir / svg_name).write_text(svg_str, encoding="utf-8")
        svg_files.append(svg_name)
        if chrome_bin is not None:
            png_name = f"{stem}.png"
            if _rasterize_figure_card_png(
                chrome_bin, theme, title, subtitle, svg_str, source, out_dir / png_name, width=920, height=height_px
            ):
                png_files.append(png_name)
        figure_blocks_html.append(mod.figure(title, subtitle, svg_str, source))

    # -----------------------------------------------------------------------
    # Fig 1: Exact 16-Mask WHT Synaptic Epistasis Heatmap (Reasoning vs. Safety)
    # -----------------------------------------------------------------------
    wht_terms = ("Q", "K", "V", "MLP", "QxK", "VxMLP", "QxV", "KxV", "QxMLP", "KxMLP")
    p6_e1 = p6["exp1_wht_synaptic_knockouts"]["schedules"]["joint_p2_p3"]
    p7_wht_suites = ink_a["exp_a2_safety_wht"]["suites"]
    heat_sources = [
        ("2-hop factual (reason)", p6_e1["by_regime"]["factual_2hop"]["continuation_logprob_wht"]),
        ("2-step arith (reason)", p6_e1["by_regime"]["arithmetic_2step"]["continuation_logprob_wht"]),
        ("All 120 reasoning", p6_e1["all_120"]["continuation_logprob_wht"]),
        ("Moral bias (safety)", p7_wht_suites["moral_valuation"]["moral_bias_delta_lo_wht"]),
        ("Verifier hack (safety)", p7_wht_suites["verifier"]["reward_hack_susceptibility_lo_wht"]),
        ("Covert anchor (safety)", p7_wht_suites["covert_sycophancy"]["sycophantic_pull_delta_lo_wht"]),
    ]
    heat_rows = [lbl for lbl, _ in heat_sources]
    heat_vals = [[float(src["terms"][t]["coalition_effect"]) for t in wht_terms] for _, src in heat_sources]

    svg1 = mod.heatmap(
        theme,
        heat_rows,
        list(wht_terms),
        heat_vals,
        id="fig1-wht-reasoning-vs-safety",
        diverging=True,
        decimals=2,
        width=840,
        cell=56,
        legend_label="Walsh-Hadamard coalition effect (nats or log-odds)",
        title="16-mask Walsh-Hadamard main effects and pairwise epistasis across reasoning and safety",
    )
    _emit(
        "fig1_wht_reasoning_vs_safety",
        "Reasoning relies on superadditive QxK (+0.89) with redundant VxMLP (-0.91), while safety bias loads onto V+MLP and positive VxMLP.",
        "Exact 16-mask Walsh-Hadamard coalition effects on Inkling-2-Mini (K = 3, Parseval error <= 8.88e-16).",
        svg1,
        "Source: Phase 6 Exp 1 (n = 120 reasoning) and Phase 7 Exp A2 (n = 64 covert, n = 64 moral, n = 96 verifier).",
        height_px=640,
    )

    # -----------------------------------------------------------------------
    # Fig 2: Non-Normal Transient Amplification Across Models & Depths
    # -----------------------------------------------------------------------
    p6_amp = p6["exp2_gain_vs_amplitude"]["non_normal_pass_amplification"]
    b1_amp = var_b1["per_pass_non_normal_gain"]["by_pass"]
    b2_amp = hug_b2["exp_h2_gain_vs_amplitude"]["non_normal_step_amplification"]

    x_steps = ["1", "2", "3", "4", "5", "6", "7", "8"]
    ink_sigma = [
        float(p6_amp["pass_1"]["sigma_max_vjp"]["mean"]),
        float(p6_amp["pass_2"]["sigma_max_vjp"]["mean"]),
        float(p6_amp["pass_3"]["sigma_max_vjp"]["mean"]),
        None,
        None,
        None,
        None,
        None,
    ]
    var_sigma = [float(b1_amp[f"pass_{p}"]["sigma_max_vjp"]["mean"]) for p in range(1, 8)] + [None]
    hug_sigma = [
        float(b2_amp["step_1"]["sigma_max_vjp"]["mean"]),
        float(b2_amp["step_2"]["sigma_max_vjp"]["mean"]),
        float(b2_amp["step_3"]["sigma_max_vjp"]["mean"]),
        float(b2_amp["step_4"]["sigma_max_vjp"]["mean"]),
        None,
        float(b2_amp["step_6"]["sigma_max_vjp"]["mean"]),
        None,
        float(b2_amp["step_8"]["sigma_max_vjp"]["mean"]),
    ]
    ink_iso = [
        float(p6_amp["pass_1"]["isotropic_rms_gain"]["mean"]),
        float(p6_amp["pass_2"]["isotropic_rms_gain"]["mean"]),
        float(p6_amp["pass_3"]["isotropic_rms_gain"]["mean"]),
        None,
        None,
        None,
        None,
        None,
    ]
    var_iso = [float(b1_amp[f"pass_{p}"]["isotropic_rms_gain"]["mean"]) for p in range(1, 8)] + [None]
    hug_iso = [
        float(b2_amp["step_1"]["isotropic_rms_gain"]["mean"]),
        float(b2_amp["step_2"]["isotropic_rms_gain"]["mean"]),
        float(b2_amp["step_3"]["isotropic_rms_gain"]["mean"]),
        float(b2_amp["step_4"]["isotropic_rms_gain"]["mean"]),
        None,
        float(b2_amp["step_6"]["isotropic_rms_gain"]["mean"]),
        None,
        float(b2_amp["step_8"]["isotropic_rms_gain"]["mean"]),
    ]

    svg2_a = mod.line_chart(
        theme,
        x_steps,
        {
            "Inkling-2-Mini": ink_sigma,
            "Varloop (K=1..7)": var_sigma,
            "Huginn-0125": hug_sigma,
        },
        id="fig2a-sigma-max",
        focus="Inkling-2-Mini",
        width=420,
        height=330,
        decimals=1,
        y_min=0,
        y_max=15,
        title="Top singular value sigma_max across recurrent passes",
    )
    svg2_b = mod.line_chart(
        theme,
        x_steps,
        {
            "Huginn-0125": hug_iso,
            "Inkling-2-Mini": ink_iso,
            "Varloop (K=1..7)": var_iso,
        },
        id="fig2b-iso-rms",
        focus="Huginn-0125",
        width=420,
        height=330,
        decimals=2,
        y_min=0,
        y_max=1.0,
        title="Contractive isotropic RMS gain across recurrent passes",
    )
    svg2 = _two_panel_svg(
        theme,
        "fig2-non-normal-cross-arch",
        "Non-normal singular amplification vs contractive isotropic RMS across three looped architectures",
        "A. Top singular gain sigma_max by pass / step",
        svg2_a,
        "B. Isotropic RMS gain (all < 1.0 contractive)",
        svg2_b,
    )
    _emit(
        "fig2_non_normal_cross_architecture",
        "All three looped architectures pair contractive isotropic RMS (0.17 to 0.73) with strong non-normal singular gain (4.5x to 13.6x).",
        "Left: Exact VJP top singular value sigma_max across passes/steps (n = 36). Right: Isotropic random-tangent RMS gain stays strictly below 1.0.",
        svg2,
        "Source: Phase 6 Exp 2 (Inkling-2-Mini) and Phase 7 Blocks B1 (Varloop K = 1..7) & B2 (Huginn-0125 K = 1..8).",
        height_px=590,
    )

    # -----------------------------------------------------------------------
    # Fig 3: Pass-2 Singular-Mode Alignment & Selective Zero-FLOP Pareto Frontier
    # -----------------------------------------------------------------------
    a1_vh = ink_a["exp_a1_singular_mode_alignment"]["cohorts"]["verifier_paired"]["hack"]["by_concept"]
    a1_cp = ink_a["exp_a1_singular_mode_alignment"]["cohorts"]["covert_anchor_paired"]["pressured"]["by_concept"]
    a1_mt = ink_a["exp_a1_singular_mode_alignment"]["cohorts"]["moral_valuation_paired"]["tainted"]["by_concept"]

    align_bars = [
        ("Authority (pressured, V8)", float(a1_cp["authority_framing"]["E_k_right"]["enrichment_over_null"])),
        ("Formatting (hack, V8)", float(a1_vh["confident_formatting"]["E_k_right"]["enrichment_over_null"])),
        ("Sycophancy (pressured, V8)", float(a1_cp["sycophancy"]["E_k_right"]["enrichment_over_null"])),
        ("Moral spite (tainted, V8)", float(a1_mt["moral_spite"]["E_k_right"]["enrichment_over_null"])),
        ("Formatting (hack, U8)", float(a1_vh["confident_formatting"]["E_k_left"]["enrichment_over_null"])),
        ("Authority (pressured, U8)", float(a1_cp["authority_framing"]["E_k_left"]["enrichment_over_null"])),
    ]
    svg3_a = mod.bar_chart(
        theme,
        align_bars,
        id="fig3a-singular-enrichment",
        focus="Authority (pressured, U8)",
        sort=False,
        width=420,
        decimals=1,
        unit="x",
        title="Pass-2 top-8 singular subspace energy enrichment over isotropic null",
    )

    pareto_conds = ink_a["exp_a2_selective_nullbridge"]["conditions"]
    label_map = {
        "unclamped": "Unclamped",
        "zff_Q_only": "Q only",
        "zff_QK": "QK",
        "zff_QKV": "QKV",
        "zff_MLP_only": "MLP only",
        "zff_V_MLP": "V+MLP",
        "zff_All4": "All-4",
    }
    pareto_pts = [
        (
            float(pareto_conds[k]["kl_eval_nats"]) * 1000.0,
            float(pareto_conds[k]["covert_capture_rate"]) * 100.0,
            label_map[k],
        )
        for k in ("unclamped", "zff_Q_only", "zff_QK", "zff_QKV", "zff_MLP_only", "zff_V_MLP", "zff_All4")
    ]
    svg3_b = mod.scatter_plot(
        theme,
        pareto_pts,
        id="fig3b-pareto-frontier",
        x_label="Held-out KL shift (millinats)",
        y_label="Covert wrong capture (%)",
        width=420,
        height=330,
        x_decimals=1,
        y_decimals=1,
        y_unit="%",
        focus="All-4",
        x_min=-0.1,
        x_max=2.1,
        y_min=66.0,
        y_max=76.0,
        title="Zero-FLOP channel-selective Nullbridge Pareto frontier on Inkling-2-Mini",
    )
    svg3 = _two_panel_svg(
        theme,
        "fig3-alignment-pareto",
        "Pass-2 top-8 singular mode alignment and zero-FLOP Nullbridge Pareto frontier",
        "A. Top-8 singular energy enrichment vs. isotropic null (x)",
        svg3_a,
        "B. Zero-FLOP fold: KL (millinats) vs. covert capture (%)",
        svg3_b,
    )
    _emit(
        "fig3_singular_alignment_and_pareto",
        "Safety directions sit nearly orthogonal to Pass-2 input modes V_8 (1.6x to 3.6x null), enabling zero-FLOP erasure at <= 1.7 millinats KL.",
        "Left: Top-8 right (V_8) vs. left (U_8) singular energy enrichment on Inkling-2-Mini (n = 32/cohort). Right: Channel-selective zero-FLOP Nullbridge Pareto sweep (n = 64).",
        svg3,
        "Source: Nullbridge Phase 7 Block A (`thinkingmachines/Inkling-2-Mini`, Exp A1 & Exp A2.2).",
        height_px=600,
    )

    # -----------------------------------------------------------------------
    # Fig 4: Loop-to-Coda Fisher Pullback Spike & Hop-1 -> Hop-2 Commitment Horizon
    # -----------------------------------------------------------------------
    f6 = p6["exp3_fisher_and_coda_shunt"]["fisher_pullback_trajectory"]
    f6_ratio = next(r["fisher_trace"]["mean"] for r in f6 if r["stage_key"] == "coda_L18") / next(
        r["fisher_trace"]["mean"] for r in f6 if r["stage_key"] == "p3_L17"
    )
    f5 = var_b1["fisher_and_commitment"]["by_depth"]["K=5"]["fisher_pullback_trajectory"]
    f5_ratio = next(r["fisher_trace"]["mean"] for r in f5 if r["stage_key"] == "coda_L18") / next(
        r["fisher_trace"]["mean"] for r in f5 if r["stage_key"] == "p5_L17"
    )
    f7 = var_b1["fisher_and_commitment"]["by_depth"]["K=7"]["fisher_pullback_trajectory"]
    f7_ratio = next(r["fisher_trace"]["mean"] for r in f7 if r["stage_key"] == "coda_L18") / next(
        r["fisher_trace"]["mean"] for r in f7 if r["stage_key"] == "p7_L17"
    )
    fh_ratio = float(hug_b2["exp_h3_fisher_and_commitment"]["loop_to_coda_fisher_spike"]["trace_ratio"])

    fisher_bars = [
        ("Varloop K=5 (6L Coda)", round(f5_ratio, 1)),
        ("Varloop K=7 (6L Coda)", round(f7_ratio, 1)),
        ("Inkling K=3 (6L Coda)", round(f6_ratio, 1)),
        ("Huginn K=8 (2L Coda)", round(fh_ratio, 2)),
    ]
    svg4_a = mod.bar_chart(
        theme,
        fisher_bars,
        id="fig4a-fisher-spike",
        focus="Varloop K=5 (6L Coda)",
        sort=False,
        width=420,
        decimals=1,
        unit="x",
        title="Loop-exit to Coda-entry Fisher pullback trace jump ratio",
    )

    comm_a = p6["exp3_fisher_and_coda_shunt"]["commitment_horizon"]["arithmetic_2step"]
    comm_f = p6["exp3_fisher_and_coda_shunt"]["commitment_horizon"]["factual_2hop"]
    svg4_b = mod.line_chart(
        theme,
        ["Stem", "Pass 1", "Pass 2", "Pass 3"],
        {
            "Arith Hop-2 final": [x * 100.0 for x in comm_a["hop2_target_prob"]],
            "Factual Hop-2 capital": [x * 100.0 for x in comm_f["hop2_target_prob"]],
            "Arith Hop-1 bridge": [x * 100.0 for x in comm_a["hop1_bridge_prob"]],
        },
        id="fig4b-commitment",
        focus="Arith Hop-2 final",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        title="Two-stage Hop-1 to Hop-2 commitment horizon on Inkling-2-Mini",
    )
    svg4 = _two_panel_svg(
        theme,
        "fig4-fisher-commitment",
        "Deep 6-layer vs shallow 2-layer Coda Fisher jump and Hop-1 to Hop-2 commitment",
        "A. Loop->Coda Fisher trace jump ratio (6L vs 2L Coda)",
        svg4_a,
        "B. Hop-1 staging (Pass 2) to Hop-2 lock-in (Pass 3, %)",
        svg4_b,
    )
    _emit(
        "fig4_fisher_pullback_and_commitment",
        "Deep 6-layer Codas trigger 87.9x to 141.7x Fisher spikes where 2-layer Huginn stays smooth (0.77x), while arithmetic relays Hop-1 to Hop-2.",
        "Left: Hutchinson VJP Fisher trace ratio across the loop-to-Coda boundary (n = 60). Right: Inkling-2-Mini Coda-transported Hop-1 vs. Hop-2 probability (%).",
        svg4,
        "Source: Phase 6 Exp 3 (Inkling-2-Mini) and Phase 7 Blocks B1 (Varloop) & B2 (Huginn-0125).",
        height_px=590,
    )

    # -----------------------------------------------------------------------
    # Fig 5: Somatic-Reset Hysteresis Shielding vs. Continuous-State Recurrence
    # -----------------------------------------------------------------------
    ink_foil = p6["exp4_hysteresis_pulse_chase"]["foil_hysteresis_sweep"]
    var_foil = var_b1["somatic_reset_hysteresis"]["by_depth"]["K=5"]["curve"]
    hug_foil = hug_b2["exp_h4_hysteresis_pulse_chase"]["by_depth"]["K=8"]["foil_curve"]

    betas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    x_beta = [f"{b:.2f}" for b in betas]

    def _interp_beta(curve: list[dict[str, Any]], key: str) -> list[float]:
        pts = sorted((float(r["beta"]), float(r[key]) * 100.0) for r in curve)
        out: list[float] = []
        for b in betas:
            exact = [v for bb, v in pts if abs(bb - b) < 1e-6]
            if exact:
                out.append(exact[0])
            else:
                lo = max((p for p in pts if p[0] < b), key=lambda p: p[0])
                hi = min((p for p in pts if p[0] > b), key=lambda p: p[0])
                frac = (b - lo[0]) / (hi[0] - lo[0])
                out.append(lo[1] + frac * (hi[1] - lo[1]))
        return out

    svg5_a = mod.line_chart(
        theme,
        x_beta,
        {
            "Inkling K=3 (reset)": _interp_beta(ink_foil, "coda_foil_capture_rate"),
            "Varloop K=5 (reset)": _interp_beta(var_foil, "coda_foil_capture_rate"),
            "Huginn K=8 (recall)": _interp_beta(hug_foil, "coda_foil_capture_rate"),
        },
        id="fig5a-foil-capture",
        focus="Inkling K=3 (reset)",
        width=420,
        height=330,
        decimals=1,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=30,
        title="Foil capture rate across Pass-1 foil interpolation beta",
    )

    p6_pulse = next(r for r in p6["exp4_hysteresis_pulse_chase"]["orthogonal_pulse_chase_sweep"] if abs(r["alpha"] - 1.0) < 1e-6)
    h4_pulse = next(r for r in hug_b2["exp_h4_hysteresis_pulse_chase"]["orthogonal_pulse_chase"] if abs(r["alpha"] - 1.0) < 1e-6)
    svg5_b = mod.line_chart(
        theme,
        ["Inject (P1/S1)", "Next (P2/S2)", "Final loop", "After Coda"],
        {
            "Inkling-2-Mini": [
                100.0,
                float(p6_pulse["rel_deviation_pass2"]) * 100.0,
                float(p6_pulse["rel_deviation_pass3"]) * 100.0,
                float(p6_pulse["rel_deviation_coda"]) * 100.0,
            ],
            "Huginn-0125": [
                100.0,
                float(h4_pulse["rel_deviation_step2"]) * 100.0,
                float(h4_pulse["rel_deviation_step8"]) * 100.0,
                float(h4_pulse["rel_deviation_coda"]) * 100.0,
            ],
        },
        id="fig5b-pulse-damping",
        focus="Huginn-0125",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        title="Orthogonal tangent pulse-chase relative deviation (alpha = 1.0)",
    )
    svg5 = _two_panel_svg(
        theme,
        "fig5-hysteresis-shielding",
        "Foil hysteresis shielding and pulse-chase damping across somatic-reset and continuous-state loops",
        "A. Foil capture rate across beta in [0, 1.25] (no bifurcation)",
        svg5_a,
        "B. Orthogonal pulse damping (alpha = 1.0, % relative norm)",
        svg5_b,
    )
    _emit(
        "fig5_hysteresis_somatic_vs_continuous",
        "Both somatic reset (Inkling 0% capture, Varloop 15%) and linear prelude recall (Huginn 10-12%) prevent foil bifurcation and damp 1.0x pulses.",
        "Left: Coda foil capture rate across Pass-1 foil interpolation beta (%, n = 60; beta_c = None in all 3 models). Right: Orthogonal pulse decay (alpha = 1.0).",
        svg5,
        "Source: Phase 6 Exp 4 (Inkling-2-Mini) and Phase 7 Blocks B1 (Varloop) & B2 (Huginn-0125).",
        height_px=590,
    )

    # -----------------------------------------------------------------------
    # Fig 6: Fixed-Weight Neuromodulatory Pass-Clock (gamma) & Coda Header Shunt
    # -----------------------------------------------------------------------
    p6_gm = p6["exp5_neuromodulatory_switching"]["allosteric_conductance_sweep"]
    h5_gm = hug_b2["exp_h5_adapter_conductance_sweep"]["adapter_conductance_sweep"]
    x_gm = [f"{r['gamma']:.2f}" for r in h5_gm]
    svg6_a = mod.line_chart(
        theme,
        x_gm,
        {
            "Inkling-2-Mini (K=3)": [float(r["hop2_top1_acc"]) * 100.0 for r in p6_gm],
            "Huginn-0125 (K=8)": [float(r["hop2_top1_acc"]) * 100.0 for r in h5_gm],
        },
        id="fig6a-gamma-inverted-u",
        focus="Inkling-2-Mini (K=3)",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=80,
        tick_every=2,
        title="Inverted-U recurrent conductance curve across gamma in [0.00, 1.50]",
    )

    hm0 = p6["exp5_neuromodulatory_switching"]["coda_header_mode_switch"]["effort_0"]["conditions"]
    shunt_bars = [
        ("Unmodified header", float(hm0["baseline_header_L23"]["format_token_prob"]) * 100.0),
        ("Slot-0 swap", float(hm0["slot0_embed_swap_L23"]["format_token_prob"]) * 100.0),
        ("Slot-0 + Slot-1 swap", float(hm0["slot0_and_slot1_somatic_swap_L23"]["format_token_prob"]) * 100.0),
        ("Carrier + loop delta", float(hm0["somatic_carrier_loop_transplant_L23"]["format_token_prob"]) * 100.0),
    ]
    svg6_b = mod.bar_chart(
        theme,
        shunt_bars,
        id="fig6b-coda-shunt-swap",
        focus="Carrier + loop delta",
        sort=False,
        width=420,
        decimals=1,
        unit="%",
        kind="percent",
        title="Inkling Coda L23 format-token shunt mass under somatic swaps",
    )
    svg6 = _two_panel_svg(
        theme,
        "fig6-neuromodulation-shunt",
        "Universal inverted-U conductance gain curve and Coda header shunt abolition",
        "A. Conductance gain gamma sweep (Hop-2 Top-1 acc, %)",
        svg6_a,
        "B. Inkling Coda L23 format-token shunt mass (%, effort 0)",
        svg6_b,
    )
    _emit(
        "fig6_neuromodulatory_clock_and_coda_shunt",
        "Both Inkling (70.0% at gamma = 1.00) and Huginn (41.7% at gamma = 0.75) follow inverted-U gain curves, while Coda swaps cut shunt mass 99.4%.",
        "Left: Recurrent message/state conductance scale gamma in [0.00, 1.50] (%, n = 60). Right: Format-token probability at <|message_model|> (%, n = 40).",
        svg6,
        "Source: Phase 6 Exp 5 (Inkling-2-Mini) and Phase 7 Exp H5 (Huginn-0125).",
        height_px=590,
    )

    gallery_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Biophysics of a Looped Transformer · Phase 7 Dawn State Figures</title>
  <style>
    :root {{
      --canvas: {theme["canvas"]};
      --well: {theme["well"]};
      --ink: {theme["text-primary"]};
      --secondary: {theme["text-secondary"]};
      --structure: {theme["structure"]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 48px 24px 80px;
      background: var(--canvas);
      color: var(--ink);
      font-family: {theme.body};
      line-height: 1.6;
    }}
    main {{ max-width: 960px; margin: 0 auto; }}
    .kicker {{
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--secondary);
      margin: 0 0 8px;
    }}
    h1 {{
      font-family: "Newsreader", Georgia, serif;
      font-size: 38px;
      font-weight: 600;
      letter-spacing: -0.03em;
      line-height: 1.18;
      margin: 0 0 12px;
    }}
    .lede {{
      font-size: 18px;
      color: var(--secondary);
      margin: 0 0 36px;
      max-width: 68ch;
    }}
    .wp-figure {{
      margin: 0 0 52px;
      padding: 0 0 32px;
      border-bottom: 1px solid rgba(44, 62, 80, 0.22);
    }}
    .wp-figure-title {{
      font-size: 20px;
      font-weight: 600;
      letter-spacing: -0.015em;
      line-height: 1.32;
      margin: 0 0 6px;
    }}
    .wp-figure-subtitle {{
      font-size: 15px;
      color: var(--secondary);
      margin: 0 0 18px;
    }}
    .wp-figure-plot {{ overflow-x: auto; padding: 4px 0; }}
    .wp-chart {{ max-width: 100%; height: auto; display: block; }}
    figcaption {{ font-size: 13.5px; color: var(--secondary); margin-top: 12px; }}
  </style>
</head>
<body>
  <main>
    <p class="kicker">Biophysics of a Looped Transformer · Phase 6 + Phase 7</p>
    <h1>Six Dawn State Plates Across Inkling-2-Mini, Varloop, and Huginn-0125</h1>
    <p class="lede">
      Exact Walsh-Hadamard synaptic epistasis, universal non-normal transient amplification, spectral orthogonal separation of latent safety, Fisher pullback boundary spikes, Labovich recall shielding, and inverted-U neuromodulatory conductance.
    </p>
    {"".join(figure_blocks_html)}
  </main>
</body>
</html>
"""
    if EM_DASH in gallery_html:
        raise ValueError("Em dash found in Phase 7 gallery HTML")
    idx_path = out_dir / "index.html"
    idx_path.write_text(gallery_html, encoding="utf-8")

    return {
        "index_html": str(idx_path),
        "svg_files": svg_files,
        "png_files": png_files,
        "n_svg": len(svg_files),
        "n_png": len(png_files),
    }
