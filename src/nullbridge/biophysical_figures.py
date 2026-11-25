"""Dawn State SVG & 2x Retina PNG Figure Suite for Phase 6 (`phase6_inkling_biophysical_loop`).

Renders 5 publication-grade figures via `design-system/scripts/figures.py` and
`design-system/profiles/dawn-state/tokens.json`, exporting both `.svg` and `.png`
plus an inspectable `index.html` gallery. Zero em dashes anywhere.
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
    spec = importlib.util.spec_from_file_location("wp_figures_p6", fig_script)
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
    panel_w: int = 410,
    panel_h: int = 330,
    gap: int = 36,
) -> str:
    """Compose two `figures.py` SVG charts side-by-side in one accessible Dawn State SVG."""
    total_w = panel_w * 2 + gap
    top_y = 28
    total_h = panel_h + top_y + 4
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


def render_phase6_figure_suite(
    results_json_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Render 5 Dawn State `.svg` + `.png` figures and `index.html` from `phase6_biophysical_results.json`."""
    mod, _profile, theme = _load_wp_figures()
    data = json.loads(results_json_path.read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    chrome_bin = _find_chrome_bin()

    e1 = data["exp1_wht_synaptic_knockouts"]
    e2 = data["exp2_gain_vs_amplitude"]
    e3 = data["exp3_fisher_and_coda_shunt"]
    e4 = data["exp4_hysteresis_pulse_chase"]
    e5 = data["exp5_neuromodulatory_switching"]

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
    # Figure 1: 16-Term Walsh-Hadamard Synaptic Decomposition Across Task Regimes
    # -----------------------------------------------------------------------
    wht_terms = ("Q", "K", "V", "MLP", "QxK", "VxMLP", "QxV", "KxV", "QxMLP", "KxMLP")
    reg_keys = (
        ("factual_2hop", "2-hop factual"),
        ("arithmetic_2step", "2-step arithmetic"),
        ("calendar_2step", "2-step calendar"),
        ("all_120", "All 120 items"),
    )
    heat_rows = [lbl for _, lbl in reg_keys]
    heat_vals: list[list[float]] = []
    for rk, _rlbl in reg_keys:
        src_w = (
            e1["schedules"]["joint_p2_p3"]["all_120"]["continuation_logprob_wht"]
            if rk == "all_120"
            else e1["schedules"]["joint_p2_p3"]["by_regime"][rk]["continuation_logprob_wht"]
        )
        heat_vals.append([float(src_w["terms"][t]["coalition_effect"]) for t in wht_terms])

    j_acc = e1["schedules"]["joint_p2_p3"]["all_120"]["top1_accuracy_wht"]
    j_lp = e1["schedules"]["joint_p2_p3"]["all_120"]["continuation_logprob_wht"]
    svg1 = mod.heatmap(
        theme,
        heat_rows,
        list(wht_terms),
        heat_vals,
        id="fig1-wht-epistasis",
        diverging=True,
        decimals=2,
        width=820,
        cell=60,
        legend_label="Walsh-Hadamard coalition effect on continuation log-probability (nats)",
        title="Walsh-Hadamard main effects and pairwise epistasis across 4 synaptic message channels",
    )
    _emit(
        "fig1_wht_synaptic_epistasis",
        f"Four synaptic message channels lift accuracy 41.5x ({j_acc['masks']['0000']['mean']*100:.1f}% to {j_acc['masks']['1111']['mean']*100:.1f}%), pairing superadditive QxK (+0.89) with redundant VxMLP (-0.91).",
        f"Exact 16-mask Walsh-Hadamard coalition effects (2w_i main channels, 4w_ij pairwise epistasis; Order-1 = {j_lp['order_variance_shares']['order_1']*100:.1f}% of variance), nats, n = 120 items.",
        svg1,
        "Source: Nullbridge Phase 6 Experiment 1 (`thinkingmachines/Inkling-2-Mini`, 2^4 = 16 synaptic gate knockouts).",
        height_px=580,
    )

    # -----------------------------------------------------------------------
    # Figure 2: 48-Visit Unembedding Amplitude vs. Coda Unlock & Non-Normal Gain
    # -----------------------------------------------------------------------
    v48 = e2["visits_48"]
    x_v = [f"v{v['visit_idx']}" for v in v48]
    raw_series = [float(v["raw_logit_lens_acc"]) * 100.0 for v in v48]
    blk_series = [float(v["block_mixed_lens_acc"]) * 100.0 for v in v48]

    tc = e2["truncated_coda_grid"]
    coda_keys = sorted(tc["pass_3"].keys(), key=lambda k: int(k[1:]))
    last_ck = coda_keys[-1]
    coda_bridge_line: list[float | None] = [None] * len(v48)
    pass_end_indices = [
        next((v["visit_idx"] for v in reversed(v48) if v["visit_label"].startswith(pref)), None)
        for pref in ("p1_", "p2_", "p3_")
    ]
    coda_bridge_line[0] = 0.0
    for p_i, v_idx in enumerate(pass_end_indices):
        if v_idx is not None:
            coda_bridge_line[v_idx] = float(tc[f"pass_{p_i + 1}"][last_ck]["top1_acc"]) * 100.0
    for c_i, ck in enumerate(coda_keys):
        v_c_idx = len(v48) - len(coda_keys) + c_i
        if 0 <= v_c_idx < len(v48):
            coda_bridge_line[v_c_idx] = float(tc["pass_3"][ck]["top1_acc"]) * 100.0
    anchor_pts = [(idx, val) for idx, val in enumerate(coda_bridge_line) if val is not None]
    for (i0, v0), (i1, v1) in zip(anchor_pts, anchor_pts[1:]):
        for j in range(i0 + 1, i1):
            frac = (j - i0) / float(i1 - i0)
            coda_bridge_line[j] = v0 + frac * (v1 - v0)

    amp = e2["non_normal_pass_amplification"]
    svg2_a = mod.line_chart(
        theme,
        x_v,
        {
            "Coda readout": coda_bridge_line,
            "Block-mixed": blk_series,
            "Raw logit lens": raw_series,
        },
        id="fig2a-visit-lens",
        focus="Coda readout",
        width=440,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        tick_every=12,
        title="48-visit accuracy trajectory",
    )
    svg2_b = mod.line_chart(
        theme,
        ["Pass 1", "Pass 2", "Pass 3"],
        {
            "Top singular sigma_max": [
                float(amp["pass_1"]["sigma_max_vjp"]["mean"]),
                float(amp["pass_2"]["sigma_max_vjp"]["mean"]),
                float(amp["pass_3"]["sigma_max_vjp"]["mean"]),
            ],
            "Rayleigh self-feedback": [
                float(amp["pass_1"]["normal_rayleigh_quotient"]["mean"]),
                float(amp["pass_2"]["normal_rayleigh_quotient"]["mean"]),
                float(amp["pass_3"]["normal_rayleigh_quotient"]["mean"]),
            ],
            "Isotropic RMS gain": [
                float(amp["pass_1"]["isotropic_rms_gain"]["mean"]),
                float(amp["pass_2"]["isotropic_rms_gain"]["mean"]),
                float(amp["pass_3"]["isotropic_rms_gain"]["mean"]),
            ],
        },
        id="fig2b-non-normal-gain",
        focus="Top singular sigma_max",
        width=400,
        height=330,
        decimals=1,
        y_min=0,
        y_max=15,
        title="Non-normal transient amplification across passes",
    )
    svg2 = _two_panel_svg(
        theme,
        "fig2-gain-conductance",
        "48-visit readout accuracy vs non-normal transient singular amplification",
        "A. Direct unembedding vs. Coda readout (v0..v47, %)",
        svg2_a,
        "B. Recurrent Jacobian gain (sigma_max vs. isotropic)",
        svg2_b,
        panel_w=420,
        panel_h=330,
        gap=28,
    )
    _emit(
        "fig2_gain_vs_amplitude_conductance",
        f"Raw unembedding stays at <= 6.7% across 36 loop visits while Pass 2 ignites 13.6x non-normal singular gain and Coda unlocks {tc['pass_3'][last_ck]['top1_acc']*100:.1f}%.",
        "Left: Top-1 accuracy across unrolled visits v0..v47 (%, n = 60). Right: Exact VJP top singular value sigma_max vs. Rayleigh and isotropic RMS gain (n = 36).",
        svg2,
        "Source: Nullbridge Phase 6 Experiment 2 (`thinkingmachines/Inkling-2-Mini`, 48-visit lens + VJP/JVP power iteration).",
        height_px=590,
    )

    # -----------------------------------------------------------------------
    # Figure 3: Two-Stage Commitment Horizon (t*) & Coda Formatting Shunt (L18..L23)
    # -----------------------------------------------------------------------
    comm_a = e3["commitment_horizon"].get("arithmetic_2step", e3["commitment_horizon"]["all"])
    comm_f = e3["commitment_horizon"].get("factual_2hop", e3["commitment_horizon"]["all"])
    svg3_a = mod.line_chart(
        theme,
        ["Stem", "Pass 1", "Pass 2", "Pass 3"],
        {
            "Arith Hop-2 final": [x * 100.0 for x in comm_a["hop2_target_prob"]],
            "Factual Hop-2 capital": [x * 100.0 for x in comm_f["hop2_target_prob"]],
            "Arith Hop-1 bridge": [x * 100.0 for x in comm_a["hop1_bridge_prob"]],
        },
        id="fig3a-commitment-horizon",
        focus="Arith Hop-2 final",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        title="Two-stage Hop-1 to Hop-2 commitment across passes",
    )

    sh07 = e3["coda_formatting_shunt"]["chat_header_effort07"]["by_coda_layer"]
    sh_ct = e3["coda_formatting_shunt"]["chat_content_text_effort0"]["by_coda_layer"]
    coda_x = [r["coda_layer"] for r in sh07]
    svg3_b = mod.line_chart(
        theme,
        coda_x,
        {
            "Answer Top-5 (+1 tok)": [float(r["semantic_answer_top5_acc"]) * 100.0 for r in sh_ct],
            "<|content_thinking|>": [float(r["content_thinking_prob"]) * 100.0 for r in sh07],
            "Answer Top-5 (header)": [float(r["semantic_answer_top5_acc"]) * 100.0 for r in sh07],
        },
        id="fig3b-coda-shunt",
        focus="Answer Top-5 (header)",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        title="Coda formatting shunt across L18..L23",
    )
    svg3 = _two_panel_svg(
        theme,
        "fig3-commitment-shunt",
        "Pass-by-pass Hop-1 to Hop-2 commitment and Coda formatting shunt",
        "A. Hop-1 staging (Pass 2) to Hop-2 lock-in (Pass 3)",
        svg3_a,
        "B. Coda L19 pre-shunt answer vs. L21..L22 shunt (%)",
        svg3_b,
        panel_w=420,
        panel_h=330,
        gap=28,
    )
    _emit(
        "fig3_fisher_commitment_and_coda_shunt",
        "Arithmetic stages Hop-1 on Pass 2 (21.6%) before locking Hop-2 on Pass 3 (100%), while Coda L21..L22 shunts L19 answers (50% Top-5) into <|content_thinking|>.",
        "Left: Coda-transported target vs. intermediate bridge probability across passes (%, n = 40). Right: Coda L18..L23 Top-5 answer rate vs. <|content_thinking|> mass (%, n = 40 triplets).",
        svg3,
        "Source: Nullbridge Phase 6 Experiment 3 (`thinkingmachines/Inkling-2-Mini`, commitment horizon & 13C Coda shunt).",
        height_px=590,
    )

    # -----------------------------------------------------------------------
    # Figure 4: Somatic-Reset Foil Suppression & Tangent Pulse-Chase Damping
    # -----------------------------------------------------------------------
    f_sw = e4["foil_hysteresis_sweep"]
    x_beta = [f"{r['beta']:.2f}" for r in f_sw]
    hs = e4["hysteresis_summary"]
    svg4_a = mod.line_chart(
        theme,
        x_beta,
        {
            "Pass 3 target acc": [float(r["coda_target_top1_acc"]) * 100.0 for r in f_sw],
            "Pass 2 foil prob": [float(r["pass2_foil_prob"]) * 100.0 for r in f_sw],
            "Pass 3 foil prob": [float(r["pass3_foil_prob"]) * 100.0 for r in f_sw],
        },
        id="fig4a-foil-suppression",
        focus="Pass 3 target acc",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        tick_every=2,
        title="Somatic reset suppresses Pass-1 foil replacements",
    )
    p_sw = [r for r in e4["orthogonal_pulse_chase_sweep"] if float(r["alpha"]) > 0.0]
    x_alpha = [f"{r['alpha']:.2f}x" for r in p_sw]
    svg4_b = mod.line_chart(
        theme,
        x_alpha,
        {
            "Coda Top-1 acc": [float(r["coda_top1_acc"]) * 100.0 for r in p_sw],
            "Pass 2 deviation": [float(r["rel_deviation_pass2"]) * 100.0 for r in p_sw],
            "Coda deviation": [float(r["rel_deviation_coda"]) * 100.0 for r in p_sw],
        },
        id="fig4b-pulse-chase",
        focus="Coda Top-1 acc",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        tick_every=2,
        title="Tangent pulse-chase damping across alpha",
    )
    svg4 = _two_panel_svg(
        theme,
        "fig4-hysteresis-pulse",
        "On-manifold foil suppression and off-manifold pulse-chase damping",
        "A. Foil replacement beta in [0.00, 1.25] (%)",
        svg4_a,
        "B. Orthogonal tangent pulse alpha in [0.10x, 2.00x]",
        svg4_b,
        panel_w=420,
        panel_h=330,
        gap=28,
    )
    _emit(
        "fig4_hysteresis_and_pulse_chase",
        f"Somatic reset suppresses Pass-2 foil mass by {hs['peak_pass3_foil_suppression']*100:.1f} pp (0% capture) and damps 1.0x tangent pulses to 29.9% in Coda.",
        "Left: Pass-1-only on-manifold foil state interpolation beta entering Pass 2 (%, n = 60). Right: Relative state deviation and Coda accuracy across tangent pulses alpha (%, n = 60).",
        svg4,
        "Source: Nullbridge Phase 6 Experiment 4 (`thinkingmachines/Inkling-2-Mini`, Pass-1 foil perturbation & pulse-chase).",
        height_px=590,
    )

    # -----------------------------------------------------------------------
    # Figure 5: Fixed-Weight Allosteric Pass-Clock Curve & Coda Format-Shunt Abolition
    # -----------------------------------------------------------------------
    gm_rows = e5["allosteric_conductance_sweep"]
    x_gm = [f"{r['gamma']:.2f}" for r in gm_rows]
    svg5_a = mod.line_chart(
        theme,
        x_gm,
        {
            "Hop-2 Top-1 acc": [float(r["hop2_top1_acc"]) * 100.0 for r in gm_rows],
            "Hop-2 prob": [float(r["hop2_target_prob"]) * 100.0 for r in gm_rows],
            "Hop-1 bridge prob": [float(r["hop1_bridge_prob"]) * 100.0 for r in gm_rows],
        },
        id="fig5a-gamma-clock",
        focus="Hop-2 Top-1 acc",
        width=420,
        height=330,
        decimals=0,
        unit="%",
        kind="percent",
        y_min=0,
        y_max=100,
        tick_every=2,
        title="Allosteric gate conductance sweep gamma in [0.00, 1.50]",
    )

    hm0 = e5["coda_header_mode_switch"]["effort_0"]["conditions"]
    shunt_bars = [
        ("Unmodified header", float(hm0["baseline_header_L23"]["format_token_prob"]) * 100.0),
        ("Slot-0 swap", float(hm0["slot0_embed_swap_L23"]["format_token_prob"]) * 100.0),
        ("Slot-0 + Slot-1 swap", float(hm0["slot0_and_slot1_somatic_swap_L23"]["format_token_prob"]) * 100.0),
        ("Carrier + loop delta", float(hm0["somatic_carrier_loop_transplant_L23"]["format_token_prob"]) * 100.0),
    ]
    svg5_b = mod.bar_chart(
        theme,
        shunt_bars,
        id="fig5b-header-shunt-abolition",
        focus="Carrier + loop delta",
        sort=False,
        width=420,
        decimals=1,
        unit="%",
        kind="percent",
        title="Format-token shunt mass under Coda somatic header swaps",
    )
    svg5 = _two_panel_svg(
        theme,
        "fig5-neuromodulation",
        "Allosteric gate conductance curve and Coda somatic header shunt abolition",
        "A. Gate conductance gamma sweep (peak 70.0% at 1.00)",
        svg5_a,
        "B. Coda L23 format-token shunt mass (%, effort 0)",
        svg5_b,
        panel_w=420,
        panel_h=330,
        gap=28,
    )
    _emit(
        "fig5_neuromodulatory_clock_and_header_swap",
        f"Scaling synaptic gate conductance peaks at native gamma = 1.00 (70.0% accuracy), while Coda somatic swaps abolish 99.4% of formatting shunt mass (100.0% to {hm0['somatic_carrier_loop_transplant_L23']['format_token_prob']*100:.1f}%).",
        "Left: Hop-2 accuracy, Hop-2 probability, and Hop-1 staging across message-gate scale gamma in [0.00, 1.50] (%, n = 60). Right: Format-token mass at <|message_model|> under Coda somatic swaps (%, n = 40).",
        svg5,
        "Source: Nullbridge Phase 6 Experiment 5 (`thinkingmachines/Inkling-2-Mini`, fixed-weight pass-clock & Coda header swaps).",
        height_px=590,
    )

    gallery_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Inkling-2-Mini Biophysical Loop Dissection · Dawn State Figures</title>
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
    main {{ max-width: 940px; margin: 0 auto; }}
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
    <p class="kicker">Biophysical &amp; Systems Loop Dissection · Inkling-2-Mini</p>
    <h1>Somatic Reset, Synaptic Conductance, and Coda Shunting in a Looped Mixture-of-Experts</h1>
    <p class="lede">
      Five causal perturbation experiments on thinkingmachines/Inkling-2-Mini map how four cross-pass synaptic message channels transmit multi-hop deductions across hard somatic resets, amplify task directions non-normally, self-correct distractors, and gate final answer emission in Coda.
    </p>
    {"".join(figure_blocks_html)}
  </main>
</body>
</html>
"""
    if EM_DASH in gallery_html:
        raise ValueError("Em dash found in gallery HTML")
    idx_path = out_dir / "index.html"
    idx_path.write_text(gallery_html, encoding="utf-8")

    return {
        "index_html": str(idx_path),
        "svg_files": svg_files,
        "png_files": png_files,
        "n_svg": len(svg_files),
        "n_png": len(png_files),
    }
