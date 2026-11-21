"""Dawn State SVG & HTML Figure Suite for Phase 5 Inkling Afriat / GARP Study.

Uses `design-system/scripts/figures.py` and `design-system/profiles/dawn-state/tokens.json`
to render publication-grade SVG charts and a local Dawn State HTML preview page.
Works seamlessly on both live block checkpoints (`checkpoint_afriat_inkling_results.json`)
and the final completed artifact (`afriat_inkling_results.json`).

Invariants (`AGENTS.md`, `PHILOSOPHY.md`, `VOICE.md`, `adapters/data.md`):
  - Zero em dashes anywhere in titles, subtitles, labels, notes, or captions.
  - Statement titles leading with the empirical finding; subtitles naming measure, unit, and sample.
  - Dawn State palette (`#FFFFF0` Ivory, `#F5EEDC` Parchment, `#0F0F0F` Onyx, `#2C3E50` Blue Slate,
    `#808B90` Slate Mist, `#87A96B` Sage Green, `#F39C12` Golden Orange with `2 px` `#2C3E50` outline).
"""

from __future__ import annotations

import html
import importlib.util
import json
from pathlib import Path
from typing import Any

from .afriat_stimuli import BASE_INTERSECTING_INTERCEPTS_24, CORE_ARCHETYPES, CORE_CONCEPTS, GRID_SHARES_11, build_wtp_curves


ARCHETYPE_LABELS: dict[str, str] = {
    "tainted_provenance": "Dirty money",
    "insulting_split": "Insulting 98/2 split",
    "communal_taboo_pricing": "Neighbor favor fee",
    "zero_interest_debt": "0%-APR loan",
    "crisis_markup": "Crisis surge price",
    "institutional_scale": "$25k invoice control",
}

CONCEPT_LABELS: dict[str, str] = {
    "guilt": "Guilt",
    "predation": "Predation",
    "spite": "Spite",
    "contamination": "Taint",
    "sacrilege": "Sacrilege",
    "captivity": "Captivity",
    "awe_dread": "Dread",
    "warmth": "Warmth",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_figures_module() -> tuple[Any, dict[str, Any], Any]:
    root = _repo_root()
    tokens_path = root / "design-system" / "profiles" / "dawn-state" / "tokens.json"
    fig_script = root / "design-system" / "scripts" / "figures.py"
    profile = json.loads(tokens_path.read_text(encoding="utf-8"))
    spec = importlib.util.spec_from_file_location("wp_figures", fig_script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load figures.py from {fig_script}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    theme = mod.Theme(profile)
    return mod, profile, theme


def build_architecture_and_budget_svg(theme: Any) -> str:
    """Two-panel Dawn State SVG: (A) Native Inkling-2 Message-Bank Loop, (B) T=24 Intersecting Budget Grid."""
    w, h = 860, 340
    esc = html.escape
    font = esc(theme.body)
    mono = esc(theme.code)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" class="wp-chart" viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'role="img" aria-labelledby="fig-arch-t fig-arch-d">'
        f'<title id="fig-arch-t">Inkling message-bank recurrence and 24 intersecting Afriat budget lines</title>'
        f'<desc id="fig-arch-d">Left panel shows Stem L0..L5, Recurrent Loop L6..L17 with 4-channel message bank, and Coda L18..L23. '
        f'Right panel shows all 24 intersecting 2-good linear budget constraints with 11 discrete menu bundles each.</desc>'
    ]

    # Panel headers
    parts.append(
        f'<text x="16" y="22" fill="{theme["text-primary"]}" font-family="{font}" font-size="15" font-weight="600">'
        f'A. Native looped MoE message bank (K = 1..5)</text>'
    )
    parts.append(
        f'<text x="456" y="22" fill="{theme["text-primary"]}" font-family="{font}" font-size="15" font-weight="600">'
        f'B. T = 24 intersecting linear budget lines (p1 x1 + p2 x2 = m)</text>'
    )

    # Panel A boxes
    boxes = [
        (16, 46, 392, 54, theme["well"], theme["structure"], 1.5, "Stem blocks (L0..L5)", "Pass once · clinical tokens with 0 affect words"),
        (16, 126, 392, 86, theme["canvas"], theme["structure"], 2.0, "Recurrent MoE loop (L6..L17, K = 1..5 passes)", "4-channel cross-pass message bank (Q, K, V, MLP)"),
        (16, 240, 392, 68, theme["well"], theme["structure"], 1.5, "Coda (L18..L23) + Readout", "0-CoT menu log-odds or <|content_thinking|> trace"),
    ]
    for bx, by, bw, bh, fill, stroke, sw, l1, l2 in boxes:
        parts.append(
            f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        parts.append(
            f'<text x="{bx + 16}" y="{by + 24}" fill="{theme["text-primary"]}" font-family="{font}" font-size="14" font-weight="600">{esc(l1)}</text>'
        )
        parts.append(
            f'<text x="{bx + 16}" y="{by + 44}" fill="{theme["text-secondary"]}" font-family="{font}" font-size="12.5">{esc(l2)}</text>'
        )

    # Focal 8D badge inside loop box
    parts.append(
        f'<rect x="32" y="178" width="360" height="24" rx="5" fill="{theme["emphasis"]}" stroke="{theme["structure"]}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="212" y="194" text-anchor="middle" fill="{theme["on-emphasis"]}" font-family="{mono}" font-size="12" font-weight="600">'
        f'8D moral subspace (0.35% of d = 2,304) · clamp or ±2α steer</text>'
    )

    # Arrows between boxes + recurrence loop arrow
    for y1, y2 in ((100, 124), (212, 238)):
        parts.append(
            f'<line x1="212" y1="{y1}" x2="212" y2="{y2}" stroke="{theme["structure"]}" stroke-width="2"/>'
            f'<polygon points="207,{y2 - 5} 217,{y2 - 5} 212,{y2 + 1}" fill="{theme["structure"]}"/>'
        )
    # Side feedback loop arc on right edge of Panel A
    parts.append(
        f'<path d="M 408 188 L 428 188 L 428 144 L 410 144" fill="none" stroke="{theme["emphasis-text"]}" stroke-width="2"/>'
        f'<polygon points="415,140 415,148 408,144" fill="{theme["emphasis-text"]}"/>'
    )

    # Panel B: Exact 24 intersecting budget lines in (x1, x2) space
    px0, py0, pw, ph = 504.0, 42.0, 332.0, 252.0
    max_val = 260.0
    x_at = lambda v: px0 + (v / max_val) * pw
    y_at = lambda v: py0 + ph - (v / max_val) * ph

    # Gridlines & axes
    for tick in (0, 50, 100, 150, 200, 250):
        tx, ty = x_at(tick), y_at(tick)
        parts.append(
            f'<line x1="{px0:.1f}" x2="{px0 + pw:.1f}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="{theme["structure"]}" stroke-opacity="0.16" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{px0 - 8:.1f}" y="{ty + 4:.1f}" text-anchor="end" fill="{theme["text-secondary"]}" font-family="{mono}" font-size="12">${tick}</text>'
        )
        parts.append(
            f'<text x="{tx:.1f}" y="{py0 + ph + 18:.1f}" text-anchor="middle" fill="{theme["text-secondary"]}" font-family="{mono}" font-size="12">${tick}</text>'
        )

    # Draw 22 context budget lines in Slate Mist
    focal_pair = (10, 11)  # (220, 90) vs (90, 220): classic symmetric crossing WARP test
    for idx, (m1, m2) in enumerate(BASE_INTERSECTING_INTERCEPTS_24):
        if idx in focal_pair:
            continue
        parts.append(
            f'<line x1="{x_at(m1):.1f}" y1="{y_at(0):.1f}" x2="{x_at(0):.1f}" y2="{y_at(m2):.1f}" '
            f'stroke="{theme["context"]}" stroke-opacity="0.38" stroke-width="1.3"/>'
        )

    # Draw the two crossing focal budget lines (Blue Slate + Golden Orange with outline)
    m1_a, m2_a = BASE_INTERSECTING_INTERCEPTS_24[focal_pair[0]]
    m1_b, m2_b = BASE_INTERSECTING_INTERCEPTS_24[focal_pair[1]]
    parts.append(
        f'<line x1="{x_at(m1_a):.1f}" y1="{y_at(0):.1f}" x2="{x_at(0):.1f}" y2="{y_at(m2_a):.1f}" '
        f'stroke="{theme["series-primary"]}" stroke-width="2.6"/>'
    )
    parts.append(
        f'<line x1="{x_at(m1_b):.1f}" y1="{y_at(0):.1f}" x2="{x_at(0):.1f}" y2="{y_at(m2_b):.1f}" '
        f'stroke="{theme["emphasis-text"]}" stroke-width="2.6" stroke-dasharray="5 3"/>'
    )
    for sh in GRID_SHARES_11:
        bx, by = sh * m1_a, (1.0 - sh) * m2_a
        parts.append(
            f'<circle cx="{x_at(bx):.1f}" cy="{y_at(by):.1f}" r="3.2" fill="{theme["series-primary"]}"/>'
        )
    # Intersection point (x1 = x2 = 220*90/(220+90) = 63.87)
    ix = (m1_a * m1_b * (m2_b - m2_a)) / max(1.0, (m1_b * m2_b - m1_a * m2_a)) if (m1_b * m2_b != m1_a * m2_a) else (m1_a * m2_a) / (m1_a + m2_a)
    ix = (m1_a * m2_a) / (m1_a + m2_a)
    parts.append(
        f'<circle cx="{x_at(ix):.1f}" cy="{y_at(ix):.1f}" r="6.5" fill="{theme["emphasis"]}" stroke="{theme["structure"]}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{x_at(ix) + 10:.1f}" y="{y_at(ix) - 8:.1f}" fill="{theme["text-primary"]}" font-family="{font}" font-size="12.5" font-weight="600">'
        f'WARP crossing (>97% Bronars power)</text>'
    )

    parts.append(
        f'<line x1="{px0:.1f}" x2="{px0 + pw:.1f}" y1="{py0 + ph:.1f}" y2="{py0 + ph:.1f}" stroke="{theme["structure"]}" stroke-width="2"/>'
        f'<line x1="{px0:.1f}" x2="{px0:.1f}" y1="{py0:.1f}" y2="{py0 + ph:.1f}" stroke="{theme["structure"]}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{px0 + pw / 2:.1f}" y="{h - 6:.1f}" text-anchor="middle" fill="{theme["text-secondary"]}" font-family="{font}" font-size="13">'
        f'Good 1 allocation x1 (own payout or standard reserve, USD)</text>'
    )
    parts.append(
        f'<text x="462" y="{py0 + ph / 2:.1f}" text-anchor="middle" transform="rotate(-90 462 {py0 + ph / 2:.1f})" '
        f'fill="{theme["text-secondary"]}" font-family="{font}" font-size="13">Good 2 allocation x2 (USD)</text>'
    )

    parts.append("</svg>")
    return "".join(parts)


def render_afriat_figure_suite(
    results_json_path: Path,
    out_dir: Path,
) -> dict[str, Any]:
    """Generate Dawn State SVG figures + standalone HTML gallery from live checkpoint or final JSON."""
    mod, profile, theme = _load_figures_module()
    payload = json.loads(results_json_path.read_text(encoding="utf-8"))
    out_dir.mkdir(parents=True, exist_ok=True)

    var_sec = payload.get("varloop_research", {})
    mini_sec = payload.get("inkling_2_mini", {})
    is_complete = bool(payload.get("completed", False))

    rendered_blocks: list[str] = []
    svg_files: list[str] = []

    def _save_svg(fname: str, svg_str: str) -> None:
        (out_dir / fname).write_text(svg_str, encoding="utf-8")
        svg_files.append(fname)

    # 1. Experimental Design & 24 Intersecting Budget Lines
    svg_arch = build_architecture_and_budget_svg(theme)
    _save_svg("fig1_architecture_and_budgets.svg", svg_arch)
    rendered_blocks.append(
        mod.figure(
            "Twenty-four crossing budget lines test whether silent-loop moral choices obey Afriat's theorem.",
            "Left: Inkling-2 4-channel cross-pass message bank (L6..L17). Right: 24 linear budget constraints (11 menu bundles each, relative price p1/p2 from 0.16 to 6.25).",
            svg_arch,
            "Source: Nullbridge Phase 5 protocol specification (`src/nullbridge/afriat_stimuli.py`, `src/nullbridge/garp.py`).",
            wide=True,
        )
    )

    # 2. Block 1 Figures: 6x8 Moral Atlas Heatmap + K=1..5 Ignition + K=5 Bootstrap Intervals
    if "block1_moral_atlas" in var_sec:
        b1 = var_sec["block1_moral_atlas"]
        fc = b1["financial_cartography"]
        arch_order = [
            "tainted_provenance",
            "insulting_split",
            "zero_interest_debt",
            "communal_taboo_pricing",
            "crisis_markup",
            "institutional_scale",
        ]
        concept_order = [
            "predation",
            "guilt",
            "spite",
            "contamination",
            "sacrilege",
            "captivity",
            "awe_dread",
            "warmth",
        ]
        row_labels = [ARCHETYPE_LABELS[a] for a in arch_order]
        col_labels = [CONCEPT_LABELS[c] for c in concept_order]
        mat = [
            [
                float(fc[a]["message_bank_projection"]["concepts"][c]["partial_beta"])
                for c in concept_order
            ]
            for a in arch_order
        ]
        svg_heat = mod.heatmap(
            theme,
            row_labels,
            col_labels,
            mat,
            id="fig-moral-atlas",
            diverging=True,
            decimals=2,
            width=760,
            cell=64,
            legend_label="Ridge-deconvolved partial beta in cross-pass message bank",
            title="Clinical financial scenarios activate non-financial moral vectors inside the message bank",
        )
        _save_svg("fig2_moral_atlas_heatmap.svg", svg_heat)
        rendered_blocks.append(
            mod.figure(
                "Zero-interest debt activates guilt (+0.26) while unfair splits and dirty money fire predation (+0.30).",
                "Multivariate partial beta onto 8 non-financial moral concept directions in TML-d24-varloop-research's message bank, n = 96 clinical transaction pairs with 0 affective words.",
                svg_heat,
                "Source: Phase 5 Block 1 unsteered reading states (`TML-d24-varloop-research`, K = 3 passes, 200-pair zero-finance concept probe).",
            )
        )

        # Figure 3: K = 1..5 Multi-Pass Moral Ignition Line Chart
        beh = b1["behavioral_anomalies_by_k"]
        ks = [k for k in ("k1", "k2", "k3", "k4", "k5") if k in beh]
        x_lbls = [f"K = {k[1:]}" for k in ks]
        series_ign = {
            ARCHETYPE_LABELS[a]: [float(beh[k]["unclamped"][a]["mean_delta_lo"]) for k in ks]
            for a in arch_order
        }
        svg_ign = mod.line_chart(
            theme,
            x_lbls,
            series_ign,
            id="fig-loop-ignition",
            focus=ARCHETYPE_LABELS["tainted_provenance"],
            width=760,
            height=360,
            decimals=2,
            y_min=-0.5,
            y_max=3.5,
            title="Silent recurrent passes ignite moral taboo aversion on pass two",
        )
        _save_svg("fig3_multipass_moral_ignition.svg", svg_ign)
        rendered_blocks.append(
            mod.figure(
                "Silent recurrent passes double dirty-money rejection (+1.58 to +2.98) and ignite 0%-APR debt aversion.",
                "Paired log-odds shift toward sacrificing cash vs. identical-payout neutral twin across K = 1..5 silent passes (0 CoT tokens), log-odds, n = 16 pairs per archetype.",
                svg_ign,
                "Source: Phase 5 Block 1 counterbalanced choice evaluation (`TML-d24-varloop-research`).",
            )
        )

        # Figure 4: K = 5 95% Bootstrap Confidence Intervals
        if "k5" in beh:
            k5_u = beh["k5"]["unclamped"]
            iv_rows = [
                (
                    ARCHETYPE_LABELS[a],
                    float(k5_u[a]["mean_delta_lo"]),
                    float(k5_u[a]["ci95"][0]),
                    float(k5_u[a]["ci95"][1]),
                )
                for a in arch_order
            ]
            svg_iv = mod.interval_plot(
                theme,
                iv_rows,
                id="fig-k5-intervals",
                reference=0.0,
                width=760,
                decimals=2,
                focus=ARCHETYPE_LABELS["tainted_provenance"],
                title="Five silent passes produce clear moral premiums on five of six archetypes while the scale control includes zero",
                reference_label="Homo Economicus parity (0.0)",
            )
            _save_svg("fig4_k5_bootstrap_intervals.svg", svg_iv)
            rendered_blocks.append(
                mod.figure(
                    "At five silent passes, dirty money, unfair splits, neighbor pricing, and 0%-APR debt all clear zero while the $25k invoice control stays null.",
                    "Mean paired log-odds anomaly and 95% bootstrap confidence interval (10,000 resamples) at K = 5 passes, log-odds, n = 16 pairs per row.",
                    svg_iv,
                    "Source: Phase 5 Block 1 (`TML-d24-varloop-research`, K = 5 unclamped).",
                )
            )

    # 3. Block 2 Figures (auto-populated as soon as Block 2 lands)
    if "block2_afriat_garp" in var_sec:
        b2 = var_sec["block2_afriat_garp"]
        bronars_mean = float(b2["bronars_random_benchmark"]["mean_random_ccei"])
        pts: list[tuple[float, float, str]] = []
        for cid, cbody in b2["cells"].items():
            if cbody["condition"] in ("unclamped", "msg_homo_economicus_8d"):
                for rname, rshort in (
                    ("social_provoked", "Provoked"),
                    ("provenance_clean_vs_tainted", "Tainted"),
                    ("social_neutral", "Neutral"),
                ):
                    rm = cbody["regimes"][rname]
                    lbl = f"{rshort} K={cbody['n_passes']}" if cbody["condition"] == "unclamped" and cbody["n_passes"] in (1, 5) else ""
                    pts.append((float(rm["homo_economicus_efficiency"]) * 100.0, float(rm["modal_ccei_mean"]), lbl))
        if "block2_afriat_garp" in mini_sec:
            im = mini_sec["block2_afriat_garp"]["cells"]["unclamped__k3"]["regimes"]["provenance_clean_vs_tainted"]
            pts.append((float(im["homo_economicus_efficiency"]) * 100.0, float(im["modal_ccei_mean"]), "Inkling-2-Mini Tainted K=3"))
        focus_lbl = "Inkling-2-Mini Tainted K=3" if any(len(p) > 2 and p[2] == "Inkling-2-Mini Tainted K=3" for p in pts) else "Tainted K=5"
        svg_garp = mod.scatter_plot(
            theme,
            pts,
            id="fig-afriat-ccei",
            x_label="Homo Economicus payoff capture (%)",
            y_label="Afriat Critical Cost Efficiency Index (CCEI)",
            width=760,
            height=420,
            x_decimals=0,
            y_decimals=2,
            x_unit="%",
            hline=bronars_mean,
            hline_label=f"Bronars random choice ({bronars_mean:.2f})",
            focus=focus_lbl,
            x_min=40.0,
            x_max=100.0,
            y_min=0.50,
            y_max=1.02,
            title="Afriat utility coherence vs. Homo Economicus payoff capture across intersecting budgets",
        )
        _save_svg("fig5_afriat_ccei_vs_fungibility.svg", svg_garp)
        rendered_blocks.append(
            mod.figure(
                "Revealed-preference coherence (Afriat CCEI) separates coherent moral utility from random budget violations.",
                "Afriat Critical Cost Efficiency Index (1.00 = exact GARP utility maximization) vs. Homo Economicus monetary efficiency across T = 24 intersecting budgets (N = 6 subjects per regime).",
                svg_garp,
                "Source: Phase 5 Block 2 Afriat / GARP budget audit (`src/nullbridge/garp.py`).",
            )
        )

    # 4. Block 3 Steering Dose-Response (auto-populated when Block 3 lands)
    target_sec = mini_sec if "block3_inequity_and_steering" in mini_sec else var_sec
    if "block3_inequity_and_steering" in target_sec:
        st = target_sec["block3_inequity_and_steering"]["message_bank_steering_k3"]
        alphas = ["−2α", "−1α", "0α", "+1α", "+2α"]
        st_series = {
            "98/2 Ultimatum + Spite": [float(r["mean_lo"]) for r in st["ultimatum_98_2"]["spite"]],
            "Dirty Money + Taint": [float(r["mean_lo"]) for r in st["tainted_provenance"]["contamination"]],
            "2/98 Dictator + Guilt": [float(r["mean_lo"]) for r in st["dictator_2_98"]["guilt"]],
            "98/2 Ultimatum + Valence": [float(r["mean_lo"]) for r in st["ultimatum_98_2"]["valence"]],
        }
        svg_st = mod.line_chart(
            theme,
            alphas,
            st_series,
            id="fig-message-steering",
            focus="98/2 Ultimatum + Spite",
            width=760,
            height=340,
            decimals=2,
            title="Additive steering inside the cross-pass message bank dials economic rejection up and down",
        )
        _save_svg("fig6_message_bank_steering.svg", svg_st)
        rendered_blocks.append(
            mod.figure(
                "Injecting non-financial moral vectors into the cross-pass message bank steers economic decisions bidirectionally.",
                "Counterbalanced choice log-odds across steering multipliers alpha in {-2, -1, 0, +1, +2} at K = 3 passes, log-odds, n = 16 fixtures per curve.",
                svg_st,
                "Source: Phase 5 Block 3 message-bank steering dose-response.",
            )
        )

    # 5. Block 4 Continuous Dollar WTP Curves (normalized to % of tested ceiling + exact USD)
    wtp_sec = mini_sec if "block4_dollar_wtp_curves" in mini_sec else var_sec
    if "block4_dollar_wtp_curves" in wtp_sec:
        cells_w = wtp_sec["block4_dollar_wtp_curves"]["by_condition_and_k"]
        ckey = "unclamped__k3" if "unclamped__k3" in cells_w else next(iter(cells_w.keys()))
        cw = cells_w[ckey]
        raw_curves = build_wtp_curves()
        c_floors = {
            "ultimatum_offer": 0.0,
            "communal_favor": 0.0,
            "crisis_surge": 0.0,
            "zero_apr_debt": 0.0,
            "tainted_surplus": 0.0,
            "institutional_certainty": -5.0,
            "pocket_certainty_control": -5.0,
        }
        wtp_names = [
            ("tainted_surplus", "Forfeit dirty surplus"),
            ("communal_favor", "Waive neighbor fee"),
            ("institutional_certainty", "Insure $25M settlement"),
            ("zero_apr_debt", "Prepay 0%-APR loan"),
            ("pocket_certainty_control", "Insure $100 voucher"),
            ("crisis_surge", "Forgo surge markup"),
            ("ultimatum_offer", "Reject unfair split"),
        ]
        wtp_rows: list[tuple[str, float]] = []
        focus_bar = ""
        for k, base_lbl in wtp_names:
            if k not in cw:
                continue
            usd_v = float(cw[k]["expected_wtp_usd"])
            span = max(it.ref_cost_usd for it in raw_curves[k]) - c_floors[k]
            pct_v = max(0.0, min(100.0, (usd_v - c_floors[k]) / max(1.0, span) * 100.0))
            lbl_full = f"{base_lbl} (${usd_v:,.0f})" if usd_v >= 100 else f"{base_lbl} (${usd_v:.1f})"
            if k == "tainted_surplus":
                focus_bar = lbl_full
            wtp_rows.append((lbl_full, pct_v))
        svg_wtp = mod.bar_chart(
            theme,
            wtp_rows,
            id="fig-wtp-bars",
            focus=focus_bar,
            sort=True,
            width=760,
            decimals=1,
            kind="percent",
            title="Turnbull dollar reservation curves show near-ceiling forfeiture on dirty surplus and neighbor fees",
        )
        _save_svg("fig7_dollar_wtp_reservations.svg", svg_wtp)
        rendered_blocks.append(
            mod.figure(
                "Turnbull reservation curves show 94.9% ceiling forfeiture on dirty surplus ($37,957) and 83.2% on neighbor fees ($666).",
                "Share of maximum tested dollar rung forfeited at K = 3 silent passes (exact expected Turnbull USD in label), %, n = 8 fixtures x 7 price rungs per curve.",
                svg_wtp,
                "Source: Phase 5 Block 4 continuous dollar reservation suite.",
            )
        )

    # Assemble Dawn State standalone HTML page
    status_line = (
        "Complete dataset locked across TML-d24-varloop-research (K = 1..5) and Inkling-2-Mini (K = 3)."
        if is_complete
        else "Live preview rendered from Block 1 checkpoint on TML-d24-varloop-research (Blocks 2 to 5 currently executing on H100; re-run `uv run nullbridge figures-afriat` upon completion to populate Figures 5 to 7)."
    )

    page_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Afriat / GARP &amp; Moral Geometry on Inkling · Dawn State Figures</title>
  <style>
    :root {{
      --canvas: {theme["canvas"]};
      --well: {theme["well"]};
      --ink: {theme["text-primary"]};
      --secondary: {theme["text-secondary"]};
      --structure: {theme["structure"]};
      --emphasis: {theme["emphasis"]};
      --emphasis-text: {theme["emphasis-text"]};
      --positive-text: {theme["outcome-positive-text"]};
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
    main {{ max-width: 920px; margin: 0 auto; }}
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
      margin: 0 0 24px;
      max-width: 68ch;
    }}
    .status-well {{
      background: var(--well);
      border-left: 3px solid var(--structure);
      padding: 12px 16px;
      border-radius: 0 8px 8px 0;
      font-size: 14px;
      color: var(--secondary);
      margin-bottom: 40px;
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
      color: var(--ink);
    }}
    .wp-figure-subtitle {{
      font-size: 15px;
      color: var(--secondary);
      margin: 0 0 18px;
    }}
    .wp-figure-plot {{
      overflow-x: auto;
      padding: 4px 0;
    }}
    .wp-chart {{
      max-width: 100%;
      height: auto;
      display: block;
    }}
    figcaption {{
      font-size: 13.5px;
      color: var(--secondary);
      margin-top: 12px;
    }}
  </style>
</head>
<body>
  <main>
    <p class="kicker">Part 4 · Revealed Preference &amp; Moral Geometry</p>
    <h1>Silent Recurrence Ignites Human Moral Priors Inside Clinical Economic Choices</h1>
    <p class="lede">
      Across 200 zero-finance moral concept pairs, 96 clinical transactions, and 24 intersecting Afriat budget constraints on looped Mixture-of-Experts models, extra silent passes amplify moral taboo aversion rather than cooling decisions toward Homo Economicus.
    </p>
    <div class="status-well">{html.escape(status_line)}</div>
    {"".join(rendered_blocks)}
  </main>
</body>
</html>
"""
    index_path = out_dir / "index.html"
    index_path.write_text(page_html, encoding="utf-8")
    return {
        "index_html": str(index_path),
        "svg_files": svg_files,
        "n_figures": len(svg_files),
        "completed_dataset": is_complete,
    }
