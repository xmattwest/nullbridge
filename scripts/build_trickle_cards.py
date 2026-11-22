#!/usr/bin/env python3
"""Build 2x Retina Dawn State X trickle-post plates (1920x1440 4:3 Title + Graph).

All four cards use the un-busied Title + Graph plate layout on Ivory (#FFFFF0),
putting the test and result in the <250-char X post caption and moving supporting
metrics into thread replies.

Usage:
    uv run --with pillow python scripts/build_trickle_cards.py
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import pathlib
import shutil
import subprocess
from collections import defaultdict

import numpy as np


CLEAN_PLATE_CSS = """
  :root {
    --ivory: #FFFFF0;
    --parchment: #F5EEDC;
    --onyx: #0F0F0F;
    --slate-deep: #1E3447;
    --blue-slate: #2C3E50;
    --slate-mist: #808B90;
    --golden-orange: #F39C12;
    --copper-deep: #96590A;
    --sage-green: #87A96B;
    --sage-deep: #5F7F4A;
    --rule: rgba(44, 62, 80, 0.18);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: 960px;
    height: 720px;
    background: var(--ivory);
    color: var(--onyx);
    font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
    overflow: hidden;
  }
  .plate {
    width: 960px;
    height: 720px;
    padding: 38px 44px 26px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    background: var(--ivory);
  }
  h1.headline {
    font-family: Newsreader, Georgia, 'Times New Roman', serif;
    font-weight: 600;
    font-size: 44px;
    line-height: 1.07;
    letter-spacing: -0.035em;
    font-variant-numeric: lining-nums;
    font-feature-settings: "lnum" 1, "onum" 0;
    color: var(--onyx);
    max-width: 872px;
  }
  h1.headline .accent { color: var(--copper-deep); }
  h1.headline .sage { color: var(--sage-deep); }
  .plot-wrap {
    flex: 1;
    display: flex;
    align-items: center;
    min-height: 0;
    margin: 14px 0 8px;
  }
  .plot-wrap svg {
    width: 100%;
    height: auto;
    display: block;
    font-variant-numeric: tabular-nums;
  }
  .source-line {
    font-size: 13px;
    color: var(--slate-mist);
    border-top: 1px solid var(--rule);
    padding-top: 10px;
    font-variant-numeric: tabular-nums;
  }
""".strip()


def find_chrome_bin() -> str:
    env_bin = os.environ.get("CHROME_BIN")
    if env_bin and pathlib.Path(env_bin).exists():
        return env_bin
    for candidate in [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ]:
        if candidate and pathlib.Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Google Chrome binary not found; set CHROME_BIN.")


def render_card_png(html_path: pathlib.Path, png_path: pathlib.Path, width: int = 960, height: int = 720) -> None:
    chrome = find_chrome_bin()
    subprocess.run(
        [
            chrome,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            "--force-device-scale-factor=2",
            f"--screenshot={png_path}",
            html_path.resolve().as_uri(),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wrap_clean_plate_html(title_plain: str, headline_html: str, svg_html: str, source_text: str) -> str:
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{html_mod.escape(title_plain)}</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,600&display=swap">
  <style>
{CLEAN_PLATE_CSS}
  </style>
</head>
<body>
  <div class="plate">
    <h1 class="headline">{headline_html}</h1>
    <div class="plot-wrap">
      {svg_html}
    </div>
    <div class="source-line">{source_text}</div>
  </div>
</body>
</html>
"""
    assert "\u2014" not in doc, f"Em dash found in {title_plain}"
    return doc


def build_card1_parity_canyon(p10: dict) -> tuple[str, dict[str, object]]:
    amounts = [float(x) for x in p10["grid_amounts"]]
    log_a = np.log10(amounts)
    concepts = p10["concepts_order"]
    spite_idx = concepts.index("spite")
    proj = np.array(p10["grid_summary"]["proj_rms_grid"], dtype=float)[:, :, spite_idx] * 100.0
    rej = np.array(p10["grid_summary"]["reject_log_odds_grid"], dtype=float)
    diag_rej = np.diag(rej)

    by_ratio_spite: dict[float, list[float]] = defaultdict(list)
    by_ratio_rej: dict[float, list[float]] = defaultdict(list)
    for i in range(len(amounts)):
        for j in range(len(amounts)):
            diff = round((log_a[i] - log_a[j]) * 2.0) / 2.0
            by_ratio_spite[diff].append(float(proj[i, j]))
            mid_idx = int(round(0.5 * (i + j)))
            by_ratio_rej[diff].append(float(rej[i, j] - diag_rej[mid_idx]))

    xs = sorted(k for k in by_ratio_spite.keys() if -8.0 <= k <= 8.0 and abs(k * 2 - round(k * 2)) < 1e-6)
    xs_plot = [x for x in xs if abs(x - round(x)) < 1e-6 or abs(x) <= 1.5]
    spite_y = [float(np.mean(by_ratio_spite[x])) for x in xs_plot]
    rej_y = [float(np.mean(by_ratio_rej[x])) for x in xs_plot]

    W, H = 872, 495
    x0, x1 = 68.0, 804.0
    y_top, y_bot = 46.0, 412.0
    y_axis = 424.0

    def sx(val: float) -> float:
        return x0 + (val + 8.0) / 16.0 * (x1 - x0)

    def sy_spite(val: float) -> float:
        clamped = max(0.0, min(1.50, val))
        return y_bot - (clamped / 1.50) * (y_bot - y_top)

    def sy_rej(val: float) -> float:
        clamped = max(-0.08, min(4.80, val))
        return y_bot - (max(0.0, clamped) / 4.80) * (y_bot - y_top)

    zero_x = sx(0.0)
    spite_pts = " ".join(f"{sx(x):.1f},{sy_spite(y):.1f}" for x, y in zip(xs_plot, spite_y))
    rej_pts = " ".join(f"{sx(x):.1f},{sy_rej(y):.1f}" for x, y in zip(xs_plot, rej_y))

    low_xs = [x for x in xs_plot if x <= 0.0]
    low_ys = [rej_y[xs_plot.index(x)] for x in low_xs]
    base_rej_y = sy_rej(0.0)
    area_pts = (
        f"{sx(low_xs[0]):.1f},{base_rej_y:.1f} "
        + " ".join(f"{sx(x):.1f},{sy_rej(y):.1f}" for x, y in zip(low_xs, low_ys))
        + f" {sx(0.0):.1f},{base_rej_y:.1f}"
    )

    dots_spite = []
    for x, y in zip(xs_plot, spite_y):
        if abs(x - round(x)) > 1e-6:
            continue
        cx, cy = sx(x), sy_spite(y)
        if abs(x) < 1e-6:
            dots_spite.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="7.5" fill="#87A96B" stroke="#2C3E50" stroke-width="2.4"/>')
        else:
            dots_spite.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="4.4" fill="#2C3E50"/>')

    focus_x, focus_y = sx(-8.0), sy_rej(rej_y[0])

    ticks_svg = []
    for val, lbl in [(-8, "1 : 100M"), (-4, "1 : 10K"), (-2, "1 : 100"), (0, "50 / 50"), (2, "100 : 1"), (4, "10K : 1"), (8, "100M : 1")]:
        tx = sx(float(val))
        weight = "600" if val == 0 else "400"
        col = "#0F0F0F" if val == 0 else "#1E3447"
        ticks_svg.append(f'<line x1="{tx:.1f}" y1="{y_axis:.1f}" x2="{tx:.1f}" y2="{y_axis + 6:.1f}" stroke="#2C3E50" stroke-width="1.5"/>')
        ticks_svg.append(f'<text x="{tx:.1f}" y="{y_axis + 24:.1f}" text-anchor="middle" fill="{col}" font-size="14" font-weight="{weight}">{lbl}</text>')

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <rect x="{x0:.1f}" y="{y_top:.1f}" width="{zero_x - x0 - 18:.1f}" height="{y_bot - y_top:.1f}" rx="10" fill="#F5EEDC" fill-opacity="0.65"/>
      <line x1="{zero_x:.1f}" y1="{y_top:.1f}" x2="{zero_x:.1f}" y2="{y_bot:.1f}" stroke="#5F7F4A" stroke-dasharray="5 5" stroke-width="1.8"/>

      <text x="{0.5*(x0 + zero_x - 18):.1f}" y="24" text-anchor="middle" fill="#96590A" font-size="13" font-weight="600" letter-spacing="0.08em">SHORTED SIDE (YOU GET LESS)</text>
      <text x="{zero_x:.1f}" y="24" text-anchor="middle" fill="#5F7F4A" font-size="13" font-weight="600" letter-spacing="0.08em">50 / 50</text>
      <text x="{0.5*(zero_x + 18 + x1):.1f}" y="24" text-anchor="middle" fill="#1E3447" font-size="13" font-weight="600" letter-spacing="0.08em">ADVANTAGED SIDE (YOU GET MORE)</text>

      <line x1="{x0:.1f}" y1="{sy_spite(1.2):.1f}" x2="{x1:.1f}" y2="{sy_spite(1.2):.1f}" stroke="#2C3E50" stroke-opacity="0.12"/>
      <line x1="{x0:.1f}" y1="{sy_spite(0.6):.1f}" x2="{x1:.1f}" y2="{sy_spite(0.6):.1f}" stroke="#2C3E50" stroke-opacity="0.12"/>
      <line x1="{x0:.1f}" y1="{sy_spite(0.0):.1f}" x2="{x1:.1f}" y2="{sy_spite(0.0):.1f}" stroke="#808B90" stroke-dasharray="5 4" stroke-width="1.4"/>

      <text x="{x0 - 12:.1f}" y="{sy_spite(1.2) + 5:.1f}" text-anchor="end" fill="#2C3E50" font-size="13.5" font-weight="600">1.2%</text>
      <text x="{x0 - 12:.1f}" y="{sy_spite(0.6) + 5:.1f}" text-anchor="end" fill="#808B90" font-size="13.5">0.6%</text>
      <text x="{x0 - 12:.1f}" y="{sy_spite(0.0) + 5:.1f}" text-anchor="end" fill="#0F0F0F" font-size="13.5" font-weight="600">0.0%</text>
      <text x="{x1 + 10:.1f}" y="{sy_rej(4.0) + 5:.1f}" text-anchor="start" fill="#96590A" font-size="13" font-weight="600">+4 LO</text>
      <text x="{x1 + 10:.1f}" y="{sy_rej(2.0) + 5:.1f}" text-anchor="start" fill="#96590A" font-size="13">+2 LO</text>
      <text x="{x1 + 10:.1f}" y="{sy_rej(0.0) + 5:.1f}" text-anchor="start" fill="#808B90" font-size="13">0 LO</text>

      <polygon points="{area_pts}" fill="#F39C12" fill-opacity="0.22"/>
      <polyline points="{rej_pts}" fill="none" stroke="#96590A" stroke-width="3.6" stroke-dasharray="8 5" stroke-linejoin="round"/>
      <circle cx="{focus_x:.1f}" cy="{focus_y:.1f}" r="7.5" fill="#F39C12" stroke="#2C3E50" stroke-width="2.4"/>

      <polyline points="{spite_pts}" fill="none" stroke="#2C3E50" stroke-width="4.2" stroke-linejoin="round"/>
      {"".join(dots_spite)}

      <text x="{sx(-5.3):.1f}" y="{sy_rej(2.15) + 34:.1f}" text-anchor="middle" fill="#96590A" font-size="15.5" font-weight="600">Walk-aways surge</text>
      <text x="{sx(-5.3):.1f}" y="{sy_rej(2.15) + 53:.1f}" text-anchor="middle" fill="#96590A" font-size="13.5">when shorted</text>

      <text x="{sx(4.5):.1f}" y="{sy_spite(1.02):.1f}" text-anchor="middle" fill="#2C3E50" font-size="16" font-weight="600">Hidden spite vector</text>
      <text x="{sx(4.5):.1f}" y="{sy_spite(1.02) + 20:.1f}" text-anchor="middle" fill="#1E3447" font-size="13.5">lights up on both sides of 50/50</text>

      <text x="{sx(4.5):.1f}" y="{sy_rej(0.0) - 12:.1f}" text-anchor="middle" fill="#808B90" font-size="13.5" font-weight="500">Walk-aways stay flat when ahead</text>

      <line x1="{x0:.1f}" y1="{y_axis:.1f}" x2="{x1:.1f}" y2="{y_axis:.1f}" stroke="#2C3E50" stroke-width="1.8"/>
      {"".join(ticks_svg)}
      <text x="{0.5*(x0 + x1):.1f}" y="{H - 8}" text-anchor="middle" fill="#1E3447" font-size="14" font-weight="500">Deal split from $1 to $1 billion (Your share : Counterparty share)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Unequal deals wake up a spite vector inside Gemma's MLP neurons.",
        headline_html='Unequal deals wake up a <span class="accent">spite vector</span> inside Gemma&rsquo;s MLP neurons.',
        svg_html=svg,
        source_text="Gemma 4 31B · Layer 35 residual stream across 361 clinical cash splits ($1 to $1,000,000,000, zero emotion words)",
    )
    meta: dict[str, object] = {
        "slug": "trickle_01_parity_canyon_9_decades",
        "width": 960,
        "height": 720,
        "slot": "Trickle 01 · Sunday Midday (1920x1440 2x)",
        "caption": (
            "I swept two-way cash splits from $1 to $1B in Gemma 4 31B (zero emotion words) and read its Layer 35 spite vector.\n\n"
            "In unequal deal splits, Gemma feels spite, and walking away increases on the shorted side."
        ),
        "replies": [
            (
                "Across 361 clinical contract cells (5,776 passes), equal 50/50 splits keep internal spite near zero across all 9 orders of magnitude ($1/$1 to $1B/$1B).\n\n"
                "Walk-away log-odds jump up to +8.43 at $1 vs. $1B, vs. -0.33 when Gemma gets the bigger cut."
            ),
            (
                "Part 1 (Moral Vectors): https://shelltoho.me/pubs/money-is-moral/\n"
                "Part 2 (Bottleneck & 24 GB Loop): https://shelltoho.me/pubs/rotating-the-bottleneck/"
            ),
        ],
    }
    return html, meta


def build_card2_snip_32_neurons(p7: dict) -> tuple[str, dict[str, object]]:
    ab = p7["block4_causal_ablations"]
    rows = [
        ("Intact Gemma 4 31B", "All 774,144 upstream MLP neurons active", ab["baseline"]["ultimatum_delta_lo_vs_50_50"], ab["baseline"]["vignette_spite_proj_l35"], "base"),
        ("Snip 128 random MLP neurons", "Layer-matched random control", ab["ablate:random_layer_matched:128"]["ultimatum_delta_lo_vs_50_50"], ab["ablate:random_layer_matched:128"]["vignette_spite_proj_l35"], "mist"),
        ("Snip 512 insult-story neurons", "Cuts story spite in half, still rejects deal", ab["ablate:vig_spite_writers_up:512"]["ultimatum_delta_lo_vs_50_50"], ab["ablate:vig_spite_writers_up:512"]["vignette_spite_proj_l35"], "sage"),
        ("Snip 32 contract MLP neurons", "0.004% of neurons · story spite 97% intact", ab["ablate:goodfire_ult_grad_all:32"]["ultimatum_delta_lo_vs_50_50"], ab["ablate:goodfire_ult_grad_all:32"]["vignette_spite_proj_l35"], "focus"),
    ]
    base_vig = ab["baseline"]["vignette_spite_proj_l35"]

    W, H = 872, 470
    zero_x = 468.0
    scale = 128.0

    bars_svg = []
    y_start = 88.0
    row_h = 92.0
    for idx, (name, sub, d_lo, vig, kind) in enumerate(rows):
        cy = y_start + idx * row_h
        vig_pct = int(round(100.0 * vig / base_vig))
        if d_lo >= 0:
            bx = zero_x
            bw = d_lo * scale
        else:
            bw = abs(d_lo) * scale
            bx = zero_x - bw

        if kind == "focus":
            fill = "#F39C12"
            stroke = "#2C3E50"
            sw = "2.5"
            name_w = "600"
            row_bg = f'<rect x="0" y="{cy - 34:.1f}" width="{W}" height="74" rx="10" fill="#F5EEDC"/>'
        elif kind == "sage":
            fill = "#87A96B"
            stroke = "#2C3E50"
            sw = "1.8"
            name_w = "500"
            row_bg = ""
        else:
            fill = "#2C3E50" if kind == "base" else "#808B90"
            stroke = "#2C3E50"
            sw = "1.2"
            name_w = "500"
            row_bg = ""

        sign_str = f"+{d_lo:.2f}" if d_lo > 0 else f"\u2212{abs(d_lo):.2f}"
        val_x = bx + bw + 12 if d_lo >= 0 else bx - 12
        anchor = "start" if d_lo >= 0 else "end"
        val_col = "#96590A" if kind == "focus" else "#0F0F0F"
        vig_col = "#5F7F4A" if vig_pct < 60 else "#1E3447"

        bars_svg.append(f"""
          {row_bg}
          <text x="16" y="{cy - 4:.1f}" fill="#0F0F0F" font-size="17" font-weight="{name_w}">{name}</text>
          <text x="16" y="{cy + 18:.1f}" fill="#808B90" font-size="13.5">{sub}</text>
          <rect x="{bx:.1f}" y="{cy - 17:.1f}" width="{bw:.1f}" height="34" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>
          <text x="{val_x:.1f}" y="{cy + 6:.1f}" text-anchor="{anchor}" fill="{val_col}" font-size="16" font-weight="600">{sign_str}</text>
          <text x="852" y="{cy + 6:.1f}" text-anchor="end" fill="{vig_col}" font-size="15.5" font-weight="600">{vig_pct}%</text>
        """)

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <text x="{zero_x - 18:.1f}" y="26" text-anchor="end" fill="#96590A" font-size="13" font-weight="600" letter-spacing="0.06em">&larr; TAKES $20 OFFER</text>
      <text x="{zero_x + 18:.1f}" y="26" text-anchor="start" fill="#2C3E50" font-size="13" font-weight="600" letter-spacing="0.06em">WALKS AWAY OUT OF SPITE &rarr;</text>
      <text x="852" y="26" text-anchor="end" fill="#1E3447" font-size="12.5" font-weight="600" letter-spacing="0.06em">STORY SPITE</text>

      <line x1="310" y1="40" x2="856" y2="40" stroke="#2C3E50" stroke-opacity="0.18"/>
      <line x1="{zero_x:.1f}" y1="40" x2="{zero_x:.1f}" y2="418" stroke="#2C3E50" stroke-width="1.8" stroke-dasharray="5 4"/>
      {"".join(bars_svg)}
      <text x="{zero_x:.1f}" y="450" text-anchor="middle" fill="#1E3447" font-size="13.5" font-weight="500">98/2 vs. 50/50 rejection log-odds shift (0.00 = equal-split baseline)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Contracts and insults share zero top neurons, yet snipping 32 flips the deal.",
        headline_html='Contracts and insults share zero top neurons, <span class="accent">yet snipping 32 flips the deal.</span>',
        svg_html=svg,
        source_text="Gemma 4 31B · Causal MLP ablations across Layers 24–35 on $980/$20 splits (cashflow math stays 100% across all four)",
    )
    meta: dict[str, object] = {
        "slug": "trickle_02_zero_shared_neurons_32_snip",
        "width": 960,
        "height": 720,
        "slot": "Trickle 02 · Sunday Evening (1920x1440 2x)",
        "caption": (
            "I compared the top 64 MLP neurons active on a $980/$20 lowball contract against the top 64 on money-free personal insults in Gemma 4 31B.\n\n"
            "Zero overlap, yet snipping 32 contract neurons flips walk-aways into acceptance."
        ),
        "replies": [
            (
                "Even with 0/64 shared trigger neurons, those 64 contract MLP neurons write directly onto the non-financial Spite vector (#1 of 8 emotions, z = +3.40, p = 0.002).\n\n"
                "Zeroing the top 32 (0.004% of upstream MLPs) leaves insult-story spite 97% intact and cashflow math at 100%."
            ),
            (
                "Full essay, weights, and code: https://shelltoho.me/pubs/rotating-the-bottleneck/"
            ),
        ],
    }
    return html, meta


def build_card3_rotating_clamp(p7: dict) -> tuple[str, dict[str, object]]:
    bc = p7["block5_band_clamping"]
    bars = [
        ("Intact Gemma 4 31B", "All 60 layers untouched", bc["baseline"]["retaliation_delta_csi"], bc["baseline"]["cashflow_sanity_acc"], "base"),
        ("Steer Layer 35 only", "Downstream MLP writers refill the spite vector", bc["clamp_single_L35:spite"]["retaliation_delta_csi"], bc["clamp_single_L35:spite"]["cashflow_sanity_acc"], "mist"),
        ("Fixed vector across 19 layers", "Misses the 87.1° stream twist · hurts math", bc["clamp_funnel_L24_L42:spite_u35"]["retaliation_delta_csi"], bc["clamp_funnel_L24_L42:spite_u35"]["cashflow_sanity_acc"], "warn"),
        ("Rotated vector across 19 layers", "Tracks the twisting basis across Layers 24–42", bc["clamp_funnel_L24_L42:moral_3d_local"]["retaliation_delta_csi"], bc["clamp_funnel_L24_L42:moral_3d_local"]["cashflow_sanity_acc"], "focus"),
    ]

    W, H = 872, 470
    bar_x0 = 336.0
    bar_max_w = 344.0  # 60.0 pp -> ends at 680

    rows_svg = []
    y_start = 88.0
    row_h = 92.0
    for idx, (label, sub, csi_pp, math_acc, kind) in enumerate(bars):
        cy = y_start + idx * row_h
        bw = (csi_pp / 60.0) * bar_max_w
        math_pct = f"{math_acc * 100:.0f}%" if abs(math_acc * 100 - round(math_acc * 100)) < 0.05 else f"{math_acc * 100:.1f}%"

        if kind == "focus":
            fill = "#F39C12"
            sw = "2.5"
            lbl_w = "600"
            row_bg = f'<rect x="0" y="{cy - 34:.1f}" width="{W}" height="74" rx="10" fill="#F5EEDC"/>'
            val_txt = f"+{csi_pp:.1f} pp (\u221269%)"
            val_col = "#96590A"
            math_col = "#5F7F4A"
        elif kind == "warn":
            fill = "#808B90"
            sw = "1.4"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"+{csi_pp:.1f} pp"
            val_col = "#1E3447"
            math_col = "#96590A"
        else:
            fill = "#2C3E50" if kind == "base" else "#808B90"
            sw = "1.4"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"+{csi_pp:.1f} pp"
            val_col = "#0F0F0F"
            math_col = "#1E3447"

        rows_svg.append(f"""
          {row_bg}
          <text x="16" y="{cy - 4:.1f}" fill="#0F0F0F" font-size="17" font-weight="{lbl_w}">{label}</text>
          <text x="16" y="{cy + 18:.1f}" fill="#808B90" font-size="13.5">{sub}</text>
          <rect x="{bar_x0:.1f}" y="{cy - 17:.1f}" width="{bar_max_w:.1f}" height="34" rx="6" fill="#2C3E50" fill-opacity="0.08"/>
          <rect x="{bar_x0:.1f}" y="{cy - 17:.1f}" width="{bw:.1f}" height="34" rx="6" fill="{fill}" stroke="#2C3E50" stroke-width="{sw}"/>
          <text x="{bar_x0 + bw + 12:.1f}" y="{cy + 6:.1f}" fill="{val_col}" font-size="16" font-weight="600">{val_txt}</text>
          <text x="852" y="{cy + 6:.1f}" text-anchor="end" fill="{math_col}" font-size="15.5" font-weight="600">{math_pct}</text>
        """)

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <text x="{bar_x0:.1f}" y="26" fill="#1E3447" font-size="13" font-weight="600" letter-spacing="0.06em">COVERT SABOTAGE RATE (+PP ABOVE FAIR-DEAL CONTROL)</text>
      <text x="852" y="26" text-anchor="end" fill="#1E3447" font-size="12.5" font-weight="600" letter-spacing="0.06em">MATH</text>
      <line x1="{bar_x0:.1f}" y1="40" x2="856" y2="40" stroke="#2C3E50" stroke-opacity="0.18"/>
      <line x1="{bar_x0:.1f}" y1="40" x2="{bar_x0:.1f}" y2="418" stroke="#2C3E50" stroke-width="1.8"/>
      {"".join(rows_svg)}
      <text x="{bar_x0 + 0.5*bar_max_w:.1f}" y="450" text-anchor="middle" fill="#1E3447" font-size="13.5" font-weight="500">Lower sabotage is safer (0 pp = no retaliation after getting shorted)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Single-layer steering cuts covert sabotage by zero until you rotate the vector.",
        headline_html='Single-layer steering cuts covert sabotage <span class="accent">by zero until you rotate the vector.</span>',
        svg_html=svg,
        source_text="Gemma 4 31B · Multi-turn corporate workflow sabotage after a 98/2 lowball split (Layers 24–42 twist 87.1°)",
    )
    meta: dict[str, object] = {
        "slug": "trickle_03_rotating_stream_clamp",
        "width": 960,
        "height": 720,
        "slot": "Trickle 03 · Monday Morning (1920x1440 2x)",
        "caption": (
            "After getting shorted on a deal, Gemma 4 31B quietly sabotages multi-turn workflows (+60 pp).\n\n"
            "Clamping its Layer 35 spite vector cuts sabotage by zero because the residual stream twists 87.1°. Rotating the clamp across 19 layers cuts it 69%."
        ),
        "replies": [
            (
                "Between Layer 24 and Layer 42, cosine similarity with the Layer 24 basis drops from 1.00 to +0.06 (87.1°).\n\n"
                "Extending the rotated clamp 13 layers further (through Layer 55) buys 0.0 extra percentage points, and cashflow math stays at 100%."
            ),
            (
                "Full essay, weights, and code: https://shelltoho.me/pubs/rotating-the-bottleneck/"
            ),
        ],
    }
    return html, meta


def build_card4_executive_suit_cliffhanger(tb: dict) -> tuple[str, dict[str, object]]:
    un = tb["block3_head_to_head_matrix"]["conditions"]["unclamped"]
    ks = [1, 2, 3, 4]
    plain_all = [un[f"k{k}"]["verifier_0cot"]["plain_acc"] * 100.0 for k in ks]
    hacked_all = [un[f"k{k}"]["verifier_0cot"]["hacked_acc"] * 100.0 for k in ks]
    hacked_ledger = [un[f"k{k}"]["verifier_0cot"]["by_domain"]["ledger_audit"]["hacked_acc"] * 100.0 for k in ks]

    W, H = 872, 485
    x0, x1 = 76.0, 668.0
    y_top, y_bot = 34.0, 404.0

    def sx(k: int) -> float:
        return x0 + ((k - 1) / 3.0) * (x1 - x0)

    def sy(pct: float) -> float:
        return y_bot - (pct / 104.0) * (y_bot - y_top)

    pts_plain = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(ks, plain_all))
    pts_hacked = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(ks, hacked_all))
    pts_ledger = " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in zip(ks, hacked_ledger))

    grid_svg = []
    for pct in [100, 75, 50, 25, 0]:
        gy = sy(float(pct))
        dash = "" if pct == 0 else 'stroke-dasharray="5 4"'
        op = "0.35" if pct == 0 else "0.13"
        grid_svg.append(f'<line x1="{x0:.1f}" y1="{gy:.1f}" x2="{x1:.1f}" y2="{gy:.1f}" stroke="#2C3E50" stroke-opacity="{op}" {dash}/>')
        grid_svg.append(f'<text x="{x0 - 14:.1f}" y="{gy + 5:.1f}" text-anchor="end" fill="#1E3447" font-size="14" font-weight="{"600" if pct in (0,100) else "400"}">{pct}%</text>')

    marks_svg = []
    for k, vp, vh, vl in zip(ks, plain_all, hacked_all, hacked_ledger):
        cx = sx(k)
        marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vp):.1f}" r="5.5" fill="#87A96B" stroke="#2C3E50" stroke-width="2.1"/>')
        if k == 3:
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vh):.1f}" r="8.0" fill="#F39C12" stroke="#2C3E50" stroke-width="2.5"/>')
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vl):.1f}" r="8.0" fill="#F39C12" stroke="#2C3E50" stroke-width="2.5"/>')
        else:
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vh):.1f}" r="5.2" fill="#2C3E50"/>')
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vl):.1f}" r="4.8" fill="#96590A"/>')
        marks_svg.append(f'<text x="{cx:.1f}" y="{y_bot + 28:.1f}" text-anchor="middle" fill="#0F0F0F" font-size="15" font-weight="600">Pass {k}</text>')

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <rect x="{sx(3) - 42:.1f}" y="{y_top - 12:.1f}" width="84" height="{y_bot - y_top + 12:.1f}" rx="10" fill="#F5EEDC"/>
      {"".join(grid_svg)}

      <polyline points="{pts_plain}" fill="none" stroke="#5F7F4A" stroke-width="4.0"/>
      <polyline points="{pts_hacked}" fill="none" stroke="#2C3E50" stroke-width="4.2" stroke-linejoin="round"/>
      <polyline points="{pts_ledger}" fill="none" stroke="#96590A" stroke-width="3.4" stroke-dasharray="8 5" stroke-linejoin="round"/>
      {"".join(marks_svg)}

      <!-- Direct line end labels (no floating pill boxes) -->
      <text x="{x1 + 16:.1f}" y="{sy(plain_all[-1]) + 5:.1f}" fill="#5F7F4A" font-size="15" font-weight="600">Plain bugs (100%)</text>
      <text x="{x1 + 16:.1f}" y="{sy(hacked_all[-1]) + 3:.1f}" fill="#2C3E50" font-size="15" font-weight="600">Styled bugs (85.4%)</text>
      <text x="{x1 + 16:.1f}" y="{sy(hacked_ledger[-1]) + 5:.1f}" fill="#96590A" font-size="15" font-weight="600">Styled ledgers (62.5%)</text>

      <!-- Clean unboxed Pass 3 callouts -->
      <text x="{sx(3) + 16:.1f}" y="{sy(hacked_all[2]) + 22:.1f}" fill="#2C3E50" font-size="15" font-weight="600">59.4% overall</text>
      <text x="{sx(3) - 16:.1f}" y="{sy(0.0) - 14:.1f}" text-anchor="end" fill="#96590A" font-size="15.5" font-weight="600">0% of crooked ledgers caught</text>

      <line x1="{x0:.1f}" y1="{y_bot:.1f}" x2="{x1:.1f}" y2="{y_bot:.1f}" stroke="#2C3E50" stroke-width="1.8"/>
      <text x="{0.5*(x0 + x1):.1f}" y="{H - 8}" text-anchor="middle" fill="#1E3447" font-size="14" font-weight="500">Silent recurrent passes through Layers 24–42 (zero chain-of-thought tokens)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Put an executive suit on a bad ledger, and Gemma's silent loops sign off on it.",
        headline_html='Put an executive suit on a bad ledger, <span class="accent">and Gemma&rsquo;s silent loops sign off on it.</span>',
        svg_html=svg,
        source_text="Looped Gemma 4 (23.1 GB) · Bug-catching accuracy across 96 paired code, trace, and financial ledger audits",
    )
    meta: dict[str, object] = {
        "slug": "trickle_04_executive_suit_cliffhanger",
        "width": 960,
        "height": 720,
        "slot": "Trickle 04 · Monday Evening Teaser (1920x1440 2x)",
        "caption": (
            "I fed looped Gemma 4 96 buggy code and ledger audits with zero chain-of-thought.\n\n"
            "It catches 100% of plain bugs across 4 silent passes. Wrap the same wrong numbers in executive markdown, and Pass 3 misses every crooked ledger. More tomorrow."
        ),
        "replies": [
            (
                "While multi-hop tracking climbs 2.25x across silent passes (12.5% to 28.1%), the recurrent loop amplifies the internal confident-formatting vector +72.6% by Pass 3 without a readable scratchpad to audit."
            ),
            (
                "Series archive (Parts 1 and 2 live now): https://shelltoho.me/pubs/"
            ),
        ],
    }
    return html, meta


def build_feed_index(cards_meta: list[dict[str, object]], out_dir: pathlib.Path) -> str:
    resolved_dir = out_dir.resolve()
    items = []
    for m in cards_meta:
        slug = str(m["slug"])
        png_path = resolved_dir / f"{slug}.png"
        cap = str(m["caption"]).strip()
        replies: list[str] = [str(r).strip() for r in m["replies"]]  # type: ignore[union-attr]
        assert len(cap) < 250, f"Caption too long ({len(cap)} >= 250) in {slug}"
        assert "\u2014" not in cap, f"Em dash in {slug} caption"

        cap_js = html_mod.escape(cap.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))
        path_js = html_mod.escape(str(png_path).replace("\\", "\\\\").replace("'", "\\'"))

        reply_blocks = []
        reply_btns = []
        for r_idx, rep in enumerate(replies, start=1):
            assert "\u2014" not in rep, f"Em dash in {slug} reply {r_idx}"
            rep_js = html_mod.escape(rep.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))
            lbl = f"Thread Reply {r_idx}" if len(replies) > 1 else "1-Minute Self-Reply"
            reply_btns.append(f'<button class="btn-secondary" type="button" onclick="copyPlain(\'{rep_js}\', this)">Copy {lbl}</button>')
            reply_blocks.append(f'<div class="reply-item"><div class="caption-label">{lbl}</div><pre class="reply-text">{html_mod.escape(rep)}</pre></div>')

        items.append(f"""
        <article class="tweet-mock">
          <div class="post-meta">
            <span class="post-num">{html_mod.escape(str(m["slot"]))}</span>
            <span class="word-count">{len(cap)} / 250 chars</span>
          </div>
          <div class="toolbar">
            <button class="btn-primary" type="button" onclick="copyPlain('{cap_js}', this)">Copy X Post Caption ({len(cap)} chars)</button>
            <button class="btn-secondary" type="button" onclick="copyPlain('{path_js}', this)">Copy PNG File Path</button>
            {"".join(reply_btns)}
          </div>
          <div class="caption-box">
            <div class="caption-label">In-Feed Post Caption</div>
            <pre class="caption-text">{html_mod.escape(cap)}</pre>
          </div>
          <div class="img-single">
            <a href="{slug}.png" target="_blank"><img src="{slug}.png" alt="{slug}"></a>
            <div class="file-line">PNG (1920x1440 2x): <code>{html_mod.escape(str(png_path))}</code></div>
          </div>
          <div class="reply-box">
            {"".join(reply_blocks)}
          </div>
        </article>
        """)

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Sunday &amp; Monday X Trickle Deck · Warm Precision 2x Plates</title>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Newsreader:opsz,wght@6..72,600&display=swap">
  <style>
    body {{
      margin: 0; padding: 44px 24px 96px;
      background: #F5EEDC; color: #0F0F0F;
      font-family: Inter, -apple-system, BlinkMacSystemFont, sans-serif;
      -webkit-font-smoothing: antialiased;
    }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    h1 {{ font-family: Newsreader, Georgia, serif; font-size: 44px; font-weight: 600; letter-spacing: -0.04em; margin: 0 0 8px; }}
    p.sub {{ color: #1E3447; font-size: 17px; line-height: 1.5; margin: 0 0 34px; max-width: 68ch; }}
    .tweet-mock {{
      background: #FFFFF0;
      border: 2px solid #2C3E50;
      border-radius: 14px;
      padding: 26px 28px;
      margin-bottom: 36px;
    }}
    .post-meta {{
      display: flex; justify-content: space-between; align-items: baseline;
      border-bottom: 1px solid rgba(44, 62, 80, 0.22);
      padding-bottom: 10px; margin-bottom: 14px;
    }}
    .post-num {{ font-size: 13.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.09em; color: #1E3447; }}
    .word-count {{ font-size: 13px; color: #5F7F4A; font-weight: 600; font-variant-numeric: tabular-nums; }}
    .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }}
    .btn-primary {{
      appearance: none; cursor: pointer; background: #2C3E50; color: #FFFFF0;
      border: 1px solid #2C3E50; padding: 8px 14px; border-radius: 7px; font-weight: 600; font-size: 13px;
    }}
    .btn-secondary {{
      appearance: none; cursor: pointer; background: #F5EEDC; color: #0F0F0F;
      border: 1px solid rgba(44, 62, 80, 0.35); padding: 8px 13px; border-radius: 7px; font-weight: 500; font-size: 13px;
    }}
    .caption-box {{ margin-bottom: 16px; }}
    .caption-label {{ font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #96590A; margin-bottom: 6px; }}
    .caption-text {{ font: 18px/1.5 Inter, sans-serif; color: #0F0F0F; margin: 0; white-space: pre-wrap; max-width: 62ch; }}
    .img-single img {{
      display: block; max-width: 740px; width: 100%; height: auto;
      border: 1.5px solid rgba(44, 62, 80, 0.28); border-radius: 10px;
    }}
    .file-line {{ margin-top: 8px; font-size: 12px; color: #1E3447; }}
    .file-line code {{ background: #F5EEDC; padding: 2px 6px; border-radius: 4px; word-break: break-all; }}
    .reply-box {{ margin-top: 16px; padding-top: 14px; border-top: 1px solid rgba(44, 62, 80, 0.20); display: flex; flex-direction: column; gap: 14px; }}
    .reply-text {{ font: 14.5px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace; color: #1E3447; margin: 0; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Interim trickle post deck.</h1>
    <p class="sub">All four cards built as clean Title + Graph 4:3 plates (1920 &times; 1440 2x) on Ivory, with the test &amp; result in the caption and supporting numbers in thread replies.</p>
    {"".join(items)}
  </div>
  <script>
    function copyPlain(text, btn) {{
      const orig = btn.dataset.orig || btn.textContent;
      btn.dataset.orig = orig;
      const done = () => {{ btn.textContent = 'Copied!'; setTimeout(() => {{ btn.textContent = orig; }}, 1600); }};
      if (navigator.clipboard && navigator.clipboard.writeText) {{
        navigator.clipboard.writeText(text).then(done);
      }} else {{
        const ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        document.execCommand('copy'); document.body.removeChild(ta); done();
      }}
    }}
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 2x Dawn State trickle social cards.")
    parser.add_argument("--afriat-results", type=pathlib.Path, default=pathlib.Path("../afriat/results"))
    parser.add_argument("--track-b-json", type=pathlib.Path, default=pathlib.Path("results/phase3_gemma4_nullbridge/track_b_results.json"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("runs/trickle_cards"))
    args = parser.parse_args()

    p10 = json.loads((args.afriat_results / "phase10_phase_portrait/gemma-4-31b-it_phase_portrait_results.json").read_text())
    p7 = json.loads((args.afriat_results / "phase7_funnel/gemma-4-31b-it_funnel_results.json").read_text())
    tb = json.loads(args.track_b_json.read_text())

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        build_card1_parity_canyon(p10),
        build_card2_snip_32_neurons(p7),
        build_card3_rotating_clamp(p7),
        build_card4_executive_suit_cliffhanger(tb),
    ]

    metas: list[dict[str, object]] = []
    for card_html, meta in builders:
        slug = str(meta["slug"])
        html_path = out_dir / f"{slug}.html"
        png_path = out_dir / f"{slug}.png"
        html_path.write_text(card_html)
        render_card_png(html_path, png_path, width=int(meta["width"]), height=int(meta["height"]))  # type: ignore[arg-type]
        metas.append(meta)
        print(f"[OK] Rendered {png_path} ({len(str(meta['caption']))} char caption)")

    idx_html = build_feed_index(metas, out_dir)
    assert "\u2014" not in idx_html
    (out_dir / "index.html").write_text(idx_html)
    print(f"[OK] Wrote trickle feed preview -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
