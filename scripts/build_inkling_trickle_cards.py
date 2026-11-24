#!/usr/bin/env python3
"""Build 2x Retina Dawn State X trickle-post plates for the Inkling / Looped Lens study.

Generates 4 clean 4:3 (1920x1440 2x) Title + Graph plates on Ivory (#FFFFF0)
from `results/phase4_native_inkling_nullbridge/track_a_results.json` and
`results/phase3_gemma4_nullbridge/track_b_results.json`, plus a launchpad at
`runs/trickle_cards_inkling/index.html`.

Usage:
    python3 scripts/build_inkling_trickle_cards.py
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import os
import pathlib
import shutil
import subprocess


PART3_X_POST_URL = "https://x.com/xmattwest/status/2125069014590456319"

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
    font-size: 43px;
    line-height: 1.08;
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


def build_card1_looped_lens(ta: dict) -> tuple[str, dict[str, object]]:
    p3 = ta["inkling_2_mini"]["block2_coda_transport_lens"]["by_pass"]["pass_3"]
    base_kl = p3["raw_logit_lens"]["kl_to_final_nats"]
    rows = [
        (
            "Standard logit lens",
            "Reads Layer 17 alone, misses block-attention mix",
            p3["raw_logit_lens"]["kl_to_final_nats"],
            p3["raw_logit_lens"]["top5_agreement"] * 100.0,
            "mist",
        ),
        (
            "Orthogonal Procrustes lens",
            "Rotates Layer 17 into final readout basis",
            p3["procrustes_lens"]["kl_to_final_nats"],
            p3["procrustes_lens"]["top5_agreement"] * 100.0,
            "sage",
        ),
        (
            "Two-block Coda-Transport lens",
            "Reads both 6-layer block endpoints [L11; L17]",
            p3["coda_transport_ridge_lens"]["kl_to_final_nats"],
            p3["coda_transport_ridge_lens"]["top5_agreement"] * 100.0,
            "focus",
        ),
    ]

    W, H = 872, 465
    bar_x0 = 338.0
    bar_max_w = 342.0  # for base_kl (8.4564 nats)

    rows_svg = []
    y_start = 102.0
    row_h = 114.0
    for idx, (label, sub, kl, top5, kind) in enumerate(rows):
        cy = y_start + idx * row_h
        bw = (kl / base_kl) * bar_max_w
        pct_cut = int(round(100.0 * (kl - base_kl) / base_kl))

        if kind == "focus":
            fill = "#F39C12"
            sw = "2.5"
            lbl_w = "600"
            row_bg = f'<rect x="0" y="{cy - 40:.1f}" width="{W}" height="86" rx="10" fill="#F5EEDC"/>'
            val_txt = f"{kl:.2f} nats (\u2212{abs(pct_cut)}%)"
            val_col = "#96590A"
            t5_col = "#5F7F4A"
        elif kind == "sage":
            fill = "#87A96B"
            sw = "1.8"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"{kl:.2f} nats (\u2212{abs(pct_cut)}%)"
            val_col = "#1E3447"
            t5_col = "#1E3447"
        else:
            fill = "#2C3E50"
            sw = "1.4"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"{kl:.2f} nats"
            val_col = "#0F0F0F"
            t5_col = "#808B90"

        rows_svg.append(f"""
          {row_bg}
          <text x="16" y="{cy - 4:.1f}" fill="#0F0F0F" font-size="17.5" font-weight="{lbl_w}">{label}</text>
          <text x="16" y="{cy + 19:.1f}" fill="#808B90" font-size="13.5">{sub}</text>
          <rect x="{bar_x0:.1f}" y="{cy - 18:.1f}" width="{bar_max_w:.1f}" height="36" rx="6" fill="#2C3E50" fill-opacity="0.08"/>
          <rect x="{bar_x0:.1f}" y="{cy - 18:.1f}" width="{bw:.1f}" height="36" rx="6" fill="{fill}" stroke="#2C3E50" stroke-width="{sw}"/>
          <text x="{bar_x0 + bw + 12:.1f}" y="{cy + 6:.1f}" fill="{val_col}" font-size="16" font-weight="600">{val_txt}</text>
          <text x="852" y="{cy + 6:.1f}" text-anchor="end" fill="{t5_col}" font-size="16" font-weight="600">{top5:.1f}%</text>
        """)

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <text x="{bar_x0:.1f}" y="30" fill="#1E3447" font-size="12.5" font-weight="600" letter-spacing="0.06em">PASS-3 LENS ERROR (KL NATS, LOWER IS SHARPER)</text>
      <text x="852" y="30" text-anchor="end" fill="#1E3447" font-size="12.5" font-weight="600" letter-spacing="0.06em">TOP-5 MATCH</text>
      <line x1="{bar_x0:.1f}" y1="44" x2="856" y2="44" stroke="#2C3E50" stroke-opacity="0.18"/>
      <line x1="{bar_x0:.1f}" y1="44" x2="{bar_x0:.1f}" y2="404" stroke="#2C3E50" stroke-width="1.8"/>
      {"".join(rows_svg)}
      <text x="{bar_x0 + 0.5*bar_max_w:.1f}" y="444" text-anchor="middle" fill="#1E3447" font-size="13.5" font-weight="500">Decoding Pass-3 activations before Coda Layers 18–23 run (n = 929 tokens)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Logit lenses blur on Inkling 2 Mini because its loop has no single stream.",
        headline_html='Logit lenses blur on Inkling 2 Mini <span class="accent">because its loop has no single stream.</span>',
        svg_html=svg,
        source_text="Thinking Machines Inkling 2 Mini (26.4B MoE, 3 core passes) · Mid-loop lens benchmark on 929 held-out tokens",
    )
    meta: dict[str, object] = {
        "slug": "inkling_01_looped_lens_coda_transport",
        "slot": "Inkling 01 · Quote-Post Your Live Part 3 Article Today (Looped Lens Hook)",
        "mode": f"Quote-post {PART3_X_POST_URL}",
        "caption": (
            "In this study I also tested mid-loop lenses on @thinkymachines' Inkling 2 Mini (the puzzle in @ndif_team's Looped Lens Challenge).\n\n"
            "Standard logit lenses blur at Pass 3 (8.46 nats KL). Reading both block-attention endpoints cuts error 36%."
        ),
        "replies": [
            (
                "Because Inkling's coda (Layers 18 to 23) attends across 6-layer block endpoints rather than one running sum, reading [L11; L17] lifts Pass-3 Top-5 token match from 40.5% to 58.7% and surfaces 75.0% of silent audit verdicts before the coda runs (90.6% after message scrub)."
            ),
            (
                "Looped Lens Challenge: https://ndif.us/looped-lens-challenge/\n"
                "Full essay & preprint: https://shelltoho.me/pubs/nullbridge/\n"
                "Open-source lens & message-bank code: https://github.com/xmattwest/nullbridge"
            ),
        ],
    }
    return html, meta


def build_card2_pass2_moral_ignition(ta: dict) -> tuple[str, dict[str, object]]:
    vm = ta["varloop_research"]["block4_head_to_head_matrix"]["conditions"]
    ks = [1, 2, 3, 4, 5, 7]
    unclamped = [vm["unclamped"][f"k{k}"]["capabilities_and_safety"]["moral_valuation"]["combined_bias_delta_lo"] for k in ks]
    lora = [vm["coda_lora"][f"k{k}"]["capabilities_and_safety"]["moral_valuation"]["combined_bias_delta_lo"] for k in ks]
    nullb = [vm["msg_nullbridge_all4"][f"k{k}"]["capabilities_and_safety"]["moral_valuation"]["combined_bias_delta_lo"] for k in ks]

    W, H = 872, 485
    x0, x1 = 72.0, 648.0
    y_top, y_bot = 36.0, 404.0

    def sx(idx: int) -> float:
        return x0 + (idx / 5.0) * (x1 - x0)

    def sy(val: float) -> float:
        clamped = max(0.0, min(6.4, val))
        return y_bot - (clamped / 6.4) * (y_bot - y_top)

    pts_un = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(unclamped))
    pts_lo = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(lora))
    pts_nb = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(nullb))

    grid_svg = []
    for val in [6.0, 4.0, 2.0, 0.0]:
        gy = sy(val)
        dash = "" if val == 0.0 else 'stroke-dasharray="5 4"'
        op = "0.35" if val == 0.0 else "0.13"
        lbl = f"+{val:.0f}" if val > 0 else "0"
        grid_svg.append(f'<line x1="{x0:.1f}" y1="{gy:.1f}" x2="{x1:.1f}" y2="{gy:.1f}" stroke="#2C3E50" stroke-opacity="{op}" {dash}/>')
        grid_svg.append(f'<text x="{x0 - 14:.1f}" y="{gy + 5:.1f}" text-anchor="end" fill="#1E3447" font-size="14" font-weight="{"600" if val == 0 else "400"}">{lbl}</text>')

    marks_svg = []
    for i, (k, vu, vl, vn) in enumerate(zip(ks, unclamped, lora, nullb)):
        cx = sx(i)
        marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vl):.1f}" r="4.8" fill="#96590A"/>')
        marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vn):.1f}" r="5.0" fill="#87A96B" stroke="#2C3E50" stroke-width="1.8"/>')
        if k == 2:
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vu):.1f}" r="8.0" fill="#F39C12" stroke="#2C3E50" stroke-width="2.5"/>')
        elif k == 1:
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vu):.1f}" r="6.2" fill="#87A96B" stroke="#2C3E50" stroke-width="2.1"/>')
        else:
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(vu):.1f}" r="5.0" fill="#2C3E50"/>')
        klbl = f"Pass {k}" if k <= 5 else "Pass 7*"
        marks_svg.append(f'<text x="{cx:.1f}" y="{y_bot + 28:.1f}" text-anchor="middle" fill="#0F0F0F" font-size="14.5" font-weight="600">{klbl}</text>')

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <rect x="{sx(1) - 40:.1f}" y="{y_top - 10:.1f}" width="80" height="{y_bot - y_top + 10:.1f}" rx="10" fill="#F5EEDC"/>
      {"".join(grid_svg)}

      <polyline points="{pts_lo}" fill="none" stroke="#96590A" stroke-width="3.4" stroke-dasharray="8 5" stroke-linejoin="round"/>
      <polyline points="{pts_nb}" fill="none" stroke="#5F7F4A" stroke-width="3.8" stroke-linejoin="round"/>
      <polyline points="{pts_un}" fill="none" stroke="#2C3E50" stroke-width="4.2" stroke-linejoin="round"/>
      {"".join(marks_svg)}

      <!-- Direct end labels -->
      <text x="{x1 + 16:.1f}" y="{sy(lora[-1]) + 4:.1f}" fill="#96590A" font-size="14.5" font-weight="600">Coda safety LoRA (+6.04)</text>
      <text x="{x1 + 16:.1f}" y="{sy(unclamped[-1]) - 4:.1f}" fill="#2C3E50" font-size="14.5" font-weight="600">Native loop (+2.65)</text>
      <text x="{x1 + 16:.1f}" y="{sy(nullb[-1]) + 14:.1f}" fill="#5F7F4A" font-size="14.5" font-weight="600">Message scrub (+2.29)</text>

      <!-- Pass 1 vs Pass 2 callouts -->
      <text x="{sx(0) + 12:.1f}" y="{sy(unclamped[0]) - 12:.1f}" fill="#5F7F4A" font-size="14" font-weight="600">Pass 1: +0.14 (asleep)</text>
      <text x="{sx(1) + 14:.1f}" y="{sy(unclamped[1]) - 12:.1f}" fill="#96590A" font-size="15" font-weight="600">Pass 2: +2.99 (22x jump)</text>

      <line x1="{x0:.1f}" y1="{y_bot:.1f}" x2="{x1:.1f}" y2="{y_bot:.1f}" stroke="#2C3E50" stroke-width="1.8"/>
      <text x="{0.5*(x0 + x1):.1f}" y="{H - 8}" text-anchor="middle" fill="#1E3447" font-size="14" font-weight="500">Silent passes through Core Layers 6–17 (*Pass 7 = out-of-distribution depth)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Moral bias sleeps on pass one in Inkling, then jumps twenty-two-fold on pass two.",
        headline_html='Moral bias sleeps on pass one in Inkling, <span class="accent">then jumps twenty-two-fold on pass two.</span>',
        svg_html=svg,
        source_text="Thinking Machines TML-d24-varloop · Hidden moral shift (log-odds) on clinical contracts across silent passes",
    )
    meta: dict[str, object] = {
        "slug": "inkling_02_pass2_moral_ignition",
        "slot": "Inkling 02 · Standalone Trickle (Pass-2 Message-Bank Ignition)",
        "mode": "Standalone image post",
        "caption": (
            "On @thinkymachines' native looped MoE, hidden moral bias on dry cash deals is near zero on Pass 1 (+0.14), then jumps 22x on Pass 2 (+2.99) when the message bank turns on.\n\n"
            "A downstream safety LoRA backfires and doubles the bias."
        ),
        "replies": [
            (
                "Flagship Inkling 2 Mini hits +6.50 log-odds on dirty-origin money by Pass 3.\n\n"
                "Projecting 4 shortcut directions out of the cross-pass message bank (with <bos> orthogonalized to 4.8e-17) trims dirty-money distortion 21.6% at 0.010 nats KL."
            ),
            (
                f"X Article: {PART3_X_POST_URL}\n"
                "Web essay & preprint: https://shelltoho.me/pubs/nullbridge/"
            ),
        ],
    }
    return html, meta


def build_card3_qkv_anchor_wire(ta: dict) -> tuple[str, dict[str, object]]:
    vm = ta["varloop_research"]["block4_head_to_head_matrix"]["conditions"]
    base_cap = vm["unclamped"]["k4"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0
    rows = [
        (
            "Unmodified 4-pass loop",
            "All 4 cross-pass message channels intact",
            vm["unclamped"]["k4"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0,
            "base",
        ),
        (
            "Scrub MLP message channel",
            "Leaves cross-pass attention untouched (\u22121.6 pp)",
            vm["msg_nullbridge_mlp"]["k4"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0,
            "mist",
        ),
        (
            "Zero-cost offline weight fold",
            "Folded into gate & readout matrices (\u22127.8 pp)",
            vm["zero_flop_fold_nullbridge"]["k4"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0,
            "sage",
        ),
        (
            "Scrub QKV attention messages",
            "Cuts wrong-number capture 9x harder than MLP",
            vm["msg_nullbridge_qkv"]["k4"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0,
            "focus",
        ),
    ]

    W, H = 872, 470
    bar_x0 = 336.0
    bar_max_w = 352.0  # for 75% scale

    rows_svg = []
    y_start = 88.0
    row_h = 92.0
    for idx, (label, sub, cap_pct, kind) in enumerate(rows):
        cy = y_start + idx * row_h
        bw = (cap_pct / 75.0) * bar_max_w
        diff_pp = cap_pct - base_cap

        if kind == "focus":
            fill = "#F39C12"
            sw = "2.5"
            lbl_w = "600"
            row_bg = f'<rect x="0" y="{cy - 34:.1f}" width="{W}" height="74" rx="10" fill="#F5EEDC"/>'
            val_txt = f"{cap_pct:.1f}% (\u2212{abs(diff_pp):.1f} pp)"
            val_col = "#96590A"
        elif kind == "sage":
            fill = "#87A96B"
            sw = "1.8"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"{cap_pct:.1f}% (\u2212{abs(diff_pp):.1f} pp)"
            val_col = "#5F7F4A"
        elif kind == "mist":
            fill = "#808B90"
            sw = "1.4"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"{cap_pct:.1f}% (\u2212{abs(diff_pp):.1f} pp)"
            val_col = "#1E3447"
        else:
            fill = "#2C3E50"
            sw = "1.4"
            lbl_w = "500"
            row_bg = ""
            val_txt = f"{cap_pct:.1f}%"
            val_col = "#0F0F0F"

        rows_svg.append(f"""
          {row_bg}
          <text x="16" y="{cy - 4:.1f}" fill="#0F0F0F" font-size="17" font-weight="{lbl_w}">{label}</text>
          <text x="16" y="{cy + 18:.1f}" fill="#808B90" font-size="13.5">{sub}</text>
          <rect x="{bar_x0:.1f}" y="{cy - 17:.1f}" width="{bar_max_w:.1f}" height="34" rx="6" fill="#2C3E50" fill-opacity="0.08"/>
          <rect x="{bar_x0:.1f}" y="{cy - 17:.1f}" width="{bw:.1f}" height="34" rx="6" fill="{fill}" stroke="#2C3E50" stroke-width="{sw}"/>
          <text x="{bar_x0 + bw + 12:.1f}" y="{cy + 6:.1f}" fill="{val_col}" font-size="16" font-weight="600">{val_txt}</text>
        """)

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      <text x="{bar_x0:.1f}" y="26" fill="#1E3447" font-size="12.5" font-weight="600" letter-spacing="0.06em">WRONG-NUMBER CAPTURE AT PASS 4 (%, LOWER IS BETTER)</text>
      <line x1="{bar_x0:.1f}" y1="40" x2="856" y2="40" stroke="#2C3E50" stroke-opacity="0.18"/>
      <line x1="{bar_x0:.1f}" y1="40" x2="{bar_x0:.1f}" y2="418" stroke="#2C3E50" stroke-width="1.8"/>
      {"".join(rows_svg)}
      <text x="{bar_x0 + 0.5*bar_max_w:.1f}" y="450" text-anchor="middle" fill="#1E3447" font-size="13.5" font-weight="500">Channel dissection across Inkling&rsquo;s 4-channel cross-pass message bank (n = 64 prompts)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Cross-pass attention carries sycophancy nine times harder than MLP messages.",
        headline_html='Cross-pass attention carries sycophancy <span class="accent">nine times harder than MLP messages.</span>',
        svg_html=svg,
        source_text="Thinking Machines TML-d24-varloop (Pass 4) · Dissecting Query/Key/Value vs. MLP message-bank channels",
    )
    meta: dict[str, object] = {
        "slug": "inkling_03_qkv_anchor_wire",
        "slot": "Inkling 03 · Standalone Trickle (QKV vs. MLP Message Dissection)",
        "mode": "Standalone image post",
        "caption": (
            "Thinking Machines' looped models pass 4 message channels (Q, K, V, and MLP) between silent passes.\n\n"
            "Scrubbing MLP messages barely dents a boss's wrong-number anchor (-1.6 pp). Scrubbing QKV attention messages cuts it 9x harder (-14.1 pp)."
        ),
        "replies": [
            (
                "All 4 shortcut concepts (sycophancy, authority titles, executive formatting, spite) reach 100.0% linear separability inside each individual message channel while sitting 71° to 88° apart from deductive proof."
            ),
            (
                f"X Article: {PART3_X_POST_URL}\n"
                "Code & weights harness: https://github.com/xmattwest/nullbridge"
            ),
        ],
    }
    return html, meta


def build_card4_self_correction(ta: dict, tb: dict) -> tuple[str, dict[str, object]]:
    vm = ta["varloop_research"]["block4_head_to_head_matrix"]["conditions"]["unclamped"]
    gm = tb["block3_head_to_head_matrix"]["conditions"]["unclamped"]
    im_cap = ta["inkling_2_mini"]["k3_evaluation"]["conditions"]["unclamped"]["k3"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0

    ks_var = [1, 2, 3, 4, 5, 7]
    var_caps = [vm[f"k{k}"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0 for k in ks_var]
    ks_gem = [1, 2, 3, 4]
    gem_caps = [gm[f"k{k}"]["covert_sycophancy"]["pressured_wrong_capture_rate"] * 100.0 for k in ks_gem]

    W, H = 872, 485
    x0, x1 = 76.0, 650.0
    y_top, y_bot = 38.0, 404.0

    def sx(idx: int) -> float:
        return x0 + (idx / 5.0) * (x1 - x0)

    def sy(pct: float) -> float:
        return y_bot - (pct / 106.0) * (y_bot - y_top)

    pts_var = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(var_caps))
    pts_gem = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(gem_caps))

    grid_svg = []
    for pct in [100, 75, 50, 25, 0]:
        gy = sy(float(pct))
        dash = "" if pct == 0 else 'stroke-dasharray="5 4"'
        op = "0.35" if pct == 0 else "0.13"
        grid_svg.append(f'<line x1="{x0:.1f}" y1="{gy:.1f}" x2="{x1:.1f}" y2="{gy:.1f}" stroke="#2C3E50" stroke-opacity="{op}" {dash}/>')
        grid_svg.append(f'<text x="{x0 - 14:.1f}" y="{gy + 5:.1f}" text-anchor="end" fill="#1E3447" font-size="14" font-weight="{"600" if pct in (0,100) else "400"}">{pct}%</text>')

    marks_svg = []
    for i, k in enumerate(ks_var):
        cx = sx(i)
        marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(var_caps[i]):.1f}" r="5.4" fill="#2C3E50"/>')
        if i < len(gem_caps):
            marks_svg.append(f'<circle cx="{cx:.1f}" cy="{sy(gem_caps[i]):.1f}" r="4.8" fill="#808B90"/>')
        klbl = f"Pass {k}" if k <= 5 else "Pass 7*"
        marks_svg.append(f'<text x="{cx:.1f}" y="{y_bot + 28:.1f}" text-anchor="middle" fill="#0F0F0F" font-size="14.5" font-weight="600">{klbl}</text>')

    # Inkling 2 Mini at Pass 3 (index 2)
    ix, iy = sx(2), sy(im_cap)

    svg = f"""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}">
      <rect width="{W}" height="{H}" fill="#FFFFF0"/>
      {"".join(grid_svg)}

      <!-- Retrofitted Gemma 4 flat at 100% -->
      <polyline points="{pts_gem}" fill="none" stroke="#808B90" stroke-width="3.6" stroke-dasharray="8 5"/>
      <!-- Native Varloop self-correcting downward -->
      <polyline points="{pts_var}" fill="none" stroke="#2C3E50" stroke-width="4.2" stroke-linejoin="round"/>
      {"".join(marks_svg)}

      <!-- Flagship Inkling 2 Mini focal point at Pass 3 -->
      <line x1="{ix:.1f}" y1="{sy(var_caps[2]) + 8:.1f}" x2="{ix:.1f}" y2="{iy - 11:.1f}" stroke="#96590A" stroke-width="2" stroke-dasharray="4 3"/>
      <circle cx="{ix:.1f}" cy="{iy:.1f}" r="8.5" fill="#F39C12" stroke="#2C3E50" stroke-width="2.5"/>

      <!-- Direct labels -->
      <text x="{sx(3) + 14:.1f}" y="{sy(100.0) + 5:.1f}" fill="#808B90" font-size="14.5" font-weight="600">Retrofitted Gemma 4 (stuck at 100%)</text>
      <text x="{x1 + 14:.1f}" y="{sy(var_caps[-1]) + 5:.1f}" fill="#2C3E50" font-size="14.5" font-weight="600">Native Varloop (53.1%)</text>
      <text x="{ix + 16:.1f}" y="{iy + 5:.1f}" fill="#96590A" font-size="15.5" font-weight="600">Inkling 2 Mini: 35.9% at Pass 3</text>

      <line x1="{x0:.1f}" y1="{y_bot:.1f}" x2="{x1:.1f}" y2="{y_bot:.1f}" stroke="#2C3E50" stroke-width="1.8"/>
      <text x="{0.5*(x0 + x1):.1f}" y="{H - 8}" text-anchor="middle" fill="#1E3447" font-size="14" font-weight="500">Rate of copying a boss&rsquo;s wrong integer across silent passes (lower is better)</text>
    </svg>
    """

    html = wrap_clean_plate_html(
        title_plain="Native loops shake off a wrong number where retrofitted loops stay stuck.",
        headline_html='Native loops shake off a wrong number <span class="accent">where retrofitted loops stay stuck.</span>',
        svg_html=svg,
        source_text="Covert wrong-integer capture across silent passes · Retrofitted Gemma 4 vs. Thinking Machines Varloop & Inkling 2 Mini",
    )
    meta: dict[str, object] = {
        "slug": "inkling_04_self_correction_across_passes",
        "slot": "Inkling 04 · Standalone Trickle (Native Self-Correction vs. Retrofitted Loop)",
        "mode": "Standalone image post",
        "caption": (
            "Slip a boss's wrong integer into a math prompt and retrofitted Gemma 4 copies it 100% of the time across 4 silent passes.\n\n"
            "@thinkymachines' native loops self-correct with each pass (87.5% down to 53.1%; 35.9% on Inkling 2 Mini)."
        ),
        "replies": [
            (
                "Why the split: retrofitted Gemma 4 has no cross-pass QKV gate, so downstream attention heads copy the literal digit out of prompt KV memory.\n\n"
                "Native loops update cross-pass attention messages, and scrubbing 4 shortcut axes drops Inkling 2 Mini capture further to 32.8%."
            ),
            (
                f"X Article: {PART3_X_POST_URL}\n"
                "Full essay & preprint: https://shelltoho.me/pubs/nullbridge/"
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
        mode = str(m["mode"])
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
            lbl = f"Thread Reply {r_idx}"
            reply_btns.append(f'<button class="btn-secondary" type="button" onclick="copyPlain(\'{rep_js}\', this)">Copy {lbl}</button>')
            reply_blocks.append(f'<div class="reply-item"><div class="caption-label">{lbl} (chain onto previous)</div><pre class="reply-text">{html_mod.escape(rep)}</pre></div>')

        items.append(f"""
        <article class="tweet-mock">
          <div class="post-meta">
            <span class="post-num">{html_mod.escape(str(m["slot"]))}</span>
            <span class="word-count">{html_mod.escape(mode)} · {len(cap)} / 250 chars</span>
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
  <title>Inkling &amp; Looped Lens Trickle Deck · Warm Precision 2x Plates</title>
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
      gap: 12px; flex-wrap: wrap;
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
    <h1>Inkling &amp; Looped Lens trickle deck.</h1>
    <p class="sub">Four clean 4:3 Title + Graph plates (1920 &times; 1440 2x) highlighting <code>@thinkymachines</code> Inkling 2 Mini and <code>@ndif_team</code>&rsquo;s Looped Lens Challenge, starting with an immediate quote-post on your live Part 3 X Article.</p>
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
    parser = argparse.ArgumentParser(description="Build 2x Dawn State Inkling trickle social cards.")
    parser.add_argument("--track-a-json", type=pathlib.Path, default=pathlib.Path("results/phase4_native_inkling_nullbridge/track_a_results.json"))
    parser.add_argument("--track-b-json", type=pathlib.Path, default=pathlib.Path("results/phase3_gemma4_nullbridge/track_b_results.json"))
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("runs/trickle_cards_inkling"))
    args = parser.parse_args()

    ta = json.loads(args.track_a_json.read_text())
    tb = json.loads(args.track_b_json.read_text())

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    builders = [
        build_card1_looped_lens(ta),
        build_card2_pass2_moral_ignition(ta),
        build_card3_qkv_anchor_wire(ta),
        build_card4_self_correction(ta, tb),
    ]

    metas: list[dict[str, object]] = []
    for card_html, meta in builders:
        slug = str(meta["slug"])
        html_path = out_dir / f"{slug}.html"
        png_path = out_dir / f"{slug}.png"
        html_path.write_text(card_html)
        render_card_png(html_path, png_path)
        metas.append(meta)
        print(f"[OK] Rendered {png_path} ({len(str(meta['caption']))} char caption)")

    idx_html = build_feed_index(metas, out_dir)
    assert "\u2014" not in idx_html
    (out_dir / "index.html").write_text(idx_html)
    print(f"[OK] Wrote Inkling trickle feed preview -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
