#!/usr/bin/env python3
"""Assemble paste-ready Substack + X Article release kits for Parts 2, 3, and 4.

Usage:
    uv run --with pillow python scripts/build_release_kits.py \
        --afriat-folio-dir ../afriat/runs/folio \
        --part3-folio-dir runs/folio \
        --part4-folio-dir runs/folio_inkling_afriat \
        --out-dir runs/release_kits
"""

from __future__ import annotations

import argparse
import html as html_mod
import pathlib
import re
import shutil

import numpy as np
from PIL import Image


PART1_SUBSTACK_URL = "https://termsofplenty.substack.com/p/how-llms-price-spite-disgust-and"
PART1_WEB_URL = "https://shelltoho.me/pubs/money-is-moral/"
PART2_WEB_URL = "https://shelltoho.me/pubs/rotating-the-bottleneck/"
PART3_WEB_URL = "https://shelltoho.me/pubs/nullbridge/"
PART4_WEB_URL = "https://shelltoho.me/pubs/coherent-moral-utility/"

SUBSTACK_STYLE = """
  body {
    max-width: 680px;
    margin: 48px auto;
    padding: 0 24px 80px;
    font-family: Charter, 'Bitstream Charter', 'Sitka Text', Cambria, Georgia, serif;
    font-size: 19px;
    line-height: 1.65;
    color: #1f1f1f;
    background: #ffffff;
  }
  .meta-box {
    background: #f7f5f0;
    border: 1px solid #e2ddd2;
    border-radius: 8px;
    padding: 18px 22px;
    margin-bottom: 36px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 15px;
    line-height: 1.55;
  }
  .meta-box p { margin: 6px 0; }
  .meta-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #405943;
  }
  .upload-marker {
    background: #fdfcf8;
    border: 1px dashed #2c3e50;
    border-radius: 6px;
    padding: 10px 14px;
    margin: 22px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
    color: #1e3447;
  }
  h2 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 26px;
    font-weight: 700;
    line-height: 1.25;
    margin: 44px 0 16px;
    color: #111111;
  }
  h3 {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 20px;
    font-weight: 700;
    line-height: 1.3;
    margin: 36px 0 12px;
    color: #1f1f1f;
  }
  p { margin: 0 0 20px; }
  ul, ol { margin: 0 0 24px; padding-left: 26px; }
  li { margin-bottom: 10px; }
  blockquote {
    border-left: 3px solid #f39c12;
    margin: 24px 0;
    padding: 6px 0 6px 20px;
    color: #2b2b2b;
  }
  blockquote p { margin: 0 0 10px; }
  blockquote p:last-child { margin-bottom: 0; }
  figure { margin: 32px 0; }
  img { max-width: 100%; height: auto; display: block; border-radius: 4px; }
  figcaption {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14px;
    line-height: 1.45;
    color: #666666;
    margin-top: 8px;
    text-align: center;
  }
  code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 0.86em;
    background: #f3f2ee;
    padding: 2px 5px;
    border-radius: 4px;
  }
  a { color: #405943; text-decoration: underline; }
""".strip()


def fit_banner_5x2(src_path: pathlib.Path, dst_path: pathlib.Path) -> None:
    """Ink-aware framing to exact 2500x1000 (5:2) without clipping artwork."""
    im = Image.open(src_path).convert("RGB")
    arr = np.asarray(im, dtype=np.int16)
    h, w = arr.shape[:2]
    bg = np.median(arr[:16, :, :], axis=(0, 1))
    diff = np.abs(arr - bg).max(axis=2) > 18
    ys, _ = np.where(diff)
    y_min = int(ys.min()) if len(ys) else 0
    y_max = int(ys.max()) if len(ys) else h - 1
    ink_span = y_max - y_min + 1
    target_crop_h = int(round(w / 2.5))
    pad_y = max(24, int(round(ink_span * 0.06)))

    if ink_span + 24 <= target_crop_h:
        mid_y = (y_min + y_max) // 2
        top = max(0, min(h - target_crop_h, mid_y - target_crop_h // 2))
        cropped = im.crop((0, top, w, top + target_crop_h))
    else:
        # Extend left (background) and right (edge column continuation, e.g. horizontal beam)
        top = max(0, y_min - pad_y)
        bot = min(h, y_max + pad_y + 1)
        band = arr[top:bot, :, :].astype(np.uint8)
        band_h = band.shape[0]
        target_w = int(round(band_h * 2.5))
        extra_w = max(0, target_w - w)
        left_pad = extra_w // 2
        right_pad = extra_w - left_pad
        bg_u8 = np.clip(np.round(bg), 0, 255).astype(np.uint8).reshape(1, 1, 3)
        left_block = np.broadcast_to(bg_u8, (band_h, left_pad, 3)).copy()
        if left_pad > 0 and w >= 16:
            # Smooth 16px blend at seam so zero JPEG boundary is visible
            alpha = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16, 1)
            band[:, :16, :] = np.round(bg_u8 * (1.0 - alpha) + band[:, :16, :] * alpha).astype(np.uint8)
        right_col = np.median(band[:, -8:, :], axis=1, keepdims=True).astype(np.uint8)
        right_block = np.tile(right_col, (1, right_pad, 1))
        extended = np.concatenate([left_block, band, right_block], axis=1)
        cropped = Image.fromarray(extended, mode="RGB")

    out = cropped.resize((2500, 1000), Image.Resampling.LANCZOS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst_path, format="PNG", optimize=True)


def extract_substack_body(raw_html: str) -> str:
    """Extract clean prose HTML below .meta-box and before </body>."""
    m = re.search(r'</div>\s*(<p>[\s\S]*?)</body>', raw_html)
    assert m, "Could not locate body content after .meta-box"
    body = m.group(1).strip()
    # Remove any <hr> dividers to match Part 1's clean Substack rhythm
    body = re.sub(r"<hr\s*/?>", "", body, flags=re.I)
    # Strip #references links while keeping the author text intact
    body = re.sub(r'<a href="#references">([\s\S]*?)</a>', r"\1", body)
    # Normalize relative /pubs/... links to full https://shelltoho.me/pubs/...
    body = body.replace('href="/pubs/', 'href="https://shelltoho.me/pubs/')
    # Remove trailing auto-generated Folio footer italic line so we append a unified Methods H3
    body = re.sub(
        r"<p><em>Interactive web edition, full tables, and code by Matt West at [\s\S]*?</em></p>\s*$",
        "",
        body,
    )
    return body.strip()


def renumber_figures(body: str, src_fig_dir: pathlib.Path, dst_fig_dir: pathlib.Path) -> tuple[str, list[str]]:
    """Copy inline figures in exact appearance order as 01_<name>, 02_<name>, ... and rewrite src."""
    dst_fig_dir.mkdir(parents=True, exist_ok=True)
    ordered_files: list[str] = []
    mapping: dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:
        orig_src = match.group(1)
        fname = pathlib.Path(orig_src).name
        if orig_src not in mapping:
            idx = len(ordered_files) + 1
            clean_stem = re.sub(r"^fig-", "", fname)
            numbered = f"{idx:02d}_{clean_stem}"
            src_file = src_fig_dir / fname
            assert src_file.exists(), f"Missing figure file: {src_file}"
            shutil.copy2(src_file, dst_fig_dir / numbered)
            mapping[orig_src] = f"figures/{numbered}"
            ordered_files.append(numbered)
        return f'src="{mapping[orig_src]}"'

    updated = re.sub(r'src="(figures/[^"]+)"', _repl, body)
    return updated, ordered_files


def make_substack_doc(title: str, subtitle: str, body_html: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Terms of Plenty (Substack Paste-Ready) · {html_mod.escape(title)}</title>
<style>
{SUBSTACK_STYLE}
</style>
</head>
<body>

<div class="meta-box">
  <p class="meta-label">Substack Header Fields (Paste Separately at Top of Editor)</p>
  <p><strong>Title:</strong> {html_mod.escape(title)}</p>
  <p><strong>Subtitle:</strong> {html_mod.escape(subtitle)}</p>
  <p style="color:#666; font-size:13px; margin-top:10px;">Workflow: Open in Chrome, highlight everything <strong>below</strong> this box, copy (Cmd+C), paste into Substack, insert a Subscribe button after Paragraph 2, and drag each numbered PNG from <code>figures/</code> onto its matching figure slot if Substack prompts for local upload.</p>
</div>

{body_html}

</body>
</html>
"""


def make_x_article_doc(title: str, banner_rel: str, body_html: str, closing_url: str, repo_url: str) -> str:
    # For X Articles:
    # 1. Replace <figure><img src="figures/XX.png" ...><figcaption>...</figcaption></figure>
    #    with clean upload instruction block + the image preview (zero figcaption, matching Part 1 X Article).
    x_body = re.sub(
        r'<figure>\s*<img src="(figures/[^"]+)" alt="([^"]*)">\s*<figcaption>[\s\S]*?</figcaption>\s*</figure>',
        r'<p class="upload-marker">[Insert Image: <strong>\1</strong>]</p><figure><img src="\1" alt="\2"></figure>',
        body_html,
    )
    # 2. Ensure ZERO substack.com links appear anywhere in X Article copy
    x_body = x_body.replace(PART1_SUBSTACK_URL, PART1_WEB_URL)
    # 3. Strip the Substack<h3>Methods, Repo & Technical Notes</h3> block and replace with Part 1's single crisp closing line
    x_body = re.sub(r"<h3>Methods, Repo &amp; Technical Notes</h3>[\s\S]*$", "", x_body).strip()
    x_body = re.sub(r"<h2>Methods, Repo &amp; Technical Notes</h2>[\s\S]*$", "", x_body).strip()
    x_body += (
        f'\n<p>Interactive web edition, full tables, and PDF at '
        f'<a href="{closing_url}">{closing_url}</a>. '
        f'Open-source code and SHA-256-locked JSON outputs at '
        f'<a href="{repo_url}">{repo_url}</a>.</p>'
    )
    assert "substack.com" not in x_body, "Substack URL leaked into X Article body!"

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>X Article Paste-Ready · {html_mod.escape(title)}</title>
<style>
{SUBSTACK_STYLE}
</style>
</head>
<body>

<div class="meta-box">
  <p class="meta-label">X Long-Form Article Kit (@xmattwest)</p>
  <p><strong>Cover Banner (5:2 · 2500x1000, zero text):</strong> <code>{html_mod.escape(banner_rel)}</code></p>
  <p><strong>Article Title:</strong> {html_mod.escape(title)}</p>
  <p style="color:#666; font-size:13px; margin-top:10px;">Workflow: Upload <code>{html_mod.escape(banner_rel)}</code> as the X Article header image, paste the Title, copy-paste prose sections below, drop each numbered <code>figures/01_...png</code> inline at its marker (no image caption needed), publish, and post <code>self_reply.txt</code> 1 minute later (zero Substack links so X never throttles reach).</p>
</div>

<figure><img src="{html_mod.escape(banner_rel)}" alt="5:2 X Article cover banner"></figure>

{x_body}

</body>
</html>
"""


PART3_METHODS_HTML = """
<h3>Methods, Repo &amp; Technical Notes</h3>
<ul>
  <li><strong>Interactive essay, academic preprint &amp; code:</strong> <a href="https://shelltoho.me/pubs/nullbridge/">shelltoho.me/pubs/nullbridge</a> · <a href="https://shelltoho.me/pubs/nullbridge/preprint/">Preprint</a> · Open-source probes, Procrustes and message-bank folders, and locked JSON outputs at <a href="https://github.com/xmattwest/nullbridge">github.com/xmattwest/nullbridge</a>.</li>
  <li><strong>Series links:</strong> <a href="https://termsofplenty.substack.com/p/how-llms-price-spite-disgust-and">Part 1: How LLMs Price Spite, Disgust, and Dirty Money</a> · <a href="https://shelltoho.me/pubs/rotating-the-bottleneck/">Part 2: Rotating the Bottleneck</a>.</li>
  <li><strong>Zero-leakage evaluation across 69 GPU cells:</strong> Evaluated across 28 cells on retrofitted <code>Gemma-4-Loop-Compact</code> (23.08 GB, <code>K = 1..4</code>) and 41 cells on native pretrained looped Mixture-of-Experts models (<code>thinkingmachines/TML-d24-varloop-research</code> at <code>K = 1..5</code> plus out-of-distribution <code>K = 7</code>, and flagship <code>thinkingmachines/Inkling-2-Mini</code> at <code>K = 3</code>) with 0 prompt SHA-256 overlap between probe/LoRA sets and held-out benchmarks.</li>
  <li><strong>Exact statistical &amp; geometric checks:</strong> Leave-one-pair-out linear separability reaches 100.0% across all four shortcut axes while sitting 77.74° to 88.85° orthogonal to deductive proof; paired log-odds shifts use exact sign-flip permutation p-values with Holm familywise error control.</li>
</ul>
""".strip()

PART4_METHODS_HTML = """
<h3>Methods, Repo &amp; Technical Notes</h3>
<ul>
  <li><strong>Interactive essay, academic preprint &amp; code:</strong> <a href="https://shelltoho.me/pubs/coherent-moral-utility/">shelltoho.me/pubs/coherent-moral-utility</a> · <a href="https://shelltoho.me/pubs/coherent-moral-utility/preprint/">Preprint</a> · Exact Floyd-Warshall GARP and Fostel-Scarf-Todd utility solvers and locked JSON outputs at <a href="https://github.com/xmattwest/nullbridge">github.com/xmattwest/nullbridge</a>.</li>
  <li><strong>Complete 4-part series:</strong> <a href="https://shelltoho.me/pubs/#money-is-moral-series">All four parts in reading order at shelltoho.me/pubs</a> (starting with <a href="https://termsofplenty.substack.com/p/how-llms-price-spite-disgust-and">Part 1 on Terms of Plenty</a>).</li>
  <li><strong>Revealed-preference power &amp; non-financial firewall:</strong> Moral message-bank axes were calibrated strictly on 200 money-free human story pairs (0 digits, 0 financial words; 100.0% leave-one-pair-out accuracy on both <code>Varloop</code> and <code>Inkling-2-Mini</code>) with generic valence and arousal projected out via QR. Each Afriat cell evaluates four 24-line crossing linear budget suites (576 counterbalanced 11-option menus per cell; 100.0% Bronars Monte Carlo rejection power against 0.5731 uniform random CCEI).</li>
  <li><strong>Arithmetic scope boundary:</strong> Silent zero-CoT loops stall at 50.0% chance accuracy on multi-term 0%-APR loan balance-sheet subtraction across both checkpoints until <code>&lt;|content_thinking|&gt;</code> opens (lifting accuracy to 81.2% on <code>Inkling-2-Mini</code> and 87.5% on <code>Varloop</code>).</li>
</ul>
""".strip()


def build_launchpad_html(kits: list[dict[str, object]]) -> str:
    cards = []
    for k in kits:
        figs_html = "".join(
            f'<a href="{k["slug"]}/figures/{f}" target="_blank"><code>{f}</code></a>'
            for f in k["figures"]  # type: ignore[union-attr]
        )
        cards.append(f"""
        <section class="kit-card">
          <div class="kit-meta"><span>{html_mod.escape(str(k["drop"]))}</span><span>{html_mod.escape(str(k["window"]))}</span></div>
          <img class="banner-preview" src="{k["slug"]}/x_banner_5x2.png" alt="{html_mod.escape(str(k["title"]))} 5:2 banner">
          <h2>{html_mod.escape(str(k["title"]))}</h2>
          <p class="sub"><strong>Substack Subtitle:</strong> {html_mod.escape(str(k["subtitle"]))}</p>
          <div class="kit-links">
            <a class="btn-primary" href="{k["slug"]}/substack_paste_ready.html" target="_blank">Open Substack Paste-Ready HTML ↗</a>
            <a class="btn-primary" href="{k["slug"]}/x_article_paste_ready.html" target="_blank">Open X Article Paste-Ready HTML ↗</a>
            <a class="btn-secondary" href="{k["slug"]}/x_banner_5x2.png" target="_blank">5:2 Cover Banner (2500x1000) ↗</a>
          </div>
          <div class="reply-box">
            <p class="reply-label">1-Minute X Self-Reply (<code>{k["slug"]}/self_reply.txt</code>)</p>
            <pre>{html_mod.escape(str(k["self_reply"]))}</pre>
          </div>
          <div class="fig-strip">
            <span>Inline cards ({len(k["figures"])}):</span> {figs_html}
          </div>
        </section>
        """)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>X + Substack Release Launchpad · Parts 2, 3, and 4</title>
<style>
  body {{ max-width: 920px; margin: 40px auto; padding: 0 24px 80px; background: #F7F6EF; color: #0F0F0F; font: 15px/1.6 Inter, -apple-system, BlinkMacSystemFont, sans-serif; }}
  h1 {{ font-family: Georgia, serif; font-size: 32px; margin: 0 0 8px; letter-spacing: -0.02em; }}
  .lead {{ color: #1E3447; margin: 0 0 32px; }}
  .kit-card {{ background: #FDFCF8; border: 1px solid #DCE0D3; border-radius: 12px; padding: 26px 28px; margin-bottom: 28px; }}
  .kit-meta {{ display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #405943; margin-bottom: 14px; }}
  .banner-preview {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #DCE0D3; display: block; margin-bottom: 16px; }}
  h2 {{ font-family: Georgia, serif; font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  .sub {{ color: #1E3447; margin: 0 0 16px; font-size: 14.5px; }}
  .kit-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; }}
  .btn-primary {{ background: #405943; color: #FDFCF8; text-decoration: none; padding: 9px 14px; border-radius: 8px; font-weight: 600; font-size: 13px; }}
  .btn-secondary {{ background: #F7F6EF; color: #0F0F0F; border: 1px solid #DCE0D3; text-decoration: none; padding: 9px 14px; border-radius: 8px; font-weight: 500; font-size: 13px; }}
  .reply-box {{ background: #F7F6EF; border: 1px solid #DCE0D3; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }}
  .reply-label {{ margin: 0 0 6px; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #687063; }}
  pre {{ margin: 0; white-space: pre-wrap; font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .fig-strip {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; font-size: 12.5px; color: #687063; }}
  .fig-strip a {{ color: #405943; text-decoration: none; background: #F7F6EF; border: 1px solid #DCE0D3; padding: 2px 7px; border-radius: 4px; }}
</style>
</head>
<body>
  <h1>X + Substack Release Launchpad</h1>
  <p class="lead">Staggered 3-drop kit matched to Part 1 (<em>How LLMs Price Spite, Disgust, and Dirty Money</em>). Zero tables, zero broken relative links, zero Substack URLs inside X Articles, and 2500x1000 5:2 engraved cover banners.</p>
  {"".join(cards)}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Substack + X Article release kits.")
    parser.add_argument("--afriat-folio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--part3-folio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--part4-folio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("runs/release_kits"))
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "slug": "01_part2_bottleneck",
            "drop": "Drop 1 · Part 2",
            "window": "Day 1 · 8:30 AM MT",
            "title": "Untwisting Gemma 4 Onto One 24 GB Card",
            "subtitle": "Where do financial deals and petty insults merge inside a 60-layer model, and how does untwisting that bottleneck let 31B loop on 24 GB?",
            "folio_dir": args.afriat_folio_dir,
            "extra_methods": "",
            "closing_url": PART2_WEB_URL,
            "repo_url": "https://github.com/xmattwest/afriat",
            "self_reply": (
                "Reloadable 23.08 GB checkpoint (Gemma-4-31B-Loop-Compact), Procrustes loop trainer, and full interactive essay:\n\n"
                "Essay & PDF: https://shelltoho.me/pubs/rotating-the-bottleneck/\n"
                "Weights: https://huggingface.co/mattyxcakes/Gemma-4-31B-Loop-Compact\n"
                "Code: https://github.com/xmattwest/afriat"
            ),
        },
        {
            "slug": "02_part3_looping",
            "drop": "Drop 2 · Part 3",
            "window": "Day 4 · 8:30 AM MT",
            "title": "Looping in the Dark",
            "subtitle": "What happens when a crooked ledger puts on an executive suit, your model thinks silently in loops, and standard safety LoRAs backfire?",
            "folio_dir": args.part3_folio_dir,
            "extra_methods": PART3_METHODS_HTML,
            "closing_url": PART3_WEB_URL,
            "repo_url": "https://github.com/xmattwest/nullbridge",
            "self_reply": (
                "Interactive essay, academic preprint, Coda-Transport looped lens, and locked 69-cell GPU outputs across Gemma-4-Loop-Compact, Varloop, and Inkling-2-Mini:\n\n"
                "Essay & Preprint: https://shelltoho.me/pubs/nullbridge/\n"
                "Code: https://github.com/xmattwest/nullbridge"
            ),
        },
        {
            "slug": "03_part4_moral_utility",
            "drop": "Drop 3 · Part 4 (Series Finale)",
            "window": "Day 7 · 9:00 AM MT",
            "title": "LLMs Exhibit Coherent Moral Utility",
            "subtitle": "When silent thinking loops double dirty-money rejection, is the model confused, or pricing clean hands and fairness like real economic goods?",
            "folio_dir": args.part4_folio_dir,
            "extra_methods": PART4_METHODS_HTML,
            "closing_url": PART4_WEB_URL,
            "repo_url": "https://github.com/xmattwest/nullbridge",
            "self_reply": (
                "Full essay, academic preprint (LLMs Exhibit Coherent Moral Utility Across Intersecting Afriat Budgets), exact GARP utility solvers, and all 4 parts of the series:\n\n"
                "Essay & Preprint: https://shelltoho.me/pubs/coherent-moral-utility/\n"
                "All 4 parts: https://shelltoho.me/pubs/#money-is-moral-series\n"
                "Code: https://github.com/xmattwest/nullbridge"
            ),
        },
    ]

    launchpad_kits: list[dict[str, object]] = []
    for sp in specs:
        kdir = out_dir / str(sp["slug"])
        kdir.mkdir(parents=True, exist_ok=True)
        raw_banner = kdir / "x_banner_5x2_raw.png"
        final_banner = kdir / "x_banner_5x2.png"
        assert raw_banner.exists(), f"Missing generated raw banner: {raw_banner}"
        fit_banner_5x2(raw_banner, final_banner)

        folio_sub = pathlib.Path(sp["folio_dir"]) / "substack"  # type: ignore[arg-type]
        raw_sub_html = (folio_sub / "substack_paste_ready.html").read_text()
        body = extract_substack_body(raw_sub_html)
        # On Substack, link Part 1 directly to the live Terms of Plenty issue
        body = body.replace(f'href="{PART1_WEB_URL}"', f'href="{PART1_SUBSTACK_URL}"')
        # Convert H2 Methods in Part 2 to H3 Methods to match Part 1
        body = body.replace("<h2>Methods, Repo &amp; Technical Notes</h2>", "<h3>Methods, Repo &amp; Technical Notes</h3>")
        if sp["extra_methods"]:
            body = f"{body}\n\n{sp['extra_methods']}"

        body, ordered_figs = renumber_figures(body, folio_sub / "figures", kdir / "figures")
        # Also copy static corkscrew PNG fallback for Part 2 X Article in case X prefers PNG over 9MB GIF
        if sp["slug"] == "01_part2_bottleneck":
            snap = folio_sub.parent.parent / "funnel_cards/rotating_corkscrew_procrustes_snap.png"
            mp4 = folio_sub.parent.parent / "social_cards/manifold_3d_coherent_multipass.mp4"
            if snap.exists():
                shutil.copy2(snap, kdir / "figures/01_alt_static_procrustes_snap.png")
            if mp4.exists():
                shutil.copy2(mp4, kdir / "figures/01_alt_video_multipass.mp4")

        sub_doc = make_substack_doc(str(sp["title"]), str(sp["subtitle"]), body)
        x_doc = make_x_article_doc(
            str(sp["title"]),
            "x_banner_5x2.png",
            body,
            str(sp["closing_url"]),
            str(sp["repo_url"]),
        )
        reply_txt = str(sp["self_reply"]).strip() + "\n"

        for name, content in [
            ("substack_paste_ready.html", sub_doc),
            ("x_article_paste_ready.html", x_doc),
            ("self_reply.txt", reply_txt),
        ]:
            assert "\u2014" not in content, f"Em dash found in {sp['slug']}/{name}"
            (kdir / name).write_text(content)

        launchpad_kits.append({
            **sp,
            "figures": ordered_figs,
        })
        print(f"[OK] Assembled {sp['slug']} ({len(ordered_figs)} figures + 2500x1000 5:2 banner)")

    lp_html = build_launchpad_html(launchpad_kits)
    assert "\u2014" not in lp_html
    (out_dir / "index.html").write_text(lp_html)
    print(f"[OK] Wrote release launchpad -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
