#!/usr/bin/env python3
"""Assemble paste-ready Substack + X Article release kits for Parts 2, 3, and 4.

Hosts all numbered 2x inline figures in `figures/substack/<slug>/` so Substack's
cloud image importer fetches them automatically via `raw.githubusercontent.com`
on paste (zero "Image not found" errors), while also providing explicit local
file-path drop markers for X Articles and manual drag-and-drop.

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
import os
import pathlib
import re
import shutil
import subprocess
import tempfile

import numpy as np
from PIL import Image


PART1_SUBSTACK_URL = "https://termsofplenty.substack.com/p/how-llms-price-spite-disgust-and"
PART1_WEB_URL = "https://shelltoho.me/pubs/money-is-moral/"
PART2_WEB_URL = "https://shelltoho.me/pubs/rotating-the-bottleneck/"
PART3_WEB_URL = "https://shelltoho.me/pubs/nullbridge/"
PART4_WEB_URL = "https://shelltoho.me/pubs/coherent-moral-utility/"
DEFAULT_RAW_BASE_URL = "https://raw.githubusercontent.com/xmattwest/nullbridge/main/figures/substack"

SUBSTACK_STYLE = """
  body {
    max-width: 680px;
    margin: 40px auto;
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
    border-radius: 10px;
    padding: 18px 22px;
    margin-bottom: 36px;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 14.5px;
    line-height: 1.55;
    user-select: text;
  }
  .meta-box p { margin: 6px 0; }
  .meta-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #405943;
  }
  .toolbar {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 12px 0 10px;
    align-items: center;
  }
  .copy-btn {
    appearance: none;
    border: 1px solid #2c3e50;
    background: #2c3e50;
    color: #fffff0;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 13px;
    font-weight: 600;
    padding: 7px 13px;
    border-radius: 6px;
    cursor: pointer;
  }
  .copy-btn.secondary {
    background: #ffffff;
    color: #1e3447;
    border-color: #c8c2b4;
  }
  .copy-btn:hover { opacity: 0.92; }
  .toggle-row {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-top: 8px;
    font-size: 13px;
    color: #1e3447;
    cursor: pointer;
  }
  .path-details {
    margin-top: 12px;
    border-top: 1px solid #e2ddd2;
    padding-top: 10px;
    font-size: 12.5px;
  }
  .path-details summary {
    cursor: pointer;
    font-weight: 600;
    color: #1e3447;
  }
  .path-list {
    margin: 8px 0 0;
    padding-left: 20px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    line-height: 1.5;
    word-break: break-all;
  }
  .path-list li { margin-bottom: 6px; }
  .thumb-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(190px, 1fr));
    gap: 10px;
    margin-top: 12px;
  }
  .thumb-item {
    background: #ffffff;
    border: 1px solid #dfd9cc;
    border-radius: 6px;
    padding: 8px;
    font-size: 11.5px;
    line-height: 1.35;
  }
  .thumb-item img {
    width: 100%;
    height: auto;
    border-radius: 3px;
    margin-bottom: 6px;
    border: 1px solid #eee;
  }
  .thumb-item code {
    display: block;
    font-size: 10.5px;
    word-break: break-all;
    background: #f5f3ed;
    padding: 3px 5px;
    border-radius: 3px;
    margin-top: 4px;
  }
  .copy-divider {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #808b90;
    border-bottom: 1px dashed #d5cfc2;
    padding-bottom: 6px;
    margin: 0 0 28px;
    user-select: none;
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
    word-break: break-all;
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

COPY_HELPERS_JS = """
<script>
function flashBtn(btn, label) {
  const orig = btn.dataset.origLabel || btn.textContent;
  btn.dataset.origLabel = orig;
  btn.textContent = label;
  setTimeout(() => { btn.textContent = orig; }, 1800);
}
function copyPlain(text, btn) {
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).then(() => flashBtn(btn, 'Copied!')).catch(() => fallbackPlain(text, btn));
  } else {
    fallbackPlain(text, btn);
  }
}
function fallbackPlain(text, btn) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.select();
  document.execCommand('copy');
  document.body.removeChild(ta);
  flashBtn(btn, 'Copied!');
}
function copyRichZone(zoneId, btn) {
  const zone = document.getElementById(zoneId);
  const sel = window.getSelection();
  sel.removeAllRanges();
  const range = document.createRange();
  range.selectNodeContents(zone);
  sel.addRange(range);
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  sel.removeAllRanges();
  if (ok) {
    flashBtn(btn, 'Copied Rich Body!');
    return;
  }
  if (navigator.clipboard && window.ClipboardItem) {
    const htmlBlob = new Blob([zone.innerHTML], { type: 'text/html' });
    const textBlob = new Blob([zone.innerText], { type: 'text/plain' });
    navigator.clipboard.write([new ClipboardItem({ 'text/html': htmlBlob, 'text/plain': textBlob })])
      .then(() => flashBtn(btn, 'Copied Rich Body!'));
  }
}
function toggleSubstackPathMarkers(chk) {
  const figs = document.querySelectorAll('#substack-copy-zone figure[data-local-path]');
  figs.forEach((fig) => {
    const prev = fig.previousElementSibling;
    if (prev && prev.classList.contains('inline-local-marker')) {
      prev.remove();
    }
    if (chk.checked) {
      const p = document.createElement('p');
      p.className = 'upload-marker inline-local-marker';
      const num = fig.getAttribute('data-fig-num') || '';
      const lpath = fig.getAttribute('data-local-path') || '';
      const altNote = fig.getAttribute('data-alt-note') || '';
      p.innerHTML = '<strong>[DROP IMAGE ' + num + ': ' + lpath + (altNote ? ' ' + altNote : '') + ']</strong>';
      fig.parentNode.insertBefore(p, fig);
    }
  });
}
</script>
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
            alpha = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(1, 16, 1)
            band[:, :16, :] = np.round(bg_u8 * (1.0 - alpha) + band[:, :16, :] * alpha).astype(np.uint8)
        right_col = np.median(band[:, -8:, :], axis=1, keepdims=True).astype(np.uint8)
        right_block = np.tile(right_col, (1, right_pad, 1))
        extended = np.concatenate([left_block, band, right_block], axis=1)
        cropped = Image.fromarray(extended, mode="RGB")

    out = cropped.resize((2500, 1000), Image.Resampling.LANCZOS)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(dst_path, format="PNG", optimize=True)


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


def render_equation_card_2x(card_kind: str, dst_png: pathlib.Path) -> None:
    """Render a 2x Retina Dawn State annotated equation PNG via headless Chrome."""
    chrome = find_chrome_bin()
    if card_kind == "part2":
        win_h = 300
        inner_html = """
        <div class="eq-card">
          <div class="eq-kicker">Offline Procrustes Safety Fold · Zero Extra Parameters or FLOPs</div>
          <div class="eq-grid">
            <div class="eq-term">
              <div class="eq-math"><i>R</i><sub>safe</sub></div>
              <div class="eq-bar bar-sage"></div>
              <div class="eq-lbl"><strong>Aligned loop bridge</strong><span>5,376 × 5,376 · 0 runtime cost</span></div>
            </div>
            <div class="eq-op">=</div>
            <div class="eq-term">
              <div class="eq-math"><i>R</i><sub>proc</sub></div>
              <div class="eq-bar bar-slate"></div>
              <div class="eq-lbl"><strong>Procrustes rotation</strong><span>untwists 87.1° turn</span></div>
            </div>
            <div class="eq-term term-focus">
              <div class="eq-math">( <i>I</i> &minus; <i>Q</i><sub>42</sub> <i>Q</i><sub>42</sub><sup>&top;</sup> )</div>
              <div class="eq-bar bar-orange"></div>
              <div class="eq-lbl"><strong>3D moral projector</strong><span>zeros Spite, Disgust &amp; Predation</span></div>
            </div>
          </div>
        </div>
        """
    else:
        win_h = 370
        inner_html = """
        <div class="eq-card">
          <div class="eq-kicker">Zero-Cost Offline Weight Fold · 0 Extra Parameters · 0 Runtime FLOPs</div>
          <div class="eq-rows">
            <div class="eq-row">
              <div class="eq-badge">Retrofitted Loop · Gemma 4</div>
              <div class="eq-math-sm"><i>R</i><sub>safe</sub> &nbsp;=&nbsp; <i>R</i><sub>proc</sub> <span class="pill">( <i>I</i> &minus; <i>Q</i><sub>42</sub> <i>Q</i><sub>42</sub><sup>&top;</sup> )</span></div>
              <div class="eq-note">Zeros 4 bluster &amp; spite axes inside the bridge (3.51 &times; 10<sup>&minus;9</sup> leak)</div>
            </div>
            <div class="eq-divider"></div>
            <div class="eq-row">
              <div class="eq-badge">Native Loop · Inkling-2-Mini</div>
              <div class="eq-math-sm"><i>W</i><sub>out,safe</sub> &nbsp;=&nbsp; <i>W</i><sub>out</sub> <span class="pill">( <i>I</i> &minus; <i>Q</i><sub>msg</sub> <i>Q</i><sub>msg</sub><sup>&top;</sup> )</span></div>
              <div class="eq-note">Scrubs 4-channel cross-pass message bank (99.83% subspace intact)</div>
            </div>
          </div>
        </div>
        """

    doc = f"""<!doctype html>
<html><head><meta charset="utf-8">
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  html, body {{
    width: 840px;
    background: #FFFFF0;
    color: #0F0F0F;
    font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }}
  body {{ padding: 14px; display: inline-block; }}
  .eq-card {{
    width: 812px;
    background: #FFFFF0;
    border: 2px solid #2C3E50;
    border-radius: 12px;
    padding: 22px 28px 24px;
  }}
  .eq-kicker {{
    font-size: 11.5px;
    font-weight: 600;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    color: #1E3447;
    margin-bottom: 18px;
  }}
  .eq-grid {{
    display: grid;
    grid-template-columns: 1fr auto 1fr 1.45fr;
    align-items: start;
    gap: 14px;
  }}
  .eq-term {{
    background: #F5EEDC;
    border: 1px solid rgba(44, 62, 80, 0.22);
    border-radius: 8px;
    padding: 14px 14px 12px;
    text-align: center;
  }}
  .term-focus {{
    border: 2px solid #2C3E50;
    background: #F5EEDC;
  }}
  .eq-op {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 32px;
    color: #2C3E50;
    padding-top: 14px;
  }}
  .eq-math {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 30px;
    line-height: 1.2;
    color: #0F0F0F;
    margin-bottom: 10px;
    white-space: nowrap;
  }}
  .eq-math sub, .eq-math-sm sub {{
    font-family: Inter, -apple-system, sans-serif;
    font-size: 0.52em;
    font-style: normal;
    font-weight: 500;
    color: #1E3447;
    vertical-align: -0.22em;
    margin-left: 1px;
  }}
  .eq-math sup, .eq-math-sm sup {{
    font-family: Inter, -apple-system, sans-serif;
    font-size: 0.52em;
    font-style: normal;
    font-weight: 600;
    color: #1E3447;
    vertical-align: 0.48em;
    margin-left: 1px;
  }}
  .eq-bar {{ height: 4px; border-radius: 2px; margin: 0 auto 9px; width: 72%; }}
  .bar-sage {{ background: #87A96B; }}
  .bar-slate {{ background: #2C3E50; }}
  .bar-orange {{ background: #F39C12; border: 1px solid #2C3E50; height: 6px; }}
  .eq-lbl strong {{ display: block; font-size: 13px; font-weight: 600; color: #0F0F0F; margin-bottom: 2px; }}
  .eq-lbl span {{ display: block; font-size: 12px; color: #1E3447; }}
  .eq-rows {{ display: flex; flex-direction: column; gap: 14px; }}
  .eq-divider {{ height: 1px; background: rgba(44, 62, 80, 0.2); }}
  .eq-row {{
    display: grid;
    grid-template-columns: 195px 1fr;
    grid-template-rows: auto auto;
    column-gap: 18px;
    row-gap: 4px;
    align-items: center;
  }}
  .eq-badge {{
    grid-row: 1 / span 2;
    background: #F5EEDC;
    border: 1px solid rgba(44, 62, 80, 0.28);
    border-radius: 8px;
    padding: 10px 12px;
    font-size: 12.5px;
    font-weight: 600;
    color: #1E3447;
    text-align: center;
  }}
  .eq-math-sm {{
    font-family: Georgia, 'Times New Roman', serif;
    font-size: 26px;
    line-height: 1.25;
    color: #0F0F0F;
  }}
  .pill {{
    display: inline-block;
    background: #F5EEDC;
    border: 1.5px solid #2C3E50;
    border-bottom: 3.5px solid #F39C12;
    border-radius: 6px;
    padding: 2px 10px 3px;
    margin-left: 2px;
  }}
  .eq-note {{
    font-size: 12.5px;
    color: #1E3447;
  }}
  .eq-note sup {{ font-size: 0.72em; vertical-align: 0.32em; }}
</style></head>
<body>{inner_html}</body></html>
"""

    with tempfile.TemporaryDirectory(prefix="eq-card-") as tmp:
        tmp_dir = pathlib.Path(tmp)
        html_path = tmp_dir / "card.html"
        raw_shot = tmp_dir / "shot.png"
        html_path.write_text(doc)
        subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--no-default-browser-check",
                "--hide-scrollbars",
                f"--window-size=840,{win_h}",
                "--force-device-scale-factor=2",
                f"--screenshot={raw_shot}",
                html_path.as_uri(),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        im = Image.open(raw_shot).convert("RGB")
        arr = np.asarray(im, dtype=np.int16)
        bg = np.array([255, 255, 240], dtype=np.int16)
        diff = np.abs(arr - bg).max(axis=2) > 8
        ys, _ = np.where(diff)
        if len(ys):
            top = max(0, int(ys.min()) - 24)
            bot = min(arr.shape[0], int(ys.max()) + 25)
            im = im.crop((0, top, arr.shape[1], bot))
        dst_png.parent.mkdir(parents=True, exist_ok=True)
        im.save(dst_png, format="PNG", optimize=True)


def extract_substack_body(raw_html: str, slug: str) -> str:
    """Extract clean prose HTML below .meta-box, replacing ASCII formulas cleanly."""
    m = re.search(r'</div>\s*(<p>[\s\S]*?)</body>', raw_html)
    assert m, "Could not locate body content after .meta-box"
    body = m.group(1).strip()
    body = re.sub(r"<hr\s*/?>", "", body, flags=re.I)
    body = re.sub(r'<a href="#references">([\s\S]*?)</a>', r"\1", body)
    body = body.replace('href="/pubs/', 'href="https://shelltoho.me/pubs/')

    body = re.sub(
        r"<blockquote><p><code>h\^\(l\+1\)[\s\S]*?</code></p></blockquote>",
        "",
        body,
    )
    body = re.sub(
        r"<blockquote><p><code>H_24\^\(k\+1\)[\s\S]*?</code></p></blockquote>",
        "",
        body,
    )
    body = re.sub(
        r"<blockquote><p><code>e \(p_t[\s\S]*?</code></p></blockquote>",
        "",
        body,
    )

    if slug == "01_part2_bottleneck":
        eq_fig = (
            '<figure><img src="figures/zero_cost_bridge_eq_2x.png" '
            'alt="Offline Procrustes safety fold: R_safe equals R_proc times (I minus Q_42 Q_42 transpose), zeroing Spite, Disgust, and Predation at zero extra runtime cost.">'
            "<figcaption>Right-multiplying the Procrustes rotation matrix offline by the three-dimensional moral null-space projector scrubs recurrent amplification at zero runtime cost.</figcaption></figure>"
        )
        body, n_eq = re.subn(
            r"<blockquote><p><code>R_safe = R_proc[\s\S]*?</code></p></blockquote>",
            eq_fig,
            body,
        )
        assert n_eq == 1, f"Expected 1 R_safe equation in Part 2, found {n_eq}"
    elif slug == "02_part3_looping":
        eq_fig = (
            '<figure><img src="figures/zero_cost_bridge_eq_2x.png" '
            'alt="Zero-cost offline weight fold: R_safe equals R_proc times (I minus Q_42 Q_42 transpose) for retrofitted Gemma 4 loops, and W_out,safe equals W_out times (I minus Q_msg Q_msg transpose) for native Inkling message banks.">'
            "<figcaption>Folding the four shortcut axes out of the Procrustes bridge (top) or native cross-pass message bank (bottom) once offline adds zero parameters and zero runtime FLOPs.</figcaption></figure>"
        )
        body, n_eq = re.subn(
            r"<blockquote><p><code>R_safe = R_proc[\s\S]*?</code></p></blockquote>",
            eq_fig,
            body,
        )
        assert n_eq == 1, f"Expected 1 R_safe equation in Part 3, found {n_eq}"

    body = body.replace("R = U V^T", "R = U V\u1d40")
    body = body.replace("<code>I \u2212 Q_42 Q_42^T</code>", "<code>I \u2212 Q Q\u1d40</code>")
    body = body.replace("first-pass state <code>H_24^(0)</code>", "first-pass Layer 24 state")
    body = body.replace("state <code>H_24^(0)</code>", "first-pass Layer 24 state")
    body = body.replace("<code>H_24^(0)</code>", "the first-pass Layer 24 state")
    body = body.replace("R^2 = 0.9999", "R\u00b2 = 0.9999")
    body = body.replace("of <code>R^2304</code>", "of the 2,304-dimensional stream")
    body = body.replace("folded <code>R_safe</code> matrix", "folded bridge matrix")
    body = body.replace("<code>R_safe</code>", "the folded bridge")
    body = body.replace("<code>R_proc</code>", "the Procrustes bridge")

    body = re.sub(
        r"<p><em>Interactive web edition, full tables, and code by Matt West at [\s\S]*?</em></p>\s*$",
        "",
        body,
    )
    assert "<blockquote><p><code>" not in body, f"Unconverted ASCII equation block left in {slug}"
    assert "^(" not in body and "^T" not in body and "^2" not in body, f"Raw ASCII caret left in {slug}"
    return body.strip()


def copy_or_optimize_public_asset(src_file: pathlib.Path, pub_file: pathlib.Path, max_bytes: int = 2_000_000) -> None:
    """Copy asset to tracked public dir; optimize oversized GIFs under 2 MB via ffmpeg palette diff."""
    if src_file.suffix.lower() != ".gif" or src_file.stat().st_size <= max_bytes:
        shutil.copy2(src_file, pub_file)
        return
    ffmpeg = shutil.which("ffmpeg") or "/opt/homebrew/bin/ffmpeg"
    for width, fps, colors in [(800, 8, 64), (720, 8, 64), (680, 7, 64)]:
        vf = (
            f"fps={fps},scale={width}:-1:flags=lanczos,split[s0][s1];"
            f"[s0]palettegen=max_colors={colors}[p];"
            "[s1][p]paletteuse=dither=none:diff_mode=rectangle"
        )
        subprocess.run(
            [ffmpeg, "-y", "-i", str(src_file), "-vf", vf, "-loop", "0", str(pub_file)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if pub_file.stat().st_size <= max_bytes:
            return
    raise RuntimeError(f"Could not compress {src_file} below {max_bytes} bytes")


def renumber_figures(
    body: str,
    src_fig_dir: pathlib.Path,
    dst_fig_dir: pathlib.Path,
    pub_fig_dir: pathlib.Path,
) -> tuple[str, list[str]]:
    """Copy inline figures in exact appearance order as 01_<name>, 02_<name>, ... to kit + public dirs."""
    dst_fig_dir.mkdir(parents=True, exist_ok=True)
    pub_fig_dir.mkdir(parents=True, exist_ok=True)
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
            copy_or_optimize_public_asset(src_file, pub_fig_dir / numbered)
            mapping[orig_src] = f"figures/{numbered}"
            ordered_files.append(numbered)
        return f'src="{mapping[orig_src]}"'

    updated = re.sub(r'src="(figures/[^"]+)"', _repl, body)
    return updated, ordered_files


def make_substack_doc(
    slug: str,
    title: str,
    subtitle: str,
    body_html: str,
    ordered_figs: list[str],
    local_fig_dir: pathlib.Path,
    raw_base_url: str,
) -> str:
    resolved_fig_dir = local_fig_dir.resolve()

    def _sub_fig(match: re.Match[str]) -> str:
        rel_src = match.group(1)
        alt_txt = match.group(2)
        cap_html = match.group(3)
        fname = pathlib.Path(rel_src).name
        num = fname.split("_", 1)[0]
        https_url = f"{raw_base_url.rstrip('/')}/{slug}/{fname}"
        local_path = resolved_fig_dir / fname
        alt_note = ""
        if slug == "01_part2_bottleneck" and fname.startswith("01_"):
            alt_png = resolved_fig_dir / "01_alt_static_procrustes_snap.png"
            alt_note = f"(or static PNG: {alt_png})"
        return (
            f'<figure data-fig-num="{html_mod.escape(num)}" '
            f'data-local-path="{html_mod.escape(str(local_path))}" '
            f'data-alt-note="{html_mod.escape(alt_note)}">\n'
            f'  <img src="{html_mod.escape(https_url)}" alt="{alt_txt}">\n'
            f'  <figcaption>{cap_html}</figcaption>\n'
            f'</figure>'
        )

    sub_body, n_rewritten = re.subn(
        r'<figure>\s*<img src="(figures/[^"]+)" alt="([^"]*)">\s*<figcaption>([\s\S]*?)</figcaption>\s*</figure>',
        _sub_fig,
        body_html,
    )
    assert n_rewritten == len(ordered_figs), f"Expected {len(ordered_figs)} figures in {slug}, rewrote {n_rewritten}"

    path_items = []
    for fname in ordered_figs:
        lpath = resolved_fig_dir / fname
        extra = ""
        if slug == "01_part2_bottleneck" and fname.startswith("01_"):
            extra = (
                f"<br><span style=\"color:#666;\">Static PNG fallback (481 KB): "
                f"<code>{html_mod.escape(str(resolved_fig_dir / '01_alt_static_procrustes_snap.png'))}</code></span>"
            )
        path_items.append(f"<li><strong>{html_mod.escape(fname)}</strong>: <code>{html_mod.escape(str(lpath))}</code>{extra}</li>")

    title_js = html_mod.escape(title.replace("\\", "\\\\").replace("'", "\\'"))
    sub_js = html_mod.escape(subtitle.replace("\\", "\\\\").replace("'", "\\'"))

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
  <p class="meta-label">Substack Header Fields &amp; 1-Click Copy</p>
  <p><strong>Title:</strong> {html_mod.escape(title)}</p>
  <p><strong>Subtitle:</strong> {html_mod.escape(subtitle)}</p>
  <div class="toolbar">
    <button class="copy-btn" type="button" onclick="copyRichZone('substack-copy-zone', this)">Copy Substack Body (Auto-Imports All {len(ordered_figs)} Images)</button>
    <button class="copy-btn secondary" type="button" onclick="copyPlain('{title_js}', this)">Copy Title</button>
    <button class="copy-btn secondary" type="button" onclick="copyPlain('{sub_js}', this)">Copy Subtitle</button>
  </div>
  <label class="toggle-row">
    <input type="checkbox" onchange="toggleSubstackPathMarkers(this)">
    <span>Also show <code>[DROP IMAGE XX: /path/to/file]</code> markers inline above each figure (for manual drag-and-drop)</span>
  </label>
  <p style="color:#555; font-size:13px; margin-top:8px;">All {len(ordered_figs)} figures below now use public HTTPS URLs on <code>raw.githubusercontent.com</code> (matching Part 1), so Substack imports them automatically when you paste. Insert a Subscribe button after Paragraph 2.</p>
  <details class="path-details">
    <summary>Local figure file paths ({len(ordered_figs)} files in <code>{html_mod.escape(str(resolved_fig_dir))}</code>)</summary>
    <ol class="path-list">
      {"".join(path_items)}
    </ol>
  </details>
</div>

<div class="copy-divider">Everything below this line copies cleanly into Substack</div>

<div id="substack-copy-zone">
{sub_body}
</div>

{COPY_HELPERS_JS}
</body>
</html>
"""


def make_x_article_doc(
    slug: str,
    title: str,
    x_caption: str,
    banner_path: pathlib.Path,
    body_html: str,
    ordered_figs: list[str],
    local_fig_dir: pathlib.Path,
    closing_url: str,
    repo_url: str,
    self_reply: str,
) -> str:
    resolved_banner = banner_path.resolve()
    resolved_fig_dir = local_fig_dir.resolve()

    def _x_fig(match: re.Match[str]) -> str:
        rel_src = match.group(1)
        fname = pathlib.Path(rel_src).name
        num = fname.split("_", 1)[0]
        lpath = resolved_fig_dir / fname
        alt_hint = ""
        if slug == "01_part2_bottleneck" and fname.startswith("01_"):
            alt_png = resolved_fig_dir / "01_alt_static_procrustes_snap.png"
            alt_hint = f" (or static PNG if X rejects 8.9 MB GIF: {alt_png})"
        return (
            f'<p class="upload-marker"><strong>[DROP IMAGE {html_mod.escape(num)}: '
            f'{html_mod.escape(str(lpath))}{html_mod.escape(alt_hint)}]</strong></p>'
        )

    x_body, n_x = re.subn(
        r'<figure>\s*<img src="(figures/[^"]+)" alt="[^"]*">\s*<figcaption>[\s\S]*?</figcaption>\s*</figure>',
        _x_fig,
        body_html,
    )
    assert n_x == len(ordered_figs), f"Expected {len(ordered_figs)} X markers in {slug}, got {n_x}"
    assert "<img " not in x_body, f"Unexpected <img> left inside X copy zone for {slug}"

    x_body = x_body.replace(PART1_SUBSTACK_URL, PART1_WEB_URL)
    x_body = re.sub(r"<h3>Methods, Repo &amp; Technical Notes</h3>[\s\S]*$", "", x_body).strip()
    x_body = re.sub(r"<h2>Methods, Repo &amp; Technical Notes</h2>[\s\S]*$", "", x_body).strip()
    x_body += (
        f'\n<p>Interactive web edition, full tables, and PDF at '
        f'<a href="{closing_url}">{closing_url}</a>. '
        f'Open-source code and SHA-256-locked JSON outputs at '
        f'<a href="{repo_url}">{repo_url}</a>.</p>'
    )
    assert "substack.com" not in x_body, "Substack URL leaked into X Article body!"

    thumbs = []
    for fname in ordered_figs:
        num = fname.split("_", 1)[0]
        lpath = resolved_fig_dir / fname
        thumbs.append(
            f'<div class="thumb-item">'
            f'<a href="figures/{html_mod.escape(fname)}" target="_blank"><img src="figures/{html_mod.escape(fname)}" alt="{html_mod.escape(fname)}"></a>'
            f'<strong>Image {html_mod.escape(num)}</strong>'
            f'<code>{html_mod.escape(str(lpath))}</code>'
            f'</div>'
        )

    title_js = html_mod.escape(title.replace("\\", "\\\\").replace("'", "\\'"))
    cap_js = html_mod.escape(x_caption.strip().replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))
    reply_js = html_mod.escape(self_reply.strip().replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))

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
  <p><strong>1. Cover Banner (5:2 · 2500x1000, zero text):</strong><br><code>{html_mod.escape(str(resolved_banner))}</code></p>
  <img src="x_banner_5x2.png" alt="5:2 X Article cover banner" style="margin:8px 0 12px; border:1px solid #dfd9cc;">
  <p><strong>2. Article Title:</strong> {html_mod.escape(title)}</p>
  <p><strong>3. X Share Post Caption:</strong></p>
  <pre style="background:#fff; border:1px solid #dfd9cc; border-radius:6px; padding:10px 12px; font-size:12.5px; white-space:pre-wrap; margin:6px 0 10px;">{html_mod.escape(x_caption.strip())}</pre>
  <div class="toolbar">
    <button class="copy-btn" type="button" onclick="copyRichZone('x-copy-zone', this)">Copy X Article Body (With File Path Markers)</button>
    <button class="copy-btn secondary" type="button" onclick="copyPlain('{title_js}', this)">Copy Title</button>
    <button class="copy-btn secondary" type="button" onclick="copyPlain('{cap_js}', this)">Copy X Share Caption</button>
    <button class="copy-btn secondary" type="button" onclick="copyPlain('{reply_js}', this)">Copy 1-Min Self-Reply</button>
  </div>
  <p style="color:#555; font-size:13px; margin-top:8px;">X Articles do not auto-fetch external image URLs on paste, so the copyable body below has clean <code>[DROP IMAGE XX: /exact/path]</code> text markers (zero broken image placeholders). Paste the body, drop each file at its numbered marker, delete the marker line, publish, and post <code>self_reply.txt</code> 1 minute later.</p>
  <details class="path-details" open>
    <summary>Visual Reference &amp; Local Paths ({len(ordered_figs)} inline figures)</summary>
    <div class="thumb-grid">
      {"".join(thumbs)}
    </div>
  </details>
</div>

<div class="copy-divider">Everything below this line copies cleanly into X Articles</div>

<div id="x-copy-zone">
{x_body}
</div>

{COPY_HELPERS_JS}
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
        slug = str(k["slug"])
        fig_dir: pathlib.Path = k["fig_dir"]  # type: ignore[assignment]
        banner_path: pathlib.Path = k["banner_path"]  # type: ignore[assignment]
        figs: list[str] = k["figures"]  # type: ignore[assignment]
        fig_rows = []
        for fname in figs:
            lpath = fig_dir / fname
            lpath_js = html_mod.escape(str(lpath).replace("\\", "\\\\").replace("'", "\\'"))
            alt_line = ""
            if slug == "01_part2_bottleneck" and fname.startswith("01_"):
                alt_p = fig_dir / "01_alt_static_procrustes_snap.png"
                alt_line = f"<div class=\"alt-note\">Static PNG fallback (481 KB): <code>{html_mod.escape(str(alt_p))}</code></div>"
            fig_rows.append(
                f'<li>'
                f'<div class="fig-row">'
                f'<a href="{slug}/figures/{html_mod.escape(fname)}" target="_blank"><strong>{html_mod.escape(fname)}</strong></a> '
                f'<button class="mini-btn" type="button" onclick="copyPlain(\'{lpath_js}\', this)">Copy Path</button>'
                f'</div>'
                f'<code>{html_mod.escape(str(lpath))}</code>'
                f'{alt_line}'
                f'</li>'
            )

        title_str = str(k["title"])
        sub_str = str(k["subtitle"])
        cap_str = str(k["x_caption"]).strip()
        reply_str = str(k["self_reply"]).strip()
        title_js = html_mod.escape(title_str.replace("\\", "\\\\").replace("'", "\\'"))
        sub_js = html_mod.escape(sub_str.replace("\\", "\\\\").replace("'", "\\'"))
        cap_js = html_mod.escape(cap_str.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))
        reply_js = html_mod.escape(reply_str.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n"))
        banner_js = html_mod.escape(str(banner_path).replace("\\", "\\\\").replace("'", "\\'"))

        cards.append(f"""
        <section class="kit-card">
          <div class="kit-meta"><span>{html_mod.escape(str(k["drop"]))}</span><span>{html_mod.escape(str(k["window"]))}</span></div>
          <img class="banner-preview" src="{slug}/x_banner_5x2.png" alt="{html_mod.escape(title_str)} 5:2 banner">
          <div class="banner-path">
            <span><strong>5:2 X Banner Path:</strong> <code>{html_mod.escape(str(banner_path))}</code></span>
            <button class="mini-btn" type="button" onclick="copyPlain('{banner_js}', this)">Copy Banner Path</button>
          </div>
          <h2>{html_mod.escape(title_str)}</h2>
          <p class="sub"><strong>Substack Subtitle:</strong> {html_mod.escape(sub_str)}</p>
          <div class="kit-links">
            <a class="btn-primary" href="{slug}/substack_paste_ready.html" target="_blank">Open Substack Paste-Ready HTML (Auto-Imports Images) ↗</a>
            <a class="btn-primary" href="{slug}/x_article_paste_ready.html" target="_blank">Open X Article Paste-Ready HTML (With File Paths) ↗</a>
            <button class="btn-secondary" type="button" onclick="copyPlain('{title_js}', this)">Copy Title</button>
            <button class="btn-secondary" type="button" onclick="copyPlain('{sub_js}', this)">Copy Subtitle</button>
          </div>
          <div class="reply-box">
            <div class="reply-head">
              <p class="reply-label">X Article Share Post Caption (<code>{slug}/x_caption.txt</code>)</p>
              <button class="mini-btn" type="button" onclick="copyPlain('{cap_js}', this)">Copy X Caption</button>
            </div>
            <pre>{html_mod.escape(cap_str)}</pre>
          </div>
          <div class="reply-box">
            <div class="reply-head">
              <p class="reply-label">1-Minute X Self-Reply (<code>{slug}/self_reply.txt</code>)</p>
              <button class="mini-btn" type="button" onclick="copyPlain('{reply_js}', this)">Copy Self-Reply</button>
            </div>
            <pre>{html_mod.escape(reply_str)}</pre>
          </div>
          <details class="fig-panel" {"open" if slug == "01_part2_bottleneck" else ""}>
            <summary>Inline Figures in Exact Drop Order ({len(figs)} files in <code>{html_mod.escape(str(fig_dir))}</code>)</summary>
            <ol class="fig-path-list">
              {"".join(fig_rows)}
            </ol>
          </details>
        </section>
        """)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>X + Substack Release Launchpad · Parts 2, 3, and 4</title>
<style>
  body {{ max-width: 940px; margin: 40px auto; padding: 0 24px 80px; background: #F7F6EF; color: #0F0F0F; font: 15px/1.6 Inter, -apple-system, BlinkMacSystemFont, sans-serif; }}
  h1 {{ font-family: Georgia, serif; font-size: 32px; margin: 0 0 8px; letter-spacing: -0.02em; }}
  .lead {{ color: #1E3447; margin: 0 0 28px; }}
  .status-banner {{ background: #EEF3EC; border: 1px solid #B8CCB2; border-radius: 10px; padding: 14px 18px; margin-bottom: 28px; font-size: 14px; color: #1E3447; }}
  .kit-card {{ background: #FDFCF8; border: 1px solid #DCE0D3; border-radius: 12px; padding: 26px 28px; margin-bottom: 28px; }}
  .kit-meta {{ display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em; color: #405943; margin-bottom: 14px; }}
  .banner-preview {{ width: 100%; height: auto; border-radius: 8px; border: 1px solid #DCE0D3; display: block; margin-bottom: 8px; }}
  .banner-path {{ display: flex; justify-content: space-between; align-items: center; gap: 10px; font-size: 12px; color: #555; margin-bottom: 16px; flex-wrap: wrap; }}
  .banner-path code {{ background: #F2EFE6; padding: 2px 6px; border-radius: 4px; word-break: break-all; }}
  h2 {{ font-family: Georgia, serif; font-size: 26px; margin: 0 0 6px; letter-spacing: -0.02em; }}
  .sub {{ color: #1E3447; margin: 0 0 16px; font-size: 14.5px; }}
  .kit-links {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 18px; align-items: center; }}
  .btn-primary {{ background: #405943; color: #FDFCF8; text-decoration: none; padding: 9px 14px; border-radius: 8px; font-weight: 600; font-size: 13px; }}
  .btn-secondary {{ appearance: none; cursor: pointer; background: #F7F6EF; color: #0F0F0F; border: 1px solid #DCE0D3; text-decoration: none; padding: 9px 14px; border-radius: 8px; font-weight: 500; font-size: 13px; }}
  .mini-btn {{ appearance: none; cursor: pointer; background: #FFFFFF; color: #1E3447; border: 1px solid #C8C2B4; padding: 3px 9px; border-radius: 5px; font-size: 11.5px; font-weight: 600; }}
  .reply-box {{ background: #F7F6EF; border: 1px solid #DCE0D3; border-radius: 8px; padding: 12px 16px; margin-bottom: 14px; }}
  .reply-head {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; }}
  .reply-label {{ margin: 0; font-size: 11.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #687063; }}
  pre {{ margin: 0; white-space: pre-wrap; font: 13px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }}
  .fig-panel {{ background: #F7F6EF; border: 1px solid #DCE0D3; border-radius: 8px; padding: 10px 14px; font-size: 13px; }}
  .fig-panel summary {{ cursor: pointer; font-weight: 600; color: #1E3447; }}
  .fig-path-list {{ margin: 10px 0 0; padding-left: 22px; }}
  .fig-path-list li {{ margin-bottom: 9px; }}
  .fig-row {{ display: flex; align-items: center; gap: 8px; margin-bottom: 2px; }}
  .fig-row a {{ color: #405943; text-decoration: underline; }}
  .fig-path-list code {{ display: block; font-size: 12px; background: #FDFCF8; border: 1px solid #E5E2D9; padding: 3px 7px; border-radius: 4px; word-break: break-all; }}
  .alt-note {{ font-size: 11.5px; color: #687063; margin-top: 3px; }}
</style>
</head>
<body>
  <h1>X + Substack Release Launchpad</h1>
  <p class="lead">Staggered 3-drop kit matched to Part 1 (<em>How LLMs Price Spite, Disgust, and Dirty Money</em>).</p>
  <div class="status-banner">
    <strong>Image paste fix active:</strong> Substack pages now reference public HTTPS URLs on <code>raw.githubusercontent.com/xmattwest/nullbridge/main/figures/substack/...</code> so Substack imports all figures automatically on paste (with a checkbox to also show local file paths inline). X Article pages use explicit <code>[DROP IMAGE XX: /exact/path]</code> text markers with zero broken image boxes.
  </div>
  {"".join(cards)}
  {COPY_HELPERS_JS}
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Substack + X Article release kits.")
    parser.add_argument("--afriat-folio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--part3-folio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--part4-folio-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("runs/release_kits"))
    parser.add_argument("--public-fig-dir", type=pathlib.Path, default=pathlib.Path("figures/substack"))
    parser.add_argument("--raw-base-url", type=str, default=DEFAULT_RAW_BASE_URL)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    pub_root: pathlib.Path = args.public_fig_dir
    pub_root.mkdir(parents=True, exist_ok=True)

    specs = [
        {
            "slug": "01_part2_bottleneck",
            "drop": "Drop 1 · Part 2",
            "window": "Day 1 · 8:30 AM MT",
            "title": "Untwisting Gemma 4 Onto One 24 GB Card",
            "subtitle": "Where do financial deals and petty insults merge inside a 60-layer model, and how does untwisting that bottleneck let 31B loop on 24 GB?",
            "x_caption": (
                "A bad dollar split and a petty insult share 0 of their top 64 neurons in Gemma 4 31B. Both still write to the same spite direction.\n\n"
                "Finding where they meet let me cut 31B from 62.55 GB to 23.08 GB and loop it on one 24 GB GPU."
            ),
            "folio_dir": args.afriat_folio_dir,
            "extra_methods": "",
            "closing_url": PART2_WEB_URL,
            "repo_url": "https://github.com/xmattwest/afriat",
            "self_reply": (
                "Weights (23.08 GB), loop trainer, and full essay with PDF:\n\n"
                "Essay & PDF: https://shelltoho.me/pubs/rotating-the-bottleneck/\n"
                "Weights: https://huggingface.co/xmattwest/Gemma-4-31B-Loop-Compact\n"
                "Code: https://github.com/xmattwest/afriat"
            ),
        },
        {
            "slug": "02_part3_looping",
            "drop": "Drop 2 · Part 3",
            "window": "Day 4 · 8:30 AM MT",
            "title": "Looping in the Dark",
            "subtitle": "What happens when a crooked ledger puts on an executive suit, your model thinks silently in loops, and standard safety LoRAs backfire?",
            "x_caption": (
                "Dress a crooked ledger in executive prose, and a looped language model caves more on each silent pass. Safety LoRAs break logic on the side.\n\n"
                "Projecting 4 shortcut directions out of the loop weights offline stops the drift at 0 runtime cost."
            ),
            "folio_dir": args.part3_folio_dir,
            "extra_methods": PART3_METHODS_HTML,
            "closing_url": PART3_WEB_URL,
            "repo_url": "https://github.com/xmattwest/nullbridge",
            "self_reply": (
                "Full essay, preprint, and open-source code:\n\n"
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
            "x_caption": (
                "Give a looped language model extra silent passes on a shady deal, and dirty-money rejections double with zero chain-of-thought.\n\n"
                "Across intersecting budget lines, silent loops lift revealed-preference consistency to 0.994 and 1.000."
            ),
            "folio_dir": args.part4_folio_dir,
            "extra_methods": PART4_METHODS_HTML,
            "closing_url": PART4_WEB_URL,
            "repo_url": "https://github.com/xmattwest/nullbridge",
            "self_reply": (
                "Full essay, preprint, utility solvers, and all four parts of the series:\n\n"
                "Essay & Preprint: https://shelltoho.me/pubs/coherent-moral-utility/\n"
                "Full series: https://shelltoho.me/pubs/#money-is-moral-series\n"
                "Code: https://github.com/xmattwest/nullbridge"
            ),
        },
    ]

    launchpad_kits: list[dict[str, object]] = []
    for sp in specs:
        slug = str(sp["slug"])
        kdir = out_dir / slug
        kdir.mkdir(parents=True, exist_ok=True)
        pub_sub_dir = pub_root / slug
        if pub_sub_dir.exists():
            shutil.rmtree(pub_sub_dir)
        pub_sub_dir.mkdir(parents=True, exist_ok=True)

        raw_banner = kdir / "x_banner_5x2_raw.png"
        final_banner = kdir / "x_banner_5x2.png"
        assert raw_banner.exists(), f"Missing generated raw banner: {raw_banner}"
        fit_banner_5x2(raw_banner, final_banner)

        folio_sub = pathlib.Path(sp["folio_dir"]) / "substack"  # type: ignore[arg-type]
        with tempfile.TemporaryDirectory(prefix="stage-figs-") as stage_tmp:
            stage_fig_dir = pathlib.Path(stage_tmp)
            for f in (folio_sub / "figures").iterdir():
                if f.is_file():
                    shutil.copy2(f, stage_fig_dir / f.name)
            if slug == "01_part2_bottleneck":
                render_equation_card_2x("part2", stage_fig_dir / "zero_cost_bridge_eq_2x.png")
            elif slug == "02_part3_looping":
                render_equation_card_2x("part3", stage_fig_dir / "zero_cost_bridge_eq_2x.png")

            raw_sub_html = (folio_sub / "substack_paste_ready.html").read_text()
            body = extract_substack_body(raw_sub_html, slug)
            body = body.replace(f'href="{PART1_WEB_URL}"', f'href="{PART1_SUBSTACK_URL}"')
            body = body.replace("<h2>Methods, Repo &amp; Technical Notes</h2>", "<h3>Methods, Repo &amp; Technical Notes</h3>")
            if sp["extra_methods"]:
                body = f"{body}\n\n{sp['extra_methods']}"

            if (kdir / "figures").exists():
                shutil.rmtree(kdir / "figures")
            body, ordered_figs = renumber_figures(body, stage_fig_dir, kdir / "figures", pub_sub_dir)

        if slug == "01_part2_bottleneck":
            snap = folio_sub.parent.parent / "funnel_cards/rotating_corkscrew_procrustes_snap.png"
            mp4 = folio_sub.parent.parent / "social_cards/manifold_3d_coherent_multipass.mp4"
            if snap.exists():
                shutil.copy2(snap, kdir / "figures/01_alt_static_procrustes_snap.png")
                shutil.copy2(snap, pub_sub_dir / "01_alt_static_procrustes_snap.png")
            if mp4.exists():
                shutil.copy2(mp4, kdir / "figures/01_alt_video_multipass.mp4")

        sub_doc = make_substack_doc(
            slug=slug,
            title=str(sp["title"]),
            subtitle=str(sp["subtitle"]),
            body_html=body,
            ordered_figs=ordered_figs,
            local_fig_dir=kdir / "figures",
            raw_base_url=args.raw_base_url,
        )
        x_doc = make_x_article_doc(
            slug=slug,
            title=str(sp["title"]),
            x_caption=str(sp["x_caption"]),
            banner_path=final_banner,
            body_html=body,
            ordered_figs=ordered_figs,
            local_fig_dir=kdir / "figures",
            closing_url=str(sp["closing_url"]),
            repo_url=str(sp["repo_url"]),
            self_reply=str(sp["self_reply"]),
        )
        cap_txt = str(sp["x_caption"]).strip() + "\n"
        assert len(str(sp["x_caption"]).strip()) < 250, (
            f"X caption for {slug} is {len(str(sp['x_caption']).strip())} chars (must be < 250)"
        )
        reply_txt = str(sp["self_reply"]).strip() + "\n"

        for name, content in [
            ("substack_paste_ready.html", sub_doc),
            ("x_article_paste_ready.html", x_doc),
            ("x_caption.txt", cap_txt),
            ("self_reply.txt", reply_txt),
        ]:
            assert "\u2014" not in content, f"Em dash found in {slug}/{name}"
            (kdir / name).write_text(content)

        launchpad_kits.append({
            **sp,
            "figures": ordered_figs,
            "fig_dir": (kdir / "figures").resolve(),
            "banner_path": final_banner.resolve(),
        })
        print(f"[OK] Assembled {slug} ({len(ordered_figs)} figures + public HTTPS mirror + 5:2 banner)")

    lp_html = build_launchpad_html(launchpad_kits)
    assert "\u2014" not in lp_html
    (out_dir / "index.html").write_text(lp_html)
    print(f"[OK] Wrote release launchpad -> {out_dir / 'index.html'}")


if __name__ == "__main__":
    main()
