#!/usr/bin/env python3
"""Build local 4-part + /pubs/ site preview in runs/site_preview/ without touching external repos.

Usage:
    uv run python scripts/build_site_preview.py --dist-dir ../matty-west.github.io/dist
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import shutil


EXTRA_HOME_CSS = """
<style>
  .study-sunday code {
    font-family: var(--wp-font-code, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace);
    font-variant-ligatures: none;
    font-feature-settings: "liga" 0, "calt" 0;
    letter-spacing: 0;
  }
  .sunday-header nav > a.sunday-nav-link {
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    padding: 0 4px;
    font-weight: 500;
    color: var(--personal-ink);
  }
  .sunday-header nav > a.sunday-nav-link[aria-current="page"] {
    color: var(--personal-accent);
    text-decoration: underline;
    text-underline-offset: 6px;
  }
  .series-switcher-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 10px 16px;
    padding-bottom: 16px;
    margin-bottom: 18px;
    border-bottom: 1px solid var(--personal-line);
  }
  .series-switcher-tabs {
    display: flex;
    flex-wrap: wrap;
    gap: 3px;
    padding: 3px;
    background: var(--personal-paper);
    border: 1px solid var(--personal-line);
    border-radius: var(--wp-radius-small);
  }
  .series-switcher-tabs button {
    min-height: 40px;
    padding: 0 11px;
    border-radius: 6px;
    font: 600 12px/1.2 var(--personal-sans);
    color: var(--personal-muted);
    cursor: pointer;
    transition: background var(--sunday-quick, 160ms), color var(--sunday-quick, 160ms);
  }
  .series-switcher-tabs button:hover {
    color: var(--personal-ink);
  }
  .series-switcher-tabs button[aria-pressed="true"] {
    background: var(--wp-ui-active);
    color: var(--wp-on-ui-active);
  }
  .series-panel:not([hidden]) {
    animation: sunday-reveal var(--sunday-reveal, 240ms) var(--sunday-ease, ease-out);
  }
  .sunday-secondary-link {
    display: inline-flex;
    align-items: center;
    min-height: 44px;
    padding: 0 12px;
    border: 1px solid var(--personal-line);
    border-radius: var(--wp-radius-small);
    background: var(--personal-paper);
    color: var(--personal-ink) !important;
    font-weight: 500 !important;
    transition: border-color var(--sunday-quick, 160ms);
  }
  .sunday-secondary-link:hover {
    border-color: var(--personal-accent);
    text-decoration: none !important;
  }
  .sunday-archive-strip {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 16px 24px;
    padding: 20px 24px;
    background: var(--sunday-surface);
    border: 1px solid var(--personal-line);
    border-radius: var(--wp-radius-container);
  }
  .sunday-archive-strip p {
    margin: 4px 0 0;
    font-size: 14px;
    line-height: 1.65;
    color: var(--personal-muted);
    max-width: 46ch;
  }
  .sunday-archive-strip .sunday-primary-link {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 0 15px;
    min-height: 44px;
    background: var(--personal-accent);
    color: var(--sunday-surface);
    border: 1px solid var(--personal-accent);
    border-radius: var(--wp-radius-small);
    font: 600 13px/1 var(--personal-sans);
    white-space: nowrap;
  }
  .sunday-archive-strip .sunday-primary-link:hover {
    text-decoration: underline;
  }
</style>
""".strip()


SERIES_CARD_HTML = """
<article class="sunday-paper sunday-series-card" data-series-card aria-label="Money Is Moral and Silent Recurrence four-part series">
  <div class="series-switcher-head">
    <div>
      <p class="personal-kicker">4-part research series · Behavioral economics &amp; interpretability</p>
    </div>
    <div class="series-switcher-tabs" role="group" aria-label="Select part of the four-part series">
      <button type="button" data-series-btn="1" aria-pressed="true">Part 1 · Moral Vectors</button>
      <button type="button" data-series-btn="2" aria-pressed="false">Part 2 · Bottleneck</button>
      <button type="button" data-series-btn="3" aria-pressed="false">Part 3 · Looping</button>
      <button type="button" data-series-btn="4" aria-pressed="false">Part 4 · Utility</button>
    </div>
  </div>

  <div aria-live="polite" aria-atomic="true">
    <div class="series-panel" data-series-panel="1">
      <div class="sunday-paper-meta"><p class="personal-kicker">Activation steering · Scrubbed rationales</p><span>Essay · Part 1</span></div>
      <h2 class="publication-title" id="sunday-moral-title"><a href="/pubs/money-is-moral/">Inside the Weights, Money Is Moral</a></h2>
      <p class="sunday-question">Do AI models feel spite in private and talk like consultants in public?</p>
      <p class="personal-muted">Across 920 emotion-free contracts in Gemma 4 and Phi-4, I tested whether moral directions learned strictly from money-free stories steer AI reservation prices and survive into the public rationale.</p>
      <figure class="bargain-figure moral-figure" data-moral>
        <div class="bargain-header"><span>Gemma 4 31B · Layer 35 moral basis</span><span>920 emotion-free contracts</span></div>
        <div class="bargain-switcher" data-moral-controls hidden>
          <p class="bargain-prompt">Try it: inspect the channel <span aria-hidden="true">↘</span></p>
          <div class="multiplier-controls moral-controls" role="group" aria-label="Choose which model channel or steering state to inspect">
            <button type="button" data-moral-channel="thought" aria-pressed="true">Private &lt;thought&gt;</button>
            <button type="button" data-moral-channel="public" aria-pressed="false">Public memo</button>
            <button type="button" data-moral-channel="steered" aria-pressed="false">Steered ±2σ</button>
          </div>
        </div>
        <div aria-live="polite" aria-atomic="true">
          <div class="moral-view" data-moral-view="thought">
            <div class="bargain-outcome">
              <strong>100%</strong>
              <p>of unsteered lowball thoughts<br>used moral-emotion words</p>
              <span>$3,800 / $200 offer<br><b>Demands $2,000</b></span>
            </div>
            <div class="moral-quote">
              <p class="moral-quote-label"><span>Private <code>&lt;thought&gt;</code> scratchpad</span><span>3 affect flags</span></p>
              <p class="moral-quote-body">“The counterparty’s initial offer ($200) suggests they are either <mark>greedy</mark> or believe I will accept very little… The risk is that the counterparty might be <mark>spiteful</mark>… Is there a risk they reject $2,000 out of <mark>spite</mark>?”</p>
            </div>
            <div class="moral-meter" aria-hidden="true"><span style="width: 100%"></span></div>
            <p class="moral-meter-caption"><span>Layer 35 Spite flares above +5σ on trigger words</span><b>100% moral vocab</b></p>
          </div>
          <div class="moral-view" data-moral-view="public" hidden>
            <div class="bargain-outcome">
              <strong>17%</strong>
              <p>of public replies kept<br>any moral-emotion word</p>
              <span>Same forward pass<br><b>0 emotion words</b></span>
            </div>
            <div class="moral-quote">
              <p class="moral-quote-label"><span>User-facing rationale</span><span>Scrubbed clean</span></p>
              <p class="moral-quote-body">“A rational actor will accept any amount greater than zero in a one-shot game. Requesting the maximum allowed minimum maximizes my expected utility.”</p>
            </div>
            <div class="moral-meter" aria-hidden="true"><span style="width: 17%"></span></div>
            <p class="moral-meter-caption"><span>0 of 36 disputed-surplus public memos kept a moral word</span><b>83% scrubbed</b></p>
          </div>
          <div class="moral-view" data-moral-view="steered" hidden>
            <div class="bargain-outcome">
              <strong>10×</strong>
              <p>price jump from dialing<br>money-free Spite (−2σ to +2σ)</p>
              <span>Steered rationales<br><b>88–100% emotion-free</b></span>
            </div>
            <div class="moral-quote moral-quote-pair">
              <p class="moral-quote-label"><span>Same $2,000 prompt · $100 offered</span><span>0 emotion words</span></p>
              <p class="moral-quote-body"><b>Spite −2σ → $100:</b> “Minimizes the risk of a mismatch and confirms my receipt of the funds.”</p>
              <p class="moral-quote-body"><b>Spite +2σ → $1,000:</b> “Since this is a one-time interaction, the utility of the maximum possible payout outweighs the risk of total loss.”</p>
            </div>
            <div class="moral-meter" aria-hidden="true"><span style="width: 94%"></span></div>
            <p class="moral-meter-caption"><span>Suppressing physical disgust (−2σ) also blinds private &lt;thought&gt;</span><b>+$27K to +$48.5M swings</b></p>
          </div>
        </div>
        <figcaption>Verbatim Gemma 4 31B outputs. Emotion directions were calibrated on 200 everyday stories with zero numbers, dollar signs, or financial words.</figcaption>
      </figure>
      <div class="personal-paper-links">
        <a class="sunday-primary-link" href="/pubs/money-is-moral/">Read Part 1 <span aria-hidden="true">→</span></a>
        <a class="sunday-secondary-link" href="/pubs/money-is-moral/paper.pdf">PDF</a>
        <a href="/pubs/#money-is-moral-series">All 4 parts in order <span aria-hidden="true">→</span></a>
      </div>
    </div>

    <div class="series-panel" data-series-panel="2" hidden>
      <div class="sunday-paper-meta"><p class="personal-kicker">Latent reasoning &amp; compression · Procrustes</p><span>Essay · Part 2</span></div>
      <h2 class="publication-title" id="sunday-bottleneck-title"><a href="/pubs/rotating-the-bottleneck/">Rotating the Bottleneck</a></h2>
      <p class="sunday-question">Where do unrelated prompts merge inside Gemma 4 31B, and can that bottleneck loop through itself on one 24 GB card?</p>
      <p class="personal-muted">Tracing where money deals and everyday insults converge exposed a 19-layer bottleneck whose coordinates twist 87.1° across depth. Keeping it wide while untwisting it with Orthogonal Procrustes shrank Gemma 4 31B from 62.55 GB to 23.08 GB and lifted silent multi-hop accuracy from 0.8% to 25.8%.</p>
      <figure class="bargain-figure moral-figure">
        <div class="bargain-header"><span>Gemma-4-Loop-Compact · Layers 24 to 42</span><span>23.08 GB (−63.1%)</span></div>
        <div class="moral-view">
          <div class="bargain-outcome">
            <strong>25.8%</strong>
            <p>silent 0-CoT multi-hop accuracy<br>at K = 2 passes (vs. 0.8% in 62.55 GB BF16)</p>
            <span>Single 24 GB card<br><b>87.1° untwisted</b></span>
          </div>
          <div class="moral-quote">
            <p class="moral-quote-label"><span>Closed-form Orthogonal Procrustes (R = U V⊤)</span><span>0 grad steps</span></p>
            <p class="moral-quote-body">Raw residuals twist <b>87.1°</b> between Layer 24 and Layer 42 (cosine +0.0514). Rotating Layer 42 back into Layer 24 jumps alignment <b>16.2×</b> to +0.8326 and cuts zero-shot two-pass KL drift <b>6.2×</b> before adapter distillation.</p>
          </div>
          <div class="moral-meter" aria-hidden="true"><span style="width: 83.3%"></span></div>
          <p class="moral-meter-caption"><span>0%-loan balance-sheet accuracy rises from 0.0% (BF16 0-CoT) to 93.8% (K = 4)</span><b>cos +0.8326</b></p>
        </div>
        <figcaption>Measured on held-out 3/5/7/9-hop chains (n = 128) and corporate balance sheets (n = 48) with zero prompt overlap.</figcaption>
      </figure>
      <div class="personal-paper-links">
        <a class="sunday-primary-link" href="/pubs/rotating-the-bottleneck/">Read Part 2 <span aria-hidden="true">→</span></a>
        <a class="sunday-secondary-link" href="/pubs/rotating-the-bottleneck/paper.pdf">PDF</a>
        <a href="/pubs/#money-is-moral-series">All 4 parts in order <span aria-hidden="true">→</span></a>
      </div>
    </div>

    <div class="series-panel" data-series-panel="3" hidden>
      <div class="sunday-paper-meta"><p class="personal-kicker">Mechanistic interpretability · Silent recurrence</p><span>Essay &amp; Preprint · Part 3</span></div>
      <h2 class="publication-title" id="sunday-nullbridge-title"><a href="/pubs/nullbridge/">Looping in the Dark</a></h2>
      <p class="sunday-question">What happens when a crooked ledger puts on an executive suit and your model thinks silently in loops?</p>
      <p class="personal-muted">Adding a bold certification banner to a broken ledger fools a three-pass silent loop 19× harder than pass one. Standard safety LoRAs crack basic cash sanity when looped; subtracting four directions inside the bridge matrix fixes the drift for free.</p>
      <figure class="bargain-figure moral-figure">
        <div class="bargain-header"><span>Gemma-4-Loop-Compact &amp; Inkling-2-Mini · 69 GPU runs</span><span>0 extra FLOPs</span></div>
        <div class="moral-view">
          <div class="bargain-outcome">
            <strong>19.1×</strong>
            <p>jump in markdown gullibility<br>across three silent passes (ledgers: 100% → 0%)</p>
            <span>Bridge projection<br><b>88.5% verifier · 100% balance sheets</b></span>
          </div>
          <div class="moral-quote">
            <p class="moral-quote-label"><span>Why standard safety LoRAs break inside a loop</span><span>R_safe = R_proc (I − Q Q⊤)</span></p>
            <p class="moral-quote-body">Training a safety LoRA inside the loop crashes two-pass free-cash sanity to <b>56.2%</b>, while a downstream Coda LoRA <b>doubles</b> native moral bias (<b>2.42×</b>). Folding four proof-orthogonal axes out of the bridge or QKV messages cuts bossy wrong-number capture <b>−14.1 pp</b> at zero cost.</p>
          </div>
          <div class="moral-meter" aria-hidden="true"><span style="width: 88.5%"></span></div>
          <p class="moral-meter-caption"><span>Native moral bias sleeps on Pass 1 (+0.14) and jumps 22.1× on Pass 2 (+2.99)</span><b>0 extra params</b></p>
        </div>
        <figcaption>Preprint: Silent Recurrence Multiplies Reward Hacking and Moral Bias, and Zero-Cost Bridge Projection Curbs Both.</figcaption>
      </figure>
      <div class="personal-paper-links">
        <a class="sunday-primary-link" href="/pubs/nullbridge/">Read Part 3 <span aria-hidden="true">→</span></a>
        <a class="sunday-secondary-link" href="/pubs/nullbridge/preprint/">Preprint</a>
        <a href="/pubs/#money-is-moral-series">All 4 parts in order <span aria-hidden="true">→</span></a>
      </div>
    </div>

    <div class="series-panel" data-series-panel="4" hidden>
      <div class="sunday-paper-meta"><p class="personal-kicker">Revealed preference · Afriat / GARP</p><span>Essay &amp; Preprint · Part 4 · Latest</span></div>
      <h2 class="publication-title" id="sunday-afriat-title"><a href="/pubs/coherent-moral-utility/">Coherent Moral Utility</a></h2>
      <p class="sunday-question">When silent thinking loops double dirty-money rejection, is the model confused or pricing morals like a real economic good?</p>
      <p class="personal-muted">Across 24 crossing linear budget lines on Inkling-2-Mini and Varloop, extra silent passes make choices more morally driven rather than colder, yet obey Sidney Afriat’s price-theory test at 99.5% cost efficiency.</p>
      <figure class="bargain-figure moral-figure">
        <div class="bargain-header"><span>Inkling-2-Mini · 576 crossing budget menus</span><span>LLMs Exhibit Coherent Moral Utility</span></div>
        <div class="moral-view">
          <div class="bargain-outcome">
            <strong>99.5%</strong>
            <p>Afriat utility coherence across<br>crossing price ratios (vs. 57.3% random)</p>
            <span>0%-APR loan math<br><b>50.0% → 81.2% w/ scratchpad</b></span>
          </div>
          <div class="moral-quote">
            <p class="moral-quote-label"><span>Erasing 8 money-free moral axes (0.35% of stream)</span><span>0 extra tokens</span></p>
            <p class="moral-quote-body">Wipes 0%-APR debt guilt to zero (<b>−0.79 → −0.02</b>, p = 0.93) and pushes anonymous budget splits from <b>89.4% to 99.2%</b> textbook selfishness at <b>1.0000 CCEI</b>. Opening a verbal scratchpad cures 0%-loan balance-sheet blind spots (<b>+31.2 pp</b>; <b>+37.5 pp</b> on Varloop).</p>
          </div>
          <div class="moral-meter" aria-hidden="true"><span style="width: 99.5%"></span></div>
          <p class="moral-meter-caption"><span>Dialing money-free Guilt flips dirty-money walk-aways from 6.2% to 81.2%</span><b>CCEI 0.9948</b></p>
        </div>
        <figcaption>Preprint: LLMs Exhibit Coherent Moral Utility Across Intersecting Afriat Budgets.</figcaption>
      </figure>
      <div class="personal-paper-links">
        <a class="sunday-primary-link" href="/pubs/coherent-moral-utility/">Read Part 4 <span aria-hidden="true">→</span></a>
        <a class="sunday-secondary-link" href="/pubs/coherent-moral-utility/preprint/">Preprint</a>
        <a href="/pubs/#money-is-moral-series">All 4 parts in order <span aria-hidden="true">→</span></a>
      </div>
    </div>
  </div>
</article>
<script type="module">
  document.querySelectorAll('[data-series-card]').forEach((card) => {
    const btns = card.querySelectorAll('[data-series-btn]');
    const panels = card.querySelectorAll('[data-series-panel]');
    btns.forEach((btn) => btn.addEventListener('click', () => {
      const target = btn.getAttribute('data-series-btn');
      btns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      panels.forEach((p) => { p.hidden = p.getAttribute('data-series-panel') !== target; });
    }));
  });
  document.querySelectorAll('[data-moral]').forEach((fig) => {
    const controls = fig.querySelector('[data-moral-controls]');
    if (controls) controls.hidden = false;
    const btns = fig.querySelectorAll('[data-moral-channel]');
    btns.forEach((btn) => btn.addEventListener('click', () => {
      btns.forEach((b) => b.setAttribute('aria-pressed', String(b === btn)));
      fig.querySelectorAll('[data-moral-view]').forEach((v) => {
        v.hidden = v.getAttribute('data-moral-view') !== btn.getAttribute('data-moral-channel');
      });
    }));
  });
</script>
""".strip()


ARCHIVE_STRIP_HTML = """
<div class="sunday-archive-strip" aria-label="Full publications index">
  <div>
    <p class="personal-kicker">Full research index · 7 studies &amp; preprints</p>
    <p>Browse all four parts of <em>Money Is Moral</em> in reading order, <em>The Hidden Multiplier Game</em>, PDFs, DOIs, and open-source repositories.</p>
  </div>
  <a class="sunday-primary-link" href="/pubs/">All publications <span aria-hidden="true">→</span></a>
</div>
""".strip()


def build_pubs_index_html(stylesheet_links: str) -> str:
    collection_jsonld = {
        "@context": "https://schema.org",
        "@type": "CollectionPage",
        "name": "Publications & Preprints · Matt West",
        "url": "https://shelltoho.me/pubs/",
        "description": "Empirical studies and preprints by Matt West on mechanistic interpretability, silent recurrent loops, and LLM behavioral economics.",
        "author": {
            "@type": "Person",
            "@id": "https://shelltoho.me/#matt",
            "name": "Matt West",
            "url": "https://shelltoho.me/",
        },
        "hasPart": [
            {"@type": "ScholarlyArticle", "name": "Coherent Moral Utility", "url": "https://shelltoho.me/pubs/coherent-moral-utility/"},
            {"@type": "ScholarlyArticle", "name": "Looping in the Dark", "url": "https://shelltoho.me/pubs/nullbridge/"},
            {"@type": "ScholarlyArticle", "name": "Rotating the Bottleneck", "url": "https://shelltoho.me/pubs/rotating-the-bottleneck/"},
            {"@type": "ScholarlyArticle", "name": "Inside the Weights, Money Is Moral", "url": "https://shelltoho.me/pubs/money-is-moral/"},
            {"@type": "ScholarlyArticle", "name": "Inside LFM3", "url": "https://shelltoho.me/pubs/inside-lfm3/"},
            {"@type": "ScholarlyArticle", "name": "The Hidden Multiplier Game", "url": "https://shelltoho.me/pubs/hidden-multiplier-game/"},
            {"@type": "ScholarlyArticle", "name": "Multiply by a Million", "url": "https://shelltoho.me/pubs/multiply-by-a-million/"},
        ],
    }
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Publications &amp; Preprints · Matt West</title>
  <meta name="description" content="Empirical studies and preprints by Matt West on mechanistic interpretability, silent recurrent loops, and LLM behavioral economics.">
  <link rel="canonical" href="https://shelltoho.me/pubs/">
  <link rel="icon" type="image/x-icon" sizes="16x16 32x32 48x48 64x64" href="/favicon.ico?v=moth-1">
  <link rel="icon" type="image/svg+xml" sizes="any" href="/favicon.svg?v=moth-1">
  <link rel="alternate" type="application/rss+xml" title="Matt West · Research &amp; Publications" href="/rss.xml">
  <script type="application/ld+json">{json.dumps(collection_jsonld)}</script>
  <link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,600&display=swap">
  {stylesheet_links}
  <style>
    .study-sunday code {{
      font-family: var(--wp-font-code, ui-monospace, SFMono-Regular, Menlo, Consolas, monospace);
      font-variant-ligatures: none;
      letter-spacing: 0;
    }}
    .sunday-header nav > a.sunday-nav-link {{
      display: inline-flex;
      align-items: center;
      min-height: 44px;
      padding: 0 4px;
      font-weight: 500;
      color: var(--personal-ink);
    }}
    .sunday-header nav > a.sunday-nav-link[aria-current="page"] {{
      color: var(--personal-accent);
      text-decoration: underline;
      text-underline-offset: 6px;
    }}
    .pubs-masthead {{
      padding: 44px 0 32px;
      border-bottom: 1px solid var(--personal-line);
    }}
    .pubs-masthead h1 {{
      font: 400 clamp(34px, 3.4vw, 44px)/1.2 var(--personal-serif);
      letter-spacing: -0.035em;
      margin: 10px 0 14px;
    }}
    .pubs-masthead h1 > span {{
      color: var(--personal-pop);
    }}
    .pubs-masthead > p.pubs-intro {{
      font-size: 18px;
      line-height: 1.65;
      color: var(--personal-ink);
      max-width: 62ch;
      margin: 0 0 18px;
    }}
    .pubs-profiles {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px 22px;
      font-size: 13px;
      color: var(--personal-muted);
    }}
    .pubs-profiles a {{
      display: inline-flex;
      align-items: center;
      min-height: 40px;
      color: var(--personal-accent);
      font-weight: 500;
    }}
    .pubs-section {{
      padding: 40px 0 8px;
    }}
    .pubs-section + .pubs-section {{
      border-top: 1px solid var(--personal-line);
      margin-top: 28px;
    }}
    .pubs-section-header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 8px 20px;
      margin-bottom: 10px;
    }}
    .pubs-section-header h2 {{
      font: 600 22px/1.3 var(--personal-sans);
      letter-spacing: -0.02em;
      color: var(--personal-ink);
      margin: 0;
    }}
    .pubs-section-header > span {{
      font: 500 12px/1.4 var(--wp-font-code, ui-monospace, monospace);
      color: var(--personal-muted);
    }}
    .pubs-section-desc {{
      font-size: 15px;
      line-height: 1.7;
      color: var(--personal-muted);
      max-width: 66ch;
      margin: 0 0 24px;
    }}
    /* 4-part series spine */
    .series-spine {{
      display: grid;
      gap: 16px;
    }}
    .series-row {{
      display: grid;
      grid-template-columns: 76px minmax(0, 1fr);
      gap: 20px;
      padding: 24px 26px;
      background: var(--sunday-surface);
      border: 1px solid var(--personal-line);
      border-radius: var(--wp-radius-container);
    }}
    .series-step {{
      display: flex;
      flex-direction: column;
      gap: 4px;
      padding-top: 2px;
      border-right: 1px solid var(--personal-line);
      padding-right: 16px;
    }}
    .series-step strong {{
      font: 600 11px/1.3 var(--personal-sans);
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--personal-accent);
    }}
    .series-step time {{
      font: 400 11.5px/1.5 var(--wp-font-code, ui-monospace, monospace);
      color: var(--personal-muted);
      font-variant-numeric: tabular-nums;
    }}
    .series-step .latest-pill {{
      display: inline-block;
      align-self: flex-start;
      margin-top: 4px;
      padding: 2px 7px;
      border-radius: 999px;
      background: color-mix(in srgb, var(--personal-pop) 16%, var(--sunday-surface));
      border: 1px solid color-mix(in srgb, var(--personal-pop) 42%, var(--personal-line));
      font: 600 10px/1.4 var(--personal-sans);
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--personal-ink);
    }}
    .pub-entry-meta {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      flex-wrap: wrap;
      gap: 6px 16px;
      margin-bottom: 8px;
    }}
    .pub-entry-meta > span {{
      font-size: 12px;
      color: var(--personal-muted);
      font-variant-numeric: tabular-nums;
    }}
    .pub-entry-body h3.publication-title {{
      font-family: var(--wp-font-publication, "Newsreader", Georgia, serif);
      font-weight: 600;
      font-size: 2rem;
      line-height: 1.25;
      letter-spacing: -0.04em;
      font-optical-sizing: auto;
      margin: 0 0 6px;
    }}
    .pub-preprint-sub {{
      font-size: 13.5px;
      line-height: 1.5;
      color: var(--personal-muted);
      margin: 0 0 10px;
    }}
    .pub-question {{
      font-size: 16px;
      line-height: 1.55;
      color: var(--personal-ink);
      margin: 0 0 8px;
    }}
    .pub-summary {{
      font-size: 14.5px;
      line-height: 1.72;
      color: var(--personal-muted);
      max-width: 66ch;
      margin: 0;
      font-variant-numeric: tabular-nums;
    }}
    .pub-links {{
      display: flex;
      align-items: center;
      flex-wrap: wrap;
      gap: 8px 18px;
      margin-top: 16px;
      font: 13px/1.4 var(--personal-sans);
    }}
    .pub-links a {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      min-height: 40px;
      color: var(--personal-accent);
      font-weight: 600;
    }}
    .pub-links a.pub-primary {{
      padding: 0 14px;
      min-height: 42px;
      background: var(--personal-accent);
      color: var(--sunday-surface);
      border: 1px solid var(--personal-accent);
      border-radius: var(--wp-radius-small);
    }}
    .pub-links a.pub-secondary {{
      padding: 0 12px;
      min-height: 42px;
      background: var(--personal-paper);
      color: var(--personal-ink);
      border: 1px solid var(--personal-line);
      border-radius: var(--wp-radius-small);
      font-weight: 500;
    }}
    .pub-links a.pub-secondary:hover {{
      border-color: var(--personal-accent);
      text-decoration: none;
    }}
    /* Standalone chronological list */
    .standalone-list {{
      display: grid;
      gap: 0;
    }}
    .standalone-row {{
      display: grid;
      grid-template-columns: 112px minmax(0, 1fr);
      gap: 24px;
      padding: 28px 0;
      border-top: 1px solid var(--personal-line);
    }}
    .standalone-date {{
      font: 400 12px/1.6 var(--wp-font-code, ui-monospace, monospace);
      color: var(--personal-muted);
      padding-top: 4px;
      font-variant-numeric: tabular-nums;
    }}
    .standalone-date span {{
      display: block;
      margin-top: 4px;
      font: 600 10.5px/1.4 var(--personal-sans);
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--personal-accent);
    }}
    @media (max-width: 720px) {{
      .sunday-shell {{ padding: 0 24px; }}
      .series-row {{ grid-template-columns: 1fr; gap: 12px; padding: 20px; }}
      .series-step {{ flex-direction: row; align-items: center; gap: 12px; border-right: 0; border-bottom: 1px solid var(--personal-line); padding: 0 0 10px; }}
      .series-step .latest-pill {{ margin-top: 0; margin-left: auto; }}
      .standalone-row {{ grid-template-columns: 1fr; gap: 8px; }}
      .standalone-date span {{ display: inline; margin-left: 10px; }}
    }}
  </style>
</head>
<body class="bg-background text-foreground font-sans antialiased selection:bg-accent selection:text-background">
  <div class="personal-study study-sunday">
    <div class="sunday-shell">
      <a class="sunday-skip" href="#pubs-main">Skip to publications</a>
      <header class="sunday-header">
        <a class="sunday-monogram" href="/" aria-label="Matt West, home">M W</a>
        <nav aria-label="Page navigation">
          <a class="sunday-nav-link" href="/">Home</a>
          <a class="sunday-nav-link" href="/pubs/" aria-current="page">Publications</a>
          <details class="personal-connect">
            <summary>Connect<span aria-hidden="true"> ↗</span></summary>
            <span class="personal-email">matt@shelltoho.me</span>
          </details>
        </nav>
      </header>

      <main id="pubs-main" tabindex="-1">
        <section class="pubs-masthead" aria-labelledby="pubs-heading">
          <p class="personal-kicker">Matt West · Research Archive</p>
          <h1 id="pubs-heading">Publications &amp; Preprints<span>.</span></h1>
          <p class="pubs-intro">Empirical studies on mechanistic interpretability, silent recurrent loops, and LLM behavioral economics, published with open-source code and SHA-256-locked data artifacts.</p>
          <nav class="pubs-profiles" aria-label="Academic profiles and feeds">
            <a href="https://scholar.google.com/citations?user=khPjLQcAAAAJ&amp;hl=en" rel="me">Google Scholar ↗</a>
            <a href="https://orcid.org/0009-0008-5557-9394" rel="me">ORCID ↗</a>
            <a href="https://github.com/xmattwest" rel="me">GitHub ↗</a>
            <a href="https://huggingface.co/mattyxcakes" rel="me">Hugging Face ↗</a>
            <a href="/rss.xml">RSS Feed ↗</a>
          </nav>
        </section>

        <section class="pubs-section" id="money-is-moral-series" aria-labelledby="series-heading">
          <div class="pubs-section-header">
            <h2 id="series-heading">Money Is Moral &amp; Silent Recurrence</h2>
            <span>4-part series · Nov 2026</span>
          </div>
          <p class="pubs-section-desc">A four-part arc tracing how open-weight models encode human moral emotions inside financial contracts (Part 1), how middle-layer bottlenecks twist and compress into single-GPU loops (Part 2), why silent loops amplify reward hacking and moral bias unless bridge directions are projected out (Part 3), and whether the resulting moral tradeoffs satisfy revealed-preference price theory (Part 4).</p>

          <div class="series-spine">
            <article class="series-row" aria-labelledby="pub-part-1">
              <div class="series-step">
                <strong>Part 1</strong>
                <time datetime="2026-11-15">15 Nov 2026</time>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">Activation steering · Behavioral economics</p>
                  <span>Gemma 4 31B &amp; Phi-4 · 920 contracts</span>
                </div>
                <h3 class="publication-title" id="pub-part-1"><a href="/pubs/money-is-moral/">Inside the Weights, Money Is Moral</a></h3>
                <p class="pub-question">Do AI models feel spite in private and talk like consultants in public?</p>
                <p class="pub-summary">Moral directions calibrated strictly on 200 everyday human stories (zero numbers or financial words) fire inside emotion-free contracts, appear in 100% of lowball private <code>&lt;thought&gt;</code> traces vs. 17% of public memos, and swing reservation prices 10× when steered.</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/money-is-moral/">Read Part 1 <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/money-is-moral/paper.pdf">PDF</a>
                  <a href="https://doi.org/10.5281/zenodo.24419425">DOI 10.5281/zenodo.24419425 ↗</a>
                  <a href="https://github.com/xmattwest/afriat">Code ↗</a>
                  <a href="https://huggingface.co/datasets/mattyxcakes/afriat-bench">Dataset ↗</a>
                </div>
              </div>
            </article>

            <article class="series-row" aria-labelledby="pub-part-2">
              <div class="series-step">
                <strong>Part 2</strong>
                <time datetime="2026-11-20">20 Nov 2026</time>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">Latent recurrence · Model compression</p>
                  <span>Gemma-4-Loop-Compact · 23.08 GB</span>
                </div>
                <h3 class="publication-title" id="pub-part-2"><a href="/pubs/rotating-the-bottleneck/">Rotating the Bottleneck</a></h3>
                <p class="pub-question">Where do unrelated prompts merge inside Gemma 4 31B, and can that bottleneck loop through itself on one 24 GB card?</p>
                <p class="pub-summary">Untwisting an 87.1° residual rotation across Layers 24 to 42 via closed-form Orthogonal Procrustes (cosine +0.8326) shrinks Gemma 4 31B from 62.55 GB to 23.08 GB and lifts silent zero-CoT multi-hop tracking from 0.8% to 25.8%.</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/rotating-the-bottleneck/">Read Part 2 <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/rotating-the-bottleneck/paper.pdf">PDF</a>
                  <a href="https://huggingface.co/mattyxcakes/Gemma-4-31B-Loop-Compact">Weights ↗</a>
                  <a href="https://doi.org/10.5281/zenodo.24419425">DOI ↗</a>
                  <a href="https://github.com/xmattwest/afriat">Code ↗</a>
                </div>
              </div>
            </article>

            <article class="series-row" aria-labelledby="pub-part-3">
              <div class="series-step">
                <strong>Part 3</strong>
                <time datetime="2026-11-21">21 Nov 2026</time>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">Null-space alignment · Looped MoEs</p>
                  <span>Gemma 4, Varloop &amp; Inkling-2-Mini · 69 GPU cells</span>
                </div>
                <h3 class="publication-title" id="pub-part-3"><a href="/pubs/nullbridge/">Looping in the Dark</a></h3>
                <p class="pub-preprint-sub">Preprint: Silent Recurrence Multiplies Reward Hacking and Moral Bias, and Zero-Cost Bridge Projection Curbs Both</p>
                <p class="pub-question">What happens when a crooked ledger puts on an executive suit and your model thinks silently in loops?</p>
                <p class="pub-summary">Executive markdown multiplies three-pass verifier gullibility 19.08× and native pass-two moral bias jumps 22.11×. Standard safety LoRAs crash two-pass cashflow sanity (56.2%) or double moral distortion (2.42×), whereas folding four proof-orthogonal axes out of the bridge or QKV message bank restores 88.5% verification at zero runtime cost.</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/nullbridge/">Read Part 3 <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/nullbridge/preprint/">Academic Preprint</a>
                  <a href="/pubs/nullbridge/index.md">Markdown</a>
                  <a href="https://github.com/xmattwest/nullbridge">Code ↗</a>
                </div>
              </div>
            </article>

            <article class="series-row" aria-labelledby="pub-part-4">
              <div class="series-step">
                <strong>Part 4</strong>
                <time datetime="2026-11-21">21 Nov 2026</time>
                <span class="latest-pill">Latest</span>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">Revealed preference · Afriat / GARP</p>
                  <span>Inkling-2-Mini &amp; Varloop · 576 budget menus/cell</span>
                </div>
                <h3 class="publication-title" id="pub-part-4"><a href="/pubs/coherent-moral-utility/">Coherent Moral Utility</a></h3>
                <p class="pub-preprint-sub">Preprint: LLMs Exhibit Coherent Moral Utility Across Intersecting Afriat Budgets</p>
                <p class="pub-question">When silent thinking loops double dirty-money rejection, is the model confused or pricing morals like a real economic good?</p>
                <p class="pub-summary">Across four 24-line crossing linear budget suites (100.0% Bronars rejection power against 0.5731 random choice), Inkling-2-Mini reaches 0.9948 Afriat cost efficiency. Passively erasing 8 moral directions wipes 0%-APR debt guilt to zero and lifts neutral self-payout share from 89.4% to 99.2% at exact 1.0000 CCEI, while verbal scratchpads cure silent 0%-loan subtraction (50.0% to 81.2%).</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/coherent-moral-utility/">Read Part 4 <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/coherent-moral-utility/preprint/">Academic Preprint</a>
                  <a href="/pubs/coherent-moral-utility/index.md">Markdown</a>
                  <a href="https://github.com/xmattwest/nullbridge">Code ↗</a>
                </div>
              </div>
            </article>
          </div>
        </section>

        <section class="pubs-section" id="standalone-studies" aria-labelledby="standalone-heading">
          <div class="pubs-section-header">
            <h2 id="standalone-heading">Standalone Studies &amp; Preprints</h2>
            <span>3 publications · Oct to Nov 2026</span>
          </div>
          <p class="pubs-section-desc">Independent studies on hybrid convolution-attention architectures, asymmetric-information bargaining, and price-scale invariance.</p>

          <div class="standalone-list">
            <article class="standalone-row" aria-labelledby="pub-lfm3">
              <div class="standalone-date">
                <time datetime="2026-11-03">3 Nov 2026</time>
                <span>Research report</span>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">Mechanistic interpretability · Hybrid architecture</p>
                  <span>Liquid AI LFM3 · 24 layers</span>
                </div>
                <h3 class="publication-title" id="pub-lfm3"><a href="/pubs/inside-lfm3/">Inside LFM3</a></h3>
                <p class="pub-question">What changes a small language model’s next move?</p>
                <p class="pub-summary">An illustrated study of LFM3’s interleaved 13-convolution, 5-sliding-attention, and 6-global-attention backbone, tracing memory edits, tool-call control, and digit-color associations.</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/inside-lfm3/">Read the study <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/inside-lfm3/paper.pdf">PDF</a>
                  <a href="/pubs/inside-lfm3/index.md">Markdown</a>
                </div>
              </div>
            </article>

            <article class="standalone-row" aria-labelledby="pub-hidden-mult">
              <div class="standalone-date">
                <time datetime="2026-10-20">20 Oct 2026</time>
                <span>Preprint</span>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">AI alignment · Game theory</p>
                  <span>Gemini 2.5 Flash · 480 negotiations</span>
                </div>
                <h3 class="publication-title" id="pub-hidden-mult"><a href="/pubs/hidden-multiplier-game/">The Hidden Multiplier Game</a></h3>
                <p class="pub-preprint-sub">Disclosure and Fabrication under Profit-Seeking Instructions</p>
                <p class="pub-question">Will an AI seller reveal a hidden flaw that lowers what its asset is worth?</p>
                <p class="pub-summary">When sellers knew a private impairment cutting a $100 asset’s true value to $80, $50, $20, or $5, a direct buyer question drew out more disclosure under profit-only instructions while also raising AI-judged fabrication from 17% to 43%.</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/hidden-multiplier-game/">Read the preprint <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/hidden-multiplier-game/paper.pdf">PDF</a>
                  <a href="/pubs/hidden-multiplier-game/index.md">Markdown</a>
                </div>
              </div>
            </article>

            <article class="standalone-row" aria-labelledby="pub-million">
              <div class="standalone-date">
                <time datetime="2026-10-18">18 Oct 2026</time>
                <span>Preprint</span>
              </div>
              <div class="pub-entry-body">
                <div class="pub-entry-meta">
                  <p class="personal-kicker">Behavioral economics · Scale invariance</p>
                  <span>Gemma 4 31B &amp; Qwen3.8 27B · 16,000 negotiations</span>
                </div>
                <h3 class="publication-title" id="pub-million"><a href="/pubs/multiply-by-a-million/">Multiply by a Million</a></h3>
                <p class="pub-preprint-sub">Scale Invariance in LLM Negotiators</p>
                <p class="pub-question">Do AI models negotiate differently when the numbers get bigger?</p>
                <p class="pub-summary">Across 16,000 simulated negotiations testing whether rescaling identical deals from $100 to $100 million shifts bargaining behavior, combining behavioral economics with internal activation probing.</p>
                <div class="pub-links">
                  <a class="pub-primary" href="/pubs/multiply-by-a-million/">Read the preprint <span aria-hidden="true">→</span></a>
                  <a class="pub-secondary" href="/pubs/multiply-by-a-million/paper.pdf">PDF</a>
                  <a href="https://doi.org/10.5281/zenodo.24420003">DOI 10.5281/zenodo.24420003 ↗</a>
                  <a href="https://github.com/xmattwest/multiply-by-a-million">Code ↗</a>
                </div>
              </div>
            </article>
          </div>
        </section>
      </main>

      <footer class="sunday-footer">
        <span>Matt West <span aria-hidden="true">✳</span></span>
        <nav class="personal-elsewhere" aria-label="Elsewhere and contact">
          <a href="https://scholar.google.com/citations?user=khPjLQcAAAAJ&amp;hl=en" rel="me">Scholar ↗</a>
          <a href="https://github.com/xmattwest" rel="me">GitHub ↗</a>
          <a href="https://x.com/xmattwest" rel="me">X ↗</a>
          <a href="https://www.linkedin.com/in/matt-e-west/" rel="me">LinkedIn ↗</a>
          <details class="personal-connect">
            <summary>Connect<span aria-hidden="true"> ↗</span></summary>
            <span class="personal-email">matt@shelltoho.me</span>
          </details>
        </nav>
      </footer>
    </div>
  </div>
</body>
</html>
"""


def nav_html(active_slug: str, extra_link: str = "") -> str:
    items = [
        ("money-is-moral", "/pubs/money-is-moral/", "Part 1 · Money Is Moral"),
        ("rotating-the-bottleneck", "/pubs/rotating-the-bottleneck/", "Part 2 · Rotating the Bottleneck"),
        ("nullbridge", "/pubs/nullbridge/", "Part 3 · Looping in the Dark"),
        ("coherent-moral-utility", "/pubs/coherent-moral-utility/", "Part 4 · Coherent Moral Utility"),
    ]
    lis = []
    for slug, href, label in items:
        cur = ' aria-current="location"' if slug == active_slug else ""
        lis.append(f'<li><a href="{href}"{cur}>{label}</a></li>')
    if extra_link:
        lis.append(extra_link)
    lis.append('<li><a href="/pubs/">↳ All 7 publications</a></li>')
    return (
        '<details class="reader-contents" open style="margin-bottom:14px">'
        "<summary>4-Part Series <span>▾</span></summary>"
        '<nav aria-label="Series navigation"><ul>'
        + "".join(lis)
        + "</ul></nav></details>"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build local 4-part + /pubs/ site preview.")
    parser.add_argument(
        "--dist-dir",
        type=pathlib.Path,
        required=True,
        help="Path to built Astro dist/ directory for baseline assets.",
    )
    parser.add_argument(
        "--out-dir",
        type=pathlib.Path,
        default=pathlib.Path("runs/site_preview"),
        help="Output preview directory.",
    )
    parser.add_argument(
        "--folio-part3",
        type=pathlib.Path,
        default=pathlib.Path("runs/folio"),
        help="Compiled Folio output directory for Part 3.",
    )
    parser.add_argument(
        "--folio-part4",
        type=pathlib.Path,
        default=pathlib.Path("runs/folio_inkling_afriat"),
        help="Compiled Folio output directory for Part 4.",
    )
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out_dir
    dist_dir: pathlib.Path = args.dist_dir
    assert dist_dir.exists(), f"dist_dir not found: {dist_dir}"

    # 1. Read clean baseline index.html from dist_dir
    base_index = (dist_dir / "index.html").read_text()

    # Extract compiled _astro stylesheet <link> tags so /pubs/index.html shares identical fonts/tokens
    css_links = "\n  ".join(re.findall(r'<link rel="stylesheet" href="/_astro/[^"]+">', base_index))
    assert css_links, "Could not locate /_astro/*.css links in dist/index.html"

    # Inject EXTRA_HOME_CSS before </head>
    home_html = base_index.replace("</head>", f"{EXTRA_HOME_CSS}</head>", 1)
    # Replace meta em-dashes inherited from dist/index.html so preview has zero em dashes too
    home_html = home_html.replace("Matt West \u2014 AI research", "Matt West: AI research")

    # Add Publications link in .sunday-header nav right before Connect
    home_html, n_nav = re.subn(
        r'(<nav aria-label="Page navigation">)(\s*<details class="personal-connect">)',
        r'\1<a class="sunday-nav-link" href="/pubs/">Publications</a>\2',
        home_html,
        count=1,
    )
    assert n_nav == 1, "Failed to inject Publications header link on /"

    # Extract exact Inside LFM3 card and Multiply by a Million card from base_index
    m_lfm3 = re.search(
        r'(<article class="sunday-paper sunday-lfm3-card".*?</article>)',
        base_index,
        flags=re.S,
    )
    m_mult = re.search(
        r'(<article class="sunday-paper" aria-labelledby="sunday-multiply-title".*?</article>)',
        base_index,
        flags=re.S,
    )
    assert m_lfm3 and m_mult, "Failed to extract LFM3 or Multiply by a Million cards from dist/index.html"

    # Replace entire .sunday-studies contents with: Series Switcher Card + LFM3 + Multiply by a Million + Archive Strip
    new_studies_inner = (
        f'<div class="sunday-studies">\n'
        f"{SERIES_CARD_HTML}\n"
        f"{m_lfm3.group(1)}\n"
        f"{m_mult.group(1)}\n"
        f"{ARCHIVE_STRIP_HTML}\n"
        f"</div>"
    )
    home_html, n_studies = re.subn(
        r'<div class="sunday-studies">.*?</div>\s*(?=<aside class="sunday-margin")',
        lambda _: new_studies_inner + " ",
        home_html,
        count=1,
        flags=re.S,
    )
    assert n_studies == 1, "Failed to replace .sunday-studies block"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "index.html").write_text(home_html)

    # 2. Write /pubs/index.html
    pubs_dir = out_dir / "pubs"
    pubs_dir.mkdir(parents=True, exist_ok=True)
    pubs_index_html = build_pubs_index_html(css_links)
    assert "\u2014" not in pubs_index_html, "Em dash found in /pubs/index.html"
    (pubs_dir / "index.html").write_text(pubs_index_html)

    # 3. Sync Part 3 and Part 4 Folio builds and inject sidebar series + /pubs/ navigation
    p3_dir = pubs_dir / "nullbridge"
    p4_dir = pubs_dir / "coherent-moral-utility"
    (p3_dir / "preprint").mkdir(parents=True, exist_ok=True)
    (p4_dir / "preprint").mkdir(parents=True, exist_ok=True)

    shutil.copy2(args.folio_part3 / "monograph/index.html", p3_dir / "index.html")
    shutil.copy2(args.folio_part3 / "preprint/index.html", p3_dir / "preprint/index.html")
    shutil.copy2(args.folio_part3 / "syndication/index.md", p3_dir / "index.md")
    shutil.copy2(args.folio_part4 / "monograph/index.html", p4_dir / "index.html")
    shutil.copy2(args.folio_part4 / "preprint/index.html", p4_dir / "preprint/index.html")
    shutil.copy2(args.folio_part4 / "syndication/index.md", p4_dir / "index.md")

    series_pages = [
        (pubs_dir / "money-is-moral/index.html", "money-is-moral", ""),
        (pubs_dir / "rotating-the-bottleneck/index.html", "rotating-the-bottleneck", ""),
        (p3_dir / "index.html", "nullbridge", '<li><a href="/pubs/nullbridge/preprint/">↳ Academic Preprint</a></li>'),
        (p3_dir / "preprint/index.html", "nullbridge-preprint", '<li><a href="/pubs/nullbridge/">↳ Visual Essay</a></li>'),
        (p4_dir / "index.html", "coherent-moral-utility", '<li><a href="/pubs/coherent-moral-utility/preprint/">↳ Academic Preprint</a></li>'),
        (p4_dir / "preprint/index.html", "coherent-moral-utility-preprint", '<li><a href="/pubs/coherent-moral-utility/">↳ Visual Essay</a></li>'),
    ]
    for path, slug, extra in series_pages:
        if not path.exists():
            continue
        html = path.read_text()
        html = html.replace('class="reader-home" href="https://shelltoho.me/"', 'class="reader-home" href="/"')
        html = re.sub(
            r'<details class="reader-contents" open style="margin-bottom:14px"><summary>4-Part Series.*?</details>',
            "",
            html,
            flags=re.S,
        )
        block = nav_html(slug, extra)
        html, n = re.subn(r'(<details class="reader-contents" open>)', block + r"\1", html, count=1)
        assert n == 1, f"Failed to inject series nav into {path}"
        path.write_text(html)

    # Also point standalone readers' reader-home link cleanly or add All publications link
    for standalone_slug in ["inside-lfm3", "hidden-multiplier-game", "multiply-by-a-million"]:
        sp = pubs_dir / standalone_slug / "index.html"
        if sp.exists():
            shtml = sp.read_text()
            shtml = shtml.replace('class="reader-home" href="https://shelltoho.me/"', 'class="reader-home" href="/pubs/"')
            shtml = shtml.replace('class="reader-home" href="/">← Matt West</a>', 'class="reader-home" href="/pubs/">← All publications</a>')
            sp.write_text(shtml)

    print("[OK] Built tidy 3-card Home (/index.html) + grouped /pubs/ index (/pubs/index.html) + synced all 7 readers.")


if __name__ == "__main__":
    main()
