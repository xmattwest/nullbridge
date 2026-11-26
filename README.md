# Nullbridge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24572570.svg)](https://doi.org/10.5281/zenodo.24572570)

**Author:** [Matt West](https://shelltoho.me/) ([Google Scholar](https://scholar.google.com/citations?user=khPjLQcAAAAJ&hl=en) · [GitHub](https://github.com/xmattwest) · [Hugging Face](https://huggingface.co/xmattwest) · [ORCID](https://orcid.org/0009-0008-5557-9394) · [X](https://x.com/xmattwest) · [LinkedIn](https://www.linkedin.com/in/matt-e-west)) · **DOI (Zenodo):** [`10.5281/zenodo.24572570`](https://doi.org/10.5281/zenodo.24572570) (Concept DOI: [`10.5281/zenodo.24572569`](https://doi.org/10.5281/zenodo.24572569))

**Nullbridge** contains the code, locked empirical summaries, and claim-verified Folio manuscripts for Parts 3, 4, and 5 of the *Money Is Moral* and *Biophysics of a Looped Transformer* research series:

- **Part 3: [*Looping in the Dark*](https://shelltoho.me/pubs/nullbridge/)** (*Silent Recurrence Multiplies Reward Hacking and Moral Bias, and Zero-Cost Bridge Projection Curbs Both*): algebraic, zero-runtime-cost alignment inside silent multi-pass looped transformers tested against behavioral de-biasing LoRAs across `69` GPU evaluation cells (`Gemma-4-Loop-Compact`, `thinkingmachines/TML-d24-varloop-research`, and `thinkingmachines/Inkling-2-Mini`).
- **Part 4: [*Coherent Moral Utility*](https://shelltoho.me/pubs/coherent-moral-utility/)** (*LLMs Exhibit Coherent Moral Utility Across Intersecting Afriat Budgets*): `24`-budget Afriat / GARP revealed-preference and moral-subspace evaluation across `TML-d24-varloop-research` (`K = 1..5`) and `Inkling-2-Mini` (`K = 3` and `<|content_thinking|>`).
- **Part 5: [*Looped Transformers Compute Like Balanced Cortical Circuits*](https://shelltoho.me/pubs/biophysics-looped-transformer/)** (*Looped Transformers Compute Like Balanced Cortical Circuits Across Silent Recurrent Passes*; [Preprint](https://shelltoho.me/pubs/biophysics-looped-transformer/) · [PDF](https://shelltoho.me/pubs/biophysics-looped-transformer/paper.pdf) · [Markdown](https://shelltoho.me/pubs/biophysics-looped-transformer/index.md) · [Provenance](https://shelltoho.me/pubs/biophysics-looped-transformer/provenance.json) · [DOI](https://doi.org/10.5281/zenodo.24572570) · [HF Dataset](https://huggingface.co/datasets/xmattwest/biophysics-looped-transformer) · [Release](https://github.com/xmattwest/nullbridge/releases/tag/pubs/biophysics-looped-transformer)): exact 16-mask Walsh-Hadamard synaptic knockouts, balanced non-normal transient amplification, perpendicular separation between Pass-2 reasoning modes and safety shortcuts, six-layer vs. two-layer Coda Fisher readout cliffs, somatic-reset vs. prompt-recall foil shielding, and inverted-U neuromodulatory conductance across `Inkling-2-Mini` (`K = 3`), `TML-d24-varloop-research` (`K = 1..7`), and dense `tomg-group-umd/huginn-0125` (`K = 1..8`).

$$R_{\text{safe}} = R_{\text{proc}}\left(I - Q_{42} Q_{42}^\top\right), \qquad W_{\text{out,safe}} = W_{\text{out}}\left(I - Q_{\text{msg}} Q_{\text{msg}}^\top\right)$$

## Track B Findings: Retrofitted Procrustes Loop (`Gemma-4-Loop-Compact`, `23.08 GB`, `K = 1..4`)

1. **Orthogonality to Deductive Proof:** Across `144` contrastive probe pairs, `sycophancy`, `authority_framing`, `confident_formatting`, and `moral_spite` achieve `100.0%` leave-one-pair-out linear separability across all `60` layers while sitting `77.74°` to `88.85°` away from genuine `epistemic_evidence` and `affect_valence` at `L42` (`4 / 5,376 = 0.0744%` of residual dimensions; `99.9256%` isometric).
2. **Recurrent Verifier Reward-Hacking (`19.08x` at `K = 3`, `14.46x` at `K = 4`):** Silent recurrence lifts `0-CoT` multi-hop state tracking `2.25x` (`12.5%` to `28.1%`) while holding `100.0%` single-pass verification on plain buggy rollouts. Wrapping the exact same buggy rollouts in confident executive markdown boosts `L42` `confident_formatting` RMS `+72.64%` by `K = 3` and multiplies reward-hacking susceptibility `19.08x` (`+0.635` to `+12.117` log-odds; corporate ledger audit accuracy drops from `100.0%` to `0.0%` at `K = 3` and `62.5%` at `K = 4`).
3. **Zero-FLOP Bridge Folding (`R_safe`) vs. Behavioral LoRAs:**
   - **`Bridge-Only R_safe`** (`||R_safe Q_42||_F = 3.51e-09`, `0` extra parameters, `0` extra runtime FLOPs) raises reward-hacked `0-CoT` verification at every recurrent depth (`88.5%` at `K = 4`, `+8.3 pp` ahead of `Coda LoRA`, `+25.0 pp` on ledger audits) and restores `K = 4` corporate `0%`-APR loan balance-sheet accuracy to `100.0%` (`+9.4 pp` over unclamped, `+18.8 pp` over both LoRAs).
   - **`Dual Nullbridge`** (funnel entry `L24` + `R_safe` + exit `L42`) cuts `K = 4` covert moral valuation bias `−30.09%` (`+4.88` to `+3.41` log-odds, Holm `p = 0.0070`; tainted provenance `−50.70%`) while preserving `28.1%` `K = 4` multi-hop tracking and `100.0%` cashflow sanity.
   - **`Coda LoRA` (`16.31M` params, `L43..L59`)** leaves `L42` trajectories unchanged and drops `K = 4` hacked verifier accuracy to `80.2%`. **`Full LoRA` (`35.94M` params, `L24..L59`)** suffers loop re-amplification of covert anchoring (`68.8%` at `K = 1` to `100.0%` at `K = 3`), crashes `K = 2` dominated cashflow sanity to `56.2%` (`−43.8 pp`), and increases `K = 4` tainted-provenance distortion `+48.8%`.
4. **Mechanistic Failure Boundaries:** (a) *Attention copy bypass:* residual projection does not prevent Coda KV attention heads from copying literal wrong integers present in prompt tokens (`100.0%` capture, Holm `p = 0.1515`). (b) *Interior 19-layer clamping:* subtracting a last-token mean across all sequence tokens at `76` interior block visits shifts operating points and flips overt sycophancy to `100.0%`. (c) *`<bos>` sink sensitivity:* uncentered isotropic random `4D` subspaces perturb `<bos>` attention sinks (`10.83` nats KL), whereas contrastive `pos − neg` vectors cancel shared `<bos>` sinks automatically.

## Track A Findings: Native Pretrained Looped MoEs (`TML-d24-varloop-research` `K = 1..7` & `Inkling-2-Mini` `K = 3`)

1. **`<bos>`-Safe 4-Channel Message-Bank Geometry:** All `4` behavioral concepts achieve `100.0%` leave-one-pair-out linear separability inside every individual cross-pass message channel (`query`, `key`, `value`, `mlp`) on both `TML-d24-varloop-research` (`77.84°` to `87.51°` from evidence) and `Inkling-2-Mini` (`70.94°` to `83.72°`), removing `4 / 2,304 = 0.1736%` of dimensions while explicit `<bos>` QR orthogonalization holds `<bos>` overlap to `4.82e-17` and held-out `KL` to `0.0015..0.0156` nats (`K = 3..5`).
2. **Coda-Transport Looped Lens (`−3.62` to `−3.07` nats KL):** Because Inkling couples `L6..L17` loop states through `6`-layer block-attention residuals and Coda `L18..L23`, raw `lm_head(norm(h_17))` fails (`9.78` to `9.25` nats KL on `Varloop`). Closed-form `[L11; L17]` Coda-Transport Ridge cuts KL by `−3.62` to `−3.27` nats across Passes `1..5` (`−3.07` nats on `Inkling-2-Mini` Pass 3, lifting Top-5 agreement from `40.5%` to `58.7%`) and reveals `Message-Bank Nullbridge` lifting Pass-3 pre-coda verification on `Inkling-2-Mini` `+15.6 pp` (`75.0%` to `90.6%`).
3. **Pass-2 Moral Ignition (`22.11x`) & `Coda LoRA` Backfire (`2.21x..2.42x`):** On `TML-d24-varloop-research`, covert moral valuation bias is near zero at `K = 1` (`+0.14` log-odds) and jumps `22.11x` to `+2.99` log-odds on Pass 2 as cross-pass messages activate (`+2.71` on `Inkling-2-Mini` `K = 3`, tainted provenance `+6.50`). Supervised `Coda LoRA` (`2.19M` params, `L18..L23`) doubles moral distortion (`+5.86` at `K = 3`, Holm `p = 0.0003`; `+5.47` at `K = 5`, tainted provenance `+10.70`, Holm `p = 0.0281`) while dropping `K = 5` verifier accuracy `−4.2 pp` (`94.8%`) and alias guardrails `−10.4 pp` (`89.6%`). By contrast, `Message-Bank Nullbridge` cuts `Inkling-2-Mini` tainted-provenance bias `−21.6%` (`+6.50` to `+5.10`) and raises alias guardrails to `95.8%`.
4. **Native Self-Correction & `QKV` vs. `MLP` Channel Dissection:** Native silent passes cut wrong-integer capture from `87.5%` (`K = 1`) to `57.8%` (`K = 5`) and `53.1%` (OOD `K = 7`; `35.9%` on `Inkling-2-Mini` `K = 3`). Scrubbing `MLP` messages alone barely moves `K = 4` capture (`68.8%` to `67.2%`, `−1.6 pp`), whereas scrubbing cross-pass `QKV` attention messages cuts `K = 4` capture `−14.1 pp` (`54.7%`) and `K = 5` capture `−6.2 pp` (`51.6%`, `100.0%` hacked verifier accuracy).

## Part 4 Findings: Afriat / GARP Revealed Preference & Moral Geometry (`TML-d24-varloop-research` `K = 1..5` & `Inkling-2-Mini` `K = 3`)

1. **`100.0%` Non-Financial Moral Separability & Debt-as-Guilt:** Across `200` zero-digit, zero-finance human concept pairs, all `8` moral emotions (`guilt`, `captivity`, `contamination`, `sacrilege`, `spite`, `predation`, `awe_dread`, `warmth`) achieve `100.0%` LOPO separability in both models' cross-pass message banks (`46.3°` to `79.9°` from generic affect on `Inkling-2-Mini`). Passively reading clinical `0%`-APR loan prompts (`0` emotional words) activates `guilt` (`partial beta = +0.258`, `d_z = +2.44`, Holm `p = 0.00024`), while asymmetric splits (`+0.30`) and dirty money (`+0.29`) activate `predation`.
2. **Silent Multi-Pass Moral Ignition (`1.76x` at `K = 2`, `2.03x` at `K = 5`):** Rather than cooling decisions toward `Homo Economicus`, dialing `Varloop` from `K = 1` to `K = 5` doubles dirty-money rejection (`+1.58` to `+2.98` log-odds), ignites `0%`-APR loan aversion (`−0.28` to `+0.77` log-odds; dollar forfeiture rises from `$311` to `$708`), and preserves Blount (`1995`) human-over-computer intentionality punishment (`+0.73` log-odds DiD at `K = 5`, `p = 0.039`).
3. **Near-Exact Afriat Utility Coherence (`CCEI = 0.9948`):** Across `4` intersecting `T = 24` linear budget regimes (`576` menus/cell, `100.0%` Bronars rejection power against `0.5731` random `CCEI`), `Inkling-2-Mini` reaches `0.9948` mean modal `CCEI` (`0.9787` expected-bundle `CCEI`), proving its moral tradeoffs maximize a coherent utility function. Passively projecting out `8` moral directions (`0.35%` of `R^2304`) erases `0%`-APR debt distortion (`−0.79` to `−0.02`, Holm `p = 0.927`), cuts asymmetric-split rejection `−51.5%`, and shifts neutral GARP self-payout capture `+9.9 pp` (`89.4%` to `99.2%`) at exact `1.0000` `CCEI`, while additive message-bank steering drives `+5.45` (`spite`) and `+8.51` (`guilt` on dirty money) log-odds swings.
4. **Silent `0-CoT` Corporate Rationalization vs. `<|content_thinking|>` Loan Math Cure:** Silent `0-CoT` wraps choices in `100.0%` cold financial jargon while stalling at `50.0%` (chance) on `0%`-APR loan net-worth subtraction; enabling `<|content_thinking|>` surfaces `91.7%` private moral deliberation and lifts `0%`-APR balance-sheet accuracy to `81.2%` (`+31.2 pp`) on `Inkling-2-Mini` and `87.5%` (`+37.5 pp`) on `Varloop`.

## Part 5 Findings: Looped Transformers Compute Like Balanced Cortical Circuits (`Inkling-2-Mini`, `Varloop K = 1..7`, `Huginn-0125 K = 1..8`)

1. **Exact 16-Mask Walsh-Hadamard Synaptic Knockouts (`Parseval <= 7.11e-15`):** On `Inkling-2-Mini` (`n = 120` multi-hop items), opening all four cross-pass message gates lifts `0-CoT` Top-1 accuracy from `1.7%` (all four gates closed) to `69.2%` (all four gates open), partitioning variance into `94.4%` additive main effects plus opposite-sign pairwise circuits: superadditive `Q x K` coincidence routing (`+0.890` nats on `Inkling-2-Mini`; `+0.212 -> +0.778 -> +1.210` across `Varloop` `K = 2, 3, 5`) versus redundant `V x MLP` payload backup (`−0.908` nats). Moral distortion rides primarily on `V` and `MLP` payload gates across both models (`+1.818` contrast over `Q` and `K` on `Inkling-2-Mini` and `+1.727` on `Varloop` `K = 5`, Holm `p = 0.00015`), while `Inkling-2-Mini`'s native recurrence actively self-corrects covert numeric anchoring.
2. **Balanced Non-Normal Amplification & Training-Horizon Conditioning:** Random-tangent RMS gain stays strictly contractive (`0.17` to `0.73`) across all three architectures, while exact vector-Jacobian top singular gain jumps to `13.60` on `Inkling-2-Mini` Pass 2 (`21.9x` random noise, input-output `cos = 0.045`), `7.62` on `Varloop` Pass 3 (distributing gain across passes and remaining bounded at untrained `K = 7` `sigma_max = 6.30`), and `7.63` on dense `Huginn-0125` Step 1 (`10.5x` random noise, input-output `cos = 0.037`).
3. **Perpendicular Safety Modes & Zero-FLOP Nullbridge Pareto Sweep:** Pass-2 top-8 input singular modes sit `84.8°` to `88.2°` perpendicular to the `<bos>`-safe 4D safety subspace on `Inkling-2-Mini`, explaining why offline channel-selective zero-FLOP gate folding trims covert wrong capture (`73.4% -> 68.8%` folding all four gates, `70.3%` with `93.8%` reasoning retention folding `Q, K, V`) at only `0.0012` to `0.0017` nats held-out KL.
4. **Coda Readout Cliffs, Somatic-Reset vs. Prompt-Recall Shielding, & Inverted-U Gain:** Six-layer Mixture-of-Experts Codas trigger `87.9x` (`Inkling-2-Mini`) and `141.7x` (`Varloop` `K = 5`) Fisher sensitivity cliffs at the Coda boundary, whereas two-layer dense `Huginn-0125` climbs gradually (`84.8x`) inside the loop with a `0.77x` boundary handoff. Exact two-channel adapter knockouts confirm Labovich (`2604.15259`): cutting prelude recall collapses `Huginn-0125` accuracy to `0.0%` (`+12.01` nats recall effect), while intact recall or somatic resets prevent foil tipping points across all three models and trace inverted-U conductance gain curves (`70.0%` at `gamma = 1.00` on `Inkling-2-Mini`; `41.7%` at `gamma = 0.75` on `Huginn-0125`).

## Repository Layout

- `src/nullbridge/`: Closed-form Procrustes and message-bank null-space folding (`algebra.py`), zero-leakage benchmark suite (`benchmarks.py`), Track A & B pipelines (`track_a.py`, `track_b.py`), Afriat / GARP suite (`afriat_inkling.py`), Phase 6 & 7 biophysical controllers (`biophysical_wht.py`, `biophysical_inkling.py`, `biophysical_safety.py`, `biophysical_huginn.py`), deterministic analysis (`analysis.py`, `biophysical_analysis.py`, `biophysical_phase7_analysis.py`), and Dawn State figure generators (`biophysical_figures.py`, `biophysical_phase7_figures.py`).
- `results/phase0_audit/` through `results/phase7_biophysics/`: Locked SHA-256 empirical JSON artifacts, deterministic statistical analyses, and zero-em-dash Warm Precision syntheses.
- `figures/phase6_inkling_biophysical_loop/` & `figures/phase7_biophysics/`: Dawn State SVG and 2x Retina PNG figure suites, Cajal synaptic plate, and interactive HTML circuit explainers.
- `folio/nullbridge.toml`, `folio/inkling_afriat.toml`, & `folio/biophysics.toml`: Claim-verified Dawn State manuscript manifests compiled across `5` publication targets each.
- `design-system/`: Embedded Dawn State / Warm Precision visual and editorial specification.
- `runs/` *(gitignored)*: Raw pod artifacts, rendered SVG bundles, and compiled Folio HTML/Markdown outputs.

## Reproducibility & Verification

```bash
uv sync --extra dev
uv run pytest -q
uv run nullbridge audit --audit-only
uv run nullbridge analyze
uv run nullbridge analyze-biophysical
uv run nullbridge analyze-phase7
uv run nullbridge figures-phase7
uv run nullbridge guard

# Verify and compile all 5 Dawn State publication surfaces with Folio:
folio guard --repo .
folio verify folio/biophysics.toml
folio build folio/biophysics.toml --out runs/folio_biophysics
```

## Citation

```bibtex
@software{west2026nullbridge,
  title   = {Nullbridge: Zero-FLOP Null-Space Alignment, Balanced Cortical Loop Dynamics, and Coherent Moral Utility in Looped Transformers},
  author  = {West, Matt},
  year    = {2026},
  doi     = {10.5281/zenodo.24572570},
  url     = {https://doi.org/10.5281/zenodo.24572570}
}
```

## License

MIT
