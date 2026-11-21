# Nullbridge

**Nullbridge** (Part 3 of *Money Is Moral* and *Rotating the Bottleneck*) tests algebraic, zero-runtime-cost alignment inside silent multi-pass looped transformers against standard behavioral de-biasing LoRAs across `69` GPU evaluation cells spanning both retrofitted Procrustes loops (`Gemma-4-Loop-Compact`, `28` cells) and native pretrained looped MoEs (`thinkingmachines/TML-d24-varloop-research` and flagship `thinkingmachines/Inkling-2-Mini`, `41` cells):

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

## Repository Layout

- `src/nullbridge/`: Closed-form Procrustes and message-bank null-space folding (`algebra.py`), zero-leakage benchmark suite (`benchmarks.py`), synthetic loop verification (`mock_loop.py`), NDIF harness (`ndif_harness.py`), `Gemma-4-Loop-Compact` loader and Track B pipeline (`gemma4_compact.py`, `track_b.py`), native `Inkling` / `Varloop` Track A pipeline (`track_a.py`), and deterministic statistical analysis (`analysis.py`).
- `results/phase0_audit/`: Locked Phase 0 mathematical, benchmark-disjointness, and `K = 1..8` synthetic message-bank audit JSON.
- `results/phase3_gemma4_nullbridge/`: Locked 28-cell GPU evaluation (`track_b_results.json`), derived statistical analysis (`statistical_analysis.json`), and synthesis (`synthesis.md`).
- `results/phase4_native_inkling_nullbridge/`: Locked 41-cell GPU evaluation across `TML-d24-varloop-research` and `Inkling-2-Mini` (`track_a_results.json`), derived statistical analysis (`track_a_analysis.json`), and synthesis (`synthesis.md`).
- `results/phase5_inkling_afriat_garp/`: Locked 5-block Afriat / GARP and *Money Is Moral* GPU evaluation (`afriat_inkling_results.json`), derived statistical analysis (`afriat_inkling_analysis.json`), and synthesis (`synthesis.md`).
- `folio/nullbridge.toml` & `folio/inkling_afriat.toml`: Claim-verified Dawn State manuscript manifests compiled across `5` publication targets each.
- `design-system/`: Embedded Dawn State / Warm Precision visual and editorial specification.
- `runs/` *(gitignored)*: Raw pod artifacts, rendered SVG bundles, and compiled Folio HTML/Markdown outputs.

## Reproducibility & Verification

```bash
uv sync --extra dev
uv run pytest -q
uv run nullbridge audit --audit-only
uv run nullbridge analyze
uv run nullbridge guard

# Verify and compile all 5 Dawn State publication surfaces with Folio:
folio verify folio/nullbridge.toml
folio build folio/nullbridge.toml --out runs/folio
```

## License

MIT
