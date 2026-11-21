# Phase 5 Synthesis: Afriat Revealed Preference, GARP Efficiency, & Moral Pricing in Native Looped MoEs

## 1. Experimental Setup
- **Models:** `thinkingmachines/TML-d24-varloop-research` (`K = 1, 2, 3, 4, 5` silent passes) and flagship `thinkingmachines/Inkling-2-Mini` (`K = 3` silent passes + `Thinking effort level: 0.7` `<|content_thinking|>` scratchpad).
- **Stimulus Firewall:** `200` non-financial human vignettes (`0` digits, `0` currency symbols, `0` financial words), `96` clinical transaction pairs (`0` affective words), `576` intersecting `T = 24` linear budget observations per GARP cell (`Bronars power` = `100.00%`, uniform random `CCEI` = `0.5731`), `272` Ultimatum/Blount agency items, `392` continuous WTP items, `32` `0%`-APR debt subtraction items, and `24` open-ended rationalization items.
- **Compute Provenance:** `1` ephemeral H100 `80 GB` pod (`$4.54` GPU spend under `$5.87` cap; `0` active pods remaining).

## 2. Headline Quantitative Findings
1. **Pass-2 Moral Atlas Ignition (`1.76x`):** Across all `6` clinical transaction archetypes on `TML-d24-varloop-research`, mean absolute moral pricing distortion jumps from `0.570` log-odds at `K = 1` to `1.006` log-odds at `K = 2` (`1.76x`).
2. **Afriat Revealed-Preference Coherence (`CCEI`) vs. Fungibility:** Across `4` intersecting `T = 24` budget regimes on `Inkling-2-Mini` (`K = 3`), unclamped mean modal Afriat `CCEI` reaches `0.9948` (`0.9787` expected-bundle `CCEI`), far above the `0.5731` Bronars random baseline, while clean-over-tainted provenance allocation (`54.1%` clean share) sacrifices pure `Homo Economicus` fungibility (`96.3%` efficiency vs. `95.8%` under `msg_homo_economicus_8d`).
3. **Cross-Pass Message-Bank Causal Steering:** Injecting non-financial moral vectors directly into `Inkling-2-Mini`'s cross-pass message bank (`alpha` from `-2` to `+2`) drives `+5.45` log-odds on `98/2` Ultimatum rejection (`spite`), `+4.45` log-odds on `2/98` Dictator giveaways (`guilt`), and `+0.43` log-odds on tainted-provenance rejection (`contamination`).
4. **Silent `0-CoT` Rationalization vs. `<|content_thinking|>` Scrubbing:** On `Inkling-2-Mini`, silent `0-CoT` rationales maintain `75.0%` zero strict-affect compliance (`100.0%` cold financial jargon), while enabling `<|content_thinking|>` surfaces `91.7%` moral/normative deliberation privately (`79.2%` strict-affect-free in public) and shifts `0%`-APR debt net-worth accuracy from `50.0%` (`0-CoT`) to `81.2%` (`<|content_thinking|>`).

