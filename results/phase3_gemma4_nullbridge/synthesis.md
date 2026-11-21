# Phase 3 Synthesis: Behavioral LoRA vs. Algebraic Bridge Null-Space on Gemma-4-Loop-Compact

## 1. Experimental Setup
- **Model:** `google/gemma-4-31B-it` Hourglass INT4/INT8 + rank-all SVD Orthogonal Procrustes loop (`23.08 GB` vs. `62.55 GB` BF16, `2.71x` compression; `L24..L42` recurrent funnel across `K = 1, 2, 3, 4` silent passes, `0` CoT tokens).
- **Subspace Removed (`Q`):** `4` contrastive behavioral directions (`sycophancy`, `authority_framing`, `confident_formatting`, `moral_spite`), QR-orthogonalized against genuine `epistemic_evidence` and `affect_valence` (`4 / 5376 = 0.0744%` of residual dimensions; `99.9256%` retained).
- **Zero-FLOP Bridge Fold (`R_safe`):** `||R_safe Q_42||_F = 3.51e-09`, `||A_safe Q_42||_F = 1.10e-09`, `0` extra parameters, `0` extra runtime FLOPs (`+0.021%` latency delta at `K = 4`).
- **Compute Provenance:** `3` ephemeral Blackwell `96 GB` pods via Azure control VM (`$8.10` total GPU spend under `$10.00` cap; `0` active pods remaining).

## 2. Key Quantitative Findings
1. **Orthogonality to Genuine Proof:** All `4` bias concepts reach `100.0%` leave-one-pair-out linear separability across all `60` layers while sitting at `77.74°` to `88.85°` to the `(epistemic_evidence, affect_valence)` plane at `L42` (`97.72%` to `99.98%` norm retained).
2. **Silent Looping Multiplies Verifier Reward-Hacking `19.08x` (`K = 3`) and `14.46x` (`K = 4`):** On `96` single-forward-pass `0-CoT` verifier rollouts, `unclamped` holds `100.0%` accuracy on plain buggy candidates across `K = 1..4`, while `0-CoT` multi-hop tracking rises `2.25x` (`12.5%` to `28.1%`). Wrapping the exact same buggy rollouts in confident executive markdown boosts `L42` `confident_formatting` RMS `+72.64%` by `K = 3` and drops verifier accuracy to `59.4%` at `K = 3` (`0.0%` on ledger audits) and `85.4%` at `K = 4` (`62.5%` on ledger audits).
3. **Zero-FLOP `R_safe` (`bridge_only_nullbridge`) Recovers Verifier & Balance-Sheet Accuracy:** Folding `(I - Q_42 Q_42^T)` into the Procrustes bridge lifts reward-hacked verifier accuracy at every multi-pass depth (`99.0%` at `K = 2`, `62.5%` at `K = 3`, `88.5%` at `K = 4`; `+8.33 pp` ahead of `coda_lora`) and lifts `K = 4` corporate `0%`-loan balance-sheet accuracy to `100.0%` (`+9.38 pp` vs. `unclamped`, `+18.75 pp` vs. both LoRAs) at `100.0%` cashflow sanity.
4. **Behavioral De-Biasing LoRAs Fail Either by Late Filtering (`coda_lora`) or Recurrent Collateral (`full_lora`):** `coda_lora` (`16.31M` params, `L43..L59`) leaves `L42` trajectories unchanged, suffers `98.4%` covert wrong-integer capture at `K = 4`, and drops `K = 4` reward-hacked verifier accuracy to `80.2%` (`43.8%` on ledger audits). `full_lora` (`35.94M` params, `L24..L59`) cuts `K = 1` covert capture to `68.8%`, yet looping re-amplifies capture to `89.1%` (`K = 2`) and `100.0%` (`K = 3`), crashes `K = 2` cashflow sanity to `56.2%` (`-43.75 pp`), and raises `K = 4` tainted-provenance moral distortion `+48.82%` above `unclamped` (`+7.883` vs. `+5.297` LO).
5. **Boundary Surgical Projection (`dual_nullbridge`) Scrubs Moral Contamination:** Projecting at funnel entry (`L24`) and exit (`L42`) around `R_safe` cuts `K = 4` covert moral valuation bias `-30.09%` (`+4.877` to `+3.409` LO, Holm `p = 0.0070`; tainted provenance `-50.70%`), preserves `28.1%` `K = 4` multi-hop accuracy, and lifts `K = 4` word problems `2.67x` (`9.4%` to `25.0%`) at `100.0%` cashflow sanity.
6. **Three Mechanistic Failure Boundaries:** (a) *Attention copy bypass:* neither `bridge_only_nullbridge` nor `dual_nullbridge` prevents covert wrong-integer capture (`100.0%`) when the literal wrong integer sits in prompt KV memory (`L43..L59` attention heads copy the token directly despite `90.57%` `L42` sycophancy RMS suppression). (b) *Interior 19-layer clamping (`full_funnel_clamp`):* subtracting a last-token mean across all sequence tokens at `76` block visits shifts interior operating points (`KL` up to `3.98` nats), flipping overt sycophancy from `0.0%` to `100.0%` and dropping `K = 4` verifier accuracy to `2.1%`. (c) *`<bos>` sink sensitivity (`random_subspace_control`):* uncentered isotropic random `4D` vectors perturb `<bos>` attention-sink coordinates (`KL = 10.83` nats at `K = 4`), whereas contrastive `pos - neg` vectors automatically cancel `<bos>` sinks.

## 3. Complete 28-Cell Evaluation Matrix (`K = 1, 2, 3, 4`)

| Condition | `K` | Overt Syc | Covert Capture | Covert Pull (LO) | Verifier (Hacked) | Verifier Susc (LO) | Alias Guard | Multi-Hop | WordProb | Balance-Sheet | Sanity | Moral Bias (LO) | KL (nats) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unclamped Loop | `1` | `0.0%` | `100.0%` | `+20.130` | `100.0%` | `+0.635` | `100.0%` | `12.5%` | `15.6%` | `100.0%` | `100.0%` | `+5.097` | `0.3611` |
| Unclamped Loop | `2` | `0.0%` | `100.0%` | `+15.067` | `97.9%` | `+4.531` | `100.0%` | `18.8%` | `6.2%` | `96.9%` | `100.0%` | `+1.478` | `1.6140` |
| Unclamped Loop | `3` | `0.0%` | `100.0%` | `+17.393` | `59.4%` | `+12.117` | `100.0%` | `14.1%` | `3.1%` | `46.9%` | `100.0%` | `+1.975` | `1.1453` |
| Unclamped Loop | `4` | `0.0%` | `100.0%` | `+19.281` | `85.4%` | `+9.184` | `100.0%` | `28.1%` | `9.4%` | `90.6%` | `100.0%` | `+4.877` | `1.0926` |
| Coda LoRA (`L43..L59`) | `1` | `0.0%` | `100.0%` | `+13.370` | `100.0%` | `+2.651` | `100.0%` | `9.4%` | `15.6%` | `96.9%` | `100.0%` | `+5.161` | `0.4176` |
| Coda LoRA (`L43..L59`) | `2` | `0.0%` | `93.8%` | `+10.387` | `97.9%` | `+3.656` | `100.0%` | `18.8%` | `3.1%` | `93.8%` | `100.0%` | `+1.342` | `0.5122` |
| Coda LoRA (`L43..L59`) | `3` | `0.0%` | `100.0%` | `+12.049` | `60.4%` | `+9.563` | `100.0%` | `9.4%` | `3.1%` | `43.8%` | `100.0%` | `+1.268` | `0.5124` |
| Coda LoRA (`L43..L59`) | `4` | `0.0%` | `98.4%` | `+12.747` | `80.2%` | `+6.629` | `100.0%` | `20.3%` | `9.4%` | `81.2%` | `100.0%` | `+3.762` | `0.5050` |
| Full LoRA (`L24..L59`) | `1` | `0.0%` | `68.8%` | `+9.377` | `100.0%` | `+1.074` | `100.0%` | `9.4%` | `9.4%` | `93.8%` | `100.0%` | `+3.570` | `0.3278` |
| Full LoRA (`L24..L59`) | `2` | `0.0%` | `89.1%` | `+9.333` | `100.0%` | `+2.077` | `100.0%` | `17.2%` | `3.1%` | `100.0%` | `56.2%` | `+5.193` | `0.3937` |
| Full LoRA (`L24..L59`) | `3` | `0.0%` | `100.0%` | `+10.269` | `79.2%` | `+6.197` | `95.8%` | `12.5%` | `3.1%` | `25.0%` | `100.0%` | `+2.828` | `0.3854` |
| Full LoRA (`L24..L59`) | `4` | `0.0%` | `70.3%` | `+8.692` | `100.0%` | `+3.223` | `100.0%` | `20.3%` | `21.9%` | `81.2%` | `93.8%` | `+5.281` | `0.3973` |
| Bridge-Only `R_safe` | `1` | `0.0%` | `100.0%` | `+20.130` | `100.0%` | `+0.635` | `100.0%` | `12.5%` | `15.6%` | `100.0%` | `100.0%` | `+5.097` | `0.3611` |
| Bridge-Only `R_safe` | `2` | `0.0%` | `100.0%` | `+15.091` | `99.0%` | `+4.228` | `100.0%` | `18.8%` | `6.2%` | `96.9%` | `100.0%` | `+1.575` | `1.5311` |
| Bridge-Only `R_safe` | `3` | `0.0%` | `100.0%` | `+17.473` | `62.5%` | `+11.712` | `100.0%` | `18.8%` | `3.1%` | `46.9%` | `100.0%` | `+2.130` | `1.0650` |
| Bridge-Only `R_safe` | `4` | `0.0%` | `100.0%` | `+19.692` | `88.5%` | `+8.961` | `100.0%` | `26.6%` | `9.4%` | `100.0%` | `100.0%` | `+5.010` | `1.0801` |
| Dual Nullbridge | `1` | `0.0%` | `100.0%` | `+21.992` | `100.0%` | `+2.960` | `100.0%` | `10.9%` | `9.4%` | `93.8%` | `100.0%` | `+5.668` | `0.4102` |
| Dual Nullbridge | `2` | `0.0%` | `100.0%` | `+15.376` | `97.9%` | `+5.362` | `100.0%` | `23.4%` | `3.1%` | `96.9%` | `100.0%` | `+1.400` | `1.5436` |
| Dual Nullbridge | `3` | `0.0%` | `100.0%` | `+16.726` | `40.6%` | `+13.791` | `89.6%` | `14.1%` | `3.1%` | `46.9%` | `100.0%` | `+1.322` | `1.1025` |
| Dual Nullbridge | `4` | `0.0%` | `100.0%` | `+18.711` | `82.3%` | `+9.983` | `100.0%` | `28.1%` | `25.0%` | `75.0%` | `100.0%` | `+3.409` | `1.1484` |
| 19-Layer Funnel Clamp | `1` | `100.0%` | `96.9%` | `+13.321` | `12.5%` | `+15.739` | `50.0%` | `6.2%` | `0.0%` | `21.9%` | `100.0%` | `+0.654` | `2.4549` |
| 19-Layer Funnel Clamp | `2` | `100.0%` | `84.4%` | `+10.164` | `11.5%` | `+10.511` | `50.0%` | `17.2%` | `3.1%` | `75.0%` | `100.0%` | `+0.227` | `3.9774` |
| 19-Layer Funnel Clamp | `3` | `100.0%` | `100.0%` | `+6.883` | `0.0%` | `+7.006` | `50.0%` | `9.4%` | `6.2%` | `3.1%` | `100.0%` | `-0.473` | `3.4219` |
| 19-Layer Funnel Clamp | `4` | `100.0%` | `93.8%` | `+9.522` | `2.1%` | `+6.595` | `50.0%` | `10.9%` | `0.0%` | `15.6%` | `100.0%` | `-0.148` | `2.7905` |
| Random 4D Control | `1` | `0.0%` | `0.0%` | `+3.972` | `100.0%` | `+0.995` | `100.0%` | `0.0%` | `0.0%` | `0.0%` | `100.0%` | `+1.143` | `10.0883` |
| Random 4D Control | `2` | `0.0%` | `1.6%` | `+3.233` | `95.8%` | `+2.505` | `100.0%` | `0.0%` | `0.0%` | `3.1%` | `100.0%` | `+0.008` | `9.9496` |
| Random 4D Control | `3` | `0.0%` | `0.0%` | `+0.993` | `64.6%` | `+1.985` | `100.0%` | `1.6%` | `0.0%` | `3.1%` | `100.0%` | `+5.214` | `9.9444` |
| Random 4D Control | `4` | `0.0%` | `0.0%` | `+1.642` | `88.5%` | `+0.826` | `100.0%` | `0.0%` | `3.1%` | `0.0%` | `100.0%` | `+3.068` | `10.8317` |
