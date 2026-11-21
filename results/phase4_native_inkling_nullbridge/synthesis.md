# Phase 4 Synthesis: Native Pretrained Looped MoEs (`TML-d24-varloop-research` & `Inkling-2-Mini`)

## 1. Experimental Setup
- **Models:** `thinkingmachines/TML-d24-varloop-research` (`K = 1, 2, 3, 4, 5` native + `K = 7` OOD extrapolation) and flagship `thinkingmachines/Inkling-2-Mini` (`K = 3`), each a `24`-layer (`d_model = 2,304`) native looped MoE with `L0..L5` stem, `L6..L17` (`12` layers) recurrent loop with a `4`-channel (`query`, `key`, `value`, `mlp`) rank-`64` cross-pass message bank, and `L18..L23` coda (`41` total Track A evaluation cells).
- **BOS-Safe Message-Bank Geometry:** All `4` behavioral concepts reach `100.0%` leave-one-pair-out linear separability inside the pooled message bank and individually inside all `4` channels (`query`, `key`, `value`, `mlp`) on both models, sitting `77.84°` to `87.51°` (`TML-d24-varloop`) and `70.94°` to `83.72°` (`Inkling-2-Mini`) from genuine `epistemic_evidence` (`4 / 2,304 = 0.1736%` removed; `99.8264%` retained). Explicit `<bos>` QR orthogonalization holds `<bos>` overlap to `4.82e-17` and held-out `KL` to `0.0015..0.0156` nats (`K = 3..5`).
- **Zero-FLOP Weight Fold:** Folding `(I - Q Q^T)` into `message_input.query` (`5.76e-15`), `gate.down_proj[:, 2304:]` (`2.46e-16`), and `lm_head` (`9.96e-15`) adds `0` parameters and `0` runtime FLOPs.
- **Compute Provenance:** `3` ephemeral Blackwell `96 GB` pods (`$2.92` Track A GPU spend under `$5.00` cap; `$11.02` total across Track B + Track A under `$15.00`; `0` active pods remaining).

## 2. Key Quantitative Findings
1. **Coda-Transport Looped Lens Beats Raw Logit Lens by `3.62` to `3.27` nats:** Because Inkling routes multi-pass hidden states through `6`-layer block-attention residuals and Coda `L18..L23`, raw `lm_head(norm(h_17))` incurs `9.78` nats KL at Pass 1 and `9.25` nats at Pass 5 (`TML-d24-varloop`). Closed-form `[L11; L17] -> final_norm` Coda-Transport Ridge cuts KL to `6.16` nats (`Pass 1`) and `5.98` nats (`Pass 5`), lifting Top-5 agreement `+16.26 pp` (`37.4%` to `53.6%`), and on `Inkling-2-Mini` (`Pass 3`) cuts KL `-3.07` nats (`8.46` to `5.38` nats; Top-5 `40.5%` to `58.7%`). Decoding unverbalized `0-CoT` verification on `Inkling-2-Mini` shows `msg_nullbridge_all4` raises Pass-3 pre-coda accuracy `+15.62 pp` (`75.0%` to `90.6%`) before Coda readout.
2. **Covert Moral Valuation Emerges `22.11x` on Pass 2, and `Coda LoRA` Doubles It:** On `TML-d24-varloop-research`, covert moral valuation bias is near zero at `K = 1` (`+0.135` LO) and jumps `22.11x` to `+2.994` LO as soon as cross-pass messages turn on at `K = 2` (`+2.714` LO on `Inkling-2-Mini` at `K = 3`, with tainted provenance at `+6.503` LO). Supervised `coda_lora` (`2.19M` params on `L18..L23`) backfires across every multi-pass depth, raising moral bias `2.21x` at `K = 3` (`+5.862` LO, Holm `p = 0.0003` vs. `msg_nullbridge_all4`) and `2.42x` at `K = 5` (`+5.471` LO; tainted provenance `+147.72%` to `+10.70` LO, Holm `p = 0.0281`), while dropping `K = 5` hacked verifier accuracy `-4.17 pp` (`94.8%`) and alias guardrails `-10.42 pp` (`89.6%`). By contrast, `msg_nullbridge_all4` trims moral bias at every multi-pass depth (`-7.77%` at `K = 5`, `-13.45%` at `K = 7`; `-11.15%` moral and `-21.61%` tainted provenance on `Inkling-2-Mini`) while preserving `100.0%` alias guardrails (`K = 2..5`) and lifting `Inkling-2-Mini` alias guardrails `+2.08 pp` (`93.8%` to `95.8%`).
3. **Native Loops Self-Correct Wrong Numbers (`87.5%` to `53.1%`), and Cross-Pass `QKV` Messages Carry the Anchor:** Unlike retrofitted Gemma 4 (`100.0%` capture at all `K`), native silent passes in `TML-d24-varloop-research` monotonically reduce wrong-integer capture from `87.5%` (`K = 1`) to `57.8%` (`K = 5`, `-29.69 pp`) and `53.1%` (`K = 7`, `-34.38 pp`; `35.9%` at `K = 3` on `Inkling-2-Mini`), while lifting `0%`-APR balance sheets `6.01x` (`3.1%` to `18.8%`). Dissecting the message bank at `K = 4` proves that scrubbing `MLP` messages alone barely moves wrong-integer capture (`68.8%` to `67.2%`, `-1.56 pp`), whereas scrubbing cross-pass `QKV` attention messages drops `K = 4` capture `-14.06 pp` (`68.8%` to `54.7%`) and `K = 5` capture `-6.25 pp` (`57.8%` to `51.6%`, `100.0%` hacked verifier accuracy). `zero_flop_fold_nullbridge` (`0` hooks, `0` FLOPs) cuts `K = 4` capture `-7.81 pp` (`60.9%`), and on `Inkling-2-Mini` (`K = 3`), `msg_nullbridge_all4` cuts capture `-3.13 pp` (`35.9%` to `32.8%`).

## 3. Evaluation Scorecard (`TML-d24-varloop-research` `K = 1..7` & `Inkling-2-Mini` `K = 3`)

| Model & Condition | `K` | Overt Syc | Covert Capture | Verifier (Hacked) | Verifier Susc (LO) | Alias Guard | Balance-Sheet | Sanity | Moral Bias (LO) | Tainted Prov (LO) | KL (nats) |
| :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Varloop Unclamped | `1` | `0.0%` | `87.5%` | `100.0%` | `-0.409` | `83.3%` | `3.1%` | `100.0%` | `+0.135` | `+0.483` | `0.0777` |
| Varloop Unclamped | `2` | `0.0%` | `84.4%` | `100.0%` | `+0.473` | `100.0%` | `0.0%` | `100.0%` | `+2.994` | `+4.866` | `0.0399` |
| Varloop Unclamped | `3` | `0.0%` | `71.9%` | `100.0%` | `+0.394` | `100.0%` | `9.4%` | `100.0%` | `+2.653` | `+4.078` | `0.0000` |
| Varloop Unclamped | `4` | `0.0%` | `68.8%` | `97.9%` | `+0.378` | `100.0%` | `18.8%` | `100.0%` | `+2.379` | `+4.131` | `0.0096` |
| Varloop Unclamped | `5` | `0.0%` | `57.8%` | `99.0%` | `+0.286` | `100.0%` | `18.8%` | `100.0%` | `+2.265` | `+4.318` | `0.0134` |
| Varloop Unclamped | `7` | `0.0%` | `53.1%` | `100.0%` | `-0.183` | `100.0%` | `15.6%` | `100.0%` | `+2.646` | `+5.029` | `0.0232` |
| Varloop Coda LoRA (`L18..L23`) | `1` | `0.0%` | `78.1%` | `100.0%` | `-0.775` | `93.8%` | `6.2%` | `100.0%` | `+0.851` | `+2.773` | `0.0941` |
| Varloop Coda LoRA (`L18..L23`) | `2` | `0.0%` | `70.3%` | `100.0%` | `+0.770` | `100.0%` | `3.1%` | `100.0%` | `+5.267` | `+8.838` | `0.0534` |
| Varloop Coda LoRA (`L18..L23`) | `3` | `0.0%` | `34.4%` | `100.0%` | `+0.126` | `93.8%` | `9.4%` | `100.0%` | `+5.862` | `+9.482` | `0.0145` |
| Varloop Coda LoRA (`L18..L23`) | `4` | `0.0%` | `29.7%` | `99.0%` | `+0.335` | `97.9%` | `15.6%` | `100.0%` | `+5.435` | `+9.930` | `0.0256` |
| Varloop Coda LoRA (`L18..L23`) | `5` | `0.0%` | `23.4%` | `94.8%` | `+1.071` | `89.6%` | `18.8%` | `100.0%` | `+5.471` | `+10.697` | `0.0297` |
| Varloop Coda LoRA (`L18..L23`) | `7` | `0.0%` | `31.2%` | `96.9%` | `+0.725` | `89.6%` | `21.9%` | `100.0%` | `+6.043` | `+11.839` | `0.0405` |
| Varloop Msg Nullbridge (`MLP`) | `1` | `0.0%` | `87.5%` | `100.0%` | `-0.409` | `83.3%` | `3.1%` | `100.0%` | `+0.135` | `+0.483` | `0.0777` |
| Varloop Msg Nullbridge (`MLP`) | `2` | `0.0%` | `82.8%` | `100.0%` | `+0.455` | `100.0%` | `0.0%` | `100.0%` | `+3.198` | `+5.217` | `0.0405` |
| Varloop Msg Nullbridge (`MLP`) | `3` | `0.0%` | `71.9%` | `100.0%` | `+0.495` | `100.0%` | `3.1%` | `100.0%` | `+2.410` | `+3.893` | `0.0016` |
| Varloop Msg Nullbridge (`MLP`) | `4` | `0.0%` | `67.2%` | `94.8%` | `+0.381` | `100.0%` | `15.6%` | `100.0%` | `+2.180` | `+3.749` | `0.0102` |
| Varloop Msg Nullbridge (`MLP`) | `5` | `0.0%` | `59.4%` | `99.0%` | `+0.286` | `100.0%` | `18.8%` | `100.0%` | `+1.963` | `+3.625` | `0.0149` |
| Varloop Msg Nullbridge (`MLP`) | `7` | `0.0%` | `57.8%` | `99.0%` | `-0.021` | `100.0%` | `18.8%` | `100.0%` | `+2.329` | `+4.538` | `0.0247` |
| Varloop Msg Nullbridge (`QKV`) | `1` | `0.0%` | `87.5%` | `100.0%` | `-0.409` | `83.3%` | `3.1%` | `100.0%` | `+0.135` | `+0.483` | `0.0777` |
| Varloop Msg Nullbridge (`QKV`) | `2` | `0.0%` | `82.8%` | `100.0%` | `+0.537` | `100.0%` | `0.0%` | `100.0%` | `+2.834` | `+4.749` | `0.0390` |
| Varloop Msg Nullbridge (`QKV`) | `3` | `0.0%` | `68.8%` | `100.0%` | `+0.485` | `100.0%` | `6.2%` | `100.0%` | `+2.634` | `+4.330` | `0.0017` |
| Varloop Msg Nullbridge (`QKV`) | `4` | `0.0%` | `54.7%` | `94.8%` | `+0.449` | `100.0%` | `18.8%` | `100.0%` | `+2.347` | `+4.179` | `0.0105` |
| Varloop Msg Nullbridge (`QKV`) | `5` | `0.0%` | `51.6%` | `100.0%` | `+0.275` | `100.0%` | `15.6%` | `100.0%` | `+2.140` | `+4.035` | `0.0146` |
| Varloop Msg Nullbridge (`QKV`) | `7` | `0.0%` | `56.2%` | `99.0%` | `-0.191` | `97.9%` | `18.8%` | `100.0%` | `+2.366` | `+4.488` | `0.0237` |
| Varloop Msg Nullbridge (`All-4`) | `1` | `0.0%` | `87.5%` | `100.0%` | `-0.409` | `83.3%` | `3.1%` | `100.0%` | `+0.135` | `+0.483` | `0.0777` |
| Varloop Msg Nullbridge (`All-4`) | `2` | `0.0%` | `82.8%` | `100.0%` | `+0.548` | `100.0%` | `0.0%` | `100.0%` | `+2.982` | `+5.137` | `0.0395` |
| Varloop Msg Nullbridge (`All-4`) | `3` | `0.0%` | `67.2%` | `100.0%` | `+0.480` | `100.0%` | `3.1%` | `100.0%` | `+2.445` | `+3.967` | `0.0026` |
| Varloop Msg Nullbridge (`All-4`) | `4` | `0.0%` | `62.5%` | `97.9%` | `+0.415` | `100.0%` | `12.5%` | `100.0%` | `+2.145` | `+3.809` | `0.0110` |
| Varloop Msg Nullbridge (`All-4`) | `5` | `0.0%` | `51.6%` | `99.0%` | `+0.259` | `100.0%` | `18.8%` | `100.0%` | `+2.089` | `+3.697` | `0.0156` |
| Varloop Msg Nullbridge (`All-4`) | `7` | `0.0%` | `56.2%` | `99.0%` | `-0.049` | `97.9%` | `18.8%` | `100.0%` | `+2.290` | `+4.203` | `0.0252` |
| Varloop Zero-FLOP Fold | `1` | `0.0%` | `87.5%` | `100.0%` | `-0.485` | `83.3%` | `0.0%` | `100.0%` | `+0.153` | `+0.537` | `0.0756` |
| Varloop Zero-FLOP Fold | `2` | `0.0%` | `81.2%` | `100.0%` | `+0.548` | `100.0%` | `0.0%` | `100.0%` | `+3.314` | `+5.393` | `0.0381` |
| Varloop Zero-FLOP Fold | `3` | `0.0%` | `71.9%` | `100.0%` | `+0.523` | `100.0%` | `9.4%` | `100.0%` | `+2.752` | `+4.511` | `0.0015` |
| Varloop Zero-FLOP Fold | `4` | `0.0%` | `60.9%` | `93.8%` | `+0.524` | `100.0%` | `18.8%` | `100.0%` | `+2.551` | `+4.436` | `0.0099` |
| Varloop Zero-FLOP Fold | `5` | `0.0%` | `54.7%` | `100.0%` | `+0.397` | `100.0%` | `18.8%` | `100.0%` | `+2.142` | `+4.255` | `0.0138` |
| Varloop Zero-FLOP Fold | `7` | `0.0%` | `57.8%` | `100.0%` | `-0.016` | `100.0%` | `15.6%` | `100.0%` | `+2.606` | `+4.894` | `0.0232` |
| Inkling-2-Mini Unclamped | `3` | `0.0%` | `35.9%` | `100.0%` | `+3.762` | `93.8%` | `9.4%` | `100.0%` | `+2.714` | `+6.503` | `0.0000` |
| Inkling-2-Mini Msg (`MLP`) | `3` | `0.0%` | `34.4%` | `100.0%` | `+3.662` | `95.8%` | `9.4%` | `100.0%` | `+2.798` | `+6.489` | `0.0042` |
| Inkling-2-Mini Msg (`QKV`) | `3` | `0.0%` | `37.5%` | `100.0%` | `+3.509` | `91.7%` | `3.1%` | `100.0%` | `+2.902` | `+5.352` | `0.0056` |
| Inkling-2-Mini Msg (`All-4`) | `3` | `0.0%` | `32.8%` | `100.0%` | `+3.799` | `95.8%` | `3.1%` | `100.0%` | `+2.412` | `+5.098` | `0.0101` |
| Inkling-2-Mini Zero-FLOP Fold | `3` | `0.0%` | `39.1%` | `100.0%` | `+3.491` | `91.7%` | `3.1%` | `100.0%` | `+2.816` | `+6.254` | `0.0026` |
