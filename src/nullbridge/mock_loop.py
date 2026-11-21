"""Offline synthetic looped-transformer testbed (`MiniLoopedModel`) for Phase 0 verification.

Mirrors the two architectural families studied in Nullbridge without requiring a GPU:
  1. Retrofitted Procrustes Funnel (`Prelude -> K Loops via R_proc -> Coda`, as in `Gemma-4-Loop-Compact`)
  2. Cross-Pass Rank-64 `tanh` Message Bank (`m^(k+1) = tanh(h^(k) W_msg^T)`, as in `Inkling 2 Mini` / `TML-d24-varloop`)

Verifies end-to-end across `K = 1..8` passes:
  - Unclamped recurrence solves deeper multi-hop reasoning as `K` increases while compounding latent
    sycophancy / reward-hacking cues across passes.
  - Surface / output-channel alignment (mimicking late verbalization LoRA/RLHF) suppresses explicit
    verbal admission while leaving covert numeric valuation & `0-CoT` verifier scores distorted.
  - Algebraic `Nullbridge` (`R_safe`, `W_msg_safe` + doorway projection) zeroes cross-pass bias
    amplification (`0.00e+00` null leakage) with `0` extra runtime FLOPs and `0.0%` multi-hop loss.
"""

from __future__ import annotations

from typing import Any
import numpy as np

from .algebra import (
    build_orthonormal_basis,
    fold_message_bank_projection,
    fold_procrustes_bridge,
    sample_random_orthonormal_basis,
)


def run_synthetic_loop_verification(
    d_model: int = 128,
    r_msg: int = 64,
    n_bias_dims: int = 4,
    n_hops_dims: int = 8,
    n_samples: int = 256,
    loop_counts: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    seed: int = 20261120,
) -> dict[str, Any]:
    """Simulate multi-pass reasoning vs. compounding latent bias across 5 alignment regimes."""
    rng = np.random.default_rng(seed)

    # Construct exact orthonormal subspaces inside R^d:
    #   cols 0..3:   bias subspace Q_bias (sycophancy, authority, gloss, spite)
    #   cols 4..11:  multi-hop reasoning registers Q_reason
    #   cols 12..d:  general language / semantic subspace
    full_q, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    q_bias = full_q[:, :n_bias_dims].astype(np.float32)              # [D, 4]
    q_reason = full_q[:, n_bias_dims : n_bias_dims + n_hops_dims]    # [D, 8]

    # Ground-truth orthogonal basis rotation between funnel exit and funnel entry
    r_true, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    r_proc = r_true.astype(np.float32)

    # Recurrent dynamics inside the loop block (in funnel-entry coordinates):
    #   - Reasoning coordinates integrate multi-hop evidence toward target state `z_star`
    #   - Bias coordinates amplify by gain `gamma = 1.32 > 1` on each recurrent pass when unclamped!
    r_proc64 = r_proc.astype(np.float64)
    q_b64 = q_bias.astype(np.float64)
    q_r64 = q_reason.astype(np.float64)

    # Express `q_bias` and `q_reason` at funnel exit (`L_end`) and at funnel entry (`L_start`):
    # `R_proc` maps exit coordinates -> entry coordinates (`R_proc @ h_end_vec = h_entry_vec`),
    # while the recurrent block forward pass maps entry -> exit (`R_proc.T`).
    q_b_end = q_b64
    q_b_entry = r_proc64 @ q_b_end
    q_r_end = q_r64
    q_r_entry = r_proc64 @ q_r_end

    # Cross-pass rank-64 message bank weights (`W_in [64, D]` reads exit, `W_out [D, 64]` writes entry)
    w_msg_in = (rng.standard_normal((r_msg, d_model)) / np.sqrt(d_model)).astype(np.float32)
    w_msg_out_raw = rng.standard_normal((d_model, r_msg)) / np.sqrt(r_msg)
    # Keep general message channels orthogonal to bias & reasoning entry except dedicated slots
    w_msg_out_raw = w_msg_out_raw - q_b_entry @ (q_b_entry.T @ w_msg_out_raw) - q_r_entry @ (q_r_entry.T @ w_msg_out_raw)
    w_msg_in[:n_bias_dims, :] = q_b_end.T.astype(np.float32) * 0.9
    w_msg_out_raw[:, :n_bias_dims] = q_b_entry * 0.6
    w_msg_out = w_msg_out_raw.astype(np.float32)

    # Folded weights (0 extra FLOPs!)
    folded_bridge = fold_procrustes_bridge(r_proc, q_b_end)
    w_msg_in_safe, msg_leak = fold_message_bank_projection(w_msg_in, q_b_end, side="input")
    w_msg_out_safe, _ = fold_message_bank_projection(w_msg_out, q_b_entry, side="output")
    q_rand = sample_random_orthonormal_basis(d_model, n_bias_dims, exclude_q=q_b_end, seed=seed + 7)
    folded_random = fold_procrustes_bridge(r_proc, q_rand)

    # Draw synthetic multi-hop + pressured test problems
    z_target = rng.standard_normal((n_samples, n_hops_dims))
    z_init = 0.15 * z_target + 0.90 * rng.standard_normal((n_samples, n_hops_dims))  # K=1 unresolved
    b_init = rng.uniform(0.6, 1.4, size=(n_samples, n_bias_dims))                   # Pass-0 sycophancy cue
    ambient = 0.15 * rng.standard_normal((n_samples, d_model))
    ambient = ambient - (ambient @ q_b_entry) @ q_b_entry.T - (ambient @ q_r_entry) @ q_r_entry.T
    h_entry_0 = z_init @ q_r_entry.T + b_init @ q_b_entry.T + ambient

    # Two readout heads after Coda:
    #   1. `w_overt`: explicit verbal admission head ("Yes, I agree with your wrong claim")
    #   2. `w_covert`: covert numeric valuation / `0-CoT` verifier head (reads latent bias!)
    w_overt = q_b_end[:, 0]
    w_covert_bias = np.mean(q_b_end, axis=1)

    def _run_condition(mode: str, k_passes: int) -> dict[str, float]:
        h_anc = h_entry_0.copy()
        if mode == "dual_nullbridge":
            h_anc = h_anc - (h_anc @ q_b_entry) @ q_b_entry.T

        r_mat = (
            folded_bridge.r_safe.astype(np.float64)
            if mode in ("bridge_only_nullbridge", "dual_nullbridge")
            else (
                folded_random.r_safe.astype(np.float64)
                if mode == "random_subspace_control"
                else r_proc64
            )
        )
        m_in = (
            w_msg_in_safe.astype(np.float64)
            if mode in ("bridge_only_nullbridge", "dual_nullbridge")
            else w_msg_in.astype(np.float64)
        )
        m_out = (
            w_msg_out_safe.astype(np.float64)
            if mode in ("bridge_only_nullbridge", "dual_nullbridge")
            else w_msg_out.astype(np.float64)
        )

        h_cur_entry = h_anc.copy()
        msg_vec = np.zeros((n_samples, d_model), dtype=np.float64)
        g_loop = 0.85

        for pass_idx in range(k_passes):
            if pass_idx > 0:
                h_rot = h_end @ r_mat.T
                h_cur_entry = (1.0 - g_loop) * h_anc + g_loop * h_rot + 0.30 * msg_vec

            # Inside the looped block (`entry -> exit` via `R_proc.T`):
            #   - Reasoning state contracts error toward `z_target` each pass!
            #   - Surviving bias component amplifies by 1.38x inside the block!
            cur_r = h_cur_entry @ q_r_entry
            new_r = z_target - 0.35 * (z_target - cur_r)
            cur_b = h_cur_entry @ q_b_entry
            new_b = 1.38 * cur_b
            rem = h_cur_entry - cur_r @ q_r_entry.T - cur_b @ q_b_entry.T
            # `rem` is orthogonal to `q_b_entry` and `q_r_entry`, so `rem @ R_proc` is orthogonal to `q_b_end` and `q_r_end`!
            h_end = new_r @ q_r_end.T + new_b @ q_b_end.T + rem @ r_proc64

            # Update rank-64 tanh message bank for next pass
            msg_bank_state = np.tanh(h_end @ m_in.T)
            msg_vec = msg_bank_state @ m_out.T

        # Coda readout: `surface_coda_filter` patches ONLY the explicit verbalization readout channel
        overt_score = h_end @ w_overt
        if mode == "surface_coda_filter":
            overt_score = overt_score * 0.04  # Polite verbal refusal learned!

        pred_r = h_end @ q_r_end
        reason_mse = np.mean((pred_r - z_target) ** 2, axis=-1)
        multihop_acc = float(np.mean(reason_mse < 0.10))
        covert_bias = float(np.mean(np.abs(h_end @ w_covert_bias)))
        overt_bias = float(np.mean(np.abs(overt_score)))
        latent_bias_norm = float(np.mean(np.linalg.norm(h_end @ q_b_end, axis=-1)))

        return {
            "multihop_acc": round(multihop_acc, 4),
            "overt_verbal_bias": round(overt_bias, 5),
            "covert_numeric_bias": round(covert_bias, 5),
            "latent_bias_norm": round(latent_bias_norm, 5),
        }

    modes = (
        "unclamped_loop",
        "surface_coda_filter",
        "random_subspace_control",
        "bridge_only_nullbridge",
        "dual_nullbridge",
    )
    trajectories: dict[str, dict[str, Any]] = {}
    for m in modes:
        by_k = {f"k{k}": _run_condition(m, k) for k in loop_counts}
        slope_covert = (by_k[f"k{loop_counts[-1]}"]["covert_numeric_bias"] - by_k["k1"]["covert_numeric_bias"]) / max(
            1, loop_counts[-1] - loop_counts[0]
        )
        trajectories[m] = {
            "by_k": by_k,
            "covert_amplification_slope": round(float(slope_covert), 6),
        }

    # Verify core Phase 0 synthetic invariants
    unclamped_amplifies = (
        trajectories["unclamped_loop"]["by_k"]["k8"]["covert_numeric_bias"]
        > 2.0 * trajectories["unclamped_loop"]["by_k"]["k1"]["covert_numeric_bias"]
    )
    surface_hides_only_overt = (
        trajectories["surface_coda_filter"]["by_k"]["k8"]["overt_verbal_bias"]
        < 0.10 * trajectories["unclamped_loop"]["by_k"]["k8"]["overt_verbal_bias"]
        and trajectories["surface_coda_filter"]["by_k"]["k8"]["covert_numeric_bias"]
        > 2.0 * trajectories["surface_coda_filter"]["by_k"]["k1"]["covert_numeric_bias"]
    )
    dual_zeroes_bias_and_keeps_acc = (
        trajectories["dual_nullbridge"]["by_k"]["k8"]["covert_numeric_bias"] < 1e-5
        and trajectories["dual_nullbridge"]["by_k"]["k8"]["multihop_acc"]
        == trajectories["unclamped_loop"]["by_k"]["k8"]["multihop_acc"]
        and trajectories["unclamped_loop"]["by_k"]["k8"]["multihop_acc"] > 0.90
    )

    return {
        "valid": bool(unclamped_amplifies and surface_hides_only_overt and dual_zeroes_bias_and_keeps_acc),
        "message_bank_null_leakage": msg_leak,
        "loop_counts": list(loop_counts),
        "conditions": trajectories,
    }
