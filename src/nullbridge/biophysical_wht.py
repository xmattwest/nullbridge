"""Exact 16-term Walsh-Hadamard Transform (WHT), commitment horizon, and hysteresis math for Phase 6.

Pure NumPy module so every algebraic identity (Parseval variance conservation, exact reconstruction,
main additive channel effects vs. pairwise QxK and VxMLP epistasis, commitment horizon interpolation,
and pulse-chase decay constants) runs deterministically on both CPU unit tests and GPU analysis.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Sequence

import numpy as np

from .algebra import bootstrap_mean_ci, exact_or_mc_signflip_p, holm_adjust


CHANNEL_NAMES: tuple[str, str, str, str] = ("Q", "K", "V", "MLP")


def mask_int_to_bits(mask_int: int) -> tuple[int, int, int, int]:
    """Map `mask_int in 0..15` to `(b_Q, b_K, b_V, b_MLP)` in `{0, 1}^4`."""
    if not (0 <= mask_int < 16):
        raise ValueError(f"mask_int must be in 0..15, got {mask_int}")
    return (
        (mask_int >> 3) & 1,
        (mask_int >> 2) & 1,
        (mask_int >> 1) & 1,
        mask_int & 1,
    )


def bits_to_mask_code(bits: Sequence[int]) -> str:
    """Format `(1, 1, 0, 0)` as `'1100'`."""
    return "".join("1" if int(b) else "0" for b in bits)


def mask_human_label(bits: Sequence[int]) -> str:
    """Readable channel subset label, e.g. `'Q+K'` or `'None (0000)'` or `'All (Q+K+V+MLP)'`."""
    active = [CHANNEL_NAMES[i] for i, b in enumerate(bits) if int(b) != 0]
    if not active:
        return "None (0000)"
    if len(active) == 4:
        return "All (Q+K+V+MLP)"
    return "+".join(active)


def enumerate_16_masks() -> list[tuple[int, tuple[int, int, int, int], str, str]]:
    """Return all 16 binary masks `(mask_int, bits, code_str, label)` ordered `0..15`."""
    out: list[tuple[int, tuple[int, int, int, int], str, str]] = []
    for m in range(16):
        bits = mask_int_to_bits(m)
        out.append((m, bits, bits_to_mask_code(bits), mask_human_label(bits)))
    return out


def wht_subsets_16() -> list[tuple[tuple[int, ...], str, int]]:
    """Canonical ordering of all 16 subsets `S` of `{Q, K, V, MLP}` by interaction order `0..4`."""
    subsets: list[tuple[tuple[int, ...], str, int]] = [((), "const", 0)]
    for order in (1, 2, 3, 4):
        for comb in combinations(range(4), order):
            lbl = "x".join(CHANNEL_NAMES[i] for i in comb)
            subsets.append((comb, lbl, order))
    return subsets


def build_walsh_hadamard_design_matrix() -> np.ndarray:
    """Build orthogonal `16 x 16` spin basis `H` (`H[m, k] = prod_{i in S_k} z_i(m)` with `z_i = 2*b_i - 1`).

    Satisfies `(1/16) H^T H = I_16`, so `w = (1/16) H^T y` is the exact Fourier-Walsh expansion
    and `y = H w` reconstructs all 16 observations to machine precision.
    Notice:
      - For order 1 (`S = {i}`), turning channel `i` from OFF (`z_i = -1`) to ON (`z_i = +1`)
        changes `f` by `+2 * w_{i}` averaged uniformly over all `2^3 = 8` partner states.
      - For order 2 (`S = {i, j}`), `4 * w_{i,j}` equals the average pairwise epistatic synergy
        `[f(1,1) - f(1,0) - f(0,1) + f(0,0)]` across the remaining `2^2 = 4` partner states.
    """
    subsets = wht_subsets_16()
    mat = np.ones((16, 16), dtype=np.float64)
    for m in range(16):
        bits = mask_int_to_bits(m)
        spins = [1.0 if b == 1 else -1.0 for b in bits]
        for k, (comb, _lbl, _ord) in enumerate(subsets):
            prod = 1.0
            for idx in comb:
                prod *= spins[idx]
            mat[m, k] = prod
    return mat


def mean_and_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 4000,
    seed: int = 202606,
) -> tuple[float, float, float]:
    """Return `(sample_mean, ci95_low, ci95_high)` using `bootstrap_mean_ci`."""
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    lo, hi = bootstrap_mean_ci(arr, n_boot=n_boot, seed=seed)
    return (float(np.mean(arr)), lo, hi)


def compute_exact_wht_16(
    item_matrix_16xn: np.ndarray,
    seed: int = 202606,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Compute per-item and population 16-term Walsh-Hadamard decomposition with bootstrap CIs & Holm tests.

    Args:
        item_matrix_16xn: shape `(16, N)` float array aligned to `mask_int = 0..15`.
    """
    arr = np.asarray(item_matrix_16xn, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 16:
        raise ValueError(f"Expected shape (16, N), got {arr.shape}")
    n_items = arr.shape[1]

    h_mat = build_walsh_hadamard_design_matrix()  # (16, 16)
    # Per-item WHT coefficients: shape (16, N)
    w_items = (h_mat.T @ arr) / 16.0
    w_mean = np.mean(w_items, axis=1)
    recon = h_mat @ w_mean
    mean_obs = np.mean(arr, axis=1)
    max_recon_err = float(np.max(np.abs(recon - mean_obs)))

    # Parseval variance spectrum on mean trajectory across masks (excluding k=0 constant)
    nonconst_sq = np.square(w_mean[1:])
    total_nonconst_var = float(np.sum(nonconst_sq))
    pop_var = float(np.var(mean_obs, ddof=0))
    parseval_err = abs(total_nonconst_var - pop_var)

    subsets = wht_subsets_16()
    raw_p_nonconst: list[float] = []
    for k in range(1, 16):
        raw_p_nonconst.append(exact_or_mc_signflip_p(w_items[k], seed=seed + k))
    holm_p_nonconst = holm_adjust(raw_p_nonconst)

    terms: dict[str, dict[str, Any]] = {}
    order_var_sum = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for k, (_comb, lbl, order) in enumerate(subsets):
        est, ci_lo, ci_hi = mean_and_ci(w_items[k], n_boot=n_boot, seed=seed + 100 + k)
        # Coalitional scale: 2*w_i for main effects, 4*w_{ij} for pairwise epistasis
        coalition_scale = float(2**order) if order >= 1 else 1.0
        var_share = (float(w_mean[k] ** 2) / total_nonconst_var) if (order >= 1 and total_nonconst_var > 1e-15) else 0.0
        if order >= 1:
            order_var_sum[order] += float(w_mean[k] ** 2)
        entry: dict[str, Any] = {
            "order": order,
            "wht_coeff": round(est, 6),
            "wht_ci95": [round(ci_lo, 6), round(ci_hi, 6)],
            "coalition_effect": round(est * coalition_scale, 6),
            "coalition_ci95": [round(ci_lo * coalition_scale, 6), round(ci_hi * coalition_scale, 6)],
            "variance_share": round(var_share, 6),
        }
        if order >= 1:
            entry["p_raw"] = round(float(raw_p_nonconst[k - 1]), 6)
            entry["p_holm"] = round(float(holm_p_nonconst[k - 1]), 6)
        terms[lbl] = entry

    order_shares = {
        f"order_{ord_k}": round(
            (order_var_sum[ord_k] / total_nonconst_var) if total_nonconst_var > 1e-15 else 0.0,
            6,
        )
        for ord_k in (1, 2, 3, 4)
    }

    # Direct monosynaptic sub-circuit contrast: QK-only (1100 = idx 12) vs V+MLP-only (0011 = idx 3)
    qk_minus_vm = arr[12] - arr[3]
    qk_est, qk_lo, qk_hi = mean_and_ci(qk_minus_vm, n_boot=n_boot, seed=seed + 501)
    total_gain = arr[15] - arr[0]
    tg_est, tg_lo, tg_hi = mean_and_ci(total_gain, n_boot=n_boot, seed=seed + 502)

    # Epistatic synergy contrast: QxK vs VxMLP
    qxk_idx = next(i for i, s in enumerate(subsets) if s[1] == "QxK")
    vxm_idx = next(i for i, s in enumerate(subsets) if s[1] == "VxMLP")
    ep_diff = 4.0 * (w_items[qxk_idx] - w_items[vxm_idx])
    ep_est, ep_lo, ep_hi = mean_and_ci(ep_diff, n_boot=n_boot, seed=seed + 503)

    by_mask: dict[str, dict[str, Any]] = {}
    for m, _bits, code, h_lbl in enumerate_16_masks():
        m_est, m_lo, m_hi = mean_and_ci(arr[m], n_boot=min(2000, n_boot), seed=seed + 700 + m)
        by_mask[code] = {
            "mask_int": m,
            "label": h_lbl,
            "mean": round(m_est, 6),
            "ci95": [round(m_lo, 6), round(m_hi, 6)],
        }

    return {
        "n_items": int(n_items),
        "max_reconstruction_error": max_recon_err,
        "parseval_variance_error": parseval_err,
        "total_loop_gain_1111_minus_0000": {
            "mean": round(tg_est, 6),
            "ci95": [round(tg_lo, 6), round(tg_hi, 6)],
            "p_signflip": round(exact_or_mc_signflip_p(total_gain, seed=seed + 601), 6),
        },
        "qk_only_minus_v_mlp_only": {
            "mean": round(qk_est, 6),
            "ci95": [round(qk_lo, 6), round(qk_hi, 6)],
            "p_signflip": round(exact_or_mc_signflip_p(qk_minus_vm, seed=seed + 602), 6),
        },
        "qxk_minus_vxmlp_synergy": {
            "mean": round(ep_est, 6),
            "ci95": [round(ep_lo, 6), round(ep_hi, 6)],
            "p_signflip": round(exact_or_mc_signflip_p(ep_diff, seed=seed + 603), 6),
        },
        "order_variance_shares": order_shares,
        "terms": terms,
        "masks": by_mask,
    }


def estimate_commitment_horizon(
    stage_labels: Sequence[str],
    hop2_minus_hop1_log_odds: Sequence[float],
) -> dict[str, Any]:
    """Locate the commitment horizon `t*` where Hop-2 log-odds overtakes Hop-1 (`delta_lo >= 0`)."""
    vals = [float(x) for x in hop2_minus_hop1_log_odds]
    if len(vals) != len(stage_labels) or len(vals) == 0:
        raise ValueError("stage_labels and hop2_minus_hop1_log_odds must have matching non-zero length")

    crossing_idx: int | None = None
    interp_pos: float | None = None
    for i, v in enumerate(vals):
        if v >= 0.0:
            crossing_idx = i
            if i == 0:
                interp_pos = 0.0
            else:
                prev = vals[i - 1]
                denom = v - prev
                frac = (0.0 - prev) / denom if abs(denom) > 1e-12 else 0.0
                interp_pos = float((i - 1) + max(0.0, min(1.0, frac)))
            break

    peak_jump_idx = int(np.argmax(np.diff(vals))) + 1 if len(vals) >= 2 else 0
    return {
        "crossed_zero": crossing_idx is not None,
        "crossing_stage_index": crossing_idx,
        "crossing_stage_label": stage_labels[crossing_idx] if crossing_idx is not None else None,
        "interpolated_stage_position": round(interp_pos, 4) if interp_pos is not None else None,
        "steepest_transition_stage": stage_labels[peak_jump_idx],
        "steepest_transition_delta_lo": round(vals[peak_jump_idx] - vals[peak_jump_idx - 1], 4)
        if len(vals) >= 2
        else 0.0,
    }


def summarize_hysteresis_curve(
    betas: Sequence[float],
    p2_foil_prob: Sequence[float],
    p3_foil_prob: Sequence[float],
    coda_target_acc: Sequence[float],
    coda_foil_capture: Sequence[float],
) -> dict[str, Any]:
    """Quantify elastic self-correction area (`Pass 2 -> Pass 3` foil reduction) and bifurcation `beta_c`."""
    b = np.asarray(betas, dtype=np.float64)
    f2 = np.asarray(p2_foil_prob, dtype=np.float64)
    f3 = np.asarray(p3_foil_prob, dtype=np.float64)
    acc = np.asarray(coda_target_acc, dtype=np.float64)
    cap = np.asarray(coda_foil_capture, dtype=np.float64)

    # Elastic recovery = reduction in foil probability between Pass 2 exit and Pass 3 exit
    recovery = f2 - f3
    area_recovery = float(np.trapezoid(recovery, b)) if hasattr(np, "trapezoid") else float(np.trapz(recovery, b))

    # Critical bifurcation beta_c where foil capture exceeds 50% (or overtakes target accuracy)
    beta_c: float | None = None
    for i in range(len(b)):
        if cap[i] >= acc[i] and i > 0:
            prev_gap = cap[i - 1] - acc[i - 1]
            cur_gap = cap[i] - acc[i]
            denom = cur_gap - prev_gap
            frac = -prev_gap / denom if abs(denom) > 1e-12 else 0.0
            beta_c = float(b[i - 1] + max(0.0, min(1.0, frac)) * (b[i] - b[i - 1]))
            break

    max_rec_idx = int(np.argmax(recovery))
    return {
        "elastic_recovery_integral": round(area_recovery, 6),
        "peak_pass3_foil_suppression": round(float(recovery[max_rec_idx]), 6),
        "peak_suppression_beta": float(b[max_rec_idx]),
        "critical_bifurcation_beta": round(beta_c, 4) if beta_c is not None else None,
        "target_retention_at_beta_05": round(
            float(np.interp(0.5, b, acc)),
            4,
        ),
        "foil_capture_at_beta_10": round(
            float(np.interp(1.0, b, cap)),
            4,
        ),
    }
