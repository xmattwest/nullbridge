"""Exact Revealed-Preference Microeconomics Engine: GARP, Afriat CCEI, Bronars Power, and Utility Recovery.

Implements Sidney Afriat's (1967) Theorem, Hal Varian's (1982) Generalized Axiom of
Revealed Preference (GARP) and Critical Cost Efficiency Index (CCEI), Houtman-Maks / pairwise
cycle diagnostics, Bronars (1987) Monte Carlo power benchmarks against uniform random choice,
and constructive Afriat utility and marginal-utility-of-income `(u_t, lambda_t)` recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class GARPAuditResult:
    """Complete revealed-preference audit for a T-observation `(prices, bundles)` dataset."""

    n_obs: int
    passes_garp: bool
    garp_violation_pairs: int
    warp_violation_pairs: int
    sarp_violation_pairs: int
    afriat_ccei: float
    afriat_utility_u: list[float]
    afriat_multiplier_lambda: list[float]
    max_afriat_inequality_violation: float
    effective_efficiency_used: float


def compute_expenditure_matrix(prices: np.ndarray, bundles: np.ndarray) -> np.ndarray:
    """Return `T x T` matrix `E` where `E[t, s] = p_t . x_s` (cost of bundle `s` at prices `t`)."""
    p = np.asarray(prices, dtype=np.float64)
    x = np.asarray(bundles, dtype=np.float64)
    if p.ndim != 2 or x.ndim != 2 or p.shape != x.shape:
        raise ValueError(f"prices {p.shape} and bundles {x.shape} must have matching 2D shape (T, K)")
    if np.any(p <= 0.0):
        raise ValueError("All prices must be strictly positive (p_t in R_{++}^K)")
    if np.any(x < -1e-12):
        raise ValueError("Bundles must be non-negative")
    return p @ x.T


def transitive_closure_bool(adj: np.ndarray) -> np.ndarray:
    """Exact Floyd-Warshall boolean transitive closure of `T x T` relation matrix."""
    reach = np.asarray(adj, dtype=bool).copy()
    t_n = reach.shape[0]
    for k in range(t_n):
        # reach[i, j] |= reach[i, k] & reach[k, j]
        reach |= reach[:, k : k + 1] & reach[k : k + 1, :]
    return reach


def count_garp_violations_at_e(
    exp_mat: np.ndarray,
    e: float = 1.0,
    tol: float = 1e-9,
) -> tuple[int, np.ndarray]:
    """Check `GARP(e)` on expenditure matrix `E[t, s] = p_t . x_s`.

    Direct revealed preference at efficiency `e in (0, 1]`:
      `x_t R^0(e) x_s` iff `e * E[t, t] >= E[t, s] - tol`
      `x_s P^0(e) x_t` iff `e * E[s, s] >  E[s, t] + tol`
    `GARP(e)` holds iff no `(t, s)` satisfies `x_t R(e) x_s` and `x_s P^0(e) x_t`.
    """
    own_exp = np.diag(exp_mat)[:, None]  # (T, 1)
    r0 = (e * own_exp + tol) >= exp_mat
    p0 = (e * own_exp - tol) > exp_mat
    np.fill_diagonal(p0, False)
    r_star = transitive_closure_bool(r0)
    violations = r_star & p0.T
    np.fill_diagonal(violations, False)
    return int(np.sum(violations)), violations


def count_warp_and_sarp_violations(
    exp_mat: np.ndarray,
    bundles: np.ndarray,
    tol: float = 1e-9,
) -> tuple[int, int]:
    """Count exact unordered WARP violation pairs and ordered SARP violation pairs at `e = 1.0`."""
    t_n = exp_mat.shape[0]
    own_exp = np.diag(exp_mat)[:, None]
    r0 = (own_exp + tol) >= exp_mat
    p0 = (own_exp - tol) > exp_mat
    np.fill_diagonal(p0, False)

    # Unordered WARP violation: `x_t != x_s` and (`x_t R^0 x_s` and `x_s P^0 x_t` or symmetric, or both `R^0` with distinct bundles under SARP)
    diff_bundle = np.linalg.norm(bundles[:, None, :] - bundles[None, :, :], axis=-1) > 1e-7
    warp_sym = diff_bundle & ((r0 & p0.T) | (r0.T & p0))
    warp_pairs = int(np.sum(np.triu(warp_sym, k=1)))

    # SARP: `x_t R^0_neq x_s` iff `p_t . x_t >= p_t . x_s` and `x_t != x_s`; transitive closure has no symmetric cycles
    r0_neq = r0 & diff_bundle
    r_neq_star = transitive_closure_bool(r0_neq)
    sarp_viol = r_neq_star & r0_neq.T
    np.fill_diagonal(sarp_viol, False)
    sarp_pairs = int(np.sum(sarp_viol))
    return warp_pairs, sarp_pairs


def compute_exact_afriat_ccei(
    prices: np.ndarray,
    bundles: np.ndarray,
    tol: float = 1e-9,
) -> float:
    """Compute exact Afriat Critical Cost Efficiency Index `e* = sup {e in (0, 1] : GARP(e) holds}`.

    Because edges in `R^0(e)` and `P^0(e)` only change at threshold ratios `E[t, s] / E[t, t]`,
    evaluating sorted candidate thresholds via binary search recovers the exact supremum `e*`.
    """
    exp_mat = compute_expenditure_matrix(prices, bundles)
    n_v1, _ = count_garp_violations_at_e(exp_mat, e=1.0, tol=tol)
    if n_v1 == 0:
        return 1.0

    own_exp = np.diag(exp_mat)[:, None]
    if np.any(own_exp <= 0.0):
        return 0.0
    ratios = exp_mat / own_exp
    mask = (ratios > 0.0) & (ratios <= 1.0)
    cand_raw = np.unique(ratios[mask])
    # Just below each threshold ratio drops at least one strict/weak edge
    eps_step = 2.5 * tol
    candidates = np.unique(np.clip(np.concatenate([[0.0], cand_raw - eps_step]), 0.0, 1.0))

    lo = 0
    hi = len(candidates) - 1
    best_e = 0.0
    while lo <= hi:
        mid = (lo + hi) // 2
        e_try = float(candidates[mid])
        v_cnt, _ = count_garp_violations_at_e(exp_mat, e=e_try, tol=tol)
        if v_cnt == 0:
            best_e = e_try
            lo = mid + 1
        else:
            hi = mid - 1
    # Refine with 32 bisection steps between `best_e` and the next threshold for machine precision
    upper = 1.0
    higher = cand_raw[cand_raw > best_e]
    if higher.size > 0:
        upper = float(min(1.0, higher[0] + eps_step))
    a, b = best_e, upper
    for _ in range(32):
        m = 0.5 * (a + b)
        v_cnt, _ = count_garp_violations_at_e(exp_mat, e=m, tol=tol)
        if v_cnt == 0:
            a = m
        else:
            b = m
    return round(float(a), 6)


def construct_afriat_utility(
    prices: np.ndarray,
    bundles: np.ndarray,
    e: float | None = None,
    tol: float = 1e-9,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Construct explicit Afriat numbers `(u_t, lambda_t > 0)` satisfying Afriat's (1967) inequalities:

        `u_s <= u_t + lambda_t * (p_t . x_s - e * p_t . x_t)`   for all `t, s`.

    Uses Varian's (1982) constructive topological-level / Fostel-Scarf-Todd algorithm on
    the acyclic `GARP(e)` preference condensation graph.
    """
    exp_mat = compute_expenditure_matrix(prices, bundles)
    t_n = exp_mat.shape[0]
    if e is None:
        ccei = compute_exact_afriat_ccei(prices, bundles, tol=tol)
        e_use = 1.0 if ccei >= 1.0 else max(1e-6, ccei - 5e-6)
    else:
        e_use = float(e)

    # Ensure strictly GARP-consistent `e_use`
    for _ in range(12):
        v_cnt, _ = count_garp_violations_at_e(exp_mat, e=e_use, tol=tol)
        if v_cnt == 0:
            break
        e_use = max(1e-6, e_use - 1e-4)

    own_exp = np.diag(exp_mat)[:, None]
    # Difference matrix `D[t, s] = E[t, s] - e_use * E[t, t]`
    # Note: `x_t R^0(e) x_s` iff `D[t, s] <= tol`; `x_t P^0(e) x_s` iff `D[t, s] < -tol`.
    diff = exp_mat - e_use * own_exp
    np.fill_diagonal(diff, 0.0)

    r0 = diff <= tol
    r_star = transitive_closure_bool(r0)

    # Equivalence classes (SCCs): `i ~ j` iff `r_star[i, j] and r_star[j, i]`
    equiv = r_star & r_star.T
    visited = np.zeros(t_n, dtype=bool)
    sccs: list[list[int]] = []
    comp_of = np.zeros(t_n, dtype=int)
    for i in range(t_n):
        if not visited[i]:
            members = [j for j in range(t_n) if equiv[i, j]]
            cid = len(sccs)
            for j in members:
                visited[j] = True
                comp_of[j] = cid
            sccs.append(members)

    n_c = len(sccs)
    # Build DAG on SCCs: edge `A -> B` (`A` preferred to `B`) if some `i in A, j in B` has `r_star[i, j]`
    dag = np.zeros((n_c, n_c), dtype=bool)
    for ca in range(n_c):
        ia = sccs[ca][0]
        for cb in range(n_c):
            if ca != cb:
                ib = sccs[cb][0]
                if r_star[ia, ib]:
                    dag[ca, cb] = True

    # Topological order from BEST (no incoming edges in `dag`) to WORST (Fostel-Scarf-Todd 2004):
    # `dag[A, B] == True` means `A` is revealed preferred to `B`.
    # If `C` has all incoming parents `{P : dag[P, C]}` already placed, then NO `t in C` can
    # reveal-prefer any `s in placed` (otherwise `dag[C, P]` would hold, contradicting placement order),
    # guaranteeing `diff[t, s] > tol > 0` strictly for every `t in C` and every `s in placed`!
    placed = np.zeros(n_c, dtype=bool)
    best_to_worst: list[int] = []
    for _ in range(n_c):
        chosen = -1
        for c in range(n_c):
            if not placed[c]:
                in_unplaced = [p_c for p_c in range(n_c) if dag[p_c, c] and not placed[p_c]]
                if not in_unplaced:
                    chosen = c
                    break
        if chosen < 0:
            chosen = int(np.where(~placed)[0][0])
        placed[chosen] = True
        best_to_worst.append(chosen)

    u = np.zeros(t_n, dtype=np.float64)
    lam = np.ones(t_n, dtype=np.float64)
    placed_obs: list[int] = []

    for step_idx, cid in enumerate(best_to_worst):
        members = sccs[cid]
        if step_idx == 0:
            for t in members:
                u[t] = 0.0
                lam[t] = 1.0
        else:
            # 1. Satisfy `u_C <= u[s] + lam[s] * diff[s, t]` for all `s in placed_obs, t in members`
            upper_bounds = [u[s] + lam[s] * diff[s, t] for s in placed_obs for t in members]
            upper_bounds.append(min(u[s] for s in placed_obs))
            u_c = float(min(upper_bounds) - 1.0)
            for t in members:
                u[t] = u_c
                # 2. Since `diff[t, s] > tol > 0` for all `s in placed_obs`, pick `lam[t]` large enough
                #    that `u[s] - u_c <= lam[t] * diff[t, s]` holds for every `s in placed_obs`.
                req_lams = [(u[s] - u_c) / max(diff[t, s], 1e-12) for s in placed_obs]
                lam[t] = float(max(1.0, max(req_lams) * 1.02 + 1e-3))
        placed_obs.extend(members)

    # Normalize `u` to `[0, 1]` and scale `lam` by the exact same positive factor `1 / u_span`
    u_min = float(np.min(u))
    u_span = float(np.max(u) - u_min)
    if u_span > 1e-12:
        u = (u - u_min) / u_span
        lam = lam / u_span

    # Verify exact Afriat inequality residual: `max_{t, s} (u_s - u_t - lam_t * (E[t, s] - e_use * E[t, t]))`
    lhs = u[None, :] - u[:, None]
    eff_diff = np.where((diff < 0.0) & (diff >= -2.0 * tol), 0.0, diff)
    max_viol = float(np.max(np.maximum(0.0, lhs - lam[:, None] * eff_diff)))
    return u, lam, max_viol, e_use


def audit_revealed_preferences(
    prices: np.ndarray,
    bundles: np.ndarray,
    tol: float = 1e-9,
) -> GARPAuditResult:
    """Run full GARP, WARP, SARP, exact Afriat CCEI, and Afriat utility recovery."""
    p = np.asarray(prices, dtype=np.float64)
    x = np.asarray(bundles, dtype=np.float64)
    exp_mat = compute_expenditure_matrix(p, x)
    garp_v, _ = count_garp_violations_at_e(exp_mat, e=1.0, tol=tol)
    warp_v, sarp_v = count_warp_and_sarp_violations(exp_mat, x, tol=tol)
    ccei = compute_exact_afriat_ccei(p, x, tol=tol)
    u, lam, max_viol, e_used = construct_afriat_utility(p, x, e=1.0 if garp_v == 0 else max(1e-6, ccei - 5e-6), tol=tol)
    return GARPAuditResult(
        n_obs=int(p.shape[0]),
        passes_garp=(garp_v == 0),
        garp_violation_pairs=garp_v,
        warp_violation_pairs=warp_v,
        sarp_violation_pairs=sarp_v,
        afriat_ccei=ccei,
        afriat_utility_u=[round(float(v), 6) for v in u],
        afriat_multiplier_lambda=[round(float(v), 6) for v in lam],
        max_afriat_inequality_violation=float(max_viol),
        effective_efficiency_used=round(float(e_used), 6),
    )


def compute_bronars_power(
    max_intercepts: np.ndarray,
    grid_shares: Sequence[float] = tuple(round(0.1 * k, 2) for k in range(11)),
    n_simulations: int = 1000,
    seed: int = 20261121,
) -> dict[str, Any]:
    """Bronars (1987) Monte Carlo benchmark: uniform random discrete choice on the budget set.

    `max_intercepts` has shape `(T, 2)` giving `(M_{t,1}, M_{t,2})`, so `p_t = (1/M_{t,1}, 1/M_{t,2})`
    with normalized expenditure `p_t . x_t = 1`.
    """
    m = np.asarray(max_intercepts, dtype=np.float64)
    t_n = m.shape[0]
    prices = 1.0 / m
    shares = np.asarray(grid_shares, dtype=np.float64)
    rng = np.random.default_rng(seed)

    cceis = np.zeros(n_simulations, dtype=np.float64)
    pass_flags = np.zeros(n_simulations, dtype=bool)
    viol_counts = np.zeros(n_simulations, dtype=np.int64)

    for sim in range(n_simulations):
        idx = rng.integers(0, len(shares), size=t_n)
        th = shares[idx]
        bundles = np.column_stack([th * m[:, 0], (1.0 - th) * m[:, 1]])
        exp_mat = prices @ bundles.T
        v_cnt, _ = count_garp_violations_at_e(exp_mat, e=1.0)
        viol_counts[sim] = v_cnt
        pass_flags[sim] = (v_cnt == 0)
        cceis[sim] = 1.0 if v_cnt == 0 else compute_exact_afriat_ccei(prices, bundles)

    return {
        "n_simulations": int(n_simulations),
        "n_budget_lines": int(t_n),
        "frac_random_pass_garp": round(float(np.mean(pass_flags)), 4),
        "bronars_power_1_minus_pass": round(float(1.0 - np.mean(pass_flags)), 4),
        "mean_random_ccei": round(float(np.mean(cceis)), 4),
        "q95_random_ccei": round(float(np.quantile(cceis, 0.95)), 4),
        "q99_random_ccei": round(float(np.quantile(cceis, 0.99)), 4),
        "mean_random_garp_violations": round(float(np.mean(viol_counts)), 2),
        "sample_cceis": cceis,  # caller strips or summarizes as needed
    }
