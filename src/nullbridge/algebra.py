"""Core linear algebra for Orthogonal Procrustes loop bridges and zero-FLOP null-space folding.

Implements:
  1. RMS-standardized SVD Orthogonal Procrustes (`R_proc = U V^T in O(d)`).
  2. Contrastive concept basis extraction with leave-one-pair-out (`LOPO`) validation
     and QR orthogonalization against nuisance/epistemic controls.
  3. Offline zero-FLOP weight folding:
       - Procrustes loop bridge: `R_safe = R_proc (I - Q Q^T)` and `A_safe = A (I - Q Q^T)`
       - Cross-pass rank-m message bank: `W_msg_safe` (`m <- m (I - Q_msg Q_msg^T)`)
       - Residual writer left-projection: `W_safe = (I - Q Q^T) W`
  4. Strict orthogonal-complement random basis controls (`Q_rand^T Q = 0`, `Q_rand^T Q_rand = I_r`)
     and exact sign-flip / Holm-Bonferroni / bootstrap inference helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np

EPS = 1e-12


@dataclass(frozen=True)
class ProcrustesFit:
    """Closed-form RMS-standardized Orthogonal Procrustes map `h_dst_hat = rms_dst * ((h_src - mu_src)/rms_src) @ r_proc.T + mu_dst`."""

    r_proc: np.ndarray            # [D, D] orthogonal matrix (`r_proc.T @ r_proc = I_D`)
    mu_src: np.ndarray            # [D]
    mu_dst: np.ndarray            # [D]
    rms_src: float
    rms_dst: float
    d_model: int
    rel_err_identity: float
    rel_err_procrustes: float
    mean_cosine_identity: float
    mean_cosine_procrustes: float


@dataclass(frozen=True)
class ConceptBasis:
    """Extracted concept directions across layers `[L, D]` with LOPO diagnostics and control orthogonalization."""

    concepts: tuple[str, ...]
    controls: tuple[str, ...]
    raw_dirs: dict[str, np.ndarray]         # concept -> [L, D] unit vectors
    controlled_dirs: dict[str, np.ndarray]  # core concept -> [L, D] unit vectors orthogonal to span(controls)
    residual_ratio: dict[str, np.ndarray]   # core concept -> [L] norm retained after control projection
    lopo_accuracy: dict[str, np.ndarray]    # concept -> [L] leave-one-pair-out sign accuracy
    lopo_cosine: dict[str, np.ndarray]      # concept -> [L] leave-one-pair-out cosine similarity
    contrast_rms: dict[str, np.ndarray]     # concept -> [L] RMS of (pos - neg)
    stream_rms: np.ndarray                  # [L] RMS of pooled activations


@dataclass(frozen=True)
class FoldedBridgeWeights:
    """Zero-FLOP null-space folded bridge matrices (`0` extra parameters, `0` extra runtime FLOPs)."""

    r_safe: np.ndarray                  # [D, D] = R_proc @ (I - Q Q^T)
    lora_a_safe: np.ndarray | None      # [rank, D] = A @ (I - Q Q^T)
    mu_src_safe: np.ndarray | None      # [D]
    q_basis: np.ndarray                 # [D, r] orthonormal columns
    rank_removed: int
    max_null_leakage_r: float           # ||R_safe @ Q||_F
    max_null_leakage_a: float           # ||A_safe @ Q||_F
    max_complement_isometry_err: float  # ||R_safe @ Q_perp - R_proc @ Q_perp||_F / sqrt(d - r)
    singular_values_head: tuple[float, ...]
    singular_values_tail: tuple[float, ...]


def _unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    norms = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norms, EPS)


def solve_orthogonal_procrustes(h_src: np.ndarray, h_dst: np.ndarray) -> ProcrustesFit:
    """Solve `min_{R in O(d)} || Y_n - X_n R^T ||_F` via exact SVD `Y_n^T X_n = U S V^T => R = U V^T`."""
    x = np.asarray(h_src, dtype=np.float64)
    y = np.asarray(h_dst, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape != y.shape:
        raise ValueError(f"Expected matching 2D arrays [N, D], got {x.shape} vs {y.shape}")
    d = int(x.shape[1])

    mu_src = np.mean(x, axis=0)
    mu_dst = np.mean(y, axis=0)
    x_c = x - mu_src
    y_c = y - mu_dst

    rms_src = float(np.sqrt(np.mean(x_c * x_c) + EPS))
    rms_dst = float(np.sqrt(np.mean(y_c * y_c) + EPS))
    x_n = x_c / rms_src
    y_n = y_c / rms_dst

    u_mat, _, vt_mat = np.linalg.svd(y_n.T @ x_n, full_matrices=False)
    r_0 = u_mat @ vt_mat

    recon = rms_dst * (x_n @ r_0.T) + mu_dst
    rel_err_proc = float(np.linalg.norm(y - recon) / max(np.linalg.norm(y), 1e-9))
    rel_err_id = float(np.linalg.norm(y - x) / max(np.linalg.norm(y), 1e-9))

    cos_id = float(
        np.mean(
            np.sum(x_n * y_n, axis=-1)
            / (np.linalg.norm(x_n, axis=-1) * np.linalg.norm(y_n, axis=-1) + EPS)
        )
    )
    x_rot = x_n @ r_0.T
    cos_proc = float(
        np.mean(
            np.sum(x_rot * y_n, axis=-1)
            / (np.linalg.norm(x_rot, axis=-1) * np.linalg.norm(y_n, axis=-1) + EPS)
        )
    )

    return ProcrustesFit(
        r_proc=r_0.astype(np.float32),
        mu_src=mu_src.astype(np.float32),
        mu_dst=mu_dst.astype(np.float32),
        rms_src=rms_src,
        rms_dst=rms_dst,
        d_model=d,
        rel_err_identity=round(rel_err_id, 6),
        rel_err_procrustes=round(rel_err_proc, 6),
        mean_cosine_identity=round(cos_id, 6),
        mean_cosine_procrustes=round(cos_proc, 6),
    )


def build_orthonormal_basis(
    directions: Sequence[np.ndarray] | np.ndarray,
    neutral_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build orthonormal columns `Q [D, r]` via reduced QR and optional neutral offset `mu_proj [D]`."""
    arr = np.asarray(directions, dtype=np.float64)
    if arr.ndim == 1:
        cols = arr[:, None]
    elif arr.ndim == 2:
        # Accept either [r, D] (list of row vectors) or [D, r] (column matrix) when r != D;
        # default convention for stacked unit vectors is [r, D] -> transpose when shape[0] <= shape[1].
        cols = arr.T if arr.shape[0] <= arr.shape[1] else arr
    else:
        raise ValueError(f"Expected 1D or 2D directions, got shape {arr.shape}")

    q_np, r_np = np.linalg.qr(cols, mode="reduced")
    # Stabilize column signs deterministically so QR sign conventions never vary across BLAS backends
    diag = np.diag(r_np)
    signs = np.where(diag >= 0.0, 1.0, -1.0)
    q_np = q_np * signs[None, :]

    if neutral_mean is not None:
        mu_vec = np.asarray(neutral_mean, dtype=np.float64).reshape(-1)
        if mu_vec.shape[0] != q_np.shape[0]:
            raise ValueError(f"neutral_mean length {mu_vec.shape[0]} != d={q_np.shape[0]}")
        mu_proj = (mu_vec @ q_np) @ q_np.T
    else:
        mu_proj = np.zeros(q_np.shape[0], dtype=np.float64)

    return q_np.astype(np.float32), mu_proj.astype(np.float32)


def sample_random_orthonormal_basis(
    d_model: int,
    rank: int,
    exclude_q: np.ndarray | None = None,
    seed: int = 20261120,
) -> np.ndarray:
    """Sample a rank-`r` random orthonormal basis `Q_rand [D, r]` strictly orthogonal to `exclude_q`."""
    if rank <= 0 or rank >= d_model:
        raise ValueError(f"Invalid rank={rank} for d_model={d_model}")
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d_model, rank)).astype(np.float64)
    if exclude_q is not None:
        q_ex = np.asarray(exclude_q, dtype=np.float64)
        if q_ex.ndim == 1:
            q_ex = q_ex[:, None]
        if q_ex.shape[0] != d_model:
            q_ex = q_ex.T
        if q_ex.shape[1] + rank > d_model:
            raise ValueError("exclude_q rank + requested rank exceeds d_model")
        g = g - q_ex @ (q_ex.T @ g)
    q_rand, r_mat = np.linalg.qr(g, mode="reduced")
    signs = np.where(np.diag(r_mat) >= 0.0, 1.0, -1.0)
    q_rand = q_rand * signs[None, :]
    return q_rand.astype(np.float32)


def fold_procrustes_bridge(
    r_proc: np.ndarray,
    q_basis: np.ndarray,
    lora_a: np.ndarray | None = None,
    mu_src: np.ndarray | None = None,
    neutral_mean: np.ndarray | None = None,
) -> FoldedBridgeWeights:
    """Fold `P_perp = I - Q Q^T` offline into `R_proc` and `lora_a` at 0 extra params and 0 extra FLOPs.

    Algebraic guarantees:
      - `R_safe @ q = 0` and `A_safe @ q = 0` for every `q in col(Q)`.
      - `R_safe @ v = R_proc @ v` isometrically (`||R_safe v||_2 = ||v||_2`) for every `v in col(Q)^perp`.
      - Singular spectrum of `R_safe` is `(1, ..., 1, 0, ..., 0)` with `d - r` ones and `r` zeros.
    """
    r64 = np.asarray(r_proc, dtype=np.float64)
    q64, _ = build_orthonormal_basis(q_basis)
    q64 = q64.astype(np.float64)
    d = int(r64.shape[0])
    r = int(q64.shape[1])
    if r64.shape != (d, d):
        raise ValueError(f"r_proc must be square [D, D], got {r64.shape}")

    # R_safe = R_proc @ (I - Q Q^T) = R_proc - (R_proc @ Q) @ Q^T  (O(d^2 r) offline, 0 runtime FLOPs)
    r_safe64 = r64 - (r64 @ q64) @ q64.T
    leak_r = float(np.linalg.norm(r_safe64 @ q64, ord="fro"))

    a_safe_f32: np.ndarray | None = None
    leak_a = 0.0
    if lora_a is not None:
        a64 = np.asarray(lora_a, dtype=np.float64)
        if a64.shape[1] != d:
            raise ValueError(f"lora_a shape {a64.shape} incompatible with d={d}")
        a_safe64 = a64 - (a64 @ q64) @ q64.T
        leak_a = float(np.linalg.norm(a_safe64 @ q64, ord="fro"))
        a_safe_f32 = a_safe64.astype(np.float32)

    mu_safe_f32: np.ndarray | None = None
    if mu_src is not None:
        mu64 = np.asarray(mu_src, dtype=np.float64).reshape(-1)
        if neutral_mean is not None:
            neu64 = np.asarray(neutral_mean, dtype=np.float64).reshape(-1)
            # Replace mu_src's component in col(Q) with neutral_mean's component (which R_safe annihilates anyway,
            # while keeping RMS denominator centered on neutral along col(Q)).
            mu64 = mu64 - (mu64 @ q64) @ q64.T + (neu64 @ q64) @ q64.T
        mu_safe_f32 = mu64.astype(np.float32)

    # Verify exact isometry on orthogonal complement via random test subspace (or full complement if d <= 512)
    n_probe = min(max(1, d - r), 64)
    q_comp = sample_random_orthonormal_basis(d, n_probe, exclude_q=q64, seed=99173).astype(np.float64)
    iso_err = float(np.linalg.norm((r_safe64 - r64) @ q_comp, ord="fro") / np.sqrt(n_probe))

    if d <= 1024:
        sv = np.linalg.svd(r_safe64, compute_uv=False)
        sv_head = tuple(round(float(x), 6) for x in sv[: min(4, len(sv))])
        sv_tail = tuple(round(float(x), 8) for x in sv[-min(r + 2, len(sv)) :])
    else:
        # Fast Gram check without O(d^3) SVD on 5376x5376 during online GPU runs
        norms_comp = np.linalg.norm(r_safe64 @ q_comp, axis=0)
        norms_null = np.linalg.norm(r_safe64 @ q64, axis=0)
        sv_head = tuple(round(float(x), 6) for x in norms_comp[: min(4, len(norms_comp))])
        sv_tail = tuple(round(float(x), 8) for x in norms_null)

    return FoldedBridgeWeights(
        r_safe=r_safe64.astype(np.float32),
        lora_a_safe=a_safe_f32,
        mu_src_safe=mu_safe_f32,
        q_basis=q64.astype(np.float32),
        rank_removed=r,
        max_null_leakage_r=leak_r,
        max_null_leakage_a=leak_a,
        max_complement_isometry_err=iso_err,
        singular_values_head=sv_head,
        singular_values_tail=sv_tail,
    )


def fold_message_bank_projection(
    w_msg: np.ndarray,
    q_basis: np.ndarray,
    side: str = "input",
) -> tuple[np.ndarray, float]:
    """Fold null-space projector into a cross-pass message bank matrix (`Inkling 2 Mini` / `TML-d24-varloop`).

    Args:
        w_msg: Either `W_in [r_msg, D]` (side='input', mapping hidden state `h [D]` -> `tanh(h @ W_in.T)`)
               or `W_out [D, r_msg]` / `[r_msg, r_msg]` (side='output', projecting message rows out of `Q`).
        q_basis: Orthonormal basis `[D, r]` (for input/residual) or `[r_msg, r]` (inside rank-64 message space).
        side: 'input' computes `W_safe = W_msg @ (I - Q Q^T)`; 'output' computes `W_safe = (I - Q Q^T) @ W_msg`.
    """
    w64 = np.asarray(w_msg, dtype=np.float64)
    q64, _ = build_orthonormal_basis(q_basis)
    q64 = q64.astype(np.float64)

    if side == "input":
        if w64.shape[1] != q64.shape[0]:
            raise ValueError(f"w_msg shape {w64.shape} incompatible with q_basis {q64.shape} on side='input'")
        w_safe = w64 - (w64 @ q64) @ q64.T
        leak = float(np.linalg.norm(w_safe @ q64, ord="fro"))
    elif side == "output":
        if w64.shape[0] != q64.shape[0]:
            raise ValueError(f"w_msg shape {w64.shape} incompatible with q_basis {q64.shape} on side='output'")
        w_safe = w64 - q64 @ (q64.T @ w64)
        leak = float(np.linalg.norm(q64.T @ w_safe, ord="fro"))
    else:
        raise ValueError(f"side must be 'input' or 'output', got {side!r}")

    return w_safe.astype(np.float32), leak


def fold_linear_writer(
    weight: np.ndarray,
    q_basis: np.ndarray,
    bias: np.ndarray | None = None,
    neutral_mean: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, float]:
    """Left-multiply an output writer `W [D_out, D_in]` by `(I - Q Q^T)` so its output cannot write into `col(Q)`."""
    w64 = np.asarray(weight, dtype=np.float64)
    q64, mu_proj = build_orthonormal_basis(q_basis, neutral_mean=neutral_mean)
    q64 = q64.astype(np.float64)
    if w64.shape[0] != q64.shape[0]:
        raise ValueError(f"Writer out_features {w64.shape[0]} != q_basis d={q64.shape[0]}")

    w_safe = w64 - q64 @ (q64.T @ w64)
    leak = float(np.linalg.norm(q64.T @ w_safe, ord="fro"))
    b_safe: np.ndarray | None = None
    if bias is not None:
        b64 = np.asarray(bias, dtype=np.float64).reshape(-1)
        b_safe = (b64 - (b64 @ q64) @ q64.T + mu_proj.astype(np.float64)).astype(np.float32)
    elif neutral_mean is not None:
        b_safe = mu_proj.astype(np.float32)
    return w_safe.astype(np.float32), b_safe, leak


def extract_controlled_basis(
    pos_by_concept: dict[str, np.ndarray],
    neg_by_concept: dict[str, np.ndarray],
    concepts: tuple[str, ...],
    controls: tuple[str, ...] = (),
) -> ConceptBasis:
    """Extract raw & control-orthogonalized unit directions `[L, D]` with LOPO accuracy & cosine."""
    sample = next(iter(pos_by_concept.values()))
    _, n_layers, d_model = sample.shape

    raw_dirs: dict[str, np.ndarray] = {}
    lopo_acc: dict[str, np.ndarray] = {}
    lopo_cos: dict[str, np.ndarray] = {}
    contrast_rms: dict[str, np.ndarray] = {}

    stream_sq_sum = np.zeros(n_layers, dtype=np.float64)
    stream_count = 0

    for c in concepts:
        pos = np.asarray(pos_by_concept[c], dtype=np.float64)
        neg = np.asarray(neg_by_concept[c], dtype=np.float64)
        if pos.shape != neg.shape:
            raise ValueError(f"Shape mismatch for {c!r}: {pos.shape} vs {neg.shape}")
        n_pairs = int(pos.shape[0])
        diffs = pos - neg  # [N, L, D]

        stream_sq_sum += np.sum(np.linalg.norm(pos, axis=-1) ** 2 + np.linalg.norm(neg, axis=-1) ** 2, axis=0)
        stream_count += 2 * n_pairs

        mean_diff = np.mean(diffs, axis=0)  # [L, D]
        raw_dirs[c] = _unit(mean_diff, axis=-1).astype(np.float32)
        contrast_rms[c] = np.sqrt(np.mean(np.sum(diffs * diffs, axis=-1), axis=0)).astype(np.float32)

        if n_pairs >= 2:
            total_sum = np.sum(diffs, axis=0, keepdims=True)
            loo_dirs = _unit(total_sum - diffs, axis=-1)
            diff_units = _unit(diffs, axis=-1)
            cos_per_pair = np.sum(diff_units * loo_dirs, axis=-1)
            lopo_acc[c] = np.mean(cos_per_pair > 0.0, axis=0).astype(np.float32)
            lopo_cos[c] = np.mean(cos_per_pair, axis=0).astype(np.float32)
        else:
            lopo_acc[c] = np.ones(n_layers, dtype=np.float32)
            lopo_cos[c] = np.ones(n_layers, dtype=np.float32)

    stream_rms = np.sqrt(stream_sq_sum / max(1, stream_count)).astype(np.float32)

    controlled_dirs: dict[str, np.ndarray] = {}
    residual_ratio: dict[str, np.ndarray] = {}
    core_concepts = [c for c in concepts if c not in controls]

    for c in core_concepts:
        ctrl_vecs = np.zeros((n_layers, d_model), dtype=np.float64)
        ratios = np.zeros(n_layers, dtype=np.float64)
        for ell in range(n_layers):
            v = raw_dirs[c][ell].astype(np.float64)
            if controls:
                ctrl_mat = np.column_stack([raw_dirs[ct][ell].astype(np.float64) for ct in controls])
                q_ctrl, _ = np.linalg.qr(ctrl_mat, mode="reduced")
                v_perp = v - q_ctrl @ (q_ctrl.T @ v)
            else:
                v_perp = v
            nrm = float(np.linalg.norm(v_perp))
            ratios[ell] = nrm
            ctrl_vecs[ell] = v_perp / max(nrm, EPS)
        controlled_dirs[c] = ctrl_vecs.astype(np.float32)
        residual_ratio[c] = ratios.astype(np.float32)

    return ConceptBasis(
        concepts=concepts,
        controls=controls,
        raw_dirs=raw_dirs,
        controlled_dirs=controlled_dirs,
        residual_ratio=residual_ratio,
        lopo_accuracy=lopo_acc,
        lopo_cosine=lopo_cos,
        contrast_rms=contrast_rms,
        stream_rms=stream_rms,
    )


def exact_or_mc_signflip_p(
    values: Sequence[float] | np.ndarray,
    two_sided: bool = True,
    max_exact_n: int = 16,
    n_mc: int = 20_000,
    seed: int = 20261120,
) -> float:
    """Exact `2^N` paired sign-flip permutation p-value (`N <= 16`) or seeded Monte Carlo (`N > 16`)."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return 1.0
    obs = float(np.mean(x))
    if abs(obs) < EPS:
        return 1.0

    if n <= max_exact_n:
        bits = ((np.arange(1 << n, dtype=np.uint32)[:, None] >> np.arange(n, dtype=np.uint32)) & 1)
        signs = 2.0 * bits.astype(np.float64) - 1.0
        null_means = (signs @ x) / n
        if two_sided:
            extreme = int(np.sum(np.abs(null_means) >= abs(obs) - 1e-15))
        else:
            extreme = int(np.sum(null_means >= obs - 1e-15)) if obs > 0 else int(np.sum(null_means <= obs + 1e-15))
        return float(extreme / (1 << n))

    rng = np.random.default_rng(seed)
    signs = rng.choice((-1.0, 1.0), size=(n_mc, n))
    null_means = (signs @ x) / n
    if two_sided:
        extreme = int(np.sum(np.abs(null_means) >= abs(obs) - 1e-15))
    else:
        extreme = int(np.sum(null_means >= obs - 1e-15)) if obs > 0 else int(np.sum(null_means <= obs + 1e-15))
    return float((extreme + 1) / (n_mc + 1))


def holm_adjust(p_values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Step-down Holm-Bonferroni FWER adjustment."""
    p = np.asarray(p_values, dtype=np.float64)
    m = int(p.size)
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    scaled = ranked * (m - np.arange(m))
    enforced = np.maximum.accumulate(scaled)
    clipped = np.clip(enforced, 0.0, 1.0)
    out = np.empty_like(clipped)
    out[order] = clipped
    return out


def bootstrap_mean_ci(
    values: Sequence[float] | np.ndarray,
    n_boot: int = 5_000,
    alpha: float = 0.05,
    seed: int = 20261120,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for a sample mean."""
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    n = int(x.size)
    if n == 0:
        return (0.0, 0.0)
    if n == 1:
        return (float(x[0]), float(x[0]))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = np.mean(x[idx], axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return (float(lo), float(hi))


def audit_algebraic_properties(d_model: int = 256, rank: int = 4, lora_rank: int = 16, r_msg: int = 64) -> dict[str, Any]:
    """Run deterministic Phase 0 mathematical verification of Procrustes & Nullbridge operators."""
    rng = np.random.default_rng(20261120)
    # Plant a ground-truth orthogonal transformation + noise
    g_true, _ = np.linalg.qr(rng.standard_normal((d_model, d_model)))
    x_src = rng.standard_normal((512, d_model)) * 1.7 + 0.4
    y_dst = 0.9 * ((x_src - 0.4) / 1.7) @ g_true.T - 0.2 + 0.02 * rng.standard_normal((512, d_model))

    proc = solve_orthogonal_procrustes(x_src, y_dst)
    raw_dirs = rng.standard_normal((rank, d_model))
    lora_a = rng.standard_normal((lora_rank, d_model)) * (1.0 / np.sqrt(d_model))
    lora_b = rng.standard_normal((d_model, lora_rank)) * 0.05
    folded = fold_procrustes_bridge(proc.r_proc, raw_dirs, lora_a=lora_a, mu_src=proc.mu_src)

    # Perturbation invariance check: B_linear(x + alpha * q) == B_linear(x) for any alpha, q in col(Q)
    x_test = rng.standard_normal((32, d_model)).astype(np.float64)
    q64 = folded.q_basis.astype(np.float64)
    alpha_coef = rng.uniform(-25.0, 25.0, size=(32, rank))
    x_perturbed = x_test + alpha_coef @ q64.T

    r_s = folded.r_safe.astype(np.float64)
    a_s = folded.lora_a_safe.astype(np.float64)  # type: ignore[union-attr]
    b64 = lora_b.astype(np.float64)

    out_clean = x_test @ r_s.T + (x_test @ a_s.T) @ b64.T
    out_pert = x_perturbed @ r_s.T + (x_perturbed @ a_s.T) @ b64.T
    max_pert_diff = float(np.max(np.abs(out_clean - out_pert)))

    # Random orthogonal control orthogonality check
    q_rand = sample_random_orthonormal_basis(d_model, rank, exclude_q=q64, seed=42)
    cross_overlap = float(np.linalg.norm(q64.T @ q_rand.astype(np.float64), ord="fro"))

    # Rank-64 cross-pass message bank fold check
    w_msg = rng.standard_normal((r_msg, d_model))
    w_msg_safe, msg_leak = fold_message_bank_projection(w_msg, q64, side="input")
    msg_clean = np.tanh(x_test @ w_msg_safe.astype(np.float64).T)
    msg_pert = np.tanh(x_perturbed @ w_msg_safe.astype(np.float64).T)
    max_msg_pert_diff = float(np.max(np.abs(msg_clean - msg_pert)))

    sv = np.linalg.svd(r_s, compute_uv=False)
    n_unit_sv = int(np.sum(np.isclose(sv, 1.0, atol=1e-5)))
    n_zero_sv = int(np.sum(np.isclose(sv, 0.0, atol=1e-6)))

    valid = (
        proc.mean_cosine_procrustes > 0.999
        and folded.max_null_leakage_r < 1e-6
        and folded.max_null_leakage_a < 1e-6
        and folded.max_complement_isometry_err < 1e-6
        and max_pert_diff < 5e-6
        and max_msg_pert_diff < 5e-6
        and cross_overlap < 1e-6
        and n_unit_sv == d_model - rank
        and n_zero_sv == rank
    )

    return {
        "valid": valid,
        "d_model": d_model,
        "rank_removed": rank,
        "retained_fraction": round((d_model - rank) / d_model, 6),
        "procrustes_cosine_before": proc.mean_cosine_identity,
        "procrustes_cosine_after": proc.mean_cosine_procrustes,
        "max_null_leakage_r": folded.max_null_leakage_r,
        "max_null_leakage_a": folded.max_null_leakage_a,
        "max_complement_isometry_err": folded.max_complement_isometry_err,
        "max_linear_perturbation_diff": max_pert_diff,
        "max_message_bank_perturbation_diff": max_msg_pert_diff,
        "message_bank_frobenius_leak": msg_leak,
        "random_control_cross_overlap": cross_overlap,
        "n_unit_singular_values": n_unit_sv,
        "n_zero_singular_values": n_zero_sv,
        "singular_values_tail": list(folded.singular_values_tail),
        "extra_parameters": 0,
        "extra_runtime_flops": 0,
    }
