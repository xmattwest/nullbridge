"""Phase 7 Experiment Block B2: Cross-Architecture Biophysical Dissection of `tomg-group-umd/huginn-0125`.

Implements `BiophysicalHuginnController` for `RavenForCausalLM` (2 prelude + 4 recurrent core + 2 coda,
`n_embd = 5,280`, linear state/recall adapter `transformer.adapter(cat([x, input_embeds], dim=-1))`,
`ln_f`, `lm_head`) and all 5 Huginn biophysical experiments (`Exp H1..H5`):

  - Preflight Equivalence: `unrolled_huginn_forward(n_steps=K)` vs `model.forward(num_steps=K)` at
    `K = 1, 4, 8` with matched deterministic `"like-init"` state `x_0` (`<= 1e-3`).
  - Zero-State Sensitivity Check: `"like-init"` (`init_scale = 1.0`) vs zero-state (`init_scale = 0.0`)
    at `K = 8` (`n = 20`).
  - Baseline Calibration Check: zero-shot `K = 8` accuracy on `factual_2hop + arithmetic_2step`;
    automatically switches to a deterministic 2-shot in-context prefix if accuracy `< 30%`.
  - Exp H1: 2-channel `(b_state, b_recall)` 4-mask adapter WHT at `K = 4, 8` with exact Parseval `< 1e-12`.
  - Exp H2: Unrolled visit lens, truncated Coda readout, Coda conductance (`p = 1, 2, 4, 8`), and
    6-step VJP/JVP `sigma_max` non-normal transient gain across `p = 1, 2, 3, 4, 6, 8`.
  - Exp H3: Hutchinson VJP Fisher pullback & Hop-1/Hop-2 commitment horizon at `K = 8`.
  - Exp H4: Labovich Pass-1-only foil interpolation (`beta in {0.0..1.25}`) at `K = 4, 8` plus
    orthogonal tangent pulse-chase (`alpha in {0.0, 0.25, 0.50, 1.00, 2.00}`) at `K = 8`.
  - Exp H5: Continuous adapter recurrent-state gain sweep (`gamma in {0.0..1.50}`) at `K = 8`.
"""

from __future__ import annotations

import gc
import json
import math
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - local CPU venv without torch
    torch = None  # type: ignore[assignment]
    nn = Any  # type: ignore[assignment,misc]
    F = None  # type: ignore[assignment]

from .algebra import exact_or_mc_signflip_p, holm_adjust
from .biophysical_inkling import first_token_variant_ids
from .biophysical_stimuli import BiophysicalItem, build_phase6_stimulus_suite
from .biophysical_wht import (
    estimate_commitment_horizon,
    mean_and_ci as bootstrap_mean_ci,
    summarize_hysteresis_curve,
)
from .track_a import _atomic_save_json, _no_grad


HUGINN_CHANNEL_NAMES: tuple[str, str] = ("state", "recall")

# Non-overlapping 2-shot prefixes (disjoint from all 120 Phase 6 items)
_TWO_SHOT_PREFIXES: dict[str, str] = {
    "factual_2hop": (
        "Fact: The national capital city of the country where the Statue of Liberty is located is Washington\n"
        "Fact: The national capital city of the country where the Sydney Opera House is located is Canberra\n"
    ),
    "arithmetic_2step": (
        "Expression: (2 * 3) + 4 = 10\n"
        "Expression: (5 * 2) - 3 = 7\n"
    ),
    "calendar_2step": (
        "Calendar sequence: Start on Monday, advance 1 day, then advance 1 more day. Final day: Wednesday\n"
        "Calendar sequence: Start in January, advance 2 months, then advance 1 more month. Final month: April\n"
    ),
}


# ---------------------------------------------------------------------------
# Pure-NumPy Exact 4-Term Walsh-Hadamard Transform for 2-Channel Adapter
# ---------------------------------------------------------------------------

def mask4_int_to_bits(mask_int: int) -> tuple[int, int]:
    """Map `mask_int in 0..3` to `(b_state, b_recall)` in `{0, 1}^2`."""
    if not (0 <= mask_int < 4):
        raise ValueError(f"mask_int must be in 0..3, got {mask_int}")
    return ((mask_int >> 1) & 1, mask_int & 1)


def enumerate_4_masks() -> list[tuple[int, tuple[int, int], str, str]]:
    """Return all 4 binary masks `(mask_int, bits, code_str, label)` ordered `0..3`."""
    labels = {
        (0, 0): "None (00)",
        (0, 1): "recall",
        (1, 0): "state",
        (1, 1): "All (state+recall)",
    }
    out: list[tuple[int, tuple[int, int], str, str]] = []
    for m in range(4):
        bits = mask4_int_to_bits(m)
        code = f"{bits[0]}{bits[1]}"
        out.append((m, bits, code, labels[bits]))
    return out


def wht_subsets_4() -> list[tuple[tuple[int, ...], str, int]]:
    """Canonical ordering of all 4 subsets of `{state, recall}` by interaction order `0..2`."""
    return [
        ((), "const", 0),
        ((0,), "state", 1),
        ((1,), "recall", 1),
        ((0, 1), "statexrecall", 2),
    ]


def build_walsh_hadamard_4() -> np.ndarray:
    """Build orthogonal `4 x 4` spin basis `H` satisfying `(1/4) H^T H = I_4`."""
    subsets = wht_subsets_4()
    mat = np.ones((4, 4), dtype=np.float64)
    for m in range(4):
        bits = mask4_int_to_bits(m)
        spins = [1.0 if b == 1 else -1.0 for b in bits]
        for k, (comb, _lbl, _ord) in enumerate(subsets):
            prod = 1.0
            for idx in comb:
                prod *= spins[idx]
            mat[m, k] = prod
    return mat


def compute_exact_wht_4(
    item_matrix_4xn: np.ndarray,
    seed: int = 202607,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Compute exact 4-term Walsh-Hadamard decomposition with Parseval conservation and bootstrap CIs."""
    arr = np.asarray(item_matrix_4xn, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] != 4:
        raise ValueError(f"Expected shape (4, N), got {arr.shape}")
    n_items = arr.shape[1]

    h_mat = build_walsh_hadamard_4()
    w_items = (h_mat.T @ arr) / 4.0
    w_mean = np.mean(w_items, axis=1)
    recon = h_mat @ w_mean
    mean_obs = np.mean(arr, axis=1)
    max_recon_err = float(np.max(np.abs(recon - mean_obs)))

    nonconst_sq = np.square(w_mean[1:])
    total_nonconst_var = float(np.sum(nonconst_sq))
    pop_var = float(np.var(mean_obs, ddof=0))
    parseval_err = abs(total_nonconst_var - pop_var)

    subsets = wht_subsets_4()
    raw_p = [exact_or_mc_signflip_p(w_items[k], seed=seed + k) for k in range(1, 4)]
    holm_p = holm_adjust(raw_p)

    terms: dict[str, dict[str, Any]] = {}
    order_var_sum = {1: 0.0, 2: 0.0}
    for k, (_comb, lbl, order) in enumerate(subsets):
        est, ci_lo, ci_hi = bootstrap_mean_ci(w_items[k], n_boot=n_boot, seed=seed + 100 + k)
        coalition_scale = float(2**order) if order >= 1 else 1.0
        var_share = (
            float(w_mean[k] ** 2) / total_nonconst_var
            if (order >= 1 and total_nonconst_var > 1e-15)
            else 0.0
        )
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
            entry["p_raw"] = round(float(raw_p[k - 1]), 6)
            entry["p_holm"] = round(float(holm_p[k - 1]), 6)
        terms[lbl] = entry

    order_shares = {
        f"order_{ord_k}": round(
            (order_var_sum[ord_k] / total_nonconst_var) if total_nonconst_var > 1e-15 else 0.0,
            6,
        )
        for ord_k in (1, 2)
    }

    total_gain = arr[3] - arr[1]  # intact (11) minus prelude-only (01)
    tg_est, tg_lo, tg_hi = bootstrap_mean_ci(total_gain, n_boot=n_boot, seed=seed + 501)
    state_minus_recall = arr[2] - arr[1]  # state-only (10) minus recall-only (01)
    sr_est, sr_lo, sr_hi = bootstrap_mean_ci(state_minus_recall, n_boot=n_boot, seed=seed + 502)

    by_mask: dict[str, dict[str, Any]] = {}
    for m, _bits, code, h_lbl in enumerate_4_masks():
        m_est, m_lo, m_hi = bootstrap_mean_ci(arr[m], n_boot=min(2000, n_boot), seed=seed + 700 + m)
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
        "recurrent_gain_11_minus_01": {
            "mean": round(tg_est, 6),
            "ci95": [round(tg_lo, 6), round(tg_hi, 6)],
            "p_signflip": round(exact_or_mc_signflip_p(total_gain, seed=seed + 601), 6),
        },
        "state_only_minus_recall_only": {
            "mean": round(sr_est, 6),
            "ci95": [round(sr_lo, 6), round(sr_hi, 6)],
            "p_signflip": round(exact_or_mc_signflip_p(state_minus_recall, seed=seed + 602), 6),
        },
        "order_variance_shares": order_shares,
        "terms": terms,
        "masks": by_mask,
    }


# ---------------------------------------------------------------------------
# BiophysicalHuginnController for RavenForCausalLM
# ---------------------------------------------------------------------------

def _select_balanced_subset(suite: dict[str, Any], per_regime: int = 20) -> list[BiophysicalItem]:
    out: list[BiophysicalItem] = []
    for r in ("factual_2hop", "arithmetic_2step", "calendar_2step"):
        out.extend(suite[r][:per_regime])
    return out


class BiophysicalHuginnController:
    """Explicit unrolled mechanistic controller for `RavenForCausalLM` (`tomg-group-umd/huginn-0125`)."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        model_label: str = "tomg-group-umd/huginn-0125",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.model_label = model_label
        self.config = model.config
        self.transformer = model.transformer
        self.lm_head = model.lm_head

        self.d_model = int(getattr(self.config, "n_embd", self.lm_head.weight.shape[1]))
        self.n_prelude = len(self.transformer.prelude)
        self.n_core = len(self.transformer.core_block)
        self.n_coda = len(self.transformer.coda)
        self.pad_token_id = int(
            getattr(tokenizer, "pad_token_id", None)
            if getattr(tokenizer, "pad_token_id", None) is not None
            else getattr(self.config, "pad_token_id", 0)
        )
        self.device = next(model.parameters()).device

    def format_prompt(self, item: BiophysicalItem, prompt_mode: str = "zero_shot", use_foil: bool = False) -> str:
        """Render item prompt in zero-shot or 2-shot completion format."""
        raw = item.foil_raw_prompt if use_foil else item.raw_prompt
        if prompt_mode == "two_shot":
            prefix = _TWO_SHOT_PREFIXES.get(item.regime, "")
            return prefix + raw
        return raw

    def _encode_right_padded(self, prompts: Sequence[str]) -> tuple[torch.Tensor, list[int]]:
        enc_list: list[list[int]] = []
        for p in prompts:
            ids = self.tokenizer.encode(p, add_special_tokens=False)
            if not ids:
                ids = [self.pad_token_id]
            enc_list.append([int(x) for x in ids])
        max_len = max(len(x) for x in enc_list)
        bsz = len(enc_list)
        inp = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long, device=self.device)
        lens: list[int] = []
        for i, ids in enumerate(enc_list):
            inp[i, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=self.device)
            lens.append(len(ids))
        return inp, lens

    def make_initial_state(
        self,
        input_embeds: torch.Tensor,
        init_scale: float = 1.0,
        seed: int = 20260701,
    ) -> torch.Tensor:
        """Deterministic fixed-seed `'like-init'` or zero-state (`init_scale = 0.0`) initial latent `x_0`."""
        if init_scale <= 0.0:
            return torch.zeros_like(input_embeds)
        init_vals = getattr(self.config, "init_values", None) or {}
        base_std = float(init_vals.get("std", 0.008703882797784892))
        std = base_std * float(init_scale)
        if std <= 0.0:
            return torch.zeros_like(input_embeds)

        gen = torch.Generator(device="cpu").manual_seed(int(seed))
        x_cpu = torch.empty(input_embeds.shape, dtype=torch.float32, device="cpu")
        torch.nn.init.trunc_normal_(x_cpu, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std, generator=gen)
        emb_scale = float(getattr(self.model, "emb_scale", 1.0))
        if emb_scale != 1.0:
            x_cpu = x_cpu * emb_scale
        return x_cpu.to(device=input_embeds.device, dtype=input_embeds.dtype)

    def _get_freqs_cis(self, seq_len: int) -> torch.Tensor:
        freqs = self.model.freqs_cis[:, :seq_len]
        return freqs.to(self.device)

    def _run_adapter_and_core_step(
        self,
        x: torch.Tensor,
        input_embeds: torch.Tensor,
        freqs_cis: torch.Tensor,
        block_idx: torch.Tensor,
        current_step: int,
        channel_bits: tuple[int, int] = (1, 1),
        gamma: float = 1.0,
        record_visits_prefix: str | None = None,
        visits_out: list[tuple[str, torch.Tensor]] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b_idx = block_idx.detach().clone()
        if hasattr(self.model, "_maybe_inject_noise"):
            x = self.model._maybe_inject_noise(x, current_step)

        b_state, b_recall = int(channel_bits[0]), int(channel_bits[1])
        x_eff = x if (b_state == 1 and abs(gamma - 1.0) <= 1e-9) else (x * (float(b_state) * float(gamma)))
        e_eff = input_embeds.to(x.device)
        if b_recall == 0:
            e_eff = torch.zeros_like(e_eff)

        x = self.transformer.adapter(torch.cat([x_eff, e_eff], dim=-1))
        for c_i, block in enumerate(self.transformer.core_block):
            b_idx = b_idx + 1
            x = block(x, freqs_cis, b_idx, None, None)
            if record_visits_prefix is not None and visits_out is not None:
                visits_out.append((f"{record_visits_prefix}_C{c_i}", x))
        return x, b_idx

    def run_coda_from_latent(
        self,
        latent_state: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor]]]:
        """Run `ln_f -> coda blocks -> ln_f` starting from any recurrent latent state."""
        x = self.transformer.ln_f(latent_state)
        b_idx = torch.tensor(0, device=torch.device("cpu"), dtype=torch.long)
        trace: list[tuple[int, torch.Tensor]] = []
        for c_i, block in enumerate(self.transformer.coda):
            b_idx = b_idx - 1
            x = block(x, freqs_cis, b_idx, None, None)
            trace.append((c_i, x))
        x = self.transformer.ln_f(x)
        return x, trace

    def readout_logits_from_hidden(
        self,
        hidden_states: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        apply_ln_f: bool = True,
    ) -> torch.Tensor:
        """Apply optional `ln_f` + `lm_head` at selected `(row_idx, col_idx)` token positions."""
        picked = hidden_states[row_idx, col_idx]
        if apply_ln_f:
            picked = self.transformer.ln_f(picked)
        return self.lm_head(picked.to(self.lm_head.weight.dtype)).float()

    def unrolled_huginn_forward(
        self,
        input_ids: torch.Tensor,
        n_steps: int = 8,
        input_states: torch.Tensor | None = None,
        init_scale: float = 1.0,
        seed: int = 20260701,
        channel_bits: tuple[int, int] = (1, 1),
        channel_bits_by_step: Sequence[tuple[int, int]] | None = None,
        gamma: float = 1.0,
        gammas_by_step: Sequence[float] | None = None,
        apply_from_step: int = 2,
        record_all_visits: bool = False,
    ) -> dict[str, Any]:
        """Exact step-by-step unrolled execution of `RavenForCausalLM` with per-step adapter hooks."""
        seq_len = input_ids.shape[1]
        freqs_cis = self._get_freqs_cis(seq_len)

        input_embeds = self.transformer.wte(input_ids)
        emb_scale = float(getattr(self.model, "emb_scale", 1.0))
        if emb_scale != 1.0:
            input_embeds = input_embeds * emb_scale

        visits: list[tuple[str, torch.Tensor]] = []
        block_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)

        # 1. Prelude blocks
        for p_i, block in enumerate(self.transformer.prelude):
            block_idx = block_idx + 1
            input_embeds = block(input_embeds, freqs_cis, block_idx, None, None)
            if record_all_visits:
                visits.append((f"prelude_L{p_i}", input_embeds))

        # 2. Initial recurrent state
        if input_states is not None:
            x0 = input_states.clone()
        else:
            x0 = self.make_initial_state(input_embeds, init_scale=init_scale, seed=seed)
        x = x0

        # 3. Recurrent core steps (1..n_steps)
        step_exits: list[torch.Tensor] = []
        step_block_indices: list[torch.Tensor] = []
        for step_idx in range(n_steps):
            step_num = step_idx + 1
            if step_num >= apply_from_step:
                if channel_bits_by_step is not None and len(channel_bits_by_step) > 0:
                    if len(channel_bits_by_step) == n_steps:
                        cb = channel_bits_by_step[step_idx]
                    else:
                        cb = channel_bits_by_step[min(step_num - apply_from_step, len(channel_bits_by_step) - 1)]
                    cur_bits = (int(cb[0]), int(cb[1]))
                else:
                    cur_bits = (int(channel_bits[0]), int(channel_bits[1]))

                if gammas_by_step is not None and len(gammas_by_step) > 0:
                    if len(gammas_by_step) == n_steps:
                        cur_gamma = float(gammas_by_step[step_idx])
                    else:
                        cur_gamma = float(gammas_by_step[min(step_num - apply_from_step, len(gammas_by_step) - 1)])
                else:
                    cur_gamma = float(gamma)
            else:
                cur_bits = (1, 1)
                cur_gamma = 1.0

            x, block_idx = self._run_adapter_and_core_step(
                x=x,
                input_embeds=input_embeds,
                freqs_cis=freqs_cis,
                block_idx=block_idx,
                current_step=step_idx,
                channel_bits=cur_bits,
                gamma=cur_gamma,
                record_visits_prefix=f"p{step_num}" if record_all_visits else None,
                visits_out=visits if record_all_visits else None,
            )
            step_exits.append(x)
            step_block_indices.append(block_idx.detach().clone())

        # 4. Post-recurrence ln_f + Coda + final ln_f
        x_fin, coda_trace = self.run_coda_from_latent(x, freqs_cis)
        if record_all_visits:
            for c_i, c_ten in coda_trace:
                visits.append((f"coda_L{c_i}", c_ten))

        return {
            "last_hidden_state": x_fin,
            "prelude_state": input_embeds,
            "initial_state": x0,
            "step_exits": step_exits,
            "step_block_indices": step_block_indices,
            "coda_states": coda_trace,
            "visits": visits,
            "freqs_cis": freqs_cis,
        }


# ---------------------------------------------------------------------------
# Preflight Equivalence, Zero-State Sensitivity, and Baseline Calibration
# ---------------------------------------------------------------------------

@_no_grad()
def run_huginn_preflight_equivalence(
    ctrl: BiophysicalHuginnController,
    depths: Sequence[int] = (1, 4, 8),
    seed: int = 20260701,
) -> dict[str, Any]:
    """Verify `unrolled_huginn_forward(n_steps=K)` matches `model.forward(num_steps=K)` with matched `x_0`."""
    sample_prompts = [
        "Expression: (3 * 2) + 1 =",
        "Fact: The national capital city of the country where the Taj Mahal is located is",
    ]
    inp, lens = ctrl._encode_right_padded(sample_prompts)
    row_idx = torch.arange(len(lens), device=ctrl.device)
    col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

    # Extract prelude shape for matched x_0
    unr_probe = ctrl.unrolled_huginn_forward(inp, n_steps=1, init_scale=1.0, seed=seed)
    x0_matched = unr_probe["initial_state"]

    diffs: dict[str, float] = {}
    for k in sorted({int(d) for d in depths if int(d) >= 1}):
        nat_out = ctrl.model(
            input_ids=inp,
            input_states=x0_matched.clone(),
            num_steps=torch.tensor(k, dtype=torch.long),
            use_cache=False,
        ).logits[row_idx, col_idx].float()

        unr_out = ctrl.unrolled_huginn_forward(
            inp,
            n_steps=k,
            input_states=x0_matched.clone(),
        )
        unr_logits = ctrl.lm_head(unr_out["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
        diff_k = float((nat_out - unr_logits).abs().max().item())
        diffs[f"K={k}"] = round(diff_k, 8)
        if diff_k > 1e-3:
            raise RuntimeError(f"Huginn preflight equivalence failed at K={k}: max_abs_diff={diff_k:.3e}")

    return {
        "valid": True,
        "depths_checked": sorted({int(d) for d in depths if int(d) >= 1}),
        "max_abs_logit_diff_by_depth": diffs,
    }


@_no_grad()
def run_huginn_zero_state_sensitivity(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    n_items: int = 20,
    n_steps: int = 8,
    batch_size: int = 10,
    prompt_mode: str = "zero_shot",
    seed: int = 20260701,
) -> dict[str, Any]:
    """Compare fixed-seed `'like-init'` (`init_scale=1.0`) vs zero-state (`init_scale=0.0`) at `K=8`."""
    per_reg = max(1, math.ceil(n_items / 3))
    items = _select_balanced_subset(suite, per_regime=per_reg)[:n_items]
    n_eff = len(items)

    like_acc = 0.0
    zero_acc = 0.0
    like_prob = 0.0
    zero_prob = 0.0
    kl_vals: list[float] = []
    max_logit_diffs: list[float] = []

    for s in range(0, n_eff, batch_size):
        e = min(n_eff, s + batch_size)
        sub = items[s:e]
        prompts = [ctrl.format_prompt(it, prompt_mode=prompt_mode) for it in sub]
        inp, lens = ctrl._encode_right_padded(prompts)
        row_idx = torch.arange(len(sub), device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
        tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]

        u_like = ctrl.unrolled_huginn_forward(inp, n_steps=n_steps, init_scale=1.0, seed=seed)
        u_zero = ctrl.unrolled_huginn_forward(inp, n_steps=n_steps, init_scale=0.0, seed=seed)

        lg_like = ctrl.lm_head(u_like["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
        lg_zero = ctrl.lm_head(u_zero["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()

        lp_like = F.log_softmax(lg_like, dim=-1)
        lp_zero = F.log_softmax(lg_zero, dim=-1)
        pr_like = torch.exp(lp_like)
        pr_zero = torch.exp(lp_zero)

        kl_batch = (pr_like * (lp_like - lp_zero)).sum(dim=-1).clamp(min=0.0)
        kl_vals.extend(kl_batch.cpu().tolist())
        max_logit_diffs.extend((lg_like - lg_zero).abs().max(dim=-1).values.cpu().tolist())

        top_like = lg_like.argmax(dim=-1).tolist()
        top_zero = lg_zero.argmax(dim=-1).tolist()
        for i in range(len(sub)):
            t_ids = tgt_sets[i]
            if top_like[i] in t_ids:
                like_acc += 1.0
            if top_zero[i] in t_ids:
                zero_acc += 1.0
            like_prob += float(pr_like[i, t_ids].sum().item())
            zero_prob += float(pr_zero[i, t_ids].sum().item())

    return {
        "n_items": n_eff,
        "n_steps": n_steps,
        "prompt_mode": prompt_mode,
        "like_init_top1_acc": round(like_acc / max(1, n_eff), 4),
        "zero_state_top1_acc": round(zero_acc / max(1, n_eff), 4),
        "like_init_target_prob": round(like_prob / max(1, n_eff), 5),
        "zero_state_target_prob": round(zero_prob / max(1, n_eff), 5),
        "mean_kl_like_vs_zero_nats": round(float(np.mean(kl_vals)) if kl_vals else 0.0, 5),
        "mean_max_abs_logit_diff": round(float(np.mean(max_logit_diffs)) if max_logit_diffs else 0.0, 5),
    }


@_no_grad()
def run_huginn_baseline_calibration(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    n_steps: int = 8,
    batch_size: int = 16,
    threshold: float = 0.30,
    seed: int = 20260701,
) -> dict[str, Any]:
    """Check zero-shot K=8 accuracy on `factual_2hop + arithmetic_2step`; add 2-shot prefix if `< 30%`."""
    cal_items = [*suite["factual_2hop"], *suite["arithmetic_2step"]]
    n_cal = len(cal_items)

    def _eval_mode(mode: str) -> tuple[float, float]:
        hits = 0.0
        prob_sum = 0.0
        for s in range(0, n_cal, batch_size):
            e = min(n_cal, s + batch_size)
            sub = cal_items[s:e]
            prompts = [ctrl.format_prompt(it, prompt_mode=mode) for it in sub]
            inp, lens = ctrl._encode_right_padded(prompts)
            row_idx = torch.arange(len(sub), device=ctrl.device)
            col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]

            unr = ctrl.unrolled_huginn_forward(inp, n_steps=n_steps, init_scale=1.0, seed=seed)
            lg = ctrl.lm_head(unr["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
            pr = F.softmax(lg, dim=-1)
            top = lg.argmax(dim=-1).tolist()
            for i in range(len(sub)):
                t_ids = tgt_sets[i]
                if top[i] in t_ids:
                    hits += 1.0
                prob_sum += float(pr[i, t_ids].sum().item())
        return hits / max(1, n_cal), prob_sum / max(1, n_cal)

    zs_acc, zs_prob = _eval_mode("zero_shot")
    selected_mode = "zero_shot"
    ts_acc: float | None = None
    ts_prob: float | None = None
    if zs_acc < threshold:
        ts_acc, ts_prob = _eval_mode("two_shot")
        selected_mode = "two_shot"

    return {
        "n_calibration_items": n_cal,
        "n_steps": n_steps,
        "threshold": threshold,
        "zero_shot_acc": round(zs_acc, 4),
        "zero_shot_target_prob": round(zs_prob, 5),
        "two_shot_acc": round(ts_acc, 4) if ts_acc is not None else None,
        "two_shot_target_prob": round(ts_prob, 5) if ts_prob is not None else None,
        "selected_prompt_mode": selected_mode,
    }


# ---------------------------------------------------------------------------
# Exp H1: 2-Channel (b_state, b_recall) 4-Mask Adapter WHT at K=4, 8
# ---------------------------------------------------------------------------

@_no_grad()
def _score_huginn_items(
    ctrl: BiophysicalHuginnController,
    items: Sequence[BiophysicalItem],
    n_steps: int,
    channel_bits: tuple[int, int] = (1, 1),
    gamma: float = 1.0,
    batch_size: int = 16,
    prompt_mode: str = "zero_shot",
    seed: int = 20260701,
) -> dict[str, np.ndarray]:
    n = len(items)
    cont_lp = np.zeros(n, dtype=np.float64)
    top1_hit = np.zeros(n, dtype=np.float64)
    target_prob = np.zeros(n, dtype=np.float64)
    hop1_prob = np.zeros(n, dtype=np.float64)
    target_vs_foil_lo = np.zeros(n, dtype=np.float64)

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sub = items[s:e]
        p_ids_list: list[list[int]] = []
        f_ids_list: list[list[int]] = []
        tgt_sets: list[list[int]] = []
        hop1_sets: list[list[int]] = []
        foil_sets: list[list[int]] = []

        for it in sub:
            pref = ctrl.format_prompt(it, prompt_mode=prompt_mode)
            p_ids = ctrl.tokenizer.encode(pref, add_special_tokens=False)
            cont_text = " " + it.target_answer.strip()
            f_ids = ctrl.tokenizer.encode(pref + cont_text, add_special_tokens=False)
            if len(f_ids) <= len(p_ids):
                f_ids = p_ids + ctrl.tokenizer.encode(cont_text, add_special_tokens=False)
            p_ids_list.append(p_ids)
            f_ids_list.append(f_ids)
            tgt_sets.append(first_token_variant_ids(ctrl.tokenizer, it.target_answer))
            hop1_sets.append(first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge))
            foil_sets.append(first_token_variant_ids(ctrl.tokenizer, it.foil_answer))

        max_l = max(len(x) for x in f_ids_list)
        bsz = len(sub)
        inp = torch.full((bsz, max_l), ctrl.pad_token_id, dtype=torch.long, device=ctrl.device)
        for i, f_ids in enumerate(f_ids_list):
            inp[i, : len(f_ids)] = torch.tensor(f_ids, dtype=torch.long, device=ctrl.device)

        unr = ctrl.unrolled_huginn_forward(
            inp,
            n_steps=n_steps,
            init_scale=1.0,
            seed=seed,
            channel_bits=channel_bits,
            gamma=gamma,
            apply_from_step=2,
        )
        h_fin = unr["last_hidden_state"]

        for i, (p_ids, f_ids) in enumerate(zip(p_ids_list, f_ids_list)):
            q_pos = len(p_ids) - 1
            span_end = len(f_ids) - 1
            h_slice = h_fin[i, q_pos:span_end].to(ctrl.lm_head.weight.dtype)
            logits_slice = ctrl.lm_head(h_slice).float()
            lp_slice = F.log_softmax(logits_slice, dim=-1)

            lp_sum = 0.0
            for step_k, tok_id in enumerate(f_ids[len(p_ids) :]):
                lp_sum += float(lp_slice[step_k, tok_id].item())
            cont_lp[s + i] = lp_sum

            probs0 = torch.exp(lp_slice[0])
            pred0 = int(logits_slice[0].argmax().item())
            t_ids = tgt_sets[i]
            h1_ids = hop1_sets[i]
            fl_ids = foil_sets[i]

            top1_hit[s + i] = 1.0 if pred0 in t_ids else 0.0
            p_t = float(probs0[t_ids].sum().item()) if t_ids else 0.0
            p_h1 = float(probs0[h1_ids].sum().item()) if h1_ids else 0.0
            p_fl = float(probs0[fl_ids].sum().item()) if fl_ids else 0.0
            target_prob[s + i] = p_t
            hop1_prob[s + i] = p_h1
            target_vs_foil_lo[s + i] = math.log(max(p_t, 1e-30)) - math.log(max(p_fl, 1e-30))

    return {
        "continuation_logprob": cont_lp,
        "top1_acc": top1_hit,
        "target_prob": target_prob,
        "hop1_prob": hop1_prob,
        "target_vs_foil_lo": target_vs_foil_lo,
    }


@_no_grad()
def run_exp_h1_adapter_wht(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    depths: Sequence[int] = (4, 8),
    batch_size: int = 16,
    prompt_mode: str = "zero_shot",
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Exp H1: Exact 2-channel `(b_state, b_recall)` 4-mask adapter WHT at `K = 4, 8`."""
    items_120: list[BiophysicalItem] = suite["reasoning_120"]
    regimes = ("factual_2hop", "arithmetic_2step", "calendar_2step")
    reg_indices = {
        r: np.asarray([idx for idx, it in enumerate(items_120) if it.regime == r], dtype=np.int64)
        for r in regimes
    }
    masks_4 = enumerate_4_masks()
    by_depth: dict[str, Any] = {}

    for d_i, k in enumerate(sorted({int(x) for x in depths if int(x) >= 2})):
        lp_cube = np.zeros((4, len(items_120)), dtype=np.float64)
        acc_cube = np.zeros((4, len(items_120)), dtype=np.float64)
        lo_cube = np.zeros((4, len(items_120)), dtype=np.float64)

        for m_int, bits, _code, _lbl in masks_4:
            sc = _score_huginn_items(
                ctrl,
                items_120,
                n_steps=k,
                channel_bits=bits,
                gamma=1.0,
                batch_size=batch_size,
                prompt_mode=prompt_mode,
            )
            lp_cube[m_int] = sc["continuation_logprob"]
            acc_cube[m_int] = sc["top1_acc"]
            lo_cube[m_int] = sc["target_vs_foil_lo"]

        lp_wht = compute_exact_wht_4(lp_cube, seed=9100 + d_i * 100, n_boot=n_boot)
        acc_wht = compute_exact_wht_4(acc_cube, seed=9200 + d_i * 100, n_boot=n_boot)
        lo_wht = compute_exact_wht_4(lo_cube, seed=9300 + d_i * 100, n_boot=n_boot)
        if lp_wht["parseval_variance_error"] >= 1e-12 or acc_wht["parseval_variance_error"] >= 1e-12:
            raise RuntimeError(f"Huginn WHT-4 Parseval error exceeded 1e-12 at K={k}")

        reg_out: dict[str, Any] = {}
        for r_i, r_name in enumerate(regimes):
            idxs = reg_indices[r_name]
            if idxs.size == 0:
                continue
            reg_out[r_name] = {
                "continuation_logprob_wht": compute_exact_wht_4(
                    lp_cube[:, idxs], seed=9400 + d_i * 100 + r_i * 20, n_boot=n_boot
                ),
                "top1_accuracy_wht": compute_exact_wht_4(
                    acc_cube[:, idxs], seed=9500 + d_i * 100 + r_i * 20, n_boot=n_boot
                ),
                "target_vs_foil_lo_wht": compute_exact_wht_4(
                    lo_cube[:, idxs], seed=9600 + d_i * 100 + r_i * 20, n_boot=n_boot
                ),
            }

        by_depth[f"K={k}"] = {
            "n_steps": k,
            "n_items": len(items_120),
            "all_120": {
                "continuation_logprob_wht": lp_wht,
                "top1_accuracy_wht": acc_wht,
                "target_vs_foil_lo_wht": lo_wht,
            },
            "by_regime": reg_out,
        }

    return {
        "n_total_items": len(items_120),
        "channel_order": list(HUGINN_CHANNEL_NAMES),
        "prompt_mode": prompt_mode,
        "by_depth": by_depth,
    }


# ---------------------------------------------------------------------------
# Exp H2: Unrolled Visit Lens, Coda Conductance (p=1,2,4,8), & Non-Normal Gain
# ---------------------------------------------------------------------------

def run_exp_h2_gain_vs_amplitude(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    n_steps: int = 8,
    lens_per_regime: int = 20,
    jacobian_per_regime: int = 12,
    conductance_steps: Sequence[int] = (1, 2, 4, 8),
    power_steps: Sequence[int] = (1, 2, 3, 4, 6, 8),
    batch_size: int = 12,
    prompt_mode: str = "zero_shot",
    n_boot: int = 4000,
    seed: int = 20260701,
) -> dict[str, Any]:
    """Exp H2: Unrolled visit lens, Coda conductance at p=1,2,4,8, and 6-step VJP/JVP sigma_max."""
    items_60 = _select_balanced_subset(suite, per_regime=lens_per_regime)
    n60 = len(items_60)

    with torch.no_grad():
        w_head_sub = ctrl.lm_head.weight[: min(32768, ctrl.lm_head.weight.shape[0])].float()
        _u_h, _s_h, vh_head = torch.linalg.svd(w_head_sub, full_matrices=False)
        u_unembed = vh_head[: min(64, vh_head.shape[0])].t().contiguous()

    visit_names: list[str] = []
    raw_acc_sum: np.ndarray | None = None
    raw_prob_sum: np.ndarray | None = None
    unembed_frac_sum: np.ndarray | None = None
    state_norm_sum: np.ndarray | None = None

    eff_cond_steps = sorted({int(p) for p in conductance_steps if 1 <= int(p) <= n_steps})
    trunc_acc = {p: 0.0 for p in eff_cond_steps}
    trunc_prob = {p: 0.0 for p in eff_cond_steps}
    trunc_hop1 = {p: 0.0 for p in eff_cond_steps}

    with torch.no_grad():
        for s in range(0, n60, batch_size):
            e = min(n60, s + batch_size)
            sub = items_60[s:e]
            prompts = [ctrl.format_prompt(it, prompt_mode=prompt_mode) for it in sub]
            inp, lens = ctrl._encode_right_padded(prompts)
            row_idx = torch.arange(len(sub), device=ctrl.device)
            col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
            h1_sets = [first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge) for it in sub]

            unr = ctrl.unrolled_huginn_forward(inp, n_steps=n_steps, init_scale=1.0, seed=seed, record_all_visits=True)
            v_list = unr["visits"]
            if not visit_names:
                visit_names = [v[0] for v in v_list]
                nv = len(visit_names)
                raw_acc_sum = np.zeros(nv, dtype=np.float64)
                raw_prob_sum = np.zeros(nv, dtype=np.float64)
                unembed_frac_sum = np.zeros(nv, dtype=np.float64)
                state_norm_sum = np.zeros(nv, dtype=np.float64)

            assert raw_acc_sum is not None and raw_prob_sum is not None
            assert unembed_frac_sum is not None and state_norm_sum is not None

            for v_i, (_vname, h_st) in enumerate(v_list):
                h_tok = h_st[row_idx, col_idx].float()
                h_nrm = torch.linalg.vector_norm(h_tok, dim=-1).clamp(min=1e-8)
                proj_nrm = torch.linalg.vector_norm(h_tok @ u_unembed, dim=-1)
                state_norm_sum[v_i] += float(h_nrm.sum().item())
                unembed_frac_sum[v_i] += float((proj_nrm / h_nrm).sum().item())

                lg_raw = ctrl.readout_logits_from_hidden(h_st, row_idx, col_idx, apply_ln_f=True)
                pr_raw = F.softmax(lg_raw, dim=-1)
                top_raw = lg_raw.argmax(dim=-1).tolist()
                for i in range(len(sub)):
                    t_ids = tgt_sets[i]
                    if top_raw[i] in t_ids:
                        raw_acc_sum[v_i] += 1.0
                    raw_prob_sum[v_i] += float(pr_raw[i, t_ids].sum().item())

            for p_step in eff_cond_steps:
                lat_p = unr["step_exits"][p_step - 1]
                fin_p, _ = ctrl.run_coda_from_latent(lat_p, unr["freqs_cis"])
                lg_c = ctrl.lm_head(fin_p[row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
                pr_c = F.softmax(lg_c, dim=-1)
                top_c = lg_c.argmax(dim=-1).tolist()
                for i in range(len(sub)):
                    t_ids = tgt_sets[i]
                    h1_ids = h1_sets[i]
                    if top_c[i] in t_ids:
                        trunc_acc[p_step] += 1.0
                    trunc_prob[p_step] += float(pr_c[i, t_ids].sum().item())
                    trunc_hop1[p_step] += float(pr_c[i, h1_ids].sum().item())

    assert raw_acc_sum is not None and raw_prob_sum is not None
    assert unembed_frac_sum is not None and state_norm_sum is not None

    visits_summary = [
        {
            "visit_idx": v_i,
            "visit_label": v_lbl,
            "raw_logit_lens_acc": round(float(raw_acc_sum[v_i] / max(1, n60)), 4),
            "raw_logit_lens_prob": round(float(raw_prob_sum[v_i] / max(1, n60)), 4),
            "unembed_svd64_amp_ratio": round(float(unembed_frac_sum[v_i] / max(1, n60)), 4),
            "mean_hidden_norm": round(float(state_norm_sum[v_i] / max(1, n60)), 3),
        }
        for v_i, v_lbl in enumerate(visit_names)
    ]

    truncated_coda_by_step = {
        f"step_{p_step}": {
            "top1_acc": round(float(trunc_acc[p_step] / max(1, n60)), 4),
            "target_prob": round(float(trunc_prob[p_step] / max(1, n60)), 4),
            "hop1_prob": round(float(trunc_hop1[p_step] / max(1, n60)), 4),
        }
        for p_step in eff_cond_steps
    }

    # Part H2B: Coda Conductance + 6-Step VJP/JVP Non-Normal Pass Gain
    items_jac = _select_balanced_subset(suite, per_regime=jacobian_per_regime)
    n_jac = len(items_jac)
    jac_bsz = min(8, max(1, n_jac))
    rng_g = torch.Generator(device="cpu").manual_seed(20260702)

    eff_pow_steps = sorted({int(p) for p in power_steps if 1 <= int(p) <= n_steps})

    chi_iso_vjp: dict[str, list[float]] = {f"step_{p}": [] for p in eff_cond_steps}
    chi_rand_fd: dict[str, list[float]] = {f"step_{p}": [] for p in eff_cond_steps}
    chi_task_fd: dict[str, list[float]] = {f"step_{p}": [] for p in eff_cond_steps}
    task_grad_norm: dict[str, list[float]] = {f"step_{p}": [] for p in eff_cond_steps}

    sigma_vjp_by_step: dict[str, list[float]] = {f"step_{p}": [] for p in eff_pow_steps}
    sigma_fd_by_step: dict[str, list[float]] = {f"step_{p}": [] for p in eff_pow_steps}
    iso_gain_by_step: dict[str, list[float]] = {f"step_{p}": [] for p in eff_pow_steps}
    io_cos_by_step: dict[str, list[float]] = {f"step_{p}": [] for p in eff_pow_steps}
    rayleigh_by_step: dict[str, list[float]] = {f"step_{p}": [] for p in eff_pow_steps}

    for s in range(0, n_jac, jac_bsz):
        e = min(n_jac, s + jac_bsz)
        sub = items_jac[s:e]
        bsz = len(sub)
        prompts = [ctrl.format_prompt(it, prompt_mode=prompt_mode) for it in sub]
        inp, lens = ctrl._encode_right_padded(prompts)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
        tgt_first = torch.tensor(
            [first_token_variant_ids(ctrl.tokenizer, it.target_answer)[0] for it in sub],
            dtype=torch.long,
            device=ctrl.device,
        )

        with torch.no_grad():
            base = ctrl.unrolled_huginn_forward(inp, n_steps=n_steps, init_scale=1.0, seed=seed)
            prelude_clean = base["prelude_state"].detach()
            x0_clean = base["initial_state"].detach()
            exits_clean = [x.detach() for x in base["step_exits"]]
            b_idx_clean = base["step_block_indices"]
            freqs_cis = base["freqs_cis"]

        # 1) Coda conductance at p in eff_cond_steps
        for p_step in eff_cond_steps:
            st_key = f"step_{p_step}"
            h_ex = exits_clean[p_step - 1]
            h_leaf = h_ex.clone().requires_grad_(True)
            with torch.enable_grad():
                fin_n, _ = ctrl.run_coda_from_latent(h_leaf, freqs_cis)
                z_last = fin_n[row_idx, col_idx].float()
                logits_last = ctrl.lm_head(z_last.to(ctrl.lm_head.weight.dtype)).float()
                lp_last = F.log_softmax(logits_last, dim=-1)
                target_lp = lp_last[row_idx, tgt_first].sum()

                g_task_full = torch.autograd.grad(target_lp, h_leaf, retain_graph=True)[0]
                g_task = g_task_full[row_idx, col_idx].float()
                g_nrm = torch.linalg.vector_norm(g_task, dim=-1, keepdim=True).clamp(min=1e-12)
                u_task = g_task / g_nrm

                iso_sq_acc = torch.zeros(bsz, dtype=torch.float32, device=ctrl.device)
                n_probes = 4
                for pr_i in range(n_probes):
                    w_cpu = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
                    w_unit = (w_cpu / torch.linalg.vector_norm(w_cpu, dim=-1, keepdim=True).clamp(min=1e-12)).to(
                        ctrl.device
                    )
                    g_w = torch.autograd.grad(
                        (z_last * w_unit).sum(),
                        h_leaf,
                        retain_graph=(pr_i < n_probes - 1),
                    )[0][row_idx, col_idx].float()
                    iso_sq_acc += g_w.square().sum(dim=-1)
                chi_iso_batch = torch.sqrt(iso_sq_acc / float(n_probes))

            with torch.no_grad():
                h_base_tok = h_ex[row_idx, col_idx].float()
                step_mag = (0.05 * torch.linalg.vector_norm(h_base_tok, dim=-1, keepdim=True)).clamp(min=0.05)
                u_r_cpu = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
                u_rand = (u_r_cpu / torch.linalg.vector_norm(u_r_cpu, dim=-1, keepdim=True).clamp(min=1e-12)).to(
                    ctrl.device
                )
                for u_vec, dest_list in ((u_rand, chi_rand_fd[st_key]), (u_task, chi_task_fd[st_key])):
                    h_plus = h_ex.clone()
                    h_minus = h_ex.clone()
                    h_plus[row_idx, col_idx] = (h_base_tok + step_mag * u_vec).to(h_ex.dtype)
                    h_minus[row_idx, col_idx] = (h_base_tok - step_mag * u_vec).to(h_ex.dtype)
                    z_p, _ = ctrl.run_coda_from_latent(h_plus, freqs_cis)
                    z_m, _ = ctrl.run_coda_from_latent(h_minus, freqs_cis)
                    dz = (z_p[row_idx, col_idx].float() - z_m[row_idx, col_idx].float()) / (2.0 * step_mag)
                    dest_list.extend(torch.linalg.vector_norm(dz, dim=-1).cpu().tolist())

            chi_iso_vjp[st_key].extend(chi_iso_batch.cpu().tolist())
            task_grad_norm[st_key].extend(g_nrm.squeeze(-1).cpu().tolist())

        # 2) 6-step VJP/JVP sigma_max across p in eff_pow_steps
        prelude_b_idx = torch.tensor(ctrl.n_prelude - 1, device=torch.device("cpu"), dtype=torch.long)
        for p_step in eff_pow_steps:
            st_key = f"step_{p_step}"
            in_ref = x0_clean if p_step == 1 else exits_clean[p_step - 2]
            in_b_idx = prelude_b_idx if p_step == 1 else b_idx_clean[p_step - 2]

            def _run_step_op(h_var: torch.Tensor) -> torch.Tensor:
                out_x, _ = ctrl._run_adapter_and_core_step(
                    x=h_var,
                    input_embeds=prelude_clean,
                    freqs_cis=freqs_cis,
                    block_idx=in_b_idx,
                    current_step=p_step - 1,
                    channel_bits=(1, 1),
                    gamma=1.0,
                )
                return out_x

            v_init = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
            v_vec = (v_init / torch.linalg.vector_norm(v_init, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)

            h_leaf_p = in_ref.clone().requires_grad_(True)
            with torch.enable_grad():
                out_p = _run_step_op(h_leaf_p)
                out_last = out_p[row_idx, col_idx].float()
                iso_p_sq = torch.zeros(bsz, dtype=torch.float32, device=ctrl.device)
                for _pr_i in range(4):
                    w_c = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
                    w_u = (w_c / torch.linalg.vector_norm(w_c, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)
                    gp = torch.autograd.grad((out_last * w_u).sum(), h_leaf_p, retain_graph=True)[0][
                        row_idx, col_idx
                    ].float()
                    iso_p_sq += gp.square().sum(dim=-1)
                iso_gain_p = torch.sqrt(iso_p_sq / 4.0)

            in_tok_base = in_ref[row_idx, col_idx].float()
            step_p = (0.05 * torch.linalg.vector_norm(in_tok_base, dim=-1, keepdim=True)).clamp(min=0.05)
            sig_fd = torch.ones(bsz, dtype=torch.float32, device=ctrl.device)

            for it_step in range(6):
                if it_step == 0:
                    with torch.enable_grad():
                        g_v = torch.autograd.grad((out_last * v_vec).sum(), h_leaf_p, retain_graph=False)[0][
                            row_idx, col_idx
                        ].float()
                else:
                    hl = in_ref.clone().requires_grad_(True)
                    with torch.enable_grad():
                        ol = _run_step_op(hl)[row_idx, col_idx].float()
                        g_v = torch.autograd.grad((ol * v_vec).sum(), hl, retain_graph=False)[0][
                            row_idx, col_idx
                        ].float()
                sig_vjp = torch.linalg.vector_norm(g_v, dim=-1).clamp(min=1e-12)
                u_vec = g_v / sig_vjp.unsqueeze(-1)

                with torch.no_grad():
                    hp = in_ref.clone()
                    hm = in_ref.clone()
                    hp[row_idx, col_idx] = (in_tok_base + step_p * u_vec).to(in_ref.dtype)
                    hm[row_idx, col_idx] = (in_tok_base - step_p * u_vec).to(in_ref.dtype)
                    yp = _run_step_op(hp)[row_idx, col_idx].float()
                    ym = _run_step_op(hm)[row_idx, col_idx].float()
                    ju = (yp - ym) / (2.0 * step_p)
                    sig_fd = torch.linalg.vector_norm(ju, dim=-1).clamp(min=1e-12)
                    v_vec = ju / sig_fd.unsqueeze(-1)

            hl_fin = in_ref.clone().requires_grad_(True)
            with torch.enable_grad():
                ol_fin = _run_step_op(hl_fin)[row_idx, col_idx].float()
                g_fin = torch.autograd.grad((ol_fin * v_vec).sum(), hl_fin, retain_graph=False)[0][
                    row_idx, col_idx
                ].float()
            sig_vjp_cert = torch.linalg.vector_norm(g_fin, dim=-1).clamp(min=1e-12)
            u_fin = g_fin / sig_vjp_cert.unsqueeze(-1)
            io_cos = (u_fin * v_vec).sum(dim=-1).abs()
            rayleigh_q = sig_vjp_cert * io_cos

            sigma_vjp_by_step[st_key].extend(sig_vjp_cert.cpu().tolist())
            sigma_fd_by_step[st_key].extend(sig_fd.cpu().tolist())
            iso_gain_by_step[st_key].extend(iso_gain_p.cpu().tolist())
            io_cos_by_step[st_key].extend(io_cos.cpu().tolist())
            rayleigh_by_step[st_key].extend(rayleigh_q.cpu().tolist())

    conductance_summary: dict[str, Any] = {}
    for idx_s, p_step in enumerate(eff_cond_steps):
        st_key = f"step_{p_step}"
        ci_vjp = bootstrap_mean_ci(chi_iso_vjp[st_key], n_boot=n_boot, seed=9700 + idx_s)
        ci_rfd = bootstrap_mean_ci(chi_rand_fd[st_key], n_boot=n_boot, seed=9720 + idx_s)
        ci_tfd = bootstrap_mean_ci(chi_task_fd[st_key], n_boot=n_boot, seed=9740 + idx_s)
        ci_tgn = bootstrap_mean_ci(task_grad_norm[st_key], n_boot=n_boot, seed=9760 + idx_s)
        task_over_rand = np.asarray(chi_task_fd[st_key]) / np.maximum(np.asarray(chi_rand_fd[st_key]), 1e-8)
        conductance_summary[st_key] = {
            "isotropic_conductance_vjp": {"mean": round(ci_vjp[0], 4), "ci95": [round(ci_vjp[1], 4), round(ci_vjp[2], 4)]},
            "random_tangent_conductance_fd": {"mean": round(ci_rfd[0], 4), "ci95": [round(ci_rfd[1], 4), round(ci_rfd[2], 4)]},
            "task_aligned_conductance_fd": {"mean": round(ci_tfd[0], 4), "ci95": [round(ci_tfd[1], 4), round(ci_tfd[2], 4)]},
            "task_over_random_conductance_ratio": round(float(np.mean(task_over_rand)), 4),
            "task_logprob_grad_norm": {"mean": round(ci_tgn[0], 4), "ci95": [round(ci_tgn[1], 4), round(ci_tgn[2], 4)]},
        }

    amplification_summary: dict[str, Any] = {}
    for idx_s, p_step in enumerate(eff_pow_steps):
        st_key = f"step_{p_step}"
        sv_ci = bootstrap_mean_ci(sigma_vjp_by_step[st_key], n_boot=n_boot, seed=9800 + idx_s)
        sf_ci = bootstrap_mean_ci(sigma_fd_by_step[st_key], n_boot=n_boot, seed=9820 + idx_s)
        ig_ci = bootstrap_mean_ci(iso_gain_by_step[st_key], n_boot=n_boot, seed=9840 + idx_s)
        cos_ci = bootstrap_mean_ci(io_cos_by_step[st_key], n_boot=n_boot, seed=9860 + idx_s)
        rq_ci = bootstrap_mean_ci(rayleigh_by_step[st_key], n_boot=n_boot, seed=9880 + idx_s)
        aniso = np.asarray(sigma_vjp_by_step[st_key]) / np.maximum(np.asarray(iso_gain_by_step[st_key]), 1e-8)
        non_norm_amp = np.asarray(sigma_vjp_by_step[st_key]) / np.maximum(np.asarray(rayleigh_by_step[st_key]), 1e-4)
        amplification_summary[st_key] = {
            "sigma_max_vjp": {"mean": round(sv_ci[0], 4), "ci95": [round(sv_ci[1], 4), round(sv_ci[2], 4)]},
            "sigma_max_fd": {"mean": round(sf_ci[0], 4), "ci95": [round(sf_ci[1], 4), round(sf_ci[2], 4)]},
            "isotropic_rms_gain": {"mean": round(ig_ci[0], 4), "ci95": [round(ig_ci[1], 4), round(ig_ci[2], 4)]},
            "anisotropy_ratio_sigma_over_iso": round(float(np.mean(aniso)), 4),
            "singular_io_cosine": {"mean": round(cos_ci[0], 4), "ci95": [round(cos_ci[1], 4), round(cos_ci[2], 4)]},
            "normal_rayleigh_quotient": {"mean": round(rq_ci[0], 4), "ci95": [round(rq_ci[1], 4), round(rq_ci[2], 4)]},
            "non_normal_transient_ratio": round(float(np.mean(non_norm_amp)), 4),
        }

    return {
        "n_steps": n_steps,
        "n_lens_items": n60,
        "n_jacobian_items": n_jac,
        "visits": visits_summary,
        "truncated_coda_by_step": truncated_coda_by_step,
        "coda_conductance": conductance_summary,
        "non_normal_step_amplification": amplification_summary,
    }


# ---------------------------------------------------------------------------
# Exp H3: Hutchinson VJP Fisher Pullback & Hop-1/Hop-2 Commitment at K=8
# ---------------------------------------------------------------------------

def run_exp_h3_fisher_and_commitment(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    n_steps: int = 8,
    fisher_per_regime: int = 20,
    n_rademacher: int = 6,
    fisher_bsz: int = 8,
    prompt_mode: str = "zero_shot",
    n_boot: int = 4000,
    seed: int = 20260701,
) -> dict[str, Any]:
    """Exp H3: Full-vocab Hutchinson VJP Fisher pullback and Hop-1/Hop-2 commitment horizon at `K=8`."""
    items_60 = _select_balanced_subset(suite, per_regime=fisher_per_regime)
    n60 = len(items_60)
    rng_f = torch.Generator(device="cpu").manual_seed(20260703)

    ckpt_keys = [
        "prelude_end",
        *[f"step_{p}" for p in range(1, n_steps + 1)],
        *[f"coda_L{c_i}" for c_i in range(ctrl.n_coda)],
    ]
    n_ckpts = len(ckpt_keys)
    fisher_trace_mat = np.zeros((n_ckpts, n60), dtype=np.float64)
    fisher_rel_mat = np.zeros((n_ckpts, n60), dtype=np.float64)

    traj_stages = ["prelude", *[f"step_{p}" for p in range(1, n_steps + 1)]]
    n_stages = len(traj_stages)
    hop1_prob_traj = {
        r: np.zeros(n_stages, dtype=np.float64)
        for r in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step")
    }
    hop2_prob_traj = {
        r: np.zeros(n_stages, dtype=np.float64)
        for r in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step")
    }
    lo_per_item = np.zeros((n_stages, n60), dtype=np.float64)

    for s in range(0, n60, fisher_bsz):
        e = min(n60, s + fisher_bsz)
        sub = items_60[s:e]
        bsz = len(sub)
        prompts = [ctrl.format_prompt(it, prompt_mode=prompt_mode) for it in sub]
        inp, lens = ctrl._encode_right_padded(prompts)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

        with torch.enable_grad():
            seq_len = inp.shape[1]
            freqs_cis = ctrl._get_freqs_cis(seq_len)
            emb = ctrl.transformer.wte(inp).detach().requires_grad_(True)
            emb_scale = float(getattr(ctrl.model, "emb_scale", 1.0))
            input_embeds = emb * emb_scale if emb_scale != 1.0 else emb

            b_idx = torch.tensor(-1, device=torch.device("cpu"), dtype=torch.long)
            for block in ctrl.transformer.prelude:
                b_idx = b_idx + 1
                input_embeds = block(input_embeds, freqs_cis, b_idx, None, None)

            ckpt_map: dict[str, torch.Tensor] = {"prelude_end": input_embeds}
            x = ctrl.make_initial_state(input_embeds, init_scale=1.0, seed=seed)
            step_latents_detached: list[torch.Tensor] = [input_embeds.detach()]

            for step_idx in range(n_steps):
                x, b_idx = ctrl._run_adapter_and_core_step(
                    x=x,
                    input_embeds=input_embeds,
                    freqs_cis=freqs_cis,
                    block_idx=b_idx,
                    current_step=step_idx,
                    channel_bits=(1, 1),
                    gamma=1.0,
                )
                ckpt_map[f"step_{step_idx + 1}"] = x
                step_latents_detached.append(x.detach())

            x_fin, coda_trace = ctrl.run_coda_from_latent(x, freqs_cis)
            for c_i, c_ten in coda_trace:
                ckpt_map[f"coda_L{c_i}"] = c_ten

            logits_last = ctrl.lm_head(x_fin[row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
            probs_last = F.softmax(logits_last, dim=-1).detach()
            sqrt_p = torch.sqrt(probs_last.clamp(min=1e-30))

            ckpt_tensors = [ckpt_map[k] for k in ckpt_keys]
            ckpt_norms_sq = [
                ckpt_tensors[c_i].detach()[row_idx, col_idx].float().square().sum(dim=-1)
                for c_i in range(n_ckpts)
            ]

            vocab_sz = logits_last.shape[-1]
            tr_accum = [torch.zeros(bsz, dtype=torch.float32, device=ctrl.device) for _ in range(n_ckpts)]
            for r_i in range(n_rademacher):
                rad_bits = torch.randint(0, 2, (bsz, vocab_sz), generator=rng_f, dtype=torch.int8)
                rad = (rad_bits.to(device=ctrl.device, dtype=torch.float32) * 2.0) - 1.0
                dot_sp = (sqrt_p * rad).sum(dim=-1, keepdim=True)
                w_r = (sqrt_p * rad) - (dot_sp * probs_last)
                scalar_obj = (logits_last * w_r).sum()
                grads = torch.autograd.grad(
                    scalar_obj,
                    ckpt_tensors,
                    retain_graph=(r_i < n_rademacher - 1),
                )
                for c_i, g_ten in enumerate(grads):
                    tr_accum[c_i] += g_ten[row_idx, col_idx].float().square().sum(dim=-1)

            for c_i in range(n_ckpts):
                tr_val = tr_accum[c_i] / float(n_rademacher)
                rel_val = tr_val * ckpt_norms_sq[c_i] / float(ctrl.d_model)
                fisher_trace_mat[c_i, s:e] = tr_val.cpu().numpy()
                fisher_rel_mat[c_i, s:e] = rel_val.cpu().numpy()

        with torch.no_grad():
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
            h1_sets = [first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge) for it in sub]
            for st_i, lat_st in enumerate(step_latents_detached):
                fin_st, _ = ctrl.run_coda_from_latent(lat_st, freqs_cis)
                lg_st = ctrl.lm_head(fin_st[row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
                pr_st = F.softmax(lg_st, dim=-1)
                for i, it in enumerate(sub):
                    p1_m = float(pr_st[i, h1_sets[i]].sum().item())
                    p2_m = float(pr_st[i, tgt_sets[i]].sum().item())
                    lo_per_item[st_i, s + i] = math.log(max(p2_m, 1e-30)) - math.log(max(p1_m, 1e-30))
                    hop1_prob_traj["all"][st_i] += p1_m
                    hop2_prob_traj["all"][st_i] += p2_m
                    hop1_prob_traj[it.regime][st_i] += p1_m
                    hop2_prob_traj[it.regime][st_i] += p2_m

    fisher_checkpoints = []
    tr_means: dict[str, float] = {}
    rel_means: dict[str, float] = {}
    for c_i, k_lbl in enumerate(ckpt_keys):
        tr_ci = bootstrap_mean_ci(fisher_trace_mat[c_i], n_boot=n_boot, seed=9900 + c_i)
        rl_ci = bootstrap_mean_ci(fisher_rel_mat[c_i], n_boot=n_boot, seed=9950 + c_i)
        tr_means[k_lbl] = float(tr_ci[0])
        rel_means[k_lbl] = float(rl_ci[0])
        fisher_checkpoints.append(
            {
                "stage_key": k_lbl,
                "fisher_trace": {"mean": round(tr_ci[0], 5), "ci95": [round(tr_ci[1], 5), round(tr_ci[2], 5)]},
                "scale_normalized_fisher_kl_curvature": {
                    "mean": round(rl_ci[0], 5),
                    "ci95": [round(rl_ci[1], 5), round(rl_ci[2], 5)],
                },
            }
        )

    loop_exit_key = f"step_{n_steps}"
    coda_entry_key = "coda_L0"
    commitment_by_regime: dict[str, Any] = {}
    for r_key in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step"):
        idxs = (
            np.arange(n60)
            if r_key == "all"
            else np.asarray([idx for idx, it in enumerate(items_60) if it.regime == r_key], dtype=np.int64)
        )
        if idxs.size == 0:
            continue
        denom = float(idxs.size)
        mean_lo_curve = [float(np.mean(lo_per_item[st_i, idxs])) for st_i in range(n_stages)]
        commitment_by_regime[r_key] = {
            "stages": traj_stages,
            "hop1_bridge_prob": [round(float(x / denom), 5) for x in hop1_prob_traj[r_key]],
            "hop2_target_prob": [round(float(x / denom), 5) for x in hop2_prob_traj[r_key]],
            "hop2_minus_hop1_log_odds": [round(v, 4) for v in mean_lo_curve],
            "horizon_estimate": estimate_commitment_horizon(traj_stages, mean_lo_curve),
        }

    return {
        "n_steps": n_steps,
        "n_fisher_items": n60,
        "loop_to_coda_fisher_spike": {
            "loop_exit_key": loop_exit_key,
            "coda_entry_key": coda_entry_key,
            "trace_ratio": round(tr_means.get(coda_entry_key, 0.0) / max(tr_means.get(loop_exit_key, 0.0), 1e-12), 4),
            "scale_normalized_ratio": round(
                rel_means.get(coda_entry_key, 0.0) / max(rel_means.get(loop_exit_key, 0.0), 1e-12), 4
            ),
        },
        "fisher_pullback_trajectory": fisher_checkpoints,
        "commitment_horizon": commitment_by_regime,
    }


# ---------------------------------------------------------------------------
# Exp H4: Labovich Pass-1 Foil Hysteresis (K=4, 8) + Orthogonal Pulse-Chase
# ---------------------------------------------------------------------------

@_no_grad()
def run_exp_h4_hysteresis_and_pulse_chase(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    depths: Sequence[int] = (4, 8),
    items_per_regime: int = 20,
    batch_size: int = 16,
    foil_betas: Sequence[float] = (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00, 1.25),
    orth_alphas: Sequence[float] = (0.0, 0.25, 0.50, 1.00, 2.00),
    prompt_mode: str = "zero_shot",
    seed: int = 20260701,
) -> dict[str, Any]:
    """Exp H4: Pass-1-only foil interpolation at K=4, 8 and orthogonal tangent pulse-chase at K=8."""
    items_60 = _select_balanced_subset(suite, per_regime=items_per_regime)
    n60 = len(items_60)
    sorted_depths = sorted({int(k) for k in depths if int(k) >= 2})
    max_k = max(sorted_depths) if sorted_depths else 8
    tracked_steps = sorted({2, *sorted_depths})

    betas_t = tuple(float(b) for b in foil_betas)
    alphas_t = tuple(float(a) for a in orth_alphas)
    n_b = len(betas_t)
    n_a = len(alphas_t)

    target_p_by_step = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_steps}
    foil_p_by_step = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_steps}
    coda_acc_by_step = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_steps}
    coda_cap_by_step = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_steps}
    coda_lo_by_step = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_steps}

    pulse_rel_by_step = {p: np.zeros((n_a, n60), dtype=np.float64) for p in tracked_steps}
    pulse_rel_coda = np.zeros((n_a, n60), dtype=np.float64)
    pulse_acc_coda = np.zeros((n_a, n60), dtype=np.float64)
    pulse_kl_coda = np.zeros((n_a, n60), dtype=np.float64)

    rng_p = torch.Generator(device="cpu").manual_seed(20260704)

    for s in range(0, n60, batch_size):
        e = min(n60, s + batch_size)
        sub = items_60[s:e]
        bsz = len(sub)
        clean_prompts = [ctrl.format_prompt(it, prompt_mode=prompt_mode, use_foil=False) for it in sub]
        foil_prompts = [ctrl.format_prompt(it, prompt_mode=prompt_mode, use_foil=True) for it in sub]

        inp_c, lens_c = ctrl._encode_right_padded(clean_prompts)
        inp_f, lens_f = ctrl._encode_right_padded(foil_prompts)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_c = torch.tensor([l_i - 1 for l_i in lens_c], dtype=torch.long, device=ctrl.device)
        col_f = torch.tensor([l_i - 1 for l_i in lens_f], dtype=torch.long, device=ctrl.device)
        tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
        foil_sets = [first_token_variant_ids(ctrl.tokenizer, it.foil_answer) for it in sub]

        clean_full = ctrl.unrolled_huginn_forward(inp_c, n_steps=max_k, init_scale=1.0, seed=seed)
        foil_p1 = ctrl.unrolled_huginn_forward(inp_f, n_steps=1, init_scale=1.0, seed=seed)

        prelude_c = clean_full["prelude_state"]
        freqs_c = clean_full["freqs_cis"]
        b_idx_after_p1 = clean_full["step_block_indices"][0]
        x1_clean = clean_full["step_exits"][0]
        x1_foil_tok = foil_p1["step_exits"][0][row_idx, col_f].float()
        x1_clean_tok = x1_clean[row_idx, col_c].float()

        clean_step_refs = {p: clean_full["step_exits"][p - 1][row_idx, col_c].float() for p in tracked_steps}
        coda_clean_ref = clean_full["last_hidden_state"][row_idx, col_c].float()
        clean_logits_ref = ctrl.lm_head(
            clean_full["last_hidden_state"][row_idx, col_c].to(ctrl.lm_head.weight.dtype)
        ).float()
        clean_lp_ref = F.log_softmax(clean_logits_ref, dim=-1)
        clean_p_ref = torch.exp(clean_lp_ref)

        # 4A: On-manifold foil interpolation entering Step 2
        for b_i, beta in enumerate(betas_t):
            x_cur = x1_clean.clone()
            x_cur[row_idx, col_c] = ((1.0 - beta) * x1_clean_tok + beta * x1_foil_tok).to(x1_clean.dtype)
            b_idx = b_idx_after_p1.clone()
            for step_num in range(2, max_k + 1):
                x_cur, b_idx = ctrl._run_adapter_and_core_step(
                    x=x_cur,
                    input_embeds=prelude_c,
                    freqs_cis=freqs_c,
                    block_idx=b_idx,
                    current_step=step_num - 1,
                    channel_bits=(1, 1),
                    gamma=1.0,
                )
                if step_num in target_p_by_step:
                    fin_p, _ = ctrl.run_coda_from_latent(x_cur, freqs_c)
                    lg_p = ctrl.lm_head(fin_p[row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float()
                    pr_p = F.softmax(lg_p, dim=-1)
                    top_p = lg_p.argmax(dim=-1).tolist()
                    for i in range(bsz):
                        t_ids = tgt_sets[i]
                        fl_ids = foil_sets[i]
                        pt = float(pr_p[i, t_ids].sum().item())
                        pf = float(pr_p[i, fl_ids].sum().item())
                        target_p_by_step[step_num][b_i, s + i] = pt
                        foil_p_by_step[step_num][b_i, s + i] = pf
                        coda_acc_by_step[step_num][b_i, s + i] = 1.0 if top_p[i] in t_ids else 0.0
                        coda_cap_by_step[step_num][b_i, s + i] = 1.0 if top_p[i] in fl_ids else 0.0
                        coda_lo_by_step[step_num][b_i, s + i] = math.log(max(pt, 1e-30)) - math.log(max(pf, 1e-30))

        # 4B: Orthogonal tangent pulse-chase entering Step 2
        raw_u = torch.randn((bsz, ctrl.d_model), generator=rng_p, dtype=torch.float32).to(ctrl.device)
        e1 = x1_clean_tok / torch.linalg.vector_norm(x1_clean_tok, dim=-1, keepdim=True).clamp(min=1e-12)
        raw_u = raw_u - (raw_u * e1).sum(dim=-1, keepdim=True) * e1
        df = x1_foil_tok - x1_clean_tok
        e2 = df / torch.linalg.vector_norm(df, dim=-1, keepdim=True).clamp(min=1e-12)
        raw_u = raw_u - (raw_u * e2).sum(dim=-1, keepdim=True) * e2
        u_orth = raw_u / torch.linalg.vector_norm(raw_u, dim=-1, keepdim=True).clamp(min=1e-12)
        base_nrm = torch.linalg.vector_norm(x1_clean_tok, dim=-1, keepdim=True).clamp(min=1e-8)

        for a_i, alpha in enumerate(alphas_t):
            x_cur = x1_clean.clone()
            delta_vec = float(alpha) * base_nrm * u_orth
            x_cur[row_idx, col_c] = (x1_clean_tok + delta_vec).to(x1_clean.dtype)
            d_in = torch.linalg.vector_norm(delta_vec, dim=-1).clamp(min=1e-8)
            b_idx = b_idx_after_p1.clone()

            for step_num in range(2, max_k + 1):
                x_cur, b_idx = ctrl._run_adapter_and_core_step(
                    x=x_cur,
                    input_embeds=prelude_c,
                    freqs_cis=freqs_c,
                    block_idx=b_idx,
                    current_step=step_num - 1,
                    channel_bits=(1, 1),
                    gamma=1.0,
                )
                if step_num in pulse_rel_by_step:
                    if alpha == 0.0:
                        pulse_rel_by_step[step_num][a_i, s:e] = 0.0
                    else:
                        d_st = torch.linalg.vector_norm(
                            x_cur[row_idx, col_c].float() - clean_step_refs[step_num], dim=-1
                        )
                        pulse_rel_by_step[step_num][a_i, s:e] = (d_st / d_in).cpu().numpy()

            fin_o, _ = ctrl.run_coda_from_latent(x_cur, freqs_c)
            if alpha == 0.0:
                pulse_rel_coda[a_i, s:e] = 0.0
            else:
                d_cd = torch.linalg.vector_norm(fin_o[row_idx, col_c].float() - coda_clean_ref, dim=-1)
                pulse_rel_coda[a_i, s:e] = (d_cd / d_in).cpu().numpy()

            lg_o = ctrl.lm_head(fin_o[row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float()
            lp_o = F.log_softmax(lg_o, dim=-1)
            kl_o = (clean_p_ref * (clean_lp_ref - lp_o)).sum(dim=-1).clamp(min=0.0)
            pulse_kl_coda[a_i, s:e] = kl_o.cpu().numpy()
            top_o = lg_o.argmax(dim=-1).tolist()
            for i in range(bsz):
                pulse_acc_coda[a_i, s + i] = 1.0 if top_o[i] in tgt_sets[i] else 0.0

    by_depth: dict[str, Any] = {}
    for k_depth in sorted_depths:
        rows = []
        for b_i, beta in enumerate(betas_t):
            rows.append(
                {
                    "beta": beta,
                    "step2_foil_prob": round(float(np.mean(foil_p_by_step[2][b_i])), 5),
                    "step_k_target_prob": round(float(np.mean(target_p_by_step[k_depth][b_i])), 5),
                    "step_k_foil_prob": round(float(np.mean(foil_p_by_step[k_depth][b_i])), 5),
                    "coda_target_top1_acc": round(float(np.mean(coda_acc_by_step[k_depth][b_i])), 4),
                    "coda_foil_capture_rate": round(float(np.mean(coda_cap_by_step[k_depth][b_i])), 4),
                    "coda_target_minus_foil_lo": round(float(np.mean(coda_lo_by_step[k_depth][b_i])), 4),
                }
            )
        summary = summarize_hysteresis_curve(
            betas_t,
            [float(np.mean(foil_p_by_step[2][i])) for i in range(n_b)],
            [float(np.mean(foil_p_by_step[k_depth][i])) for i in range(n_b)],
            [float(np.mean(coda_acc_by_step[k_depth][i])) for i in range(n_b)],
            [float(np.mean(coda_cap_by_step[k_depth][i])) for i in range(n_b)],
        )
        by_depth[f"K={k_depth}"] = {
            "n_steps": k_depth,
            "foil_curve": rows,
            "hysteresis_summary": summary,
        }

    pulse_rows = []
    for a_i, alpha in enumerate(alphas_t):
        r2 = float(np.mean(pulse_rel_by_step[2][a_i]))
        rk = float(np.mean(pulse_rel_by_step[max_k][a_i]))
        rc = float(np.mean(pulse_rel_coda[a_i]))
        row_p: dict[str, Any] = {
            "alpha": alpha,
            "rel_deviation_step2": round(r2, 5),
            f"rel_deviation_step{max_k}": round(rk, 5),
            "rel_deviation_coda": round(rc, 5),
            "final_over_step2_damping_ratio": round(rk / max(r2, 1e-8), 4) if alpha > 0.0 else 0.0,
            "coda_target_top1_acc": round(float(np.mean(pulse_acc_coda[a_i])), 4),
            "coda_kl_from_clean_nats": round(float(np.mean(pulse_kl_coda[a_i])), 5),
        }
        pulse_rows.append(row_p)

    return {
        "n_items": n60,
        "depths": sorted_depths,
        "by_depth": by_depth,
        "orthogonal_pulse_chase": pulse_rows,
    }


# ---------------------------------------------------------------------------
# Exp H5: Continuous Adapter Recurrent-State Gain Sweep (gamma in 0..1.5)
# ---------------------------------------------------------------------------

@_no_grad()
def run_exp_h5_adapter_conductance_sweep(
    ctrl: BiophysicalHuginnController,
    suite: dict[str, Any],
    n_steps: int = 8,
    items_per_regime: int = 20,
    batch_size: int = 16,
    gammas: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50),
    prompt_mode: str = "zero_shot",
) -> dict[str, Any]:
    """Exp H5: Continuous adapter recurrent-state gain sweep `gamma in {0.0..1.50}` at `K=8`."""
    items_60 = _select_balanced_subset(suite, per_regime=items_per_regime)
    gamma_rows = []
    for gm_val in gammas:
        sc = _score_huginn_items(
            ctrl,
            items_60,
            n_steps=n_steps,
            channel_bits=(1, 1),
            gamma=float(gm_val),
            batch_size=batch_size,
            prompt_mode=prompt_mode,
        )
        gamma_rows.append(
            {
                "gamma": float(gm_val),
                "hop2_top1_acc": round(float(np.mean(sc["top1_acc"])), 4),
                "hop2_target_prob": round(float(np.mean(sc["target_prob"])), 5),
                "hop1_bridge_prob": round(float(np.mean(sc["hop1_prob"])), 5),
                "target_vs_foil_lo": round(float(np.mean(sc["target_vs_foil_lo"])), 4),
            }
        )
    return {
        "n_steps": n_steps,
        "n_items": len(items_60),
        "prompt_mode": prompt_mode,
        "adapter_conductance_sweep": gamma_rows,
    }


# ---------------------------------------------------------------------------
# Unified Resumable Phase 7 Huginn Suite Runner
# ---------------------------------------------------------------------------

def load_huginn_controller(
    model_dir: Path | str,
    model_label: str = "tomg-group-umd/huginn-0125",
) -> tuple[BiophysicalHuginnController, dict[str, Any]]:
    """Load `RavenForCausalLM` from a local verified snapshot directory."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True, trust_remote_code=True)
    dev = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        torch_dtype=dt,
        device_map=dev,
    )
    model.eval()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    ctrl = BiophysicalHuginnController(model=model, tokenizer=tok, model_label=model_label)
    gpu_name = torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else "cpu"
    meta = {
        "model_id": model_label,
        "gpu_name": gpu_name,
        "d_model": ctrl.d_model,
        "n_prelude": ctrl.n_prelude,
        "n_core": ctrl.n_core,
        "n_coda": ctrl.n_coda,
        "cuda_allocated_gb": round(float(torch.cuda.memory_allocated() / 1e9), 3)
        if (torch is not None and torch.cuda.is_available())
        else 0.0,
        "load_seconds": round(time.perf_counter() - t0, 2),
    }
    return ctrl, meta


def run_phase7_huginn_suite(
    controller: BiophysicalHuginnController,
    tokenizer: Any = None,
    stimuli: dict[str, Any] | None = None,
    batch_size: int = 16,
    lens_per_regime: int = 20,
    jacobian_per_regime: int = 12,
    fisher_per_regime: int = 20,
    n_rademacher: int = 6,
    fisher_bsz: int = 8,
    hysteresis_per_regime: int = 20,
    zero_state_items: int = 20,
    n_boot: int = 4000,
    checkpoint_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run all Phase 7 Block B2 Huginn-0125 experiments (Preflight + H1..H5) with atomic JSON checkpoints."""
    _ = tokenizer
    t0 = time.perf_counter()
    suite = stimuli if stimuli is not None else build_phase6_stimulus_suite()
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None

    if ckpt is not None and ckpt.is_file():
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        payload["completed"] = False
    else:
        payload = {
            "phase": "phase7_huginn_biophysical_replication",
            "model_label": controller.model_label,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed": False,
        }

    def _save_ckpt() -> None:
        if ckpt is not None:
            _atomic_save_json(ckpt, payload)

    if "preflight_equivalence" not in payload:
        payload["preflight_equivalence"] = run_huginn_preflight_equivalence(controller, depths=(1, 4, 8))
        _save_ckpt()

    if "baseline_calibration" not in payload:
        payload["baseline_calibration"] = run_huginn_baseline_calibration(
            controller, suite, n_steps=8, batch_size=batch_size, threshold=0.30
        )
        payload["prompt_mode"] = payload["baseline_calibration"]["selected_prompt_mode"]
        _save_ckpt()

    prompt_mode = str(payload.get("prompt_mode", "zero_shot"))

    if "zero_state_sensitivity" not in payload:
        payload["zero_state_sensitivity"] = run_huginn_zero_state_sensitivity(
            controller,
            suite,
            n_items=zero_state_items,
            n_steps=8,
            batch_size=min(10, batch_size),
            prompt_mode=prompt_mode,
        )
        _save_ckpt()

    if "exp_h1_adapter_wht" not in payload:
        payload["exp_h1_adapter_wht"] = run_exp_h1_adapter_wht(
            controller,
            suite,
            depths=(4, 8),
            batch_size=batch_size,
            prompt_mode=prompt_mode,
            n_boot=n_boot,
        )
        _save_ckpt()

    if "exp_h2_gain_vs_amplitude" not in payload:
        payload["exp_h2_gain_vs_amplitude"] = run_exp_h2_gain_vs_amplitude(
            controller,
            suite,
            n_steps=8,
            lens_per_regime=lens_per_regime,
            jacobian_per_regime=jacobian_per_regime,
            conductance_steps=(1, 2, 4, 8),
            power_steps=(1, 2, 3, 4, 6, 8),
            batch_size=min(12, batch_size),
            prompt_mode=prompt_mode,
            n_boot=n_boot,
        )
        _save_ckpt()

    if "exp_h3_fisher_and_commitment" not in payload:
        payload["exp_h3_fisher_and_commitment"] = run_exp_h3_fisher_and_commitment(
            controller,
            suite,
            n_steps=8,
            fisher_per_regime=fisher_per_regime,
            n_rademacher=n_rademacher,
            fisher_bsz=min(fisher_bsz, batch_size),
            prompt_mode=prompt_mode,
            n_boot=n_boot,
        )
        _save_ckpt()

    if "exp_h4_hysteresis_pulse_chase" not in payload:
        payload["exp_h4_hysteresis_pulse_chase"] = run_exp_h4_hysteresis_and_pulse_chase(
            controller,
            suite,
            depths=(4, 8),
            items_per_regime=hysteresis_per_regime,
            batch_size=batch_size,
            prompt_mode=prompt_mode,
        )
        _save_ckpt()

    if "exp_h5_adapter_conductance_sweep" not in payload:
        payload["exp_h5_adapter_conductance_sweep"] = run_exp_h5_adapter_conductance_sweep(
            controller,
            suite,
            n_steps=8,
            items_per_regime=hysteresis_per_regime,
            batch_size=batch_size,
            prompt_mode=prompt_mode,
        )
        _save_ckpt()

    payload["completed"] = True
    payload["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    _save_ckpt()
    return payload
