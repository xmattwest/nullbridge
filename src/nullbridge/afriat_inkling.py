"""Part 4 GPU Pipeline: Afriat Revealed Preference, GARP Efficiency, & Moral Pricing in Native Looped MoEs.

Executes a 5-block zero-leakage study across `thinkingmachines/TML-d24-varloop-research`
(`K = 1, 2, 3, 4, 5` silent passes) and flagship `thinkingmachines/Inkling-2-Mini` (`K = 3`):
  - Block 1: 200-pair non-financial 8-axis moral geometry (`guilt`, `captivity`, `contamination`,
    `sacrilege`, `spite`, `predation`, `awe_dread`, `warmth` orthogonalized against `valence`,
    `arousal`, and `<bos>` sinks) across all 4 cross-pass message channels (`Q, K, V, MLP`) +
    96-pair 6x8 clinical financial Moral Atlas across `K = 1..5`.
  - Block 2: Formal Afriat (1967) / Varian (1982) Revealed-Preference & GARP audit across 4
    intersecting `T = 24` linear budget-set regimes (`576` observations/cell), computing exact
    Afriat `CCEI`, Warshall GARP/WARP/SARP violations, Bronars Monte Carlo power, constructive
    Afriat `(u_t, lambda_t)` utility recovery, and Homo Economicus fungibility efficiency under
    `unclamped`, `msg_homo_economicus_8d`, `msg_qkv_8d`, `msg_mlp_8d`, and `zero_flop_8d`.
  - Block 3: Continuous Ultimatum disparity (`50/50` to `99.9/0.1`), Blount (1995) human-vs.-computer
    intentionality DiD across `K = 1..5`, and bidirectional cross-pass message-bank steering
    double dissociation (`spite`, `guilt`, `captivity`, `contamination`, `valence`).
  - Block 4: 6 continuous 7-rung Dollar Willingness-to-Pay (`WTP`) reservation curves + `$100`
    pocket-scale control (`392` items) across `K = 1..5`, `8D` clamping, and message-bank steering.
  - Block 5: Silent `0-CoT` corporate rationalization vs. private `<|content_thinking|>` scratchpad
    moral scrubbing (`24` open-ended pricing tasks) + `32`-item `0%`-APR debt subtraction cure.
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import re
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

from .afriat_stimuli import (
    ALL_CONCEPTS,
    BANNED_AFFECTIVE_TERMS,
    BASE_INTERSECTING_INTERCEPTS_24,
    BROAD_MORAL_LEXICON,
    CONTROL_CONCEPTS,
    CORE_ARCHETYPES,
    CORE_CONCEPTS,
    CURVE_PERMILLE,
    GARPBudgetItem,
    GRID_SHARES_11,
    InequityItem,
    MENU_LETTERS_11,
    OpenDecisionItem,
    RATIONALIZATION_JARGON,
    WtpCurveItem,
    audit_all_afriat_inkling_stimuli,
    build_debt_math_suite,
    build_garp_budget_suites,
    build_inequity_suite,
    build_open_decision_suite,
    build_wtp_curves,
    default_stimuli_dir,
    load_concepts,
    load_sanity_items,
    load_transactions,
    tokenize_words,
)
from .algebra import (
    bootstrap_mean_ci,
    exact_or_mc_signflip_p,
    extract_controlled_basis,
    holm_adjust,
)
from .garp import (
    audit_revealed_preferences,
    compute_bronars_power,
)
from .gemma4_compact import build_scenario_reading_prompt
from .track_a import (
    InklingLoopController,
    _atomic_save_json,
    _no_grad,
    collect_pass_states_and_messages,
)


EPS = 1e-12
LEADING_NUM_RE = re.compile(r"^\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
DECISION_USD_RE = re.compile(r"DECISION_USD\s*:\s*\$?\s*([0-9][0-9,]*(?:\.[0-9]+)?)", re.IGNORECASE)
ANY_DOLLAR_RE = re.compile(r"\$\s*([0-9][0-9,]*(?:\.[0-9]+)?)")
RATIONALE_SPLIT_RE = re.compile(r"RATIONALE\s*:\s*", re.IGNORECASE)
INKLING_THINK_CLOSED_RE = re.compile(
    r"<\|content_thinking\|>(.*?)<\|end_message\|>\s*(?:<\|message_model\|>)?(?:<\|content_text\|>)?(.*)",
    re.DOTALL,
)
INKLING_THINK_OPEN_RE = re.compile(r"<\|content_thinking\|>(.*)", re.DOTALL)
INKLING_SPECIAL_CLEAN_RE = re.compile(
    r"(?:<\|message_model\|>|<\|content_text\|>|<\|end_message\|>|<\|content_model_end_sampling\|>|<\|endoftext\|>|<\|begin_of_text\|>)"
)


def _clean_inkling_text(raw: str) -> str:
    return INKLING_SPECIAL_CLEAN_RE.sub("", raw).strip()


def split_inkling_thought_and_public(raw_decoded: str) -> tuple[str, str, bool, bool]:
    """Split Inkling `<|content_thinking|>...<|end_message|><|message_model|><|content_text|>...` output."""
    m_closed = INKLING_THINK_CLOSED_RE.search(raw_decoded)
    if m_closed is not None:
        return (
            _clean_inkling_text(m_closed.group(1)),
            _clean_inkling_text(m_closed.group(2)),
            True,
            True,
        )
    m_open = INKLING_THINK_OPEN_RE.search(raw_decoded)
    if m_open is not None:
        return _clean_inkling_text(m_open.group(1)), "", True, False
    return "", _clean_inkling_text(raw_decoded), False, True


class AfriatInklingController(InklingLoopController):
    """Extends `InklingLoopController` with 11-way GARP menus, message steering, and `<|content_thinking|>` decoding."""

    def __init__(self, model: nn.Module, tokenizer: Any, model_label: str) -> None:
        super().__init__(model=model, tokenizer=tokenizer, model_label=model_label)
        # Keep the 0.92 GB backup copy of `lm_head` in host RAM instead of GPU VRAM
        self._orig_lm_head = self._orig_lm_head.cpu()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.active_msg_steer: torch.Tensor | None = None   # (d_model,) float32 on device
        self.active_msg_steer_channels: tuple[int, ...] = (0, 1, 2, 3)
        self.active_final_steer: torch.Tensor | None = None

    def reset_interventions(self) -> None:
        self.active_msg_q = None
        self.active_msg_channels = ()
        self.active_final_q = None
        self.set_active_lora(None)
        self.active_msg_steer = None
        self.active_msg_steer_channels = (0, 1, 2, 3)
        self.active_final_steer = None
        with torch.no_grad():
            self.text_model.message_input.query.copy_(self._orig_msg_query)
            self.lm_head.weight.copy_(self._orig_lm_head.to(device=self.device, dtype=self.lm_head.weight.dtype))
            for (l_idx, g_name), orig_slice in self._orig_gate_down_msg.items():
                gate_mod = getattr(self.text_model.layers[l_idx], g_name)
                gate_mod.down_proj.weight[:, self.d_model :].copy_(orig_slice)

    def apply_zero_flop_gate_fold(
        self,
        q_msg: np.ndarray,
        channels: Sequence[int] = (0, 1, 2, 3),
        fold_lm_head_q: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Memory-safe zero-FLOP fold supporting CPU-backed `_orig_lm_head`."""
        q_m = torch.tensor(q_msg, dtype=torch.float64, device=self.device)
        proj_m = torch.eye(self.d_model, dtype=torch.float64, device=self.device) - (q_m @ q_m.t())
        gate_names = ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate")
        max_query_leak = 0.0
        max_gate_leak = 0.0
        max_head_leak = 0.0

        with torch.no_grad():
            q_orig = self._orig_msg_query.to(dtype=torch.float64)
            q_new = q_orig.clone()
            for ch in channels:
                q_new[:, ch, :] = q_orig[:, ch, :] @ proj_m
                max_query_leak = max(max_query_leak, float(torch.linalg.norm(q_new[:, ch, :] @ q_m, ord="fro").item()))
            self.text_model.message_input.query.copy_(q_new.to(dtype=self._orig_msg_query.dtype))

            for l_idx in range(self.loop_start, self.loop_end):
                layer = self.text_model.layers[l_idx]
                for ch in channels:
                    g_name = gate_names[ch]
                    orig_s = self._orig_gate_down_msg[(l_idx, g_name)].to(dtype=torch.float64)
                    folded_s = orig_s @ proj_m
                    max_gate_leak = max(max_gate_leak, float(torch.linalg.norm(folded_s @ q_m, ord="fro").item()))
                    gate_mod = getattr(layer, g_name)
                    gate_mod.down_proj.weight[:, self.d_model :].copy_(folded_s.to(dtype=gate_mod.down_proj.weight.dtype))

            if fold_lm_head_q is not None:
                q_f = torch.tensor(fold_lm_head_q, dtype=torch.float64, device=self.device)
                proj_f = torch.eye(self.d_model, dtype=torch.float64, device=self.device) - (q_f @ q_f.t())
                w_orig_cpu = self._orig_lm_head
                target_dtype = self.lm_head.weight.dtype
                chunk = 8192
                for s in range(0, w_orig_cpu.shape[0], chunk):
                    e = min(w_orig_cpu.shape[0], s + chunk)
                    blk = w_orig_cpu[s:e].to(device=self.device, dtype=torch.float64) @ proj_f
                    max_head_leak = max(max_head_leak, float(torch.linalg.norm(blk @ q_f, ord="fro").item()))
                    self.lm_head.weight[s:e].copy_(blk.to(dtype=target_dtype))

        return {
            "max_query_null_leakage": max_query_leak,
            "max_gate_down_null_leakage": max_gate_leak,
            "max_lm_head_null_leakage": max_head_leak,
        }

    @_no_grad()
    def _forward_last_token_logits(self, rendered_prompts: Sequence[str], max_length: int = 1536) -> torch.Tensor:
        """Compute `(bsz, vocab)` float32 logits ONLY at each sequence's final token (`870x` less logit VRAM)."""
        inp, msk, lens = self._encode_right_padded(rendered_prompts, max_length=max_length)
        out = self.text_model(input_ids=inp, attention_mask=msk, use_cache=False)
        h = out[0] if isinstance(out, tuple) else out.last_hidden_state
        row_idx = torch.arange(len(lens), device=self.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=self.device)
        h_last = h[row_idx, col_idx]  # (bsz, d_model)
        return self.logit_softcap(self.lm_head(h_last)).float()

    @_no_grad()
    def score_counterbalanced_choices(
        self,
        scenarios: Sequence[str],
        target_options: Sequence[str],
        reference_options: Sequence[str],
        n_passes: int,
        batch_size: int = 16,
    ) -> np.ndarray:
        """Memory-safe position-counterbalanced `0-CoT` binary choice log-odds (`target` minus `reference`)."""
        prompts_fwd = [
            f"{sc}\n\nOption A: {tgt}\nOption B: {ref}\n\nRespond with a single letter ONLY (A or B)."
            for sc, tgt, ref in zip(scenarios, target_options, reference_options)
        ]
        prompts_rev = [
            f"{sc}\n\nOption A: {ref}\nOption B: {tgt}\n\nRespond with a single letter ONLY (A or B)."
            for sc, tgt, ref in zip(scenarios, target_options, reference_options)
        ]
        tok_a = self.tokenizer.encode("A", add_special_tokens=False)[0]
        tok_b = self.tokenizer.encode("B", add_special_tokens=False)[0]

        def _ab_lo(p_list: Sequence[str]) -> np.ndarray:
            arr = np.zeros(len(p_list), dtype=np.float64)
            for s in range(0, len(p_list), batch_size):
                e = min(len(p_list), s + batch_size)
                rendered = [self.format_0cot_prompt(p) for p in p_list[s:e]]
                last_l = self._forward_last_token_logits(rendered, max_length=1280)
                arr[s:e] = (last_l[:, tok_a] - last_l[:, tok_b]).cpu().numpy()
            return arr

        with self.active_passes(n_passes):
            lo_fwd = _ab_lo(prompts_fwd)
            lo_rev = _ab_lo(prompts_rev)
        return 0.5 * (lo_fwd - lo_rev)

    def _on_message_input(self, _mod: nn.Module, _inp: Any, output: torch.Tensor) -> torch.Tensor:
        out = super()._on_message_input(_mod, _inp, output)
        if self.active_msg_steer is not None and self.active_msg_steer_channels:
            out = out.clone()
            ch_idx = list(self.active_msg_steer_channels)
            sub = out[..., ch_idx, :].float() + self.active_msg_steer[None, None, None, :]
            out[..., ch_idx, :] = sub.to(out.dtype)
        return out

    def _on_final_norm(self, _mod: nn.Module, _inp: Any, output: torch.Tensor) -> torch.Tensor:
        out = super()._on_final_norm(_mod, _inp, output)
        if self.active_final_steer is not None:
            xf = out.float() + self.active_final_steer[None, None, :]
            out = xf.to(out.dtype)
        return out

    @contextmanager
    def message_steering(
        self,
        msg_vec: np.ndarray,
        delta_scale: float,
        channels: Sequence[int] = (0, 1, 2, 3),
        final_vec: np.ndarray | None = None,
        final_scale: float = 0.0,
    ):
        prev_ms = self.active_msg_steer
        prev_ch = self.active_msg_steer_channels
        prev_fs = self.active_final_steer
        try:
            self.active_msg_steer = torch.tensor(
                np.asarray(msg_vec, dtype=np.float32) * float(delta_scale),
                dtype=torch.float32,
                device=self.device,
            )
            self.active_msg_steer_channels = tuple(int(c) for c in channels)
            if final_vec is not None and abs(final_scale) > 1e-12:
                self.active_final_steer = torch.tensor(
                    np.asarray(final_vec, dtype=np.float32) * float(final_scale),
                    dtype=torch.float32,
                    device=self.device,
                )
            yield
        finally:
            self.active_msg_steer = prev_ms
            self.active_msg_steer_channels = prev_ch
            self.active_final_steer = prev_fs

    def format_thinking_prompt(self, user_text: str, reasoning_effort: str = "medium") -> str:
        """Format Inkling prompt with active `<|content_thinking|>` effort (`0.7` for `medium`)."""
        try:
            rendered = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": user_text}],
                tokenize=False,
                add_generation_prompt=False,
                reasoning_effort=reasoning_effort,
            )
            return rendered
        except Exception:
            eff_val = "0.7" if reasoning_effort == "medium" else "0.9"
            return (
                f"<|message_system|><|content_text|>Thinking effort level: {eff_val}<|end_message|>"
                f"<|message_user|><|content_text|>{user_text}<|end_message|>"
            )

    @_no_grad()
    def score_counterbalanced_11way_menus(
        self,
        items: Sequence[GARPBudgetItem],
        n_passes: int,
        batch_size: int = 32,
    ) -> dict[str, np.ndarray]:
        """Score 11-option frontier menus in both `(0%..100%)` and `(100%..0%)` letter orders."""
        n = len(items)
        letter_ids = [self.tokenizer.encode(ch, add_special_tokens=False)[0] for ch in MENU_LETTERS_11]
        letter_tensor = torch.tensor(letter_ids, dtype=torch.long, device=self.device)

        prompts_fwd: list[str] = []
        prompts_rev: list[str] = []
        footer = "\n\nRespond with a single letter ONLY (A, B, C, D, E, F, G, H, I, J, or K)."
        for it in items:
            lines_f = [f"Option {MENU_LETTERS_11[k]}: {it.bundle_options[k]}" for k in range(11)]
            lines_r = [f"Option {MENU_LETTERS_11[k]}: {it.bundle_options[10 - k]}" for k in range(11)]
            prompts_fwd.append(f"{it.scenario}\n\n" + "\n".join(lines_f) + footer)
            prompts_rev.append(f"{it.scenario}\n\n" + "\n".join(lines_r) + footer)

        def _menu_logits(p_list: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
            mat = np.zeros((len(p_list), 11), dtype=np.float64)
            mass = np.zeros(len(p_list), dtype=np.float64)
            for s in range(0, len(p_list), batch_size):
                e = min(len(p_list), s + batch_size)
                rendered = [self.format_0cot_prompt(p) for p in p_list[s:e]]
                last_l = self._forward_last_token_logits(rendered, max_length=1536)
                last_p = F.softmax(last_l, dim=-1)
                mat[s:e] = last_l.index_select(1, letter_tensor).cpu().numpy()
                mass[s:e] = last_p.index_select(1, letter_tensor).sum(dim=-1).cpu().numpy()
            return mat, mass

        with self.active_passes(n_passes):
            lf, mf = _menu_logits(prompts_fwd)
            lr, mr = _menu_logits(prompts_rev)

        # Align reverse ordering `(A->10, ..., K->0)` back to canonical `0..10` (`share = 0.0..1.0`)
        avg_logits = 0.5 * (lf + lr[:, ::-1])
        shifted = avg_logits - np.max(avg_logits, axis=1, keepdims=True)
        exp_l = np.exp(shifted)
        probs = exp_l / np.maximum(np.sum(exp_l, axis=1, keepdims=True), EPS)
        modal_idx = np.argmax(avg_logits, axis=1)
        shares_vec = np.asarray(GRID_SHARES_11, dtype=np.float64)
        expected_share = probs @ shares_vec
        modal_share = shares_vec[modal_idx]

        return {
            "probs": probs,                       # (N, 11)
            "modal_idx": modal_idx,               # (N,)
            "modal_share": modal_share,           # (N,)
            "expected_share": expected_share,     # (N,)
            "letter_mass": 0.5 * (mf + mr),       # (N,)
        }

    @_no_grad()
    def generate_completions(
        self,
        user_prompts: Sequence[str],
        n_passes: int,
        reasoning_effort: str = "none",
        assistant_prefix: str = "",
        rescue_prefix: str = "DECISION_USD: $",
        max_new_tokens: int = 128,
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        """KV-cached greedy generation supporting both `0-CoT` and `<|content_thinking|>` modes."""
        pad_id = self.tokenizer.pad_token_id or 0
        stop_ids: set[int] = set()
        if reasoning_effort == "none":
            stop_tokens = ("<|end_message|>", "<|content_model_end_sampling|>", "<|endoftext|>")
        else:
            # Must NOT stop at `<|end_message|>` because `<|end_message|>` closes `<|content_thinking|>` before public text!
            stop_tokens = ("<|content_model_end_sampling|>", "<|endoftext|>")
        for tok_s in stop_tokens:
            ids = self.tokenizer.encode(tok_s, add_special_tokens=False)
            if len(ids) == 1:
                stop_ids.add(ids[0])
        if getattr(self.tokenizer, "eos_token_id", None) is not None and reasoning_effort == "none":
            stop_ids.add(int(self.tokenizer.eos_token_id))

        def _decode_batch(rendered_list: Sequence[str], cur_max_new: int) -> list[list[int]]:
            enc_all = [self.tokenizer.encode(r, add_special_tokens=False) for r in rendered_list]
            order = sorted(range(len(enc_all)), key=lambda idx: len(enc_all[idx]))
            out_by_idx: dict[int, list[int]] = {}

            for s in range(0, len(order), batch_size):
                b_idxs = order[s : min(len(order), s + batch_size)]
                b_seqs = [enc_all[i] for i in b_idxs]
                max_l = max(len(x) for x in b_seqs)
                bsz = len(b_seqs)
                inp = torch.full((bsz, max_l), pad_id, dtype=torch.long, device=self.device)
                msk = torch.zeros((bsz, max_l), dtype=torch.long, device=self.device)
                for i, seq in enumerate(b_seqs):
                    inp[i, max_l - len(seq) :] = torch.tensor(seq, dtype=torch.long, device=self.device)
                    msk[i, max_l - len(seq) :] = 1

                out = self.text_model(input_ids=inp, attention_mask=msk, use_cache=True)
                h_pref = out[0] if isinstance(out, tuple) else out.last_hidden_state
                cache = out.cache if hasattr(out, "cache") else getattr(out, "past_key_values", None)
                next_tok = self.logit_softcap(self.lm_head(h_pref[:, -1, :])).argmax(dim=-1, keepdim=True)
                gen_steps = [next_tok]
                done = torch.zeros(bsz, dtype=torch.bool, device=self.device)
                for sid in stop_ids:
                    done = done | (next_tok.squeeze(-1) == sid)

                for _ in range(cur_max_new - 1):
                    if bool(done.all()):
                        break
                    msk = torch.cat([msk, torch.ones((bsz, 1), dtype=torch.long, device=self.device)], dim=1)
                    if cache is not None:
                        out = self.text_model(input_ids=next_tok, attention_mask=msk, cache=cache, use_cache=True)
                        cache = out.cache if hasattr(out, "cache") else getattr(out, "past_key_values", None)
                    else:
                        inp = torch.cat([inp, next_tok], dim=1)
                        out = self.text_model(input_ids=inp, attention_mask=msk, use_cache=False)
                    h_step = out[0] if isinstance(out, tuple) else out.last_hidden_state
                    next_tok = self.logit_softcap(self.lm_head(h_step[:, -1, :])).argmax(dim=-1, keepdim=True)
                    gen_steps.append(next_tok)
                    for sid in stop_ids:
                        done = done | (next_tok.squeeze(-1) == sid)

                gen_mat = torch.cat(gen_steps, dim=1).tolist()
                for i, orig_i in enumerate(b_idxs):
                    trimmed: list[int] = []
                    for tid in gen_mat[i]:
                        if tid in stop_ids:
                            break
                        trimmed.append(int(tid))
                    out_by_idx[orig_i] = trimmed

            return [out_by_idx[j] for j in range(len(rendered_list))]

        with self.active_passes(n_passes):
            if reasoning_effort == "none":
                rendered = [self.format_0cot_prompt(u, assistant_prefix=assistant_prefix) for u in user_prompts]
            else:
                rendered = [self.format_thinking_prompt(u, reasoning_effort=reasoning_effort) for u in user_prompts]

            gen_id_lists = _decode_batch(rendered, max_new_tokens)

            records: list[dict[str, Any]] = []
            rescue_indices: list[int] = []
            rescue_rendered: list[str] = []

            for idx, (r_prompt, g_ids) in enumerate(zip(rendered, gen_id_lists)):
                raw_txt = self.tokenizer.decode(g_ids, skip_special_tokens=False)
                if reasoning_effort == "none":
                    records.append(
                        {
                            "thought_text": "",
                            "public_text": _clean_inkling_text(assistant_prefix + raw_txt),
                            "thought_present": False,
                            "thought_closed": True,
                            "thought_rescued": False,
                            "n_gen_tokens": len(g_ids),
                        }
                    )
                else:
                    t_txt, p_txt, t_pres, t_closed = split_inkling_thought_and_public(raw_txt)
                    records.append(
                        {
                            "thought_text": t_txt,
                            "public_text": p_txt,
                            "thought_present": t_pres,
                            "thought_closed": t_closed,
                            "thought_rescued": False,
                            "n_gen_tokens": len(g_ids),
                        }
                    )
                    if (not t_closed) or (not p_txt.strip()):
                        rescue_indices.append(idx)
                        rescue_rendered.append(
                            r_prompt + raw_txt + f"<|end_message|><|message_model|><|content_text|>{rescue_prefix}"
                        )

            # Rescue any unclosed thinking trace cleanly so 100% of items emit public decision + rationale
            if rescue_rendered:
                prev_stops = set(stop_ids)
                for tok_s in ("<|end_message|>", "<|content_model_end_sampling|>"):
                    ids = self.tokenizer.encode(tok_s, add_special_tokens=False)
                    if len(ids) == 1:
                        stop_ids.add(ids[0])
                rescued_ids = _decode_batch(rescue_rendered, 96)
                stop_ids = prev_stops
                for r_i, g_r in zip(rescue_indices, rescued_ids):
                    pub_cont = _clean_inkling_text(rescue_prefix + self.tokenizer.decode(g_r, skip_special_tokens=False))
                    records[r_i]["public_text"] = pub_cont
                    records[r_i]["thought_rescued"] = True
                    records[r_i]["n_gen_tokens"] += len(g_r)

        return records


# ---------------------------------------------------------------------------
# Block 1: 8-Axis Non-Financial Moral Basis + 6x8 Financial Moral Atlas
# ---------------------------------------------------------------------------

def _project_multivariate_cartography(
    delta_mat: np.ndarray,            # (N, d)
    unit_dirs: dict[str, np.ndarray], # concept -> (d,)
    stream_rms: float,
    ridge: float = 0.05,
) -> dict[str, Any]:
    """Compute univariate `Cohen's d_z`, normalized RMS projection, and ridge-deconvolved partial `beta`."""
    n = delta_mat.shape[0]
    norms = np.maximum(np.linalg.norm(delta_mat, axis=-1), EPS)
    m_mat = np.stack([unit_dirs[c] for c in CORE_CONCEPTS], axis=0)  # (8, d)
    g_inv = np.linalg.inv(m_mat @ m_mat.T + ridge * np.eye(len(CORE_CONCEPTS)))

    cos_mat = np.zeros((n, len(CORE_CONCEPTS)), dtype=np.float64)
    stats: dict[str, dict[str, Any]] = {}
    p_raw_list: list[float] = []

    for j, c in enumerate(CORE_CONCEPTS):
        u = unit_dirs[c]
        dots = delta_mat @ u
        norm_p = dots / max(stream_rms, EPS)
        cos_p = dots / norms
        cos_mat[:, j] = cos_p
        mean_v = float(np.mean(norm_p))
        sd_v = float(np.std(norm_p, ddof=1)) if n > 1 else 0.0
        dz = mean_v / sd_v if sd_v > EPS else 0.0
        p_raw = exact_or_mc_signflip_p(norm_p)
        p_raw_list.append(p_raw)
        ci_l, ci_h = bootstrap_mean_ci(norm_p)
        stats[c] = {
            "cohen_dz": round(dz, 4),
            "mean_proj_rms": round(mean_v, 6),
            "ci95_rms": [round(ci_l, 6), round(ci_h, 6)],
            "mean_cosine": round(float(np.mean(cos_p)), 4),
            "p_signflip": round(p_raw, 6),
        }

    beta_per_fix = (g_inv @ cos_mat.T).T  # (N, 8)
    p_holm = holm_adjust(p_raw_list)
    p_part_raw = [exact_or_mc_signflip_p(beta_per_fix[:, j]) for j in range(len(CORE_CONCEPTS))]
    p_part_holm = holm_adjust(p_part_raw)

    for j, c in enumerate(CORE_CONCEPTS):
        bv = beta_per_fix[:, j]
        bl, bh = bootstrap_mean_ci(bv)
        stats[c]["p_holm"] = round(float(p_holm[j]), 6)
        stats[c]["partial_beta"] = round(float(np.mean(bv)), 4)
        stats[c]["partial_ci95"] = [round(bl, 4), round(bh, 4)]
        stats[c]["partial_p_holm"] = round(float(p_part_holm[j]), 6)

    top_concept = max(CORE_CONCEPTS, key=lambda c: abs(stats[c]["partial_beta"]))
    return {
        "top_partial_concept": top_concept,
        "concepts": stats,
    }


def run_block1_moral_atlas(
    controller: AfriatInklingController,
    pass_depths: Sequence[int],
    probe_passes: int = 3,
    batch_size: int = 32,
    stimuli_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract `<bos>`-safe 8D moral subspace and run 6x8 clinical financial Moral Atlas across `K`."""
    controller.reset_interventions()
    sdir = stimuli_dir or default_stimuli_dir()
    concepts = load_concepts(sdir)
    transactions = load_transactions(sdir)
    sanity_items = load_sanity_items(sdir)

    # 1. Sanity gate across pass_depths
    sanity_by_k: dict[str, Any] = {}
    for k_p in pass_depths:
        lo_san = controller.score_counterbalanced_choices(
            scenarios=[it.scenario for it in sanity_items],
            target_options=[it.better_option for it in sanity_items],
            reference_options=[it.worse_option for it in sanity_items],
            n_passes=k_p,
            batch_size=batch_size,
        )
        sanity_by_k[f"k{k_p}"] = {
            "accuracy": round(float(np.mean(lo_san > 0.0)), 4),
            "mean_log_odds": round(float(np.mean(lo_san)), 4),
            "n": len(sanity_items),
        }

    # 2. Extract 200-pair non-financial concept states (`400` vignettes, 0 digits, 0 finance words)
    eff_probe_p = max(2, probe_passes)
    pos_st = collect_pass_states_and_messages(
        controller,
        [build_scenario_reading_prompt(c.positive) for c in concepts],
        n_passes=eff_probe_p,
        batch_size=batch_size,
    )
    neg_st = collect_pass_states_and_messages(
        controller,
        [build_scenario_reading_prompt(c.negative) for c in concepts],
        n_passes=eff_probe_p,
        batch_size=batch_size,
    )

    pos_msg = pos_st["msg_last_by_pass"].mean(axis=(0, 2))  # (200, d)
    neg_msg = neg_st["msg_last_by_pass"].mean(axis=(0, 2))
    v_bos_msg = 0.5 * (pos_st["v_bos_msg"] + neg_st["v_bos_msg"])
    v_bos_msg = v_bos_msg / max(EPS, float(np.linalg.norm(v_bos_msg)))

    pos_fin = pos_st["final_last"]
    neg_fin = neg_st["final_last"]
    v_bos_fin = 0.5 * (pos_st["v_bos_final"] + neg_st["v_bos_final"])
    v_bos_fin = v_bos_fin / max(EPS, float(np.linalg.norm(v_bos_fin)))

    msg_rms = float(np.sqrt(np.mean(0.5 * (np.sum(pos_msg.astype(np.float64) ** 2, axis=-1) + np.sum(neg_msg.astype(np.float64) ** 2, axis=-1)))))
    fin_rms = float(np.sqrt(np.mean(0.5 * (np.sum(pos_fin.astype(np.float64) ** 2, axis=-1) + np.sum(neg_fin.astype(np.float64) ** 2, axis=-1)))))

    def _extract_8d_basis(p_arr: np.ndarray, n_arr: np.ndarray, v_bos: np.ndarray):
        p_map: dict[str, np.ndarray] = {}
        n_map: dict[str, np.ndarray] = {}
        for cname in ALL_CONCEPTS:
            idxs = [i for i, c in enumerate(concepts) if c.concept == cname]
            p_map[cname] = p_arr[idxs, None, :].astype(np.float64)
            n_map[cname] = n_arr[idxs, None, :].astype(np.float64)

        cb_affect = extract_controlled_basis(
            p_map, n_map, concepts=ALL_CONCEPTS, controls=CONTROL_CONCEPTS
        )
        p_bos = dict(p_map)
        n_bos = dict(n_map)
        p_bos["bos_sink"] = np.stack([v_bos, v_bos], axis=0)[:, None, :]
        n_bos["bos_sink"] = np.zeros((2, 1, controller.d_model), dtype=np.float64)
        cb_full = extract_controlled_basis(
            p_bos,
            n_bos,
            concepts=(*ALL_CONCEPTS, "bos_sink"),
            controls=(*CONTROL_CONCEPTS, "bos_sink"),
        )

        # Guarantee `< 1e-16` `<bos>` orthogonality on both individual unit vectors and the 8D basis `Q`
        ub = v_bos.astype(np.float64) / max(EPS, float(np.linalg.norm(v_bos)))
        unit_dirs: dict[str, np.ndarray] = {}
        for cname in CORE_CONCEPTS:
            v = cb_full.controlled_dirs[cname][0].astype(np.float64)
            v = v - float(ub @ v) * ub
            unit_dirs[cname] = v / max(EPS, float(np.linalg.norm(v)))
        for ctrl_c in CONTROL_CONCEPTS:
            v = cb_full.raw_dirs[ctrl_c][0].astype(np.float64)
            v = v - float(ub @ v) * ub
            unit_dirs[ctrl_c] = v / max(EPS, float(np.linalg.norm(v)))

        dirs64 = np.column_stack([unit_dirs[c] for c in CORE_CONCEPTS])
        aug = np.column_stack([ub, dirs64])
        q_aug, r_aug = np.linalg.qr(aug, mode="reduced")
        signs = np.where(np.diag(r_aug) >= 0.0, 1.0, -1.0)
        q_aug = q_aug * signs[None, :]
        q8 = q_aug[:, 1:]

        # Natural contrast step magnitude = mean L2 norm of `pos - neg` across the 8 core concepts
        contrast_norms = []
        for cname in CORE_CONCEPTS:
            idxs = [i for i, c in enumerate(concepts) if c.concept == cname]
            diffs = p_arr[idxs].astype(np.float64) - n_arr[idxs].astype(np.float64)
            contrast_norms.append(float(np.mean(np.linalg.norm(diffs, axis=-1))))
        nat_step = float(np.mean(contrast_norms))
        return cb_affect, cb_full, unit_dirs, q8, nat_step

    msg_aff, msg_cb, msg_units, q_msg8, nat_msg_step = _extract_8d_basis(pos_msg, neg_msg, v_bos_msg)
    _fin_aff, fin_cb, fin_units, q_fin8, nat_fin_step = _extract_8d_basis(pos_fin, neg_fin, v_bos_fin)

    # Per-channel LOPO accuracy across `Q, K, V, MLP` on Pass 2
    ch_names = ("query", "key", "value", "mlp")
    per_channel_lopo: dict[str, dict[str, float]] = {}
    per_channel_units: dict[str, dict[str, np.ndarray]] = {}
    for ch_i, ch_lbl in enumerate(ch_names):
        p_ch = pos_st["msg_last_by_pass"][0, :, ch_i, :]
        n_ch = neg_st["msg_last_by_pass"][0, :, ch_i, :]
        pm = {c: p_ch[[i for i, x in enumerate(concepts) if x.concept == c], None, :].astype(np.float64) for c in ALL_CONCEPTS}
        nm = {c: n_ch[[i for i, x in enumerate(concepts) if x.concept == c], None, :].astype(np.float64) for c in ALL_CONCEPTS}
        cb_ch = extract_controlled_basis(pm, nm, concepts=ALL_CONCEPTS, controls=CONTROL_CONCEPTS)
        per_channel_lopo[ch_lbl] = {
            c: round(float(cb_ch.lopo_accuracy[c][0]), 4) for c in CORE_CONCEPTS
        }
        per_channel_units[ch_lbl] = {
            c: cb_ch.controlled_dirs[c][0].astype(np.float64) for c in CORE_CONCEPTS
        }

    # Audit zero-FLOP 8D weight fold leakage and restore
    fold_diag = controller.apply_zero_flop_gate_fold(q_msg8, channels=(0, 1, 2, 3), fold_lm_head_q=q_fin8)
    controller.reset_interventions()

    concept_diag: dict[str, Any] = {}
    for cname in CORE_CONCEPTS:
        r_aff = min(1.0, max(0.0, float(msg_aff.residual_ratio[cname][0])))
        concept_diag[cname] = {
            "msg_lopo_acc": round(float(msg_cb.lopo_accuracy[cname][0]), 4),
            "msg_lopo_cosine": round(float(msg_cb.lopo_cosine[cname][0]), 4),
            "final_lopo_acc": round(float(fin_cb.lopo_accuracy[cname][0]), 4),
            "affect_retained_norm": round(r_aff, 4),
            "angle_to_affect_deg": round(math.degrees(math.asin(r_aff)), 2),
        }

    # 3. 6x8 Clinical Financial Cartography (`96` pairs, 0 emotional words) + Behavioral Choices across `K`
    cartography_by_arch: dict[str, Any] = {}
    behavioral_by_k: dict[str, Any] = {f"k{k_p}": {"unclamped": {}, "msg_homo_economicus_8d": {}} for k_p in pass_depths}

    for arch in CORE_ARCHETYPES:
        # CRITICAL: Always reset interventions before collecting unmasked reading representations!
        controller.reset_interventions()
        sub = [t for t in transactions if t.archetype == arch]
        st_a = collect_pass_states_and_messages(
            controller,
            [build_scenario_reading_prompt(t.archetype_scenario) for t in sub],
            n_passes=eff_probe_p,
            batch_size=batch_size,
        )
        st_c = collect_pass_states_and_messages(
            controller,
            [build_scenario_reading_prompt(t.control_scenario) for t in sub],
            n_passes=eff_probe_p,
            batch_size=batch_size,
        )
        delta_msg = (st_a["msg_last_by_pass"].mean(axis=(0, 2)) - st_c["msg_last_by_pass"].mean(axis=(0, 2))).astype(np.float64)
        delta_fin = (st_a["final_last"] - st_c["final_last"]).astype(np.float64)

        ch_top_betas: dict[str, dict[str, float]] = {}
        for ch_i, ch_lbl in enumerate(ch_names):
            d_ch = (st_a["msg_last_by_pass"][0, :, ch_i, :] - st_c["msg_last_by_pass"][0, :, ch_i, :]).astype(np.float64)
            c_proj = _project_multivariate_cartography(d_ch, per_channel_units[ch_lbl], msg_rms)
            ch_top_betas[ch_lbl] = {
                c: c_proj["concepts"][c]["partial_beta"] for c in CORE_CONCEPTS
            }

        cartography_by_arch[arch] = {
            "n_fixtures": len(sub),
            "message_bank_projection": _project_multivariate_cartography(delta_msg, msg_units, msg_rms),
            "final_norm_projection": _project_multivariate_cartography(delta_fin, fin_units, fin_rms),
            "per_channel_partial_beta": ch_top_betas,
        }

        # Score behavioral anomaly (`arch_lo - ctrl_lo`) across `K` under `unclamped` & `msg_homo_economicus_8d`
        scens_a = [t.archetype_scenario for t in sub]
        scens_c = [t.control_scenario for t in sub]
        devs = [t.deviation_option for t in sub]
        rats = [t.rational_option for t in sub]

        for cond_key, use_clamp in (("unclamped", False), ("msg_homo_economicus_8d", True)):
            controller.reset_interventions()
            if use_clamp:
                controller.active_msg_q = torch.tensor(q_msg8, dtype=torch.float32, device=controller.device)
                controller.active_msg_channels = (0, 1, 2, 3)
                controller.active_final_q = torch.tensor(q_fin8, dtype=torch.float32, device=controller.device)
            for k_p in pass_depths:
                lo_a = controller.score_counterbalanced_choices(scens_a, devs, rats, n_passes=k_p, batch_size=batch_size)
                lo_c = controller.score_counterbalanced_choices(scens_c, devs, rats, n_passes=k_p, batch_size=batch_size)
                deltas = lo_a - lo_c
                cl, ch = bootstrap_mean_ci(deltas)
                behavioral_by_k[f"k{k_p}"][cond_key][arch] = {
                    "mean_delta_lo": round(float(np.mean(deltas)), 4),
                    "ci95": [round(cl, 4), round(ch, 4)],
                    "mean_arch_lo": round(float(np.mean(lo_a)), 4),
                    "mean_ctrl_lo": round(float(np.mean(lo_c)), 4),
                    "p_signflip": round(exact_or_mc_signflip_p(deltas), 6),
                }
    controller.reset_interventions()

    # Holm-adjust across the 6 archetypes within each (k, cond)
    for k_key, cond_dict in behavioral_by_k.items():
        for cond_key, arch_dict in cond_dict.items():
            ps = [arch_dict[a]["p_signflip"] for a in CORE_ARCHETYPES]
            ph = holm_adjust(ps)
            for i, a in enumerate(CORE_ARCHETYPES):
                arch_dict[a]["p_holm"] = round(float(ph[i]), 6)

    basis_state = {
        "q_msg8": q_msg8,
        "q_fin8": q_fin8,
        "msg_units": msg_units,
        "fin_units": fin_units,
        "nat_msg_step": nat_msg_step,
        "nat_fin_step": nat_fin_step,
        "msg_rms": msg_rms,
        "fin_rms": fin_rms,
    }
    summary = {
        "d_model": controller.d_model,
        "rank_removed": 8,
        "removed_pct": round(100.0 * 8.0 / controller.d_model, 4),
        "retained_pct": round(100.0 * (controller.d_model - 8.0) / controller.d_model, 4),
        "bos_sink_orthogonality": {
            "msg_8d_bos_overlap": float(np.linalg.norm(q_msg8.T @ v_bos_msg)),
            "fin_8d_bos_overlap": float(np.linalg.norm(q_fin8.T @ v_bos_fin)),
        },
        "zero_flop_8d_fold_audit": fold_diag,
        "sanity_gate_by_k": sanity_by_k,
        "concept_diagnostics": concept_diag,
        "per_channel_lopo_acc": per_channel_lopo,
        "financial_cartography": cartography_by_arch,
        "behavioral_anomalies_by_k": behavioral_by_k,
    }
    return summary, basis_state


# ---------------------------------------------------------------------------
# Block 2: Formal Afriat / GARP Revealed-Preference Evaluation
# ---------------------------------------------------------------------------

def _activate_afriat_condition(
    controller: AfriatInklingController,
    cond_name: str,
    basis_state: dict[str, Any],
) -> None:
    controller.reset_interventions()
    q_m = basis_state["q_msg8"]
    q_f = basis_state["q_fin8"]
    if cond_name == "unclamped":
        return
    if cond_name == "msg_homo_economicus_8d":
        controller.active_msg_q = torch.tensor(q_m, dtype=torch.float32, device=controller.device)
        controller.active_msg_channels = (0, 1, 2, 3)
        controller.active_final_q = torch.tensor(q_f, dtype=torch.float32, device=controller.device)
        return
    if cond_name == "msg_qkv_8d":
        controller.active_msg_q = torch.tensor(q_m, dtype=torch.float32, device=controller.device)
        controller.active_msg_channels = (0, 1, 2)
        return
    if cond_name == "msg_mlp_8d":
        controller.active_msg_q = torch.tensor(q_m, dtype=torch.float32, device=controller.device)
        controller.active_msg_channels = (3,)
        return
    if cond_name == "zero_flop_8d":
        controller.apply_zero_flop_gate_fold(q_m, channels=(0, 1, 2, 3), fold_lm_head_q=q_f)
        return
    raise ValueError(f"Unknown condition: {cond_name}")


def _summarize_regime_garp(
    items: Sequence[GARPBudgetItem],
    menu_scores: dict[str, np.ndarray],
    random_sample_cceis: np.ndarray,
) -> dict[str, Any]:
    """Audit exact GARP, Afriat CCEI, utility recovery, law of demand, and Homo Economicus payoff."""
    subjects = sorted({it.subject_id for it in items})
    modal_idx = menu_scores["modal_idx"]
    modal_share = menu_scores["modal_share"]
    exp_share = menu_scores["expected_share"]
    letter_mass = menu_scores["letter_mass"]

    subj_modal_ccei: list[float] = []
    subj_exp_ccei: list[float] = []
    subj_garp_pass: list[bool] = []
    subj_garp_viols: list[int] = []
    subj_warp_viols: list[int] = []
    subj_max_u_viol: list[float] = []
    subj_he_eff: list[float] = []
    subj_good1_share: list[float] = []
    subj_law_demand: list[float] = []

    sample_afriat_u: list[float] = []
    sample_afriat_lam: list[float] = []

    for s_idx, sid in enumerate(subjects):
        mask = [i for i, it in enumerate(items) if it.subject_id == sid]
        mask.sort(key=lambda i: items[i].obs_idx)
        sub_items = [items[i] for i in mask]

        prices = np.array([[it.price1, it.price2] for it in sub_items], dtype=np.float64)
        m_sh = modal_share[mask]
        e_sh = exp_share[mask]
        m_bundles = np.array(
            [[sh * it.max_good1, (1.0 - sh) * it.max_good2] for sh, it in zip(m_sh, sub_items)],
            dtype=np.float64,
        )
        e_bundles = np.array(
            [[sh * it.max_good1, (1.0 - sh) * it.max_good2] for sh, it in zip(e_sh, sub_items)],
            dtype=np.float64,
        )

        res_m = audit_revealed_preferences(prices, m_bundles)
        res_e = audit_revealed_preferences(prices, e_bundles)

        subj_modal_ccei.append(res_m.afriat_ccei)
        subj_exp_ccei.append(res_e.afriat_ccei)
        subj_garp_pass.append(res_m.passes_garp)
        subj_garp_viols.append(res_m.garp_violation_pairs)
        subj_warp_viols.append(res_m.warp_violation_pairs)
        subj_max_u_viol.append(res_m.max_afriat_inequality_violation)
        if s_idx == 0:
            sample_afriat_u = res_m.afriat_utility_u
            sample_afriat_lam = res_m.afriat_multiplier_lambda

        # Homo Economicus monetary efficiency:
        # On social regimes, Homo Economicus maximizes `x_1` (`own cash` vs `M_1`).
        # On provenance/decoy regimes, Homo Economicus maximizes `x_1 + x_2` vs `max(M_1, M_2)`.
        he_ratios: list[float] = []
        for sh, it in zip(e_sh, sub_items):
            x1 = sh * it.max_good1
            x2 = (1.0 - sh) * it.max_good2
            if it.regime.startswith("social"):
                he_ratios.append(x1 / max(EPS, it.max_good1))
            else:
                he_ratios.append((x1 + x2) / max(EPS, max(it.max_good1, it.max_good2)))
        subj_he_eff.append(float(np.mean(he_ratios)))
        subj_good1_share.append(float(np.mean(e_sh)))

        # Law of demand check across pairs with distinct relative prices `p1/p2`:
        # When `p1/p2` rises, expenditure share / quantity ratio `x1 / (x1 + x2)` should not rise (`<= +0.02` tol)
        log_p = np.log(np.array([it.price_ratio_p1_p2 for it in sub_items], dtype=np.float64))
        lod_ok = 0
        lod_tot = 0
        for a in range(len(sub_items)):
            for b in range(a + 1, len(sub_items)):
                if abs(log_p[a] - log_p[b]) > 0.15:
                    lod_tot += 1
                    dp = log_p[b] - log_p[a]
                    dsh = e_sh[b] - e_sh[a]
                    if dp * dsh <= 0.02 * abs(dp):
                        lod_ok += 1
        subj_law_demand.append(float(lod_ok / max(1, lod_tot)))

    mean_m_ccei = float(np.mean(subj_modal_ccei))
    p_vs_rand = float(np.mean(random_sample_cceis >= mean_m_ccei - 1e-12))

    return {
        "n_subjects": len(subjects),
        "n_obs_per_subject": 24,
        "modal_ccei_mean": round(mean_m_ccei, 4),
        "modal_ccei_per_subject": [round(v, 4) for v in subj_modal_ccei],
        "expected_bundle_ccei_mean": round(float(np.mean(subj_exp_ccei)), 4),
        "expected_ccei_per_subject": [round(v, 4) for v in subj_exp_ccei],
        "frac_subjects_exact_garp": round(float(np.mean(subj_garp_pass)), 4),
        "mean_garp_violation_pairs": round(float(np.mean(subj_garp_viols)), 2),
        "mean_warp_violation_pairs": round(float(np.mean(subj_warp_viols)), 2),
        "max_afriat_inequality_violation": float(np.max(subj_max_u_viol)),
        "p_vs_bronars_random": round(p_vs_rand, 6),
        "homo_economicus_efficiency": round(float(np.mean(subj_he_eff)), 4),
        "he_efficiency_per_subject": [round(v, 4) for v in subj_he_eff],
        "mean_good1_share": round(float(np.mean(subj_good1_share)), 4),
        "good1_share_per_subject": [round(v, 4) for v in subj_good1_share],
        "law_of_demand_consistency": round(float(np.mean(subj_law_demand)), 4),
        "mean_menu_letter_mass": round(float(np.mean(letter_mass)), 4),
        "subject0_afriat_u": sample_afriat_u,
        "subject0_afriat_lambda": sample_afriat_lam,
    }


def run_block2_afriat_garp(
    controller: AfriatInklingController,
    basis_state: dict[str, Any],
    eval_cells: Sequence[tuple[str, int]],
    batch_size: int = 32,
) -> dict[str, Any]:
    """Run formal Afriat / GARP evaluation across all `(condition, K)` cells."""
    suites = build_garp_budget_suites()
    bronars = compute_bronars_power(
        np.asarray(BASE_INTERSECTING_INTERCEPTS_24, dtype=np.float64),
        grid_shares=GRID_SHARES_11,
        n_simulations=2000,
        seed=20261121,
    )
    rand_cceis = bronars.pop("sample_cceis")

    cells_out: dict[str, dict[str, Any]] = {}
    for cond_name, k_p in eval_cells:
        cell_id = f"{cond_name}__k{k_p}"
        print(f"    [Block 2 GARP] {controller.model_label} | {cond_name} | K={k_p}...", flush=True)
        _activate_afriat_condition(controller, cond_name, basis_state)
        reg_out: dict[str, Any] = {}
        for reg_name, items in suites.items():
            m_scores = controller.score_counterbalanced_11way_menus(items, n_passes=k_p, batch_size=batch_size)
            reg_out[reg_name] = _summarize_regime_garp(items, m_scores, rand_cceis)
        cells_out[cell_id] = {
            "condition": cond_name,
            "n_passes": int(k_p),
            "regimes": reg_out,
        }
    controller.reset_interventions()
    return {
        "bronars_random_benchmark": bronars,
        "cells": cells_out,
    }


# ---------------------------------------------------------------------------
# Block 3: Disparity, Blount (1995) Agency DiD, & Message-Bank Steering
# ---------------------------------------------------------------------------

def run_block3_inequity_and_steering(
    controller: AfriatInklingController,
    basis_state: dict[str, Any],
    pass_depths: Sequence[int],
    steer_passes: int = 3,
    batch_size: int = 32,
    stimuli_dir: Path | None = None,
) -> dict[str, Any]:
    """Evaluate Ultimatum curve, Blount Human-vs.-Computer DiD across `K`, and message-bank steering."""
    controller.reset_interventions()
    ineq = build_inequity_suite()
    txs = load_transactions(stimuli_dir or default_stimuli_dir())
    tainted_sub = [t for t in txs if t.archetype == "tainted_provenance"]

    by_k: dict[str, Any] = {}
    for k_p in pass_depths:
        scores_k: dict[str, np.ndarray] = {}
        for cell_key, items in ineq.items():
            scores_k[cell_key] = controller.score_counterbalanced_choices(
                scenarios=[it.scenario for it in items],
                target_options=[it.deviation_option for it in items],
                reference_options=[it.rational_option for it in items],
                n_passes=k_p,
                batch_size=batch_size,
            )

        curve_rows: list[dict[str, Any]] = []
        for pm in CURVE_PERMILLE:
            deltas = scores_k[f"human_{pm}"] - scores_k["human_500"]
            cl, ch = bootstrap_mean_ci(deltas)
            curve_rows.append(
                {
                    "you_permille": pm,
                    "mean_reject_lo": round(float(np.mean(scores_k[f"human_{pm}"])), 4),
                    "reject_rate": round(float(np.mean(scores_k[f"human_{pm}"] > 0.0)), 4),
                    "delta_vs_50_50_lo": round(float(np.mean(deltas)), 4),
                    "ci95": [round(cl, 4), round(ch, 4)],
                }
            )

        # Blount (1995) Intentionality DiD: `(human_pm - human_500) - (random_pm - random_500)`
        blount_did: dict[str, Any] = {}
        for tag, pm in (("98_2", 20), ("999_1", 1)):
            h_gap = scores_k[f"human_{pm}"] - scores_k["human_500"]
            r_gap = scores_k[f"random_{pm}"] - scores_k["random_500"]
            did = h_gap - r_gap
            cl, ch = bootstrap_mean_ci(did)
            blount_did[tag] = {
                "human_gap_lo": round(float(np.mean(h_gap)), 4),
                "computer_random_gap_lo": round(float(np.mean(r_gap)), 4),
                "did_human_minus_computer_lo": round(float(np.mean(did)), 4),
                "ci95": [round(cl, 4), round(ch, 4)],
                "p_signflip": round(exact_or_mc_signflip_p(did), 6),
                "n_human_higher_of_16": int(np.sum(did > 0.0)),
            }

        dict_gap = scores_k["dictator_980"] - scores_k["dictator_500"]
        by_k[f"k{k_p}"] = {
            "disparity_curve": curve_rows,
            "blount_intentionality_did": blount_did,
            "dictator_giveaway_98_2_gap_lo": round(float(np.mean(dict_gap)), 4),
            "dictator_giveaway_rate": round(float(np.mean(scores_k["dictator_980"] > 0.0)), 4),
        }

    # Bidirectional Cross-Pass Message-Bank Steering Dose-Response (`alpha in {-2, -1, 0, +1, +2}`)
    alphas = (-2.0, -1.0, 0.0, 1.0, 2.0)
    unit = float(basis_state["nat_msg_step"])
    steer_vecs = ("spite", "guilt", "captivity", "contamination", "valence")
    eff_sp = max(2, steer_passes)

    steering_out: dict[str, dict[str, list[dict[str, Any]]]] = {
        "ultimatum_98_2": {},
        "dictator_2_98": {},
        "tainted_provenance": {},
    }

    target_specs = [
        (
            "ultimatum_98_2",
            [it.scenario for it in ineq["human_20"]],
            [it.deviation_option for it in ineq["human_20"]],
            [it.rational_option for it in ineq["human_20"]],
        ),
        (
            "dictator_2_98",
            [it.scenario for it in ineq["dictator_980"]],
            [it.deviation_option for it in ineq["dictator_980"]],
            [it.rational_option for it in ineq["dictator_980"]],
        ),
        (
            "tainted_provenance",
            [t.archetype_scenario for t in tainted_sub],
            [t.deviation_option for t in tainted_sub],
            [t.rational_option for t in tainted_sub],
        ),
    ]

    for t_name, scens, devs, rats in target_specs:
        base_lo = controller.score_counterbalanced_choices(scens, devs, rats, n_passes=eff_sp, batch_size=batch_size)
        for v_name in steer_vecs:
            vec = basis_state["msg_units"][v_name]
            rows_v: list[dict[str, Any]] = []
            for al in alphas:
                if abs(al) < 1e-12:
                    cur_lo = base_lo
                else:
                    with controller.message_steering(vec, delta_scale=float(al * unit)):
                        cur_lo = controller.score_counterbalanced_choices(scens, devs, rats, n_passes=eff_sp, batch_size=batch_size)
                shift = cur_lo - base_lo
                rows_v.append(
                    {
                        "alpha": float(al),
                        "mean_lo": round(float(np.mean(cur_lo)), 4),
                        "deviate_rate": round(float(np.mean(cur_lo > 0.0)), 4),
                        "shift_vs_zero_lo": round(float(np.mean(shift)), 4),
                        "p_signflip": 1.0 if abs(al) < 1e-12 else round(exact_or_mc_signflip_p(shift), 6),
                    }
                )
            steering_out[t_name][v_name] = rows_v

    controller.reset_interventions()
    return {
        "by_k": by_k,
        "message_bank_steering_k3": steering_out,
    }


# ---------------------------------------------------------------------------
# Block 4: Continuous Dollar Reservation Curves (Turnbull WTP)
# ---------------------------------------------------------------------------

def _summarize_wtp_curve(
    items: Sequence[WtpCurveItem],
    lo_scores: np.ndarray,
    c_floor: float = 0.0,
) -> dict[str, Any]:
    rungs = sorted({it.rung_idx for it in items})
    fixtures = sorted({it.fixture_id for it in items})
    f_map = {f: i for i, f in enumerate(fixtures)}
    n_f, n_r = len(fixtures), len(rungs)

    lo_mat = np.zeros((n_f, n_r), dtype=np.float64)
    prob_mat = np.zeros((n_f, n_r), dtype=np.float64)
    ref_costs = np.zeros(n_r, dtype=np.float64)

    for it, lo_val in zip(items, lo_scores):
        i = f_map[it.fixture_id]
        r = it.rung_idx
        lo_mat[i, r] = float(lo_val)
        prob_mat[i, r] = float(1.0 / (1.0 + math.exp(-max(-40.0, min(40.0, float(lo_val))))))
        ref_costs[r] = float(it.ref_cost_usd)

    wtp_per_fix = np.zeros(n_f, dtype=np.float64)
    mono_ok = 0
    mono_tot = n_f * (n_r - 1)
    for i in range(n_f):
        p = prob_mat[i]
        lead = max(0.0, ref_costs[0] - c_floor) * p[0]
        trap = float(np.trapezoid(p, ref_costs)) if hasattr(np, "trapezoid") else float(np.trapz(p, ref_costs))
        wtp_per_fix[i] = float(c_floor + lead + trap)
        for k in range(n_r - 1):
            if lo_mat[i, k + 1] <= lo_mat[i, k] + 0.10:
                mono_ok += 1

    cl, ch = bootstrap_mean_ci(wtp_per_fix)
    return {
        "expected_wtp_usd": round(float(np.mean(wtp_per_fix)), 2),
        "ci95_usd": [round(cl, 2), round(ch, 2)],
        "per_fixture_wtp_usd": [round(float(v), 2) for v in wtp_per_fix],
        "garp_1d_monotonicity": round(float(mono_ok / max(1, mono_tot)), 4),
        "mean_log_odds_by_rung": [round(float(v), 4) for v in np.mean(lo_mat, axis=0)],
    }


def run_block4_wtp_curves(
    controller: AfriatInklingController,
    basis_state: dict[str, Any],
    wtp_k_depths: Sequence[int],
    steer_passes: int = 3,
    batch_size: int = 32,
) -> dict[str, Any]:
    """Evaluate 6 continuous Dollar WTP curves + `$100` Pocket Control across `K`, clamping, and steering."""
    curves = build_wtp_curves()
    c_floors = {
        "ultimatum_offer": 0.0,
        "communal_favor": 0.0,
        "crisis_surge": 0.0,
        "zero_apr_debt": 0.0,
        "tainted_surplus": 0.0,
        "institutional_certainty": -5.0,
        "pocket_certainty_control": -5.0,
    }

    out_cells: dict[str, Any] = {}
    for cond_name in ("unclamped", "msg_homo_economicus_8d"):
        for k_p in wtp_k_depths:
            if cond_name == "msg_homo_economicus_8d" and k_p == 1 and len(wtp_k_depths) > 1:
                continue
            _activate_afriat_condition(controller, cond_name, basis_state)
            cell_res: dict[str, Any] = {}
            for arch_k, items in curves.items():
                lo_s = controller.score_counterbalanced_choices(
                    scenarios=[it.scenario for it in items],
                    target_options=[it.deviation_option for it in items],
                    reference_options=[it.rational_option for it in items],
                    n_passes=k_p,
                    batch_size=batch_size,
                )
                cell_res[arch_k] = _summarize_wtp_curve(items, lo_s, c_floor=c_floors[arch_k])
            out_cells[f"{cond_name}__k{k_p}"] = cell_res

    # Bidirectional message-bank dollar WTP swings at `steer_passes` (`alpha in {-2.0, +2.0}`)
    controller.reset_interventions()
    eff_sp = max(2, steer_passes)
    unit = float(basis_state["nat_msg_step"])
    steered_wtp: dict[str, Any] = {}
    steer_pairs = (
        ("ultimatum_offer", "spite"),
        ("tainted_surplus", "contamination"),
        ("communal_favor", "guilt"),
        ("institutional_certainty", "awe_dread"),
    )
    for arch_k, vec_name in steer_pairs:
        items = curves[arch_k]
        vec = basis_state["msg_units"][vec_name]
        steered_wtp[arch_k] = {"vector": vec_name}
        for al_lbl, al_val in (("minus_2", -2.0), ("plus_2", 2.0)):
            with controller.message_steering(vec, delta_scale=float(al_val * unit)):
                lo_s = controller.score_counterbalanced_choices(
                    scenarios=[it.scenario for it in items],
                    target_options=[it.deviation_option for it in items],
                    reference_options=[it.rational_option for it in items],
                    n_passes=eff_sp,
                    batch_size=batch_size,
                )
            steered_wtp[arch_k][al_lbl] = _summarize_wtp_curve(items, lo_s, c_floor=c_floors[arch_k])

    controller.reset_interventions()
    return {
        "by_condition_and_k": out_cells,
        "message_bank_steered_wtp": steered_wtp,
    }


# ---------------------------------------------------------------------------
# Block 5: Silent `0-CoT` vs. `<|content_thinking|>` Rationalization + Debt Math
# ---------------------------------------------------------------------------

def _parse_open_record(rec: dict[str, Any], item: OpenDecisionItem) -> dict[str, Any]:
    pub = rec["public_text"]
    tht = rec["thought_text"]

    m = DECISION_USD_RE.search(pub)
    if m is None:
        first_line = pub.splitlines()[0] if pub else ""
        m = LEADING_NUM_RE.match(first_line) or ANY_DOLLAR_RE.search(pub)
    parsed_ok = m is not None
    val_raw = float(m.group(1).replace(",", "")) if m is not None else 0.5 * (item.actual_min_usd + item.actual_max_usd)
    clamped = float(np.clip(val_raw, float(item.actual_min_usd), float(item.actual_max_usd)))
    ref_usd = float(clamped / item.scale_factor)

    parts = RATIONALE_SPLIT_RE.split(pub, maxsplit=1)
    rationale = parts[1].strip() if len(parts) == 2 else " ".join(pub.splitlines()[1:]).strip()
    rationale = rationale.split("\nDECISION")[0].strip()

    pub_words = set(tokenize_words(rationale))
    tht_words = set(tokenize_words(tht))

    return {
        "fixture_id": item.fixture_id,
        "parsed_ok": parsed_ok,
        "ref_usd": round(ref_usd, 2),
        "public_strict_affect": len(pub_words & BANNED_AFFECTIVE_TERMS) > 0,
        "public_broad_moral": len(pub_words & BROAD_MORAL_LEXICON) > 0,
        "public_cold_jargon": len(pub_words & RATIONALIZATION_JARGON) > 0,
        "thought_present": bool(rec["thought_present"]),
        "thought_strict_affect": len(tht_words & BANNED_AFFECTIVE_TERMS) > 0,
        "thought_broad_moral": len(tht_words & BROAD_MORAL_LEXICON) > 0,
        "n_gen_tokens": int(rec["n_gen_tokens"]),
        "sample_thought_excerpt": tht[:220],
        "sample_public_rationale": rationale[:220],
    }


def _summarize_open_task(parsed_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usd = np.array([r["ref_usd"] for r in parsed_rows], dtype=np.float64)
    cl, ch = bootstrap_mean_ci(usd)
    return {
        "n": len(parsed_rows),
        "parse_rate": round(float(np.mean([r["parsed_ok"] for r in parsed_rows])), 4),
        "mean_ref_usd": round(float(np.mean(usd)), 2),
        "ci95_usd": [round(cl, 2), round(ch, 2)],
        "public_zero_strict_affect_rate": round(float(np.mean([not r["public_strict_affect"] for r in parsed_rows])), 4),
        "public_broad_moral_rate": round(float(np.mean([r["public_broad_moral"] for r in parsed_rows])), 4),
        "public_cold_jargon_rate": round(float(np.mean([r["public_cold_jargon"] for r in parsed_rows])), 4),
        "thought_present_rate": round(float(np.mean([r["thought_present"] for r in parsed_rows])), 4),
        "thought_strict_affect_rate": round(float(np.mean([r["thought_strict_affect"] for r in parsed_rows])), 4),
        "thought_broad_moral_rate": round(float(np.mean([r["thought_broad_moral"] for r in parsed_rows])), 4),
        "mean_gen_tokens": round(float(np.mean([r["n_gen_tokens"] for r in parsed_rows])), 1),
        "example_thought": parsed_rows[0]["sample_thought_excerpt"] if parsed_rows else "",
        "example_rationale": parsed_rows[0]["sample_public_rationale"] if parsed_rows else "",
    }


def run_block5_rationalization_and_debt(
    controller: AfriatInklingController,
    basis_state: dict[str, Any],
    silent_k_depths: Sequence[int],
    think_passes: int = 3,
    batch_size: int = 16,
) -> dict[str, Any]:
    """Compare Silent `0-CoT` Loops vs. Private `<|content_thinking|>` Scratchpad + `0%`-APR Debt Math."""
    controller.reset_interventions()
    open_items = build_open_decision_suite()
    debt_items = build_debt_math_suite()
    tasks = ("open_ultimatum_min", "open_tainted_keep", "open_communal_quote")
    task_vec = {
        "open_ultimatum_min": "spite",
        "open_tainted_keep": "contamination",
        "open_communal_quote": "guilt",
    }

    open_runs: dict[str, Any] = {}

    # 1. Unsteered Silent `0-CoT` across `silent_k_depths`
    for k_p in silent_k_depths:
        recs = controller.generate_completions(
            [it.user_prompt for it in open_items],
            n_passes=k_p,
            reasoning_effort="none",
            assistant_prefix="DECISION_USD: $",
            max_new_tokens=96,
            batch_size=batch_size,
        )
        by_t: dict[str, Any] = {}
        for t_id in tasks:
            sub_parsed = [_parse_open_record(r, it) for r, it in zip(recs, open_items) if it.task_id == t_id]
            by_t[t_id] = _summarize_open_task(sub_parsed)
        open_runs[f"silent_0cot__k{k_p}"] = by_t

    # 2. Message-Bank Steered Silent `0-CoT` (`alpha in {-2.0, +2.0}` at `think_passes`)
    eff_tp = max(2, think_passes)
    unit = float(basis_state["nat_msg_step"])
    for al_lbl, al_val in (("minus_2", -2.0), ("plus_2", 2.0)):
        by_t_s: dict[str, Any] = {}
        for t_id in tasks:
            t_sub = [it for it in open_items if it.task_id == t_id]
            vec = basis_state["msg_units"][task_vec[t_id]]
            with controller.message_steering(vec, delta_scale=float(al_val * unit)):
                recs_s = controller.generate_completions(
                    [it.user_prompt for it in t_sub],
                    n_passes=eff_tp,
                    reasoning_effort="none",
                    assistant_prefix="DECISION_USD: $",
                    max_new_tokens=96,
                    batch_size=batch_size,
                )
            by_t_s[t_id] = _summarize_open_task([_parse_open_record(r, it) for r, it in zip(recs_s, t_sub)])
        open_runs[f"silent_0cot_steered_{al_lbl}__k{eff_tp}"] = by_t_s

    # 3. Explicit `<|content_thinking|>` Scratchpad (`reasoning_effort="medium"` at `think_passes`)
    recs_th = controller.generate_completions(
        [it.user_prompt for it in open_items],
        n_passes=eff_tp,
        reasoning_effort="medium",
        rescue_prefix="DECISION_USD: $",
        max_new_tokens=384,
        batch_size=max(1, batch_size // 2),
    )
    by_t_th: dict[str, Any] = {}
    for t_id in tasks:
        sub_parsed = [_parse_open_record(r, it) for r, it in zip(recs_th, open_items) if it.task_id == t_id]
        by_t_th[t_id] = _summarize_open_task(sub_parsed)
    open_runs[f"vocal_thinking_medium__k{eff_tp}"] = by_t_th

    # 4. 32-Item `0%`-APR Debt Balance-Sheet vs. Cashflow Subtraction (`0-CoT` across K vs. `<|content_thinking|>`)
    debt_out: dict[str, Any] = {}
    for cond_name in ("unclamped", "msg_homo_economicus_8d"):
        for k_p in silent_k_depths:
            _activate_afriat_condition(controller, cond_name, basis_state)
            lo_d = controller.score_counterbalanced_choices(
                scenarios=[it.scenario for it in debt_items],
                target_options=[it.better_option for it in debt_items],
                reference_options=[it.worse_option for it in debt_items],
                n_passes=k_p,
                batch_size=batch_size * 2,
            )
            debt_mask = np.array([it.domain == "debt_nw" for it in debt_items])
            cash_mask = np.array([it.domain == "cashflow_twin" for it in debt_items])
            unenc_mask = np.array([it.domain == "debt_nw" and it.winner_arm == "unencumbered_wins" for it in debt_items])
            gross_mask = np.array([it.domain == "debt_nw" and it.winner_arm == "gross_minus_obligation_wins" for it in debt_items])
            corr = lo_d > 0.0
            debt_out[f"{cond_name}_0cot__k{k_p}"] = {
                "debt_nw_acc": round(float(np.mean(corr[debt_mask])), 4),
                "cashflow_twin_acc": round(float(np.mean(corr[cash_mask])), 4),
                "debt_when_unencumbered_wins_acc": round(float(np.mean(corr[unenc_mask])), 4),
                "debt_when_gross_minus_loan_wins_acc": round(float(np.mean(corr[gross_mask])), 4),
            }
    controller.reset_interventions()

    # Also test `<|content_thinking|>` explicit scratchpad on the 16 `debt_nw` items (A/B counterbalanced)!
    debt_nw_only = [it for it in debt_items if it.domain == "debt_nw"]
    d_prompts: list[str] = []
    d_expected: list[str] = []
    for idx_d, it in enumerate(debt_nw_only):
        if idx_d % 2 == 0:
            opt_a, opt_b, exp_let = it.better_option, it.worse_option, "A"
        else:
            opt_a, opt_b, exp_let = it.worse_option, it.better_option, "B"
        d_prompts.append(
            f"{it.scenario}\n\nOption A: {opt_a}\nOption B: {opt_b}\n\n"
            f"After your analysis, end your final answer with CHOICE: A or CHOICE: B."
        )
        d_expected.append(exp_let)
    d_recs = controller.generate_completions(
        d_prompts,
        n_passes=eff_tp,
        reasoning_effort="medium",
        rescue_prefix="CHOICE: ",
        max_new_tokens=320,
        batch_size=max(1, batch_size // 2),
    )
    choice_re = re.compile(r"CHOICE\s*:\s*([AB])\b", re.IGNORECASE)
    hits = 0
    for rec, exp_let in zip(d_recs, d_expected):
        pub_u = rec["public_text"].strip().upper()
        matches = choice_re.findall(pub_u)
        picked = matches[-1].upper() if matches else (pub_u[0] if pub_u[:1] in ("A", "B") else "")
        if picked == exp_let:
            hits += 1
    debt_out[f"vocal_thinking_medium__k{eff_tp}"] = {
        "debt_nw_acc": round(float(hits / max(1, len(debt_nw_only))), 4),
        "thought_present_rate": round(float(np.mean([r["thought_present"] for r in d_recs])), 4),
        "mean_gen_tokens": round(float(np.mean([r["n_gen_tokens"] for r in d_recs])), 1),
    }

    return {
        "open_pricing_and_rationalization": open_runs,
        "zero_apr_debt_vs_cashflow": debt_out,
    }


# ---------------------------------------------------------------------------
# End-to-End Model & Pipeline Runner
# ---------------------------------------------------------------------------

def _load_afriat_inkling_controller(model_dir: Path, model_label: str) -> tuple[AfriatInklingController, dict[str, Any]]:
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=torch.bfloat16,
        device_map="cuda:0",
    )
    model.eval()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ctrl = AfriatInklingController(model=model, tokenizer=tok, model_label=model_label)
    meta = {
        "model_id": model_label,
        "d_model": ctrl.d_model,
        "n_layers": ctrl.n_layers,
        "loop_start": ctrl.loop_start,
        "loop_end": ctrl.loop_end,
        "default_passes": ctrl.default_passes,
        "gate_rank": ctrl.gate_rank,
        "cuda_allocated_gb": round(float(torch.cuda.memory_allocated() / 1e9), 3) if torch.cuda.is_available() else 0.0,
        "load_seconds": round(time.perf_counter() - t0, 2),
    }
    return ctrl, meta


def run_afriat_inkling_pipeline(
    varloop_model_dir: Path,
    inkling_mini_dir: Path | None,
    varloop_label: str = "thinkingmachines/TML-d24-varloop-research",
    inkling_label: str = "thinkingmachines/Inkling-2-Mini",
    out_dir: Path = Path("runs/afriat_inkling_garp"),
    batch_size: int = 32,
    pass_depths: Sequence[int] = (1, 2, 3, 4, 5),
    resume_json: Path | None = None,
) -> dict[str, Any]:
    """Execute the complete 5-block Afriat / GARP + Money Is Moral study on Varloop & Inkling-2-Mini."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint_afriat_inkling_results.json"
    final_path = out_dir / "afriat_inkling_results.json"

    stim_audit = audit_all_afriat_inkling_stimuli()
    if not stim_audit["valid"]:
        raise RuntimeError(f"Stimulus firewall audit failed: {stim_audit['errors']}")

    payload: dict[str, Any] = {
        "phase": "phase5_inkling_afriat_garp",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stimulus_audit": stim_audit,
        "completed": False,
    }
    if resume_json is not None and resume_json.is_file():
        payload = json.loads(resume_json.read_text(encoding="utf-8"))
        payload["completed"] = False

    # -----------------------------------------------------------------------
    # Part 1: TML-d24-varloop-research (`K = 1, 2, 3, 4, 5`)
    # -----------------------------------------------------------------------
    if not payload.get("varloop_research", {}).get("completed", False):
        print(f"[Afriat-Inkling] Loading {varloop_label} from {varloop_model_dir}...", flush=True)
        ctrl_v, meta_v = _load_afriat_inkling_controller(varloop_model_dir, varloop_label)
        v_sec: dict[str, Any] = payload.setdefault("varloop_research", {})
        v_sec["metadata"] = meta_v
        v_sec["pass_depths"] = list(pass_depths)

        print("[Afriat-Inkling][Varloop] Block 1: 8-axis non-financial moral basis + 6x8 Moral Atlas...", flush=True)
        b1_v, basis_v = run_block1_moral_atlas(
            ctrl_v, pass_depths=pass_depths, probe_passes=3, batch_size=batch_size
        )
        v_sec["block1_moral_atlas"] = b1_v
        _atomic_save_json(ckpt_path, payload)

        if "block2_afriat_garp" not in v_sec:
            print("[Afriat-Inkling][Varloop] Block 2: Intersecting T=24 Afriat / GARP budget regimes...", flush=True)
            garp_cells_v: list[tuple[str, int]] = [("unclamped", k) for k in pass_depths]
            for k_sub in [k for k in (3, 5) if k in pass_depths]:
                for c_sub in ("msg_homo_economicus_8d", "msg_qkv_8d", "msg_mlp_8d", "zero_flop_8d"):
                    garp_cells_v.append((c_sub, k_sub))
            v_sec["block2_afriat_garp"] = run_block2_afriat_garp(
                ctrl_v, basis_v, eval_cells=garp_cells_v, batch_size=batch_size
            )
            _atomic_save_json(ckpt_path, payload)

        if "block3_inequity_and_steering" not in v_sec:
            print("[Afriat-Inkling][Varloop] Block 3: Disparity, Blount agency DiD, & message-bank steering...", flush=True)
            ineq_ks = [k for k in (1, 2, 3, 5) if k in pass_depths] or list(pass_depths)
            v_sec["block3_inequity_and_steering"] = run_block3_inequity_and_steering(
                ctrl_v, basis_v, pass_depths=ineq_ks, steer_passes=3, batch_size=batch_size
            )
            _atomic_save_json(ckpt_path, payload)

        if "block4_dollar_wtp_curves" not in v_sec:
            print("[Afriat-Inkling][Varloop] Block 4: 6 Continuous Dollar WTP reservation curves...", flush=True)
            wtp_ks = [k for k in (1, 2, 3, 5) if k in pass_depths] or list(pass_depths)
            v_sec["block4_dollar_wtp_curves"] = run_block4_wtp_curves(
                ctrl_v, basis_v, wtp_k_depths=wtp_ks, steer_passes=3, batch_size=batch_size
            )
            _atomic_save_json(ckpt_path, payload)

        if "block5_rationalization_and_debt" not in v_sec:
            print("[Afriat-Inkling][Varloop] Block 5: Silent 0-CoT vs. <|content_thinking|> rationalization + Debt math...", flush=True)
            rat_ks = [k for k in (1, 3, 5) if k in pass_depths] or [pass_depths[0]]
            v_sec["block5_rationalization_and_debt"] = run_block5_rationalization_and_debt(
                ctrl_v, basis_v, silent_k_depths=rat_ks, think_passes=3, batch_size=max(8, batch_size // 2)
            )
            _atomic_save_json(ckpt_path, payload)

        v_sec["completed"] = True
        _atomic_save_json(ckpt_path, payload)

        ctrl_v.reset_interventions()
        ctrl_v.model.to("cpu")
        del ctrl_v
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Part 2: Flagship Inkling-2-Mini (`K = 3` + `<|content_thinking|>`)
    # -----------------------------------------------------------------------
    if inkling_mini_dir is not None and inkling_mini_dir.exists() and not payload.get("inkling_2_mini", {}).get("completed", False):
        print(f"[Afriat-Inkling] Loading flagship {inkling_label} from {inkling_mini_dir}...", flush=True)
        ctrl_i, meta_i = _load_afriat_inkling_controller(inkling_mini_dir, inkling_label)
        i_sec: dict[str, Any] = payload.setdefault("inkling_2_mini", {})
        i_sec["metadata"] = meta_i

        print("[Afriat-Inkling][Inkling-2-Mini] Block 1: 8-axis moral basis + 6x8 Moral Atlas (K=3)...", flush=True)
        b1_i, basis_i = run_block1_moral_atlas(
            ctrl_i, pass_depths=(3,), probe_passes=3, batch_size=batch_size
        )
        i_sec["block1_moral_atlas"] = b1_i
        _atomic_save_json(ckpt_path, payload)

        if "block2_afriat_garp" not in i_sec:
            print("[Afriat-Inkling][Inkling-2-Mini] Block 2: Intersecting T=24 Afriat / GARP regimes (K=3)...", flush=True)
            garp_cells_i = [
                ("unclamped", 3),
                ("msg_homo_economicus_8d", 3),
                ("msg_qkv_8d", 3),
                ("msg_mlp_8d", 3),
                ("zero_flop_8d", 3),
            ]
            i_sec["block2_afriat_garp"] = run_block2_afriat_garp(
                ctrl_i, basis_i, eval_cells=garp_cells_i, batch_size=batch_size
            )
            _atomic_save_json(ckpt_path, payload)

        if "block3_inequity_and_steering" not in i_sec:
            print("[Afriat-Inkling][Inkling-2-Mini] Block 3: Disparity, Blount agency DiD, & message-bank steering...", flush=True)
            i_sec["block3_inequity_and_steering"] = run_block3_inequity_and_steering(
                ctrl_i, basis_i, pass_depths=(3,), steer_passes=3, batch_size=batch_size
            )
            _atomic_save_json(ckpt_path, payload)

        if "block4_dollar_wtp_curves" not in i_sec:
            print("[Afriat-Inkling][Inkling-2-Mini] Block 4: Continuous Dollar WTP reservation curves...", flush=True)
            i_sec["block4_dollar_wtp_curves"] = run_block4_wtp_curves(
                ctrl_i, basis_i, wtp_k_depths=(3,), steer_passes=3, batch_size=batch_size
            )
            _atomic_save_json(ckpt_path, payload)

        if "block5_rationalization_and_debt" not in i_sec:
            print("[Afriat-Inkling][Inkling-2-Mini] Block 5: Silent 0-CoT vs. <|content_thinking|> rationalization + Debt math...", flush=True)
            i_sec["block5_rationalization_and_debt"] = run_block5_rationalization_and_debt(
                ctrl_i, basis_i, silent_k_depths=(3,), think_passes=3, batch_size=max(8, batch_size // 2)
            )
            _atomic_save_json(ckpt_path, payload)

        i_sec["completed"] = True
        _atomic_save_json(ckpt_path, payload)

        ctrl_i.reset_interventions()
        ctrl_i.model.to("cpu")
        del ctrl_i
        gc.collect()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["completed"] = True
    payload["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    _atomic_save_json(final_path, payload)
    _atomic_save_json(ckpt_path, payload)
    print(f"[Afriat-Inkling] COMPLETE in {payload['runtime_seconds']}s -> {final_path}", flush=True)
    return payload
