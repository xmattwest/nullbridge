"""Phase 6 GPU Engine: Biophysical & Systems Loop Dissection of `thinkingmachines/Inkling-2-Mini`.

Executes 5 interconnected mechanistic experiments on native `Inkling2LoopForConditionalGeneration`
(`24` physical layers: `L0..L5` stem, `L6..L17` 3-pass recurrent core with somatic reset and
`4`-channel `Q, K, V, MLP` synaptic message bank, and `L18..L23` post-loop Coda):

  1. Exp 1 (`exp1_wht_synaptic_knockouts`): Full `2^4 = 16` Walsh-Hadamard synaptic channel
     knockouts across Pass 2 + Pass 3 (`joint_p2_p3`), `pass2_only`, and `pass3_only` (`48` sweeps,
     `N = 120` items across `factual_2hop`, `arithmetic_2step`, and `calendar_2step`), plus a
     zero-payload forget-gate damping control.
  2. Exp 2 (`exp2_gain_vs_amplitude`): All `48` unrolled layer visits (`v0..v47`) raw logit lens vs.
     block-mixed `final_input` lens vs. `3 x 6` truncated Coda readouts (`L18..L23`), paired with
     analytical VJP + patch-clamp directional Coda conductance `chi_t(u)` and 6-step singular value
     power iteration for non-normal transient amplification `sigma_max(J_pass_t)`.
  3. Exp 3 (`exp3_fisher_and_coda_shunt`): Full-vocabulary Hutchinson VJP Fisher pullback curvature
     `G_t` across `13` unrolled checkpoints (`N = 60`), Hop-1 -> Hop-2 commitment horizon `t*`, and
     `13C` Coda Layer-23 formatting shunt tracing across `N = 40` matched chat/raw triplets.
  4. Exp 4 (`exp4_hysteresis_pulse_chase`): Controlled Pass-1-only on-manifold foil concept
     perturbations (`beta in 0..1.25`) and orthogonal tangent pulses (`alpha in 0..2.0`) entering
     Pass 2 (`N = 60`), leaving Pass 3 unperturbed to measure elastic self-correction vs.
     path-dependent hysteretic foil capture and pulse-chase relaxation kinetics.
  5. Exp 5 (`exp5_neuromodulatory_switching`): Fixed-weight neuromodulatory interventions:
     (5a) Pass-clock gate telemetry (`L6..L17`), continuous message-gate conductance scaling
          (`gamma in 0..1.5`), and Pass-1 premature self-message clock transplant (`N = 60`).
     (5b) Coda `<|message_model|>` -> `<|content_text|>` header mode-switch (`slot0_embed_swap`,
          `slot0_and_slot1_somatic_swap`, and `somatic_carrier_loop_transplant`, `N = 40`).
"""

from __future__ import annotations

from contextlib import contextmanager
import gc
import json
import math
from pathlib import Path
import sys
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

from .algebra import exact_or_mc_signflip_p
from .biophysical_stimuli import (
    BiophysicalItem,
    ChatShuntTriplet,
    audit_biophysical_stimuli,
    build_phase6_stimulus_suite,
)
from .biophysical_wht import (
    CHANNEL_NAMES,
    bits_to_mask_code,
    compute_exact_wht_16,
    enumerate_16_masks,
    estimate_commitment_horizon,
    mean_and_ci as bootstrap_mean_ci,
    summarize_hysteresis_curve,
)
from .track_a import InklingLoopController, _atomic_save_json, _no_grad


EPS = 1e-12
GATE_ATTR_NAMES: tuple[str, str, str, str] = (
    "q_message_gate",
    "k_message_gate",
    "v_message_gate",
    "mlp_message_gate",
)


def first_token_variant_ids(tokenizer: Any, answer_text: str) -> list[int]:
    """Return deduplicated leading-token IDs for both bare and space-prefixed surface forms."""
    clean = answer_text.strip()
    out: list[int] = []
    for candidate in (clean, " " + clean):
        ids = tokenizer.encode(candidate, add_special_tokens=False)
        if ids and int(ids[0]) not in out:
            out.append(int(ids[0]))
    return out


class BiophysicalInklingController(InklingLoopController):
    """Explicit unrolled mechanistic controller for `Inkling2LoopForConditionalGeneration`."""

    def __init__(self, model: nn.Module, tokenizer: Any, model_label: str) -> None:
        super().__init__(model=model, tokenizer=tokenizer, model_label=model_label)
        # Keep backup weights on CPU to free ~1 GB of HBM3 VRAM for autograd Jacobians
        self._orig_lm_head = self._orig_lm_head.cpu()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.special_ids: dict[str, int] = {}
        for sp_tok in (
            "<|message_model|>",
            "<|content_text|>",
            "<|content_thinking|>",
            "<|end_message|>",
            "<|content_model_end_sampling|>",
        ):
            enc = self.tokenizer.encode(sp_tok, add_special_tokens=False)
            if len(enc) == 1:
                self.special_ids[sp_tok] = int(enc[0])

    def format_chat_header_only(self, user_text: str, effort: str = "0.7") -> str:
        """Render chat prompt ending right at `<|message_model|>` (before `<|content_thinking|>` or `<|content_text|>`)."""
        return (
            f"<|message_system|><|content_text|>Thinking effort level: {effort}<|end_message|>"
            f"<|message_user|><|content_text|>{user_text}<|end_message|>"
            "<|message_model|>"
        )

    def _build_masks_and_distances(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, Any], dict[str, Any]]:
        """Construct exact Inkling2 causal/sliding/conv masks and relative distance matrices."""
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        mod = sys.modules.get(type(self.text_model).__module__)
        create_causal = getattr(mod, "create_causal_mask", None) if mod is not None else None
        create_sliding = getattr(mod, "create_sliding_window_causal_mask", None) if mod is not None else None
        create_recurrent = getattr(mod, "create_recurrent_attention_mask", None) if mod is not None else None
        get_rel_dist = getattr(mod, "get_relative_distance", None) if mod is not None else None

        if create_causal is not None and create_sliding is not None and create_recurrent is not None:
            mask_kwargs = {
                "config": self.text_cfg,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache": None,
                "position_ids": position_ids,
            }
            causal_map = {
                "full_attention": create_causal(**mask_kwargs),
                "sliding_attention": create_sliding(**mask_kwargs),
                "linear_attention": create_recurrent(**mask_kwargs),
            }
        else:
            causal_map = {
                "full_attention": attention_mask,
                "sliding_attention": attention_mask,
                "linear_attention": attention_mask,
            }

        if get_rel_dist is not None:
            rel_map = {
                "full_attention": get_rel_dist(inputs_embeds, None, is_sliding=False),
                "sliding_attention": get_rel_dist(inputs_embeds, None, is_sliding=True),
            }
        else:
            q_len = inputs_embeds.shape[1]
            pos = torch.arange(q_len, device=inputs_embeds.device)
            dist = pos[:, None] - pos[None, :]
            rel_map = {"full_attention": dist, "sliding_attention": dist}

        return position_ids, causal_map, rel_map

    def _call_decoder_layer(
        self,
        layer: nn.Module,
        hidden_states: torch.Tensor,
        block_states: torch.Tensor,
        causal_map: dict[str, Any],
        rel_map: dict[str, Any],
        position_ids: torch.Tensor,
        messages: torch.Tensor | None = None,
        is_loop_call: bool = False,
    ) -> torch.Tensor:
        attn_type = getattr(layer, "attention_type", "full_attention")
        if is_loop_call:
            return layer(
                hidden_states,
                block_states,
                messages,
                attention_mask=causal_map[attn_type],
                relative_distance=rel_map[attn_type],
                conv_mask=causal_map["linear_attention"],
                cache=None,
                cache_offset=0,
                position_ids=position_ids,
            )
        return layer(
            hidden_states,
            block_states,
            attention_mask=causal_map[attn_type],
            relative_distance=rel_map[attn_type],
            conv_mask=causal_map["linear_attention"],
            cache=None,
            position_ids=position_ids,
        )

    def readout_logits_at_positions(
        self,
        block_states: torch.Tensor,
        hidden_states: torch.Tensor,
        row_idx: torch.Tensor,
        col_idx: torch.Tensor,
        use_block_final_input: bool = True,
    ) -> torch.Tensor:
        """Apply `(final_input -> norm -> lm_head -> logit_softcap)` at selected `(row_idx, col_idx)` tokens."""
        if use_block_final_input and hasattr(self.text_model, "final_input"):
            mixed = self.text_model.final_input(block_states, hidden_states)[0]
        else:
            mixed = hidden_states
        normed = self.text_model.norm(mixed).to(self.lm_head.weight.dtype)
        picked = normed[row_idx, col_idx]
        return self.logit_softcap(self.lm_head(picked).float())

    @contextmanager
    def gate_channel_intervention(
        self,
        channel_bits: tuple[int, int, int, int] = (1, 1, 1, 1),
        mode: str = "bypass",
        gamma_scale: float = 1.0,
    ):
        """Temporarily control the 4 synaptic message gates `(Q, K, V, MLP)` across `layers[loop_start:loop_end]`.

        Modes:
          - `'bypass'`: when `channel_bits[c] == 0`, gate returns `inputs[0]` (exact Pass-1 identity bypass).
          - `'zero_payload'`: when `channel_bits[c] == 0`, runs `gate(h, 0)` so `forget_gate(h, 0)` still damps `h`.
          - `'scale'`: interpolates `(1 - gamma) * h_in + gamma * gate_out` across active channels.
        """
        handles: list[Any] = []

        def _make_hook(c_idx: int):
            bit = int(channel_bits[c_idx])

            def _hook(mod: nn.Module, inputs: tuple[torch.Tensor, torch.Tensor], output: torch.Tensor) -> torch.Tensor:
                h_in, m_in = inputs[0], inputs[1]
                if mode == "bypass":
                    if bit == 0:
                        return h_in.float() if output.dtype == torch.float32 else h_in.to(output.dtype)
                    return output
                if mode == "zero_payload":
                    if bit == 0:
                        z = torch.zeros_like(m_in)
                        hf, mf = h_in.float(), z.float()
                        gh = torch.cat([mod.norm(hf), mod.norm(mf)], dim=-1)
                        gh = mod.down_proj(gh.to(mod.down_proj.weight.dtype)).float()
                        fg = torch.tanh(mod.forget_gate_proj(gh))
                        out_z = (1.0 - fg) * hf
                        return out_z if output.dtype == torch.float32 else out_z.to(output.dtype)
                    return output
                if mode == "scale":
                    hf = h_in.float()
                    of = output.float()
                    scaled = hf + float(gamma_scale) * (of - hf)
                    return scaled if output.dtype == torch.float32 else scaled.to(output.dtype)
                raise ValueError(f"Unknown gate intervention mode: {mode}")

            return _hook

        try:
            for l_idx in range(self.loop_start, self.loop_end):
                layer = self.text_model.layers[l_idx]
                for c_i, g_attr in enumerate(GATE_ATTR_NAMES):
                    gate_mod = getattr(layer, g_attr)
                    handles.append(gate_mod.register_forward_hook(_make_hook(c_i)))
            yield
        finally:
            for h in handles:
                h.remove()

    def unrolled_forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        n_passes: int = 3,
        mask_p2: tuple[int, int, int, int] = (1, 1, 1, 1),
        mask_p3: tuple[int, int, int, int] = (1, 1, 1, 1),
        mask_mode: str = "bypass",
        gamma_p2: float = 1.0,
        gamma_p3: float = 1.0,
        masks_by_pass: Sequence[tuple[int, int, int, int]] | None = None,
        gammas_by_pass: Sequence[float] | None = None,
        pass1_self_message: bool = False,
        record_all_visits: bool = False,
    ) -> dict[str, Any]:
        """Exact step-by-step unrolled execution of `Inkling2LoopTextModel` with per-pass surgical hooks."""
        inputs_embeds = self.text_model.embed_tokens(input_ids)
        if getattr(self.text_model, "_is_mock_tiny", False) and attention_mask is not None:
            inputs_embeds = inputs_embeds * attention_mask.unsqueeze(-1).to(inputs_embeds.dtype)
        position_ids, causal_map, rel_map = self._build_masks_and_distances(inputs_embeds, attention_mask)

        hidden_states = inputs_embeds
        block_states = inputs_embeds.new_zeros(*inputs_embeds.shape[:-1], 0, inputs_embeds.shape[-1])

        visits: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        # 1. Pre-loop stem layers `L0..L5`
        for l_idx in range(0, self.loop_start):
            layer = self.text_model.layers[l_idx]
            new_h = self._call_decoder_layer(
                layer, hidden_states, block_states, causal_map, rel_map, position_ids, is_loop_call=False
            )
            if layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_h
            if record_all_visits:
                visits.append((f"stem_L{l_idx}", block_states, hidden_states))

        pre_loop_h, pre_loop_b = hidden_states, block_states
        num_pre_blocks = pre_loop_b.shape[-2]

        pass_exits: list[tuple[torch.Tensor, torch.Tensor]] = []

        # 2. Pass 1 (`L6..L17`)
        if n_passes >= 1:
            for li, l_idx in enumerate(range(self.loop_start, self.loop_end)):
                layer = self.text_model.layers[l_idx]
                msgs_p1 = (
                    self.text_model.message_input(pre_loop_b[..., :0, :], pre_loop_h, li)
                    if pass1_self_message
                    else None
                )
                new_h = self._call_decoder_layer(
                    layer,
                    hidden_states,
                    block_states,
                    causal_map,
                    rel_map,
                    position_ids,
                    messages=msgs_p1,
                    is_loop_call=True,
                )
                if layer.is_block_boundary:
                    block_states = torch.cat(
                        [block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2
                    )
                hidden_states = new_h
                if record_all_visits:
                    visits.append((f"p1_L{l_idx}", block_states, hidden_states))
            pass_exits.append((block_states, hidden_states))

        # 3. Extra recurrent passes (`loop_idx = 1..n_passes-1`) with somatic reset + synaptic messages
        for loop_idx in range(1, n_passes):
            prev_b = block_states[..., num_pre_blocks + 1 :, :]
            prev_h = hidden_states
            hidden_states, block_states = pre_loop_h, pre_loop_b

            if masks_by_pass is not None and len(masks_by_pass) > 0:
                if len(masks_by_pass) == n_passes:
                    raw_bits = masks_by_pass[loop_idx]
                else:
                    raw_bits = masks_by_pass[min(loop_idx - 1, len(masks_by_pass) - 1)]
                cur_bits: tuple[int, int, int, int] = (
                    int(raw_bits[0]),
                    int(raw_bits[1]),
                    int(raw_bits[2]),
                    int(raw_bits[3]),
                )
            else:
                cur_bits = mask_p2 if loop_idx == 1 else mask_p3

            if gammas_by_pass is not None and len(gammas_by_pass) > 0:
                if len(gammas_by_pass) == n_passes:
                    cur_gamma = float(gammas_by_pass[loop_idx])
                else:
                    cur_gamma = float(gammas_by_pass[min(loop_idx - 1, len(gammas_by_pass) - 1)])
            else:
                cur_gamma = gamma_p2 if loop_idx == 1 else gamma_p3
            eff_mode = "scale" if abs(cur_gamma - 1.0) > 1e-9 and cur_bits == (1, 1, 1, 1) else mask_mode

            with self.gate_channel_intervention(channel_bits=cur_bits, mode=eff_mode, gamma_scale=cur_gamma):
                for li, l_idx in enumerate(range(self.loop_start, self.loop_end)):
                    layer = self.text_model.layers[l_idx]
                    msgs = self.text_model.message_input(prev_b, prev_h, li)
                    new_h = self._call_decoder_layer(
                        layer,
                        hidden_states,
                        block_states,
                        causal_map,
                        rel_map,
                        position_ids,
                        messages=msgs,
                        is_loop_call=True,
                    )
                    if layer.is_block_boundary:
                        block_states = torch.cat(
                            [block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2
                        )
                    hidden_states = new_h
                    if record_all_visits:
                        visits.append((f"p{loop_idx + 1}_L{l_idx}", block_states, hidden_states))
            pass_exits.append((block_states, hidden_states))

        # 4. Post-loop Coda (`L18..L23`)
        coda_states: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for l_idx in range(self.loop_end, self.n_layers):
            layer = self.text_model.layers[l_idx]
            new_h = self._call_decoder_layer(
                layer, hidden_states, block_states, causal_map, rel_map, position_ids, is_loop_call=False
            )
            if layer.is_block_boundary:
                block_states = torch.cat([block_states.to(hidden_states.device), hidden_states.unsqueeze(-2)], dim=-2)
            hidden_states = new_h
            coda_states.append((l_idx, block_states, hidden_states))
            if record_all_visits:
                visits.append((f"coda_L{l_idx}", block_states, hidden_states))

        fin_mixed = (
            self.text_model.final_input(block_states, hidden_states)[0]
            if hasattr(self.text_model, "final_input")
            else hidden_states
        )
        fin_norm = self.text_model.norm(fin_mixed).to(inputs_embeds.dtype)
        return {
            "last_hidden_state": fin_norm,
            "inputs_embeds": inputs_embeds,
            "pre_loop": (pre_loop_b, pre_loop_h),
            "pass_exits": pass_exits,
            "coda_states": coda_states,
            "visits": visits,
            "position_ids": position_ids,
            "causal_map": causal_map,
            "rel_map": rel_map,
        }

    def run_coda_from_state(
        self,
        block_states: torch.Tensor,
        hidden_states: torch.Tensor,
        causal_map: dict[str, Any],
        rel_map: dict[str, Any],
        position_ids: torch.Tensor,
        stop_coda_layer: int | None = None,
    ) -> tuple[torch.Tensor, list[tuple[int, torch.Tensor, torch.Tensor]]]:
        """Run Coda `L18..stop_coda_layer` starting from any loop-exit `(block_states, hidden_states)`."""
        end_exclusive = self.n_layers if stop_coda_layer is None else (int(stop_coda_layer) + 1)
        cur_b = block_states
        cur_h = hidden_states
        trace: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for l_idx in range(self.loop_end, end_exclusive):
            layer = self.text_model.layers[l_idx]
            new_h = self._call_decoder_layer(
                layer, cur_h, cur_b, causal_map, rel_map, position_ids, is_loop_call=False
            )
            if layer.is_block_boundary:
                cur_b = torch.cat([cur_b.to(cur_h.device), cur_h.unsqueeze(-2)], dim=-2)
            cur_h = new_h
            trace.append((l_idx, cur_b, cur_h))
        fin_mixed = (
            self.text_model.final_input(cur_b, cur_h)[0]
            if hasattr(self.text_model, "final_input")
            else cur_h
        )
        fin_norm = self.text_model.norm(fin_mixed)
        return fin_norm, trace


@_no_grad()
def run_preflight_equivalence_audit(ctrl: BiophysicalInklingController) -> dict[str, Any]:
    """Verify that `unrolled_forward` matches native `model.forward` at `K=3` and `K=1` (`0000` bypass)."""
    sample_prompts = [
        ctrl.format_0cot_prompt("Compute (3 * 2) + 1. Reply with the single integer only."),
        ctrl.format_0cot_prompt("Name the national capital city of the country where the Taj Mahal is located."),
    ]
    inp, msk, lens = ctrl._encode_right_padded(sample_prompts)
    row_idx = torch.arange(len(lens), device=ctrl.device)
    col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

    with ctrl.active_passes(3):
        nat_out_k3 = ctrl.model(input_ids=inp, attention_mask=msk, use_cache=False).logits[row_idx, col_idx].float()
    with ctrl.active_passes(1):
        nat_out_k1 = ctrl.model(input_ids=inp, attention_mask=msk, use_cache=False).logits[row_idx, col_idx].float()

    unr_k3 = ctrl.unrolled_forward(
        inp, msk, n_passes=3, mask_p2=(1, 1, 1, 1), mask_p3=(1, 1, 1, 1), mask_mode="bypass"
    )
    unr_logits_k3 = ctrl.logit_softcap(ctrl.lm_head(unr_k3["last_hidden_state"][row_idx, col_idx]).float())

    unr_0000 = ctrl.unrolled_forward(
        inp, msk, n_passes=3, mask_p2=(0, 0, 0, 0), mask_p3=(0, 0, 0, 0), mask_mode="bypass"
    )
    unr_logits_0000 = ctrl.logit_softcap(ctrl.lm_head(unr_0000["last_hidden_state"][row_idx, col_idx]).float())

    diff_k3 = float((nat_out_k3 - unr_logits_k3).abs().max().item())
    diff_k1_vs_0000 = float((nat_out_k1 - unr_logits_0000).abs().max().item())

    if diff_k3 > 1e-3 or diff_k1_vs_0000 > 1e-3:
        raise RuntimeError(
            f"Unrolled equivalence failed: diff_k3={diff_k3:.3e}, diff_k1_vs_0000={diff_k1_vs_0000:.3e}"
        )

    return {
        "valid": True,
        "max_abs_logit_diff_k3_vs_unrolled_1111": round(diff_k3, 8),
        "max_abs_logit_diff_k1_vs_unrolled_0000": round(diff_k1_vs_0000, 8),
    }


# ---------------------------------------------------------------------------
# Experiment 1: Full 2^4 = 16 Walsh-Hadamard Synaptic Knockouts & Epistasis
# ---------------------------------------------------------------------------

@_no_grad()
def _score_items_under_mask(
    ctrl: BiophysicalInklingController,
    items: Sequence[BiophysicalItem],
    mask_p2: tuple[int, int, int, int],
    mask_p3: tuple[int, int, int, int],
    mask_mode: str = "bypass",
    batch_size: int = 40,
    n_passes: int = 3,
    masks_by_pass: Sequence[tuple[int, int, int, int]] | None = None,
) -> dict[str, np.ndarray]:
    """Score teacher-forced continuation log-prob, first-token Top-1 accuracy, and target-vs-foil log-odds."""
    n = len(items)
    cont_lp = np.zeros(n, dtype=np.float64)
    top1_hit = np.zeros(n, dtype=np.float64)
    target_prob = np.zeros(n, dtype=np.float64)
    hop1_prob = np.zeros(n, dtype=np.float64)
    target_vs_foil_lo = np.zeros(n, dtype=np.float64)
    pad_id = ctrl.tokenizer.pad_token_id or 0

    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        sub = items[s:e]
        p_ids_list: list[list[int]] = []
        f_ids_list: list[list[int]] = []
        tgt_sets: list[list[int]] = []
        hop1_sets: list[list[int]] = []
        foil_sets: list[list[int]] = []

        for it in sub:
            pref = ctrl.format_0cot_prompt(it.user_prompt)
            p_ids = ctrl.tokenizer.encode(pref, add_special_tokens=False)
            f_ids = ctrl.tokenizer.encode(pref + it.target_answer, add_special_tokens=False)
            if len(f_ids) <= len(p_ids):
                f_ids = p_ids + ctrl.tokenizer.encode(it.target_answer, add_special_tokens=False)
            p_ids_list.append(p_ids)
            f_ids_list.append(f_ids)
            tgt_sets.append(first_token_variant_ids(ctrl.tokenizer, it.target_answer))
            hop1_sets.append(first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge))
            foil_sets.append(first_token_variant_ids(ctrl.tokenizer, it.foil_answer))

        max_l = max(len(x) for x in f_ids_list)
        bsz = len(sub)
        inp = torch.full((bsz, max_l), pad_id, dtype=torch.long, device=ctrl.device)
        msk = torch.zeros((bsz, max_l), dtype=torch.long, device=ctrl.device)
        for i, f_ids in enumerate(f_ids_list):
            inp[i, : len(f_ids)] = torch.tensor(f_ids, dtype=torch.long, device=ctrl.device)
            msk[i, : len(f_ids)] = 1

        unr = ctrl.unrolled_forward(
            inp,
            msk,
            n_passes=n_passes,
            mask_p2=mask_p2,
            mask_p3=mask_p3,
            mask_mode=mask_mode,
            masks_by_pass=masks_by_pass,
        )
        h_fin = unr["last_hidden_state"]

        for i, (p_ids, f_ids) in enumerate(zip(p_ids_list, f_ids_list)):
            q_pos = len(p_ids) - 1
            span_end = len(f_ids) - 1
            h_slice = h_fin[i, q_pos:span_end].to(ctrl.lm_head.weight.dtype)
            logits_slice = ctrl.logit_softcap(ctrl.lm_head(h_slice).float())
            lp_slice = F.log_softmax(logits_slice, dim=-1)

            # Teacher-forced full continuation log-prob
            lp_sum = 0.0
            for step_k, tok_id in enumerate(f_ids[len(p_ids) :]):
                lp_sum += float(lp_slice[step_k, tok_id].item())
            cont_lp[s + i] = lp_sum

            # Prompt-boundary distribution at `q_pos`
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
def run_exp1_wht_synaptic_knockouts(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    batch_size: int = 40,
) -> dict[str, Any]:
    """Run all 16 binary channel masks across Joint (P2+P3), Pass-2-only, and Pass-3-only schedules."""
    items_120: list[BiophysicalItem] = suite["reasoning_120"]
    regimes = ("factual_2hop", "arithmetic_2step", "calendar_2step")
    reg_indices = {
        r: np.asarray([idx for idx, it in enumerate(items_120) if it.regime == r], dtype=np.int64)
        for r in regimes
    }

    schedules = ("joint_p2_p3", "pass2_only", "pass3_only")
    cached_scores: dict[tuple[tuple[int, int, int, int], tuple[int, int, int, int]], dict[str, np.ndarray]] = {}

    masks_16 = enumerate_16_masks()
    out_by_schedule: dict[str, Any] = {}

    for sched in schedules:
        lp_cube = np.zeros((16, len(items_120)), dtype=np.float64)
        acc_cube = np.zeros((16, len(items_120)), dtype=np.float64)
        lo_cube = np.zeros((16, len(items_120)), dtype=np.float64)

        for m_int, bits, _code, _lbl in masks_16:
            if sched == "joint_p2_p3":
                mp2, mp3 = bits, bits
            elif sched == "pass2_only":
                mp2, mp3 = bits, (1, 1, 1, 1)
            else:
                mp2, mp3 = (1, 1, 1, 1), bits

            key = (mp2, mp3)
            if key not in cached_scores:
                cached_scores[key] = _score_items_under_mask(
                    ctrl, items_120, mask_p2=mp2, mask_p3=mp3, mask_mode="bypass", batch_size=batch_size
                )
            sc = cached_scores[key]
            lp_cube[m_int] = sc["continuation_logprob"]
            acc_cube[m_int] = sc["top1_acc"]
            lo_cube[m_int] = sc["target_vs_foil_lo"]

        sched_block: dict[str, Any] = {
            "all_120": {
                "continuation_logprob_wht": compute_exact_wht_16(lp_cube, seed=101),
                "top1_accuracy_wht": compute_exact_wht_16(acc_cube, seed=201),
                "target_vs_foil_lo_wht": compute_exact_wht_16(lo_cube, seed=301),
            },
            "by_regime": {},
        }
        for r_i, r_name in enumerate(regimes):
            idxs = reg_indices[r_name]
            sched_block["by_regime"][r_name] = {
                "continuation_logprob_wht": compute_exact_wht_16(lp_cube[:, idxs], seed=400 + r_i * 30),
                "top1_accuracy_wht": compute_exact_wht_16(acc_cube[:, idxs], seed=500 + r_i * 30),
                "target_vs_foil_lo_wht": compute_exact_wht_16(lo_cube[:, idxs], seed=600 + r_i * 30),
            }
        out_by_schedule[sched] = sched_block

    # Zero-payload somatic forget-gate damping control (`m = 0` vs full `bypass`)
    zero_payload_0000 = _score_items_under_mask(
        ctrl, items_120, mask_p2=(0, 0, 0, 0), mask_p3=(0, 0, 0, 0), mask_mode="zero_payload", batch_size=batch_size
    )
    bypass_0000 = cached_scores[((0, 0, 0, 0), (0, 0, 0, 0))]
    intact_1111 = cached_scores[((1, 1, 1, 1), (1, 1, 1, 1))]

    # Cross-regime dissociation: (QK_only - V_MLP_only) on factual_2hop vs arithmetic_2step
    qk_sc = cached_scores[((1, 1, 0, 0), (1, 1, 0, 0))]
    vm_sc = cached_scores[((0, 0, 1, 1), (0, 0, 1, 1))]
    fact_qk_adv = qk_sc["continuation_logprob"][reg_indices["factual_2hop"]] - vm_sc["continuation_logprob"][reg_indices["factual_2hop"]]
    arith_qk_adv = qk_sc["continuation_logprob"][reg_indices["arithmetic_2step"]] - vm_sc["continuation_logprob"][reg_indices["arithmetic_2step"]]

    return {
        "n_total_items": len(items_120),
        "channel_order": list(CHANNEL_NAMES),
        "schedules": out_by_schedule,
        "forget_gate_somatic_damping_control": {
            "bypass_0000_acc": round(float(np.mean(bypass_0000["top1_acc"])), 4),
            "zero_payload_0000_acc": round(float(np.mean(zero_payload_0000["top1_acc"])), 4),
            "intact_1111_acc": round(float(np.mean(intact_1111["top1_acc"])), 4),
            "bypass_0000_mean_lp": round(float(np.mean(bypass_0000["continuation_logprob"])), 4),
            "zero_payload_0000_mean_lp": round(float(np.mean(zero_payload_0000["continuation_logprob"])), 4),
            "intact_1111_mean_lp": round(float(np.mean(intact_1111["continuation_logprob"])), 4),
        },
        "regime_subcircuit_dissociation": {
            "factual_2hop_qk_minus_vmlp_lp": round(float(np.mean(fact_qk_adv)), 4),
            "arithmetic_2step_qk_minus_vmlp_lp": round(float(np.mean(arith_qk_adv)), 4),
            "factual_vs_arithmetic_did_lp": round(float(np.mean(fact_qk_adv) - np.mean(arith_qk_adv)), 4),
        },
    }


# ---------------------------------------------------------------------------
# Experiment 2: Gain vs. Amplitude (48-Visit Lens, Coda Conductance, Non-Normal Amplification)
# ---------------------------------------------------------------------------

def _select_balanced_subset(suite: dict[str, Any], per_regime: int = 20) -> list[BiophysicalItem]:
    out: list[BiophysicalItem] = []
    for r in ("factual_2hop", "arithmetic_2step", "calendar_2step"):
        out.extend(suite[r][:per_regime])
    return out


def run_exp2_gain_vs_amplitude(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    lens_per_regime: int = 20,
    jacobian_per_regime: int = 12,
    batch_size: int = 20,
) -> dict[str, Any]:
    """Compare 48-visit unembedding amplitude vs. truncated Coda readouts, Coda conductance, and non-normal gain."""
    items_60 = _select_balanced_subset(suite, per_regime=lens_per_regime)
    n60 = len(items_60)

    # Precompute top-128 SVD unembedding subspace of `lm_head` to measure unembedding amplitude alignment
    with torch.no_grad():
        w_head_sub = ctrl.lm_head.weight[: min(32768, ctrl.lm_head.weight.shape[0])].float()
        _u_h, _s_h, vh_head = torch.linalg.svd(w_head_sub, full_matrices=False)
        u_unembed = vh_head[: min(64, vh_head.shape[0])].t().contiguous()  # (d_model, 64)

    # Part 2A: All 48 layer visits (`raw_logit_lens` vs `block_mixed_lens`) + `3 passes x Coda layers` truncations
    visit_names: list[str] = []
    raw_acc_sum: np.ndarray | None = None
    blk_acc_sum: np.ndarray | None = None
    raw_prob_sum: np.ndarray | None = None
    blk_prob_sum: np.ndarray | None = None
    unembed_frac_sum: np.ndarray | None = None
    state_norm_sum: np.ndarray | None = None

    n_coda_layers = ctrl.n_layers - ctrl.loop_end
    trunc_acc = np.zeros((3, n_coda_layers), dtype=np.float64)
    trunc_prob = np.zeros((3, n_coda_layers), dtype=np.float64)
    trunc_hop1_prob = np.zeros((3, n_coda_layers), dtype=np.float64)

    with torch.no_grad():
        for s in range(0, n60, batch_size):
            e = min(n60, s + batch_size)
            sub = items_60[s:e]
            rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
            inp, msk, lens = ctrl._encode_right_padded(rendered)
            row_idx = torch.arange(len(lens), device=ctrl.device)
            col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
            h1_sets = [first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge) for it in sub]

            unr = ctrl.unrolled_forward(inp, msk, n_passes=3, record_all_visits=True)
            v_list = unr["visits"]
            if not visit_names:
                visit_names = [v[0] for v in v_list]
                nv = len(visit_names)
                raw_acc_sum = np.zeros(nv, dtype=np.float64)
                blk_acc_sum = np.zeros(nv, dtype=np.float64)
                raw_prob_sum = np.zeros(nv, dtype=np.float64)
                blk_prob_sum = np.zeros(nv, dtype=np.float64)
                unembed_frac_sum = np.zeros(nv, dtype=np.float64)
                state_norm_sum = np.zeros(nv, dtype=np.float64)

            assert raw_acc_sum is not None and blk_acc_sum is not None
            assert raw_prob_sum is not None and blk_prob_sum is not None
            assert unembed_frac_sum is not None and state_norm_sum is not None

            for v_i, (_vname, b_st, h_st) in enumerate(v_list):
                h_tok = h_st[row_idx, col_idx].float()
                h_nrm = torch.linalg.vector_norm(h_tok, dim=-1).clamp(min=1e-8)
                proj_nrm = torch.linalg.vector_norm(h_tok @ u_unembed, dim=-1)
                state_norm_sum[v_i] += float(h_nrm.sum().item())
                unembed_frac_sum[v_i] += float((proj_nrm / h_nrm).sum().item())

                lg_raw = ctrl.readout_logits_at_positions(b_st, h_st, row_idx, col_idx, use_block_final_input=False)
                lg_blk = ctrl.readout_logits_at_positions(b_st, h_st, row_idx, col_idx, use_block_final_input=True)
                pr_raw = F.softmax(lg_raw, dim=-1)
                pr_blk = F.softmax(lg_blk, dim=-1)
                top_raw = lg_raw.argmax(dim=-1).tolist()
                top_blk = lg_blk.argmax(dim=-1).tolist()
                for i in range(len(sub)):
                    t_ids = tgt_sets[i]
                    if top_raw[i] in t_ids:
                        raw_acc_sum[v_i] += 1.0
                    if top_blk[i] in t_ids:
                        blk_acc_sum[v_i] += 1.0
                    raw_prob_sum[v_i] += float(pr_raw[i, t_ids].sum().item())
                    blk_prob_sum[v_i] += float(pr_blk[i, t_ids].sum().item())

            # Truncated Coda grid across Pass 1, Pass 2, Pass 3 exits
            for p_idx, (b_exit, h_exit) in enumerate(unr["pass_exits"]):
                _fin, c_trace = ctrl.run_coda_from_state(
                    b_exit, h_exit, unr["causal_map"], unr["rel_map"], unr["position_ids"]
                )
                for c_k, (_l_idx, cb_st, ch_st) in enumerate(c_trace):
                    lg_c = ctrl.readout_logits_at_positions(cb_st, ch_st, row_idx, col_idx, use_block_final_input=True)
                    pr_c = F.softmax(lg_c, dim=-1)
                    top_c = lg_c.argmax(dim=-1).tolist()
                    for i in range(len(sub)):
                        t_ids = tgt_sets[i]
                        h1_ids = h1_sets[i]
                        if top_c[i] in t_ids:
                            trunc_acc[p_idx, c_k] += 1.0
                        trunc_prob[p_idx, c_k] += float(pr_c[i, t_ids].sum().item())
                        trunc_hop1_prob[p_idx, c_k] += float(pr_c[i, h1_ids].sum().item())

    assert raw_acc_sum is not None and blk_acc_sum is not None
    assert raw_prob_sum is not None and blk_prob_sum is not None
    assert unembed_frac_sum is not None and state_norm_sum is not None

    visits_summary = []
    for v_i, v_lbl in enumerate(visit_names):
        visits_summary.append(
            {
                "visit_idx": v_i,
                "visit_label": v_lbl,
                "raw_logit_lens_acc": round(float(raw_acc_sum[v_i] / n60), 4),
                "raw_logit_lens_prob": round(float(raw_prob_sum[v_i] / n60), 4),
                "block_mixed_lens_acc": round(float(blk_acc_sum[v_i] / n60), 4),
                "block_mixed_lens_prob": round(float(blk_prob_sum[v_i] / n60), 4),
                "unembed_svd64_amp_ratio": round(float(unembed_frac_sum[v_i] / n60), 4),
                "mean_hidden_norm": round(float(state_norm_sum[v_i] / n60), 3),
            }
        )

    truncated_coda_grid: dict[str, dict[str, dict[str, float]]] = {}
    for p_idx in range(3):
        p_key = f"pass_{p_idx + 1}"
        truncated_coda_grid[p_key] = {}
        for c_k in range(n_coda_layers):
            l_num = ctrl.loop_end + c_k
            truncated_coda_grid[p_key][f"L{l_num}"] = {
                "top1_acc": round(float(trunc_acc[p_idx, c_k] / n60), 4),
                "target_prob": round(float(trunc_prob[p_idx, c_k] / n60), 4),
                "hop1_prob": round(float(trunc_hop1_prob[p_idx, c_k] / n60), 4),
            }

    # Part 2B: Analytical VJP + Patch-Clamp Coda Conductance & 6-Step Singular Value Power Iteration
    items_jac = _select_balanced_subset(suite, per_regime=jacobian_per_regime)
    n_jac = len(items_jac)
    jac_bsz = min(12, n_jac)
    rng_g = torch.Generator(device="cpu").manual_seed(20260602)

    # Checkpoints for Coda Conductance: pass_1, pass_2, pass_3 exits -> final Coda normalized readout
    cond_stages = ("pass_1", "pass_2", "pass_3")
    chi_iso_vjp: dict[str, list[float]] = {k: [] for k in cond_stages}
    chi_rand_fd: dict[str, list[float]] = {k: [] for k in cond_stages}
    chi_task_fd: dict[str, list[float]] = {k: [] for k in cond_stages}
    task_grad_norm: dict[str, list[float]] = {k: [] for k in cond_stages}

    # Recurrent Pass Operators (`pass_1`, `pass_2`, `pass_3`) for Non-Normal Transient Amplification
    sigma_vjp_by_pass: dict[str, list[float]] = {k: [] for k in cond_stages}
    sigma_fd_by_pass: dict[str, list[float]] = {k: [] for k in cond_stages}
    iso_gain_by_pass: dict[str, list[float]] = {k: [] for k in cond_stages}
    io_cos_by_pass: dict[str, list[float]] = {k: [] for k in cond_stages}
    rayleigh_by_pass: dict[str, list[float]] = {k: [] for k in cond_stages}

    for s in range(0, n_jac, jac_bsz):
        e = min(n_jac, s + jac_bsz)
        sub = items_jac[s:e]
        bsz = len(sub)
        rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        inp, msk, lens = ctrl._encode_right_padded(rendered)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
        tgt_first = torch.tensor(
            [first_token_variant_ids(ctrl.tokenizer, it.target_answer)[0] for it in sub],
            dtype=torch.long,
            device=ctrl.device,
        )

        with torch.no_grad():
            base = ctrl.unrolled_forward(inp, msk, n_passes=3, record_all_visits=False)
            pre_b_clean, pre_h_clean = base["pre_loop"]
            c_map, r_map, pos_ids = base["causal_map"], base["rel_map"], base["position_ids"]
            exits_clean = [(b.detach(), h.detach()) for b, h in base["pass_exits"]]

        # 1) Coda Conductance `chi_p(u) = ||J_{p -> coda}(h_p) u||_2` for p = 1, 2, 3
        for p_idx, st_key in enumerate(cond_stages):
            b_ex, h_ex = exits_clean[p_idx]
            h_leaf = h_ex.detach().clone().requires_grad_(True)
            with torch.enable_grad():
                fin_n, _ = ctrl.run_coda_from_state(b_ex, h_leaf, c_map, r_map, pos_ids)
                z_last = fin_n[row_idx, col_idx].float()
                logits_last = ctrl.logit_softcap(ctrl.lm_head(z_last.to(ctrl.lm_head.weight.dtype)).float())
                lp_last = F.log_softmax(logits_last, dim=-1)
                target_lp = lp_last[row_idx, tgt_first].sum()

                # Task-aligned VJP gradient
                g_task_full = torch.autograd.grad(target_lp, h_leaf, retain_graph=True)[0]
                g_task = g_task_full[row_idx, col_idx].float()
                g_nrm = torch.linalg.vector_norm(g_task, dim=-1, keepdim=True).clamp(min=1e-12)
                u_task = g_task / g_nrm

                # Exact analytical isotropic conductance via 4 Rademacher/unit output probes
                iso_sq_acc = torch.zeros(bsz, dtype=torch.float32, device=ctrl.device)
                n_probes = 4
                for pr_i in range(n_probes):
                    w_cpu = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
                    w_unit = (w_cpu / torch.linalg.vector_norm(w_cpu, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)
                    g_w = torch.autograd.grad(
                        (z_last * w_unit).sum(),
                        h_leaf,
                        retain_graph=(pr_i < n_probes - 1),
                    )[0][row_idx, col_idx].float()
                    iso_sq_acc += g_w.square().sum(dim=-1)
                # Note: E_{w unit}[||J^T w||^2] = (1/d) tr(J^T J) = E_{u unit}[||J u||^2]
                chi_iso_batch = torch.sqrt(iso_sq_acc / float(n_probes))

            # Centered directional patch-clamp Coda conductance along `u_task` vs `u_rand`
            with torch.no_grad():
                h_base_tok = h_ex[row_idx, col_idx].float()
                step_mag = (0.05 * torch.linalg.vector_norm(h_base_tok, dim=-1, keepdim=True)).clamp(min=0.05)
                u_r_cpu = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
                u_rand = (u_r_cpu / torch.linalg.vector_norm(u_r_cpu, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)

                for u_vec, dest_list in ((u_rand, chi_rand_fd[st_key]), (u_task, chi_task_fd[st_key])):
                    h_plus = h_ex.clone()
                    h_minus = h_ex.clone()
                    h_plus[row_idx, col_idx] = (h_base_tok + step_mag * u_vec).to(h_ex.dtype)
                    h_minus[row_idx, col_idx] = (h_base_tok - step_mag * u_vec).to(h_ex.dtype)
                    z_p, _ = ctrl.run_coda_from_state(b_ex, h_plus, c_map, r_map, pos_ids)
                    z_m, _ = ctrl.run_coda_from_state(b_ex, h_minus, c_map, r_map, pos_ids)
                    dz = (z_p[row_idx, col_idx].float() - z_m[row_idx, col_idx].float()) / (2.0 * step_mag)
                    dest_list.extend(torch.linalg.vector_norm(dz, dim=-1).cpu().tolist())

            chi_iso_vjp[st_key].extend(chi_iso_batch.cpu().tolist())
            task_grad_norm[st_key].extend(g_nrm.squeeze(-1).cpu().tolist())

        # 2) Non-Normal Transient Amplification `sigma_max(J_pass_p)` across Pass 1, Pass 2, Pass 3
        num_pre_blocks = pre_b_clean.shape[-2]
        for p_idx, st_key in enumerate(cond_stages):
            if p_idx == 0:
                in_ref = pre_h_clean.detach()
                prev_b_fixed = None
            else:
                prev_b_full, prev_h_full = exits_clean[p_idx - 1]
                prev_b_fixed = prev_b_full[..., num_pre_blocks + 1 :, :].detach()
                in_ref = prev_h_full.detach()

            def _run_single_pass_op(h_var: torch.Tensor) -> torch.Tensor:
                if p_idx == 0:
                    cur_h, cur_b = h_var, pre_b_clean
                    for l_idx in range(ctrl.loop_start, ctrl.loop_end):
                        lyr = ctrl.text_model.layers[l_idx]
                        nh = ctrl._call_decoder_layer(
                            lyr, cur_h, cur_b, c_map, r_map, pos_ids, messages=None, is_loop_call=True
                        )
                        if lyr.is_block_boundary:
                            cur_b = torch.cat([cur_b.to(cur_h.device), cur_h.unsqueeze(-2)], dim=-2)
                        cur_h = nh
                    return cur_h
                cur_h, cur_b = pre_h_clean, pre_b_clean
                assert prev_b_fixed is not None
                for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                    lyr = ctrl.text_model.layers[l_idx]
                    msgs = ctrl.text_model.message_input(prev_b_fixed, h_var, li)
                    nh = ctrl._call_decoder_layer(
                        lyr, cur_h, cur_b, c_map, r_map, pos_ids, messages=msgs, is_loop_call=True
                    )
                    if lyr.is_block_boundary:
                        cur_b = torch.cat([cur_b.to(cur_h.device), cur_h.unsqueeze(-2)], dim=-2)
                    cur_h = nh
                return cur_h

            # Initialize unit `v_vec` and compute isotropic RMS gain on Pass `p`
            v_init = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
            v_vec = (v_init / torch.linalg.vector_norm(v_init, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)

            h_leaf_p = in_ref.clone().requires_grad_(True)
            with torch.enable_grad():
                out_p = _run_single_pass_op(h_leaf_p)
                out_last = out_p[row_idx, col_idx].float()
                iso_p_sq = torch.zeros(bsz, dtype=torch.float32, device=ctrl.device)
                for pr_i in range(4):
                    w_c = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
                    w_u = (w_c / torch.linalg.vector_norm(w_c, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)
                    gp = torch.autograd.grad((out_last * w_u).sum(), h_leaf_p, retain_graph=True)[0][row_idx, col_idx].float()
                    iso_p_sq += gp.square().sum(dim=-1)
                iso_gain_p = torch.sqrt(iso_p_sq / 4.0)

            # 6-step VJP + directional JVP power iteration
            in_tok_base = in_ref[row_idx, col_idx].float()
            step_p = (0.05 * torch.linalg.vector_norm(in_tok_base, dim=-1, keepdim=True)).clamp(min=0.05)
            u_vec = v_vec
            sig_vjp = torch.ones(bsz, dtype=torch.float32, device=ctrl.device)
            sig_fd = torch.ones(bsz, dtype=torch.float32, device=ctrl.device)

            for it_step in range(6):
                # Exact VJP half-step reusing `out_last` on step 0 or fresh graph after `v_vec` updates
                if it_step == 0:
                    with torch.enable_grad():
                        g_v = torch.autograd.grad((out_last * v_vec).sum(), h_leaf_p, retain_graph=False)[0][row_idx, col_idx].float()
                else:
                    hl = in_ref.clone().requires_grad_(True)
                    with torch.enable_grad():
                        ol = _run_single_pass_op(hl)[row_idx, col_idx].float()
                        g_v = torch.autograd.grad((ol * v_vec).sum(), hl, retain_graph=False)[0][row_idx, col_idx].float()
                sig_vjp = torch.linalg.vector_norm(g_v, dim=-1).clamp(min=1e-12)
                u_vec = g_v / sig_vjp.unsqueeze(-1)

                with torch.no_grad():
                    hp = in_ref.clone()
                    hm = in_ref.clone()
                    hp[row_idx, col_idx] = (in_tok_base + step_p * u_vec).to(in_ref.dtype)
                    hm[row_idx, col_idx] = (in_tok_base - step_p * u_vec).to(in_ref.dtype)
                    yp = _run_single_pass_op(hp)[row_idx, col_idx].float()
                    ym = _run_single_pass_op(hm)[row_idx, col_idx].float()
                    ju = (yp - ym) / (2.0 * step_p)
                    sig_fd = torch.linalg.vector_norm(ju, dim=-1).clamp(min=1e-12)
                    v_vec = ju / sig_fd.unsqueeze(-1)

            # Final exact analytical VJP certificate at converged `v_vec`
            hl_fin = in_ref.clone().requires_grad_(True)
            with torch.enable_grad():
                ol_fin = _run_single_pass_op(hl_fin)[row_idx, col_idx].float()
                g_fin = torch.autograd.grad((ol_fin * v_vec).sum(), hl_fin, retain_graph=False)[0][row_idx, col_idx].float()
            sig_vjp_cert = torch.linalg.vector_norm(g_fin, dim=-1).clamp(min=1e-12)
            u_fin = g_fin / sig_vjp_cert.unsqueeze(-1)
            io_cos = (u_fin * v_vec).sum(dim=-1).abs()
            rayleigh_q = sig_vjp_cert * io_cos

            sigma_vjp_by_pass[st_key].extend(sig_vjp_cert.cpu().tolist())
            sigma_fd_by_pass[st_key].extend(sig_fd.cpu().tolist())
            iso_gain_by_pass[st_key].extend(iso_gain_p.cpu().tolist())
            io_cos_by_pass[st_key].extend(io_cos.cpu().tolist())
            rayleigh_by_pass[st_key].extend(rayleigh_q.cpu().tolist())

    conductance_summary: dict[str, Any] = {}
    amplification_summary: dict[str, Any] = {}
    for idx_s, st_key in enumerate(cond_stages):
        ci_vjp = bootstrap_mean_ci(chi_iso_vjp[st_key], seed=2100 + idx_s)
        ci_rfd = bootstrap_mean_ci(chi_rand_fd[st_key], seed=2200 + idx_s)
        ci_tfd = bootstrap_mean_ci(chi_task_fd[st_key], seed=2300 + idx_s)
        ci_tgn = bootstrap_mean_ci(task_grad_norm[st_key], seed=2400 + idx_s)
        task_over_rand = np.asarray(chi_task_fd[st_key]) / np.maximum(np.asarray(chi_rand_fd[st_key]), 1e-8)
        conductance_summary[st_key] = {
            "isotropic_conductance_vjp": {"mean": round(ci_vjp[0], 4), "ci95": [round(ci_vjp[1], 4), round(ci_vjp[2], 4)]},
            "random_tangent_conductance_fd": {"mean": round(ci_rfd[0], 4), "ci95": [round(ci_rfd[1], 4), round(ci_rfd[2], 4)]},
            "task_aligned_conductance_fd": {"mean": round(ci_tfd[0], 4), "ci95": [round(ci_tfd[1], 4), round(ci_tfd[2], 4)]},
            "task_over_random_conductance_ratio": round(float(np.mean(task_over_rand)), 4),
            "task_logprob_grad_norm": {"mean": round(ci_tgn[0], 4), "ci95": [round(ci_tgn[1], 4), round(ci_tgn[2], 4)]},
        }

        sv_ci = bootstrap_mean_ci(sigma_vjp_by_pass[st_key], seed=2500 + idx_s)
        sf_ci = bootstrap_mean_ci(sigma_fd_by_pass[st_key], seed=2600 + idx_s)
        ig_ci = bootstrap_mean_ci(iso_gain_by_pass[st_key], seed=2700 + idx_s)
        cos_ci = bootstrap_mean_ci(io_cos_by_pass[st_key], seed=2800 + idx_s)
        rq_ci = bootstrap_mean_ci(rayleigh_by_pass[st_key], seed=2900 + idx_s)
        aniso = np.asarray(sigma_vjp_by_pass[st_key]) / np.maximum(np.asarray(iso_gain_by_pass[st_key]), 1e-8)
        non_norm_amp = np.asarray(sigma_vjp_by_pass[st_key]) / np.maximum(np.asarray(rayleigh_by_pass[st_key]), 1e-4)
        amplification_summary[st_key] = {
            "sigma_max_vjp": {"mean": round(sv_ci[0], 4), "ci95": [round(sv_ci[1], 4), round(sv_ci[2], 4)]},
            "sigma_max_fd": {"mean": round(sf_ci[0], 4), "ci95": [round(sf_ci[1], 4), round(sf_ci[2], 4)]},
            "isotropic_rms_gain": {"mean": round(ig_ci[0], 4), "ci95": [round(ig_ci[1], 4), round(ig_ci[2], 4)]},
            "anisotropy_ratio_sigma_over_iso": round(float(np.mean(aniso)), 4),
            "singular_io_alignment_cosine": {"mean": round(cos_ci[0], 4), "ci95": [round(cos_ci[1], 4), round(cos_ci[2], 4)]},
            "normal_rayleigh_quotient": {"mean": round(rq_ci[0], 4), "ci95": [round(rq_ci[1], 4), round(rq_ci[2], 4)]},
            "non_normal_transient_ratio_sigma_over_rayleigh": round(float(np.mean(non_norm_amp)), 4),
        }

    return {
        "n_lens_items": n60,
        "n_jacobian_items": n_jac,
        "visits_48": visits_summary,
        "truncated_coda_grid": truncated_coda_grid,
        "coda_conductance": conductance_summary,
        "non_normal_pass_amplification": amplification_summary,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Fisher Pullback Commitment Horizon (t*) & 13C Coda Shunt Tracing
# ---------------------------------------------------------------------------

def run_exp3_fisher_and_coda_shunt(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    fisher_per_regime: int = 20,
    n_rademacher: int = 6,
    fisher_bsz: int = 8,
) -> dict[str, Any]:
    """Compute 13-stage full-vocab Fisher pullback sensitivity, Hop-1->Hop-2 `t*`, and Layer-23 Coda shunt."""
    items_60 = _select_balanced_subset(suite, per_regime=fisher_per_regime)
    n60 = len(items_60)
    rng_f = torch.Generator(device="cpu").manual_seed(20260603)

    mid_loop_l = ctrl.loop_start + max(0, (ctrl.loop_end - ctrl.loop_start) // 2 - 1)
    end_loop_l = ctrl.loop_end - 1
    stem_end_l = ctrl.loop_start - 1

    target_visit_keys = [
        f"stem_L{stem_end_l}",
        f"p1_L{mid_loop_l}",
        f"p1_L{end_loop_l}",
        f"p2_L{mid_loop_l}",
        f"p2_L{end_loop_l}",
        f"p3_L{mid_loop_l}",
        f"p3_L{end_loop_l}",
        *[f"coda_L{l_i}" for l_i in range(ctrl.loop_end, ctrl.n_layers)],
    ]
    n_ckpts = len(target_visit_keys)
    fisher_trace_mat = np.zeros((n_ckpts, n60), dtype=np.float64)
    fisher_rel_mat = np.zeros((n_ckpts, n60), dtype=np.float64)

    # Also track Hop-1 vs Hop-2 probability trajectory across Stem, Pass 1, Pass 2, Pass 3 (via pre-shunt Coda L22 & full L23)
    # plus Coda L18..L23 direct readouts
    coda_probe_layer = max(ctrl.loop_end, ctrl.n_layers - 2)  # L22 on 24-layer Inkling-2-Mini
    traj_stages = ["stem", "pass_1", "pass_2", "pass_3"]
    hop1_prob_traj = {r: np.zeros(4, dtype=np.float64) for r in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step")}
    hop2_prob_traj = {r: np.zeros(4, dtype=np.float64) for r in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step")}
    hop2_minus_hop1_lo_traj = {
        r: np.zeros((4, 0), dtype=np.float64) for r in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step")
    }
    lo_per_item = np.zeros((4, n60), dtype=np.float64)

    for s in range(0, n60, fisher_bsz):
        e = min(n60, s + fisher_bsz)
        sub = items_60[s:e]
        bsz = len(sub)
        rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        inp, msk, lens = ctrl._encode_right_padded(rendered)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

        # Single autograd-enabled forward pass capturing all 13 checkpoints simultaneously
        with torch.enable_grad():
            emb = ctrl.text_model.embed_tokens(inp).detach().requires_grad_(True)
            if getattr(ctrl.text_model, "_is_mock_tiny", False) and msk is not None:
                emb_run = emb * msk.unsqueeze(-1).to(emb.dtype)
            else:
                emb_run = emb
            pos_ids, c_map, r_map = ctrl._build_masks_and_distances(emb_run, msk)

            h_cur = emb_run
            b_cur = emb_run.new_zeros(*emb_run.shape[:-1], 0, emb_run.shape[-1])
            visit_dict: dict[str, torch.Tensor] = {}
            stem_pair: tuple[torch.Tensor, torch.Tensor] | None = None
            pass_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

            for l_idx in range(0, ctrl.loop_start):
                lyr = ctrl.text_model.layers[l_idx]
                nh = ctrl._call_decoder_layer(lyr, h_cur, b_cur, c_map, r_map, pos_ids, is_loop_call=False)
                if lyr.is_block_boundary:
                    b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                h_cur = nh
                visit_dict[f"stem_L{l_idx}"] = h_cur
            pre_h, pre_b = h_cur, b_cur
            stem_pair = (b_cur.detach(), h_cur.detach())
            num_pre = pre_b.shape[-2]

            for l_idx in range(ctrl.loop_start, ctrl.loop_end):
                lyr = ctrl.text_model.layers[l_idx]
                nh = ctrl._call_decoder_layer(lyr, h_cur, b_cur, c_map, r_map, pos_ids, messages=None, is_loop_call=True)
                if lyr.is_block_boundary:
                    b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                h_cur = nh
                visit_dict[f"p1_L{l_idx}"] = h_cur
            pass_pairs.append((b_cur.detach(), h_cur.detach()))

            for loop_idx in range(1, 3):
                prev_b = b_cur[..., num_pre + 1 :, :]
                prev_h = h_cur
                h_cur, b_cur = pre_h, pre_b
                for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                    lyr = ctrl.text_model.layers[l_idx]
                    msgs = ctrl.text_model.message_input(prev_b, prev_h, li)
                    nh = ctrl._call_decoder_layer(lyr, h_cur, b_cur, c_map, r_map, pos_ids, messages=msgs, is_loop_call=True)
                    if lyr.is_block_boundary:
                        b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                    h_cur = nh
                    visit_dict[f"p{loop_idx + 1}_L{l_idx}"] = h_cur
                pass_pairs.append((b_cur.detach(), h_cur.detach()))

            for l_idx in range(ctrl.loop_end, ctrl.n_layers):
                lyr = ctrl.text_model.layers[l_idx]
                nh = ctrl._call_decoder_layer(lyr, h_cur, b_cur, c_map, r_map, pos_ids, is_loop_call=False)
                if lyr.is_block_boundary:
                    b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                h_cur = nh
                visit_dict[f"coda_L{l_idx}"] = h_cur

            fin_m = (
                ctrl.text_model.final_input(b_cur, h_cur)[0]
                if hasattr(ctrl.text_model, "final_input")
                else h_cur
            )
            fin_n = ctrl.text_model.norm(fin_m).to(ctrl.lm_head.weight.dtype)
            logits_last = ctrl.logit_softcap(ctrl.lm_head(fin_n[row_idx, col_idx]).float())
            probs_last = F.softmax(logits_last, dim=-1).detach()
            sqrt_p = torch.sqrt(probs_last.clamp(min=1e-30))

            ckpt_tensors = [visit_dict[k] for k in target_visit_keys]
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
                # Exact Fisher covariance factor: E[w w^T] = diag(p) - p p^T
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

        # Hop-1 vs Hop-2 basin trajectories across [stem, pass_1, pass_2, pass_3] via Coda readout
        with torch.no_grad():
            assert stem_pair is not None
            # Build 3-slot block state for stem so Coda sees consistent [slot0, slot1, slot2] shape
            b_stem_pad = stem_pair[0]
            while b_stem_pad.shape[-2] < pass_pairs[0][0].shape[-2]:
                b_stem_pad = torch.cat([b_stem_pad, stem_pair[1].unsqueeze(-2)], dim=-2)
            four_pairs = [(b_stem_pad, stem_pair[1]), *pass_pairs]
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
            h1_sets = [first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge) for it in sub]

            for st_i, (bp, hp) in enumerate(four_pairs):
                fin_st, _ = ctrl.run_coda_from_state(bp, hp, c_map, r_map, pos_ids, stop_coda_layer=ctrl.n_layers - 1)
                lg_st = ctrl.logit_softcap(ctrl.lm_head(fin_st[row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float())
                pr_st = F.softmax(lg_st, dim=-1)
                for i, it in enumerate(sub):
                    p1_m = float(pr_st[i, h1_sets[i]].sum().item())
                    p2_m = float(pr_st[i, tgt_sets[i]].sum().item())
                    lo_d = math.log(max(p2_m, 1e-30)) - math.log(max(p1_m, 1e-30))
                    lo_per_item[st_i, s + i] = lo_d
                    hop1_prob_traj["all"][st_i] += p1_m
                    hop2_prob_traj["all"][st_i] += p2_m
                    hop1_prob_traj[it.regime][st_i] += p1_m
                    hop2_prob_traj[it.regime][st_i] += p2_m

    _ = coda_probe_layer, hop2_minus_hop1_lo_traj

    fisher_checkpoints = []
    for c_i, v_key in enumerate(target_visit_keys):
        tr_ci = bootstrap_mean_ci(fisher_trace_mat[c_i], seed=3100 + c_i)
        rl_ci = bootstrap_mean_ci(fisher_rel_mat[c_i], seed=3200 + c_i)
        fisher_checkpoints.append(
            {
                "stage_key": v_key,
                "fisher_trace": {"mean": round(tr_ci[0], 5), "ci95": [round(tr_ci[1], 5), round(tr_ci[2], 5)]},
                "scale_normalized_fisher_kl_curvature": {
                    "mean": round(rl_ci[0], 5),
                    "ci95": [round(rl_ci[1], 5), round(rl_ci[2], 5)],
                },
            }
        )

    commitment_by_regime: dict[str, Any] = {}
    for r_key in ("all", "factual_2hop", "arithmetic_2step", "calendar_2step"):
        denom = n60 if r_key == "all" else fisher_per_regime
        idxs = (
            np.arange(n60)
            if r_key == "all"
            else np.asarray([idx for idx, it in enumerate(items_60) if it.regime == r_key])
        )
        mean_lo_curve = [float(np.mean(lo_per_item[st_i, idxs])) for st_i in range(4)]
        commitment_by_regime[r_key] = {
            "stages": traj_stages,
            "hop1_bridge_prob": [round(float(x / denom), 5) for x in hop1_prob_traj[r_key]],
            "hop2_target_prob": [round(float(x / denom), 5) for x in hop2_prob_traj[r_key]],
            "hop2_minus_hop1_log_odds": [round(v, 4) for v in mean_lo_curve],
            "horizon_estimate": estimate_commitment_horizon(traj_stages, mean_lo_curve),
        }

    # Part 3B: `13C` Coda Shunt Tracing across `N = 40` matched triplets
    triplets: list[ChatShuntTriplet] = suite["chat_shunt_triplets"]
    n_tr = len(triplets)
    n_coda = ctrl.n_layers - ctrl.loop_end
    coda_labels = [f"L{ctrl.loop_end + c_k}" for c_k in range(n_coda)]

    id_think = ctrl.special_ids.get("<|content_thinking|>")
    id_ctext = ctrl.special_ids.get("<|content_text|>")
    id_endm = ctrl.special_ids.get("<|end_message|>")

    shunt_modes = (
        "chat_header_effort07",
        "chat_header_effort0",
        "chat_content_text_effort0",
        "raw_completion",
    )
    shunt_results: dict[str, Any] = {}

    with torch.no_grad():
        for mode_name in shunt_modes:
            ans_prob_by_l = np.zeros((n_coda, n_tr), dtype=np.float64)
            ans_top1_by_l = np.zeros((n_coda, n_tr), dtype=np.float64)
            ans_top5_by_l = np.zeros((n_coda, n_tr), dtype=np.float64)
            think_prob_by_l = np.zeros((n_coda, n_tr), dtype=np.float64)
            ctext_prob_by_l = np.zeros((n_coda, n_tr), dtype=np.float64)
            fmt_prob_by_l = np.zeros((n_coda, n_tr), dtype=np.float64)

            for s in range(0, n_tr, 20):
                e = min(n_tr, s + 20)
                sub_t = triplets[s:e]
                if mode_name == "chat_header_effort07":
                    prompts = [ctrl.format_chat_header_only(t.user_question, effort="0.7") for t in sub_t]
                elif mode_name == "chat_header_effort0":
                    prompts = [ctrl.format_chat_header_only(t.user_question, effort="0") for t in sub_t]
                elif mode_name == "chat_content_text_effort0":
                    prompts = [ctrl.format_0cot_prompt(t.user_question) for t in sub_t]
                else:
                    prompts = [t.raw_completion_prompt for t in sub_t]

                inp, msk, lens = ctrl._encode_right_padded(prompts)
                row_idx = torch.arange(len(lens), device=ctrl.device)
                col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
                tgt_sets = [first_token_variant_ids(ctrl.tokenizer, t.target_answer) for t in sub_t]

                unr = ctrl.unrolled_forward(inp, msk, n_passes=3)
                for c_k, (_l_idx, cb_st, ch_st) in enumerate(unr["coda_states"]):
                    lg = ctrl.readout_logits_at_positions(cb_st, ch_st, row_idx, col_idx, use_block_final_input=True)
                    pr = F.softmax(lg, dim=-1)
                    top1 = lg.argmax(dim=-1).tolist()
                    top5 = torch.topk(lg, k=min(5, lg.shape[-1]), dim=-1).indices.tolist()
                    for i in range(len(sub_t)):
                        t_ids = tgt_sets[i]
                        ans_prob_by_l[c_k, s + i] = float(pr[i, t_ids].sum().item())
                        ans_top1_by_l[c_k, s + i] = 1.0 if top1[i] in t_ids else 0.0
                        ans_top5_by_l[c_k, s + i] = 1.0 if any(tid in top5[i] for tid in t_ids) else 0.0
                        p_th = float(pr[i, id_think].item()) if id_think is not None else 0.0
                        p_ct = float(pr[i, id_ctext].item()) if id_ctext is not None else 0.0
                        p_em = float(pr[i, id_endm].item()) if id_endm is not None else 0.0
                        think_prob_by_l[c_k, s + i] = p_th
                        ctext_prob_by_l[c_k, s + i] = p_ct
                        fmt_prob_by_l[c_k, s + i] = min(1.0, p_th + p_ct + p_em)

            layer_rows = []
            for c_k, clbl in enumerate(coda_labels):
                layer_rows.append(
                    {
                        "coda_layer": clbl,
                        "semantic_answer_prob": round(float(np.mean(ans_prob_by_l[c_k])), 5),
                        "semantic_answer_top1_acc": round(float(np.mean(ans_top1_by_l[c_k])), 4),
                        "semantic_answer_top5_acc": round(float(np.mean(ans_top5_by_l[c_k])), 4),
                        "content_thinking_prob": round(float(np.mean(think_prob_by_l[c_k])), 5),
                        "content_text_prob": round(float(np.mean(ctext_prob_by_l[c_k])), 5),
                        "format_shunt_total_prob": round(float(np.mean(fmt_prob_by_l[c_k])), 5),
                    }
                )
            peak_pre_l23 = float(np.max(np.mean(ans_prob_by_l[:-1], axis=1))) if n_coda >= 2 else float(np.mean(ans_prob_by_l[0]))
            l23_val = float(np.mean(ans_prob_by_l[-1]))
            shunt_results[mode_name] = {
                "by_coda_layer": layer_rows,
                "pre_final_peak_answer_prob": round(peak_pre_l23, 5),
                "final_layer_answer_prob": round(l23_val, 5),
                "layer23_shunt_drop_prob": round(peak_pre_l23 - l23_val, 5),
            }

    return {
        "n_fisher_items": n60,
        "n_shunt_triplets": n_tr,
        "fisher_pullback_trajectory": fisher_checkpoints,
        "commitment_horizon": commitment_by_regime,
        "coda_formatting_shunt": shunt_results,
    }


# ---------------------------------------------------------------------------
# Experiment 4: Hysteresis, Pulse-Chase Kinetics, & Self-Correction Across Passes
# ---------------------------------------------------------------------------

@_no_grad()
def run_exp4_hysteresis_and_pulse_chase(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    items_per_regime: int = 20,
    batch_size: int = 20,
) -> dict[str, Any]:
    """Inject Pass-1-only foil attractors and orthogonal pulses entering Pass 2, leaving Pass 3 unperturbed."""
    items_60 = _select_balanced_subset(suite, per_regime=items_per_regime)
    n60 = len(items_60)

    foil_betas = (0.0, 0.15, 0.30, 0.50, 0.70, 0.85, 1.00, 1.25)
    orth_alphas = (0.0, 0.10, 0.25, 0.50, 1.00, 1.50, 2.00)

    p2_target_p = np.zeros((len(foil_betas), n60), dtype=np.float64)
    p2_foil_p = np.zeros((len(foil_betas), n60), dtype=np.float64)
    p3_target_p = np.zeros((len(foil_betas), n60), dtype=np.float64)
    p3_foil_p = np.zeros((len(foil_betas), n60), dtype=np.float64)
    coda_target_acc = np.zeros((len(foil_betas), n60), dtype=np.float64)
    coda_foil_cap = np.zeros((len(foil_betas), n60), dtype=np.float64)
    coda_target_minus_foil_lo = np.zeros((len(foil_betas), n60), dtype=np.float64)

    # Pulse-chase state deviation `||h_perturbed - h_clean|| / ||delta_p1||` across Pass 2 -> Pass 3 -> Coda
    pulse_rel_p2 = np.zeros((len(orth_alphas), n60), dtype=np.float64)
    pulse_rel_p3 = np.zeros((len(orth_alphas), n60), dtype=np.float64)
    pulse_rel_coda = np.zeros((len(orth_alphas), n60), dtype=np.float64)
    pulse_acc_coda = np.zeros((len(orth_alphas), n60), dtype=np.float64)
    pulse_kl_coda = np.zeros((len(orth_alphas), n60), dtype=np.float64)

    rng_p = torch.Generator(device="cpu").manual_seed(20260604)

    for s in range(0, n60, batch_size):
        e = min(n60, s + batch_size)
        sub = items_60[s:e]
        bsz = len(sub)
        clean_prompts = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        foil_prompts = [ctrl.format_0cot_prompt(it.foil_user_prompt) for it in sub]

        inp_c, msk_c, lens_c = ctrl._encode_right_padded(clean_prompts)
        inp_f, msk_f, lens_f = ctrl._encode_right_padded(foil_prompts)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_c = torch.tensor([l_i - 1 for l_i in lens_c], dtype=torch.long, device=ctrl.device)
        col_f = torch.tensor([l_i - 1 for l_i in lens_f], dtype=torch.long, device=ctrl.device)

        tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
        foil_sets = [first_token_variant_ids(ctrl.tokenizer, it.foil_answer) for it in sub]

        # Run clean 3-pass trajectory and foil 1-pass state extraction
        clean_full = ctrl.unrolled_forward(inp_c, msk_c, n_passes=3)
        foil_p1 = ctrl.unrolled_forward(inp_f, msk_f, n_passes=1)

        pre_b_c, pre_h_c = clean_full["pre_loop"]
        num_pre = pre_b_c.shape[-2]
        c_map, r_map, pos_ids = clean_full["causal_map"], clean_full["rel_map"], clean_full["position_ids"]

        p1_b_clean_full, p1_h_clean = clean_full["pass_exits"][0]
        p1_b_clean = p1_b_clean_full[..., num_pre + 1 :, :]
        p2_h_clean_ref = clean_full["pass_exits"][1][1][row_idx, col_c].float()
        p3_h_clean_ref = clean_full["pass_exits"][2][1][row_idx, col_c].float()
        coda_clean_ref = clean_full["last_hidden_state"][row_idx, col_c].float()
        clean_logits_ref = ctrl.logit_softcap(
            ctrl.lm_head(clean_full["last_hidden_state"][row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float()
        )
        clean_lp_ref = F.log_softmax(clean_logits_ref, dim=-1)
        clean_p_ref = torch.exp(clean_lp_ref)

        p1_b_foil_full, p1_h_foil = foil_p1["pass_exits"][0]
        p1_b_foil_last = p1_b_foil_full[..., num_pre + 1 :, :][row_idx, col_f]  # (bsz, slots, d)
        p1_h_foil_last = p1_h_foil[row_idx, col_f]                              # (bsz, d)

        def _continue_p2_p3_coda(prev_b_into_p2: torch.Tensor, prev_h_into_p2: torch.Tensor):
            # Pass 2 (receives perturbed Pass-1 messages, somatic reset to clean `pre_h_c, pre_b_c`)
            h2, b2 = pre_h_c, pre_b_c
            for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                lyr = ctrl.text_model.layers[l_idx]
                msgs2 = ctrl.text_model.message_input(prev_b_into_p2, prev_h_into_p2, li)
                nh2 = ctrl._call_decoder_layer(lyr, h2, b2, c_map, r_map, pos_ids, messages=msgs2, is_loop_call=True)
                if lyr.is_block_boundary:
                    b2 = torch.cat([b2.to(h2.device), h2.unsqueeze(-2)], dim=-2)
                h2 = nh2

            # Pass 3 (100% unperturbed weights & somatic reset to clean `pre_h_c, pre_b_c`)
            prev_b3 = b2[..., num_pre + 1 :, :]
            prev_h3 = h2
            h3, b3 = pre_h_c, pre_b_c
            for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                lyr = ctrl.text_model.layers[l_idx]
                msgs3 = ctrl.text_model.message_input(prev_b3, prev_h3, li)
                nh3 = ctrl._call_decoder_layer(lyr, h3, b3, c_map, r_map, pos_ids, messages=msgs3, is_loop_call=True)
                if lyr.is_block_boundary:
                    b3 = torch.cat([b3.to(h3.device), h3.unsqueeze(-2)], dim=-2)
                h3 = nh3

            fin_p2, _ = ctrl.run_coda_from_state(b2, h2, c_map, r_map, pos_ids)
            fin_p3, _ = ctrl.run_coda_from_state(b3, h3, c_map, r_map, pos_ids)
            return h2, h3, fin_p2, fin_p3

        # Sweep 4A: On-manifold Foil Attractor Interpolation (`beta`)
        c_h_tok = p1_h_clean[row_idx, col_c].float()
        f_h_tok = p1_h_foil_last.float()
        c_b_tok = p1_b_clean[row_idx, col_c].float()
        f_b_tok = p1_b_foil_last.float()

        for b_i, beta in enumerate(foil_betas):
            pb_mod = p1_b_clean.clone()
            ph_mod = p1_h_clean.clone()
            ph_mod[row_idx, col_c] = ((1.0 - beta) * c_h_tok + beta * f_h_tok).to(p1_h_clean.dtype)
            if pb_mod.shape[-2] > 0 and f_b_tok.shape[-2] == pb_mod.shape[-2]:
                pb_mod[row_idx, col_c] = ((1.0 - beta) * c_b_tok + beta * f_b_tok).to(p1_b_clean.dtype)

            _h2, _h3, fin2, fin3 = _continue_p2_p3_coda(pb_mod, ph_mod)
            lg2 = ctrl.logit_softcap(ctrl.lm_head(fin2[row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float())
            lg3 = ctrl.logit_softcap(ctrl.lm_head(fin3[row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float())
            pr2 = F.softmax(lg2, dim=-1)
            pr3 = F.softmax(lg3, dim=-1)
            top3 = lg3.argmax(dim=-1).tolist()

            for i in range(bsz):
                t_ids = tgt_sets[i]
                fl_ids = foil_sets[i]
                pt2 = float(pr2[i, t_ids].sum().item())
                pf2 = float(pr2[i, fl_ids].sum().item())
                pt3 = float(pr3[i, t_ids].sum().item())
                pf3 = float(pr3[i, fl_ids].sum().item())
                p2_target_p[b_i, s + i] = pt2
                p2_foil_p[b_i, s + i] = pf2
                p3_target_p[b_i, s + i] = pt3
                p3_foil_p[b_i, s + i] = pf3
                coda_target_acc[b_i, s + i] = 1.0 if top3[i] in t_ids else 0.0
                coda_foil_cap[b_i, s + i] = 1.0 if top3[i] in fl_ids else 0.0
                coda_target_minus_foil_lo[b_i, s + i] = math.log(max(pt3, 1e-30)) - math.log(max(pf3, 1e-30))

        # Sweep 4B: Orthogonal Tangent Pulse-Chase (`alpha`)
        raw_u = torch.randn((bsz, ctrl.d_model), generator=rng_p, dtype=torch.float32).to(ctrl.device)
        # Gram-Schmidt orthogonalize `raw_u` against both `c_h_tok` and `(f_h_tok - c_h_tok)`
        e1 = c_h_tok / torch.linalg.vector_norm(c_h_tok, dim=-1, keepdim=True).clamp(min=1e-12)
        raw_u = raw_u - (raw_u * e1).sum(dim=-1, keepdim=True) * e1
        df = f_h_tok - c_h_tok
        e2 = df / torch.linalg.vector_norm(df, dim=-1, keepdim=True).clamp(min=1e-12)
        raw_u = raw_u - (raw_u * e2).sum(dim=-1, keepdim=True) * e2
        u_orth = raw_u / torch.linalg.vector_norm(raw_u, dim=-1, keepdim=True).clamp(min=1e-12)
        base_nrm = torch.linalg.vector_norm(c_h_tok, dim=-1, keepdim=True).clamp(min=1e-8)

        for a_i, alpha in enumerate(orth_alphas):
            ph_orth = p1_h_clean.clone()
            delta_vec = float(alpha) * base_nrm * u_orth
            ph_orth[row_idx, col_c] = (c_h_tok + delta_vec).to(p1_h_clean.dtype)
            h2_o, h3_o, _fin2_o, fin3_o = _continue_p2_p3_coda(p1_b_clean, ph_orth)

            d_in = torch.linalg.vector_norm(delta_vec, dim=-1).clamp(min=1e-8)
            d_p2 = torch.linalg.vector_norm(h2_o[row_idx, col_c].float() - p2_h_clean_ref, dim=-1)
            d_p3 = torch.linalg.vector_norm(h3_o[row_idx, col_c].float() - p3_h_clean_ref, dim=-1)
            d_cd = torch.linalg.vector_norm(fin3_o[row_idx, col_c].float() - coda_clean_ref, dim=-1)

            if alpha == 0.0:
                pulse_rel_p2[a_i, s:e] = 0.0
                pulse_rel_p3[a_i, s:e] = 0.0
                pulse_rel_coda[a_i, s:e] = 0.0
            else:
                pulse_rel_p2[a_i, s:e] = (d_p2 / d_in).cpu().numpy()
                pulse_rel_p3[a_i, s:e] = (d_p3 / d_in).cpu().numpy()
                pulse_rel_coda[a_i, s:e] = (d_cd / d_in).cpu().numpy()

            lg_o = ctrl.logit_softcap(ctrl.lm_head(fin3_o[row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float())
            lp_o = F.log_softmax(lg_o, dim=-1)
            kl_o = (clean_p_ref * (clean_lp_ref - lp_o)).sum(dim=-1).clamp(min=0.0)
            pulse_kl_coda[a_i, s:e] = kl_o.cpu().numpy()
            top_o = lg_o.argmax(dim=-1).tolist()
            for i in range(bsz):
                pulse_acc_coda[a_i, s + i] = 1.0 if top_o[i] in tgt_sets[i] else 0.0

    foil_curve_rows = []
    for b_i, beta in enumerate(foil_betas):
        foil_curve_rows.append(
            {
                "beta": beta,
                "pass2_target_prob": round(float(np.mean(p2_target_p[b_i])), 5),
                "pass2_foil_prob": round(float(np.mean(p2_foil_p[b_i])), 5),
                "pass3_target_prob": round(float(np.mean(p3_target_p[b_i])), 5),
                "pass3_foil_prob": round(float(np.mean(p3_foil_p[b_i])), 5),
                "coda_target_top1_acc": round(float(np.mean(coda_target_acc[b_i])), 4),
                "coda_foil_capture_rate": round(float(np.mean(coda_foil_cap[b_i])), 4),
                "coda_target_minus_foil_lo": round(float(np.mean(coda_target_minus_foil_lo[b_i])), 4),
            }
        )

    hyst_metrics = summarize_hysteresis_curve(
        foil_betas,
        [float(np.mean(p2_foil_p[i])) for i in range(len(foil_betas))],
        [float(np.mean(p3_foil_p[i])) for i in range(len(foil_betas))],
        [float(np.mean(coda_target_acc[i])) for i in range(len(foil_betas))],
        [float(np.mean(coda_foil_cap[i])) for i in range(len(foil_betas))],
    )

    pulse_rows = []
    for a_i, alpha in enumerate(orth_alphas):
        r2 = float(np.mean(pulse_rel_p2[a_i]))
        r3 = float(np.mean(pulse_rel_p3[a_i]))
        rc = float(np.mean(pulse_rel_coda[a_i]))
        pulse_rows.append(
            {
                "alpha": alpha,
                "rel_deviation_pass2": round(r2, 5),
                "rel_deviation_pass3": round(r3, 5),
                "rel_deviation_coda": round(rc, 5),
                "pass3_over_pass2_damping_ratio": round(r3 / max(r2, 1e-8), 4) if alpha > 0 else 0.0,
                "coda_top1_acc": round(float(np.mean(pulse_acc_coda[a_i])), 4),
                "output_kl_nats": round(float(np.mean(pulse_kl_coda[a_i])), 5),
            }
        )

    return {
        "n_items": n60,
        "foil_hysteresis_sweep": foil_curve_rows,
        "hysteresis_summary": hyst_metrics,
        "orthogonal_pulse_chase_sweep": pulse_rows,
    }


# ---------------------------------------------------------------------------
# Experiment 5: Neuromodulatory Mode-Switching on Fixed Weights (Clock & Header Swap)
# ---------------------------------------------------------------------------

@_no_grad()
def run_exp5_neuromodulatory_mode_switching(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    clock_per_regime: int = 20,
    batch_size: int = 20,
) -> dict[str, Any]:
    """Run (5a) Allosteric Pass-Clock Gate Modulation and (5b) Coda Header Slot-0/Somatic Mode-Switch."""
    items_60 = _select_balanced_subset(suite, per_regime=clock_per_regime)
    n60 = len(items_60)
    num_loop_l = ctrl.loop_end - ctrl.loop_start

    # 5A.1: Native Pass-2 vs Pass-3 Gate Open-Fraction Telemetry across L6..L17 x (Q, K, V, MLP)
    mg_abs_accum = np.zeros((2, num_loop_l, 4), dtype=np.float64)
    fg_val_accum = np.zeros((2, num_loop_l, 4), dtype=np.float64)
    retain_accum = np.zeros((2, num_loop_l, 4), dtype=np.float64)
    p2_p3_cos_accum = np.zeros((num_loop_l, 4), dtype=np.float64)

    for s in range(0, n60, batch_size):
        e = min(n60, s + batch_size)
        sub = items_60[s:e]
        bsz = len(sub)
        rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        inp, msk, lens = ctrl._encode_right_padded(rendered)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

        gate_calls: dict[tuple[int, int], list[tuple[torch.Tensor, torch.Tensor]]] = {
            (li, ci): [] for li in range(num_loop_l) for ci in range(4)
        }
        handles = []
        for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
            layer = ctrl.text_model.layers[l_idx]
            for ci, g_attr in enumerate(GATE_ATTR_NAMES):
                gm = getattr(layer, g_attr)

                def _make_tele_hook(li_k: int, ci_k: int):
                    def _thook(mod: nn.Module, inputs: tuple[torch.Tensor, torch.Tensor], _out: torch.Tensor) -> None:
                        hf, mf = inputs[0].float(), inputs[1].float()
                        gh = torch.cat([mod.norm(hf), mod.norm(mf)], dim=-1)
                        gh = mod.down_proj(gh.to(mod.down_proj.weight.dtype)).float()
                        fg = torch.tanh(mod.forget_gate_proj(gh))[row_idx, col_idx]
                        mg = torch.tanh(mod.message_gate_proj(gh))[row_idx, col_idx]
                        gate_calls[(li_k, ci_k)].append((mg.detach(), fg.detach()))

                    return _thook

                handles.append(gm.register_forward_hook(_make_tele_hook(li, ci)))
        try:
            ctrl.unrolled_forward(inp, msk, n_passes=3)
        finally:
            for h in handles:
                h.remove()

        for li in range(num_loop_l):
            for ci in range(4):
                (mg2, fg2), (mg3, fg3) = gate_calls[(li, ci)][0], gate_calls[(li, ci)][1]
                for p_slot_i, (mg_t, fg_t) in enumerate(((mg2, fg2), (mg3, fg3))):
                    mg_abs_accum[p_slot_i, li, ci] += float(mg_t.abs().mean(dim=-1).sum().item())
                    fg_val_accum[p_slot_i, li, ci] += float(fg_t.mean(dim=-1).sum().item())
                    retain_accum[p_slot_i, li, ci] += float((1.0 - fg_t).mean(dim=-1).sum().item())
                num_d = (mg2 * mg3).sum(dim=-1)
                den_d = (torch.linalg.vector_norm(mg2, dim=-1) * torch.linalg.vector_norm(mg3, dim=-1)).clamp(min=1e-12)
                p2_p3_cos_accum[li, ci] += float((num_d / den_d).sum().item())

    gate_telemetry_by_channel: dict[str, Any] = {}
    for ci, ch_lbl in enumerate(CHANNEL_NAMES):
        gate_telemetry_by_channel[ch_lbl] = {
            "pass2_mean_abs_message_gate": round(float(np.mean(mg_abs_accum[0, :, ci] / n60)), 5),
            "pass3_mean_abs_message_gate": round(float(np.mean(mg_abs_accum[1, :, ci] / n60)), 5),
            "pass2_mean_somatic_retention_1_minus_fg": round(float(np.mean(retain_accum[0, :, ci] / n60)), 5),
            "pass3_mean_somatic_retention_1_minus_fg": round(float(np.mean(retain_accum[1, :, ci] / n60)), 5),
            "pass2_vs_pass3_gate_cosine": round(float(np.mean(p2_p3_cos_accum[:, ci] / n60)), 5),
        }

    # 5A.2: Continuous Allosteric Gate Conductance Sweep (`gamma in 0..1.5`) + Pass-1 Clock Transplant
    gammas = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
    gamma_rows = []
    for gm_val in gammas:
        acc_hits = 0.0
        p_t_sum = 0.0
        p_h1_sum = 0.0
        for s in range(0, n60, batch_size):
            e = min(n60, s + batch_size)
            sub = items_60[s:e]
            rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
            inp, msk, lens = ctrl._encode_right_padded(rendered)
            row_idx = torch.arange(len(lens), device=ctrl.device)
            col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
            h1_sets = [first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge) for it in sub]

            if gm_val == 0.0:
                unr = ctrl.unrolled_forward(
                    inp, msk, n_passes=3, mask_p2=(0, 0, 0, 0), mask_p3=(0, 0, 0, 0), mask_mode="bypass"
                )
            else:
                unr = ctrl.unrolled_forward(inp, msk, n_passes=3, gamma_p2=gm_val, gamma_p3=gm_val)
            lg = ctrl.logit_softcap(
                ctrl.lm_head(unr["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
            )
            pr = F.softmax(lg, dim=-1)
            top = lg.argmax(dim=-1).tolist()
            for i in range(len(sub)):
                if top[i] in tgt_sets[i]:
                    acc_hits += 1.0
                p_t_sum += float(pr[i, tgt_sets[i]].sum().item())
                p_h1_sum += float(pr[i, h1_sets[i]].sum().item())
        gamma_rows.append(
            {
                "gamma": gm_val,
                "hop2_top1_acc": round(acc_hits / n60, 4),
                "hop2_target_prob": round(p_t_sum / n60, 5),
                "hop1_bridge_prob": round(p_h1_sum / n60, 5),
            }
        )

    # Pass-1 premature self-message clock transplant (`pass1_self_message=True` at K=1 vs clean K=1)
    p1_clean_acc = 0.0
    p1_clock_transplant_acc = 0.0
    for s in range(0, n60, batch_size):
        e = min(n60, s + batch_size)
        sub = items_60[s:e]
        rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        inp, msk, lens = ctrl._encode_right_padded(rendered)
        row_idx = torch.arange(len(lens), device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)
        tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]

        u_c1 = ctrl.unrolled_forward(inp, msk, n_passes=1, pass1_self_message=False)
        u_t1 = ctrl.unrolled_forward(inp, msk, n_passes=1, pass1_self_message=True)
        top_c1 = ctrl.logit_softcap(
            ctrl.lm_head(u_c1["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
        ).argmax(dim=-1).tolist()
        top_t1 = ctrl.logit_softcap(
            ctrl.lm_head(u_t1["last_hidden_state"][row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
        ).argmax(dim=-1).tolist()
        for i in range(len(sub)):
            if top_c1[i] in tgt_sets[i]:
                p1_clean_acc += 1.0
            if top_t1[i] in tgt_sets[i]:
                p1_clock_transplant_acc += 1.0

    # 5B: Coda Header Mode-Switch (`<|message_model|>` -> `<|content_text|>`) on `N = 40` Chat Triplets
    triplets: list[ChatShuntTriplet] = suite["chat_shunt_triplets"]
    n_tr = len(triplets)
    id_ctext = ctrl.special_ids.get("<|content_text|>")
    id_think = ctrl.special_ids.get("<|content_thinking|>")
    id_endm = ctrl.special_ids.get("<|end_message|>")

    if id_ctext is not None:
        ctext_tok_tensor = torch.tensor([[id_ctext]], dtype=torch.long, device=ctrl.device)
        emb_ctext_vec = ctrl.text_model.embed_tokens(ctext_tok_tensor)[0, 0].detach()
    else:
        ctext_ids_fb = ctrl.tokenizer.encode("<|content_text|>", add_special_tokens=False)
        emb_ctext_vec = ctrl.text_model.embed_tokens(
            torch.tensor([[ctext_ids_fb[-1]]], dtype=torch.long, device=ctrl.device)
        )[0, 0].detach()

    header_modes: dict[str, Any] = {}
    for eff_lbl in ("0", "0.7"):
        cond_names = (
            "baseline_header_L23",
            "slot0_embed_swap_L23",
            "slot0_and_slot1_somatic_swap_L23",
            "somatic_carrier_loop_transplant_L23",
            "pre_shunt_coda_L22",
            "oracle_next_token_content_text",
        )
        ans_probs = {c: np.zeros(n_tr, dtype=np.float64) for c in cond_names}
        ans_top1 = {c: np.zeros(n_tr, dtype=np.float64) for c in cond_names}
        ans_top5 = {c: np.zeros(n_tr, dtype=np.float64) for c in cond_names}
        fmt_probs = {c: np.zeros(n_tr, dtype=np.float64) for c in cond_names}
        anchor_examples: list[dict[str, Any]] = []

        for s in range(0, n_tr, 20):
            e = min(n_tr, s + 20)
            sub_t = triplets[s:e]
            bsz = len(sub_t)
            hdr_prompts = [ctrl.format_chat_header_only(t.user_question, effort=eff_lbl) for t in sub_t]
            orc_prompts = [
                f"<|message_system|><|content_text|>Thinking effort level: {eff_lbl}<|end_message|>"
                f"<|message_user|><|content_text|>{t.user_question}<|end_message|>"
                "<|message_model|><|content_text|>"
                for t in sub_t
            ]
            # Single-token substituted prompt where `<|message_model|>` at `-1` is replaced by `<|content_text|>`
            # used ONLY to read stem (`L0..L5`) somatic baseline `pre_loop_h_ct` with 0 recurrent passes
            sub_ct_prompts = [
                f"<|message_system|><|content_text|>Thinking effort level: {eff_lbl}<|end_message|>"
                f"<|message_user|><|content_text|>{t.user_question}<|end_message|>"
                "<|content_text|>"
                for t in sub_t
            ]

            inp_h, msk_h, lens_h = ctrl._encode_right_padded(hdr_prompts)
            inp_s, msk_s, lens_s = ctrl._encode_right_padded(sub_ct_prompts)
            inp_o, msk_o, lens_o = ctrl._encode_right_padded(orc_prompts)
            row_idx = torch.arange(bsz, device=ctrl.device)
            col_h = torch.tensor([l_i - 1 for l_i in lens_h], dtype=torch.long, device=ctrl.device)
            col_s = torch.tensor([l_i - 1 for l_i in lens_s], dtype=torch.long, device=ctrl.device)
            col_o = torch.tensor([l_i - 1 for l_i in lens_o], dtype=torch.long, device=ctrl.device)
            tgt_sets = [first_token_variant_ids(ctrl.tokenizer, t.target_answer) for t in sub_t]

            unr_h = ctrl.unrolled_forward(inp_h, msk_h, n_passes=3)
            unr_s_stem = ctrl.unrolled_forward(inp_s, msk_s, n_passes=1)
            unr_o = ctrl.unrolled_forward(inp_o, msk_o, n_passes=3)

            b_p3, h_p3 = unr_h["pass_exits"][2]
            pre_h_mm = unr_h["pre_loop"][1][row_idx, col_h]
            pre_h_ct = unr_s_stem["pre_loop"][1][row_idx, col_s]
            c_map, r_map, pos_ids = unr_h["causal_map"], unr_h["rel_map"], unr_h["position_ids"]

            # 1) Baseline unmodified `<|message_model|>`
            fin_base = unr_h["last_hidden_state"]

            # 2) Pure Slot-0 `block_states[:, -1, 0, :] <- embed_tokens(<|content_text|>)` entering Coda
            b_slot0 = b_p3.clone()
            if b_slot0.shape[-2] >= 1:
                b_slot0[row_idx, col_h, 0, :] = emb_ctext_vec.to(b_slot0.dtype)
            fin_slot0, _ = ctrl.run_coda_from_state(b_slot0, h_p3, c_map, r_map, pos_ids)

            # 3) Slot-0 + Slot-1 Somatic Anchor Swap (keeping `<|message_model|>` Pass-3 `slot2=h_11^(3)` & `h_17^(3)`)
            b_slot01 = b_slot0.clone()
            if b_slot01.shape[-2] >= 2:
                b_slot01[row_idx, col_h, 1, :] = pre_h_ct.to(b_slot01.dtype)
            fin_slot01, _ = ctrl.run_coda_from_state(b_slot01, h_p3, c_map, r_map, pos_ids)

            # 4) Full Somatic Carrier + `<|message_model|>` Pass-3 Loop Delta (`pre_h_ct + (h_17^(3)_mm - pre_h_mm)`)
            h_carrier = h_p3.clone()
            loop_delta_mm = h_p3[row_idx, col_h].float() - pre_h_mm.float()
            h_carrier[row_idx, col_h] = (pre_h_ct.float() + loop_delta_mm).to(h_p3.dtype)
            fin_carrier, _ = ctrl.run_coda_from_state(b_slot01, h_carrier, c_map, r_map, pos_ids)

            # 5) Pre-shunt Coda `L22` (`n_layers - 2`) readout on unmodified `<|message_model|>`
            pre_shunt_idx = max(0, len(unr_h["coda_states"]) - 2)
            _l22_id, cb22, ch22 = unr_h["coda_states"][pre_shunt_idx]
            fin_l22 = (
                ctrl.text_model.norm(ctrl.text_model.final_input(cb22, ch22)[0])
                if hasattr(ctrl.text_model, "final_input")
                else ctrl.text_model.norm(ch22)
            )

            # 6) Oracle `<|message_model|><|content_text|>` (1 token later)
            fin_orc = unr_o["last_hidden_state"]

            cond_tensors = [
                ("baseline_header_L23", fin_base, col_h),
                ("slot0_embed_swap_L23", fin_slot0, col_h),
                ("slot0_and_slot1_somatic_swap_L23", fin_slot01, col_h),
                ("somatic_carrier_loop_transplant_L23", fin_carrier, col_h),
                ("pre_shunt_coda_L22", fin_l22, col_h),
                ("oracle_next_token_content_text", fin_orc, col_o),
            ]

            decoded_top1_per_cond: dict[str, list[str]] = {}
            for cname, f_ten, cols in cond_tensors:
                lg = ctrl.logit_softcap(ctrl.lm_head(f_ten[row_idx, cols].to(ctrl.lm_head.weight.dtype)).float())
                pr = F.softmax(lg, dim=-1)
                top1 = lg.argmax(dim=-1).tolist()
                top5 = torch.topk(lg, k=min(5, lg.shape[-1]), dim=-1).indices.tolist()
                decoded_top1_per_cond[cname] = [ctrl.tokenizer.decode([tid], skip_special_tokens=False) for tid in top1]
                for i in range(bsz):
                    t_ids = tgt_sets[i]
                    ans_probs[cname][s + i] = float(pr[i, t_ids].sum().item())
                    ans_top1[cname][s + i] = 1.0 if top1[i] in t_ids else 0.0
                    ans_top5[cname][s + i] = 1.0 if any(tid in top5[i] for tid in t_ids) else 0.0
                    p_th = float(pr[i, id_think].item()) if id_think is not None else 0.0
                    p_ct = float(pr[i, id_ctext].item()) if id_ctext is not None else 0.0
                    p_em = float(pr[i, id_endm].item()) if id_endm is not None else 0.0
                    fmt_probs[cname][s + i] = min(1.0, p_th + p_ct + p_em)

            if s == 0:
                for i in range(min(3, bsz)):
                    anchor_examples.append(
                        {
                            "item_id": sub_t[i].item_id,
                            "target_answer": sub_t[i].target_answer,
                            "baseline_top1_token": decoded_top1_per_cond["baseline_header_L23"][i],
                            "slot0_swap_top1_token": decoded_top1_per_cond["slot0_embed_swap_L23"][i],
                            "slot01_swap_top1_token": decoded_top1_per_cond["slot0_and_slot1_somatic_swap_L23"][i],
                            "carrier_transplant_top1_token": decoded_top1_per_cond["somatic_carrier_loop_transplant_L23"][i],
                            "pre_shunt_L22_top1_token": decoded_top1_per_cond["pre_shunt_coda_L22"][i],
                            "oracle_top1_token": decoded_top1_per_cond["oracle_next_token_content_text"][i],
                        }
                    )

        cond_summary: dict[str, Any] = {}
        base_vec = ans_probs["baseline_header_L23"]
        for c_i, cname in enumerate(cond_names):
            pr_ci = bootstrap_mean_ci(ans_probs[cname], seed=5100 + c_i)
            ac_ci = bootstrap_mean_ci(ans_top1[cname], seed=5200 + c_i)
            diff_p = exact_or_mc_signflip_p(ans_probs[cname] - base_vec, seed=5300 + c_i)
            cond_summary[cname] = {
                "semantic_answer_prob": {"mean": round(pr_ci[0], 5), "ci95": [round(pr_ci[1], 5), round(pr_ci[2], 5)]},
                "semantic_answer_top1_acc": {"mean": round(ac_ci[0], 4), "ci95": [round(ac_ci[1], 4), round(ac_ci[2], 4)]},
                "semantic_answer_top5_acc": round(float(np.mean(ans_top5[cname])), 4),
                "format_token_prob": round(float(np.mean(fmt_probs[cname])), 5),
                "p_vs_baseline_signflip": round(diff_p, 6),
            }
        header_modes[f"effort_{eff_lbl}"] = {
            "conditions": cond_summary,
            "canonical_anchor_predictions": anchor_examples,
        }

    return {
        "n_clock_items": n60,
        "n_header_items": n_tr,
        "pass_clock_gate_telemetry": gate_telemetry_by_channel,
        "allosteric_conductance_sweep": gamma_rows,
        "pass1_self_message_clock_transplant": {
            "clean_pass1_acc": round(p1_clean_acc / n60, 4),
            "premature_self_message_pass1_acc": round(p1_clock_transplant_acc / n60, 4),
            "delta_acc": round((p1_clock_transplant_acc - p1_clean_acc) / n60, 4),
        },
        "coda_header_mode_switch": header_modes,
    }


# ---------------------------------------------------------------------------
# End-to-End Phase 6 GPU Pipeline Runner
# ---------------------------------------------------------------------------

def _load_biophysical_controller(
    model_dir: Path | str,
    model_label: str,
) -> tuple[BiophysicalInklingController, dict[str, Any]]:
    from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration

    t0 = time.perf_counter()
    tok = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    dev = "cuda:0" if (torch is not None and torch.cuda.is_available()) else "cpu"
    dt = torch.bfloat16 if dev.startswith("cuda") else torch.float32
    model = Inkling2LoopForConditionalGeneration.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=dt,
        device_map=dev,
    )
    model.eval()
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    ctrl = BiophysicalInklingController(model=model, tokenizer=tok, model_label=model_label)
    gpu_name = torch.cuda.get_device_name(0) if (torch is not None and torch.cuda.is_available()) else "cpu"
    meta = {
        "model_id": model_label,
        "gpu_name": gpu_name,
        "d_model": ctrl.d_model,
        "n_layers": ctrl.n_layers,
        "loop_start": ctrl.loop_start,
        "loop_end": ctrl.loop_end,
        "default_passes": ctrl.default_passes,
        "gate_rank": ctrl.gate_rank,
        "cuda_allocated_gb": round(float(torch.cuda.memory_allocated() / 1e9), 3)
        if (torch is not None and torch.cuda.is_available())
        else 0.0,
        "load_seconds": round(time.perf_counter() - t0, 2),
    }
    return ctrl, meta


def run_phase6_biophysical_pipeline(
    inkling_mini_dir: Path | str,
    inkling_label: str = "thinkingmachines/Inkling-2-Mini",
    out_dir: Path = Path("runs/phase6_inkling_biophysical"),
    batch_size: int = 32,
    resume_json: Path | None = None,
) -> dict[str, Any]:
    """Execute the complete 5-experiment Phase 6 Biophysical Loop study on `Inkling-2-Mini`."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "checkpoint_phase6_biophysical_results.json"
    final_path = out_dir / "phase6_biophysical_results.json"

    suite = build_phase6_stimulus_suite()
    stim_audit = audit_biophysical_stimuli(suite)
    if not stim_audit["valid"]:
        raise RuntimeError(f"Phase 6 stimulus firewall audit failed: {stim_audit['errors']}")

    payload: dict[str, Any] = {
        "phase": "phase6_inkling_biophysical_loop",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "stimulus_audit": stim_audit,
        "completed": False,
    }
    if resume_json is not None and Path(resume_json).is_file():
        payload = json.loads(Path(resume_json).read_text(encoding="utf-8"))
        payload["completed"] = False

    print(f"[Phase 6 Biophysical] Loading {inkling_label} from {inkling_mini_dir}...", flush=True)
    ctrl, meta = _load_biophysical_controller(inkling_mini_dir, inkling_label)
    payload["metadata"] = meta
    print(
        f"  -> Loaded {inkling_label} on {meta['gpu_name']}: {meta['cuda_allocated_gb']} GB VRAM ({meta['load_seconds']}s)",
        flush=True,
    )

    preflight = run_preflight_equivalence_audit(ctrl)
    payload["preflight_equivalence"] = preflight
    print(
        f"  -> Preflight bit-equivalence OK: k3_diff={preflight['max_abs_logit_diff_k3_vs_unrolled_1111']:.2e}, "
        f"k1_vs_0000_diff={preflight['max_abs_logit_diff_k1_vs_unrolled_0000']:.2e}",
        flush=True,
    )
    _atomic_save_json(ckpt_path, payload)

    if "exp1_wht_synaptic_knockouts" not in payload:
        te = time.perf_counter()
        print("[Phase 6 | Exp 1] Running 2^4=16 Walsh-Hadamard Synaptic Knockouts (Joint, P2-only, P3-only, N=120)...", flush=True)
        payload["exp1_wht_synaptic_knockouts"] = run_exp1_wht_synaptic_knockouts(
            ctrl, suite, batch_size=batch_size
        )
        _atomic_save_json(ckpt_path, payload)
        j_acc = payload["exp1_wht_synaptic_knockouts"]["schedules"]["joint_p2_p3"]["all_120"]["top1_accuracy_wht"]
        print(
            f"  -> Exp 1 done in {time.perf_counter() - te:.1f}s | "
            f"Acc(0000->1111): {j_acc['masks']['0000']['mean']*100:.1f}% -> {j_acc['masks']['1111']['mean']*100:.1f}%",
            flush=True,
        )

    if "exp2_gain_vs_amplitude" not in payload:
        te = time.perf_counter()
        print("[Phase 6 | Exp 2] Running 48-Visit Amplitude Lens, Truncated Coda Grid, Conductance & Power Iteration...", flush=True)
        payload["exp2_gain_vs_amplitude"] = run_exp2_gain_vs_amplitude(
            ctrl, suite, lens_per_regime=20, jacobian_per_regime=12, batch_size=min(20, batch_size)
        )
        _atomic_save_json(ckpt_path, payload)
        amp = payload["exp2_gain_vs_amplitude"]["non_normal_pass_amplification"]
        print(
            f"  -> Exp 2 done in {time.perf_counter() - te:.1f}s | "
            f"sigma_max(P1,P2,P3)=({amp['pass_1']['sigma_max_vjp']['mean']:.2f}, "
            f"{amp['pass_2']['sigma_max_vjp']['mean']:.2f}, {amp['pass_3']['sigma_max_vjp']['mean']:.2f})",
            flush=True,
        )

    if "exp3_fisher_and_coda_shunt" not in payload:
        te = time.perf_counter()
        print("[Phase 6 | Exp 3] Running 13-Stage Fisher Pullback (t*) & Layer-23 Coda Shunt Tracing...", flush=True)
        payload["exp3_fisher_and_coda_shunt"] = run_exp3_fisher_and_coda_shunt(
            ctrl, suite, fisher_per_regime=20, n_rademacher=6, fisher_bsz=min(8, batch_size)
        )
        _atomic_save_json(ckpt_path, payload)
        sh07 = payload["exp3_fisher_and_coda_shunt"]["coda_formatting_shunt"]["chat_header_effort07"]
        print(
            f"  -> Exp 3 done in {time.perf_counter() - te:.1f}s | "
            f"Coda pre-L23 peak={sh07['pre_final_peak_answer_prob']*100:.2f}% -> L23={sh07['final_layer_answer_prob']*100:.2f}%",
            flush=True,
        )

    if "exp4_hysteresis_pulse_chase" not in payload:
        te = time.perf_counter()
        print("[Phase 6 | Exp 4] Running Pass-1 Foil Hysteresis & Orthogonal Pulse-Chase Sweeps (N=60)...", flush=True)
        payload["exp4_hysteresis_pulse_chase"] = run_exp4_hysteresis_and_pulse_chase(
            ctrl, suite, items_per_regime=20, batch_size=min(20, batch_size)
        )
        _atomic_save_json(ckpt_path, payload)
        hs = payload["exp4_hysteresis_pulse_chase"]["hysteresis_summary"]
        print(
            f"  -> Exp 4 done in {time.perf_counter() - te:.1f}s | "
            f"peak_P3_foil_suppression={hs['peak_pass3_foil_suppression']*100:.2f} pp, beta_c={hs['critical_bifurcation_beta']}",
            flush=True,
        )

    if "exp5_neuromodulatory_switching" not in payload:
        te = time.perf_counter()
        print("[Phase 6 | Exp 5] Running Allosteric Pass-Clock Gate Sweep & Coda Header Swap (N=60 + 40)...", flush=True)
        payload["exp5_neuromodulatory_switching"] = run_exp5_neuromodulatory_mode_switching(
            ctrl, suite, clock_per_regime=20, batch_size=min(20, batch_size)
        )
        _atomic_save_json(ckpt_path, payload)
        hm0 = payload["exp5_neuromodulatory_switching"]["coda_header_mode_switch"]["effort_0"]["conditions"]
        print(
            f"  -> Exp 5 done in {time.perf_counter() - te:.1f}s | "
            f"Header->Text swap top1 acc (eff=0): base={hm0['baseline_header_L23']['semantic_answer_top1_acc']['mean']*100:.1f}% "
            f"slot0={hm0['slot0_embed_swap_L23']['semantic_answer_top1_acc']['mean']*100:.1f}% "
            f"carrier={hm0['somatic_carrier_loop_transplant_L23']['semantic_answer_top1_acc']['mean']*100:.1f}%",
            flush=True,
        )

    ctrl.reset_interventions()
    ctrl.model.to("cpu")
    del ctrl
    gc.collect()
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()

    payload["completed"] = True
    payload["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    _atomic_save_json(final_path, payload)
    _atomic_save_json(ckpt_path, payload)
    print(f"[Phase 6 Biophysical] COMPLETE in {payload['runtime_seconds']}s -> {final_path}", flush=True)
    return payload


# ---------------------------------------------------------------------------
# Phase 7 Experiment Block B1: Varloop K=1..7 Depth Trajectory Suite
# ---------------------------------------------------------------------------

@_no_grad()
def run_varloop_preflight_equivalence(
    ctrl: BiophysicalInklingController,
    depths: Sequence[int] = (1, 3, 5, 7),
) -> dict[str, Any]:
    """Verify `unrolled_forward` matches native `model.forward` across multi-K depths (target 0.00e+00)."""
    sample_prompts = [
        ctrl.format_0cot_prompt("Compute (3 * 2) + 1. Reply with the single integer only."),
        ctrl.format_0cot_prompt("Name the national capital city of the country where the Taj Mahal is located."),
    ]
    inp, msk, lens = ctrl._encode_right_padded(sample_prompts)
    row_idx = torch.arange(len(lens), device=ctrl.device)
    col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

    per_depth_diffs: dict[str, float] = {}
    nat_k1: torch.Tensor | None = None
    sorted_depths = sorted({int(k) for k in depths if int(k) >= 1})

    for k in sorted_depths:
        with ctrl.active_passes(k):
            nat_out = ctrl.model(input_ids=inp, attention_mask=msk, use_cache=False).logits[row_idx, col_idx].float()
        if k == 1:
            nat_k1 = nat_out
        masks_k = [(1, 1, 1, 1)] * max(0, k - 1) if k > 1 else None
        unr_k = ctrl.unrolled_forward(
            inp,
            msk,
            n_passes=k,
            masks_by_pass=masks_k,
            mask_mode="bypass",
        )
        unr_logits = ctrl.logit_softcap(ctrl.lm_head(unr_k["last_hidden_state"]))[row_idx, col_idx].float()
        diff_k = float((nat_out - unr_logits).abs().max().item())
        per_depth_diffs[f"K={k}"] = round(diff_k, 8)
        if diff_k > 1e-3:
            raise RuntimeError(f"Varloop preflight equivalence failed at K={k}: max_abs_diff={diff_k:.3e}")

    max_k = max(sorted_depths) if sorted_depths else 3
    if nat_k1 is None:
        with ctrl.active_passes(1):
            nat_k1 = ctrl.model(input_ids=inp, attention_mask=msk, use_cache=False).logits[row_idx, col_idx].float()
    unr_zero = ctrl.unrolled_forward(
        inp,
        msk,
        n_passes=max_k,
        masks_by_pass=[(0, 0, 0, 0)] * max(0, max_k - 1) if max_k > 1 else None,
        mask_mode="bypass",
    )
    unr_zero_logits = ctrl.logit_softcap(ctrl.lm_head(unr_zero["last_hidden_state"]))[row_idx, col_idx].float()
    diff_zero_vs_k1 = float((nat_k1 - unr_zero_logits).abs().max().item())
    if diff_zero_vs_k1 > 1e-3:
        raise RuntimeError(
            f"Varloop 0000-bypass equivalence failed at K={max_k} vs K=1: max_abs_diff={diff_zero_vs_k1:.3e}"
        )

    return {
        "valid": True,
        "depths_checked": sorted_depths,
        "max_abs_logit_diff_by_depth": per_depth_diffs,
        f"max_abs_logit_diff_k1_vs_k{max_k}_0000_bypass": round(diff_zero_vs_k1, 8),
    }


@_no_grad()
def run_varloop_wht_depth_trajectory(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    depths: Sequence[int] = (2, 3, 5, 7),
    batch_size: int = 32,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Run exact 16-mask WHT depth trajectory (`K in depths`, joint mask on Passes `2..K`)."""
    items_120: list[BiophysicalItem] = suite["reasoning_120"]
    regimes = ("factual_2hop", "arithmetic_2step", "calendar_2step")
    reg_indices = {
        r: np.asarray([idx for idx, it in enumerate(items_120) if it.regime == r], dtype=np.int64)
        for r in regimes
    }
    masks_16 = enumerate_16_masks()
    by_depth: dict[str, Any] = {}

    for d_i, k in enumerate(sorted({int(x) for x in depths if int(x) >= 2})):
        lp_cube = np.zeros((16, len(items_120)), dtype=np.float64)
        acc_cube = np.zeros((16, len(items_120)), dtype=np.float64)
        lo_cube = np.zeros((16, len(items_120)), dtype=np.float64)

        for m_int, bits, _code, _lbl in masks_16:
            sc = _score_items_under_mask(
                ctrl,
                items_120,
                mask_p2=bits,
                mask_p3=bits,
                mask_mode="bypass",
                batch_size=batch_size,
                n_passes=k,
                masks_by_pass=[bits] * (k - 1),
            )
            lp_cube[m_int] = sc["continuation_logprob"]
            acc_cube[m_int] = sc["top1_acc"]
            lo_cube[m_int] = sc["target_vs_foil_lo"]

        lp_wht = compute_exact_wht_16(lp_cube, seed=7100 + d_i * 100, n_boot=n_boot)
        acc_wht = compute_exact_wht_16(acc_cube, seed=7200 + d_i * 100, n_boot=n_boot)
        lo_wht = compute_exact_wht_16(lo_cube, seed=7300 + d_i * 100, n_boot=n_boot)
        if lp_wht["parseval_variance_error"] >= 1e-12 or acc_wht["parseval_variance_error"] >= 1e-12:
            raise RuntimeError(f"Varloop WHT Parseval error exceeded 1e-12 at K={k}")

        reg_out: dict[str, Any] = {}
        for r_i, r_name in enumerate(regimes):
            idxs = reg_indices[r_name]
            if idxs.size == 0:
                continue
            reg_out[r_name] = {
                "continuation_logprob_wht": compute_exact_wht_16(
                    lp_cube[:, idxs], seed=7400 + d_i * 100 + r_i * 20, n_boot=n_boot
                ),
                "top1_accuracy_wht": compute_exact_wht_16(
                    acc_cube[:, idxs], seed=7500 + d_i * 100 + r_i * 20, n_boot=n_boot
                ),
                "target_vs_foil_lo_wht": compute_exact_wht_16(
                    lo_cube[:, idxs], seed=7600 + d_i * 100 + r_i * 20, n_boot=n_boot
                ),
            }

        by_depth[f"K={k}"] = {
            "n_passes": k,
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
        "channel_order": list(CHANNEL_NAMES),
        "depths": sorted({int(x) for x in depths if int(x) >= 2}),
        "by_depth": by_depth,
    }


def run_varloop_per_pass_non_normal_gain(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    max_passes: int = 7,
    jacobian_per_regime: int = 12,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Measure per-pass non-normal transient amplification spectrum across `p = 1..max_passes`."""
    items_jac = _select_balanced_subset(suite, per_regime=jacobian_per_regime)
    n_jac = len(items_jac)
    jac_bsz = min(12, max(1, n_jac))
    rng_g = torch.Generator(device="cpu").manual_seed(20260702)

    pass_keys = [f"pass_{p}" for p in range(1, max_passes + 1)]
    sigma_vjp_by_pass: dict[str, list[float]] = {k: [] for k in pass_keys}
    sigma_fd_by_pass: dict[str, list[float]] = {k: [] for k in pass_keys}
    iso_gain_by_pass: dict[str, list[float]] = {k: [] for k in pass_keys}
    io_cos_by_pass: dict[str, list[float]] = {k: [] for k in pass_keys}
    rayleigh_by_pass: dict[str, list[float]] = {k: [] for k in pass_keys}

    for s in range(0, n_jac, jac_bsz):
        e = min(n_jac, s + jac_bsz)
        sub = items_jac[s:e]
        bsz = len(sub)
        rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        inp, msk, lens = ctrl._encode_right_padded(rendered)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

        with torch.no_grad():
            base = ctrl.unrolled_forward(inp, msk, n_passes=max_passes, record_all_visits=False)
            pre_b_clean, pre_h_clean = base["pre_loop"]
            c_map, r_map, pos_ids = base["causal_map"], base["rel_map"], base["position_ids"]
            exits_clean = [(b.detach(), h.detach()) for b, h in base["pass_exits"]]

        num_pre_blocks = pre_b_clean.shape[-2]
        for p_idx, st_key in enumerate(pass_keys):
            if p_idx == 0:
                in_ref = pre_h_clean.detach()
                prev_b_fixed = None
            else:
                prev_b_full, prev_h_full = exits_clean[p_idx - 1]
                prev_b_fixed = prev_b_full[..., num_pre_blocks + 1 :, :].detach()
                in_ref = prev_h_full.detach()

            def _run_single_pass_op(h_var: torch.Tensor) -> torch.Tensor:
                if p_idx == 0:
                    cur_h, cur_b = h_var, pre_b_clean
                    for l_idx in range(ctrl.loop_start, ctrl.loop_end):
                        lyr = ctrl.text_model.layers[l_idx]
                        nh = ctrl._call_decoder_layer(
                            lyr, cur_h, cur_b, c_map, r_map, pos_ids, messages=None, is_loop_call=True
                        )
                        if lyr.is_block_boundary:
                            cur_b = torch.cat([cur_b.to(cur_h.device), cur_h.unsqueeze(-2)], dim=-2)
                        cur_h = nh
                    return cur_h
                cur_h, cur_b = pre_h_clean, pre_b_clean
                assert prev_b_fixed is not None
                for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                    lyr = ctrl.text_model.layers[l_idx]
                    msgs = ctrl.text_model.message_input(prev_b_fixed, h_var, li)
                    nh = ctrl._call_decoder_layer(
                        lyr, cur_h, cur_b, c_map, r_map, pos_ids, messages=msgs, is_loop_call=True
                    )
                    if lyr.is_block_boundary:
                        cur_b = torch.cat([cur_b.to(cur_h.device), cur_h.unsqueeze(-2)], dim=-2)
                    cur_h = nh
                return cur_h

            v_init = torch.randn((bsz, ctrl.d_model), generator=rng_g, dtype=torch.float32)
            v_vec = (v_init / torch.linalg.vector_norm(v_init, dim=-1, keepdim=True).clamp(min=1e-12)).to(ctrl.device)

            h_leaf_p = in_ref.clone().requires_grad_(True)
            with torch.enable_grad():
                out_p = _run_single_pass_op(h_leaf_p)
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
                        ol = _run_single_pass_op(hl)[row_idx, col_idx].float()
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
                    yp = _run_single_pass_op(hp)[row_idx, col_idx].float()
                    ym = _run_single_pass_op(hm)[row_idx, col_idx].float()
                    ju = (yp - ym) / (2.0 * step_p)
                    sig_fd = torch.linalg.vector_norm(ju, dim=-1).clamp(min=1e-12)
                    v_vec = ju / sig_fd.unsqueeze(-1)

            hl_fin = in_ref.clone().requires_grad_(True)
            with torch.enable_grad():
                ol_fin = _run_single_pass_op(hl_fin)[row_idx, col_idx].float()
                g_fin = torch.autograd.grad((ol_fin * v_vec).sum(), hl_fin, retain_graph=False)[0][
                    row_idx, col_idx
                ].float()
            sig_vjp_cert = torch.linalg.vector_norm(g_fin, dim=-1).clamp(min=1e-12)
            u_fin = g_fin / sig_vjp_cert.unsqueeze(-1)
            io_cos = (u_fin * v_vec).sum(dim=-1).abs()
            rayleigh_q = sig_vjp_cert * io_cos

            sigma_vjp_by_pass[st_key].extend(sig_vjp_cert.cpu().tolist())
            sigma_fd_by_pass[st_key].extend(sig_fd.cpu().tolist())
            iso_gain_by_pass[st_key].extend(iso_gain_p.cpu().tolist())
            io_cos_by_pass[st_key].extend(io_cos.cpu().tolist())
            rayleigh_by_pass[st_key].extend(rayleigh_q.cpu().tolist())

    by_pass: dict[str, Any] = {}
    for idx_s, st_key in enumerate(pass_keys):
        sv_ci = bootstrap_mean_ci(sigma_vjp_by_pass[st_key], n_boot=n_boot, seed=8100 + idx_s)
        sf_ci = bootstrap_mean_ci(sigma_fd_by_pass[st_key], n_boot=n_boot, seed=8200 + idx_s)
        ig_ci = bootstrap_mean_ci(iso_gain_by_pass[st_key], n_boot=n_boot, seed=8300 + idx_s)
        cos_ci = bootstrap_mean_ci(io_cos_by_pass[st_key], n_boot=n_boot, seed=8400 + idx_s)
        rq_ci = bootstrap_mean_ci(rayleigh_by_pass[st_key], n_boot=n_boot, seed=8500 + idx_s)
        aniso = np.asarray(sigma_vjp_by_pass[st_key]) / np.maximum(np.asarray(iso_gain_by_pass[st_key]), 1e-8)
        non_norm_amp = np.asarray(sigma_vjp_by_pass[st_key]) / np.maximum(np.asarray(rayleigh_by_pass[st_key]), 1e-4)
        by_pass[st_key] = {
            "sigma_max_vjp": {"mean": round(sv_ci[0], 4), "ci95": [round(sv_ci[1], 4), round(sv_ci[2], 4)]},
            "sigma_max_fd": {"mean": round(sf_ci[0], 4), "ci95": [round(sf_ci[1], 4), round(sf_ci[2], 4)]},
            "isotropic_rms_gain": {"mean": round(ig_ci[0], 4), "ci95": [round(ig_ci[1], 4), round(ig_ci[2], 4)]},
            "anisotropy_ratio_sigma_over_iso": round(float(np.mean(aniso)), 4),
            "singular_io_cosine": {"mean": round(cos_ci[0], 4), "ci95": [round(cos_ci[1], 4), round(cos_ci[2], 4)]},
            "normal_rayleigh_quotient": {"mean": round(rq_ci[0], 4), "ci95": [round(rq_ci[1], 4), round(rq_ci[2], 4)]},
            "non_normal_transient_ratio": round(float(np.mean(non_norm_amp)), 4),
        }

    return {
        "n_jacobian_items": n_jac,
        "max_passes": max_passes,
        "by_pass": by_pass,
    }


def run_varloop_fisher_and_commitment(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    depths: Sequence[int] = (5, 7),
    fisher_per_regime: int = 20,
    n_rademacher: int = 6,
    fisher_bsz: int = 8,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Compute multi-K Hutchinson VJP Fisher pullback, Loop->Coda spike, and Hop-1/Hop-2 commitment."""
    items_60 = _select_balanced_subset(suite, per_regime=fisher_per_regime)
    n60 = len(items_60)
    mid_loop_l = ctrl.loop_start + max(0, (ctrl.loop_end - ctrl.loop_start) // 2 - 1)
    end_loop_l = ctrl.loop_end - 1
    stem_end_l = ctrl.loop_start - 1

    by_depth: dict[str, Any] = {}
    for d_i, k_depth in enumerate(sorted({int(x) for x in depths if int(x) >= 1})):
        rng_f = torch.Generator(device="cpu").manual_seed(20260703 + d_i * 17)
        target_visit_keys = [f"stem_L{stem_end_l}"]
        for p_num in range(1, k_depth + 1):
            if mid_loop_l != end_loop_l:
                target_visit_keys.append(f"p{p_num}_L{mid_loop_l}")
            target_visit_keys.append(f"p{p_num}_L{end_loop_l}")
        target_visit_keys.extend(f"coda_L{l_i}" for l_i in range(ctrl.loop_end, ctrl.n_layers))

        n_ckpts = len(target_visit_keys)
        fisher_trace_mat = np.zeros((n_ckpts, n60), dtype=np.float64)
        fisher_rel_mat = np.zeros((n_ckpts, n60), dtype=np.float64)

        traj_stages = ["stem", *[f"pass_{p_num}" for p_num in range(1, k_depth + 1)]]
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
            rendered = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
            inp, msk, lens = ctrl._encode_right_padded(rendered)
            row_idx = torch.arange(bsz, device=ctrl.device)
            col_idx = torch.tensor([l_i - 1 for l_i in lens], dtype=torch.long, device=ctrl.device)

            with torch.enable_grad():
                emb = ctrl.text_model.embed_tokens(inp).detach().requires_grad_(True)
                if getattr(ctrl.text_model, "_is_mock_tiny", False) and msk is not None:
                    emb_run = emb * msk.unsqueeze(-1).to(emb.dtype)
                else:
                    emb_run = emb
                pos_ids, c_map, r_map = ctrl._build_masks_and_distances(emb_run, msk)

                h_cur = emb_run
                b_cur = emb_run.new_zeros(*emb_run.shape[:-1], 0, emb_run.shape[-1])
                visit_dict: dict[str, torch.Tensor] = {}
                pass_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

                for l_idx in range(0, ctrl.loop_start):
                    lyr = ctrl.text_model.layers[l_idx]
                    nh = ctrl._call_decoder_layer(lyr, h_cur, b_cur, c_map, r_map, pos_ids, is_loop_call=False)
                    if lyr.is_block_boundary:
                        b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                    h_cur = nh
                    visit_dict[f"stem_L{l_idx}"] = h_cur
                pre_h, pre_b = h_cur, b_cur
                stem_pair = (b_cur.detach(), h_cur.detach())
                num_pre = pre_b.shape[-2]

                for l_idx in range(ctrl.loop_start, ctrl.loop_end):
                    lyr = ctrl.text_model.layers[l_idx]
                    nh = ctrl._call_decoder_layer(
                        lyr, h_cur, b_cur, c_map, r_map, pos_ids, messages=None, is_loop_call=True
                    )
                    if lyr.is_block_boundary:
                        b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                    h_cur = nh
                    visit_dict[f"p1_L{l_idx}"] = h_cur
                pass_pairs.append((b_cur.detach(), h_cur.detach()))

                for loop_idx in range(1, k_depth):
                    prev_b = b_cur[..., num_pre + 1 :, :]
                    prev_h = h_cur
                    h_cur, b_cur = pre_h, pre_b
                    for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                        lyr = ctrl.text_model.layers[l_idx]
                        msgs = ctrl.text_model.message_input(prev_b, prev_h, li)
                        nh = ctrl._call_decoder_layer(
                            lyr, h_cur, b_cur, c_map, r_map, pos_ids, messages=msgs, is_loop_call=True
                        )
                        if lyr.is_block_boundary:
                            b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                        h_cur = nh
                        visit_dict[f"p{loop_idx + 1}_L{l_idx}"] = h_cur
                    pass_pairs.append((b_cur.detach(), h_cur.detach()))

                for l_idx in range(ctrl.loop_end, ctrl.n_layers):
                    lyr = ctrl.text_model.layers[l_idx]
                    nh = ctrl._call_decoder_layer(lyr, h_cur, b_cur, c_map, r_map, pos_ids, is_loop_call=False)
                    if lyr.is_block_boundary:
                        b_cur = torch.cat([b_cur.to(h_cur.device), h_cur.unsqueeze(-2)], dim=-2)
                    h_cur = nh
                    visit_dict[f"coda_L{l_idx}"] = h_cur

                fin_m = (
                    ctrl.text_model.final_input(b_cur, h_cur)[0]
                    if hasattr(ctrl.text_model, "final_input")
                    else h_cur
                )
                fin_n = ctrl.text_model.norm(fin_m).to(ctrl.lm_head.weight.dtype)
                logits_last = ctrl.logit_softcap(ctrl.lm_head(fin_n[row_idx, col_idx]).float())
                probs_last = F.softmax(logits_last, dim=-1).detach()
                sqrt_p = torch.sqrt(probs_last.clamp(min=1e-30))

                ckpt_tensors = [visit_dict[k] for k in target_visit_keys]
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
                b_stem_pad = stem_pair[0]
                while b_stem_pad.shape[-2] < pass_pairs[0][0].shape[-2]:
                    b_stem_pad = torch.cat([b_stem_pad, stem_pair[1].unsqueeze(-2)], dim=-2)
                all_pairs = [(b_stem_pad, stem_pair[1]), *pass_pairs]
                tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
                h1_sets = [first_token_variant_ids(ctrl.tokenizer, it.hop1_bridge) for it in sub]

                for st_i, (bp, hp) in enumerate(all_pairs):
                    fin_st, _ = ctrl.run_coda_from_state(
                        bp, hp, c_map, r_map, pos_ids, stop_coda_layer=ctrl.n_layers - 1
                    )
                    lg_st = ctrl.logit_softcap(
                        ctrl.lm_head(fin_st[row_idx, col_idx].to(ctrl.lm_head.weight.dtype)).float()
                    )
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
        stage_rel_mean: dict[str, float] = {}
        stage_tr_mean: dict[str, float] = {}
        for c_i, v_key in enumerate(target_visit_keys):
            tr_ci = bootstrap_mean_ci(fisher_trace_mat[c_i], n_boot=n_boot, seed=8600 + d_i * 100 + c_i)
            rl_ci = bootstrap_mean_ci(fisher_rel_mat[c_i], n_boot=n_boot, seed=8700 + d_i * 100 + c_i)
            stage_tr_mean[v_key] = float(tr_ci[0])
            stage_rel_mean[v_key] = float(rl_ci[0])
            fisher_checkpoints.append(
                {
                    "stage_key": v_key,
                    "fisher_trace": {"mean": round(tr_ci[0], 5), "ci95": [round(tr_ci[1], 5), round(tr_ci[2], 5)]},
                    "scale_normalized_fisher_kl_curvature": {
                        "mean": round(rl_ci[0], 5),
                        "ci95": [round(rl_ci[1], 5), round(rl_ci[2], 5)],
                    },
                }
            )

        loop_exit_key = f"p{k_depth}_L{end_loop_l}"
        coda_entry_key = f"coda_L{ctrl.loop_end}"
        spike_trace_ratio = stage_tr_mean.get(coda_entry_key, 0.0) / max(stage_tr_mean.get(loop_exit_key, 0.0), 1e-12)
        spike_rel_ratio = stage_rel_mean.get(coda_entry_key, 0.0) / max(stage_rel_mean.get(loop_exit_key, 0.0), 1e-12)

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

        by_depth[f"K={k_depth}"] = {
            "n_passes": k_depth,
            "n_fisher_items": n60,
            "loop_to_coda_fisher_spike": {
                "loop_exit_key": loop_exit_key,
                "coda_entry_key": coda_entry_key,
                "trace_ratio": round(float(spike_trace_ratio), 4),
                "scale_normalized_ratio": round(float(spike_rel_ratio), 4),
            },
            "fisher_pullback_trajectory": fisher_checkpoints,
            "commitment_horizon": commitment_by_regime,
        }

    return {
        "depths": sorted({int(x) for x in depths if int(x) >= 1}),
        "by_depth": by_depth,
    }


@_no_grad()
def run_varloop_somatic_reset_hysteresis(
    ctrl: BiophysicalInklingController,
    suite: dict[str, Any],
    depths: Sequence[int] = (3, 5, 7),
    items_per_regime: int = 20,
    batch_size: int = 20,
    foil_betas: Sequence[float] = (0.0, 0.25, 0.50, 0.75, 1.00, 1.25),
) -> dict[str, Any]:
    """Pass-1-only foil replacement entering Pass 2, tracking somatic-reset foil decay across P2..PK."""
    items_60 = _select_balanced_subset(suite, per_regime=items_per_regime)
    n60 = len(items_60)
    sorted_depths = sorted({int(k) for k in depths if int(k) >= 2})
    max_k = max(sorted_depths) if sorted_depths else 7
    tracked_passes = sorted({2, *sorted_depths})
    betas_tuple = tuple(float(b) for b in foil_betas)
    n_b = len(betas_tuple)

    target_prob_by_pass = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_passes}
    foil_prob_by_pass = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_passes}
    coda_acc_by_pass = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_passes}
    coda_cap_by_pass = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_passes}
    coda_lo_by_pass = {p: np.zeros((n_b, n60), dtype=np.float64) for p in tracked_passes}

    for s in range(0, n60, batch_size):
        e = min(n60, s + batch_size)
        sub = items_60[s:e]
        bsz = len(sub)
        clean_prompts = [ctrl.format_0cot_prompt(it.user_prompt) for it in sub]
        foil_prompts = [ctrl.format_0cot_prompt(it.foil_user_prompt) for it in sub]

        inp_c, msk_c, lens_c = ctrl._encode_right_padded(clean_prompts)
        inp_f, msk_f, lens_f = ctrl._encode_right_padded(foil_prompts)
        row_idx = torch.arange(bsz, device=ctrl.device)
        col_c = torch.tensor([l_i - 1 for l_i in lens_c], dtype=torch.long, device=ctrl.device)
        col_f = torch.tensor([l_i - 1 for l_i in lens_f], dtype=torch.long, device=ctrl.device)

        tgt_sets = [first_token_variant_ids(ctrl.tokenizer, it.target_answer) for it in sub]
        foil_sets = [first_token_variant_ids(ctrl.tokenizer, it.foil_answer) for it in sub]

        clean_p1 = ctrl.unrolled_forward(inp_c, msk_c, n_passes=1)
        foil_p1 = ctrl.unrolled_forward(inp_f, msk_f, n_passes=1)

        pre_b_c, pre_h_c = clean_p1["pre_loop"]
        num_pre = pre_b_c.shape[-2]
        c_map, r_map, pos_ids = clean_p1["causal_map"], clean_p1["rel_map"], clean_p1["position_ids"]

        p1_b_clean_full, p1_h_clean = clean_p1["pass_exits"][0]
        p1_b_clean = p1_b_clean_full[..., num_pre + 1 :, :]
        p1_b_foil_full, p1_h_foil = foil_p1["pass_exits"][0]
        p1_b_foil_last = p1_b_foil_full[..., num_pre + 1 :, :][row_idx, col_f]
        p1_h_foil_last = p1_h_foil[row_idx, col_f]

        c_h_tok = p1_h_clean[row_idx, col_c].float()
        f_h_tok = p1_h_foil_last.float()
        c_b_tok = p1_b_clean[row_idx, col_c].float()
        f_b_tok = p1_b_foil_last.float()

        for b_i, beta in enumerate(betas_tuple):
            prev_b = p1_b_clean.clone()
            prev_h = p1_h_clean.clone()
            prev_h[row_idx, col_c] = ((1.0 - beta) * c_h_tok + beta * f_h_tok).to(p1_h_clean.dtype)
            if prev_b.shape[-2] > 0 and f_b_tok.shape[-2] == prev_b.shape[-2]:
                prev_b[row_idx, col_c] = ((1.0 - beta) * c_b_tok + beta * f_b_tok).to(p1_b_clean.dtype)

            for p_num in range(2, max_k + 1):
                cur_h, cur_b = pre_h_c, pre_b_c
                for li, l_idx in enumerate(range(ctrl.loop_start, ctrl.loop_end)):
                    lyr = ctrl.text_model.layers[l_idx]
                    msgs = ctrl.text_model.message_input(prev_b, prev_h, li)
                    nh = ctrl._call_decoder_layer(
                        lyr, cur_h, cur_b, c_map, r_map, pos_ids, messages=msgs, is_loop_call=True
                    )
                    if lyr.is_block_boundary:
                        cur_b = torch.cat([cur_b.to(cur_h.device), cur_h.unsqueeze(-2)], dim=-2)
                    cur_h = nh

                if p_num in target_prob_by_pass:
                    fin_p, _ = ctrl.run_coda_from_state(cur_b, cur_h, c_map, r_map, pos_ids)
                    lg_p = ctrl.logit_softcap(
                        ctrl.lm_head(fin_p[row_idx, col_c].to(ctrl.lm_head.weight.dtype)).float()
                    )
                    pr_p = F.softmax(lg_p, dim=-1)
                    top_p = lg_p.argmax(dim=-1).tolist()
                    for i in range(bsz):
                        t_ids = tgt_sets[i]
                        fl_ids = foil_sets[i]
                        pt = float(pr_p[i, t_ids].sum().item())
                        pf = float(pr_p[i, fl_ids].sum().item())
                        target_prob_by_pass[p_num][b_i, s + i] = pt
                        foil_prob_by_pass[p_num][b_i, s + i] = pf
                        coda_acc_by_pass[p_num][b_i, s + i] = 1.0 if top_p[i] in t_ids else 0.0
                        coda_cap_by_pass[p_num][b_i, s + i] = 1.0 if top_p[i] in fl_ids else 0.0
                        coda_lo_by_pass[p_num][b_i, s + i] = math.log(max(pt, 1e-30)) - math.log(max(pf, 1e-30))

                prev_b = cur_b[..., num_pre + 1 :, :]
                prev_h = cur_h

    by_depth: dict[str, Any] = {}
    for k_depth in sorted_depths:
        rows = []
        for b_i, beta in enumerate(betas_tuple):
            rows.append(
                {
                    "beta": beta,
                    "pass2_foil_prob": round(float(np.mean(foil_prob_by_pass[2][b_i])), 5),
                    "pass_k_target_prob": round(float(np.mean(target_prob_by_pass[k_depth][b_i])), 5),
                    "pass_k_foil_prob": round(float(np.mean(foil_prob_by_pass[k_depth][b_i])), 5),
                    "coda_target_top1_acc": round(float(np.mean(coda_acc_by_pass[k_depth][b_i])), 4),
                    "coda_foil_capture_rate": round(float(np.mean(coda_cap_by_pass[k_depth][b_i])), 4),
                    "coda_target_minus_foil_lo": round(float(np.mean(coda_lo_by_pass[k_depth][b_i])), 4),
                }
            )
        summary = summarize_hysteresis_curve(
            betas_tuple,
            [float(np.mean(foil_prob_by_pass[2][i])) for i in range(n_b)],
            [float(np.mean(foil_prob_by_pass[k_depth][i])) for i in range(n_b)],
            [float(np.mean(coda_acc_by_pass[k_depth][i])) for i in range(n_b)],
            [float(np.mean(coda_cap_by_pass[k_depth][i])) for i in range(n_b)],
        )
        by_depth[f"K={k_depth}"] = {
            "n_passes": k_depth,
            "curve": rows,
            "hysteresis_summary": summary,
        }

    foil_decay_by_beta: list[dict[str, Any]] = []
    for b_i, beta in enumerate(betas_tuple):
        entry: dict[str, Any] = {"beta": beta}
        for p_num in tracked_passes:
            entry[f"pass_{p_num}_foil_prob"] = round(float(np.mean(foil_prob_by_pass[p_num][b_i])), 5)
            entry[f"pass_{p_num}_target_acc"] = round(float(np.mean(coda_acc_by_pass[p_num][b_i])), 4)
        foil_decay_by_beta.append(entry)

    return {
        "n_items": n60,
        "depths": sorted_depths,
        "tracked_passes": tracked_passes,
        "betas": list(betas_tuple),
        "foil_decay_trajectory": foil_decay_by_beta,
        "by_depth": by_depth,
    }


def run_phase7_varloop_suite(
    controller: BiophysicalInklingController,
    tokenizer: Any = None,
    stimuli: dict[str, Any] | None = None,
    depths: Sequence[int] = (1, 2, 3, 5, 7),
    wht_depths: Sequence[int] = (2, 3, 5, 7),
    fisher_depths: Sequence[int] = (5, 7),
    hysteresis_depths: Sequence[int] = (3, 5, 7),
    batch_size: int = 32,
    jacobian_per_regime: int = 12,
    fisher_per_regime: int = 20,
    n_rademacher: int = 6,
    fisher_bsz: int = 8,
    hysteresis_per_regime: int = 20,
    n_boot: int = 4000,
    checkpoint_path: Path | str | None = None,
) -> dict[str, Any]:
    """Run Phase 7 Experiment Block B1 (Varloop K=1..7) with atomic resumable JSON checkpoints."""
    _ = tokenizer
    t0 = time.perf_counter()
    suite = stimuli if stimuli is not None else build_phase6_stimulus_suite()
    ckpt = Path(checkpoint_path) if checkpoint_path is not None else None

    if ckpt is not None and ckpt.is_file():
        payload = json.loads(ckpt.read_text(encoding="utf-8"))
        payload["completed"] = False
    else:
        payload = {
            "phase": "phase7_varloop_biophysical_depth",
            "model_label": controller.model_label,
            "depths": [int(d) for d in depths],
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "completed": False,
        }

    def _save_ckpt() -> None:
        if ckpt is not None:
            _atomic_save_json(ckpt, payload)

    sorted_depths = sorted({int(d) for d in depths if int(d) >= 1})
    preflight_depths = [k for k in (1, 3, 5, 7) if k in sorted_depths] or sorted_depths

    if "preflight_equivalence" not in payload:
        payload["preflight_equivalence"] = run_varloop_preflight_equivalence(controller, depths=preflight_depths)
        _save_ckpt()

    if "wht_depth_trajectory" not in payload:
        eff_wht_depths = [int(k) for k in wht_depths if int(k) >= 2]
        payload["wht_depth_trajectory"] = run_varloop_wht_depth_trajectory(
            controller,
            suite,
            depths=eff_wht_depths,
            batch_size=batch_size,
            n_boot=n_boot,
        )
        _save_ckpt()

    if "per_pass_non_normal_gain" not in payload:
        payload["per_pass_non_normal_gain"] = run_varloop_per_pass_non_normal_gain(
            controller,
            suite,
            max_passes=max(sorted_depths) if sorted_depths else 7,
            jacobian_per_regime=jacobian_per_regime,
            n_boot=n_boot,
        )
        _save_ckpt()

    if "fisher_and_commitment" not in payload:
        payload["fisher_and_commitment"] = run_varloop_fisher_and_commitment(
            controller,
            suite,
            depths=fisher_depths,
            fisher_per_regime=fisher_per_regime,
            n_rademacher=n_rademacher,
            fisher_bsz=fisher_bsz,
            n_boot=n_boot,
        )
        _save_ckpt()

    if "somatic_reset_hysteresis" not in payload:
        payload["somatic_reset_hysteresis"] = run_varloop_somatic_reset_hysteresis(
            controller,
            suite,
            depths=hysteresis_depths,
            items_per_regime=hysteresis_per_regime,
            batch_size=min(20, batch_size),
        )
        _save_ckpt()

    payload["completed"] = True
    payload["runtime_seconds"] = round(time.perf_counter() - t0, 2)
    _save_ckpt()
    return payload
