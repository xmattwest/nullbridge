"""Self-contained loader, CompactLinear autograd, Procrustes Nullbridge, and Behavioral LoRA controller for `Gemma-4-Loop-Compact`.

Loads Phase 9's standalone `gemma4_loop_compact_pkg` (`23.08 GB` VRAM footprint) without importing
external project packages, and supports toggling:
  1. Zero-FLOP offline bridge null-space weight folding (`R_safe = R_proc (I - Q Q^T)`, `A_safe = A (I - Q Q^T)`).
  2. Multi-bank Behavioral Alignment LoRAs (`coda_lora` on `L43..L59` and `full_lora` on `L24..L59`).
  3. Funnel boundary subspace projections (`L24` entry + `L42` exit, `0` internal `L24..L41` hooks)
     vs. 19-layer internal full-funnel clamping (`L24..L42` across all `K` passes).
  4. Recurrent step-gate overrides (`gate_override`) for high-coupling stress tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import gc
import json
import math
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np


def resolve_decoder_layers(model: Any) -> Sequence[Any]:
    """Locate the text decoder block list across standard and multimodal-unified HF wrappers."""
    cfg = getattr(model, "config", None)
    text_cfg = getattr(cfg, "text_config", None) or cfg
    expected_n = getattr(text_cfg, "num_hidden_layers", None)

    candidate_paths = (
        ("model", "language_model", "layers"),
        ("model", "language_model", "model", "layers"),
        ("language_model", "model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("transformer", "h"),
        ("layers",),
    )
    for path in candidate_paths:
        cur = model
        ok = True
        for attr in path:
            if hasattr(cur, attr):
                cur = getattr(cur, attr)
            else:
                ok = False
                break
        if ok and hasattr(cur, "__len__") and len(cur) >= 2:
            if expected_n is None or len(cur) == int(expected_n):
                return cur

    if hasattr(model, "named_modules"):
        for name, mod in model.named_modules():
            if (
                name.endswith("layers")
                and "vision" not in name
                and "audio" not in name
                and hasattr(mod, "__len__")
                and len(mod) >= 2
                and (expected_n is None or len(mod) == int(expected_n))
            ):
                return mod

    raise AttributeError(f"Could not locate decoder layers on {type(model).__name__}")


def _single_token_candidates(tokenizer: Any, letter: str) -> list[int]:
    ids: list[int] = []
    for surface in (letter, f" {letter}"):
        toks = tokenizer.encode(surface, add_special_tokens=False)
        if len(toks) == 1 and int(toks[0]) not in ids:
            ids.append(int(toks[0]))
    if not ids:
        toks = tokenizer.encode(letter, add_special_tokens=False)
        if not toks:
            raise ValueError(f"Empty tokenization for {letter!r}")
        ids.append(int(toks[-1]))
    return ids


def format_chat_prompt(tokenizer: Any, user_content: str, assistant_prefill: str = "") -> str:
    """Format single-turn prompt with `<thought>` scratchpad disabled (`0-CoT`)."""
    if hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": user_content}]
        for extra_kwargs in ({"enable_thinking": False}, {"thinking": False}, {}):
            try:
                rendered = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **extra_kwargs,
                )
                return str(rendered) + assistant_prefill
            except TypeError:
                continue
    return f"User: {user_content}\nAssistant: {assistant_prefill}"


def build_choice_user_prompt(scenario: str, option_a: str, option_b: str) -> str:
    return (
        f"{scenario}\n\n"
        f"Options:\n"
        f"A) {option_a}\n"
        f"B) {option_b}\n\n"
        f"Reply with only the single letter A or B."
    )


def build_scenario_reading_prompt(scenario: str) -> str:
    return (
        "Read the following situation carefully and consider your position in it:\n\n"
        f"{scenario}"
    )


@dataclass(frozen=True)
class ChoiceScore:
    target_log_odds: float
    target_prob: float
    order_ab_log_odds: float
    order_ba_log_odds: float
    ab_mass: float


class CompactHarness:
    """Thin wrapper around a causal LM supporting batched residual extraction & counterbalanced A/B log-odds."""

    def __init__(self, model: Any, tokenizer: Any, model_id: str, device: str | None = None) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.layers = resolve_decoder_layers(model)
        self.n_layers = len(self.layers)

        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = "left"

        if device is not None:
            self.input_device = torch.device(device)
        else:
            self.input_device = next(model.parameters()).device

        self.a_token_ids = _single_token_candidates(tokenizer, "A")
        self.b_token_ids = _single_token_candidates(tokenizer, "B")

    @classmethod
    def from_pretrained(
        cls,
        model_path_or_id: str,
        dtype: str = "bfloat16",
        device_map: str = "cuda",
        model_id_label: str | None = None,
    ) -> "CompactHarness":
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        torch_dtype = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[dtype.lower()]

        if device_map == "cuda" and not torch.cuda.is_available():
            device_map = "auto"

        cfg = AutoConfig.from_pretrained(model_path_or_id)
        tokenizer = AutoTokenizer.from_pretrained(model_path_or_id)

        loader_classes: list[Any] = []
        for arch in (getattr(cfg, "architectures", None) or []):
            arch_cls = getattr(transformers, arch, None)
            if arch_cls is not None and arch_cls not in loader_classes:
                loader_classes.append(arch_cls)
        if AutoModelForCausalLM not in loader_classes:
            loader_classes.append(AutoModelForCausalLM)
        for fb_name in ("AutoModelForImageTextToText", "AutoModelForMultimodalLM"):
            if hasattr(transformers, fb_name):
                fb = getattr(transformers, fb_name)
                if fb not in loader_classes:
                    loader_classes.append(fb)

        model = None
        errs: list[str] = []
        for loader_cls in loader_classes:
            for d_kw in ({"dtype": torch_dtype}, {"torch_dtype": torch_dtype}):
                try:
                    model = loader_cls.from_pretrained(model_path_or_id, device_map=device_map, **d_kw)
                    break
                except TypeError:
                    continue
                except Exception as exc:
                    errs.append(f"{loader_cls.__name__}: {exc}")
                    break
            if model is not None:
                break

        if model is None:
            raise RuntimeError(f"Failed to load {model_path_or_id}: {'; '.join(errs)}")

        model.eval()
        torch.set_grad_enabled(False)
        return cls(model=model, tokenizer=tokenizer, model_id=model_id_label or model_path_or_id)

    def resolve_stop_token_ids(self) -> list[int]:
        stop_ids: list[int] = []

        def _add(v: Any) -> None:
            if isinstance(v, int) and v >= 0 and v not in stop_ids:
                stop_ids.append(v)
            elif isinstance(v, (list, tuple, set)):
                for x in v:
                    _add(x)

        _add(getattr(self.tokenizer, "eos_token_id", None))
        gen_cfg = getattr(self.model, "generation_config", None)
        _add(getattr(gen_cfg, "eos_token_id", None))
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            for sym in ("<turn|>", "<end_of_turn>", "<|im_end|>", "<|end|>", "<|eot_id|>"):
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(sym)
                except Exception:
                    continue
                if isinstance(tid, int) and tid >= 0 and tid != unk_id:
                    _add(tid)
        return stop_ids

    def tokenize_rendered_batch(self, prompts: Sequence[str], max_length: int = 1536) -> dict[str, Any]:
        has_chat_tmpl = bool(getattr(self.tokenizer, "chat_template", None))
        enc = self.tokenizer(
            list(prompts),
            add_special_tokens=not has_chat_tmpl,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        return {k: v.to(self.input_device) for k, v in enc.items() if k in ("input_ids", "attention_mask")}

    def forward_last_logits(self, enc: dict[str, Any]) -> Any:
        try:
            return self.model(**enc, use_cache=False, logits_to_keep=1)
        except TypeError:
            return self.model(**enc, use_cache=False)

    def extract_last_token_states(self, prompts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        """Return `[N, L, D]` float32 residual states at the final prompt token."""
        torch = self.torch
        batches: list[np.ndarray] = []
        for start in range(0, len(prompts), batch_size):
            enc = self.tokenize_rendered_batch(prompts[start : start + batch_size])
            captured: dict[int, Any] = {}
            handles = []

            def _mk_cap(idx: int):
                def _cap(_m: Any, _i: Any, out: Any) -> None:
                    h = out[0] if isinstance(out, tuple) else out
                    captured[idx] = h[:, -1, :].detach().to(torch.float32).cpu()
                return _cap

            for ell in range(self.n_layers):
                handles.append(self.layers[ell].register_forward_hook(_mk_cap(ell)))
            try:
                with torch.no_grad():
                    self.forward_last_logits(enc)
            finally:
                for h in handles:
                    h.remove()

            stacked = torch.stack([captured[ell] for ell in range(self.n_layers)], dim=1).numpy()
            batches.append(stacked)
        return np.concatenate(batches, axis=0)

    def score_counterbalanced_choices(
        self,
        scenarios: Sequence[str],
        target_options: Sequence[str],
        reference_options: Sequence[str],
        batch_size: int = 32,
    ) -> list[ChoiceScore]:
        """Score binary choices across both `(A=target, B=ref)` and `(A=ref, B=target)` orderings (`0` generated tokens)."""
        torch = self.torch
        n = len(scenarios)
        prompts: list[str] = []
        for scen, tgt, ref in zip(scenarios, target_options, reference_options):
            prompts.append(format_chat_prompt(self.tokenizer, build_choice_user_prompt(scen, tgt, ref)))
            prompts.append(format_chat_prompt(self.tokenizer, build_choice_user_prompt(scen, ref, tgt)))

        lpa_all: list[float] = []
        lpb_all: list[float] = []
        mass_all: list[float] = []

        for start in range(0, len(prompts), batch_size):
            enc = self.tokenize_rendered_batch(prompts[start : start + batch_size])
            with torch.no_grad():
                out = self.forward_last_logits(enc)
            logits = out.logits[:, -1, :].to(torch.float32)
            lp = torch.log_softmax(logits, dim=-1)
            lpa = torch.logsumexp(lp[:, self.a_token_ids], dim=-1).cpu().numpy()
            lpb = torch.logsumexp(lp[:, self.b_token_ids], dim=-1).cpu().numpy()
            mass = np.clip(np.exp(lpa) + np.exp(lpb), 0.0, 1.0)
            lpa_all.extend(float(x) for x in lpa)
            lpb_all.extend(float(x) for x in lpb)
            mass_all.extend(float(x) for x in mass)

        scores: list[ChoiceScore] = []
        for i in range(n):
            lo_ab = lpa_all[2 * i] - lpb_all[2 * i]
            lo_ba = lpb_all[2 * i + 1] - lpa_all[2 * i + 1]
            mean_lo = 0.5 * (lo_ab + lo_ba)
            prob = float(1.0 / (1.0 + np.exp(-np.clip(mean_lo, -40.0, 40.0))))
            scores.append(
                ChoiceScore(
                    target_log_odds=float(mean_lo),
                    target_prob=prob,
                    order_ab_log_odds=float(lo_ab),
                    order_ba_log_odds=float(lo_ba),
                    ab_mass=float(0.5 * (mass_all[2 * i] + mass_all[2 * i + 1])),
                )
            )
        return scores


# ---------------------------------------------------------------------------
# Packed INT4 / INT8 `CompactLinear` with Loop-LoRA + Multi-Bank Behavioral LoRA
# ---------------------------------------------------------------------------

def _choose_group_size(in_features: int, preferred: int = 64) -> int:
    for g in (preferred, 128, 32, 16, 8, 4, 2, 1):
        if in_features % g == 0 and (g % 2 == 0 or g == 1):
            return g
    return 1


def pack_weight_tensor(torch: Any, weight: Any, bits: int = 4, preferred_group: int = 64) -> dict[str, Any]:
    w = weight.detach().to(torch.float32)
    out_f, in_f = int(w.shape[0]), int(w.shape[1])

    if bits == 4 and in_f % 2 == 0:
        g = _choose_group_size(in_f, preferred_group)
        if g >= 2:
            n_g = in_f // g
            wg = w.reshape(out_f, n_g, g)
            w_min = wg.amin(dim=-1, keepdim=True)
            w_max = wg.amax(dim=-1, keepdim=True)
            scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
            q = torch.round((wg - w_min) / scale).clamp(0.0, 15.0).to(torch.uint8)
            q_packed = ((q[..., 0::2] << 4) | q[..., 1::2]).reshape(out_f, in_f // 2).contiguous()
            return {
                "bits": 4,
                "group_size": g,
                "out_features": out_f,
                "in_features": in_f,
                "qweight": q_packed,
                "scales": scale.squeeze(-1).to(torch.bfloat16).contiguous(),
                "zeros": w_min.squeeze(-1).to(torch.bfloat16).contiguous(),
            }

    g = _choose_group_size(in_f, preferred_group)
    n_g = in_f // g
    wg = w.reshape(out_f, n_g, g)
    amax = wg.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / 127.0
    q8 = torch.round(wg / scale).clamp(-127.0, 127.0).to(torch.int8).reshape(out_f, in_f).contiguous()
    return {
        "bits": 8,
        "group_size": g,
        "out_features": out_f,
        "in_features": in_f,
        "qweight": q8,
        "scales": scale.squeeze(-1).to(torch.bfloat16).contiguous(),
        "zeros": None,
    }


def dequantize_packed_weight(
    torch: Any,
    qweight: Any,
    scales: Any,
    zeros: Any | None,
    bits: int,
    out_features: int,
    in_features: int,
    group_size: int,
    target_dtype: Any,
) -> Any:
    n_g = in_features // group_size
    sc = scales.to(target_dtype).unsqueeze(-1)
    if bits == 4:
        hi = (qweight >> 4).to(target_dtype)
        lo = (qweight & 0x0F).to(target_dtype)
        qu = torch.stack((hi, lo), dim=-1).reshape(out_features, n_g, group_size)
        zr = zeros.to(target_dtype).unsqueeze(-1)
        return (qu * sc + zr).reshape(out_features, in_features)
    qu8 = qweight.to(target_dtype).reshape(out_features, n_g, group_size)
    return (qu8 * sc).reshape(out_features, in_features)


def build_compact_linear_class(torch: Any) -> Any:
    nn = torch.nn
    F = torch.nn.functional

    class _CompactLinearFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx: Any, x: Any, qweight: Any, scales: Any, zeros: Any, bias: Any, bits: int, out_f: int, in_f: int, g_sz: int) -> Any:
            if zeros is not None:
                ctx.save_for_backward(qweight, scales, zeros)
                ctx.has_zeros = True
            else:
                ctx.save_for_backward(qweight, scales)
                ctx.has_zeros = False
            ctx.bits = bits
            ctx.out_f = out_f
            ctx.in_f = in_f
            ctx.g_sz = g_sz
            ctx.has_bias = bias is not None
            w_deq = dequantize_packed_weight(torch, qweight, scales, zeros, bits, out_f, in_f, g_sz, x.dtype)
            b = bias.to(x.dtype) if bias is not None else None
            return F.linear(x, w_deq, b)

        @staticmethod
        def backward(ctx: Any, grad_out: Any) -> tuple[Any, ...]:
            if ctx.has_zeros:
                qweight, scales, zeros = ctx.saved_tensors
            else:
                qweight, scales = ctx.saved_tensors
                zeros = None
            w_deq = dequantize_packed_weight(
                torch, qweight, scales, zeros, ctx.bits, ctx.out_f, ctx.out_f if False else ctx.in_f, ctx.g_sz, grad_out.dtype
            )
            grad_x = torch.matmul(grad_out, w_deq)
            grad_b = grad_out.sum(dim=tuple(range(grad_out.ndim - 1))) if ctx.has_bias else None
            return grad_x, None, None, None, grad_b, None, None, None, None

    class CompactLinear(nn.Module):
        def __init__(
            self,
            packed: dict[str, Any],
            bias: Any | None = None,
            max_loops: int = 0,
            lora_rank: int = 0,
            lora_alpha: float = 16.0,
            loop_ctx: dict[str, Any] | None = None,
            device: Any = None,
        ) -> None:
            super().__init__()
            self.bits = int(packed["bits"])
            self.group_size = int(packed["group_size"])
            self.out_features = int(packed["out_features"])
            self.in_features = int(packed["in_features"])
            self.register_buffer("qweight", packed["qweight"])
            self.register_buffer("scales", packed["scales"])
            if packed["zeros"] is not None:
                self.register_buffer("zeros", packed["zeros"])
            else:
                self.zeros = None
            if bias is not None:
                self.register_buffer("bias_buf", bias.detach().to(torch.bfloat16).contiguous())
            else:
                self.bias_buf = None

            self.max_loops = int(max_loops)
            self.lora_rank = int(lora_rank)
            self.lora_scaling = float(lora_alpha) / max(1, self.lora_rank)
            self.loop_ctx = loop_ctx

            if self.max_loops > 0 and self.lora_rank > 0:
                self.init_loop_lora(self.max_loops, self.lora_rank, lora_alpha)
            else:
                self.lora_A = None
                self.lora_B = None

            # Optional trainable or stashed Behavioral Alignment LoRA banks
            self.train_beh_A: Any | None = None
            self.train_beh_B: Any | None = None
            self.train_beh_scale: float = 1.0
            self.beh_banks: dict[str, tuple[Any, Any, float]] = {}

            if device is not None:
                self.to(device)

        def init_loop_lora(self, max_loops: int, lora_rank: int, lora_alpha: float = 16.0) -> None:
            self.max_loops = int(max_loops)
            self.lora_rank = int(lora_rank)
            self.lora_scaling = float(lora_alpha) / max(1, self.lora_rank)
            dev = self.qweight.device
            a_list, b_list = [], []
            for _ in range(self.max_loops):
                p_a = nn.Parameter(torch.empty((self.lora_rank, self.in_features), dtype=torch.bfloat16, device=dev), requires_grad=False)
                p_b = nn.Parameter(torch.zeros((self.out_features, self.lora_rank), dtype=torch.bfloat16, device=dev), requires_grad=False)
                a_list.append(p_a)
                b_list.append(p_b)
            self.lora_A = nn.ParameterList(a_list)
            self.lora_B = nn.ParameterList(b_list)

        def start_behavioral_lora_training(self, rank: int = 16, alpha: float = 16.0) -> list[Any]:
            dev = self.qweight.device
            self.train_beh_scale = float(alpha) / max(1, int(rank))
            self.train_beh_A = nn.Parameter(torch.empty((rank, self.in_features), dtype=torch.float32, device=dev))
            nn.init.kaiming_uniform_(self.train_beh_A, a=math.sqrt(5))
            self.train_beh_B = nn.Parameter(torch.zeros((self.out_features, rank), dtype=torch.float32, device=dev))
            return [self.train_beh_A, self.train_beh_B]

        def stash_behavioral_lora(self, bank_name: str) -> None:
            if self.train_beh_A is not None and self.train_beh_B is not None:
                self.beh_banks[bank_name] = (
                    self.train_beh_A.detach().to(torch.bfloat16).contiguous(),
                    self.train_beh_B.detach().to(torch.bfloat16).contiguous(),
                    float(self.train_beh_scale),
                )
            self.train_beh_A = None
            self.train_beh_B = None

        def forward(self, x: Any) -> Any:
            out = _CompactLinearFn.apply(
                x,
                self.qweight,
                self.scales,
                self.zeros,
                self.bias_buf,
                self.bits,
                self.out_features,
                self.in_features,
                self.group_size,
            )
            if self.lora_A is not None and self.lora_B is not None:
                k_idx = 0
                if self.loop_ctx is not None:
                    k_idx = min(int(self.loop_ctx.get("loop_idx", 0)), self.max_loops - 1)
                a_k = self.lora_A[k_idx].to(x.dtype)
                b_k = self.lora_B[k_idx].to(x.dtype)
                out = out + torch.matmul(torch.matmul(x, a_k.t()), b_k.t()) * self.lora_scaling

            if self.train_beh_A is not None and self.train_beh_B is not None:
                a_t = self.train_beh_A.to(x.dtype)
                b_t = self.train_beh_B.to(x.dtype)
                out = out + torch.matmul(torch.matmul(x, a_t.t()), b_t.t()) * self.train_beh_scale
            elif self.loop_ctx is not None:
                active_bank = self.loop_ctx.get("beh_bank")
                if active_bank and active_bank in self.beh_banks:
                    a_b, b_b, sc_b = self.beh_banks[active_bank]
                    out = out + torch.matmul(torch.matmul(x, a_b.to(x.dtype).t()), b_b.to(x.dtype).t()) * sc_b

            return out

    return CompactLinear


def strip_unused_multimodal_towers_inplace(model: Any) -> list[str]:
    removed: list[str] = []
    for holder in (model, getattr(model, "model", None)):
        if holder is None:
            continue
        for attr in ("vision_tower", "multi_modal_projector", "audio_tower", "audio_projector"):
            if hasattr(holder, attr) and getattr(holder, attr) is not None:
                setattr(holder, attr, None)
                removed.append(f"{type(holder).__name__}.{attr}")
    return removed


def compute_model_storage_bytes(model: Any) -> dict[str, Any]:
    seen: set[int] = set()
    p_bytes = 0
    b_bytes = 0
    for p in model.parameters():
        ptr = int(p.data_ptr())
        if ptr not in seen:
            seen.add(ptr)
            p_bytes += int(p.numel() * p.element_size())
    for b in model.buffers():
        ptr = int(b.data_ptr())
        if ptr not in seen:
            seen.add(ptr)
            b_bytes += int(b.numel() * b.element_size())
    tot = p_bytes + b_bytes
    return {
        "total_bytes": tot,
        "total_gb": round(tot / 1e9, 3),
        "param_gb": round(p_bytes / 1e9, 3),
        "buffer_gb": round(b_bytes / 1e9, 3),
    }


def compress_decoder_layers_inplace(
    harness: CompactHarness,
    importance: dict[int, np.ndarray],
    plan: dict[int, dict[str, int]],
    funnel_start: int,
    funnel_end: int,
    max_loops: int = 4,
    lora_rank: int = 16,
    lora_alpha: float = 16.0,
    loop_ctx: dict[str, Any] | None = None,
) -> dict[str, Any]:
    torch = harness.torch
    nn = torch.nn
    CompactLinear = build_compact_linear_class(torch)
    removed_towers = strip_unused_multimodal_towers_inplace(harness.model)
    n_replaced = 0

    for ell in range(harness.n_layers):
        blk = harness.layers[ell]
        spec = plan[ell]
        keep_k = int(spec["d_ff_keep"])
        bits = int(spec["bits"])
        in_funnel = funnel_start <= ell <= funnel_end

        mlp = blk.mlp
        orig_dff = int(mlp.down_proj.weight.shape[1])
        if keep_k < orig_dff:
            imp_vec = importance[ell]
            top_idx = np.sort(np.argpartition(imp_vec, -keep_k)[-keep_k:]).astype(np.int64)
            idx_t = torch.tensor(top_idx, dtype=torch.long, device=mlp.down_proj.weight.device)

            for proj_name in ("gate_proj", "up_proj"):
                if hasattr(mlp, proj_name):
                    lin = getattr(mlp, proj_name)
                    w_sub = lin.weight.detach().index_select(0, idx_t)
                    b_sub = lin.bias.detach().index_select(0, idx_t) if lin.bias is not None else None
                    pk = pack_weight_tensor(torch, w_sub, bits=bits)
                    setattr(
                        mlp,
                        proj_name,
                        CompactLinear(
                            pk,
                            bias=b_sub,
                            max_loops=max_loops if in_funnel else 0,
                            lora_rank=lora_rank if in_funnel else 0,
                            lora_alpha=lora_alpha,
                            loop_ctx=loop_ctx,
                            device=w_sub.device,
                        ),
                    )
                    del lin, w_sub
                    n_replaced += 1

            lin_d = mlp.down_proj
            wd_sub = lin_d.weight.detach().index_select(1, idx_t)
            bd_sub = lin_d.bias.detach() if lin_d.bias is not None else None
            pk_d = pack_weight_tensor(torch, wd_sub, bits=bits)
            mlp.down_proj = CompactLinear(
                pk_d,
                bias=bd_sub,
                max_loops=max_loops if in_funnel else 0,
                lora_rank=lora_rank if in_funnel else 0,
                lora_alpha=lora_alpha,
                loop_ctx=loop_ctx,
                device=wd_sub.device,
            )
            del lin_d, wd_sub
            n_replaced += 1

        for _p_name, p_mod in list(blk.named_modules()):
            for c_name, c_mod in list(p_mod.named_children()):
                if isinstance(c_mod, nn.Linear):
                    dev = c_mod.weight.device
                    pk = pack_weight_tensor(torch, c_mod.weight.detach(), bits=bits)
                    new_mod = CompactLinear(
                        pk,
                        bias=c_mod.bias.detach() if c_mod.bias is not None else None,
                        max_loops=max_loops if in_funnel else 0,
                        lora_rank=lora_rank if in_funnel else 0,
                        lora_alpha=lora_alpha,
                        loop_ctx=loop_ctx,
                        device=dev,
                    )
                    setattr(p_mod, c_name, new_mod)
                    del c_mod
                    n_replaced += 1

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    storage = compute_model_storage_bytes(harness.model)
    if torch.cuda.is_available():
        storage["cuda_allocated_gb"] = round(float(torch.cuda.memory_allocated()) / 1e9, 3)
    return {
        "n_linear_compressed": n_replaced,
        "removed_multimodal_towers": removed_towers,
        "storage": storage,
    }


# ---------------------------------------------------------------------------
# Procrustes Nullbridge Step Bridge & Recurrent Controller
# ---------------------------------------------------------------------------

def build_nullbridge_step_modules(
    torch: Any,
    procrustes_fit: dict[str, Any],
    max_loops: int = 4,
    rank: int = 128,
    halting_hidden: int = 128,
    device: Any = None,
) -> tuple[Any, Any]:
    nn = torch.nn
    F = torch.nn.functional
    d = int(procrustes_fit["d_model"])
    n_transitions = max(1, max_loops - 1)

    class ProcrustesNullBridge(nn.Module):
        """Procrustes + Rank-`r` step bridge supporting strictly zero-FLOP offline null-space weight folding."""

        def __init__(self) -> None:
            super().__init__()
            self.d_model = d
            self.max_loops = max_loops
            self.rank = rank

            r_init = torch.as_tensor(procrustes_fit["r_0"], dtype=torch.float32).detach().cpu().clone()
            mu_s = torch.as_tensor(procrustes_fit["mu_src"], dtype=torch.float32).detach().cpu().clone()
            mu_d = torch.as_tensor(procrustes_fit["mu_dst"], dtype=torch.float32).detach().cpu().clone()
            self.register_buffer("r_ortho_base", r_init.clone())
            self.register_buffer("r_ortho", r_init.clone())
            self.register_buffer("mu_src", mu_s.clone())
            self.register_buffer("mu_dst", mu_d.clone())
            self.register_buffer("rms_src", torch.tensor(float(procrustes_fit["rms_src"]), dtype=torch.float32))

            self.log_rms_dst = nn.Parameter(
                torch.full((n_transitions, d), math.log(max(float(procrustes_fit["rms_dst"]), 1e-6)), dtype=torch.float32),
                requires_grad=False,
            )
            self.lora_a = nn.Parameter(torch.zeros(rank, d, dtype=torch.float32), requires_grad=False)
            self.lora_b = nn.Parameter(torch.zeros(d, rank, dtype=torch.float32), requires_grad=False)
            self.step_emb = nn.Parameter(torch.zeros(n_transitions, d, dtype=torch.float32), requires_grad=False)

            self.gate_scalar = nn.Parameter(torch.zeros((n_transitions, 1), dtype=torch.float32), requires_grad=False)
            self.gate_logits = nn.Parameter(torch.zeros((n_transitions, d), dtype=torch.float32), requires_grad=False)

            self.register_buffer("lora_a_base", torch.zeros(rank, d, dtype=torch.float32))
            self.register_buffer("mu_src_base", mu_s.clone())
            self.gate_override: float | None = None

        def snapshot_base_weights(self) -> None:
            with torch.no_grad():
                self.r_ortho.copy_(self.r_ortho_base)
                self.lora_a_base.copy_(self.lora_a.detach())
                self.mu_src_base.copy_(self.mu_src.detach())

        def apply_nullbridge_fold(
            self,
            q_mat_np: np.ndarray | None,
            neutral_mean_np: np.ndarray | None = None,
        ) -> None:
            """Fold `P_perp = I - Q Q^T` offline into `r_ortho` and `lora_a` (`0` runtime FLOPs!)."""
            with torch.no_grad():
                if q_mat_np is None:
                    self.r_ortho.copy_(self.r_ortho_base)
                    self.lora_a.copy_(self.lora_a_base)
                    self.mu_src.copy_(self.mu_src_base)
                    return

                q64 = torch.tensor(np.asarray(q_mat_np, dtype=np.float64), dtype=torch.float64, device=self.r_ortho.device)
                rb64 = self.r_ortho_base.to(torch.float64)
                ab64 = self.lora_a_base.to(torch.float64)
                r_safe = rb64 - torch.matmul(torch.matmul(rb64, q64), q64.t())
                a_safe = ab64 - torch.matmul(torch.matmul(ab64, q64), q64.t())
                self.r_ortho.copy_(r_safe.to(torch.float32))
                self.lora_a.copy_(a_safe.to(torch.float32))

                mu64 = self.mu_src_base.to(torch.float64)
                if neutral_mean_np is not None:
                    neu64 = torch.tensor(np.asarray(neutral_mean_np, dtype=np.float64), dtype=torch.float64, device=self.r_ortho.device)
                    mu64 = mu64 - torch.matmul(mu64, q64) @ q64.t() + torch.matmul(neu64, q64) @ q64.t()
                self.mu_src.copy_(mu64.to(torch.float32))

        def effective_gate(self, transition_idx: int) -> Any:
            if self.gate_override is not None:
                return float(self.gate_override)
            t_idx = min(max(0, int(transition_idx)), n_transitions - 1)
            return torch.sigmoid(self.gate_scalar[t_idx] + self.gate_logits[t_idx])

        def forward(self, h_anchor: Any, h_end: Any, transition_idx: int = 0) -> Any:
            """Identical runtime graph whether `apply_nullbridge_fold` is active or not (`0` extra FLOPs)."""
            t_idx = min(max(0, int(transition_idx)), n_transitions - 1)
            orig_dtype = h_anchor.dtype
            anc = h_anchor.to(torch.float32)
            x = h_end.to(torch.float32)

            x_c = x - self.mu_src
            cur_rms = torch.sqrt(torch.mean(x_c * x_c, dim=-1, keepdim=True) + 1e-8)
            x_n = x_c / cur_rms

            rot = torch.matmul(x_n, self.r_ortho.t())
            delta = torch.matmul(torch.matmul(x_n, self.lora_a.t()), self.lora_b.t())
            mapped = torch.exp(self.log_rms_dst[t_idx]) * (rot + delta) + self.mu_dst + self.step_emb[t_idx]

            g = self.effective_gate(t_idx)
            blended = (1.0 - g) * anc + g * mapped
            return blended.to(orig_dtype)

    class AdaptiveHaltingHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc1 = nn.Linear(d + max_loops, halting_hidden, dtype=torch.float32)
            self.fc2 = nn.Linear(halting_hidden, 1, dtype=torch.float32)

        def forward(self, h_end_last: Any, loop_idx: int) -> Any:
            x = h_end_last.to(torch.float32)
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
            x_n = x / rms
            one_hot = torch.zeros((x.shape[0], max_loops), dtype=torch.float32, device=x.device)
            one_hot[:, min(max(0, loop_idx), max_loops - 1)] = 1.0
            return torch.sigmoid(self.fc2(F.silu(self.fc1(torch.cat([x_n, one_hot], dim=-1))))).squeeze(-1)

    bridge = ProcrustesNullBridge()
    halting = AdaptiveHaltingHead()
    if device is not None:
        bridge = bridge.to(device)
        halting = halting.to(device)
    return bridge, halting


class Gemma4NullbridgeController:
    """Recurrent controller managing `K = 1..4` passes, Nullbridge folds, boundary clamps, and Behavioral LoRAs."""

    def __init__(
        self,
        harness: CompactHarness,
        bridge: Any,
        halting_head: Any,
        loop_ctx: dict[str, Any],
        start_layer: int = 24,
        end_layer: int = 42,
        max_loops: int = 4,
    ) -> None:
        self.harness = harness
        self.bridge = bridge
        self.halting_head = halting_head
        self.loop_ctx = loop_ctx
        self.start_layer = int(start_layer)
        self.end_layer = int(end_layer)
        self.max_loops = int(max_loops)

        # Boundary-only projection tensors (`entry` at L24 Pass-0 input, `exit` at L42 final Pass-K output)
        self.entry_q: Any | None = None
        self.entry_mu: Any | None = None
        self.exit_q: Any | None = None
        self.exit_mu: Any | None = None

        # 19-layer internal full-funnel clamp comparator (`L24..L42` across every pass)
        self.funnel_clamp_active: bool = False
        self.funnel_q_by_layer: dict[int, Any] = {}
        self.funnel_mu_by_layer: dict[int, Any] = {}

    def reset_interventions(self) -> None:
        self.loop_ctx["beh_bank"] = None
        self.bridge.apply_nullbridge_fold(None)
        self.bridge.gate_override = None
        self.entry_q = None
        self.entry_mu = None
        self.exit_q = None
        self.exit_mu = None
        self.funnel_clamp_active = False

    def set_behavioral_lora_bank(self, bank_name: str | None) -> None:
        self.loop_ctx["beh_bank"] = bank_name

    def set_boundary_clamps(
        self,
        entry_q_np: np.ndarray | None = None,
        entry_mu_np: np.ndarray | None = None,
        exit_q_np: np.ndarray | None = None,
        exit_mu_np: np.ndarray | None = None,
    ) -> None:
        torch = self.harness.torch
        dev = self.harness.input_device
        if entry_q_np is not None:
            self.entry_q = torch.tensor(entry_q_np, dtype=torch.float32, device=dev)
            self.entry_mu = (
                torch.tensor(entry_mu_np, dtype=torch.float32, device=dev)
                if entry_mu_np is not None
                else torch.zeros(entry_q_np.shape[0], dtype=torch.float32, device=dev)
            )
        else:
            self.entry_q = None
            self.entry_mu = None

        if exit_q_np is not None:
            self.exit_q = torch.tensor(exit_q_np, dtype=torch.float32, device=dev)
            self.exit_mu = (
                torch.tensor(exit_mu_np, dtype=torch.float32, device=dev)
                if exit_mu_np is not None
                else torch.zeros(exit_q_np.shape[0], dtype=torch.float32, device=dev)
            )
        else:
            self.exit_q = None
            self.exit_mu = None

    def configure_full_funnel_clamp(
        self,
        q_by_layer_np: dict[int, np.ndarray],
        mu_by_layer_np: dict[int, np.ndarray],
        active: bool = False,
    ) -> None:
        torch = self.harness.torch
        dev = self.harness.input_device
        for ell in range(self.start_layer, self.end_layer + 1):
            self.funnel_q_by_layer[ell] = torch.tensor(q_by_layer_np[ell], dtype=torch.float32, device=dev)
            self.funnel_mu_by_layer[ell] = torch.tensor(mu_by_layer_np[ell], dtype=torch.float32, device=dev)
        self.funnel_clamp_active = active

    @contextmanager
    def active_recurrence(
        self,
        n_loops: int = 1,
        capture_loop_states: list[Any] | None = None,
    ) -> Iterator[None]:
        """Execute `Prelude (L0..L23) -> K Passes of L24..L42 -> Coda (L43..L59)`."""
        torch = self.harness.torch
        self.loop_ctx["loop_idx"] = 0
        target_k = min(max(1, int(n_loops)), self.max_loops)

        needs_hooks = (
            target_k > 1
            or self.entry_q is not None
            or self.exit_q is not None
            or self.funnel_clamp_active
            or capture_loop_states is not None
        )
        if not needs_hooks:
            yield
            return

        state_box: dict[str, Any] = {
            "in_extra_loop": False,
            "h_anchor": None,
            "call_specs": {},
        }
        handles = []

        def _project_out(h_tensor: Any, q_t: Any, mu_t: Any) -> Any:
            h32 = h_tensor.to(torch.float32)
            proj = torch.matmul(torch.matmul(h32, q_t), q_t.t()) - mu_t
            return (h32 - proj).to(h_tensor.dtype)

        def _maybe_internal_clamp(ell: int, h_tensor: Any) -> Any:
            if not self.funnel_clamp_active or ell not in self.funnel_q_by_layer:
                return h_tensor
            return _project_out(h_tensor, self.funnel_q_by_layer[ell], self.funnel_mu_by_layer[ell])

        for ell in range(self.start_layer, self.end_layer + 1):
            blk = self.harness.layers[ell]

            def _make_pre(idx: int):
                def _pre(_mod: Any, args: Any, kwargs: Any):
                    if state_box["in_extra_loop"]:
                        return None
                    if len(args) >= 1:
                        h_in = args[0]
                        had_pos = True
                        rest_args = args[1:]
                    else:
                        h_in = kwargs["hidden_states"]
                        had_pos = False
                        rest_args = ()

                    clean_kw = {}
                    for k, v in kwargs.items():
                        if k == "hidden_states" and not had_pos:
                            continue
                        if k in ("past_key_value", "past_key_values"):
                            clean_kw[k] = None
                        elif k == "use_cache":
                            clean_kw[k] = False
                        else:
                            clean_kw[k] = v

                    if idx == self.start_layer:
                        self.loop_ctx["loop_idx"] = 0
                        if self.entry_q is not None and self.entry_mu is not None:
                            h_in = _project_out(h_in, self.entry_q, self.entry_mu)
                            state_box["h_anchor"] = h_in
                            state_box["call_specs"][idx] = (had_pos, rest_args, clean_kw)
                            if had_pos:
                                return (h_in,) + rest_args, kwargs
                            new_kw = dict(kwargs)
                            new_kw["hidden_states"] = h_in
                            return args, new_kw
                        state_box["h_anchor"] = h_in
                    state_box["call_specs"][idx] = (had_pos, rest_args, clean_kw)
                    return None
                return _pre

            handles.append(blk.register_forward_pre_hook(_make_pre(ell), with_kwargs=True))

            if self.funnel_clamp_active and ell < self.end_layer:
                def _make_mid(idx: int):
                    def _post_m(_m: Any, _a: Any, out: Any):
                        if state_box["in_extra_loop"]:
                            return out
                        wt = isinstance(out, tuple)
                        h0 = _maybe_internal_clamp(idx, out[0] if wt else out)
                        return (h0,) + out[1:] if wt else h0
                    return _post_m
                handles.append(blk.register_forward_hook(_make_mid(ell)))

        def _end_post_hook(_mod: Any, _args: Any, out: Any):
            if state_box["in_extra_loop"]:
                return out
            was_tuple = isinstance(out, tuple)
            h_cur = out[0] if was_tuple else out
            rest_tuple = out[1:] if was_tuple else ()
            h_cur = _maybe_internal_clamp(self.end_layer, h_cur)
            h_anchor = state_box["h_anchor"]

            if capture_loop_states is not None:
                h_snap = (
                    _project_out(h_cur, self.exit_q, self.exit_mu)
                    if (self.exit_q is not None and self.exit_mu is not None)
                    else h_cur
                )
                capture_loop_states.append(h_snap[:, -1, :].detach().to(torch.float32).cpu())

            if target_k > 1:
                state_box["in_extra_loop"] = True
                try:
                    for loop_i in range(1, target_k):
                        self.loop_ctx["loop_idx"] = loop_i
                        h_next = self.bridge(h_anchor, h_cur, transition_idx=loop_i - 1)
                        for ell in range(self.start_layer, self.end_layer + 1):
                            blk_l = self.harness.layers[ell]
                            had_pos, rest_args, clean_kw = state_box["call_specs"][ell]
                            if had_pos:
                                res = blk_l(h_next, *rest_args, **clean_kw)
                            else:
                                res = blk_l(hidden_states=h_next, **clean_kw)
                            h_next = res[0] if isinstance(res, tuple) else res
                            h_next = _maybe_internal_clamp(ell, h_next)
                        h_cur = h_next
                        if capture_loop_states is not None:
                            h_snap = (
                                _project_out(h_cur, self.exit_q, self.exit_mu)
                                if (self.exit_q is not None and self.exit_mu is not None)
                                else h_cur
                            )
                            capture_loop_states.append(h_snap[:, -1, :].detach().to(torch.float32).cpu())
                finally:
                    state_box["in_extra_loop"] = False
                    self.loop_ctx["loop_idx"] = 0
                    state_box["h_anchor"] = None
                    state_box["call_specs"].clear()
            else:
                self.loop_ctx["loop_idx"] = 0
                state_box["h_anchor"] = None
                state_box["call_specs"].clear()

            if self.exit_q is not None and self.exit_mu is not None:
                h_cur = _project_out(h_cur, self.exit_q, self.exit_mu)

            return (h_cur,) + rest_tuple if was_tuple else h_cur

        handles.append(self.harness.layers[self.end_layer].register_forward_hook(_end_post_hook))
        try:
            yield
        finally:
            self.loop_ctx["loop_idx"] = 0
            for h in handles:
                h.remove()


def precompute_kl_references(
    harness: CompactHarness,
    passages: Sequence[str],
    top_m: int = 256,
    batch_size: int = 2,
) -> list[dict[str, Any]]:
    """Cache intact `BF16` top-M token probabilities for exact held-out & anchor KL measurement."""
    torch = harness.torch
    F = torch.nn.functional
    tok = harness.tokenizer
    has_chat_tmpl = bool(getattr(tok, "chat_template", None))
    refs: list[dict[str, Any]] = []

    prompts = [format_chat_prompt(tok, "Technical reference passage:", assistant_prefill=p) for p in passages]
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i : i + batch_size]
        enc = tok(
            chunk,
            add_special_tokens=not has_chat_tmpl,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )
        inp = enc["input_ids"].to(harness.input_device)
        msk = enc["attention_mask"].to(harness.input_device)
        with torch.no_grad():
            out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
            logits = out.logits[:, :-1, :].to(torch.float32)
            lp = F.log_softmax(logits, dim=-1)
            probs = torch.exp(lp)
            k_eff = min(top_m, int(probs.shape[-1]))
            top_p, top_idx = torch.topk(probs, k=k_eff, dim=-1)
            top_lp = torch.gather(lp, dim=-1, index=top_idx)

        valid_mask = (msk[:, 1:] > 0) & (msk[:, :-1] > 0)
        for b_i in range(inp.shape[0]):
            refs.append({
                "input_ids": inp[b_i : b_i + 1].detach().cpu(),
                "attention_mask": msk[b_i : b_i + 1].detach().cpu(),
                "valid_mask": valid_mask[b_i : b_i + 1].detach().cpu(),
                "top_p": top_p[b_i : b_i + 1].to(torch.float16).detach().cpu(),
                "top_lp": top_lp[b_i : b_i + 1].to(torch.float16).detach().cpu(),
                "top_idx": top_idx[b_i : b_i + 1].to(torch.int32).detach().cpu(),
            })
    return refs


def evaluate_kl_against_refs(harness: CompactHarness, refs: Sequence[dict[str, Any]]) -> dict[str, float]:
    torch = harness.torch
    F = torch.nn.functional
    tok_kls: list[float] = []
    for ref in refs:
        inp = ref["input_ids"].to(harness.input_device)
        msk = ref["attention_mask"].to(harness.input_device)
        val = ref["valid_mask"].to(harness.input_device)
        top_p = ref["top_p"].to(device=harness.input_device, dtype=torch.float32)
        top_lp = ref["top_lp"].to(device=harness.input_device, dtype=torch.float32)
        top_idx = ref["top_idx"].to(device=harness.input_device, dtype=torch.long)
        with torch.no_grad():
            out = harness.model(input_ids=inp, attention_mask=msk, use_cache=False)
            q_lp_all = F.log_softmax(out.logits[:, :-1, :].to(torch.float32), dim=-1)
            q_lp = torch.gather(q_lp_all, dim=-1, index=top_idx)
            kl_t = (top_p * (top_lp - q_lp)).sum(dim=-1).clamp(min=0.0)
            if bool(val.any().item()):
                tok_kls.extend(kl_t[val].detach().cpu().numpy().tolist())
    mean_kl = float(np.mean(tok_kls)) if tok_kls else 0.0
    return {
        "kl_nats": round(mean_kl, 4),
        "ppl_ratio": round(float(math.exp(min(mean_kl, 12.0))), 4),
    }


def load_gemma4_loop_compact_with_kl_refs(
    package_dir: str | Path,
    base_model_id: str,
    eval_kl_passages: Sequence[str],
    train_kl_passages: Sequence[str],
    device_map: str = "cuda",
    model_label: str | None = None,
) -> tuple[Gemma4NullbridgeController, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Load intact `BF16` `Gemma 4 31B`, cache exact `BF16` KL references, then compress & load `Gemma-4-Loop-Compact`."""
    pkg = Path(package_dir)
    cfg = json.loads((pkg / "config_loop_compact.json").read_text(encoding="utf-8"))
    label = model_label or cfg.get("base_model_id", base_model_id)

    harness = CompactHarness.from_pretrained(base_model_id, dtype="bfloat16", device_map=device_map, model_id_label=label)
    torch = harness.torch
    bf16_storage = compute_model_storage_bytes(harness.model)

    eval_kl_refs = precompute_kl_references(harness, eval_kl_passages, top_m=256, batch_size=2)
    train_kl_refs = precompute_kl_references(harness, train_kl_passages, top_m=128, batch_size=2)

    imp_npz = np.load(pkg / "neuron_importance.npz")
    importance = {int(k.split("_")[1]): imp_npz[k] for k in imp_npz.files}
    plan = {int(k): v for k, v in cfg["compression_plan"].items()}
    loop_ctx: dict[str, Any] = {"loop_idx": 0, "beh_bank": None}

    comp_info = compress_decoder_layers_inplace(
        harness,
        importance=importance,
        plan=plan,
        funnel_start=int(cfg["funnel_start"]),
        funnel_end=int(cfg["funnel_end"]),
        max_loops=int(cfg["max_loops"]),
        lora_rank=int(cfg["lora_rank"]),
        lora_alpha=float(cfg["lora_alpha"]),
        loop_ctx=loop_ctx,
    )

    state_bundle = torch.load(pkg / "loop_compact_adapters.pt", map_location=harness.input_device, weights_only=True)
    b_sd = dict(state_bundle["bridge_state_dict"])
    b_sd.pop("safety_q", None)
    if "r_ortho" not in b_sd and "r_ortho_base" in b_sd:
        b_sd["r_ortho"] = b_sd["r_ortho_base"].clone()
    p_fit = dict(state_bundle["procrustes_meta"])
    p_fit["r_0"] = b_sd["r_ortho_base"].cpu().numpy()

    bridge, halting = build_nullbridge_step_modules(
        torch,
        procrustes_fit=p_fit,
        max_loops=int(cfg["max_loops"]),
        rank=int(cfg["bridge_rank"]),
        device=harness.input_device,
    )
    bridge.load_state_dict(b_sd, strict=False)
    bridge.snapshot_base_weights()
    halting.load_state_dict(state_bundle["halting_state_dict"])

    for ell_str, sd in state_bundle["funnel_lora_state_dicts"].items():
        ell = int(ell_str)
        cast_sd = {k: v.to(dtype=torch.bfloat16, device=harness.input_device) for k, v in sd.items()}
        harness.layers[ell].load_state_dict(cast_sd, strict=False)

    controller = Gemma4NullbridgeController(
        harness=harness,
        bridge=bridge,
        halting_head=halting,
        loop_ctx=loop_ctx,
        start_layer=int(cfg["funnel_start"]),
        end_layer=int(cfg["funnel_end"]),
        max_loops=int(cfg["max_loops"]),
    )
    learned_gates = [
        round(float(bridge.effective_gate(t).detach().mean().item()), 4)
        for t in range(bridge.gate_logits.shape[0])
    ]
    load_meta = {
        "base_model_id": label,
        "n_layers": harness.n_layers,
        "d_model": int(p_fit["d_model"]),
        "funnel_start": int(cfg["funnel_start"]),
        "funnel_end": int(cfg["funnel_end"]),
        "max_loops": int(cfg["max_loops"]),
        "bf16_storage_gb": bf16_storage["total_gb"],
        "compact_storage": comp_info["storage"],
        "learned_step_gates": learned_gates,
    }
    return controller, eval_kl_refs, train_kl_refs, load_meta
