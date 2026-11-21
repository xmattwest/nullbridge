"""CPU unit tests for Track A (`_parse_first_int` + full `InklingLoopController` mock when `torch` is installed)."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from nullbridge.benchmarks import build_nullbridge_benchmarks
from nullbridge.track_a import _parse_first_int


def test_parse_first_int_handles_inkling_end_tokens() -> None:
    assert _parse_first_int("  -148<|end_message|>extra 999") == -148
    assert _parse_first_int("1,250\nnext line") == 1250
    assert _parse_first_int("no digits here") is None


def test_track_a_controller_lens_lora_and_conditions_cpu() -> None:
    try:
        import torch
    except ImportError:
        import pytest
        pytest.skip("torch not installed in lightweight local venv")
    nn = torch.nn
    from nullbridge.track_a import (
        InklingLoopController,
        build_kl_reference_cache,
        configure_track_a_condition,
        evaluate_track_a_cell,
        run_block1_message_bank_geometry,
        run_block2_coda_transport_looped_lens,
        train_inkling_coda_lora,
    )

    class _DummyInklingTokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if text == "A":
                return [10]
            if text == "B":
                return [11]
            if text in ("<|end_message|>", "<|content_model_end_sampling|>"):
                return [2]
            return [12 + ((ord(ch) + idx * 7) % 115) for idx, ch in enumerate(text[-128:])] or [12]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            digits = [str((i - 12) % 10) for i in ids if i != 2]
            return "".join(digits[:3]) if digits else "42"

    class _TinyMessageGate(nn.Module):
        def __init__(self, d: int, rank: int) -> None:
            super().__init__()
            self.down_proj = nn.Linear(2 * d, rank, bias=False)
            self.forget_gate_proj = nn.Linear(rank, d, bias=False)
            self.message_gate_proj = nn.Linear(rank, d, bias=False)

        def forward(self, h, m):
            hf, mf = h.float(), m.float()
            nh = hf * torch.rsqrt(hf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            nm = mf * torch.rsqrt(mf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            gh = self.down_proj(torch.cat([nh, nm], dim=-1))
            fg = torch.tanh(self.forget_gate_proj(gh))
            mg = torch.tanh(self.message_gate_proj(gh))
            return ((1.0 - fg) * hf + mg * mf).to(h.dtype)

    class _TinyMessageInput(nn.Module):
        def __init__(self, n_loop: int, d: int) -> None:
            super().__init__()
            self.query = nn.Parameter(torch.randn(n_loop, 4, d) * 0.05)

        def forward(self, prev_blocks, prev_h, loop_layer_idx: int):
            stacked = torch.cat([prev_blocks, prev_h.unsqueeze(-2)], dim=-2).float()
            stacked = stacked * torch.rsqrt(stacked.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
            q = self.query[loop_layer_idx].float()
            scores = torch.einsum("btsd,cd->btcs", stacked, q) * (stacked.shape[-1] ** -0.5)
            probs = torch.softmax(scores, dim=-1)
            return torch.einsum("btcs,btsd->btcd", probs, stacked)

    class _TinySelfAttn(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.q_proj = SimpleNamespace(linear=nn.Linear(d, d, bias=False))
            self.k_proj = SimpleNamespace(linear=nn.Linear(d, d, bias=False))
            self.v_proj = SimpleNamespace(linear=nn.Linear(d, d, bias=False))
            self.o_proj = SimpleNamespace(linear=nn.Linear(d, d, bias=False), gate_proj=nn.Linear(d, d, bias=False))
            self._mods = nn.ModuleList(
                [self.q_proj.linear, self.k_proj.linear, self.v_proj.linear, self.o_proj.linear, self.o_proj.gate_proj]
            )

        def forward(self, qh, kh, vh):
            q = self.q_proj.linear(qh)
            k = self.k_proj.linear(kh)
            v = self.v_proj.linear(vh)
            att = 0.5 * (q + k) * v
            gate = torch.sigmoid(self.o_proj.gate_proj(qh))
            return self.o_proj.linear(att * gate)

    class _TinyInklingLayer(nn.Module):
        def __init__(self, d: int, rank: int, is_loop: bool, is_boundary: bool) -> None:
            super().__init__()
            self.is_loop = is_loop
            self.is_block_boundary = is_boundary
            self.self_attn = _TinySelfAttn(d)
            self.mlp = nn.Linear(d, d, bias=False)
            if is_loop:
                self.q_message_gate = _TinyMessageGate(d, rank)
                self.k_message_gate = _TinyMessageGate(d, rank)
                self.v_message_gate = _TinyMessageGate(d, rank)
                self.mlp_message_gate = _TinyMessageGate(d, rank)

        def forward(self, h, messages=None):
            if self.is_loop and messages is not None:
                qm, km, vm, _mm = messages.unbind(-2)
                qh = self.q_message_gate(h, qm)
                kh = self.k_message_gate(h, km)
                vh = self.v_message_gate(h, vh if False else vm)
            else:
                qh = kh = vh = h
            a_out = self.self_attn(qh, kh, vh)
            h2 = h + 0.2 * a_out
            mh = self.mlp_message_gate(h2, messages[..., 3, :]) if (self.is_loop and messages is not None) else h2
            return h2 + 0.2 * torch.tanh(self.mlp(mh))

    class _TinyRMSNorm(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            xf = x.float()
            return (xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * self.weight).to(x.dtype)

    @dataclass
    class _TinyOut:
        logits: object
        cache: None = None

    class _TinyInklingConditionalGen(nn.Module):
        def __init__(self, d: int = 32, vocab: int = 128) -> None:
            super().__init__()
            text_cfg = SimpleNamespace(
                hidden_size=d,
                num_hidden_layers=8,
                loop_layer_index_start=2,
                loop_layer_index_end=6,
                loop_num_passes=3,
                loop_gate_rank=8,
            )
            self.config = SimpleNamespace(text_config=text_cfg)
            embed = nn.Embedding(vocab, d)
            layers = nn.ModuleList(
                [_TinyInklingLayer(d, 8, is_loop=(2 <= i < 6), is_boundary=(i % 2 == 0)) for i in range(8)]
            )
            msg_in = _TinyMessageInput(4, d)
            norm = _TinyRMSNorm(d)
            self.language_model = SimpleNamespace(
                config=text_cfg,
                embed_tokens=embed,
                layers=layers,
                message_input=msg_in,
                norm=norm,
            )
            self._submods = nn.ModuleList([embed, layers, msg_in, norm])
            self.model = SimpleNamespace(language_model=self.language_model)
            self.lm_head = nn.Linear(d, vocab, bias=False)
            self.logit_softcap = nn.Identity()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            _ = kwargs
            lm = self.language_model
            h = lm.embed_tokens(input_ids)
            if attention_mask is not None:
                h = h * attention_mask.unsqueeze(-1).to(h.dtype)
            blocks = h.new_zeros(*h.shape[:-1], 0, h.shape[-1])
            for lyr in lm.layers[:2]:
                nh = lyr(h)
                if lyr.is_block_boundary:
                    blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
                h = nh
            pre_h, pre_b = h, blocks
            n_pre = pre_b.shape[-2]
            for lyr in lm.layers[2:6]:
                nh = lyr(h)
                if lyr.is_block_boundary:
                    blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
                h = nh
            for _p in range(1, lm.config.loop_num_passes):
                prev_b = blocks[..., n_pre + 1 :, :]
                prev_h = h
                h, blocks = pre_h, pre_b
                for li, lyr in enumerate(lm.layers[2:6]):
                    msgs = lm.message_input(prev_b, prev_h, li)
                    nh = lyr(h, msgs)
                    if lyr.is_block_boundary:
                        blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
                    h = nh
            for lyr in lm.layers[6:]:
                nh = lyr(h)
                if lyr.is_block_boundary:
                    blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
                h = nh
            fin = lm.norm(h)
            return _TinyOut(logits=self.logit_softcap(self.lm_head(fin)))

    torch.manual_seed(7)
    model = _TinyInklingConditionalGen(d=32, vocab=128)
    tok = _DummyInklingTokenizer()
    ctrl = InklingLoopController(model, tok, model_label="tiny-inkling-test")

    bm_full = build_nullbridge_benchmarks()
    bm_mini = {
        "probe_pairs": [p for p in bm_full["probe_pairs"] if p.pair_id.endswith(("00", "01", "02", "03"))],
        "lora_train": bm_full["lora_train"][:8],
        "train_kl_passages": bm_full["train_kl_passages"][:4],
        "eval_kl_passages": bm_full["eval_kl_passages"][:4],
        "eval_overt_sycophancy": bm_full["eval_overt_sycophancy"][:4],
        "eval_covert_sycophancy": bm_full["eval_covert_sycophancy"][:4],
        "eval_verifier": bm_full["eval_verifier"][:6],
        "eval_alias_guardrail": bm_full["eval_alias_guardrail"][:6],
        "eval_multihop": bm_full["eval_multihop"][:8],
        "eval_wordprob": bm_full["eval_wordprob"][:4],
        "eval_balsheet": bm_full["eval_balsheet"][:4],
        "eval_cashflow_sanity": bm_full["eval_cashflow_sanity"][:4],
        "eval_moral_pairs": bm_full["eval_moral_pairs"][:4],
    }

    b1, rt = run_block1_message_bank_geometry(ctrl, bm_mini, probe_passes=3, max_traj_passes=3, batch_size=8)
    assert b1["bos_sink_orthogonality"]["msg_basis_bos_overlap"] < 1e-12
    assert b1["zero_flop_fold_audit"]["max_query_null_leakage"] < 1e-6
    assert b1["zero_flop_fold_audit"]["max_gate_down_null_leakage"] < 1e-6
    assert b1["zero_flop_fold_audit"]["max_lm_head_null_leakage"] < 1e-6

    lens = run_block2_coda_transport_looped_lens(ctrl, bm_mini, rt, n_passes=3, max_tokens=32)
    assert "pass_1" in lens["by_pass"] and "pass_3" in lens["by_pass"]

    kl_refs = build_kl_reference_cache(ctrl, bm_mini["eval_kl_passages"], ref_passes=3, max_tokens=32, top_k=16)
    lora_res = train_inkling_coda_lora(
        ctrl, lora_train_items=bm_mini["lora_train"], train_kl_refs=kl_refs, n_steps=3, batch_size=2, rank=4
    )
    assert lora_res["trainable_params"] > 0

    for cond in (
        "unclamped",
        "coda_lora",
        "msg_nullbridge_mlp",
        "msg_nullbridge_qkv",
        "msg_nullbridge_all4",
        "zero_flop_fold_nullbridge",
    ):
        configure_track_a_condition(ctrl, cond, rt)
        cell, rvecs = evaluate_track_a_cell(ctrl, bm_mini, kl_refs, n_passes=2, batch_size=4)
        assert cell["n_passes"] == 2
        assert isinstance(rvecs["verifier_hack_suscept"], np.ndarray)
    ctrl.reset_interventions()


if __name__ == "__main__":
    test_parse_first_int_handles_inkling_end_tokens()
    test_track_a_controller_lens_lora_and_conditions_cpu()
    print("[OK] Track A PyTorch self-test passed.")

