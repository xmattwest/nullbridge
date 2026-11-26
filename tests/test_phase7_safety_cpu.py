"""CPU + mock-PyTorch unit tests for Phase 7 Block A (`biophysical_safety.py`).

Verifies:
  1. Zero em-dash invariant and `n = 64` zero-overlap Phase 7 moral pair construction.
  2. Synthetic `24 x 24` random-matrix SVD accuracy via `deflated_topk_svd_jvp_vjp`
     (`< 1e-4` vs `torch.linalg.svdvals`, `||V^T V - I_k||_F < 1e-5`, `||U^T U - I_k||_F < 1e-5`).
  3. End-to-end `_TinyBiophysicalModel` execution of Exp A1, Exp A2.1 (`Parseval < 1e-12`),
     Exp A2.2 (all 7 selective zero-FLOP Nullbridge conditions + `reset_interventions()`
     restoring `0.00e+00` logits), and `run_phase7_block_a` atomic checkpoint resumability.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from nullbridge.benchmarks import build_nullbridge_benchmarks
from nullbridge.biophysical_safety import (
    SELECTIVE_NULLBRIDGE_CONDITIONS,
    build_phase7_moral_pairs,
)
from nullbridge.biophysical_stimuli import EM_DASH, build_phase6_stimulus_suite


def test_phase7_safety_zero_em_dash_and_moral_pairs_audit() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for rel in (
        "src/nullbridge/biophysical_safety.py",
        "tests/test_phase7_safety_cpu.py",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert EM_DASH not in txt, f"Em dash found in {rel}"

    bm = build_nullbridge_benchmarks()
    pairs = build_phase7_moral_pairs(bm)
    assert len(pairs) == 64

    # Verify zero SHA-256 overlap across all 128 scenarios
    hashes = set()
    for t_it, c_it in pairs:
        ht = hashlib.sha256(t_it.scenario.strip().encode("utf-8")).hexdigest()
        hc = hashlib.sha256(c_it.scenario.strip().encode("utf-8")).hexdigest()
        assert ht not in hashes
        assert hc not in hashes
        assert ht != hc
        hashes.add(ht)
        hashes.add(hc)


def test_deflated_topk_svd_synthetic_24x24() -> None:
    try:
        import torch
    except ImportError:
        try:
            import pytest
            pytest.skip("torch not installed in lightweight local venv")
        except ImportError:
            return

    from nullbridge.biophysical_safety import deflated_topk_svd_jvp_vjp

    torch.manual_seed(202607)
    d = 24
    k = 8
    # Construct a well-conditioned 24x24 random matrix with distinct spectrum
    mat = torch.randn((d, d), dtype=torch.float64)
    true_sv = torch.linalg.svdvals(mat)[:k]

    res = deflated_topk_svd_jvp_vjp(
        jvp_fn=lambda v: mat @ v.to(torch.float64),
        vjp_fn=lambda u: mat.t() @ u.to(torch.float64),
        d_in=d,
        d_out=d,
        k=k,
        n_iters=24,
        oversample=6,
        batch_size=None,
        device="cpu",
        dtype=torch.float64,
        seed=17,
    )
    est_sv = res["sigma"].to(torch.float64)
    cert_sv = res["vjp_cert_sigma"].to(torch.float64)
    max_sv_err = float((est_sv - true_sv).abs().max().item())
    max_cert_err = float((cert_sv - true_sv).abs().max().item())

    assert max_sv_err < 1e-4, f"Singular value error too large: {max_sv_err}"
    assert max_cert_err < 1e-4, f"VJP certificate error too large: {max_cert_err}"
    assert res["v_orth_err"] < 1e-5, f"V orthonormality error: {res['v_orth_err']}"
    assert res["u_orth_err"] < 1e-5, f"U orthonormality error: {res['u_orth_err']}"

    # Batched check (B = 3, float32 operator)
    mats_b = torch.randn((3, d, d), dtype=torch.float32)
    true_sv_b = torch.linalg.svdvals(mats_b)[:, :k]
    res_b = deflated_topk_svd_jvp_vjp(
        jvp_fn=lambda v: torch.bmm(mats_b, v.unsqueeze(-1)).squeeze(-1),
        vjp_fn=lambda u: torch.bmm(mats_b.transpose(-2, -1), u.unsqueeze(-1)).squeeze(-1),
        d_in=d,
        d_out=d,
        k=k,
        n_iters=24,
        oversample=6,
        batch_size=3,
        device="cpu",
        dtype=torch.float32,
        seed=23,
    )
    max_sv_err_b = float((res_b["sigma"] - true_sv_b).abs().max().item())
    assert max_sv_err_b < 1e-4, f"Batched singular value error too large: {max_sv_err_b}"
    assert res_b["v_orth_err"] < 1e-5
    assert res_b["u_orth_err"] < 1e-5


def _build_tiny_controller():
    import torch
    from nullbridge.biophysical_inkling import BiophysicalInklingController

    nn = torch.nn

    class _MockTok:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            special = {
                "<|message_model|>": [3],
                "<|content_text|>": [4],
                "<|content_thinking|>": [5],
                "<|end_message|>": [2],
                "<|content_model_end_sampling|>": [6],
                "A": [8],
                "B": [9],
            }
            if text in special:
                return special[text]
            s = text.strip()
            if not s:
                return [7]
            h = sum((i + 1) * ord(c) for i, c in enumerate(s[:24]))
            lead = 10 + (h % 110)
            rest = [10 + ((ord(ch) * 13 + idx) % 110) for idx, ch in enumerate(text[-32:])]
            return [lead, *rest[:10]]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            return f"tok_{ids[0]}" if ids else ""

    class _TinyNorm(nn.Module):
        def __init__(self, d: int = 0) -> None:
            super().__init__()
            self.w = nn.Parameter(torch.ones(d)) if d > 0 else None

        def forward(self, x):
            xf = x.float()
            nrm = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + 1e-6)
            return (nrm * self.w).to(x.dtype) if self.w is not None else nrm.to(x.dtype)

    class _TinyGate(nn.Module):
        def __init__(self, d: int, rank: int) -> None:
            super().__init__()
            self.norm = _TinyNorm(0)
            self.down_proj = nn.Linear(2 * d, rank, bias=False)
            self.forget_gate_proj = nn.Linear(rank, d, bias=False)
            self.message_gate_proj = nn.Linear(rank, d, bias=False)

        def forward(self, h, m):
            hf, mf = h.float(), m.float()
            gh = self.down_proj(torch.cat([self.norm(hf), self.norm(mf)], dim=-1))
            fg = torch.tanh(self.forget_gate_proj(gh))
            mg = torch.tanh(self.message_gate_proj(gh))
            return (1.0 - fg) * hf + mg * mf

    class _TinyBlockInput(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.query = nn.Parameter(torch.randn(d) * 0.05)

        def forward(self, b_st, h_st):
            stk = torch.cat([b_st, h_st.unsqueeze(-2)], dim=-2).float()
            stk = stk * torch.rsqrt(stk.square().mean(dim=-1, keepdim=True) + 1e-6)
            sc = torch.einsum("btsd,d->bts", stk, self.query.float()) * (stk.shape[-1] ** -0.5)
            pr = torch.softmax(sc, dim=-1)
            return (torch.einsum("bts,btsd->btd", pr, stk),)

    class _TinyMsgInput(nn.Module):
        def __init__(self, n_loop: int, d: int) -> None:
            super().__init__()
            self.query = nn.Parameter(torch.randn(n_loop, 4, d) * 0.05)

        def forward(self, prev_b, prev_h, li: int):
            stk = torch.cat([prev_b, prev_h.unsqueeze(-2)], dim=-2).float()
            stk = stk * torch.rsqrt(stk.square().mean(dim=-1, keepdim=True) + 1e-6)
            sc = torch.einsum("btsd,cd->btcs", stk, self.query[li].float()) * (stk.shape[-1] ** -0.5)
            pr = torch.softmax(sc, dim=-1)
            return torch.einsum("btcs,btsd->btcd", pr, stk)

    class _TinyLayer(nn.Module):
        attention_type = "full_attention"

        def __init__(self, d: int, rank: int, is_loop: bool, is_boundary: bool) -> None:
            super().__init__()
            self.is_loop = is_loop
            self.is_block_boundary = is_boundary
            self.b_in = _TinyBlockInput(d)
            self.wq = nn.Linear(d, d, bias=False)
            self.wk = nn.Linear(d, d, bias=False)
            self.wv = nn.Linear(d, d, bias=False)
            self.mlp = nn.Linear(d, d, bias=False)
            if is_loop:
                self.q_message_gate = _TinyGate(d, rank)
                self.k_message_gate = _TinyGate(d, rank)
                self.v_message_gate = _TinyGate(d, rank)
                self.mlp_message_gate = _TinyGate(d, rank)

        def forward(self, hidden_states, block_states, *args, **kwargs):
            _ = kwargs
            messages = args[0] if (self.is_loop and len(args) >= 1) else None
            base = self.b_in(block_states, hidden_states)[0] if block_states.shape[-2] > 0 else hidden_states.float()
            if self.is_loop and messages is not None:
                qm, km, vm, mm = messages.unbind(-2)
                qh = self.q_message_gate(base, qm)
                kh = self.k_message_gate(base, km)
                vh = self.v_message_gate(base, vm)
            else:
                qh = kh = vh = base
                mm = None
            a_out = 0.5 * (self.wq(qh.to(hidden_states.dtype)) + self.wk(kh.to(hidden_states.dtype))) * torch.tanh(
                self.wv(vh.to(hidden_states.dtype))
            )
            h2 = hidden_states.float() + 0.25 * a_out.float()
            m_in = self.mlp_message_gate(h2, mm) if (self.is_loop and mm is not None) else h2
            return (h2 + 0.25 * torch.tanh(self.mlp(m_in.to(hidden_states.dtype)).float())).to(hidden_states.dtype)

    @dataclass
    class _Out:
        logits: object

    class _TinyBiophysicalModel(nn.Module):
        def __init__(self, d: int = 24, vocab: int = 128) -> None:
            super().__init__()
            tcfg = SimpleNamespace(
                hidden_size=d,
                num_hidden_layers=8,
                loop_layer_index_start=2,
                loop_layer_index_end=6,
                loop_num_passes=3,
                loop_gate_rank=6,
            )
            self.config = SimpleNamespace(text_config=tcfg)
            embed = nn.Embedding(vocab, d)
            layers = nn.ModuleList(
                [_TinyLayer(d, 6, is_loop=(2 <= i < 6), is_boundary=(i % 2 == 0)) for i in range(8)]
            )
            msg_in = _TinyMsgInput(4, d)
            fin_in = _TinyBlockInput(d)
            norm = _TinyNorm(d)
            self.language_model = SimpleNamespace(
                _is_mock_tiny=True,
                config=tcfg,
                embed_tokens=embed,
                layers=layers,
                message_input=msg_in,
                final_input=fin_in,
                norm=norm,
            )
            self._mods = nn.ModuleList([embed, layers, msg_in, fin_in, norm])
            self.model = SimpleNamespace(language_model=self.language_model)
            self.lm_head = nn.Linear(d, vocab, bias=False)
            self.logit_softcap = nn.Identity()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            _ = kwargs
            lm = self.language_model
            h = lm.embed_tokens(input_ids)
            if attention_mask is not None:
                h = h * attention_mask.unsqueeze(-1).to(h.dtype)
            b = h.new_zeros(*h.shape[:-1], 0, h.shape[-1])
            for lyr in lm.layers[:2]:
                nh = lyr(h, b)
                if lyr.is_block_boundary:
                    b = torch.cat([b, h.unsqueeze(-2)], dim=-2)
                h = nh
            pre_h, pre_b = h, b
            n_pre = pre_b.shape[-2]
            for lyr in lm.layers[2:6]:
                nh = lyr(h, b, None)
                if lyr.is_block_boundary:
                    b = torch.cat([b, h.unsqueeze(-2)], dim=-2)
                h = nh
            for _p in range(1, lm.config.loop_num_passes):
                prev_b = b[..., n_pre + 1 :, :]
                prev_h = h
                h, b = pre_h, pre_b
                for li, lyr in enumerate(lm.layers[2:6]):
                    msgs = lm.message_input(prev_b, prev_h, li)
                    nh = lyr(h, b, msgs)
                    if lyr.is_block_boundary:
                        b = torch.cat([b, h.unsqueeze(-2)], dim=-2)
                    h = nh
            for lyr in lm.layers[6:]:
                nh = lyr(h, b)
                if lyr.is_block_boundary:
                    b = torch.cat([b, h.unsqueeze(-2)], dim=-2)
                h = nh
            fin = lm.norm(lm.final_input(b, h)[0])
            return _Out(logits=self.logit_softcap(self.lm_head(fin)))

    torch.manual_seed(31)
    model = _TinyBiophysicalModel(d=24, vocab=128)
    tok = _MockTok()
    return BiophysicalInklingController(model, tok, model_label="tiny-phase7-safety-test")


def test_phase7_block_a_end_to_end_and_resumability() -> None:
    try:
        import torch  # noqa: F401
    except ImportError:
        try:
            import pytest
            pytest.skip("torch not installed in lightweight local venv")
        except ImportError:
            return

    from nullbridge.biophysical_safety import (
        run_exp_a1_singular_mode_alignment,
        run_exp_a2_safety_wht,
        run_exp_a2_selective_nullbridge,
        run_phase7_block_a,
    )

    ctrl = _build_tiny_controller()
    bm_full = build_nullbridge_benchmarks()
    bio_full = build_phase6_stimulus_suite()

    # Keep probe pairs small (3 per concept = 18) for fast CPU execution
    mini_probes = []
    for cname in (
        "sycophancy",
        "authority_framing",
        "confident_formatting",
        "moral_spite",
        "epistemic_evidence",
        "affect_valence",
    ):
        mini_probes.extend([p for p in bm_full["probe_pairs"] if p.concept == cname][:3])

    bm_mini = {
        **bm_full,
        "probe_pairs": mini_probes,
        "eval_covert_sycophancy": bm_full["eval_covert_sycophancy"][:4],
        "eval_verifier": bm_full["eval_verifier"][:4],
        "eval_alias_guardrail": bm_full["eval_alias_guardrail"][:4],
        "eval_moral_pairs": bm_full["eval_moral_pairs"][:4],
        "eval_kl_passages": bm_full["eval_kl_passages"][:2],
    }
    bio_mini = {
        "factual_2hop": bio_full["factual_2hop"][:4],
        "arithmetic_2step": bio_full["arithmetic_2step"][:4],
    }

    # 1. Standalone A1, A2.1, A2.2 smoke checks
    a1 = run_exp_a1_singular_mode_alignment(
        ctrl,
        bm=bm_mini,
        bio_suite=bio_mini,
        n_per_cohort=2,
        top_k=4,
        n_svd_iters=6,
        oversample=2,
        batch_size=2,
        n_boot=200,
    )
    assert set(a1["cohorts"].keys()) == {
        "neutral_reasoning",
        "verifier_paired",
        "covert_anchor_paired",
        "moral_valuation_paired",
    }
    assert a1["isotropic_null_baselines"]["k"] == 4
    assert a1["cohorts"]["neutral_reasoning"]["svd_diagnostics"]["max_v_orth_err"] < 1e-5
    assert a1["cohorts"]["neutral_reasoning"]["svd_diagnostics"]["max_u_orth_err"] < 1e-5

    a2_wht = run_exp_a2_safety_wht(
        ctrl,
        bm=bm_mini,
        n_passes=3,
        n_covert=4,
        n_moral=4,
        n_verifier=4,
        batch_size=4,
        n_boot=200,
    )
    assert a2_wht["algebraic_audit"]["valid"] is True
    assert a2_wht["algebraic_audit"]["max_parseval_variance_error"] < 1e-12
    assert a2_wht["algebraic_audit"]["max_reconstruction_error"] < 1e-12
    assert "P1_qxk_minus_vxmlp_synergy" in a2_wht["predictions"]
    assert "P2_vmlp_gt_qk_moral_main_effects" in a2_wht["predictions"]

    a2_sel = run_exp_a2_selective_nullbridge(
        ctrl,
        bm=bm_mini,
        bio_suite=bio_mini,
        n_passes=3,
        n_covert=4,
        n_moral=4,
        n_verifier=4,
        n_alias=4,
        n_reasoning_per_regime=3,
        n_kl_passages=2,
        batch_size=4,
    )
    expected_conds = [c for c, _ch in SELECTIVE_NULLBRIDGE_CONDITIONS]
    assert list(a2_sel["conditions"].keys()) == expected_conds
    assert a2_sel["reset_restoration_max_abs_logit_diff"] == 0.0
    for cname in expected_conds:
        c_obj = a2_sel["conditions"][cname]
        for req_key in (
            "kl_eval_nats",
            "covert_capture_rate",
            "moral_bias_delta_lo",
            "tainted_provenance_delta_lo",
            "verifier_hacked_acc",
            "verifier_susceptibility_lo",
            "alias_guardrail_acc",
            "biophysical_reasoning_acc",
        ):
            assert req_key in c_obj

    # 2. Unified resumable runner `run_phase7_block_a` with atomic checkpointing
    with tempfile.TemporaryDirectory() as td:
        ckpt = Path(td) / "phase7_block_a.json"
        res1 = run_phase7_block_a(
            ctrl,
            bm=bm_mini,
            bio_suite=bio_mini,
            checkpoint_path=ckpt,
            n_passes=3,
            a1_n_per_cohort=2,
            a1_top_k=4,
            a1_n_svd_iters=5,
            a1_oversample=2,
            a1_batch_size=2,
            a2_n_covert=4,
            a2_n_moral=4,
            a2_n_verifier=4,
            a2_n_alias=4,
            a2_n_reasoning_per_regime=3,
            a2_n_kl_passages=2,
            batch_size=4,
            n_boot=200,
        )
        assert res1["completed"] is True
        assert ckpt.exists()
        assert ckpt.with_suffix(".subspace.npz").exists()

        # Simulate interruption before `exp_a2_selective_nullbridge` and verify clean resume
        saved = json.loads(ckpt.read_text(encoding="utf-8"))
        saved.pop("exp_a2_selective_nullbridge")
        saved["completed"] = False
        saved["exp_a1_singular_mode_alignment"]["_sentinel_preserved"] = True
        ckpt.write_text(json.dumps(saved, indent=2), encoding="utf-8")

        res2 = run_phase7_block_a(
            ctrl,
            bm=bm_mini,
            bio_suite=bio_mini,
            checkpoint_path=ckpt,
            n_passes=3,
            a1_n_per_cohort=2,
            a1_top_k=4,
            a1_n_svd_iters=5,
            a1_oversample=2,
            a1_batch_size=2,
            a2_n_covert=4,
            a2_n_moral=4,
            a2_n_verifier=4,
            a2_n_alias=4,
            a2_n_reasoning_per_regime=3,
            a2_n_kl_passages=2,
            batch_size=4,
            n_boot=200,
        )
        assert res2["completed"] is True
        assert res2["exp_a1_singular_mode_alignment"].get("_sentinel_preserved") is True
        assert "exp_a2_selective_nullbridge" in res2
        assert res2["exp_a2_selective_nullbridge"]["reset_restoration_max_abs_logit_diff"] == 0.0


if __name__ == "__main__":
    test_phase7_safety_zero_em_dash_and_moral_pairs_audit()
    test_deflated_topk_svd_synthetic_24x24()
    test_phase7_block_a_end_to_end_and_resumability()
    print("[OK] Phase 7 Block A Safety-Biophysics CPU & mock-PyTorch self-test passed.")
