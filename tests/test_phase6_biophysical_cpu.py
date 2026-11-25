"""CPU + mock-PyTorch unit tests for Phase 6 (`phase6_inkling_biophysical_loop`).

Runs in both the lightweight local `.venv` (NumPy-only) and on the RunPod pre-flight
environment (where PyTorch is installed, exercising the full 5-experiment unrolled
controller, autograd VJP Fisher/conductance/power-iteration routines, analysis, and figures).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile
from types import SimpleNamespace

import numpy as np

from nullbridge.biophysical_stimuli import (
    EM_DASH,
    audit_biophysical_stimuli,
    build_phase6_stimulus_suite,
)
from nullbridge.biophysical_wht import (
    build_walsh_hadamard_design_matrix,
    compute_exact_wht_16,
    enumerate_16_masks,
    estimate_commitment_horizon,
    summarize_hysteresis_curve,
)


def test_phase6_stimulus_suite_audit_and_zero_em_dash() -> None:
    suite = build_phase6_stimulus_suite()
    audit = audit_biophysical_stimuli(suite)
    assert audit["valid"], f"Stimulus audit failed: {audit['errors']}"
    assert audit["total_paired_items"] == 160
    assert audit["canonical_anchors"] == ["New Delhi", "7", "Saturday"]

    # Also verify zero em dashes across all Phase 6 Python source files
    repo_root = Path(__file__).resolve().parents[1]
    for rel in (
        "src/nullbridge/biophysical_stimuli.py",
        "src/nullbridge/biophysical_wht.py",
        "src/nullbridge/biophysical_inkling.py",
        "src/nullbridge/biophysical_analysis.py",
        "src/nullbridge/biophysical_figures.py",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert EM_DASH not in txt, f"Em dash found in {rel}"


def test_exact_16_term_walsh_hadamard_algebra_and_parseval() -> None:
    h_mat = build_walsh_hadamard_design_matrix()
    assert h_mat.shape == (16, 16)
    gram = (h_mat.T @ h_mat) / 16.0
    assert np.allclose(gram, np.eye(16), atol=1e-12)

    # Construct synthetic ground-truth function:
    # f(z) = 1.0 + 0.5*z_Q + 0.4*z_K + 0.35*(z_Q * z_K) + 0.1*z_V + noise
    rng = np.random.default_rng(42)
    n_items = 40
    cube = np.zeros((16, n_items), dtype=np.float64)
    for m, bits, _code, _lbl in enumerate_16_masks():
        zq, zk, zv, zm = [1.0 if b == 1 else -1.0 for b in bits]
        base = 1.0 + 0.5 * zq + 0.4 * zk + 0.35 * (zq * zk) + 0.1 * zv - 0.05 * (zv * zm)
        cube[m] = base + rng.normal(0.0, 0.03, size=n_items)

    res = compute_exact_wht_16(cube, n_boot=400, seed=11)
    assert res["max_reconstruction_error"] < 1e-12
    assert res["parseval_variance_error"] < 1e-12
    # Coalition effect for Q is ~ 2 * 0.5 = 1.0; for QxK is ~ 4 * 0.35 = 1.4
    assert abs(res["terms"]["Q"]["coalition_effect"] - 1.0) < 0.05
    assert abs(res["terms"]["QxK"]["coalition_effect"] - 1.4) < 0.06
    assert res["qxk_minus_vxmlp_synergy"]["mean"] > 1.2
    assert abs(sum(res["order_variance_shares"].values()) - 1.0) < 1e-5


def test_commitment_horizon_and_hysteresis_summaries() -> None:
    hz = estimate_commitment_horizon(["stem", "pass_1", "pass_2", "pass_3"], [-3.0, -1.0, 2.0, 4.5])
    assert hz["crossed_zero"] is True
    assert hz["crossing_stage_label"] == "pass_2"
    assert abs(hz["interpolated_stage_position"] - (1.0 + 1.0 / 3.0)) < 1e-3

    hy = summarize_hysteresis_curve(
        betas=[0.0, 0.25, 0.5, 0.75, 1.0],
        p2_foil_prob=[0.02, 0.18, 0.42, 0.65, 0.85],
        p3_foil_prob=[0.01, 0.04, 0.12, 0.48, 0.82],
        coda_target_acc=[0.95, 0.92, 0.86, 0.45, 0.12],
        coda_foil_capture=[0.01, 0.03, 0.10, 0.50, 0.84],
    )
    assert hy["peak_pass3_foil_suppression"] >= 0.29
    assert hy["critical_bifurcation_beta"] is not None
    assert 0.5 < hy["critical_bifurcation_beta"] < 0.8


def test_phase6_biophysical_end_to_end_mock_pytorch() -> None:
    try:
        import torch
    except ImportError:
        try:
            import pytest
            pytest.skip("torch not installed in lightweight local venv")
        except ImportError:
            return

    nn = torch.nn
    from nullbridge.biophysical_analysis import run_phase6_biophysical_analysis
    from nullbridge.biophysical_figures import render_phase6_figure_suite
    from nullbridge.biophysical_inkling import (
        BiophysicalInklingController,
        run_exp1_wht_synaptic_knockouts,
        run_exp2_gain_vs_amplitude,
        run_exp3_fisher_and_coda_shunt,
        run_exp4_hysteresis_and_pulse_chase,
        run_exp5_neuromodulatory_mode_switching,
        run_preflight_equivalence_audit,
    )

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
            }
            if text in special:
                return special[text]
            s = text.strip()
            if not s:
                return [7]
            # Deterministic distinct token IDs from hashed prefix + tail
            h = sum((i + 1) * ord(c) for i, c in enumerate(s[:24]))
            lead = 10 + (h % 110)
            rest = [10 + ((ord(ch) * 13 + idx) % 110) for idx, ch in enumerate(text[-32:])]
            return [lead, *rest[:12]]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            rev = {3: "<|message_model|>", 4: "<|content_text|>", 5: "<|content_thinking|>", 2: "<|end_message|>"}
            if len(ids) == 1 and ids[0] in rev:
                return rev[ids[0]]
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

    torch.manual_seed(19)
    m = _TinyBiophysicalModel(d=24, vocab=128)
    tok = _MockTok()
    ctrl = BiophysicalInklingController(m, tok, model_label="tiny-biophys-test")

    eq = run_preflight_equivalence_audit(ctrl)
    assert eq["valid"]

    full_suite = build_phase6_stimulus_suite()
    mini_suite = {
        "factual_2hop": full_suite["factual_2hop"][:4],
        "arithmetic_2step": full_suite["arithmetic_2step"][:4],
        "calendar_2step": full_suite["calendar_2step"][:4],
        "reasoning_120": [
            *full_suite["factual_2hop"][:4],
            *full_suite["arithmetic_2step"][:4],
            *full_suite["calendar_2step"][:4],
        ],
        "chat_shunt_triplets": full_suite["chat_shunt_triplets"][:4],
    }

    e1 = run_exp1_wht_synaptic_knockouts(ctrl, mini_suite, batch_size=6)
    e2 = run_exp2_gain_vs_amplitude(ctrl, mini_suite, lens_per_regime=3, jacobian_per_regime=2, batch_size=6)
    e3 = run_exp3_fisher_and_coda_shunt(ctrl, mini_suite, fisher_per_regime=2, n_rademacher=2, fisher_bsz=4)
    e4 = run_exp4_hysteresis_and_pulse_chase(ctrl, mini_suite, items_per_regime=2, batch_size=6)
    e5 = run_exp5_neuromodulatory_mode_switching(ctrl, mini_suite, clock_per_regime=2, batch_size=6)

    payload = {
        "phase": "phase6_inkling_biophysical_loop",
        "metadata": {
            "model_id": "tiny-biophys-test",
            "gpu_name": "cpu",
            "d_model": 24,
            "n_layers": 8,
            "loop_start": 2,
            "loop_end": 6,
            "default_passes": 3,
            "gate_rank": 6,
            "cuda_allocated_gb": 0.0,
            "load_seconds": 0.01,
        },
        "stimulus_audit": audit_biophysical_stimuli(full_suite),
        "preflight_equivalence": eq,
        "exp1_wht_synaptic_knockouts": e1,
        "exp2_gain_vs_amplitude": e2,
        "exp3_fisher_and_coda_shunt": e3,
        "exp4_hysteresis_pulse_chase": e4,
        "exp5_neuromodulatory_switching": e5,
        "runtime_seconds": 0.5,
        "completed": True,
    }

    with tempfile.TemporaryDirectory() as td:
        td_p = Path(td)
        res_p = td_p / "phase6_biophysical_results.json"
        import json

        res_p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        summary = run_phase6_biophysical_analysis(
            res_p, td_p / "biophysical_analysis.json", td_p / "synthesis.md", gpu_spend_usd=0.0, pod_wall_minutes=0.1
        )
        assert summary["phase"] == "phase6_inkling_biophysical_loop"
        os_env_prev = None
        import os

        os_env_prev = os.environ.get("CHROME_BIN")
        os.environ["CHROME_BIN"] = "/nonexistent_chrome_skip_png_in_unit_test"
        try:
            fmeta = render_phase6_figure_suite(res_p, td_p / "figs")
            assert fmeta["n_svg"] == 5
        finally:
            if os_env_prev is None:
                os.environ.pop("CHROME_BIN", None)
            else:
                os.environ["CHROME_BIN"] = os_env_prev


if __name__ == "__main__":
    test_phase6_stimulus_suite_audit_and_zero_em_dash()
    test_exact_16_term_walsh_hadamard_algebra_and_parseval()
    test_commitment_horizon_and_hysteresis_summaries()
    test_phase6_biophysical_end_to_end_mock_pytorch()
    print("[OK] Phase 6 Biophysical CPU & mock-PyTorch self-test passed.")
