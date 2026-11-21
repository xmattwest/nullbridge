"""CPU Unit Tests for Formal Afriat / GARP Engine, Lexical Firewall, and Thought Parsing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nullbridge.afriat_inkling import split_inkling_thought_and_public
from nullbridge.afriat_stimuli import (
    BASE_INTERSECTING_INTERCEPTS_24,
    GRID_SHARES_11,
    audit_all_afriat_inkling_stimuli,
    build_garp_budget_suites,
)
from nullbridge.garp import (
    audit_revealed_preferences,
    compute_bronars_power,
    compute_exact_afriat_ccei,
)


def test_afriat_inkling_stimulus_firewall_and_counts() -> None:
    report = audit_all_afriat_inkling_stimuli()
    assert report["valid"], f"Firewall errors: {report['errors']}"
    assert report["counts"]["concept_pairs"] == 200
    assert report["counts"]["transaction_pairs"] == 96
    assert report["counts"]["sanity_items"] == 24
    assert report["counts"]["garp_regimes"] == 4
    assert report["counts"]["garp_total_budget_lines"] == 576
    assert report["counts"]["wtp_total_items"] == 392
    assert report["counts"]["debt_math_items"] == 32
    assert report["counts"]["open_decision_items"] == 24


def test_afriat_theorem_exact_rational_agents_pass_garp_with_ccei_one() -> None:
    m = np.asarray(BASE_INTERSECTING_INTERCEPTS_24, dtype=np.float64)
    prices = 1.0 / m
    shares = np.asarray(GRID_SHARES_11, dtype=np.float64)
    chosen_cd = []
    chosen_lin = []
    chosen_taint = []
    for m1, m2 in m:
        cands = np.column_stack([shares * m1, (1.0 - shares) * m2])
        u_cd = (np.maximum(cands[:, 0], 1e-6) ** 0.63) * (np.maximum(cands[:, 1], 1e-6) ** 0.37)
        chosen_cd.append(cands[int(np.argmax(u_cd))])
        u_lin = cands[:, 0] + cands[:, 1]
        chosen_lin.append(cands[int(np.argmax(u_lin))])
        u_tnt = cands[:, 0] + 0.55 * cands[:, 1]
        chosen_taint.append(cands[int(np.argmax(u_tnt))])

    for arr in (chosen_cd, chosen_lin, chosen_taint):
        res = audit_revealed_preferences(prices, np.asarray(arr))
        assert res.passes_garp
        assert res.garp_violation_pairs == 0
        assert res.afriat_ccei == 1.0
        assert res.max_afriat_inequality_violation <= 1e-7


def test_warp_and_garp_violation_detection_and_exact_ccei() -> None:
    prices = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    bundles = np.array([[1.0, 2.5], [2.5, 1.0]], dtype=np.float64)

    ccei = compute_exact_afriat_ccei(prices, bundles)
    assert abs(ccei - 0.75) < 1e-5

    res = audit_revealed_preferences(prices, bundles)
    assert not res.passes_garp
    assert res.garp_violation_pairs == 2
    assert res.warp_violation_pairs == 1
    assert abs(res.afriat_ccei - 0.75) < 1e-5
    assert res.max_afriat_inequality_violation <= 1e-6


def test_bronars_power_on_24_intersecting_budgets() -> None:
    bronars = compute_bronars_power(
        np.asarray(BASE_INTERSECTING_INTERCEPTS_24, dtype=np.float64),
        grid_shares=GRID_SHARES_11,
        n_simulations=400,
        seed=123,
    )
    assert bronars["bronars_power_1_minus_pass"] > 0.97
    assert bronars["mean_random_ccei"] < 0.80

    suites = build_garp_budget_suites()
    assert set(suites.keys()) == {
        "social_neutral",
        "social_provoked",
        "provenance_clean_vs_tainted",
        "framed_compliance_decoy",
    }


def test_inkling_thinking_parser() -> None:
    raw_closed = (
        "<|message_model|><|content_thinking|>The split is unfair, yet $20 beats $0.<|end_message|>"
        "<|message_model|><|content_text|>DECISION_USD: $150\nRATIONALE: Standard liquidity reserve.<|end_message|>"
    )
    tht, pub, pres, closed = split_inkling_thought_and_public(raw_closed)
    assert pres and closed
    assert "unfair" in tht
    assert "DECISION_USD: $150" in pub

    raw_open = "<|message_model|><|content_thinking|>Still deliberating on the balance sheet..."
    tht2, pub2, pres2, closed2 = split_inkling_thought_and_public(raw_open)
    assert pres2 and not closed2
    assert pub2 == ""
    assert "deliberating" in tht2


def test_afriat_inkling_controller_smoke_with_torch() -> None:
    try:
        import torch
    except ImportError:
        try:
            import pytest
            pytest.skip("torch not installed in local CPU venv")
        except ImportError:
            return

    from dataclasses import dataclass
    from types import SimpleNamespace
    nn = torch.nn
    from nullbridge.afriat_inkling import (
        AfriatInklingController,
        _activate_afriat_condition,
        run_block1_moral_atlas,
    )
    from nullbridge.afriat_stimuli import MENU_LETTERS_11, build_garp_budget_suites

    class _DummyTok:
        pad_token_id = 0
        eos_token_id = 2

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            _ = add_special_tokens
            if text in MENU_LETTERS_11:
                return [10 + MENU_LETTERS_11.index(text)]
            if text in ("<|end_message|>", "<|content_model_end_sampling|>", "<|endoftext|>"):
                return [2]
            return [25 + ((ord(ch) + idx * 11) % 95) for idx, ch in enumerate(text[:512])] or [25]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            return "120\nRATIONALE: Standard liquidity allocation."

    class _TinyGate(nn.Module):
        def __init__(self, d: int, r: int) -> None:
            super().__init__()
            self.down_proj = nn.Linear(2 * d, r, bias=False)
            self.fg = nn.Linear(r, d, bias=False)
            self.mg = nn.Linear(r, d, bias=False)

        def forward(self, h, m):
            hf, mf = h.float(), m.float()
            gh = self.down_proj(torch.cat([hf, mf], dim=-1))
            return ((1.0 - torch.tanh(self.fg(gh))) * hf + torch.tanh(self.mg(gh)) * mf).to(h.dtype)

    class _TinyMsgIn(nn.Module):
        def __init__(self, n_l: int, d: int) -> None:
            super().__init__()
            self.query = nn.Parameter(torch.randn(n_l, 4, d) * 0.05)

        def forward(self, prev_b, prev_h, li: int):
            st = torch.cat([prev_b, prev_h.unsqueeze(-2)], dim=-2).float()
            q = self.query[li].float()
            p = torch.softmax(torch.einsum("btsd,cd->btcs", st, q), dim=-1)
            return torch.einsum("btcs,btsd->btcd", p, st)

    class _TinyLyr(nn.Module):
        def __init__(self, d: int, is_loop: bool) -> None:
            super().__init__()
            self.is_loop = is_loop
            self.lin = nn.Linear(d, d, bias=False)
            if is_loop:
                self.q_message_gate = _TinyGate(d, 6)
                self.k_message_gate = _TinyGate(d, 6)
                self.v_message_gate = _TinyGate(d, 6)
                self.mlp_message_gate = _TinyGate(d, 6)

        def forward(self, h, msgs=None):
            if self.is_loop and msgs is not None:
                h = self.q_message_gate(h, msgs[..., 0, :])
                h = self.mlp_message_gate(h, msgs[..., 3, :])
            pos = torch.arange(1, h.shape[1] + 1, device=h.device, dtype=h.dtype).view(1, -1, 1)
            causal_ctx = torch.cumsum(h, dim=1) / pos
            return h + 0.15 * torch.tanh(self.lin(0.5 * (h + causal_ctx)))

    @dataclass
    class _Out:
        logits: object = None
        last_hidden_state: object = None
        cache: None = None

    class _TinyTextModel(nn.Module):
        def __init__(self, tcfg, emb, lyrs, msg_in, norm) -> None:
            super().__init__()
            self.config = tcfg
            self.embed_tokens = emb
            self.layers = lyrs
            self.message_input = msg_in
            self.norm = norm

        def forward(self, input_ids, attention_mask=None, **kwargs):
            _ = kwargs
            h = self.embed_tokens(input_ids)
            if attention_mask is not None:
                h = h * attention_mask.unsqueeze(-1).to(h.dtype)
            h = self.layers[0](h)
            pre_h = h
            b_list = []
            for lyr in self.layers[1:5]:
                h = lyr(h)
                b_list.append(h.unsqueeze(-2))
            for _ in range(1, self.config.loop_num_passes):
                prev_b = torch.cat(b_list, dim=-2)
                prev_h = h
                h = pre_h
                b_list = []
                for li, lyr in enumerate(self.layers[1:5]):
                    m = self.message_input(prev_b, prev_h, li)
                    h = lyr(h, m)
                    b_list.append(h.unsqueeze(-2))
            h = self.layers[5](h)
            return _Out(last_hidden_state=self.norm(h))

    class _TinyModel(nn.Module):
        def __init__(self, d: int = 32, vocab: int = 128) -> None:
            super().__init__()
            tcfg = SimpleNamespace(
                hidden_size=d,
                num_hidden_layers=6,
                loop_layer_index_start=1,
                loop_layer_index_end=5,
                loop_num_passes=3,
                loop_gate_rank=6,
            )
            self.config = SimpleNamespace(text_config=tcfg)
            emb = nn.Embedding(vocab, d)
            lyrs = nn.ModuleList([_TinyLyr(d, is_loop=(1 <= i < 5)) for i in range(6)])
            msg_in = _TinyMsgIn(4, d)
            norm = nn.LayerNorm(d)
            self.language_model = _TinyTextModel(tcfg, emb, lyrs, msg_in, norm)
            self.model = SimpleNamespace(language_model=self.language_model)
            self.lm_head = nn.Linear(d, vocab, bias=False)
            self.logit_softcap = nn.Identity()

        def forward(self, input_ids, attention_mask=None, **kwargs):
            tout = self.language_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
            return _Out(logits=self.lm_head(tout.last_hidden_state), last_hidden_state=tout.last_hidden_state)

    torch.manual_seed(11)
    ctrl = AfriatInklingController(_TinyModel(), _DummyTok(), model_label="tiny-afriat-inkling")
    b1, basis = run_block1_moral_atlas(ctrl, pass_depths=(1, 3), probe_passes=3, batch_size=32)
    assert b1["bos_sink_orthogonality"]["msg_8d_bos_overlap"] < 1e-12
    assert b1["zero_flop_8d_fold_audit"]["max_query_null_leakage"] < 1e-6
    for arch_name, arch_c in b1["financial_cartography"].items():
        assert any(abs(v["partial_beta"]) > 1e-6 for v in arch_c["message_bank_projection"]["concepts"].values()), arch_name

    suites = build_garp_budget_suites()
    m_scores = ctrl.score_counterbalanced_11way_menus(suites["social_neutral"][:6], n_passes=3, batch_size=6)
    assert m_scores["probs"].shape == (6, 11)

    for cname in ("unclamped", "msg_homo_economicus_8d", "msg_qkv_8d", "msg_mlp_8d", "zero_flop_8d"):
        _activate_afriat_condition(ctrl, cname, basis)
    ctrl.reset_interventions()

    with ctrl.message_steering(basis["msg_units"]["spite"], delta_scale=0.5):
        recs = ctrl.generate_completions(["Test prompt"], n_passes=3, reasoning_effort="none", assistant_prefix="DECISION_USD: $", max_new_tokens=6)
        assert len(recs) == 1
    recs_th = ctrl.generate_completions(["Test prompt"], n_passes=3, reasoning_effort="medium", max_new_tokens=6)
    assert len(recs_th) == 1 and recs_th[0]["public_text"] != ""


def test_locked_phase5_afriat_inkling_artifacts() -> None:
    root = Path(__file__).resolve().parents[1]
    res_p = root / "results" / "phase5_inkling_afriat_garp" / "afriat_inkling_results.json"
    ana_p = root / "results" / "phase5_inkling_afriat_garp" / "afriat_inkling_analysis.json"
    if not res_p.is_file():
        return
    res = json.loads(res_p.read_text(encoding="utf-8"))
    ana = json.loads(ana_p.read_text(encoding="utf-8"))
    assert res["completed"] is True
    assert res["varloop_research"]["completed"] is True
    assert res["inkling_2_mini"]["completed"] is True
    assert ana["garp_summary"]["bronars_power_pct"] >= 97.0
    assert ana["garp_summary"]["inkling_k3_by_condition"]["unclamped"]["mean_modal_ccei"] > 0.95
    assert ana["compute_accounting"]["active_pods_remaining"] == 0



if __name__ == "__main__":
    test_afriat_inkling_stimulus_firewall_and_counts()
    test_afriat_theorem_exact_rational_agents_pass_garp_with_ccei_one()
    test_warp_and_garp_violation_detection_and_exact_ccei()
    test_bronars_power_on_24_intersecting_budgets()
    test_inkling_thinking_parser()
    test_afriat_inkling_controller_smoke_with_torch()
    print("[OK] All Afriat-Inkling CPU & synthetic PyTorch pre-flight checks passed.")
