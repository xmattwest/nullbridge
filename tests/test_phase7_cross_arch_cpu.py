"""CPU + mock-PyTorch unit tests for Phase 7 Experiment Block B (Varloop K=1..7 + Huginn-0125 K=1..8).

Validates:
  1. Pure-NumPy 4-term Walsh-Hadamard algebra (`Parseval < 1e-12`, exact reconstruction `< 1e-12`)
     and zero em dashes across all Block B files.
  2. Verified TSV schema and SHA-256 format of `config/manifests/huginn-0125.tsv`.
  3. Multi-K (`K = 1, 2, 3, 5, 7`) Varloop `_TinyBiophysicalModel` equivalence (`0.00e+00`),
     `masks_by_pass` / `gammas_by_pass`, and resumable `run_phase7_varloop_suite`.
  4. Mock `_TinyRavenModel` (`d = 24`, 2 prelude, 4 core, 2 coda, `adapter: Linear(48, 24)`)
     verifying `unrolled_huginn_forward` equivalence (`<= 1e-3`), 4-mask WHT Parseval `< 1e-12`,
     and resumable `run_phase7_huginn_suite` (`H1..H5`).
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

import numpy as np

from nullbridge.biophysical_huginn import (
    build_walsh_hadamard_4,
    compute_exact_wht_4,
    enumerate_4_masks,
)
from nullbridge.biophysical_stimuli import EM_DASH, build_phase6_stimulus_suite


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def test_huginn_wht4_algebra_parseval_and_zero_em_dash() -> None:
    h4 = build_walsh_hadamard_4()
    assert h4.shape == (4, 4)
    assert np.allclose((h4.T @ h4) / 4.0, np.eye(4), atol=1e-12)

    rng = np.random.default_rng(77)
    n_items = 30
    cube = np.zeros((4, n_items), dtype=np.float64)
    for m, bits, _code, _lbl in enumerate_4_masks():
        z_s, z_r = [1.0 if b == 1 else -1.0 for b in bits]
        base = 0.8 + 0.45 * z_s + 0.30 * z_r + 0.25 * (z_s * z_r)
        cube[m] = base + rng.normal(0.0, 0.02, size=n_items)

    res = compute_exact_wht_4(cube, seed=17, n_boot=300)
    assert res["max_reconstruction_error"] < 1e-12
    assert res["parseval_variance_error"] < 1e-12
    assert abs(res["terms"]["state"]["coalition_effect"] - 0.90) < 0.05
    assert abs(res["terms"]["recall"]["coalition_effect"] - 0.60) < 0.05
    assert abs(res["terms"]["statexrecall"]["coalition_effect"] - 1.00) < 0.08
    assert abs(sum(res["order_variance_shares"].values()) - 1.0) < 1e-5

    repo_root = Path(__file__).resolve().parents[1]
    for rel in (
        "src/nullbridge/biophysical_inkling.py",
        "src/nullbridge/biophysical_huginn.py",
        "tests/test_phase7_cross_arch_cpu.py",
        "config/manifests/huginn-0125.tsv",
    ):
        txt = (repo_root / rel).read_text(encoding="utf-8")
        assert EM_DASH not in txt, f"Em dash found in {rel}"


def test_huginn_manifest_format_and_required_files() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    man_path = repo_root / "config/manifests/huginn-0125.tsv"
    assert man_path.is_file(), "Missing config/manifests/huginn-0125.tsv"

    lines = [ln for ln in man_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert lines[0] == "# path\tsize_bytes\tsha256\tsource"

    entries: dict[str, tuple[int, str, str]] = {}
    for ln in lines[1:]:
        parts = ln.split("\t")
        assert len(parts) == 4, f"Expected 4 tab-separated fields, got: {ln}"
        fpath, size_s, sha_s, src = parts
        size_i = int(size_s)
        assert size_i > 0, f"Non-positive size for {fpath}"
        assert _SHA256_RE.match(sha_s), f"Invalid SHA-256 for {fpath}: {sha_s}"
        assert src in ("git", "lfs"), f"Invalid source for {fpath}: {src}"
        entries[fpath] = (size_i, sha_s, src)

    required = (
        "config.json",
        "generation_config.json",
        "model-00001-of-00004.safetensors",
        "model-00002-of-00004.safetensors",
        "model-00003-of-00004.safetensors",
        "model-00004-of-00004.safetensors",
        "model.safetensors.index.json",
        "raven_config_minimal.py",
        "raven_modeling_minimal.py",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    for req in required:
        assert req in entries, f"Missing required Huginn manifest entry: {req}"
    for shard_i in range(1, 5):
        assert entries[f"model-0000{shard_i}-of-00004.safetensors"][2] == "lfs"


def _make_mock_tokenizer():
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
            h = sum((i + 1) * ord(c) for i, c in enumerate(s[:24]))
            lead = 10 + (h % 110)
            rest = [10 + ((ord(ch) * 13 + idx) % 110) for idx, ch in enumerate(text[-24:])]
            return [lead, *rest[:10]]

        def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
            _ = skip_special_tokens
            return f"tok_{ids[0]}" if ids else ""

    return _MockTok()


def _make_mini_suite() -> dict[str, object]:
    full_suite = build_phase6_stimulus_suite()
    return {
        "factual_2hop": full_suite["factual_2hop"][:3],
        "arithmetic_2step": full_suite["arithmetic_2step"][:3],
        "calendar_2step": full_suite["calendar_2step"][:3],
        "reasoning_120": [
            *full_suite["factual_2hop"][:3],
            *full_suite["arithmetic_2step"][:3],
            *full_suite["calendar_2step"][:3],
        ],
        "chat_shunt_triplets": full_suite["chat_shunt_triplets"][:3],
    }


def test_phase7_varloop_multi_k_equivalence_and_suite_mock_pytorch() -> None:
    try:
        import torch
    except ImportError:
        try:
            import pytest

            pytest.skip("torch not installed in lightweight local venv")
        except ImportError:
            return

    nn = torch.nn
    from nullbridge.biophysical_inkling import (
        BiophysicalInklingController,
        run_phase7_varloop_suite,
        run_varloop_preflight_equivalence,
    )

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

    torch.manual_seed(23)
    m = _TinyBiophysicalModel(d=24, vocab=128)
    tok = _make_mock_tokenizer()
    ctrl = BiophysicalInklingController(m, tok, model_label="tiny-varloop-test")

    eq = run_varloop_preflight_equivalence(ctrl, depths=(1, 2, 3, 5, 7))
    assert eq["valid"]
    for k in (1, 2, 3, 5, 7):
        assert eq["max_abs_logit_diff_by_depth"][f"K={k}"] == 0.0
    assert eq["max_abs_logit_diff_k1_vs_k7_0000_bypass"] == 0.0

    # Verify masks_by_pass and gammas_by_pass match legacy mask_p2/mask_p3/gamma_p2/gamma_p3
    inp, msk, _lens = ctrl._encode_right_padded(["Compute (3 * 2) + 1."])
    u_leg = ctrl.unrolled_forward(inp, msk, n_passes=3, mask_p2=(1, 0, 1, 0), mask_p3=(0, 1, 0, 1))
    u_new = ctrl.unrolled_forward(inp, msk, n_passes=3, masks_by_pass=[(1, 0, 1, 0), (0, 1, 0, 1)])
    assert float((u_leg["last_hidden_state"] - u_new["last_hidden_state"]).abs().max().item()) == 0.0

    g_leg = ctrl.unrolled_forward(inp, msk, n_passes=3, gamma_p2=0.5, gamma_p3=1.25)
    g_new = ctrl.unrolled_forward(inp, msk, n_passes=3, gammas_by_pass=[0.5, 1.25])
    assert float((g_leg["last_hidden_state"] - g_new["last_hidden_state"]).abs().max().item()) == 0.0

    mini_suite = _make_mini_suite()
    with tempfile.TemporaryDirectory() as td:
        ckpt_p = Path(td) / "varloop_ckpt.json"
        res = run_phase7_varloop_suite(
            ctrl,
            tok,
            mini_suite,
            depths=(1, 2, 3, 5, 7),
            wht_depths=(2, 3, 5, 7),
            fisher_depths=(5, 7),
            hysteresis_depths=(3, 5, 7),
            batch_size=6,
            jacobian_per_regime=2,
            fisher_per_regime=2,
            n_rademacher=2,
            fisher_bsz=4,
            hysteresis_per_regime=2,
            n_boot=200,
            checkpoint_path=ckpt_p,
        )
        assert res["completed"] is True
        assert ckpt_p.is_file()
        for k in (2, 3, 5, 7):
            wht_k = res["wht_depth_trajectory"]["by_depth"][f"K={k}"]["all_120"]["continuation_logprob_wht"]
            assert wht_k["parseval_variance_error"] < 1e-12
            assert wht_k["max_reconstruction_error"] < 1e-12
        assert len(res["per_pass_non_normal_gain"]["by_pass"]) == 7
        assert "K=5" in res["fisher_and_commitment"]["by_depth"]
        assert "K=7" in res["fisher_and_commitment"]["by_depth"]
        assert "K=3" in res["somatic_reset_hysteresis"]["by_depth"]
        assert "K=7" in res["somatic_reset_hysteresis"]["by_depth"]

        # Resuming from completed checkpoint returns immediately with same data
        res2 = run_phase7_varloop_suite(ctrl, tok, mini_suite, checkpoint_path=ckpt_p)
        assert res2["completed"] is True
        assert res2["preflight_equivalence"] == res["preflight_equivalence"]


def test_phase7_huginn_equivalence_wht4_and_suite_mock_pytorch() -> None:
    try:
        import torch
    except ImportError:
        try:
            import pytest

            pytest.skip("torch not installed in lightweight local venv")
        except ImportError:
            return

    nn = torch.nn
    from nullbridge.biophysical_huginn import (
        BiophysicalHuginnController,
        run_huginn_preflight_equivalence,
        run_phase7_huginn_suite,
    )

    class _TinyRMSNorm(nn.Module):
        def __init__(self, d: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d))

        def forward(self, x):
            xf = x.float()
            out = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
            return (out * self.weight.float()).to(x.dtype)

    class _TinySandwichBlock(nn.Module):
        def __init__(self, d: int, layer_id: int) -> None:
            super().__init__()
            self.layer_id = layer_id
            self.norm_1 = _TinyRMSNorm(d)
            self.attn = nn.Linear(d, d, bias=False)
            self.norm_2 = _TinyRMSNorm(d)
            self.norm_3 = _TinyRMSNorm(d)
            self.mlp = nn.Sequential(nn.Linear(d, d * 2, bias=False), nn.SiLU(), nn.Linear(d * 2, d, bias=False))
            self.norm_4 = _TinyRMSNorm(d)

        def forward(self, x, freqs_cis=None, block_idx=None, mask=None, past_key_values=None):
            _ = freqs_cis, block_idx, mask, past_key_values
            x = self.norm_2(x + 0.25 * self.attn(self.norm_1(x)))
            x = self.norm_4(x + 0.25 * self.mlp(self.norm_3(x)))
            return x

    @dataclass
    class _RavenOut:
        logits: object
        latent_states: object

    class _TinyRavenModel(nn.Module):
        def __init__(self, d: int = 24, vocab: int = 128) -> None:
            super().__init__()
            self.config = SimpleNamespace(
                n_embd=d,
                n_layers_in_prelude=2,
                n_layers_in_recurrent_block=4,
                n_layers_in_coda=2,
                padded_vocab_size=vocab,
                pad_token_id=0,
                test_time_noise=0.0,
                init_values={"std": 0.02, "embed_scale": 1.0},
            )
            self.emb_scale = 1.0
            self.transformer = nn.ModuleDict(
                dict(
                    wte=nn.Embedding(vocab, d),
                    prelude=nn.ModuleList([_TinySandwichBlock(d, i) for i in range(2)]),
                    adapter=nn.Linear(d * 2, d, bias=False),
                    core_block=nn.ModuleList([_TinySandwichBlock(d, i + 2) for i in range(4)]),
                    coda=nn.ModuleList([_TinySandwichBlock(d, i + 10) for i in range(2)]),
                    ln_f=_TinyRMSNorm(d),
                )
            )
            self.lm_head = nn.Linear(d, vocab, bias=False)
            self.register_buffer("freqs_cis", torch.ones(1, 256, d // 2, 2))

        def _maybe_inject_noise(self, x, current_step, renorm=True):
            _ = current_step, renorm
            return x

        def forward(
            self,
            input_ids,
            input_states=None,
            num_steps=None,
            init_scale: float = 1.0,
            use_cache: bool = False,
            **kwargs,
        ):
            _ = use_cache, kwargs
            freqs_cis = self.freqs_cis[:, : input_ids.shape[1]]
            input_embeds = self.transformer.wte(input_ids) * self.emb_scale
            block_idx = torch.tensor(-1, dtype=torch.long)
            for blk in self.transformer.prelude:
                block_idx = block_idx + 1
                input_embeds = blk(input_embeds, freqs_cis, block_idx, None, None)

            if input_states is None:
                x = torch.zeros_like(input_embeds) if init_scale == 0.0 else torch.randn_like(input_embeds) * 0.02
            else:
                x = input_states.clone()

            k = int(num_steps.item()) if hasattr(num_steps, "item") else int(num_steps or 8)
            for _step in range(k):
                x = self.transformer.adapter(torch.cat([x, input_embeds], dim=-1))
                for blk in self.transformer.core_block:
                    block_idx = block_idx + 1
                    x = blk(x, freqs_cis, block_idx, None, None)

            x = self.transformer.ln_f(x)
            latent_states = x.detach().clone()
            block_idx = torch.tensor(0, dtype=torch.long)
            for blk in self.transformer.coda:
                block_idx = block_idx - 1
                x = blk(x, freqs_cis, block_idx, None, None)
            x = self.transformer.ln_f(x)
            logits = self.lm_head(x).float()
            return _RavenOut(logits=logits, latent_states=latent_states)

    torch.manual_seed(31)
    rm = _TinyRavenModel(d=24, vocab=128)
    tok = _make_mock_tokenizer()
    ctrl = BiophysicalHuginnController(rm, tok, model_label="tiny-raven-test")

    eq = run_huginn_preflight_equivalence(ctrl, depths=(1, 4, 8))
    assert eq["valid"]
    for k in (1, 4, 8):
        assert eq["max_abs_logit_diff_by_depth"][f"K={k}"] <= 1e-3

    mini_suite = _make_mini_suite()
    with tempfile.TemporaryDirectory() as td:
        ckpt_p = Path(td) / "huginn_ckpt.json"
        res = run_phase7_huginn_suite(
            ctrl,
            tok,
            mini_suite,
            batch_size=6,
            lens_per_regime=2,
            jacobian_per_regime=2,
            fisher_per_regime=2,
            n_rademacher=2,
            fisher_bsz=4,
            hysteresis_per_regime=2,
            zero_state_items=6,
            n_boot=200,
            checkpoint_path=ckpt_p,
        )
        assert res["completed"] is True
        assert ckpt_p.is_file()
        assert res["prompt_mode"] in ("zero_shot", "two_shot")
        for k in (4, 8):
            wht_k = res["exp_h1_adapter_wht"]["by_depth"][f"K={k}"]["all_120"]["continuation_logprob_wht"]
            assert wht_k["parseval_variance_error"] < 1e-12
            assert wht_k["max_reconstruction_error"] < 1e-12
        assert set(res["exp_h2_gain_vs_amplitude"]["coda_conductance"].keys()) == {
            "step_1",
            "step_2",
            "step_4",
            "step_8",
        }
        assert set(res["exp_h2_gain_vs_amplitude"]["non_normal_step_amplification"].keys()) == {
            "step_1",
            "step_2",
            "step_3",
            "step_4",
            "step_6",
            "step_8",
        }
        assert "loop_to_coda_fisher_spike" in res["exp_h3_fisher_and_commitment"]
        assert "K=4" in res["exp_h4_hysteresis_pulse_chase"]["by_depth"]
        assert "K=8" in res["exp_h4_hysteresis_pulse_chase"]["by_depth"]
        assert len(res["exp_h5_adapter_conductance_sweep"]["adapter_conductance_sweep"]) == 7

        # Verify JSON serialization has zero em dashes and resumability works
        raw_json = ckpt_p.read_text(encoding="utf-8")
        assert EM_DASH not in raw_json
        loaded = json.loads(raw_json)
        assert loaded["completed"] is True

        res2 = run_phase7_huginn_suite(ctrl, tok, mini_suite, checkpoint_path=ckpt_p)
        assert res2["completed"] is True
        assert res2["preflight_equivalence"] == res["preflight_equivalence"]


if __name__ == "__main__":
    test_huginn_wht4_algebra_parseval_and_zero_em_dash()
    test_huginn_manifest_format_and_required_files()
    test_phase7_varloop_multi_k_equivalence_and_suite_mock_pytorch()
    test_phase7_huginn_equivalence_wht4_and_suite_mock_pytorch()
    print("[OK] Phase 7 Cross-Architecture CPU & mock-PyTorch self-test passed.")
