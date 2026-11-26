"""CLI entrypoint, repository hygiene guard, and Phase 0 audit runner for Nullbridge."""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any

BLOCKED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("api_secret_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("runpod_key", re.compile(r"\brpa_[A-Za-z0-9_]{15,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("ndif_key_assignment", re.compile(r"\bNDIF_API_KEY\s*=\s*['\"]?[A-Za-z0-9_-]{16,}")),
    ("azure_sas_signature", re.compile(r"[?&]sig=[A-Za-z0-9%+/=]{20,}")),
    ("local_home_path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")),
)


def audit_repo(repo_root: Path) -> list[str]:
    """Scan tracked files for accidental keys, SAS tokens, or local machine paths."""
    proc = subprocess.run(
        ["git", "-C", str(repo_root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    findings: list[str] = []
    for raw_rel in proc.stdout.decode("utf-8", errors="replace").split("\0"):
        if not raw_rel:
            continue
        file_path = repo_root / raw_rel
        if not file_path.is_file():
            continue
        try:
            text = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in BLOCKED_PATTERNS:
            for match in pattern.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"{raw_rel}:{line_no}: blocked pattern '{label}'")
    return findings


def run_phase0_audit(out_path: Path | None = None) -> dict[str, object]:
    """Execute Phase 0 algebraic, synthetic-loop, benchmark-disjointness, and NDIF readiness checks."""
    from .algebra import audit_algebraic_properties
    from .benchmarks import audit_nullbridge_benchmarks
    from .mock_loop import run_synthetic_loop_verification
    from .ndif_harness import check_ndif_readiness

    t0 = time.perf_counter()
    alg = audit_algebraic_properties()
    bm = audit_nullbridge_benchmarks()
    syn = run_synthetic_loop_verification()
    ndif = check_ndif_readiness()

    payload: dict[str, object] = {
        "phase": "phase0_audit",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "all_valid": bool(alg["valid"] and bm["valid"] and syn["valid"]),
        "algebra_audit": alg,
        "benchmark_audit": bm,
        "synthetic_loop_audit": syn,
        "ndif_readiness": ndif,
        "runtime_seconds": round(time.perf_counter() - t0, 3),
    }

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


_PHASE7_STAGES: tuple[str, ...] = (
    "inkling_block_a",
    "varloop_block_a",
    "varloop_block_b1",
    "huginn_block_b2",
)

_PHASE7_DEFAULT_LABELS: dict[str, str] = {
    "inkling_block_a": "thinkingmachines/Inkling-2-Mini",
    "varloop_block_a": "thinkingmachines/TML-d24-varloop-research",
    "varloop_block_b1": "thinkingmachines/TML-d24-varloop-research",
    "huginn_block_b2": "tomg-group-umd/huginn-0125",
}


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_everything(seed: int) -> None:
    try:
        import numpy as np

        np.random.seed(int(seed) % (2**32))
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    except ImportError:
        pass


def _cleanup_controller(ctrl: Any) -> None:
    if hasattr(ctrl, "reset_interventions"):
        try:
            ctrl.reset_interventions()
        except Exception:
            pass
    model = getattr(ctrl, "model", None)
    if model is not None:
        try:
            model.to("cpu")
        except Exception:
            pass
    del ctrl
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _stage_json_path(checkpoint_dir: Path, stage: str) -> Path:
    return checkpoint_dir / f"{stage}_results.json"


def _stage_already_done(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return bool(data.get("completed") is True)
    except Exception:
        return False


def _run_phase7_single_stage(
    stage: str,
    model_path: Path,
    checkpoint_dir: Path,
    model_label: str | None = None,
    output_path: Path | None = None,
    batch_size: int | None = None,
    n_passes: int | None = None,
    dtype: str = "auto",
    seed: int = 20260701,
    n_boot: int = 4000,
) -> dict[str, Any]:
    """Load one model, execute its Phase 7 stage with atomic checkpoints, and release GPU memory."""
    _seed_everything(seed)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stage_ckpt = _stage_json_path(checkpoint_dir, stage)
    label = model_label or _PHASE7_DEFAULT_LABELS[stage]
    t0 = time.perf_counter()

    if stage in ("inkling_block_a", "varloop_block_a"):
        from .biophysical_inkling import _load_biophysical_controller
        from .biophysical_safety import run_exp_a2_safety_wht, run_phase7_block_a

        eff_passes = int(n_passes) if n_passes is not None else (3 if stage == "inkling_block_a" else 5)
        eff_bsz = int(batch_size) if batch_size is not None else 32
        print(f"[Phase 7 | {stage}] Loading {label} from {model_path}...", flush=True)
        ctrl, meta = _load_biophysical_controller(model_path, label)
        try:
            res = run_phase7_block_a(
                ctrl,
                checkpoint_path=stage_ckpt,
                n_passes=eff_passes,
                batch_size=eff_bsz,
                n_boot=n_boot,
            )
            if stage == "varloop_block_a" and eff_passes != 3 and "exp_a2_safety_wht_k3" not in res:
                res["completed"] = False
                res["exp_a2_safety_wht_k3"] = run_exp_a2_safety_wht(
                    ctrl,
                    n_passes=3,
                    batch_size=eff_bsz,
                    n_boot=n_boot,
                )
                if "exp_a2_safety_wht_k3" not in res.get("completed_stages", []):
                    res.setdefault("completed_stages", []).append("exp_a2_safety_wht_k3")
                res["completed"] = True
            res["stage"] = stage
            res["metadata"] = {**meta, "dtype_arg": dtype, "seed": int(seed)}
            res["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            res["total_stage_seconds"] = round(time.perf_counter() - t0, 2)
            _atomic_write_json(stage_ckpt, res)
        finally:
            _cleanup_controller(ctrl)

    elif stage == "varloop_block_b1":
        from .biophysical_inkling import _load_biophysical_controller, run_phase7_varloop_suite

        eff_bsz = int(batch_size) if batch_size is not None else 32
        print(f"[Phase 7 | {stage}] Loading {label} from {model_path}...", flush=True)
        ctrl, meta = _load_biophysical_controller(model_path, label)
        try:
            res = run_phase7_varloop_suite(
                ctrl,
                batch_size=eff_bsz,
                n_boot=n_boot,
                checkpoint_path=stage_ckpt,
            )
            res["stage"] = stage
            res["metadata"] = {**meta, "dtype_arg": dtype, "seed": int(seed)}
            res["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            res["total_stage_seconds"] = round(time.perf_counter() - t0, 2)
            _atomic_write_json(stage_ckpt, res)
        finally:
            _cleanup_controller(ctrl)

    elif stage == "huginn_block_b2":
        from .biophysical_huginn import load_huginn_controller, run_phase7_huginn_suite

        eff_bsz = int(batch_size) if batch_size is not None else 16
        print(f"[Phase 7 | {stage}] Loading {label} from {model_path}...", flush=True)
        ctrl, meta = load_huginn_controller(model_path, label)
        try:
            res = run_phase7_huginn_suite(
                ctrl,
                batch_size=eff_bsz,
                n_boot=n_boot,
                checkpoint_path=stage_ckpt,
            )
            res["stage"] = stage
            res["metadata"] = {**meta, "dtype_arg": dtype, "seed": int(seed)}
            res["timestamp_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            res["total_stage_seconds"] = round(time.perf_counter() - t0, 2)
            _atomic_write_json(stage_ckpt, res)
        finally:
            _cleanup_controller(ctrl)
    else:
        raise ValueError(f"Unsupported Phase 7 stage: {stage}")

    if output_path is not None and output_path.resolve() != stage_ckpt.resolve():
        _atomic_write_json(output_path, res)
    stage_sha = _sha256_file(stage_ckpt)
    print(
        f"[OK] Phase 7 stage '{stage}' complete in {res['total_stage_seconds']}s -> "
        f"{stage_ckpt} (sha256={stage_sha[:12]}...)",
        flush=True,
    )
    return res


def _merge_phase7_stages(
    checkpoint_dir: Path,
    output_path: Path,
    allow_partial: bool = False,
) -> tuple[dict[str, Any], str]:
    """Combine all 4 Phase 7 stage JSONs into a single unified JSON with SHA-256 metadata."""
    stage_payloads: dict[str, dict[str, Any]] = {}
    stage_sha256: dict[str, str] = {}
    missing: list[str] = []
    incomplete: list[str] = []

    for stage in _PHASE7_STAGES:
        p = _stage_json_path(checkpoint_dir, stage)
        if not p.is_file():
            missing.append(f"{stage} ({p})")
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("completed") is not True:
            incomplete.append(f"{stage} ({p})")
            if not allow_partial:
                continue
        stage_payloads[stage] = data
        stage_sha256[stage] = _sha256_file(p)

    if (missing or incomplete) and not allow_partial:
        parts = []
        if missing:
            parts.append("missing: " + ", ".join(missing))
        if incomplete:
            parts.append("incomplete: " + ", ".join(incomplete))
        raise RuntimeError("Cannot merge Phase 7 stages (" + "; ".join(parts) + ")")

    all_complete = len(stage_payloads) == len(_PHASE7_STAGES) and not missing and not incomplete
    merged: dict[str, Any] = {
        "phase": "phase7_biophysics",
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "completed": all_complete,
        "stages_included": list(stage_payloads.keys()),
        "stage_sha256": stage_sha256,
        **stage_payloads,
    }
    _atomic_write_json(output_path, merged)
    merged_sha = _sha256_file(output_path)
    return merged, merged_sha


def _call_with_supported_kwargs(fn: Any, candidate_kwargs: dict[str, Any]) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(**candidate_kwargs)
    filtered = {k: v for k, v in candidate_kwargs.items() if k in params}
    return fn(**filtered)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nullbridge", description="Nullbridge research CLI.")
    parser.add_argument("--version", action="version", version="nullbridge 0.1.0")
    sub = parser.add_subparsers(dest="command")

    guard_p = sub.add_parser("guard", help="Verify tracked files contain no keys or local machine paths.")
    guard_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root to scan.")

    audit_p = sub.add_parser("audit", help="Run Phase 0 mathematical, synthetic-loop, and disjointness audit.")
    audit_p.add_argument(
        "--out",
        type=Path,
        default=Path("results/phase0_audit/benchmark_and_algebra_audit.json"),
        help="Output JSON path for locked Phase 0 audit.",
    )
    audit_p.add_argument("--audit-only", action="store_true", help="Validate without writing JSON.")

    tb_p = sub.add_parser("track-b", help="Run or render Track B (Gemma-4-Loop-Compact LoRA vs. Nullbridge).")
    tb_p.add_argument("--package-dir", type=Path, default=None, help="Path to gemma4_loop_compact_pkg directory.")
    tb_p.add_argument("--base-model", type=str, default="google/gemma-4-31B-it", help="Base model path or HF repo.")
    tb_p.add_argument("--model-label", type=str, default="google/gemma-4-31B-it")
    tb_p.add_argument("--out-dir", type=Path, default=Path("runs/track_b_gemma4"))
    tb_p.add_argument("--lora-steps", type=int, default=180)
    tb_p.add_argument("--batch-size", type=int, default=16)
    tb_p.add_argument("--no-plots", action="store_true")
    tb_p.add_argument("--resume-json", type=Path, default=None, help="Resume from checkpoint_track_b_results.json.")
    tb_p.add_argument("--render-dir", type=Path, default=None, help="Render Dawn State SVG/HTML/MD from a completed run dir.")

    ta_p = sub.add_parser("track-a", help="Run Track A native looped MoEs (TML-d24-varloop-research & Inkling-2-Mini).")
    ta_p.add_argument("--varloop-dir", type=Path, required=True, help="Local path or repo for TML-d24-varloop-research.")
    ta_p.add_argument("--inkling-mini-dir", type=Path, default=None, help="Optional path for flagship Inkling-2-Mini.")
    ta_p.add_argument("--varloop-label", type=str, default="thinkingmachines/TML-d24-varloop-research")
    ta_p.add_argument("--inkling-label", type=str, default="thinkingmachines/Inkling-2-Mini")
    ta_p.add_argument("--out-dir", type=Path, default=Path("runs/track_a_inkling"))
    ta_p.add_argument("--lora-steps", type=int, default=112)
    ta_p.add_argument("--batch-size", type=int, default=32)
    ta_p.add_argument("--passes", type=str, default="1,2,3,4,5,7")
    ta_p.add_argument("--resume-json", type=Path, default=None)

    ai_p = sub.add_parser("afriat-inkling", help="Run Part 4 Afriat / GARP & Money Is Moral suite on Varloop & Inkling-2-Mini.")
    ai_p.add_argument("--varloop-dir", type=Path, required=True, help="Local path for TML-d24-varloop-research.")
    ai_p.add_argument("--inkling-mini-dir", type=Path, default=None, help="Optional path for flagship Inkling-2-Mini.")
    ai_p.add_argument("--varloop-label", type=str, default="thinkingmachines/TML-d24-varloop-research")
    ai_p.add_argument("--inkling-label", type=str, default="thinkingmachines/Inkling-2-Mini")
    ai_p.add_argument("--out-dir", type=Path, default=Path("runs/afriat_inkling_garp"))
    ai_p.add_argument("--batch-size", type=int, default=32)
    ai_p.add_argument("--passes", type=str, default="1,2,3,4,5")
    ai_p.add_argument("--resume-json", type=Path, default=None)

    an_p = sub.add_parser("analyze", help="Compute deterministic Phase 3 & Phase 4 statistical analysis and synthesis.")
    an_p.add_argument(
        "--track-b-json",
        type=Path,
        default=Path("results/phase3_gemma4_nullbridge/track_b_results.json"),
    )
    an_p.add_argument(
        "--track-a-json",
        type=Path,
        default=Path("results/phase4_native_inkling_nullbridge/track_a_results.json"),
    )
    an_p.add_argument(
        "--phase0-json",
        type=Path,
        default=Path("results/phase0_audit/benchmark_and_algebra_audit.json"),
    )
    an_p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/phase3_gemma4_nullbridge/statistical_analysis.json"),
    )
    an_p.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/phase3_gemma4_nullbridge/synthesis.md"),
    )
    an_p.add_argument(
        "--out-track-a-json",
        type=Path,
        default=Path("results/phase4_native_inkling_nullbridge/track_a_analysis.json"),
    )
    an_p.add_argument(
        "--out-track-a-md",
        type=Path,
        default=Path("results/phase4_native_inkling_nullbridge/synthesis.md"),
    )

    aa_p = sub.add_parser("analyze-afriat", help="Compute deterministic Phase 5 Afriat / GARP Inkling statistical analysis.")
    aa_p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/phase5_inkling_afriat_garp/afriat_inkling_results.json"),
    )
    aa_p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/phase5_inkling_afriat_garp/afriat_inkling_analysis.json"),
    )
    aa_p.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/phase5_inkling_afriat_garp/synthesis.md"),
    )
    aa_p.add_argument("--gpu-spend-usd", type=float, default=0.0)
    aa_p.add_argument("--budget-cap-usd", type=float, default=5.0)

    fa_p = sub.add_parser("figures-afriat", help="Render Dawn State SVG figures and HTML preview for Phase 5 Afriat / GARP study.")
    fa_p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/phase5_inkling_afriat_garp/afriat_inkling_results.json"),
    )
    fa_p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/afriat_figures_preview"),
    )

    bp_p = sub.add_parser("biophysical-inkling", help="Run Phase 6 5-experiment biophysical loop study on Inkling-2-Mini.")
    bp_p.add_argument("--inkling-mini-dir", type=Path, required=True, help="Local path for Inkling-2-Mini.")
    bp_p.add_argument("--inkling-label", type=str, default="thinkingmachines/Inkling-2-Mini")
    bp_p.add_argument("--out-dir", type=Path, default=Path("runs/phase6_inkling_biophysical"))
    bp_p.add_argument("--batch-size", type=int, default=32)
    bp_p.add_argument("--resume-json", type=Path, default=None)

    ab_p = sub.add_parser("analyze-biophysical", help="Build locked Phase 6 statistical summary JSON and synthesis.md.")
    ab_p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/phase6_inkling_biophysical_loop/phase6_biophysical_results.json"),
    )
    ab_p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/phase6_inkling_biophysical_loop/biophysical_analysis.json"),
    )
    ab_p.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/phase6_inkling_biophysical_loop/synthesis.md"),
    )
    ab_p.add_argument("--gpu-spend-usd", type=float, default=0.0)
    ab_p.add_argument("--pod-wall-minutes", type=float, default=0.0)

    fb_p = sub.add_parser("figures-biophysical", help="Render Dawn State SVG and PNG figures for Phase 6 biophysical study.")
    fb_p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/phase6_inkling_biophysical_loop/phase6_biophysical_results.json"),
    )
    fb_p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/phase6_inkling_biophysical_loop"),
    )

    p7_p = sub.add_parser(
        "biophysical-phase7",
        help="Run Phase 7 Biophysics of a Looped Transformer resumable stages or merge outputs.",
    )
    p7_p.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=("all", "inkling_block_a", "varloop_block_a", "varloop_block_b1", "huginn_block_b2", "merge"),
        help="Execution stage to run (or 'merge' to combine all 4 stage JSONs).",
    )
    p7_p.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local verified model snapshot directory for single-stage execution.",
    )
    p7_p.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Optional HuggingFace repository label override for metadata.",
    )
    p7_p.add_argument(
        "--inkling-dir",
        type=Path,
        default=None,
        help="Local Inkling-2-Mini directory (used by --stage all).",
    )
    p7_p.add_argument(
        "--varloop-dir",
        type=Path,
        default=None,
        help="Local TML-d24-varloop-research directory (used by --stage all).",
    )
    p7_p.add_argument(
        "--huginn-dir",
        type=Path,
        default=None,
        help="Local Huginn-0125 directory (used by --stage all).",
    )
    p7_p.add_argument(
        "--output",
        type=Path,
        default=Path("results/phase7_biophysics/phase7_biophysics_results.json"),
        help="Output JSON path (unified merged JSON for 'merge'/'all', or extra copy for single stage).",
    )
    p7_p.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("runs/phase7_biophysics/checkpoints"),
        help="Directory storing per-stage atomic resumable JSON checkpoints.",
    )
    p7_p.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=("auto", "bf16", "bfloat16", "fp32", "float32", "fp16", "float16"),
        help="Preferred compute dtype metadata ('auto' selects bf16 on CUDA, fp32 on CPU).",
    )
    p7_p.add_argument("--seed", type=int, default=20260701, help="Deterministic RNG seed.")
    p7_p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size override (defaults to 32 for Inkling/Varloop and 16 for Huginn).",
    )
    p7_p.add_argument(
        "--n-passes",
        type=int,
        default=None,
        help="Optional recurrent pass count override for Block A (defaults: 3 Inkling, 5 Varloop).",
    )
    p7_p.add_argument("--n-boot", type=int, default=4000, help="Bootstrap resamples for 95%% CIs.")
    p7_p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Allow --stage merge to emit a partial JSON with completed=false when stages are missing.",
    )

    ap7_p = sub.add_parser(
        "analyze-phase7",
        help="Build locked Phase 7 statistical analysis JSON and synthesis.md.",
    )
    ap7_p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/phase7_biophysics/phase7_biophysics_results.json"),
    )
    ap7_p.add_argument(
        "--phase6-json",
        type=Path,
        default=Path("results/phase6_inkling_biophysical_loop/phase6_biophysical_results.json"),
    )
    ap7_p.add_argument(
        "--out-json",
        type=Path,
        default=Path("results/phase7_biophysics/phase7_biophysics_analysis.json"),
    )
    ap7_p.add_argument(
        "--out-md",
        type=Path,
        default=Path("results/phase7_biophysics/synthesis.md"),
    )
    ap7_p.add_argument("--gpu-spend-usd", type=float, default=0.0)
    ap7_p.add_argument("--pod-wall-minutes", type=float, default=0.0)

    fp7_p = sub.add_parser(
        "figures-phase7",
        help="Render Dawn State SVG and 2x PNG figures for Phase 7 biophysics study.",
    )
    fp7_p.add_argument(
        "--results-json",
        type=Path,
        default=Path("results/phase7_biophysics/phase7_biophysics_results.json"),
    )
    fp7_p.add_argument(
        "--phase6-json",
        type=Path,
        default=Path("results/phase6_inkling_biophysical_loop/phase6_biophysical_results.json"),
    )
    fp7_p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures/phase7_biophysics"),
    )

    args = parser.parse_args(argv)
    if args.command == "guard":
        findings = audit_repo(args.repo.resolve())
        if findings:
            for item in findings:
                print(f"[FAIL] {item}", file=sys.stderr)
            return 1
        print("[OK] Nullbridge hygiene check passed (0 secrets or local paths).")
        return 0

    if args.command == "audit":
        target = None if args.audit_only else args.out
        res = run_phase0_audit(out_path=target)
        if not res["all_valid"]:
            print(json.dumps(res, indent=2), file=sys.stderr)
            return 2
        msg = f"[OK] Phase 0 audit passed in {res['runtime_seconds']}s"
        if target is not None:
            msg += f" -> {target}"
        print(msg)
        return 0

    if args.command == "track-b":
        from .track_b import render_track_b_bundle, run_track_b_pipeline

        if args.render_dir is not None:
            render_track_b_bundle(args.render_dir)
            return 0
        if args.package_dir is None:
            print("[FAIL] --package-dir is required when running Track B on GPU.", file=sys.stderr)
            return 2
        run_track_b_pipeline(
            package_dir=args.package_dir,
            base_model=args.base_model,
            model_label=args.model_label,
            out_dir=args.out_dir,
            lora_steps=args.lora_steps,
            batch_size=args.batch_size,
            no_plots=args.no_plots,
            resume_json=args.resume_json,
        )
        return 0

    if args.command == "track-a":
        from .track_a import run_track_a_pipeline

        pass_tuple = tuple(int(x.strip()) for x in args.passes.split(",") if x.strip())
        run_track_a_pipeline(
            varloop_model_dir=args.varloop_dir,
            inkling_mini_dir=args.inkling_mini_dir,
            varloop_label=args.varloop_label,
            inkling_label=args.inkling_label,
            out_dir=args.out_dir,
            lora_steps=args.lora_steps,
            batch_size=args.batch_size,
            pass_depths=pass_tuple,
            resume_json=args.resume_json,
        )
        return 0

    if args.command == "afriat-inkling":
        from .afriat_inkling import run_afriat_inkling_pipeline

        pass_tuple = tuple(int(x.strip()) for x in args.passes.split(",") if x.strip())
        run_afriat_inkling_pipeline(
            varloop_model_dir=args.varloop_dir,
            inkling_mini_dir=args.inkling_mini_dir,
            varloop_label=args.varloop_label,
            inkling_label=args.inkling_label,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            pass_depths=pass_tuple,
            resume_json=args.resume_json,
        )
        return 0

    if args.command == "analyze":
        from .analysis import run_phase3_analysis, run_phase4_analysis

        analysis = run_phase3_analysis(
            track_b_path=args.track_b_json,
            phase0_path=args.phase0_json,
            out_json_path=args.out_json,
            out_md_path=args.out_md,
        )
        print(
            f"[OK] Phase 3 analysis locked -> {args.out_json} & {args.out_md} "
            f"(upstream track_b sha256={analysis['upstream_sha256']['track_b_results_json'][:12]}...)"
        )
        if args.track_a_json.is_file():
            analysis_a = run_phase4_analysis(
                track_a_path=args.track_a_json,
                track_b_path=args.track_b_json,
                phase0_path=args.phase0_json,
                out_json_path=args.out_track_a_json,
                out_md_path=args.out_track_a_md,
            )
            print(
                f"[OK] Phase 4 analysis locked -> {args.out_track_a_json} & {args.out_track_a_md} "
                f"(upstream track_a sha256={analysis_a['upstream_sha256']['track_a_results_json'][:12]}...)"
            )
        return 0

    if args.command == "analyze-afriat":
        from .afriat_analysis import run_phase5_afriat_analysis

        res_a5 = run_phase5_afriat_analysis(
            results_path=args.results_json,
            out_json_path=args.out_json,
            out_md_path=args.out_md,
            gpu_spend_usd=args.gpu_spend_usd,
            budget_cap_usd=args.budget_cap_usd,
        )
        print(
            f"[OK] Phase 5 Afriat-Inkling analysis locked -> {args.out_json} & {args.out_md} "
            f"(upstream sha256={res_a5['upstream_sha256']['afriat_inkling_results_json'][:12]}...)"
        )
        return 0

    if args.command == "figures-afriat":
        from .afriat_figures import render_afriat_figure_suite

        meta_f = render_afriat_figure_suite(
            results_json_path=args.results_json,
            out_dir=args.out_dir,
        )
        print(
            f"[OK] Rendered {meta_f['n_figures']} Dawn State SVG figures -> {meta_f['index_html']}"
        )
        return 0

    if args.command == "biophysical-inkling":
        from .biophysical_inkling import run_phase6_biophysical_pipeline

        run_phase6_biophysical_pipeline(
            inkling_mini_dir=args.inkling_mini_dir,
            inkling_label=args.inkling_label,
            out_dir=args.out_dir,
            batch_size=args.batch_size,
            resume_json=args.resume_json,
        )
        return 0

    if args.command == "analyze-biophysical":
        from .biophysical_analysis import run_phase6_biophysical_analysis

        res_b6 = run_phase6_biophysical_analysis(
            results_path=args.results_json,
            out_json_path=args.out_json,
            out_md_path=args.out_md,
            gpu_spend_usd=args.gpu_spend_usd,
            pod_wall_minutes=args.pod_wall_minutes,
        )
        print(
            f"[OK] Phase 6 Biophysical analysis locked -> {args.out_json} & {args.out_md} "
            f"(upstream sha256={res_b6['upstream_sha256']['phase6_biophysical_results_json'][:12]}...)"
        )
        return 0

    if args.command == "figures-biophysical":
        from .biophysical_figures import render_phase6_figure_suite

        meta_b6 = render_phase6_figure_suite(
            results_json_path=args.results_json,
            out_dir=args.out_dir,
        )
        print(
            f"[OK] Rendered {meta_b6['n_svg']} SVGs and {meta_b6['n_png']} PNGs -> {meta_b6['index_html']}"
        )
        return 0

    if args.command == "biophysical-phase7":
        if args.stage == "merge":
            merged, merged_sha = _merge_phase7_stages(
                checkpoint_dir=args.checkpoint_dir,
                output_path=args.output,
                allow_partial=args.allow_partial,
            )
            print(
                f"[OK] Phase 7 merged ({len(merged['stages_included'])}/4 stages, "
                f"completed={merged['completed']}) -> {args.output} (sha256={merged_sha[:12]}...)"
            )
            return 0

        if args.stage == "all":
            stage_paths = {
                "inkling_block_a": args.inkling_dir or args.model_path,
                "varloop_block_a": args.varloop_dir or args.model_path,
                "varloop_block_b1": args.varloop_dir or args.model_path,
                "huginn_block_b2": args.huginn_dir or args.model_path,
            }
            for st_name in _PHASE7_STAGES:
                ckpt_p = _stage_json_path(args.checkpoint_dir, st_name)
                if _stage_already_done(ckpt_p):
                    print(f"[Phase 7 | {st_name}] Already completed at {ckpt_p}; skipping.", flush=True)
                    continue
                m_path = stage_paths[st_name]
                if m_path is None:
                    print(
                        f"[FAIL] Missing model directory for stage '{st_name}'. "
                        "Provide --inkling-dir, --varloop-dir, and --huginn-dir (or run per-stage with --model-path).",
                        file=sys.stderr,
                    )
                    return 2
                _run_phase7_single_stage(
                    stage=st_name,
                    model_path=m_path,
                    checkpoint_dir=args.checkpoint_dir,
                    model_label=args.model_label,
                    output_path=None,
                    batch_size=args.batch_size,
                    n_passes=args.n_passes,
                    dtype=args.dtype,
                    seed=args.seed,
                    n_boot=args.n_boot,
                )
            merged, merged_sha = _merge_phase7_stages(
                checkpoint_dir=args.checkpoint_dir,
                output_path=args.output,
                allow_partial=args.allow_partial,
            )
            print(
                f"[OK] Phase 7 full pipeline complete ({len(merged['stages_included'])}/4 stages, "
                f"completed={merged['completed']}) -> {args.output} (sha256={merged_sha[:12]}...)"
            )
            return 0

        if args.model_path is None:
            print(f"[FAIL] --model-path is required for --stage {args.stage}.", file=sys.stderr)
            return 2

        default_merged = Path("results/phase7_biophysics/phase7_biophysics_results.json")
        extra_out = None if args.output == default_merged else args.output
        _run_phase7_single_stage(
            stage=args.stage,
            model_path=args.model_path,
            checkpoint_dir=args.checkpoint_dir,
            model_label=args.model_label,
            output_path=extra_out,
            batch_size=args.batch_size,
            n_passes=args.n_passes,
            dtype=args.dtype,
            seed=args.seed,
            n_boot=args.n_boot,
        )
        return 0

    if args.command == "analyze-phase7":
        from .biophysical_phase7_analysis import run_phase7_biophysical_analysis

        res_p7 = _call_with_supported_kwargs(
            run_phase7_biophysical_analysis,
            {
                "results_path": args.results_json,
                "phase6_results_path": args.phase6_json,
                "out_json_path": args.out_json,
                "out_md_path": args.out_md,
                "gpu_spend_usd": args.gpu_spend_usd,
                "pod_wall_minutes": args.pod_wall_minutes,
            },
        )
        sha_map = res_p7.get("upstream_sha256", {}) if isinstance(res_p7, dict) else {}
        first_sha = next(iter(sha_map.values()), "") if isinstance(sha_map, dict) else ""
        sha_note = f" (upstream sha256={str(first_sha)[:12]}...)" if first_sha else ""
        print(f"[OK] Phase 7 Biophysical analysis locked -> {args.out_json} & {args.out_md}{sha_note}")
        return 0

    if args.command == "figures-phase7":
        from .biophysical_phase7_figures import render_phase7_figure_suite

        meta_p7 = _call_with_supported_kwargs(
            render_phase7_figure_suite,
            {
                "results_json_path": args.results_json,
                "phase6_results_json_path": args.phase6_json,
                "out_dir": args.out_dir,
            },
        )
        if isinstance(meta_p7, dict) and "index_html" in meta_p7:
            print(
                f"[OK] Rendered {meta_p7.get('n_svg', '?')} SVGs and {meta_p7.get('n_png', '?')} PNGs -> "
                f"{meta_p7['index_html']}"
            )
        else:
            print(f"[OK] Rendered Phase 7 figure suite -> {args.out_dir}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
