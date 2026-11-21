"""CLI entrypoint, repository hygiene guard, and Phase 0 audit runner for Nullbridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
