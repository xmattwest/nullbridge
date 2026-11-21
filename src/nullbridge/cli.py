"""CLI bootstrap and pre-push repository hygiene guard for Nullbridge."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

BLOCKED_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private_key_block", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("github_token", re.compile(r"\bgh[pousr]_[A-Za-z0-9_]{20,}\b")),
    ("api_secret_key", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("huggingface_token", re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")),
    ("runpod_key", re.compile(r"\brpa_[A-Za-z0-9_]{15,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("local_home_path", re.compile(r"/(?:Users|home)/[A-Za-z0-9._-]+/")),
)


def audit_repo(repo_root: Path) -> list[str]:
    """Scan tracked files for accidental keys, tokens, or local machine paths."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="nullbridge", description="Nullbridge research CLI.")
    parser.add_argument("--version", action="version", version="nullbridge 0.1.0")
    sub = parser.add_subparsers(dest="command")

    guard_p = sub.add_parser("guard", help="Verify tracked files contain no keys or local machine paths.")
    guard_p.add_argument("--repo", type=Path, default=Path.cwd(), help="Repository root to scan.")

    args = parser.parse_args(argv)
    if args.command == "guard":
        findings = audit_repo(args.repo.resolve())
        if findings:
            for item in findings:
                print(f"[FAIL] {item}", file=sys.stderr)
            return 1
        print("[OK] Nullbridge hygiene check passed (0 secrets or local paths).")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
