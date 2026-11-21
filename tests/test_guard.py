from pathlib import Path

from nullbridge.cli import audit_repo


def test_repo_hygiene_guard() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert audit_repo(repo_root) == []
