"""Turnkey NDIF + `nnsight` remote harness for Track A (`Inkling 2 Mini` & `TML-d24-varloop-research`).

Ready for the NDIF Looped Lens Challenge and cross-pass `message_input` Nullbridge interventions.
Credentials (`NDIF_API_KEY`, optional `HF_TOKEN`) are resolved strictly at runtime from environment
variables or macOS Keychain (`ndif-api-key`) without ever printing or persisting secret values.
"""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
from typing import Any

INKLING_MINI_ID = "thinkingmachines/inkling-2-mini"
VARLOOP_RESEARCH_ID = "thinkingmachines/TML-d24-varloop-research"


def resolve_secret(env_name: str, keychain_service: str, dotenv_path: Path | None = None) -> str | None:
    """Resolve an API token from process env, gitignored `.env`, or macOS Keychain without printing."""
    val = os.environ.get(env_name, "").strip()
    if val:
        return val

    env_file = dotenv_path or (Path.cwd() / ".env")
    if env_file.is_file():
        for raw in env_file.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == env_name:
                cleaned = v.strip().strip("'\"")
                if cleaned:
                    return cleaned

    try:
        proc = subprocess.run(
            ["security", "find-generic-password", "-s", keychain_service, "-w"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except (FileNotFoundError, subprocess.SubprocessError):
        pass

    return None


def check_ndif_readiness() -> dict[str, Any]:
    """Non-secret readiness report for Track A (`nnsight` package + `NDIF_API_KEY` availability)."""
    has_key = resolve_secret("NDIF_API_KEY", "ndif-api-key") is not None
    has_hf = resolve_secret("HF_TOKEN", "hf-token") is not None
    try:
        import nnsight  # type: ignore[import-not-found]
        nnsight_version = getattr(nnsight, "__version__", "installed")
    except ImportError:
        nnsight_version = None

    return {
        "models": [INKLING_MINI_ID, VARLOOP_RESEARCH_ID],
        "nnsight_version": nnsight_version,
        "ndif_api_key_configured": has_key,
        "hf_token_configured": has_hf,
        "ready_for_remote_trace": bool(has_key and nnsight_version is not None),
    }
