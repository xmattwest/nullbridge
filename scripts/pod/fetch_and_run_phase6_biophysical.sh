#!/bin/bash
# fetch_and_run_phase6_biophysical.sh (POD-SIDE):
#   1. Installs pinned `transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`
#      (`Inkling2LoopForConditionalGeneration`) and runs full PyTorch Phase 6 pre-flight tests.
#   2. Streams and SHA-256 verifies pinned `thinkingmachines/Inkling-2-Mini` into `/models/`
#      using 24 persistent-TLS CDN range workers.
#   3. Executes `nullbridge.cli biophysical-inkling` strictly offline (`HF_HUB_OFFLINE=1`).

set +x +v +a
set -euo pipefail
export LC_ALL=C

MINI_NAME="${1:-inkling-2-mini}"
MINI_REPO="${2:-thinkingmachines/Inkling-2-Mini}"
MINI_REV="${3:?missing inkling-2-mini 40-char revision}"
BATCH_SIZE="${4:-32}"

LOGDIR="/var/log/afriat"
OUT="/workspace/runs_phase6_biophysical/inkling_biophysical"
mkdir -p "$LOGDIR" /models "$OUT"

{
  export PATH="/root/.local/bin:/root/.cargo/bin:/usr/local/bin:$PATH"
  printf '[nb-p6] Installing pinned transformers@046bdb8cc2ce (Inkling2Loop support, HTTPS tarball)...\n'
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
  fi
  PY_BIN="$(command -v python3)"
  uv pip install --python "$PY_BIN" --system --break-system-packages --no-cache \
    "https://github.com/huggingface/transformers/archive/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3.tar.gz" \
    tiktoken pillow accelerate fla-core kernels
  uv pip install --python "$PY_BIN" --system --break-system-packages --no-cache --only-binary=:all: causal-conv1d 2>/dev/null || true

  if ! python3 -c "import torch; from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration" 2>/dev/null; then
    uv pip uninstall --python "$PY_BIN" --system --break-system-packages torchvision || true
  fi
  python3 -c "import torch; from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration; print('[nb-p6] Verified torch', torch.__version__, '+ Inkling2LoopForConditionalGeneration')"

  python3 - <<'PY' || true
import torch
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopConfig
from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopTextModel
if torch.cuda.is_available():
    cfg = Inkling2LoopConfig().text_config
    cfg.num_hidden_layers = 2
    cfg.loop_layer_index_start = 0
    cfg.loop_layer_index_end = 2
    cfg.hidden_size = 128
    cfg.intermediate_size = 256
    cfg.moe_intermediate_size = 128
    cfg.num_experts = 4
    cfg.num_experts_per_tok = 2
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    cfg.head_dim = 32
    cfg.vocab_size = 256
    m = Inkling2LoopTextModel(cfg).to(device="cuda:0", dtype=torch.bfloat16).eval()
    with torch.no_grad():
        _ = m(torch.ones((1, 8), dtype=torch.long, device="cuda:0"))
    del m
    torch.cuda.empty_cache()
    print("[nb-p6] Pre-warmed Hub/Triton CUDA kernels")
PY

  PYTHONPATH=/workspace/src python3 /workspace/tests/test_phase6_biophysical_cpu.py

  MAN="/workspace/config/manifests/${MINI_NAME}.tsv"
  DST="/models/${MINI_NAME}"
  [ -f "$MAN" ] || { printf '[nb-p6] ERROR: manifest missing: %s\n' "$MAN" >&2; exit 2; }
  rm -rf "$DST"
  mkdir -p "$DST"
  printf '[nb-p6] Fast-streaming & SHA-256 verifying %s (%s @ %s)...\n' "$MINI_NAME" "$MINI_REPO" "${MINI_REV:0:12}"

  python3 - "$MAN" "$MINI_REPO" "$MINI_REV" "$DST" "$MINI_NAME" <<'PY'
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import os
import re
import sys
import threading
import time
import urllib.request

try:
    import urllib3
    _TLS = threading.local()

    def get_thread_http():
        pool = getattr(_TLS, "pool", None)
        if pool is None:
            pool = urllib3.PoolManager(
                num_pools=8,
                maxsize=8,
                retries=urllib3.Retry(total=2, redirect=5, connect=2, read=2),
                timeout=urllib3.Timeout(connect=15.0, read=90.0),
            )
            _TLS.pool = pool
        return pool
except Exception:
    get_thread_http = None

man_path, repo, rev, dest_root, model_name = sys.argv[1:6]
RE_PATH = re.compile(r"^[A-Za-z0-9._-]+(/[A-Za-z0-9._-]+)*$")
RE_INT = re.compile(r"^[0-9]+$")
RE_SHA = re.compile(r"^[0-9a-f]{64}$")

entries = []
with open(man_path, "r", encoding="utf-8") as fh:
    for raw_line in fh:
        line = raw_line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        rel_path, size_s, sha, _src = line.split("\t")
        if not RE_PATH.match(rel_path) or not RE_INT.match(size_s) or not RE_SHA.match(sha):
            raise SystemExit(f"[nb-p6] invalid manifest row: {line!r}")
        entries.append((rel_path, int(size_s), sha))

t0 = time.time()
SMALL_CUTOFF = 32 * 1024 * 1024
CHUNK_SZ = 32 * 1024 * 1024
UA = {"User-Agent": "nullbridge-fastfetch/1.0"}


def fetch_bytes_range(url: str, start: int, end: int) -> bytes:
    headers = {"Range": f"bytes={start}-{end}", "User-Agent": "nullbridge-fastfetch/1.0"}
    expected = end - start + 1
    for attempt in range(4):
        try:
            http = get_thread_http() if get_thread_http is not None else None
            if http is not None:
                resp = http.request("GET", url, headers=headers, preload_content=False)
                try:
                    if resp.status not in (200, 206):
                        raise RuntimeError(f"HTTP {resp.status}")
                    chunks = []
                    while True:
                        blk = resp.read(1024 * 1024)
                        if not blk:
                            break
                        chunks.append(blk)
                    data = b"".join(chunks)
                finally:
                    resp.release_conn()
            else:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as r:
                    data = r.read()
            if len(data) != expected:
                raise RuntimeError(f"short read {len(data)} != {expected}")
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("unreachable")


def verify_and_commit_file(part_path: str, dest_path: str, exp_size: int, exp_sha: str) -> None:
    st = os.stat(part_path)
    if st.st_size != exp_size:
        raise RuntimeError(f"size mismatch on {dest_path}: {st.st_size} != {exp_size}")
    with open(part_path, "rb") as f:
        got = hashlib.file_digest(f, "sha256").hexdigest()
    if got != exp_sha:
        raise RuntimeError(f"SHA-256 mismatch on {dest_path}: {got} != {exp_sha}")
    os.replace(part_path, dest_path)


small_entries = [e for e in entries if e[1] <= SMALL_CUTOFF]
large_entries = [e for e in entries if e[1] > SMALL_CUTOFF]

for rel_path, sz, sha in small_entries:
    url = f"https://huggingface.co/{repo}/resolve/{rev}/{rel_path}"
    dest = os.path.join(dest_root, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r, open(part, "wb") as out_f:
        data = r.read()
        out_f.write(data)
    verify_and_commit_file(part, dest, sz, sha)

if large_entries:
    hash_pool = ThreadPoolExecutor(max_workers=4)
    hash_futs = []
    shard_states = []
    slice_tasks = []
    for s_idx, (rel_path, sz, sha) in enumerate(large_entries):
        dest = os.path.join(dest_root, rel_path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        fd = os.open(part, os.O_CREAT | os.O_TRUNC | os.O_RDWR, 0o644)
        os.ftruncate(fd, sz)
        n_slices = (sz + CHUNK_SZ - 1) // CHUNK_SZ
        st = {
            "fd": fd,
            "part": part,
            "dest": dest,
            "size": sz,
            "sha": sha,
            "rem_slices": n_slices,
            "lock": threading.Lock(),
            "url": f"https://huggingface.co/{repo}/resolve/{rev}/{rel_path}",
        }
        shard_states.append(st)
        for c_i in range(n_slices):
            start = c_i * CHUNK_SZ
            end = min(sz, start + CHUNK_SZ) - 1
            slice_tasks.append((s_idx, start, end))

    def run_slice(task):
        s_idx, pos, end = task
        st = shard_states[s_idx]
        buf = fetch_bytes_range(st["url"], pos, end)
        mv = memoryview(buf)
        written = 0
        while written < len(mv):
            written += os.pwrite(st["fd"], mv[written:], pos + written)
        with st["lock"]:
            st["rem_slices"] -= 1
            done_shard = st["rem_slices"] == 0
        if done_shard:
            os.close(st["fd"])
            st["fd"] = -1
            return hash_pool.submit(verify_and_commit_file, st["part"], st["dest"], st["size"], st["sha"])
        return None

    try:
        with ThreadPoolExecutor(max_workers=24) as net_pool:
            futs = [net_pool.submit(run_slice, t) for t in slice_tasks]
            for fut in as_completed(futs):
                h_fut = fut.result()
                if h_fut is not None:
                    hash_futs.append(h_fut)
        for hf in as_completed(hash_futs):
            hf.result()
    finally:
        hash_pool.shutdown(wait=True)
        for st in shard_states:
            if st["fd"] >= 0:
                try:
                    os.close(st["fd"])
                except OSError:
                    pass

total_bytes = sum(e[1] for e in entries)
dt = max(0.1, time.time() - t0)
print(
    f"[nb-p6] WEIGHTS_OK model={model_name} files={len(entries)} gb={total_bytes/1e9:.2f} secs={dt:.1f} rate_MBps={(total_bytes/1e6)/dt:.0f}",
    flush=True,
)
PY

  RESUME_ARG=()
  if [ -s "/workspace/resume_phase6_biophysical_results.json" ]; then
    RESUME_ARG=(--resume-json "/workspace/resume_phase6_biophysical_results.json")
  fi

  printf '[nb-p6] Starting offline Phase 6 Biophysical Inkling-2-Mini pipeline...\n'
  (
    cd /workspace
    env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH=/workspace/src \
        python3 -m nullbridge.cli biophysical-inkling \
          --inkling-mini-dir "$DST" \
          --inkling-label "$MINI_REPO" \
          --batch-size "$BATCH_SIZE" \
          "${RESUME_ARG[@]}" \
          --out-dir "$OUT"
  )

  [ -s "$OUT/phase6_biophysical_results.json" ] || { printf '[nb-p6] ERROR: phase6_biophysical_results.json missing\n' >&2; exit 4; }
  (
    cd /workspace
    PYTHONPATH=/workspace/src python3 -m nullbridge.cli analyze-biophysical \
      --results-json "$OUT/phase6_biophysical_results.json" \
      --out-json "$OUT/biophysical_analysis.json" \
      --out-md "$OUT/synthesis.md"
    PYTHONPATH=/workspace/src python3 -m nullbridge.cli figures-biophysical \
      --results-json "$OUT/phase6_biophysical_results.json" \
      --out-dir "$OUT/figures"
  ) || true

  rm -rf "$DST"
  printf '[nb-p6] PHASE6_BIOPHYSICAL_DONE\n'
} 2>&1 | tee "$LOGDIR/nullbridge_phase6_biophysical.log"
