#!/bin/bash
# fetch_and_run_phase7_biophysics.sh (POD-SIDE):
#   1. Installs pinned `transformers@046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3`
#      (`Inkling2LoopForConditionalGeneration`) in the primary environment.
#   2. Runs full mock-PyTorch unit tests (`test_phase6_biophysical_cpu.py`,
#      `test_phase7_safety_cpu.py`, `test_phase7_cross_arch_cpu.py`) BEFORE
#      downloading any model weights.
#   3. Streams and SHA-256 verifies ONE model at a time from
#      `config/manifests/{inkling-2-mini,tml-d24-varloop-research,huginn-0125}.tsv`
#      using 24 persistent-TLS CDN range workers, executes strictly offline
#      (`HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`), saves atomic resumable JSON
#      checkpoints, and deletes `/models/<name>` immediately after each model
#      completes so local disk usage stays below 60 GB.
#   4. Tests `RavenForCausalLM` compatibility in the primary env for `Huginn-0125`;
#      if `transformers` API drift breaks `raven_modeling_minimal.py`, creates and
#      uses a sibling virtualenv `/opt/huginn-venv` (`transformers==4.48.3` or
#      `4.44.2`) rather than downgrading the Inkling environment.
#   5. Runs `biophysical-phase7 --stage merge` to emit the final unified
#      `phase7_biophysics_results.json` with `completed: true`.

set +x +v +a
set -euo pipefail
export LC_ALL=C

INKLING_NAME="${INKLING_NAME:-inkling-2-mini}"
INKLING_REPO="${INKLING_REPO:-thinkingmachines/Inkling-2-Mini}"
INKLING_REV="${INKLING_REV:-c9d028302cd0f035fdf8e7e91f089a4ce94d2195}"

VARLOOP_NAME="${VARLOOP_NAME:-tml-d24-varloop-research}"
VARLOOP_REPO="${VARLOOP_REPO:-thinkingmachines/TML-d24-varloop-research}"
VARLOOP_REV="${VARLOOP_REV:-91a57fe2b779bbd12c0f74d1f518cbd5b84e8d0d}"

HUGINN_NAME="${HUGINN_NAME:-huginn-0125}"
HUGINN_REPO="${HUGINN_REPO:-tomg-group-umd/huginn-0125}"
HUGINN_REV="${HUGINN_REV:-bb6621b65e90b6a4b9b29ef88dc83866d450470c}"

BATCH_SIZE="${1:-32}"
HUGINN_BATCH_SIZE="${2:-16}"
SEED="${3:-20260701}"
DTYPE="${4:-auto}"

LOGDIR="${LOGDIR:-/var/log/nullbridge}"
OUT="${OUT:-/workspace/runs_phase7_biophysics}"
CKPT_DIR="${CKPT_DIR:-$OUT/checkpoints}"
mkdir -p "$LOGDIR" /models "$OUT" "$CKPT_DIR"

stage_is_completed() {
  local json_path="$1"
  [ -s "$json_path" ] || return 1
  python3 - "$json_path" <<'PY'
import json
import sys
try:
    data = json.load(open(sys.argv[1], "r", encoding="utf-8"))
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if data.get("completed") is True else 1)
PY
}

fast_fetch_model() {
  local model_name="$1"
  local repo="$2"
  local rev="$3"
  local dest_root="$4"
  local man_path="/workspace/config/manifests/${model_name}.tsv"

  [ -f "$man_path" ] || { printf '[nb-p7] ERROR: manifest missing: %s\n' "$man_path" >&2; return 2; }
  rm -rf "$dest_root"
  mkdir -p "$dest_root"
  printf '[nb-p7] Fast-streaming & SHA-256 verifying %s (%s @ %s)...\n' "$model_name" "$repo" "${rev:0:12}"

  python3 - "$man_path" "$repo" "$rev" "$dest_root" "$model_name" <<'PY'
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
            raise SystemExit(f"[nb-p7] invalid manifest row: {line!r}")
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
    f"[nb-p7] WEIGHTS_OK model={model_name} files={len(entries)} gb={total_bytes/1e9:.2f} secs={dt:.1f} rate_MBps={(total_bytes/1e6)/dt:.0f}",
    flush=True,
)
PY
}

probe_huginn_compat() {
  local py_exec="$1"
  local model_dir="$2"
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "$py_exec" - "$model_dir" <<'PY'
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_dir = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
cfg = AutoConfig.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
# Build a tiny 1-block RavenConfig clone to verify RavenForCausalLM forward compatibility
cfg.n_embd = 64
cfg.n_heads = 4
cfg.intermediate_size = 128
cfg.n_layers_in_prelude = 1
cfg.n_layers_in_recurrent_block = 1
cfg.n_layers_in_coda = 1
cfg.vocab_size = 256
cfg.padded_vocab_size = 256
cfg.block_size = 32
m = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True).eval()
with torch.no_grad():
    out = m(input_ids=torch.ones((1, 4), dtype=torch.long), num_steps=torch.tensor(2, dtype=torch.long), use_cache=False)
assert out.logits.shape == (1, 4, 256), f"unexpected logits shape: {out.logits.shape}"
print("[nb-p7] Huginn RavenForCausalLM compatibility OK under", sys.executable)
PY
}

{
  export PATH="/root/.local/bin:/root/.cargo/bin:/usr/local/bin:$PATH"
  printf '[nb-p7] Installing pinned transformers@046bdb8cc2ce (Inkling2Loop support, HTTPS tarball)...\n'
  if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="/root/.local/bin:/root/.cargo/bin:$PATH"
  fi
  PY_BIN="$(command -v python3)"
  uv pip install --python "$PY_BIN" --system --break-system-packages --no-cache \
    "https://github.com/huggingface/transformers/archive/046bdb8cc2cee29d233964bb77fe6fd37cfe4aa3.tar.gz" \
    tiktoken pillow accelerate fla-core kernels pytest
  uv pip install --python "$PY_BIN" --system --break-system-packages --no-cache --only-binary=:all: causal-conv1d 2>/dev/null || true

  if ! python3 -c "import torch; from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration" 2>/dev/null; then
    uv pip uninstall --python "$PY_BIN" --system --break-system-packages torchvision || true
  fi
  python3 -c "import torch; from transformers import AutoTokenizer, Inkling2LoopForConditionalGeneration; print('[nb-p7] Verified torch', torch.__version__, '+ Inkling2LoopForConditionalGeneration')"

  # Pre-warm Hub/Triton CUDA kernels before offline execution
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
    print("[nb-p7] Pre-warmed Hub/Triton CUDA kernels")
PY

  # Mandatory pre-flight mock-PyTorch unit tests BEFORE downloading any weights
  printf '[nb-p7] Running pre-flight PyTorch unit tests before any weight downloads...\n'
  (
    cd /workspace
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    PYTHONPATH=/workspace/src python3 -m pytest -q \
      tests/test_phase6_biophysical_cpu.py \
      tests/test_phase7_safety_cpu.py \
      tests/test_phase7_cross_arch_cpu.py
  )
  printf '[nb-p7] Pre-flight unit tests PASSED.\n'

  # -------------------------------------------------------------------------
  # Model 1: Inkling-2-Mini (Block A: Safety-Biophysics Unification)
  # -------------------------------------------------------------------------
  if stage_is_completed "$CKPT_DIR/inkling_block_a_results.json"; then
    printf '[nb-p7] Skipping inkling_block_a (already completed in %s).\n' "$CKPT_DIR"
  else
    INK_DST="/models/${INKLING_NAME}"
    fast_fetch_model "$INKLING_NAME" "$INKLING_REPO" "$INKLING_REV" "$INK_DST"
    printf '[nb-p7] Starting offline Phase 7 stage: inkling_block_a...\n'
    (
      cd /workspace
      env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          PYTHONPATH=/workspace/src \
          python3 -m nullbridge.cli biophysical-phase7 \
            --stage inkling_block_a \
            --model-path "$INK_DST" \
            --model-label "$INKLING_REPO" \
            --checkpoint-dir "$CKPT_DIR" \
            --batch-size "$BATCH_SIZE" \
            --dtype "$DTYPE" \
            --seed "$SEED"
    )
    rm -rf "$INK_DST"
    printf '[nb-p7] Deleted %s (disk reclaimed).\n' "$INK_DST"
  fi

  # -------------------------------------------------------------------------
  # Model 2: TML-d24-varloop-research (Block A + Block B1)
  # -------------------------------------------------------------------------
  if stage_is_completed "$CKPT_DIR/varloop_block_a_results.json" && \
     stage_is_completed "$CKPT_DIR/varloop_block_b1_results.json"; then
    printf '[nb-p7] Skipping varloop_block_a and varloop_block_b1 (already completed in %s).\n' "$CKPT_DIR"
  else
    VAR_DST="/models/${VARLOOP_NAME}"
    fast_fetch_model "$VARLOOP_NAME" "$VARLOOP_REPO" "$VARLOOP_REV" "$VAR_DST"

    if ! stage_is_completed "$CKPT_DIR/varloop_block_a_results.json"; then
      printf '[nb-p7] Starting offline Phase 7 stage: varloop_block_a...\n'
      (
        cd /workspace
        env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            PYTHONPATH=/workspace/src \
            python3 -m nullbridge.cli biophysical-phase7 \
              --stage varloop_block_a \
              --model-path "$VAR_DST" \
              --model-label "$VARLOOP_REPO" \
              --checkpoint-dir "$CKPT_DIR" \
              --batch-size "$BATCH_SIZE" \
              --dtype "$DTYPE" \
              --seed "$SEED"
      )
    fi

    if ! stage_is_completed "$CKPT_DIR/varloop_block_b1_results.json"; then
      printf '[nb-p7] Starting offline Phase 7 stage: varloop_block_b1...\n'
      (
        cd /workspace
        env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
            PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            PYTHONPATH=/workspace/src \
            python3 -m nullbridge.cli biophysical-phase7 \
              --stage varloop_block_b1 \
              --model-path "$VAR_DST" \
              --model-label "$VARLOOP_REPO" \
              --checkpoint-dir "$CKPT_DIR" \
              --batch-size "$BATCH_SIZE" \
              --dtype "$DTYPE" \
              --seed "$SEED"
      )
    fi

    rm -rf "$VAR_DST"
    printf '[nb-p7] Deleted %s (disk reclaimed).\n' "$VAR_DST"
  fi

  # -------------------------------------------------------------------------
  # Model 3: Huginn-0125 (Block B2: Non-Somatic-Reset Control)
  # -------------------------------------------------------------------------
  if stage_is_completed "$CKPT_DIR/huginn_block_b2_results.json"; then
    printf '[nb-p7] Skipping huginn_block_b2 (already completed in %s).\n' "$CKPT_DIR"
  else
    HUG_DST="/models/${HUGINN_NAME}"
    fast_fetch_model "$HUGINN_NAME" "$HUGINN_REPO" "$HUGINN_REV" "$HUG_DST"

    HUGINN_PY="$PY_BIN"
    if probe_huginn_compat "$PY_BIN" "$HUG_DST"; then
      printf '[nb-p7] Primary env passed Huginn RavenForCausalLM compatibility check.\n'
    else
      printf '[nb-p7] Primary env hit transformers API drift on RavenForCausalLM; creating /opt/huginn-venv...\n'
      rm -rf /opt/huginn-venv
      uv venv --system-site-packages --python "$PY_BIN" /opt/huginn-venv
      HUGINN_PY="/opt/huginn-venv/bin/python"
      if ! (uv pip install --python "$HUGINN_PY" --no-cache "transformers==4.48.3" accelerate && \
            probe_huginn_compat "$HUGINN_PY" "$HUG_DST"); then
        printf '[nb-p7] Retrying /opt/huginn-venv with transformers==4.44.2...\n'
        uv pip install --python "$HUGINN_PY" --no-cache "transformers==4.44.2" accelerate
        probe_huginn_compat "$HUGINN_PY" "$HUG_DST"
      fi
    fi

    printf '[nb-p7] Starting offline Phase 7 stage: huginn_block_b2 (python=%s)...\n' "$HUGINN_PY"
    (
      cd /workspace
      env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
          PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
          PYTHONPATH=/workspace/src \
          "$HUGINN_PY" -m nullbridge.cli biophysical-phase7 \
            --stage huginn_block_b2 \
            --model-path "$HUG_DST" \
            --model-label "$HUGINN_REPO" \
            --checkpoint-dir "$CKPT_DIR" \
            --batch-size "$HUGINN_BATCH_SIZE" \
            --dtype "$DTYPE" \
            --seed "$SEED"
    )
    rm -rf "$HUG_DST"
    printf '[nb-p7] Deleted %s (disk reclaimed).\n' "$HUG_DST"
  fi

  # -------------------------------------------------------------------------
  # Final Merge: Unified phase7_biophysics_results.json
  # -------------------------------------------------------------------------
  printf '[nb-p7] Merging all 4 Phase 7 stage JSONs into %s/phase7_biophysics_results.json...\n' "$OUT"
  (
    cd /workspace
    PYTHONPATH=/workspace/src python3 -m nullbridge.cli biophysical-phase7 \
      --stage merge \
      --checkpoint-dir "$CKPT_DIR" \
      --output "$OUT/phase7_biophysics_results.json"
  )

  [ -s "$OUT/phase7_biophysics_results.json" ] || { printf '[nb-p7] ERROR: phase7_biophysics_results.json missing\n' >&2; exit 4; }
  stage_is_completed "$OUT/phase7_biophysics_results.json" || { printf '[nb-p7] ERROR: merged JSON completed != true\n' >&2; exit 5; }
  sha256sum "$OUT/phase7_biophysics_results.json"
  printf '[nb-p7] PHASE7_BIOPHYSICS_DONE\n'
} 2>&1 | tee "$LOGDIR/nullbridge_phase7_biophysics.log"
