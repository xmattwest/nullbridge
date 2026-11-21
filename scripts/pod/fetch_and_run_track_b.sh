#!/bin/bash
# fetch_and_run_track_b.sh (POD-SIDE):
#   1. Streams & SHA-256 verifies pinned HuggingFace weights (`google/gemma-4-31B-it`) into
#      /dev/shm (if >=72 GB free) or /models using 24 persistent-TLS CDN range workers + 4-thread
#      OpenSSL file_digest.
#   2. Runs Nullbridge Track B (`nullbridge.cli track-b`) strictly offline against the shipped
#      Phase 9 `gemma4_loop_compact_pkg` checkpoint.
#   3. Cleans base weights immediately while preserving `track_b_results.json`.

set +x +v +a
set -euo pipefail
export LC_ALL=C

NAME="${1:?missing model name}"
REPO="${2:?missing model repo}"
REV="${3:?missing 40-char revision}"
BATCH_SIZE="${4:-32}"
LORA_STEPS="${5:-160}"

MAN="/workspace/config/manifests/${NAME}.tsv"
PKG="/workspace/gemma4_loop_compact_pkg"
OUT="/workspace/runs_track_b/${NAME}"
LOGDIR="/var/log/afriat"
mkdir -p "$LOGDIR" /models /workspace/runs_track_b

[ -f "$MAN" ] || { printf '[nb-tb] ERROR: manifest missing: %s\n' "$MAN" >&2; exit 2; }
[ -f "$PKG/loop_compact_adapters.pt" ] || { printf '[nb-tb] ERROR: missing %s/loop_compact_adapters.pt\n' "$PKG" >&2; exit 2; }

SHM_FREE_KB="$(df -Pk /dev/shm 2>/dev/null | awk 'NR==2 {print $4}' || echo 0)"
MEM_AVAIL_KB="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null | awk '{print $1}' || echo 0)"
if [ "${SHM_FREE_KB:-0}" -ge 75497472 ] && [ "${MEM_AVAIL_KB:-0}" -ge 89128960 ]; then
  DIR="/dev/shm/nb_models/${NAME}"
  STORAGE_TAG="tmpfs:/dev/shm"
else
  DIR="/models/${NAME}"
  STORAGE_TAG="disk:/models"
fi

rm -rf "$DIR"
mkdir -p "$DIR"

printf '[nb-tb] Fast-streaming & SHA-256 verifying %s (%s @ %s, target=%s)...\n' "$NAME" "$REPO" "${REV:0:12}" "$STORAGE_TAG"

python3 - "$MAN" "$REPO" "$REV" "$DIR" "$NAME" <<'PY'
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
seen = set()
with open(man_path, "r", encoding="utf-8") as fh:
    for raw_line in fh:
        line = raw_line.rstrip("\r\n")
        if not line or line.startswith("#"):
            continue
        cols = line.split("\t")
        if len(cols) != 4:
            raise SystemExit(f"[nb-tb] malformed manifest row: {line!r}")
        rel_path, size_s, sha, _src = cols
        if not RE_PATH.match(rel_path) or "/./" in f"/{rel_path}/" or "/../" in f"/{rel_path}/":
            raise SystemExit(f"[nb-tb] bad path in manifest: {rel_path}")
        if not RE_INT.match(size_s) or not RE_SHA.match(sha):
            raise SystemExit(f"[nb-tb] bad size/sha for {rel_path}")
        if rel_path in seen:
            raise SystemExit(f"[nb-tb] duplicate path in manifest: {rel_path}")
        seen.add(rel_path)
        entries.append((rel_path, int(size_s), sha))

if not entries:
    raise SystemExit("[nb-tb] empty manifest")

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
                    st_code = resp.status
                    if st_code not in (200, 206):
                        raise RuntimeError(f"HTTP {st_code}")
                    chunks = []
                    while True:
                        buf = resp.read(4 * 1024 * 1024)
                        if not buf:
                            break
                        chunks.append(buf)
                    data = b"".join(chunks)
                    resp.drain_conn()
                finally:
                    resp.release_conn()
            else:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=90) as resp:
                    st_code = resp.status
                    data = resp.read()
            if st_code == 200 and start == 0 and len(data) == expected:
                return data
            if len(data) != expected:
                raise RuntimeError(f"short read {len(data)} != {expected}")
            return data
        except Exception:
            if get_thread_http is not None and hasattr(_TLS, "pool"):
                try:
                    _TLS.pool.clear()
                except Exception:
                    pass
                _TLS.pool = None
            if attempt == 3:
                raise
            time.sleep(0.6 * (attempt + 1))
    raise RuntimeError("unreachable")


def resolve_cdn_url(hf_url: str) -> str:
    try:
        req = urllib.request.Request(hf_url, headers={"Range": "bytes=0-0", **UA})
        with urllib.request.urlopen(req, timeout=30) as resp:
            return resp.geturl() or hf_url
    except Exception:
        return hf_url


def verify_and_commit_file(part_path: str, final_path: str, expected_size: int, expected_sha: str) -> None:
    st_size = os.path.getsize(part_path)
    if st_size != expected_size:
        raise RuntimeError(f"size mismatch on {final_path}: {st_size} != {expected_size}")
    with open(part_path, "rb") as f:
        h = hashlib.file_digest(f, "sha256")
    got_sha = h.hexdigest()
    if got_sha != expected_sha:
        raise RuntimeError(f"SHA-256 mismatch on {final_path}: {got_sha} != {expected_sha}")
    os.replace(part_path, final_path)


small_entries = [e for e in entries if e[1] <= SMALL_CUTOFF]
large_entries = [e for e in entries if e[1] > SMALL_CUTOFF]


def download_small(entry):
    rel_path, size, sha = entry
    hf_url = f"https://huggingface.co/{repo}/resolve/{rev}/{rel_path}"
    dest = os.path.join(dest_root, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    data = b"" if size == 0 else fetch_bytes_range(hf_url, 0, size - 1)
    if len(data) != size or hashlib.sha256(data).hexdigest() != sha:
        raise RuntimeError(f"Verification failed on small file {rel_path}")
    part = dest + ".part"
    with open(part, "wb") as f:
        f.write(data)
    os.replace(part, dest)


if small_entries:
    with ThreadPoolExecutor(max_workers=min(8, len(small_entries))) as pool:
        for fut in as_completed([pool.submit(download_small, e) for e in small_entries]):
            fut.result()


def prep_large(entry):
    rel_path, size, sha = entry
    hf_url = f"https://huggingface.co/{repo}/resolve/{rev}/{rel_path}"
    cdn_url = resolve_cdn_url(hf_url)
    dest = os.path.join(dest_root, rel_path)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    part = dest + ".part"
    fd = os.open(part, os.O_CREAT | os.O_RDWR | os.O_TRUNC, 0o644)
    os.ftruncate(fd, size)
    return {
        "rel_path": rel_path, "size": size, "sha": sha,
        "hf_url": hf_url, "cdn_url": cdn_url, "dest": dest, "part": part,
        "fd": fd, "rem_slices": 0, "lock": threading.Lock(),
    }


if large_entries:
    with ThreadPoolExecutor(max_workers=min(12, len(large_entries))) as rpool:
        shard_states = list(rpool.map(prep_large, large_entries))

    hash_pool = ThreadPoolExecutor(max_workers=4)
    hash_futs = []
    slice_tasks = []
    for st in shard_states:
        sz = st["size"]
        ranges = [(s, min(s + CHUNK_SZ - 1, sz - 1)) for s in range(0, sz, CHUNK_SZ)]
        st["rem_slices"] = len(ranges)
        for start, end in ranges:
            slice_tasks.append((st, start, end))

    def run_slice(task):
        st, start, end = task
        try:
            data = fetch_bytes_range(st["cdn_url"], start, end)
        except Exception:
            data = fetch_bytes_range(st["hf_url"], start, end)
        pos = start
        mv = memoryview(data)
        written = 0
        while written < len(mv):
            written += os.pwrite(st["fd"], mv[written:], pos + written)
        with st["lock"]:
            st["rem_slices"] -= 1
            done_shard = (st["rem_slices"] == 0)
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

actual_files = [os.path.join(r, fn) for r, _d, fs in os.walk(dest_root) for fn in fs]
if len(actual_files) != len(entries):
    raise SystemExit(f"[nb-tb] unexpected file count {len(actual_files)} != {len(entries)}")

total_bytes = sum(e[1] for e in entries)
dt = max(0.1, time.time() - t0)
print(
    f"[nb-tb] WEIGHTS_OK model={model_name} files={len(entries)} gb={total_bytes/1e9:.2f} secs={dt:.1f} rate_MBps={(total_bytes/1e6)/dt:.0f}",
    flush=True,
)
PY

rm -rf "$OUT"
mkdir -p "$OUT"

RESUME_ARG=()
if [ -s "/workspace/resume_track_b_results.json" ]; then
  RESUME_ARG=(--resume-json "/workspace/resume_track_b_results.json")
fi

printf '[nb-tb] Starting offline Nullbridge Track B pipeline for %s...\n' "$NAME"
(
  cd /workspace
  env HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 \
      PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
      PYTHONPATH=/workspace/src \
      python3 -m nullbridge.cli track-b \
        --package-dir "$PKG" \
        --base-model "$DIR" \
        --model-label "$REPO" \
        --batch-size "$BATCH_SIZE" \
        --lora-steps "$LORA_STEPS" \
        --no-plots \
        "${RESUME_ARG[@]}" \
        --out-dir "$OUT"
) 2>&1 | tee "$LOGDIR/nullbridge_track_b_${NAME}.log"

[ -s "$OUT/track_b_results.json" ] || { printf '[nb-tb] ERROR: track_b_results.json missing for %s\n' "$NAME" >&2; exit 4; }
rm -rf "$DIR"
printf '[nb-tb] MODEL_DONE %s\n' "$NAME"
