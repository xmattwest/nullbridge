#!/bin/bash
# run_autonomous_exfil.sh (POD-SIDE):
#   Wraps any inner pod runner command (`"$@"`), streams live status + block
#   checkpoints + final `results.tgz` + `var_log_afriat.tgz` directly to Azure Blob
#   Storage using a short-lived, write-only (`sp=cw`) User Delegation SAS stored in
#   0600 tmpfs `/run/afriat/exfil_env`, and immediately triggers `/run/afriat/delete_now`
#   so the pod self-deletes cleanly even when the user's laptop is offline.

set +x +v +a
set -uo pipefail
export LC_ALL=C

EXFIL_ENV="/run/afriat/exfil_env"
LOGDIR="/var/log/afriat"
mkdir -p "$LOGDIR"

log_ex() {
  printf '[af-exfil] %s %s\n' "$(date -u +%H:%M:%SZ)" "$*" >> "$LOGDIR/exfil.log"
}

if [ ! -f "$EXFIL_ENV" ]; then
  log_ex "ERROR: missing $EXFIL_ENV"
  exit 2
fi

# shellcheck disable=SC1090
. "$EXFIL_ENV"

: "${EXFIL_CONTAINER_URL:?}"
: "${EXFIL_SAS:?}"
: "${EXFIL_PREFIX:?}"
: "${EXFIL_TASK:?}"
: "${EXFIL_MODEL:?}"
: "${EXFIL_OUT_DIR:?}"
: "${EXFIL_RES_JSON:?}"

az_put_file() {
  local local_path="$1"
  local remote_name="$2"
  local ctype="${3:-application/octet-stream}"
  local try code
  [ -f "$local_path" ] || return 1
  for try in 1 2 3 4; do
    code="$(
      printf 'url = "%s/%s/%s?%s"\n' "$EXFIL_CONTAINER_URL" "$EXFIL_PREFIX" "$remote_name" "$EXFIL_SAS" |
        curl -q -sS --proto '=https' --tlsv1.2 --connect-timeout 15 --max-time 300 \
          -o /dev/null -w '%{http_code}' -X PUT \
          -H "x-ms-blob-type: BlockBlob" \
          -H "Content-Type: $ctype" \
          --data-binary @"$local_path" \
          --config - 2>/dev/null
    )" || true
    case "$code" in
      200|201) return 0 ;;
      *) log_ex "PUT $remote_name try=$try HTTP=${code:-000}"; sleep $((try * 3)) ;;
    esac
  done
  return 1
}

write_status_json() {
  local state="$1"
  local rc="$2"
  local res_ok="$3"
  local res_sha="$4"
  local msg="$5"
  local tmp="/run/afriat/status_payload.json"
  python3 - "$tmp" "$state" "$rc" "$res_ok" "$res_sha" "$msg" "$STARTED_UTC" "$EXFIL_TASK" "$EXFIL_MODEL" "$EXFIL_PREFIX" <<'PY'
import datetime, json, sys
out_p, state, rc, res_ok, res_sha, msg, started, task, model, prefix = sys.argv[1:11]
now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
payload = {
    "state": state,
    "task": task,
    "model": model,
    "prefix": prefix,
    "started_utc": started,
    "updated_utc": now,
    "exit_code": int(rc),
    "results_uploaded": bool(int(res_ok)),
    "results_tgz_sha256": res_sha,
    "note": msg,
}
with open(out_p, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
PY
  az_put_file "$tmp" "status.json" "application/json" || log_ex "WARN: failed to PUT status.json ($state)"
}

STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
log_ex "Autonomous runner starting task=$EXFIL_TASK model=$EXFIL_MODEL prefix=$EXFIL_PREFIX"
write_status_json "running" 0 0 "" "job started on pod"

# Background heartbeat & block-checkpoint uploader (every 25s)
heartbeat_loop() {
  while :; do
    sleep 25
    [ -f "/run/afriat/job_done_flag" ] && break
    if [ -s "$EXFIL_OUT_DIR/$EXFIL_RES_JSON" ]; then
      cp "$EXFIL_OUT_DIR/$EXFIL_RES_JSON" /run/afriat/ckpt_latest.json 2>/dev/null &&
        az_put_file /run/afriat/ckpt_latest.json "checkpoint_${EXFIL_RES_JSON}" "application/json" || true
    elif [ -s "$EXFIL_OUT_DIR/checkpoint_${EXFIL_RES_JSON}" ]; then
      cp "$EXFIL_OUT_DIR/checkpoint_${EXFIL_RES_JSON}" /run/afriat/ckpt_latest.json 2>/dev/null &&
        az_put_file /run/afriat/ckpt_latest.json "checkpoint_${EXFIL_RES_JSON}" "application/json" || true
    fi
    if [ -s "$LOGDIR/${EXFIL_TASK}_${EXFIL_MODEL}.log" ]; then
      tail -c 65536 "$LOGDIR/${EXFIL_TASK}_${EXFIL_MODEL}.log" > /run/afriat/live_tail.log 2>/dev/null &&
        az_put_file /run/afriat/live_tail.log "live_tail.log" "text/plain; charset=utf-8" || true
    fi
    write_status_json "running" 0 0 "" "heartbeat"
  done
}

rm -f /run/afriat/job_done_flag
heartbeat_loop &
HB_PID=$!

# Execute inner model runner (keep EXFIL_SAS unexported so child procs never see it!)
"$@"
JOB_RC=$?

touch /run/afriat/job_done_flag
kill "$HB_PID" 2>/dev/null || true
wait "$HB_PID" 2>/dev/null || true

RES_UPLOADED=0
RES_SHA=""
if [ -d "$EXFIL_OUT_DIR" ] && [ "$(ls -A "$EXFIL_OUT_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
  if tar -C "$EXFIL_OUT_DIR" -czf /tmp/af_results.tgz . 2>>"$LOGDIR/exfil.log"; then
    if command -v sha256sum >/dev/null 2>&1; then
      RES_SHA="$(sha256sum /tmp/af_results.tgz | cut -d' ' -f1)"
    else
      RES_SHA="$(shasum -a 256 /tmp/af_results.tgz | cut -d' ' -f1)"
    fi
    if az_put_file /tmp/af_results.tgz "results.tgz" "application/gzip"; then
      RES_UPLOADED=1
      log_ex "Uploaded results.tgz (sha256=$RES_SHA)"
    else
      log_ex "ERROR: failed to upload results.tgz"
    fi
    rm -f /tmp/af_results.tgz
  fi
fi

if tar -C "$LOGDIR" -czf /tmp/af_var_log.tgz . 2>/dev/null; then
  az_put_file /tmp/af_var_log.tgz "var_log_afriat.tgz" "application/gzip" &&
    log_ex "Uploaded var_log_afriat.tgz" || log_ex "WARN: failed to upload var_log_afriat.tgz"
  rm -f /tmp/af_var_log.tgz
fi

if [ "$JOB_RC" = 0 ] && [ "$RES_UPLOADED" = 1 ]; then
  write_status_json "succeeded" 0 1 "$RES_SHA" "completed and exfiltrated cleanly"
else
  write_status_json "failed" "$JOB_RC" "$RES_UPLOADED" "$RES_SHA" "job exited rc=$JOB_RC res_uploaded=$RES_UPLOADED"
fi

rm -f "$EXFIL_ENV" /run/afriat/ckpt_latest.json /run/afriat/live_tail.log /run/afriat/status_payload.json
log_ex "Triggering immediate pod self-delete (rc=$JOB_RC)"
printf 'autonomous-exfil-done rc=%s\n' "$JOB_RC" > /run/afriat/delete_now
exit "$JOB_RC"
