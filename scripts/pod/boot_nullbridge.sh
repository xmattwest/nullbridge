#!/bin/bash
# boot_afriat.sh: startup script inside the pinned vllm/vllm-openai image for Project Afriat.
#
# Security & lifecycle guardrails:
#   1. Captures pod-scoped RUNPOD_API_KEY into an unexported variable, unsets secret env vars,
#      and starts the lifetime watchdog immediately (/run/afriat/delete_now or AF_LIFETIME_S).
#   2. Installs openssh-server, generates a fresh ed25519 host key, pins our single public key
#      (AF_SSH_KEY_FPR) in /etc/ssh/af_authorized_keys (ignoring /root/.ssh/authorized_keys),
#      logs SSH_HOST_KEY and SSH_READY, and waits on the watchdog while the laptop runs
#      manifest-verified offline discovery jobs over pinned SSH.

set +x +v +a
set -uo pipefail
umask 022
export LC_ALL=C
PATH="$PATH:/usr/sbin:/sbin"

RUN="/run/afriat"
LOGDIR="/var/log/afriat"
SSHDIR="/etc/ssh"
MODELS="/models"
WORK="/workspace"
TICK=5
RETRY=30
readonly API="https://api.runpod.io/v2"

log() {
  printf '[af-boot] %s\n' "$*"
  [ -d "$LOGDIR" ] && printf '[af-boot] %s\n' "$*" >> "$LOGDIR/boot.log"
  return 0
}

POD_KEY="${RUNPOD_API_KEY:-}"
unset RUNPOD_API_KEY HF_TOKEN HUGGING_FACE_HUB_TOKEN

die_early() {
  log "BOOT_FAILED $*"
  log "cannot self-delete; the laptop launcher will delete this pod"
  exit 1
}

LIFETIME="${AF_LIFETIME_S:-}"
re_int='^[0-9]+$'
if ! [[ "$LIFETIME" =~ $re_int ]] || [ "$LIFETIME" -lt 60 ] || [ "$LIFETIME" -gt 21600 ]; then
  die_early "AF_LIFETIME_S must be 60..21600"
fi
POD_ID="${RUNPOD_POD_ID:-}"
re_id='^[A-Za-z0-9]{1,64}$'
[[ "$POD_ID" =~ $re_id ]] || die_early "RUNPOD_POD_ID missing or malformed"
[ -n "$POD_KEY" ] || die_early "RUNPOD_API_KEY (pod-scoped) missing"

mkdir -p "$RUN" "$LOGDIR" "$MODELS" "$WORK" || die_early "cannot create runtime dirs"
rm -f "$RUN/delete_now"

self_delete() {
  printf 'Authorization: Bearer %s\n' "$POD_KEY" |
    curl -q -sS --proto '=https' --tlsv1.2 --connect-timeout 10 --max-time 30 \
      -o /dev/null -w '%{http_code}' -X DELETE -H @- "$API/pods/$POD_ID" 2>/dev/null
}

 watchdog() {
  local end reason code now
  log "watchdog started: lifetime $LIFETIME s"
  end=$(( $(date +%s) + LIFETIME ))
  while :; do
    now="$(date +%s)"
    if [ "$now" -ge "$end" ]; then reason="lifetime of $LIFETIME s reached"; break; fi
    if [ -e "$RUN/delete_now" ]; then
      reason="delete requested ($(head -c 120 "$RUN/delete_now" | tr -cd 'A-Za-z0-9 ._:-'))"
      break
    fi
    sleep "$TICK"
  done
  log "WATCHDOG $reason; deleting this pod"
  if [ -f "$RUN/exfil_env" ]; then
    (
      # Best-effort emergency log upload on lifetime timeout before pod deletion
      # shellcheck disable=SC1090
      . "$RUN/exfil_env"
      if [ -n "${EXFIL_CONTAINER_URL:-}" ] && [ -n "${EXFIL_SAS:-}" ] && [ -n "${EXFIL_PREFIX:-}" ]; then
        if [ -n "${EXFIL_OUT_DIR:-}" ] && [ -d "$EXFIL_OUT_DIR" ] && [ "$(ls -A "$EXFIL_OUT_DIR" 2>/dev/null | wc -l)" -gt 0 ]; then
          tar -C "$EXFIL_OUT_DIR" -czf /tmp/af_wd_res.tgz . 2>/dev/null &&
            printf 'url = "%s/%s/results_partial.tgz?%s"\n' "$EXFIL_CONTAINER_URL" "$EXFIL_PREFIX" "$EXFIL_SAS" |
              curl -q -sS --proto '=https' --tlsv1.2 --max-time 60 -o /dev/null -X PUT \
                -H 'x-ms-blob-type: BlockBlob' -H 'Content-Type: application/gzip' \
                --data-binary @/tmp/af_wd_res.tgz --config - 2>/dev/null || true
          rm -f /tmp/af_wd_res.tgz
        fi
        tar -C "$LOGDIR" -czf /tmp/af_wd_logs.tgz . 2>/dev/null &&
          printf 'url = "%s/%s/var_log_afriat.tgz?%s"\n' "$EXFIL_CONTAINER_URL" "$EXFIL_PREFIX" "$EXFIL_SAS" |
            curl -q -sS --proto '=https' --tlsv1.2 --max-time 45 -o /dev/null -X PUT \
              -H 'x-ms-blob-type: BlockBlob' -H 'Content-Type: application/gzip' \
              --data-binary @/tmp/af_wd_logs.tgz --config - 2>/dev/null || true
        printf '{"state":"timeout","task":"%s","model":"%s","prefix":"%s","exit_code":124,"results_uploaded":false,"note":"watchdog lifetime reached"}\n' \
          "${EXFIL_TASK:-?}" "${EXFIL_MODEL:-?}" "${EXFIL_PREFIX:-}" > /tmp/af_wd_status.json
        printf 'url = "%s/%s/status.json?%s"\n' "$EXFIL_CONTAINER_URL" "$EXFIL_PREFIX" "$EXFIL_SAS" |
          curl -q -sS --proto '=https' --tlsv1.2 --max-time 20 -o /dev/null -X PUT \
            -H 'x-ms-blob-type: BlockBlob' -H 'Content-Type: application/json' \
            --data-binary @/tmp/af_wd_status.json --config - 2>/dev/null || true
      fi
      rm -f "$RUN/exfil_env" /tmp/af_wd_logs.tgz /tmp/af_wd_status.json
    ) || true
  fi
  while :; do
    code="$(self_delete)" || true
    case "$code" in
      200|204) log "SELF_DELETE ok (HTTP $code)"; exit 0 ;;
      *)       log "SELF_DELETE FAILED (HTTP ${code:-none}); retrying in $RETRY s" ;;
    esac
    sleep "$RETRY"
  done
}

watchdog &
WD_PID=$!
sleep 0.2

fail() {
  log "BOOT_FAILED $*"
  printf 'boot failed\n' > "$RUN/delete_now"
  wait "$WD_PID"
  exit 1
}

FPR="${AF_SSH_KEY_FPR:-}"
KEY="${PUBLIC_KEY:-}"
re_key='^ssh-ed25519 [A-Za-z0-9+/]+={0,2}( [A-Za-z0-9._@-]{1,64})?$'
[ "$(printf '%s' "$KEY" | grep -c .)" = "1" ] || fail "PUBLIC_KEY must hold exactly one key"
[[ "$KEY" =~ $re_key ]] || fail "PUBLIC_KEY is not a single ssh-ed25519 key"
re_fpr='^SHA256:[A-Za-z0-9+/]{43}$'
[[ "$FPR" =~ $re_fpr ]] || fail "AF_SSH_KEY_FPR malformed"

mkdir -p "$SSHDIR" || fail "cannot create $SSHDIR"
if command -v sshd >/dev/null 2>&1; then
  log "openssh-server pre-installed in image (0s apt skip)"
else
  log "installing openssh-server"
  APT_OPTS=(-o Acquire::ForceIPv4=true -o Acquire::http::Timeout=20 -o Acquire::Retries=3)
  DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" update -qq > "$LOGDIR/apt.log" 2>&1 &&
    DEBIAN_FRONTEND=noninteractive apt-get "${APT_OPTS[@]}" install -y -qq --no-install-recommends openssh-server \
      >> "$LOGDIR/apt.log" 2>&1 || fail "apt-get could not install openssh-server"
fi

AUTH="$SSHDIR/af_authorized_keys"
printf '%s\n' "$KEY" > "$RUN/key.pub"
got="$(ssh-keygen -lf "$RUN/key.pub" 2>/dev/null | cut -d' ' -f2)"
[ "$got" = "$FPR" ] || fail "PUBLIC_KEY fingerprint does not match AF_SSH_KEY_FPR"
printf '%s\n' "$KEY" > "$AUTH" || fail "cannot write $AUTH"
chmod 644 "$AUTH"

HOSTKEY="$SSHDIR/af_host_ed25519_key"
rm -f "$HOSTKEY" "$HOSTKEY.pub"
ssh-keygen -q -t ed25519 -N '' -C af-pod-host -f "$HOSTKEY" >/dev/null 2>&1 || fail "cannot make host key"

CONF="$SSHDIR/af_sshd_config"
cat > "$CONF" <<EOF || fail "cannot write $CONF"
Port 22
ListenAddress 0.0.0.0
HostKey $HOSTKEY
PidFile $RUN/sshd.pid
AuthorizedKeysFile $AUTH
AuthorizedKeysCommand none
AuthorizedPrincipalsFile none
PubkeyAuthentication yes
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitEmptyPasswords no
HostbasedAuthentication no
UsePAM no
PermitRootLogin prohibit-password
AllowUsers root
StrictModes yes
AllowAgentForwarding no
AllowTcpForwarding no
GatewayPorts no
X11Forwarding no
PermitTunnel no
PermitUserEnvironment no
MaxAuthTries 3
LoginGraceTime 20
ClientAliveInterval 30
ClientAliveCountMax 6
LogLevel VERBOSE
EOF

mkdir -p /run/sshd
SSHD_BIN="$(command -v sshd)"
case "$SSHD_BIN" in /*) ;; *) fail "sshd not found after install" ;; esac
"$SSHD_BIN" -t -f "$CONF" > "$LOGDIR/sshd_test.log" 2>&1 || fail "sshd rejected its config"
"$SSHD_BIN" -f "$CONF" -E "$LOGDIR/sshd.log" || fail "sshd did not start"

log "SSH_HOST_KEY $(cut -d' ' -f1,2 "$HOSTKEY.pub")"
log "SSH_HOST_KEY_FPR $(ssh-keygen -lf "$HOSTKEY.pub" | cut -d' ' -f2)"
log "SSH_READY (key $FPR only)"
log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -4 | tr '\n' ';' || echo unknown)"
log "host CUDA: $(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9.]*' | head -1 || echo unknown)"

wait "$WD_PID"
