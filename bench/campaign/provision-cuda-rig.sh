#!/bin/sh
set -eu

# Provision the second campaign host without changing the campaign registry or
# starting Athena. Run this as root on the already-rented Ubuntu CUDA host.
CANDIDATE_USER=${CANDIDATE_USER:-ck-candidate}
CONTROLLER_USER=${CONTROLLER_USER:-${SUDO_USER:-root}}
CAMPAIGN_ROOT=${CAMPAIGN_ROOT:-/opt/ck-campaign}
REPO_URL=${REPO_URL:-https://github.com/cortexkit/synapse.git}
SUBCONSCIOUS_URL=${SUBCONSCIOUS_URL:-https://github.com/cortexkit/subconscious.git}
COMMONS_URL=${COMMONS_URL:-https://github.com/cortexkit/commons.git}
M1_RUNNER_SSH_TARGET=${M1_RUNNER_SSH_TARGET:-}
M1_RUNNER_SOURCE_DIR=${M1_RUNNER_SOURCE_DIR:-[bench-user-home]/ck-campaign/rig/candidate-runner}
SYNAPSE_SOURCE=${SYNAPSE_SOURCE:-}
SIBLING_SUBCONSCIOUS_SOURCE=${SIBLING_SUBCONSCIOUS_SOURCE:-}
SIBLING_COMMONS_SOURCE=${SIBLING_COMMONS_SOURCE:-}
VAST_RENTAL_RATE_USD_PER_HOUR=${VAST_RENTAL_RATE_USD_PER_HOUR:-}
VAST_PLAN_HOURS=${VAST_PLAN_HOURS:-12}
VAST_RELIABILITY=${VAST_RELIABILITY:-}
VAST_SSH_COORDINATES=${VAST_SSH_COORDINATES:-}
MODEL_REVISION=c1899de289a04d12100db370d81485cdf75e47ca
MODEL_DIGEST=0d7d1359007f579fba9f6eceef44c87b947362da893cc565d27656284e4d6f86

usage() {
    printf '%s\n' "usage: provision-cuda-rig.sh [--self-test]"
    printf '%s\n' "Environment controls paths; see the script header for defaults."
}

if [ "${1:-}" = --help ]; then
    usage
    exit 0
fi
if [ "${1:-}" = --self-test ]; then
    script_dir=$(unset CDPATH; cd -- "$(dirname -- "$0")" && pwd)
    /bin/sh -n "$script_dir/provision-cuda-rig.sh"
    /bin/sh -n "$script_dir/cuda-quant-harness.sh"
    printf '%s\n' "provision-cuda-rig self-test passed (POSIX shell syntax)"
    exit 0
fi
if [ "$#" -ne 0 ]; then
    usage >&2
    exit 2
fi

if [ "$(id -u)" -ne 0 ]; then
    printf '%s\n' "provision-cuda-rig.sh must run as root" >&2
    exit 1
fi

log() { printf '[cuda-provision] %s\n' "$*"; }
fatal() { printf '[cuda-provision] ERROR: %s\n' "$*" >&2; exit 1; }

SCRIPT_DIR=$(unset CDPATH; cd -- "$(dirname -- "$0")" && pwd)
REPO_DIR=$CAMPAIGN_ROOT/synapse
# Cargo workspace path dependencies resolve beside the workspace checkout.
SIBLINGS_DIR=$CAMPAIGN_ROOT
SHARED_RUSTUP=$CAMPAIGN_ROOT/rustup
SHARED_CARGO=$CAMPAIGN_ROOT/cargo
MODEL_CACHE=${MODEL_CACHE:-$CAMPAIGN_ROOT/huggingface}
MODEL_DIR=${MODEL_DIR:-$MODEL_CACHE/models--Qwen--Qwen3-0.6B/snapshots/$MODEL_REVISION}
RUNNER_DIR=$CAMPAIGN_ROOT/candidate-runner
RUNNER_COPY=$RUNNER_DIR/m1-runner.sh
SCRATCH_DIR=$CAMPAIGN_ROOT/scratch-master
RESULT_DIR=$CAMPAIGN_ROOT/results/master-baseline
RESULT_PATH=$RESULT_DIR/result.json
IDENTITY_DIR=$CAMPAIGN_ROOT/smoke/identity-cwd
SUDOERS_FILE=/etc/sudoers.d/ck-campaign-cuda

[ -x /usr/bin/python3 ] || fatal "python3 is required"
[ -x /usr/sbin/visudo ] || fatal "visudo is required"
[ -x /usr/bin/git ] || fatal "git is required"
[ -x /usr/bin/curl ] || fatal "curl is required for rustup/model hydration"
if ! command -v iptables >/dev/null 2>&1 || ! command -v ip6tables >/dev/null 2>&1; then
    command -v apt-get >/dev/null 2>&1 || fatal "iptables/ip6tables are missing and apt-get is unavailable"
    apt-get update
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends iptables
fi

controller_home=$(getent passwd "$CONTROLLER_USER" | cut -d: -f6)
[ -n "$controller_home" ] && [ -d "$controller_home" ] || fatal "controller user has no home: $CONTROLLER_USER"

# The controller invokes the runner as this user. Keeping this wrapper explicit
# makes the smoke tests exercise the same non-interactive sudo path as ALF.
controller_exec() {
    if [ "$CONTROLLER_USER" = root ]; then
        "$@"
    else
        runuser -u "$CONTROLLER_USER" -- "$@"
    fi
}

candidate_exec() {
    controller_exec sudo -n -u "$CANDIDATE_USER" -- "$@"
}

root_from_controller() {
    controller_exec sudo -n -- "$@"
}

ensure_line() {
    line=$1
    file=$2
    touch "$file"
    chmod 600 "$file"
    if ! /usr/bin/grep -Fqx "$line" "$file"; then
        printf '%s\n' "$line" >>"$file"
    fi
}

log "start=$(date -u +%Y-%m-%dT%H:%M:%SZ) rental_rate_usd_per_hour=$VAST_RENTAL_RATE_USD_PER_HOUR"
log "ssh_coordinates=${VAST_SSH_COORDINATES:-not-supplied}; spend_cap_usd=8; rig is intentionally left running"

[ -n "$VAST_RELIABILITY" ] || fatal "set VAST_RELIABILITY from the Vast offer; required minimum is > 0.99"
[ -n "$VAST_RENTAL_RATE_USD_PER_HOUR" ] || fatal "set VAST_RENTAL_RATE_USD_PER_HOUR for spend visibility"
/usr/bin/python3 - "$VAST_RELIABILITY" "$VAST_RENTAL_RATE_USD_PER_HOUR" "$VAST_PLAN_HOURS" <<'PY'
import sys
reliability = float(sys.argv[1])
rate = float(sys.argv[2])
hours = float(sys.argv[3])
if reliability <= 0.99:
    raise SystemExit("Vast reliability must be > 0.99")
if hours < 10 or hours > 12:
    raise SystemExit("overnight plan must be 10-12 hours")
if rate <= 0 or rate * hours > 8.0:
    raise SystemExit("planned rental exceeds the $8 cap: rate*hours=%.3f" % (rate * hours))
PY

# A rented host is admitted only when it has the requested class of GPU and
# driver. The controller records the complete state later in scene.json.
command -v nvidia-smi >/dev/null 2>&1 || fatal "nvidia-smi is missing"
GPU_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader,nounits | head -n 1)
/usr/bin/python3 - "$GPU_DRIVER" <<'PY'
import sys
raw = sys.argv[1].strip()
try:
    major = int(raw.split('.', 1)[0])
except ValueError:
    raise SystemExit("could not parse nvidia driver version: " + raw)
if major < 570:
    raise SystemExit("driver must be >= 570, got " + raw)
PY
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
case "$GPU_NAME" in
    *4090*) : ;;
    *) fatal "expected an RTX 4090, got: $GPU_NAME" ;;
esac

install -d -m 755 "$CAMPAIGN_ROOT" "$SIBLINGS_DIR" "$RUNNER_DIR" "$RESULT_DIR" "$MODEL_DIR"

if ! id "$CANDIDATE_USER" >/dev/null 2>&1; then
    useradd --create-home --home-dir "/home/$CANDIDATE_USER" --shell /usr/sbin/nologin "$CANDIDATE_USER"
fi
install -d -o "$CANDIDATE_USER" -g "$CANDIDATE_USER" -m 700 "/home/$CANDIDATE_USER"

# This is deliberately a command allowlist. Candidate execution is scoped to
# the unprivileged account; only process observation/kill, firewall rules, and
# ownership repair are permitted as root.
cat >"$SUDOERS_FILE" <<EOF
Defaults:${CONTROLLER_USER} !requiretty
Cmnd_Alias CK_CANDIDATE_EXEC = /usr/bin/env *, /bin/sh *, /bin/pwd, /usr/bin/pwd, /bin/sleep *, /usr/bin/sleep *, /usr/bin/python3 *, /usr/bin/curl *, /usr/bin/git *, /usr/bin/cargo *, /usr/bin/nvidia-smi *, /bin/mkdir *, /bin/cp *, /bin/chmod *, /bin/rm *, /bin/ls *, /tmp/*, $CAMPAIGN_ROOT/*
Cmnd_Alias CK_CANDIDATE_PROBE = /usr/bin/pkill -0 -u $CANDIDATE_USER, /usr/bin/pkill -0 -u $CANDIDATE_USER *, /usr/bin/pkill -KILL -u $CANDIDATE_USER, /usr/bin/pkill -KILL -u $CANDIDATE_USER *
Cmnd_Alias CK_CANDIDATE_FIREWALL = /usr/sbin/iptables -w -I OUTPUT -m owner --uid-owner $CANDIDATE_USER -o lo -j ACCEPT, /usr/sbin/iptables -w -A OUTPUT -m owner --uid-owner $CANDIDATE_USER -j REJECT, /usr/sbin/iptables -w -D OUTPUT -m owner --uid-owner $CANDIDATE_USER -o lo -j ACCEPT, /usr/sbin/iptables -w -D OUTPUT -m owner --uid-owner $CANDIDATE_USER -j REJECT, /usr/sbin/ip6tables -w -I OUTPUT -m owner --uid-owner $CANDIDATE_USER -o lo -j ACCEPT, /usr/sbin/ip6tables -w -A OUTPUT -m owner --uid-owner $CANDIDATE_USER -j REJECT, /usr/sbin/ip6tables -w -D OUTPUT -m owner --uid-owner $CANDIDATE_USER -o lo -j ACCEPT, /usr/sbin/ip6tables -w -D OUTPUT -m owner --uid-owner $CANDIDATE_USER -j REJECT
Cmnd_Alias CK_CANDIDATE_CHOWN = /usr/bin/chown *, /bin/chown *
${CONTROLLER_USER} ALL=($CANDIDATE_USER) NOPASSWD: CK_CANDIDATE_EXEC
${CONTROLLER_USER} ALL=(root) NOPASSWD: CK_CANDIDATE_PROBE, CK_CANDIDATE_FIREWALL, CK_CANDIDATE_CHOWN
EOF
chmod 440 "$SUDOERS_FILE"
visudo -cf "$SUDOERS_FILE"

# The delayed-re-entry guard is required before a runner can admit work.
ensure_line "$CANDIDATE_USER" /etc/at.deny
ensure_line "$CANDIDATE_USER" /etc/cron.deny
chown root:root /etc/at.deny /etc/cron.deny
chmod 600 /etc/at.deny /etc/cron.deny

install_rustup_for_user() {
    user=$1
    home=$2
    rustup_bin=$home/.cargo/bin/rustup
    if [ ! -x "$rustup_bin" ]; then
        runuser -u "$user" -- env HOME="$home" sh -c 'curl --fail --silent --show-error https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable'
    fi
    runuser -u "$user" -- env HOME="$home" "$rustup_bin" toolchain install stable --profile minimal
}

install_shared_rustup() {
    if [ ! -x "$SHARED_CARGO/bin/cargo" ]; then
        install -d -m 755 "$SHARED_RUSTUP" "$SHARED_CARGO"
        env HOME=/root RUSTUP_HOME="$SHARED_RUSTUP" CARGO_HOME="$SHARED_CARGO" \
            /usr/bin/curl --fail --silent --show-error https://sh.rustup.rs | \
            env HOME=/root RUSTUP_HOME="$SHARED_RUSTUP" CARGO_HOME="$SHARED_CARGO" \
            sh -s -- -y --profile minimal --default-toolchain stable
    fi
    env RUSTUP_HOME="$SHARED_RUSTUP" CARGO_HOME="$SHARED_CARGO" \
        "$SHARED_CARGO/bin/rustup" toolchain install stable --profile minimal
    chmod -R a+rX "$SHARED_RUSTUP" "$SHARED_CARGO"
}

log "hydrating stable Rust for controller=$CONTROLLER_USER and shared candidate-readable paths"
install_rustup_for_user "$CONTROLLER_USER" "$controller_home"
install_shared_rustup

clone_or_stage() {
    source=$1
    destination=$2
    url=$3
    if [ -n "$source" ]; then
        [ -d "$source" ] || fatal "source path is missing: $source"
        install -d -m 755 "$(dirname "$destination")"
        rsync -a --delete "$source/" "$destination/"
    elif [ ! -d "$destination/.git" ]; then
        git clone --depth 1 "$url" "$destination"
    fi
    [ -d "$destination" ] || fatal "staging failed: $destination"
    chmod -R a+rX "$destination"
}

if [ -n "$SYNAPSE_SOURCE" ]; then
    clone_or_stage "$SYNAPSE_SOURCE" "$REPO_DIR" "$REPO_URL"
elif [ ! -d "$REPO_DIR/.git" ]; then
    git clone --depth 1 "$REPO_URL" "$REPO_DIR"
fi
chmod -R a+rX "$REPO_DIR"
clone_or_stage "$SIBLING_SUBCONSCIOUS_SOURCE" "$SIBLINGS_DIR/subconscious" "$SUBCONSCIOUS_URL"
clone_or_stage "$SIBLING_COMMONS_SOURCE" "$SIBLINGS_DIR/commons" "$COMMONS_URL"

log "prehydrating offline Cargo registry"
env RUSTUP_HOME="$SHARED_RUSTUP" CARGO_HOME="$SHARED_CARGO" \
    "$SHARED_CARGO/bin/cargo" fetch --locked --manifest-path "$REPO_DIR/bench/spikes/unified-rt/Cargo.toml"

verify_snapshot() {
    /usr/bin/python3 - "$1" "$MODEL_DIGEST" <<'PY'
import hashlib
import pathlib
import sys
root = pathlib.Path(sys.argv[1])
expected = sys.argv[2]
files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.relative_to(root).as_posix())
if not files:
    raise SystemExit("model snapshot is empty: " + str(root))
h = hashlib.sha256()
for path in files:
    h.update(path.relative_to(root).as_posix().encode())
    h.update(b"\0")
    with path.open("rb") as stream:
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    h.update(b"\0")
actual = h.hexdigest()
if actual != expected:
    raise SystemExit("model snapshot SHA-256 mismatch: expected %s got %s" % (expected, actual))
print(actual)
PY
}

if ! verify_snapshot "$MODEL_DIR" >/dev/null 2>&1; then
    if ! command -v hf >/dev/null 2>&1 && ! command -v huggingface-cli >/dev/null 2>&1; then
        command -v pip3 >/dev/null 2>&1 || fatal "Hugging Face CLI is missing and pip3 is unavailable"
        pip3 install --disable-pip-version-check --no-input huggingface_hub
    fi
    HF_CLI=$(command -v hf || command -v huggingface-cli || true)
    [ -n "$HF_CLI" ] || fatal "Hugging Face CLI is required to hydrate Qwen3-0.6B"
    log "downloading Qwen/Qwen3-0.6B revision $MODEL_REVISION with $HF_CLI"
    "$HF_CLI" download Qwen/Qwen3-0.6B \
        --revision "$MODEL_REVISION" \
        --cache-dir "$MODEL_CACHE"
fi
verify_snapshot "$MODEL_DIR"
chmod -R a+rX "$MODEL_DIR"

log "running harness self-test"
"$SCRIPT_DIR/cuda-quant-harness.sh" --self-test

# Install the newest M1 runner as an immutable copy. The runner is ALF's TCB;
# this script never edits it and refuses to substitute a pass-through runner.
if [ -n "$M1_RUNNER_SSH_TARGET" ]; then
    remote_runner=$(ssh "$M1_RUNNER_SSH_TARGET" /bin/sh -s -- "$M1_RUNNER_SOURCE_DIR" <<'REMOTE_RUNNER_QUERY'
find "$1" -maxdepth 2 -type f -perm -111 -print | sort | tail -n 1
REMOTE_RUNNER_QUERY
    )
    [ -n "$remote_runner" ] || fatal "M1 runner directory has no executable runner"
    scp "$M1_RUNNER_SSH_TARGET:$remote_runner" "$RUNNER_COPY"
elif [ -d "$M1_RUNNER_SOURCE_DIR" ]; then
    newest_runner=$(find "$M1_RUNNER_SOURCE_DIR" -maxdepth 2 -type f -perm -111 -print | sort | tail -n 1)
    [ -n "$newest_runner" ] || fatal "M1 runner directory has no executable runner"
    cp -- "$newest_runner" "$RUNNER_COPY"
else
    fatal "set M1_RUNNER_SSH_TARGET or provide M1_RUNNER_SOURCE_DIR; runner copy is mandatory"
fi
chmod 755 "$RUNNER_COPY"
RUNNER_SHA256=$(sha256sum "$RUNNER_COPY" | cut -d' ' -f1)
log "copied ALF M1 runner sha256=$RUNNER_SHA256 path=$RUNNER_COPY"

# Linux-specific runner evidence is kept verbatim in the provisioning log.
UNAME=$(uname -s)
[ "$UNAME" = Linux ] || fatal "this provisioning script requires Linux, got $UNAME"
if /usr/bin/grep -Eq 'ps[[:space:]]+-e[of][[:space:]]|ps[[:space:]]+eo[[:space:]]|ps[[:space:]]+ef|ps[[:space:]]+-U[[:space:]].*-axo' "$RUNNER_COPY"; then
    log "linux finding ps: procps-compatible survivor classifier found (including BSD-compatible -U/-axo form)"
else
    fatal "linux finding ps: runner has no procps-compatible survivor classifier"
fi
if /usr/bin/grep -Eq 'launchd|Darwin|darwin' "$RUNNER_COPY"; then
    log "linux finding launchd: Darwin allowlist is present and Linux branch can fail closed"
else
    fatal "linux finding launchd: runner lacks the Darwin-branched survivor allowlist"
fi
if /usr/bin/grep -Eq '(^|[^[:alnum:]_])timeout([[:space:]]|\()' "$RUNNER_COPY"; then
    log "linux finding timeout: copied runner contains a timeout(1) branch; Linux has /usr/bin/timeout and exercised that branch verbatim"
else
    log "linux finding timeout: no timeout(1) dependency; runner uses an in-script poll loop"
fi
SH_DIALECT=$(readlink -f /bin/sh 2>/dev/null || true)
case "$SH_DIALECT" in
    */dash) log "linux finding /bin/sh: dash ($SH_DIALECT); POSIX shell smoke is required" ;;
    *) fatal "linux finding /bin/sh: expected dash, got ${SH_DIALECT:-unknown}" ;;
esac

# (a) Identity-drop mechanics: controller creates a mode-700 cwd, then gives
# that directory to the candidate only through the same normalized exec path.
install -d -m 700 -o "$CONTROLLER_USER" -g "$CONTROLLER_USER" "$IDENTITY_DIR"
IDENTITY_DEADLINE_MS=$((($(date +%s) + 60) * 1000))
identity_output=$(cd "$IDENTITY_DIR" && env \
    ALFONSO_CANDIDATE_USER="$CANDIDATE_USER" \
    ALFONSO_CANDIDATE_HOME="/home/$CANDIDATE_USER" \
    ALFONSO_CANDIDATE_TMPDIR=/tmp \
    ALFONSO_CANDIDATE_DEADLINE_MS="$IDENTITY_DEADLINE_MS" \
    "$RUNNER_COPY" /bin/sh -c "printf 'HOME=%s PWD=%s\\n' \"\$HOME\" \"\$(/bin/pwd)\"")
case "$identity_output" in
    "HOME=/home/$CANDIDATE_USER PWD=/home/$CANDIDATE_USER") log "smoke (a) identity-drop: PASS evidence=$identity_output" ;;
    *) fatal "smoke (a) identity-drop failed verbatim: $identity_output" ;;
esac

# (b) Candidate process groups must be killable and fully reaped.
candidate_exec /usr/bin/env -i HOME="/home/$CANDIDATE_USER" PATH=/usr/bin:/bin /bin/sh -c '(sleep 300 &) ; sleep 300' &
SMOKE_SHELL_PID=$!
sleep 1
root_from_controller /usr/bin/pkill -0 -u "$CANDIDATE_USER" >/dev/null 2>&1 || fatal "smoke (b) pkill -0 did not observe candidate sleeper"
root_from_controller /usr/bin/pkill -KILL -u "$CANDIDATE_USER" >/dev/null 2>&1 || true
wait "$SMOKE_SHELL_PID" 2>/dev/null || true
if pgrep -u "$CANDIDATE_USER" >/dev/null 2>&1; then
    survivors=$(pgrep -a -u "$CANDIDATE_USER" || true)
    fatal "smoke (b) process-group reap failed verbatim: $survivors"
fi
log "smoke (b) process-group kill/reap: PASS evidence=pkill -0 observed; pkill -KILL left zero candidate survivors"

FIREWALL_CLEANED=0
cleanup_firewall() {
    [ "$FIREWALL_CLEANED" -eq 1 ] && return 0
    root_from_controller /usr/sbin/iptables -w -D OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -j REJECT >/dev/null 2>&1 || true
    root_from_controller /usr/sbin/iptables -w -D OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -o lo -j ACCEPT >/dev/null 2>&1 || true
    root_from_controller /usr/sbin/ip6tables -w -D OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -j REJECT >/dev/null 2>&1 || true
    root_from_controller /usr/sbin/ip6tables -w -D OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -o lo -j ACCEPT >/dev/null 2>&1 || true
    FIREWALL_CLEANED=1
}
trap cleanup_firewall EXIT INT TERM

# (c) Candidate-only egress deny must be installed by the fenced sudo verbs;
# loopback is explicitly allowed so a local package/cache path remains usable.
root_from_controller /usr/sbin/iptables -w -I OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -o lo -j ACCEPT
root_from_controller /usr/sbin/iptables -w -A OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -j REJECT
root_from_controller /usr/sbin/ip6tables -w -I OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -o lo -j ACCEPT
root_from_controller /usr/sbin/ip6tables -w -A OUTPUT -m owner --uid-owner "$CANDIDATE_USER" -j REJECT
candidate_exec /usr/bin/env -i HOME="/home/$CANDIDATE_USER" PATH=/usr/bin:/bin /bin/sh -c 'python3 -m http.server 18765 --bind 127.0.0.1 >/tmp/ck-loopback.log 2>&1 &'
sleep 1
candidate_exec /usr/bin/env -i HOME="/home/$CANDIDATE_USER" PATH=/usr/bin:/bin /usr/bin/curl --fail --silent --show-error --connect-timeout 3 http://127.0.0.1:18765/ >/dev/null
if candidate_exec /usr/bin/env -i HOME="/home/$CANDIDATE_USER" PATH=/usr/bin:/bin /usr/bin/curl --fail --silent --show-error --connect-timeout 3 --max-time 5 https://1.1.1.1/ >/tmp/ck-egress-proof.out 2>/tmp/ck-egress-proof.err; then
    fatal "smoke (c) egress proof unexpectedly succeeded"
fi
log "smoke (c) iptables/ip6tables egress: PASS evidence=external curl failed; loopback curl succeeded"
root_from_controller /usr/bin/pkill -KILL -u "$CANDIDATE_USER" >/dev/null 2>&1 || true

# (d) The full sandwich is the round-trip evidence: candidate-owned staging,
# build, quantized gates, hooks, samples, result, and failure-scene retention.
rm -rf -- "$SCRATCH_DIR"
git clone --depth 1 --branch master "$REPO_URL" "$SCRATCH_DIR"
chown -R "$CANDIDATE_USER:$CANDIDATE_USER" "$SCRATCH_DIR"
chmod -R u+rwX,go-rwx "$SCRATCH_DIR"
install -d -m 755 "$RESULT_DIR"
rm -f -- "$RESULT_PATH"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export SYNAPSE_CAMPAIGN_BASELINE_TOK_S=343.8
export SYNAPSE_CAMPAIGN_MODEL="$MODEL_DIR"
export SYNAPSE_CAMPAIGN_MODEL_SHA256="$MODEL_DIGEST"
export SYNAPSE_CAMPAIGN_CARGO="$SHARED_CARGO/bin/cargo"
export RUSTUP_HOME="$SHARED_RUSTUP"
export CARGO_HOME="$SHARED_CARGO"
export CARGO_NET_OFFLINE=1
export SYNAPSE_CAMPAIGN_SIBLINGS="$SIBLINGS_DIR/subconscious:$SIBLINGS_DIR/commons"
export ALFONSO_CANDIDATE_USER="$CANDIDATE_USER"
export ALFONSO_CANDIDATE_HOME="/home/$CANDIDATE_USER"
export ALFONSO_CANDIDATE_TMPDIR=/tmp
export ALFONSO_CANDIDATE_DEADLINE_MS=$((($(date +%s) + 1800) * 1000))
set +e
"$SCRIPT_DIR/cuda-quant-harness.sh" "$SCRATCH_DIR" "$RUNNER_COPY" "$RESULT_PATH"
HARNESS_STATUS=$?
set -e
if [ "$HARNESS_STATUS" -ne 0 ]; then
    fatal "smoke (d) full runner round-trip failed with status $HARNESS_STATUS; result=$RESULT_PATH"
fi
/usr/bin/python3 - "$RESULT_PATH" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
if payload.get("gate_passed") is not True or payload.get("hooks_passed") is not True:
    raise SystemExit("full round-trip result did not pass gates: " + json.dumps(payload))
if not isinstance(payload.get("median_tok_s"), (int, float)):
    raise SystemExit("full round-trip result has no numeric median")
print("full round-trip result gate_passed=true hooks_passed=true median_tok_s=%.3f" % payload["median_tok_s"])
PY
[ -s "$RESULT_DIR/scene.json" ] || fatal "smoke (d) did not write scene.json"
[ -s "$RESULT_DIR/result.json" ] || fatal "smoke (d) did not write result.json"
log "smoke (d) full runner round-trip: PASS evidence=result+scene+samples written at $RESULT_DIR"
log "Linux runner findings complete; rig remains provisioned for overnight use"
log "registration is intentionally not edited and Athena is intentionally not started"

HARNESS_SHA256=$(sha256sum "$SCRIPT_DIR/cuda-quant-harness.sh" | cut -d' ' -f1)
printf '%s\n' "registration block (report only; not applied):"
cat <<EOF
{
  "host": "${VAST_SSH_COORDINATES:-<set SSH coordinates>}",
  "harness_path": "$SCRIPT_DIR/cuda-quant-harness.sh",
  "harness_sha256": "$HARNESS_SHA256",
  "commands": {
    "parity": ["{harness}", "{workspace}", "{candidate_runner}", "{result}"],
    "objective": ["{harness}", "{workspace}", "{candidate_runner}", "{result}"]
  },
  "env": {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "SYNAPSE_CAMPAIGN_BASELINE_TOK_S": "343.8",
    "SYNAPSE_CAMPAIGN_MODEL": "$MODEL_DIR",
    "SYNAPSE_CAMPAIGN_MODEL_SHA256": "$MODEL_DIGEST",
    "SYNAPSE_CAMPAIGN_CARGO": "$SHARED_CARGO/bin/cargo",
    "RUSTUP_HOME": "$SHARED_RUSTUP",
    "CARGO_HOME": "$SHARED_CARGO",
    "CARGO_NET_OFFLINE": "1",
    "SYNAPSE_CAMPAIGN_SIBLINGS": "$SIBLINGS_DIR/subconscious:$SIBLINGS_DIR/commons"
  },
  "load_threshold_1m": 8
}
EOF
