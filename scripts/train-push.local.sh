# Repo-local preflight sourced by scripts/train-push.sh before a train is
# pushed: the same workflow-shape assertions CI runs as its first step, run
# here first because a drifted workflow would fail in seconds on CI anyway and
# failing locally is free.
bash scripts/check-train-preconditions.sh || refuse "train preconditions failed — see scripts/check-train-preconditions.sh"
bash scripts/check-no-external-path-deps.sh || refuse "a Cargo path dependency resolves outside the repository — see scripts/check-no-external-path-deps.sh"
python3 scripts/check-module-runtime-deps.py || refuse "shipped module includes daemon or presence — see scripts/check-module-runtime-deps.py"

# CI compiles 7 of the workspace members. The Metal engine and the ANE worker
# need a Mac, and the macos runner left with the M1 box, so this
# preflight is the ONLY automated gate they have — synapse-engine-owned is the
# primary production engine on this platform and would otherwise reach master
# having been compiled nowhere but an author's terminal. Warm cost is about ten
# seconds; a cold build is slower and still cheaper than shipping it unbuilt.
# Delete this block the day a Mac runner rejoins CI, not before.
if [ "$(uname -s)" = "Darwin" ]; then
  # The MLX worker crate this block used to name is gone: it was deleted from
  # the workspace, so there is no longer an MLX lane to gate here.
  mac_crates="-p synapse-engine-owned -p synapse-worker-ane"
  # The Metal toolchain lives in full Xcode; Command Line Tools alone cannot
  # compile the shaders, and the failure is a confusing linker error rather
  # than a missing-tool message.
  if [ -d /Applications/Xcode.app ]; then
    export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
  fi
  # --locked: the subc and commons crates are exact crates.io pins in the root
  # Cargo.toml (no sibling path checkouts), so a Cargo.lock that would need
  # rewriting means the committed lock is wrong, which is worth refusing.
  # shellcheck disable=SC2086
  cargo clippy --locked $mac_crates --all-targets -- -D warnings \
    || refuse "macOS-only crates failed clippy (CI cannot run these; see scripts/train-push.local.sh)"
  cargo test --locked -p synapse-engine-owned --lib \
    || refuse "synapse-engine-owned lib tests failed (CI cannot run these)"
fi

# Linux clippy, run here because otherwise a Linux-only failure (code behind
# cfg(not(target_os = "macos")), or test helpers left dead when their callers
# are macOS-only) is found only by a CI train about ten minutes later. It
# checks the crates CI lints, read from tests.yml so the two lists cannot
# drift. ring's build script needs a Linux C compiler, which zig provides
# through the scripts/lib/zig-*.sh shims. Without zig or the rust target this is
# a notice, not a refusal: CI still gates the train.
linux_target=x86_64-unknown-linux-gnu
if command -v zig >/dev/null 2>&1 && rustup target list --installed 2>/dev/null | grep -qx "$linux_target"; then
  ci_crates="$(sed -n 's/^ *SYNAPSE_CRATES: *//p' .github/workflows/tests.yml | head -1)"
  [ -n "$ci_crates" ] || refuse "could not read SYNAPSE_CRATES from .github/workflows/tests.yml for the Linux clippy check"
  # The llama worker's build script compiles llama.cpp through CMake, which
  # zig cannot cross-build from here; that one crate stays CI-only.
  ci_crates="$(printf '%s\n' "$ci_crates" | sed 's/-p synapse-worker-llama//')"
  # shellcheck disable=SC2086
  CC_x86_64_unknown_linux_gnu="$PWD/scripts/lib/zig-cc-linux.sh" \
  AR_x86_64_unknown_linux_gnu="$PWD/scripts/lib/zig-ar.sh" \
    cargo clippy --locked --target "$linux_target" $ci_crates --all-targets -- -D warnings \
    || refuse "Linux-target clippy failed (the same check CI's linux job runs; see scripts/train-push.local.sh)"
else
  say "NOTICE: skipping the Linux clippy preflight (needs zig and 'rustup target add $linux_target'); CI will be the first Linux check"
fi

# Windows clippy, for the same reason: code behind cfg(windows) or
# cfg(unix), or a helper whose callers are Unix-only, otherwise fails only on
# CI's windows job. CI builds with MSVC, which can't run here, so this checks
# the windows-gnu target instead: cfg(windows) and cfg(unix) resolve the same
# way on both, so it catches those lints, but not anything specific to MSVC.
# Same crate list as the Linux check, minus the llama worker.
windows_target=x86_64-pc-windows-gnu
if command -v zig >/dev/null 2>&1 && rustup target list --installed 2>/dev/null | grep -qx "$windows_target"; then
  ci_crates="$(sed -n 's/^ *SYNAPSE_CRATES: *//p' .github/workflows/tests.yml | head -1)"
  [ -n "$ci_crates" ] || refuse "could not read SYNAPSE_CRATES from .github/workflows/tests.yml for the Windows clippy check"
  ci_crates="$(printf '%s\n' "$ci_crates" | sed 's/-p synapse-worker-llama//')"
  # shellcheck disable=SC2086
  CC_x86_64_pc_windows_gnu="$PWD/scripts/lib/zig-cc-windows.sh" \
  AR_x86_64_pc_windows_gnu="$PWD/scripts/lib/zig-ar.sh" \
    cargo clippy --locked --target "$windows_target" $ci_crates --all-targets -- -D warnings \
    || refuse "Windows-target clippy failed (windows-gnu, standing in for CI's MSVC windows job; see scripts/train-push.local.sh)"
else
  say "NOTICE: skipping the Windows clippy preflight (needs zig and 'rustup target add $windows_target'); CI will be the first Windows check"
fi

# The daily cron on tests.yml is this repository's only scheduled sample of
# master against the current toolchain and runner images, which a push-only
# CI never sees change. Its result reaches nobody unless something reads it:
# a red scheduled run raises no alert, and repositories in this fleet have sat
# red for days before anyone looked. Read it here, where every landing passes.
# NOTICES, not refusals: a scheduled red is about what already landed, and the
# push in front of you may be its fix. --event schedule is filtered
# server-side, so a burst of push runs cannot crowd the last scheduled run out
# of the list. The 48 h age bound is above GitHub's routine queue lag (a cron
# landing five hours late is healthy) and catches the case a green last run
# hides: a cron that fired, then stopped.
if command -v gh >/dev/null 2>&1; then
  sched="$(gh run list --event schedule --limit 1 --json conclusion,createdAt,headSha \
    --jq '.[] | "\(.conclusion) \(.createdAt[0:16]) \(.headSha[0:8])"' 2>/dev/null || true)"
  if [ -z "$sched" ]; then
    if grep -q "schedule:" .github/workflows/*.yml 2>/dev/null; then
      say "NOTICE: a cron is declared in .github/workflows but NO scheduled run exists; a schedule that never fires looks exactly like one that passes"
    fi
  else
    sched_ts="${sched#* }"; sched_ts="${sched_ts%% *}"
    sched_age_h="$(python3 -c "
import sys,datetime
t=datetime.datetime.fromisoformat(sys.argv[1]+':00+00:00')
print(int((datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()//3600))" "$sched_ts" 2>/dev/null || true)"
    if [ -n "${sched_age_h:-}" ] && [ "$sched_age_h" -gt "${TRAIN_SCHEDULE_MAX_AGE_H:-48}" ] 2>/dev/null; then
      say "NOTICE: the last SCHEDULED run is ${sched_age_h}h old ($sched); a cron that stopped firing leaves a green last run"
    fi
    case "$sched" in
      success*) : ;;
      *) say "NOTICE: the last SCHEDULED run was not green: $sched (the tips-siblings sample; a lock wave may have broken master while pushes stay green)" ;;
    esac
  fi
fi
