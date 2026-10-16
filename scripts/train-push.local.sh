# Repo-local preflight sourced by scripts/train-push.sh before a train is
# pushed: the same workflow-shape assertions CI runs as its first step, run
# here first because a drifted workflow would fail in seconds on CI anyway and
# failing locally is free.
bash scripts/check-train-preconditions.sh || refuse "train preconditions failed — see scripts/check-train-preconditions.sh"
bash scripts/check-no-external-path-deps.sh || refuse "a Cargo path dependency resolves outside the repository — see scripts/check-no-external-path-deps.sh"
python3 scripts/check-module-runtime-deps.py || refuse "shipped module includes daemon or presence — see scripts/check-module-runtime-deps.py"

# CI compiles the portable workspace members. The Metal engine and Neural Engine
# worker require a Mac, so this preflight is their only automated compile gate.
# Keep it until CI has a Mac runner; otherwise changes to these crates could
# merge without any automated native build.
if [ "$(uname -s)" = "Darwin" ]; then
  cargo fmt --all --check || refuse "macOS formatting failed"
  # The MLX worker crate this block used to name is gone: it was deleted from
  # the workspace, so there is no longer an MLX lane to gate here.
  # Module clippy moved out of the default landing script, so keep its native
  # macOS lint coverage here alongside the Mac-only engines.
  mac_crates="-p synapse-module -p synapse-engine-owned -p synapse-worker-ane"
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
    || refuse "native macOS module/engine crates failed clippy (see scripts/train-push.local.sh)"
  cargo test --locked -p synapse-engine-owned --lib \
    || refuse "synapse-engine-owned lib tests failed (CI cannot run these)"
fi

# Native Linux and MSVC clippy run in parallel CI lanes before changes can merge.
# Repeating them with the zig cross-compiler serialized the push and did not check MSVC.
# Keep the macOS-only gates above: neither CI lane can compile Metal/ANE crates.
say "Linux and Windows clippy are CI gates; macOS-only clippy remains local"

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
