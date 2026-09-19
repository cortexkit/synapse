# Repo-local preflight sourced by scripts/train-push.sh before a train is
# pushed: the same workflow-shape assertions CI runs as its first step, run
# here first because a drifted workflow would fail in seconds on CI anyway and
# failing locally is free.
bash scripts/check-train-preconditions.sh || refuse "train preconditions failed — see scripts/check-train-preconditions.sh"

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
  # These two commands rewrite Cargo.lock as a side effect, because the sibling
  # path dependencies (../subconscious, ../commons) are other agents' working
  # checkouts and routinely sit ahead of siblings.lock. That rewrite dirties the
  # tree, and the NEXT train then refuses on a change this script made itself.
  #
  # --locked does not solve it: it refuses to run at all whenever a sibling has
  # moved, which here is most of the time, and those checkouts are not ours to
  # roll back. CI is unaffected either way because it checks the siblings out at
  # the pinned commits. So verify against whatever is on disk, then put the lock
  # back exactly as it was — and leave a deliberate lock edit alone.
  lock_was_clean=no
  git diff --quiet -- Cargo.lock 2>/dev/null && lock_was_clean=yes
  restore_lock() {
    if [ "$lock_was_clean" = yes ]; then
      git checkout -- Cargo.lock 2>/dev/null || true
    fi
  }
  # shellcheck disable=SC2086
  cargo clippy $mac_crates --all-targets -- -D warnings \
    || { restore_lock; refuse "macOS-only crates failed clippy (CI cannot run these; see scripts/train-push.local.sh)"; }
  cargo test -p synapse-engine-owned --lib \
    || { restore_lock; refuse "synapse-engine-owned lib tests failed (CI cannot run these)"; }
  restore_lock
fi

# The daily cron on tests.yml is this repository's only sample of "same sha,
# siblings at their tips" (pushes build against siblings.lock, so a lock wave
# that breaks the tips is invisible to them). Its result reaches nobody unless
# something reads it, and three seats found multi-day red streaks this way
# (fleet notices #486, #487, #490). Read it here, where every landing passes.
# NOTICES, not refusals: a scheduled red is about what already landed, and the
# push in front of you may be its fix. --event schedule is filtered server-side,
# so a push flood cannot empty the window (#489). The 48 h age bound is above
# GitHub's routine queue lag (a cron landing five hours late is healthy) and
# catches the case a green last run hides: a cron that fired, then stopped.
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
