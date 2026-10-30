# This repository-owned hook is sourced by the shared train driver. The fast
# preflight is the default until the build server's Windows images carry what
# Synapse's Windows CI job needs (Python and LLVM's libclang); until then the
# full gate's Windows job stops on a missing prerequisite, not on our code.
# TRAIN_GATE_FULL=1 runs full push-CI parity.
if [ "${TRAIN_GATE_FULL:-0}" != 1 ]; then
  bash scripts/gate.sh --fast || refuse "fast repository preflight failed"
  say "Fast repository preflight passed (not full push-CI parity)"
else
  bash scripts/gate.sh || refuse "repository gate failed (see named job/step and phase timings above)"
  say "Full push-CI parity gate passed"
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
