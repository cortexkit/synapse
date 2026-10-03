#!/usr/bin/env bash
# Interleaved dispatch-gap A/B on the full direct-ANE ModernBERT encoder.
#
# usage: gap_ab.sh OUT_DIR GAPS_MS BLOCKS CALLS_PER_BLOCK
#   e.g. gap_ab.sh "$OUT" 0,3 20 400
#
# Requires MODEL_SNAPSHOT (a gte-modernbert-base snapshot directory) and a
# release build of modernbert_full. Waits until the 5-minute load average is
# below 12 and no campaign holds the Neural Engine rig claim, samples load and
# the rig claim every 15 s while the run lasts, and afterwards copies every
# production Synapse log line stamped inside the run window, so a reader can
# see whether production served requests on any lane meanwhile.
set -euo pipefail

out=${1:?OUT_DIR}
gaps=${2:?GAPS_MS}
blocks=${3:?BLOCKS}
calls=${4:?CALLS_PER_BLOCK}
: "${MODEL_SNAPSHOT:?set MODEL_SNAPSHOT}"

here=$(cd "$(dirname "$0")" && pwd)
rig_db="$HOME/.local/share/cortexkit/prefrontal-core/store.db"
synapse_logs="$HOME/.local/share/cortexkit/synapse/logs"
mkdir -p "$out"

# Prints the number of rig claims, or "unknown" when the store cannot be read
# (it is briefly unopenable while its owner rewrites it). The start gate only
# accepts "0", so an unreadable store makes the script keep waiting.
rig_claims() {
    local attempt
    for attempt in 1 2 3; do
        if sqlite3 "file:$rig_db?mode=ro" "SELECT count(*) FROM campaign_rig_claim" 2>/dev/null; then
            return 0
        fi
        sleep 2
    done
    echo unknown
}

while true; do
    read -r _ l1 l5 l15 _ < <(sysctl -n vm.loadavg)
    claims=$(rig_claims)
    if awk -v a="$l5" 'BEGIN{exit !(a<12)}' && [ "$claims" = "0" ]; then
        break
    fi
    echo "waiting: load $l1/$l5/$l15 rig claims $claims at $(date +%H:%M:%S)" >&2
    sleep 60
done

start_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "start $start_utc load $l1/$l5/$l15 rig claims $claims" | tee "$out/run.log"

(
    while true; do
        read -r _ s1 s5 s15 _ < <(sysctl -n vm.loadavg)
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) load $s1 $s5 $s15 rig_claims $(rig_claims)"
        sleep 15
    done
) > "$out/monitor.log" &
monitor=$!
trap 'kill $monitor 2>/dev/null || true' EXIT

status=0
"$here/target/release/modernbert_full" "$MODEL_SNAPSHOT" "$here/rows.jsonl" \
    --seq 512 --layers-per-executable 1 --warm-repetitions 5 \
    --block-gaps-ms "$gaps" --blocks "$blocks" --calls-per-block "$calls" \
    --report "$out/report.json" --per-call-out "$out/per-call.json" \
    > /dev/null 2>> "$out/run.log" || status=$?

end_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "end $end_utc exit $status" | tee -a "$out/run.log"

# Log lines start with an ISO-8601 UTC timestamp, so string comparison selects
# the run window. Every log written in the last two days is read because the
# daily file name need not follow the UTC date.
find "$synapse_logs" -name 'synapse*.log' -mtime -2 -exec cat {} + 2>/dev/null \
    | sort -u \
    | awk -v s="$start_utc" -v e="$end_utc" 'substr($1,1,19)"Z" >= s && substr($1,1,19)"Z" <= e' \
    > "$out/synapse-during-run.log" || true
echo "synapse log lines during run: $(wc -l < "$out/synapse-during-run.log")" | tee -a "$out/run.log"
exit "$status"
