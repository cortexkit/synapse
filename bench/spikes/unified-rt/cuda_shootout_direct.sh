#!/usr/bin/env bash
# Wrap a corpus CLI with the campaign's 500 ms nvidia-smi sampler.
set -euo pipefail
result=$1
shift
mkdir -p "$(dirname "$result")"
raw="${result%.json}.lane.json"
telemetry="${result%.json}.smi.csv"
stdout="${result%.json}.stdout"
stderr="${result%.json}.stderr"
while true; do
  util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
  processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -c '[0-9]' || true)
  if [[ $util -le 2 && $processes -eq 0 ]]; then break; fi
  sleep 2
done
load_start=$(cut -d' ' -f1-3 /proc/loadavg)
start_ns=$(date +%s%N)
: > "$telemetry"
(
  while true; do
    values=$(nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits | tr -d ' ')
    processes=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -c '[0-9]' || true)
    printf '%s,%s,%s\n' "$(date +%s%N)" "$values" "$processes" >> "$telemetry"
    sleep 0.5
  done
) &
sampler=$!
trap 'kill "$sampler" 2>/dev/null || true' EXIT
set +e
"$@" --out "$raw" > "$stdout" 2> "$stderr"
status=$?
set -e
kill "$sampler" 2>/dev/null || true
wait "$sampler" 2>/dev/null || true
trap - EXIT
end_ns=$(date +%s%N)
load_end=$(cut -d' ' -f1-3 /proc/loadavg)
python3 - "$raw" "$telemetry" "$result" "$status" "$start_ns" "$end_ns" "$load_start" "$load_end" <<'PY'
import csv,json,statistics,sys
raw,telemetry,out,status,start,end,load_start,load_end=sys.argv[1:]
data=json.load(open(raw)) if int(status)==0 else None
samples=[]
with open(telemetry) as source:
 for row in csv.reader(source):
  if len(row)==5: samples.append((float(row[1]),float(row[2]),float(row[3]),int(row[4])))
result={"exit_status":int(status),"process_wall_s":(int(end)-int(start))/1e9,"host_load_start":[float(x) for x in load_start.split()],"host_load_end":[float(x) for x in load_end.split()],"gpu_samples":len(samples),"avg_gpu_watts":statistics.mean(x[0] for x in samples),"peak_gpu_watts":max(x[0] for x in samples),"avg_gpu_util_pct":statistics.mean(x[1] for x in samples),"peak_gpu_util_pct":max(x[1] for x in samples),"peak_vram_mib":max(x[2] for x in samples),"max_compute_processes":max(x[3] for x in samples),"contaminated":max(x[3] for x in samples)>1,"lane_result":data}
json.dump(result,open(out,"w"),indent=2)
PY
exit "$status"
