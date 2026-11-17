#!/usr/bin/env bash
# usage: start_batch_v2.sh <experiment> [<experiment> ...]
# Detaches the runner from the launching session (one wsl.exe call, then none
# needed), so a dropped or hung wsl.exe cannot stop the batch. Output and exit
# status are captured in the runs directory.
#
# Handshake: closing the wsl.exe session within milliseconds of launch kills the
# child before setsid takes effect (observed, see checks/runner_diag2). This
# script therefore does not return until the runner has recorded activity in
# batch_status.tsv, and exits 1 if it does not within 30 s.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${B3_RUNS_DIR:-$ROOT/runs}"
mkdir -p "$RUNS"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$RUNS/batch_$STAMP.log"
STATUS="$RUNS/batch_status.tsv"
BEFORE=$( { wc -l < "$STATUS"; } 2>/dev/null || echo 0)
setsid nohup bash -c 'bash "$0" "$@"; echo "batch_exit_status=$?" >> "'"$LOG"'"' \
  "$ROOT/scripts/run_b3_v2.sh" "$@" > "$LOG" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$RUNS/batch_$STAMP.pid"
for _ in $(seq 1 60); do
  NOW=$( { wc -l < "$STATUS"; } 2>/dev/null || echo 0)
  if [ "$NOW" -gt "$BEFORE" ]; then
    echo "started pid=$PID log=$LOG"
    exit 0
  fi
  sleep 0.5
done
echo "runner did not record a start within 30 s; pid=$PID log=$LOG" >&2
exit 1
