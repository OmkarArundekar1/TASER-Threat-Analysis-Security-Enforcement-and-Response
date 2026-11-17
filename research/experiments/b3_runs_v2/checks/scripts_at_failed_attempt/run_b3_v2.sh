#!/usr/bin/env bash
# usage: run_b3_v2.sh <experiment> [<experiment> ...]
# experiments: CUR_restart A_norestart A_restart B_norestart B_restart
#
# There is no default list: nothing runs unless it is named explicitly. Each
# experiment gets its own fresh database and its own run directory; an existing
# run directory is never reused. Stops the batch on any failure.
#
# Per run: run.log (stdout+stderr, unbuffered), diagnostics.log (harness
# progress, restart markers, watchdog stacks), neo4j_resources.txt, preflight.txt,
# exit_status. Batch: batch_status.tsv.
#
# B3_DRY_RUN=<rc>: test the runner itself, no container and no replay: the
# "experiment" just exits with <rc> after B3_DRY_SLEEP seconds.
# B3_RUNS_DIR: override the output directory (runner tests only).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${B3_RUNS_DIR:-$ROOT/runs}"
PY="$HOME/capstone/soc_env/bin/python"
ALERTS="$HOME/capstone/isfcr/CYUKTI/data/raw_logs/alerts.json"
MIN_AVAILABLE_KB=1500000
RESTART_AT=4500
STATUS="$RUNS/batch_status.tsv"
mkdir -p "$RUNS"

[ $# -gt 0 ] || { echo "usage: $0 <experiment> [...]" >&2; exit 2; }

record() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" >> "$STATUS"; }

for EXPERIMENT in "$@"; do
  case "$EXPERIMENT" in
    CUR_restart) VARIANT=CUR; EXTRA="--restart-at $RESTART_AT" ;;
    A_norestart) VARIANT=A;   EXTRA="" ;;
    A_restart)   VARIANT=A;   EXTRA="--restart-at $RESTART_AT" ;;
    B_norestart) VARIANT=B;   EXTRA="" ;;
    B_restart)   VARIANT=B;   EXTRA="--restart-at $RESTART_AT" ;;
    *) echo "unknown experiment: $EXPERIMENT" >&2; record "$EXPERIMENT" REJECTED unknown; exit 2 ;;
  esac
  RUN="$RUNS/$EXPERIMENT"
  if [ -e "$RUN" ]; then
    echo "refusing to reuse existing run directory: $RUN" >&2
    record "$EXPERIMENT" REFUSED run_dir_exists
    exit 3
  fi
  mkdir -p "$RUN"

  { date -u +%FT%TZ; free -k; cat /proc/buddyinfo; } > "$RUN/preflight.txt"
  AVAILABLE=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$AVAILABLE" -lt "$MIN_AVAILABLE_KB" ]; then
    echo LOW_MEMORY > "$RUN/exit_status"
    record "$EXPERIMENT" LOW_MEMORY "available_kb=$AVAILABLE"
    exit 4
  fi

  record "$EXPERIMENT" START "available_kb=$AVAILABLE"
  if [ -n "${B3_DRY_RUN:-}" ]; then
    bash -c "echo dry-run $EXPERIMENT; sleep ${B3_DRY_SLEEP:-0}; exit $B3_DRY_RUN" > "$RUN/run.log" 2>&1
    RC=$?
  else
    SHA=$(sha256sum "$ALERTS" | cut -d' ' -f1)
    if [ "$SHA" != "$(cat "$ROOT/V1_INPUT_SHA256")" ]; then
      echo INPUT_CHANGED > "$RUN/exit_status"
      record "$EXPERIMENT" INPUT_CHANGED "$SHA"
      exit 5
    fi
    (
      cd "$ROOT/variants/$VARIANT" || exit 9
      B3_VARIANT=$VARIANT bash "$ROOT/scripts/with_fresh_neo4j_v2.sh" "$EXPERIMENT" "$RUN" \
        "$PY" -u -m replay.harness --alerts "$ALERTS" --out "$RUN" $EXTRA
    ) > "$RUN/run.log" 2>&1
    RC=$?
  fi
  echo "$RC" > "$RUN/exit_status"
  record "$EXPERIMENT" FINISHED "rc=$RC"
  [ "$RC" -eq 0 ] || exit "$RC"
done
record BATCH DONE "rc=0"
