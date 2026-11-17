#!/usr/bin/env bash
# usage: run_b3_v2.sh <experiment> [<experiment> ...]
# experiments: CUR_restart_r2 CUR_norestart A_norestart A_restart B_norestart B_restart
#
# There is no default list: nothing runs unless it is named explicitly. Each
# experiment gets its own fresh database and its own run directory; an existing
# run directory is never reused. The batch stops at the first status other than
# COMPLETED, and nothing is retried.
#
# Per run: run.log (stdout+stderr, unbuffered), diagnostics.log, stage,
# neo4j_resources.txt, container_status, cleanup_status, preflight.txt,
# status (classification), exit_status (actual exit code), status.json, SHA256SUMS.
# Batch: batch_status.tsv.
#
# Test modes (refused unless B3_RUNS_DIR points away from the real runs/):
#   B3_DRY_RUN=<rc>         no container: the "experiment" exits <rc> after B3_DRY_SLEEP s
#   B3_TEST_COMMAND=<cmd>   real wrapper and disposable container, <cmd> instead of the harness
# B3_EXPERIMENT_TIMEOUT overrides the 30-minute limit (seconds).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="${B3_RUNS_DIR:-$ROOT/runs}"
PY="$HOME/capstone/soc_env/bin/python"
ALERTS="$HOME/capstone/isfcr/CYUKTI/data/raw_logs/alerts.json"
MIN_AVAILABLE_KB=1500000
RESTART_AT=4500
EXPERIMENT_TIMEOUT="${B3_EXPERIMENT_TIMEOUT:-1800}"
KILL_AFTER=45
STATUS="$RUNS/batch_status.tsv"
CLASSIFY="$PY $ROOT/scripts/classify_run.py"
MODE=full
[ -n "${B3_DRY_RUN:-}" ] && MODE=dry
[ -n "${B3_TEST_COMMAND:-}" ] && MODE=test

[ $# -gt 0 ] || { echo "usage: $0 <experiment> [...]" >&2; exit 2; }
if [ "$MODE" != full ] && { [ -z "${B3_RUNS_DIR:-}" ] || [ "$RUNS" = "$ROOT/runs" ]; }; then
  echo "test modes require B3_RUNS_DIR outside $ROOT/runs" >&2
  exit 2
fi
mkdir -p "$RUNS"

record() { printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$1" "$2" "$3" >> "$STATUS"; }

for EXPERIMENT in "$@"; do
  case "$EXPERIMENT" in
    CUR_restart_r2) VARIANT=CUR; EXTRA="--restart-at $RESTART_AT" ;;
    CUR_norestart)  VARIANT=CUR; EXTRA="" ;;
    A_norestart)    VARIANT=A;   EXTRA="" ;;
    A_restart)      VARIANT=A;   EXTRA="--restart-at $RESTART_AT" ;;
    B_norestart)    VARIANT=B;   EXTRA="" ;;
    B_restart)      VARIANT=B;   EXTRA="--restart-at $RESTART_AT" ;;
    *) echo "unknown experiment: $EXPERIMENT" >&2; record "$EXPERIMENT" REJECTED unknown; exit 2 ;;
  esac
  RUN="$RUNS/$EXPERIMENT"
  NAME="cyukti-b3v2-$EXPERIMENT"
  if [ -e "$RUN" ]; then
    echo "refusing to reuse existing run directory: $RUN" >&2
    record "$EXPERIMENT" REFUSED run_dir_exists
    exit 3
  fi
  mkdir -p "$RUN"
  STARTED=$(date -u +%FT%TZ)
  { echo "$STARTED"; free -k; cat /proc/buddyinfo; } > "$RUN/preflight.txt"

  # ---- runner-level prechecks
  AVAILABLE=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  PRECHECK=""
  [ "$AVAILABLE" -lt "$MIN_AVAILABLE_KB" ] && PRECHECK="low memory: available_kb=$AVAILABLE"
  if [ "$MODE" = full ]; then
    SHA=$(sha256sum "$ALERTS" | cut -d' ' -f1)
    [ "$SHA" = "$(cat "$ROOT/V1_INPUT_SHA256")" ] || PRECHECK="input changed: $SHA"
  fi
  if [ -n "$PRECHECK" ]; then
    RESULT=$($CLASSIFY "$RUN" --exit-code 20 --mode "$MODE" --started "$STARTED" --ended "$(date -u +%FT%TZ)" \
      --force-status PRECHECK_FAILED --force-detail "$PRECHECK")
    record "$EXPERIMENT" "$RESULT" "rc=20 $PRECHECK"
    exit 20
  fi

  record "$EXPERIMENT" START "available_kb=$AVAILABLE mode=$MODE timeout_s=$EXPERIMENT_TIMEOUT"
  BEGIN=$(date +%s)
  if [ "$MODE" = dry ]; then
    timeout --kill-after=$KILL_AFTER "$EXPERIMENT_TIMEOUT" \
      bash -c "echo dry-run $EXPERIMENT; sleep ${B3_DRY_SLEEP:-0}; exit $B3_DRY_RUN" > "$RUN/run.log" 2>&1
    RC=$?
  else
    if [ "$MODE" = test ]; then
      COMMAND=(bash -c "$B3_TEST_COMMAND")
    else
      COMMAND=("$PY" -u -m replay.harness --alerts "$ALERTS" --out "$RUN" $EXTRA)
    fi
    (
      cd "$ROOT/variants/$VARIANT" || exit 21
      # GNU timeout leads its own process group and signals the whole group.
      B3_VARIANT=$VARIANT B3_RUN_DIR="$RUN" timeout --kill-after=$KILL_AFTER "$EXPERIMENT_TIMEOUT" \
        bash "$ROOT/scripts/with_fresh_neo4j_v2.sh" "$EXPERIMENT" "$RUN" "${COMMAND[@]}"
    ) > "$RUN/run.log" 2>&1
    RC=$?
  fi
  ELAPSED=$(( $(date +%s) - BEGIN ))

  TIMED_OUT=""
  if [ "$RC" -eq 124 ] || { [ "$RC" -eq 137 ] && [ "$ELAPSED" -ge "$EXPERIMENT_TIMEOUT" ]; }; then
    TIMED_OUT=--timed-out
  fi

  # Independent of the wrapper's own cleanup (which a KILL would prevent): remove
  # only this run's container, by exact name, with its anonymous volumes.
  if [ "$MODE" != dry ]; then
    LEFT=$(timeout 30 docker ps -aq --filter "name=^/?${NAME}\$" 2>/dev/null | wc -l)
    if [ "$LEFT" -gt 0 ]; then
      timeout 30 docker rm -f -v "$NAME" >/dev/null 2>&1
      echo "runner_cleanup removed=$NAME rc=$? containers_remaining=$(timeout 30 docker ps -aq --filter "name=^/?${NAME}\$" 2>/dev/null | wc -l)" >> "$RUN/runner_cleanup"
    fi
  fi

  RESULT=$($CLASSIFY "$RUN" --exit-code "$RC" --mode "$MODE" $TIMED_OUT --experiment-timeout "$EXPERIMENT_TIMEOUT" \
    --started "$STARTED" --ended "$(date -u +%FT%TZ)")
  (cd "$RUN" && find . -maxdepth 1 -type f ! -name SHA256SUMS ! -name '.*' -print0 | sort -z | xargs -0 sha256sum > .SHA256SUMS.tmp && mv -f .SHA256SUMS.tmp SHA256SUMS)
  record "$EXPERIMENT" "$RESULT" "rc=$RC stage=$(cat "$RUN/stage" 2>/dev/null) elapsed_s=$ELAPSED"
  if [ "$RESULT" != COMPLETED ]; then
    record BATCH STOPPED "at=$EXPERIMENT status=$RESULT"
    exit 1
  fi
done
record BATCH DONE "rc=0"
