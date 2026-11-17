#!/usr/bin/env bash
# External monitor for the four-experiment batch. Not part of scripts/: changes
# nothing in the harness or runner. Stops the batch's own process session on a
# traceback, a PROCESS ALERT ERROR, or a watchdog stack. The runner itself already
# stops on any status other than COMPLETED.
# usage: monitor_four.sh <runs_dir> <batch_pid> <log>
set -u
RUNS=$1; PID=$2; LOG=$3
say() { echo "$(date -u +%T) $*" >> "$LOG"; }
say "monitor start batch_pid=$PID"
LAST=""
while kill -0 "$PID" 2>/dev/null; do
  for E in A_norestart A_restart B_norestart B_restart; do
    R="$RUNS/$E"
    [ -d "$R" ] || continue
    TB=$(grep -c 'Traceback (most recent' "$R/run.log" 2>/dev/null); TB=${TB:-0}
    PE=$(grep -c 'PROCESS ALERT ERROR' "$R/run.log" 2>/dev/null); PE=${PE:-0}
    WD=$(grep -c '^Timeout (' "$R/diagnostics.log" 2>/dev/null); WD=${WD:-0}
    if [ "$TB" -gt 0 ] || [ "$PE" -gt 0 ] || [ "$WD" -gt 0 ]; then
      say "VIOLATION $E tracebacks=$TB process_alert_errors=$PE watchdog=$WD: stopping batch session"
      kill -TERM -"$PID" 2>/dev/null
      echo "VIOLATION $E" > "$LOG.done"
      exit 1
    fi
    if [ -f "$R/status" ]; then
      NOW="$E=$(cat "$R/status")/$(cat "$R/exit_status")"
      case " $LAST " in *" $NOW "*) ;; *) say "$NOW"; LAST="$LAST $NOW" ;; esac
    fi
  done
  sleep 3
done
say "batch process ended: $(tail -1 "$RUNS/batch_status.tsv" | cut -c1-90)"
echo "ENDED" > "$LOG.done"
