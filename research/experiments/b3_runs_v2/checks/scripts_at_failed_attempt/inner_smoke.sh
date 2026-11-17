#!/usr/bin/env bash
# Runs INSIDE with_fresh_neo4j_v2.sh, against its one disposable database.
# usage: inner_smoke.sh <workspace_root> <python> <smoke_dir>
set -u
ROOT=$1; PY=$2; SMOKE=$3
ALERTS="$SMOKE/synthetic_alerts.jsonl"
"$PY" "$ROOT/scripts/make_smoke_alerts.py" "$ALERTS"

echo "=== connectivity"
"$PY" "$ROOT/scripts/smoke_db.py" "$SMOKE/db_smoke.json"; echo "connectivity_rc=$?"

for V in CUR A B; do
  echo "=== duplicate-update test, variant $V"
  (cd "$ROOT/variants/$V" && "$PY" "$ROOT/scripts/test_dup_update.py" "$V" "$SMOKE/dup_update_$V.json" 2>/dev/null | tail -1)
done

for V in CUR A B; do
  echo "=== mini harness, variant $V, restart before index 4"
  mkdir -p "$SMOKE/$V"
  (cd "$ROOT/variants/$V" && B3_VARIANT=$V "$PY" -u -m replay.harness --alerts "$ALERTS" --out "$SMOKE/$V" --restart-at 4 > "$SMOKE/$V/run.log" 2>&1)
  echo "harness_${V}_rc=$?"
  if [ "$V" = B ]; then
    echo "=== non-empty database must be refused"
    mkdir -p "$SMOKE/refused_nonempty"
    (cd "$ROOT/variants/CUR" && B3_VARIANT=CUR "$PY" -u -m replay.harness --alerts "$ALERTS" --out "$SMOKE/refused_nonempty" > "$SMOKE/refused_nonempty/run.log" 2>&1)
    echo "refusal_rc=$?"
    tail -1 "$SMOKE/refused_nonempty/run.log" | cut -c1-140
  fi
  "$PY" "$ROOT/scripts/smoke_db.py" x --wipe-only
done
exit 0
