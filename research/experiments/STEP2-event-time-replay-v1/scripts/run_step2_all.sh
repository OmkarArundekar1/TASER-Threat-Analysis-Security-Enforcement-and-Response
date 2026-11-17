#!/usr/bin/env bash
set -u
SP=/mnt/c/Users/omkar/AppData/Local/Temp/claude/bb82ff1e15780ae8/d148550c-7fcb-4867-9c51-0a458b210f8a/scratchpad
PY=~/capstone/soc_env/bin/python
BACK=~/capstone/isfcr/CYUKTI/backend
ALERTS=~/capstone/isfcr/CYUKTI/data/raw_logs/alerts.json
OUT=$SP/step2_runs
mkdir -p $OUT
cd $BACK
run_after() { # name speed maint
  local args=(--alerts $ALERTS --out $OUT/$1)
  [ "$2" != none ] && args+=(--speed $2)
  [ "$3" != none ] && args+=(--maintenance-interval $3)
  echo "== $(date -u +%T) after/$1 speed=$2 maint=$3"
  bash $SP/with_fresh_neo4j.sh "$1" $PY -m replay.harness "${args[@]}" > $OUT/$1.log 2>&1; echo "rc=$?"
}
run_before() { # name speed
  echo "== $(date -u +%T) before/$1 speed=$2"
  (cd $SP && bash $SP/with_fresh_neo4j.sh "$1" $PY $SP/feed_before.py $SP/step1_backend $BACK $ALERTS $OUT/$1 $2 > $OUT/$1.log 2>&1); echo "rc=$?"
}
run_after A_after_unpaced none none
run_after B_after_unpaced_rerun none none
run_after C_after_x2000_maint005 2000 0.05
run_after D_after_x200_maint05 200 0.5
run_before E_before_unpaced none
run_before F_before_x200 200
echo "== $(date -u +%T) done"
