#!/usr/bin/env bash
# Phase B infrastructure validation. No archive data, no harness replay, nothing
# written to runs/. usage: test_infra_v3.sh <workspace_root>
set -u
ROOT=$1
T="$ROOT/checks/infra_v3"
mkdir -p "$T"
PY="$HOME/capstone/soc_env/bin/python"
RUNNER="$ROOT/scripts/run_b3_v2.sh"
WRAP="$ROOT/scripts/with_fresh_neo4j_v2.sh"
PRE="$HOME/capstone/research/experiments/STEP2A-dedup-deadlock-v1/scripts/dedup_engine_prefix.py"
FAILS=0

check() { # label expected actual
  if [ "$2" = "$3" ]; then echo "PASS $1 [$3]"; else echo "FAIL $1 expected=[$2] actual=[$3]"; FAILS=$((FAILS + 1)); fi
}
field() { "$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]))" "$1/status.json" "$2"; }
stub() { # dir, version-body, image-body, run-body
  mkdir -p "$1"
  cat > "$1/docker" <<EOF
#!/usr/bin/env bash
case "\$1" in
  version) $2 ;;
  image) $3 ;;
  run) $4 ;;
  *) exit 0 ;;
esac
EOF
  chmod +x "$1/docker"
}
docker_clean() {
  check "$1: no temporary containers" 0 "$(timeout 30 docker ps -aq --filter 'name=cyukti' | wc -l)"
  check "$1: volumes equal baseline" same "$(timeout 30 docker volume ls -q | sort | cmp -s - "$ROOT/checks/volumes_baseline.txt" && echo same || echo different)"
  check "$1: port 17690 closed" closed "$( (exec 3<>/dev/tcp/127.0.0.1/17690) 2>/dev/null && echo open || echo closed)"
}

echo "=== T0 classifier unit tests"
"$PY" "$ROOT/scripts/test_classify.py"; check "T0 classifier" 0 $?

echo "=== T1 dry runs, guards on the runner itself"
B3_RUNS_DIR="$T/t1" B3_DRY_RUN=0 bash "$RUNNER" CUR_norestart > /dev/null; check "T1 CUR_norestart accepted, rc" 0 $?
check "T1 status" COMPLETED "$(cat "$T/t1/CUR_norestart/status")"
check "T1 exit_status separate" 0 "$(cat "$T/t1/CUR_norestart/exit_status")"
B3_RUNS_DIR="$T/t1b" B3_DRY_RUN=7 bash "$RUNNER" A_restart B_restart > /dev/null; check "T1 failure stops batch, rc" 1 $?
check "T1 classified" HARNESS_FAILED "$(cat "$T/t1b/A_restart/status")"
check "T1 actual code kept" 7 "$(cat "$T/t1b/A_restart/exit_status")"
check "T1 second not started" no "$([ -e "$T/t1b/B_restart" ] && echo yes || echo no)"
B3_RUNS_DIR="$T/t1" B3_DRY_RUN=0 bash "$RUNNER" CUR_norestart > /dev/null 2>&1; check "T1 existing dir refused" 3 $?
B3_RUNS_DIR="$T/t1" bash "$RUNNER" NOPE > /dev/null 2>&1; check "T1 unknown refused" 2 $?
B3_DRY_RUN=0 bash "$RUNNER" CUR_norestart > /dev/null 2>&1; check "T1 test mode refused on real runs/" 2 $?
check "T1 real runs/ has no CUR_norestart" no "$([ -e "$ROOT/runs/CUR_norestart" ] && echo yes || echo no)"

echo "=== T2 exact platform guard"
echo "real platform: $(timeout 30 docker version --format '{{.Server.Platform.Name}}')"
B3_PRECHECK_ONLY=1 bash "$WRAP" guard "$T/t2_real" true > /dev/null 2>&1; check "T2 real Docker Desktop accepted" 0 $?
stub "$T/stub_native" 'echo "Docker Engine - Community"' 'exit 0' 'exit 0'
PATH="$T/stub_native:$PATH" B3_PRECHECK_ONLY=1 bash "$WRAP" guard "$T/t2_native" true > /dev/null 2>&1; check "T2 native engine refused" 20 $?
cut -c1-70 "$T/t2_native/wrapper_detail"
stub "$T/stub_hang" 'sleep 100' 'exit 0' 'exit 0'
S=$(date +%s); PATH="$T/stub_hang:$PATH" B3_DOCKER_TIMEOUT=2 bash "$WRAP" guard "$T/t2_hang" true > /dev/null 2>&1; RC=$?
check "T2 hung docker version refused" 20 $RC; echo "   returned in $(( $(date +%s) - S )) s: $(cat "$T/t2_hang/wrapper_detail")"
stub "$T/stub_noimage" 'echo "Docker Desktop 9"' 'exit 1' 'exit 0'
PATH="$T/stub_noimage:$PATH" bash "$WRAP" guard "$T/t2_noimage" true > /dev/null 2>&1; check "T2 missing image refused" 20 $?
"$PY" -c "import socket,time;s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',17690));s.listen();time.sleep(6)" &
LISTENER=$!; sleep 1
bash "$WRAP" guard "$T/t2_port" true > /dev/null 2>&1; check "T2 busy port refused" 20 $?
kill "$LISTENER" 2>/dev/null; wait "$LISTENER" 2>/dev/null

echo "=== T3 blocked docker run, classified through the runner"
stub "$T/stub_runhang" 'echo "Docker Desktop 9"' 'exit 0' 'sleep 100'
S=$(date +%s); PATH="$T/stub_runhang:$PATH" B3_DOCKER_TIMEOUT=2 B3_RUNS_DIR="$T/t3" B3_TEST_COMMAND=true bash "$RUNNER" A_norestart > /dev/null 2>&1
echo "   returned in $(( $(date +%s) - S )) s"
check "T3 status" CONTAINER_START_FAILED "$(cat "$T/t3/A_norestart/status")"
check "T3 exit code" 21 "$(cat "$T/t3/A_norestart/exit_status")"
check "T3 stage" CONTAINER_START "$(cat "$T/t3/A_norestart/stage")"

echo "=== T4 overall timeout, no container"
S=$(date +%s); B3_RUNS_DIR="$T/t4" B3_DRY_RUN=0 B3_DRY_SLEEP=300 B3_EXPERIMENT_TIMEOUT=3 bash "$RUNNER" A_norestart > /dev/null 2>&1
ELAPSED=$(( $(date +%s) - S ))
check "T4 status" EXPERIMENT_TIMEOUT "$(cat "$T/t4/A_norestart/status")"
check "T4 actual code" 124 "$(cat "$T/t4/A_norestart/exit_status")"
check "T4 bounded" yes "$([ "$ELAPSED" -le 10 ] && echo yes || echo no)"

echo "=== T5 overall timeout, blocked command on a real disposable container"
B3_RUNS_DIR="$T/t5" B3_EXPERIMENT_TIMEOUT=60 B3_TEST_COMMAND='echo blocked-start; sleep 1789' bash "$RUNNER" A_norestart > /dev/null 2>&1
check "T5 status" EXPERIMENT_TIMEOUT "$(cat "$T/t5/A_norestart/status")"
check "T5 actual code" 124 "$(cat "$T/t5/A_norestart/exit_status")"
check "T5 timeout stage recorded" HARNESS "$(field "$T/t5/A_norestart" stage_at_end)"
check "T5 log preserved" blocked-start "$(grep -m1 blocked-start "$T/t5/A_norestart/run.log")"
check "T5 process group gone" 0 "$(ps -eo cmd | grep -c '[s]leep 1789')"
echo "   cleanup: $(cat "$T/t5/A_norestart/cleanup_status")"
docker_clean T5

echo "=== T6 command ignoring TERM: KILL escalation, wrapper cleanup prevented, runner cleans up"
B3_RUNS_DIR="$T/t6" B3_EXPERIMENT_TIMEOUT=40 B3_TEST_COMMAND='trap "" TERM; while :; do sleep 1; done' bash "$RUNNER" A_norestart > /dev/null 2>&1
check "T6 status" EXPERIMENT_TIMEOUT "$(cat "$T/t6/A_norestart/status")"
check "T6 actual code" 137 "$(cat "$T/t6/A_norestart/exit_status")"
echo "   runner cleanup: $(cat "$T/t6/A_norestart/runner_cleanup" 2>/dev/null)"
docker_clean T6

echo "=== T7 connectivity from WSL on a real disposable container"
B3_RUNS_DIR="$T/t7" B3_TEST_COMMAND="$PY $ROOT/scripts/smoke_db.py \$B3_RUN_DIR/db_smoke.json" bash "$RUNNER" A_norestart > /dev/null 2>&1
check "T7 status" COMPLETED "$(cat "$T/t7/A_norestart/status")"
check "T7 exit code" 0 "$(cat "$T/t7/A_norestart/exit_status")"
"$PY" - "$T/t7/A_norestart/db_smoke.json" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("   tcp", d["tcp_from_wsl"]["connected"], "| bad password rejected", d["wrong_password_rejected"], "| auth", d["bolt_auth_ok"],
      "| version", d["server"][0]["versions"], "| nodes start", d["nodes_at_start"], "| read", d["read_back"], "| nodes end", d["nodes_after_delete"])
print("   settings", {s["name"].split(".")[-2] + "." + s["name"].split(".")[-1]: s["value"] for s in d["settings"]})
EOF
cut -c1-230 "$T/t7/A_norestart/neo4j_resources.txt" | sed 's/^/   /'
sed 's/^/   /' "$T/t7/A_norestart/container_status"
echo "   cleanup: $(cat "$T/t7/A_norestart/cleanup_status")"
docker_clean T7

echo "=== T8 watchdog classification, real pre-fix deadlock, real disposable container"
B3_RUNS_DIR="$T/t8" B3_TEST_COMMAND="$PY -u $ROOT/scripts/test_watchdog.py $PRE \$B3_RUN_DIR/diagnostics.log" bash "$RUNNER" A_norestart > /dev/null 2>&1
check "T8 status" WATCHDOG_TIMEOUT "$(cat "$T/t8/A_norestart/status")"
check "T8 actual code" 1 "$(cat "$T/t8/A_norestart/exit_status")"
check "T8 stack names the lock site" yes "$(grep -q reload_cache "$T/t8/A_norestart/diagnostics.log" && echo yes || echo no)"
docker_clean T8

echo "=== final"
check "real runs/ untouched by tests" "CUR_restart batch_20261104T065800Z.log batch_20261104T065800Z.pid batch_status.tsv" "$(ls "$ROOT/runs" | tr '\n' ' ' | sed 's/ $//')"
(cd "$ROOT/runs" && cd .. && sha256sum -c checks/failed_attempt_checksums.txt --quiet > /dev/null 2>&1); check "failed attempt unchanged" 0 $?
check "CUR variant unchanged" "$(cat "$ROOT/checks/variant_CUR_tree_hash.txt")" "$(cd "$ROOT/variants" && find CUR -type f | sort | xargs sha256sum | sha256sum | cut -c1-16)"
echo "FAILS=$FAILS"
exit "$FAILS"
