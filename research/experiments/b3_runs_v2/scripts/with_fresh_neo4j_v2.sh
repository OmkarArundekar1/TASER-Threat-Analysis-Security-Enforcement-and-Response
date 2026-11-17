#!/usr/bin/env bash
# usage: with_fresh_neo4j_v2.sh <name> <run_dir> <command...>
#
# Starts one brand-new disposable Neo4j on Docker Desktop, verifies its limits
# are really applied, runs the command with NEO4J_* pointing at it, records how
# the container ended, then removes only this container and its anonymous
# volumes. neo4j-soc runs on Ubuntu's separate native dockerd and is not
# reachable through this CLI.
#
# Every docker CLI call is bounded by B3_DOCKER_TIMEOUT (default 30 s).
# exit: 20 precheck failed | 21 container start/readiness/limits failed
#       otherwise the command's own status. Details: wrapper_detail, stage,
#       container_status, cleanup_status, neo4j_resources.txt in <run_dir>.
set -u
IMAGE=neo4j:5.26.27
PORT=17690
HEAP=384m
PAGECACHE=128m
MEMORY_BYTES=1073741824
DOCKER_TIMEOUT="${B3_DOCKER_TIMEOUT:-30}"
NAME="cyukti-b3v2-$1"
RUN_DIR=$2
shift 2
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/neo4j_resources.txt"
CREATED=0
VOLUMES=""

write_atomic() { local target="$RUN_DIR/$1" tmp="$RUN_DIR/.$1.tmp.$$"; printf '%s\n' "$2" > "$tmp" && mv -f "$tmp" "$target"; }
stage() { write_atomic stage "$1"; }
detail() { write_atomic wrapper_detail "$1"; }
dk() { timeout --kill-after=5 "$DOCKER_TIMEOUT" docker "$@"; }

cleanup() {
  local rc=$? rm_rc=skipped remaining=n/a left=0 v
  trap - EXIT
  if [ "$CREATED" = 1 ]; then
    dk rm -f -v "$NAME" >/dev/null 2>&1; rm_rc=$?
    remaining=$(dk ps -aq --filter "name=^/?${NAME}\$" 2>/dev/null | wc -l)
    for v in $VOLUMES; do dk volume inspect "$v" >/dev/null 2>&1 && left=$((left + 1)); done
  fi
  write_atomic cleanup_status "container=$NAME rm_rc=$rm_rc containers_remaining=$remaining run_volumes_remaining=$left"
  exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

# ---------------------------------------------------------------- precheck
stage PRECHECK
PLATFORM=$(dk version --format '{{.Server.Platform.Name}}' 2>/dev/null); PLATFORM_RC=$?
if [ "$PLATFORM_RC" -ne 0 ]; then
  detail "docker version failed or timed out rc=$PLATFORM_RC"
  echo "Refusing to start: docker version rc=$PLATFORM_RC" >&2
  exit 20
fi
case "$PLATFORM" in
  "Docker Desktop "*) ;;
  *) detail "wrong docker platform: '$PLATFORM'"
     echo "Refusing to start: docker platform is '$PLATFORM', expected Docker Desktop." >&2
     exit 20 ;;
esac
dk image inspect "$IMAGE" >/dev/null 2>&1; IMAGE_RC=$?
if [ "$IMAGE_RC" -ne 0 ]; then
  detail "image $IMAGE not usable rc=$IMAGE_RC (no implicit pull)"
  echo "Refusing to start: $IMAGE rc=$IMAGE_RC." >&2
  exit 20
fi
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  detail "port $PORT already in use"
  echo "Refusing to start: port $PORT already in use." >&2
  exit 20
fi
[ -n "${B3_PRECHECK_ONLY:-}" ] && { detail "precheck only: platform='$PLATFORM'"; echo "precheck ok: $PLATFORM"; exit 0; }

# ---------------------------------------------------------------- container
stage CONTAINER_START
PW=$(head -c 18 /dev/urandom | base64 | tr -dc A-Za-z0-9)
CREATED=1
dk run -d --name "$NAME" \
  --memory=1g --memory-swap=1g \
  -p "127.0.0.1:$PORT:7687" \
  -e NEO4J_AUTH="neo4j/$PW" \
  -e NEO4J_server_memory_heap_initial__size=$HEAP \
  -e NEO4J_server_memory_heap_max__size=$HEAP \
  -e NEO4J_server_memory_pagecache_size=$PAGECACHE \
  "$IMAGE" >/dev/null
RUN_RC=$?
if [ "$RUN_RC" -ne 0 ]; then
  detail "docker run failed or timed out rc=$RUN_RC"
  exit 21
fi
VOLUMES=$(dk inspect "$NAME" --format '{{range .Mounts}}{{.Name}} {{end}}' 2>/dev/null)

READY=0
for _ in $(seq 1 90); do
  if dk logs "$NAME" 2>&1 | grep -q 'Started\.'; then READY=1; break; fi
  sleep 2
done
if [ $READY -ne 1 ]; then
  dk logs "$NAME" > "$RUN_DIR/neo4j_start_failure.log" 2>&1
  detail "neo4j not ready; see neo4j_start_failure.log"
  exit 21
fi

# Read the limits back from the running container, not from this script. The
# JVM command line comes from docker top (outside the container): the exec shell
# cannot resolve the JVM's /proc entry.
stage VERIFY_LIMITS
CGROUP_MAX=$(dk exec "$NAME" cat /sys/fs/cgroup/memory.max)
CGROUP_SWAP=$(dk exec "$NAME" sh -c 'cat /sys/fs/cgroup/memory.swap.max 2>/dev/null || echo n/a')
JAVA_ARGS=$(dk top "$NAME" | grep 'org.neo4j.server')
XMS=$(echo "$JAVA_ARGS" | grep -oE -- '-Xms[0-9]+[a-zA-Z]*|-XX:InitialHeapSize=[0-9]+' | head -1)
XMX=$(echo "$JAVA_ARGS" | grep -oE -- '-Xmx[0-9]+[a-zA-Z]*|-XX:MaxHeapSize=[0-9]+' | head -1)
CONF_PAGECACHE=$(dk exec "$NAME" sh -c "grep -h '^server.memory.pagecache.size' /var/lib/neo4j/conf/neo4j.conf | tail -1")
{
  echo "container=$NAME platform=$PLATFORM"
  dk inspect "$NAME" --format 'image={{.Config.Image}} image_id={{.Image}} container_id={{.Id}} memory={{.HostConfig.Memory}} memory_swap={{.HostConfig.MemorySwap}} ports={{json .HostConfig.PortBindings}} mounts={{range .Mounts}}{{.Type}}:{{.Name}}:{{.Destination}} {{end}}'
  echo "cgroup_memory_max=$CGROUP_MAX cgroup_swap_max=$CGROUP_SWAP"
  echo "jvm $XMS $XMX"
  echo "conf $CONF_PAGECACHE"
} > "$REPORT"

APPLIED=1
[ "$CGROUP_MAX" = "$MEMORY_BYTES" ] || APPLIED=0
case "$XMS" in -Xms384m|-Xms384M|-Xms402653184|-Xms393216k|-XX:InitialHeapSize=402653184) ;; *) APPLIED=0 ;; esac
case "$XMX" in -Xmx384m|-Xmx384M|-Xmx402653184|-Xmx393216k|-XX:MaxHeapSize=402653184) ;; *) APPLIED=0 ;; esac
[ "$CONF_PAGECACHE" = "server.memory.pagecache.size=$PAGECACHE" ] || APPLIED=0
if [ $APPLIED -ne 1 ]; then
  detail "limits not applied; see neo4j_resources.txt"
  exit 21
fi

# ---------------------------------------------------------------- command
export NEO4J_URI="bolt://127.0.0.1:$PORT" NEO4J_USERNAME=neo4j NEO4J_PASSWORD="$PW"
export GNN_ENABLED=false MISP_URL=https://127.0.0.1:9 SHUFFLE_WEBHOOK= SHUFFLE_BASE_URL= SHUFFLE_API_KEY=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
stage HARNESS
"$@"
RC=$?
unset NEO4J_PASSWORD

# ---------------------------------------------------------------- how it ended
stage POST_RUN
FMT='state={{.State.Status}} oom_killed={{.State.OOMKilled}} exit_code={{.State.ExitCode}} restarts={{.RestartCount}}'
AFTER_RUN=$(dk inspect "$NAME" --format "$FMT" 2>/dev/null)
{
  echo "after_run memory_peak=$(dk exec "$NAME" cat /sys/fs/cgroup/memory.peak 2>/dev/null) memory_current=$(dk exec "$NAME" cat /sys/fs/cgroup/memory.current 2>/dev/null)"
  echo "after_run memory_stat $(dk exec "$NAME" sh -c "grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat" 2>/dev/null | tr '\n' ' ')"
  echo "after_run $AFTER_RUN"
} >> "$REPORT"
dk stop -t 20 "$NAME" >/dev/null 2>&1; STOP_RC=$?
AFTER_STOP=$(dk inspect "$NAME" --format "$FMT" 2>/dev/null)
write_atomic container_status "after_run: $AFTER_RUN
stop_rc=$STOP_RC
after_stop: $AFTER_STOP"
stage CLEANUP
exit $RC
