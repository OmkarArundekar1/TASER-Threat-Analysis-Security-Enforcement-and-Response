#!/usr/bin/env bash
# usage: with_fresh_neo4j_v2.sh <name> <run_dir> <command...>
#
# Starts one brand-new disposable Neo4j on Docker Desktop, verifies its limits
# are really applied, runs the command with NEO4J_* pointing at it, then removes
# the container and its anonymous volumes. Touches nothing else: neo4j-soc runs
# on Ubuntu's separate native dockerd and is not reachable through this CLI.
#
# exit: 3 start failed | 4 wrong engine | 5 image missing | 6 port busy
#       7 not ready    | 8 limits not applied | otherwise the command's status
set -u
IMAGE=neo4j:5.26.27
PORT=17690
HEAP=384m
PAGECACHE=128m
MEMORY_BYTES=1073741824
NAME="cyukti-b3v2-$1"
RUN_DIR=$2
shift 2
mkdir -p "$RUN_DIR"
REPORT="$RUN_DIR/neo4j_resources.txt"

ENGINE=$(docker info --format '{{.Name}}' 2>/dev/null)
if [ "$ENGINE" != "docker-desktop" ]; then
  echo "Refusing to start: docker CLI reaches engine '$ENGINE', expected 'docker-desktop'." >&2
  exit 4
fi
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "Refusing to start: $IMAGE not present locally (no implicit pull)." >&2; exit 5; }
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
  echo "Refusing to start: port $PORT already in use." >&2
  exit 6
fi

cleanup() { docker rm -f -v "$NAME" >/dev/null 2>&1; }
trap cleanup EXIT
trap 'exit 130' INT TERM

PW=$(head -c 18 /dev/urandom | base64 | tr -dc A-Za-z0-9)
docker run -d --rm --name "$NAME" \
  --memory=1g --memory-swap=1g \
  -p "127.0.0.1:$PORT:7687" \
  -e NEO4J_AUTH="neo4j/$PW" \
  -e NEO4J_server_memory_heap_initial__size=$HEAP \
  -e NEO4J_server_memory_heap_max__size=$HEAP \
  -e NEO4J_server_memory_pagecache_size=$PAGECACHE \
  "$IMAGE" >/dev/null || exit 3

READY=0
for _ in $(seq 1 90); do
  if docker logs "$NAME" 2>&1 | grep -q 'Started\.'; then READY=1; break; fi
  sleep 2
done
if [ $READY -ne 1 ]; then
  docker logs "$NAME" > "$RUN_DIR/neo4j_start_failure.log" 2>&1
  echo "Neo4j did not become ready; see $RUN_DIR/neo4j_start_failure.log" >&2
  exit 7
fi

# Read the limits back from the running container, not from this script.
CGROUP_MAX=$(docker exec "$NAME" cat /sys/fs/cgroup/memory.max)
CGROUP_SWAP=$(docker exec "$NAME" sh -c 'cat /sys/fs/cgroup/memory.swap.max 2>/dev/null || echo n/a')
# Read the JVM's real command line from outside the container (the exec shell
# cannot resolve the JVM's /proc entry, and a command-line pattern inside it would
# match the probe itself).
JAVA_ARGS=$(docker top "$NAME" | grep 'org.neo4j.server')
XMS=$(echo "$JAVA_ARGS" | grep -oE -- '-Xms[0-9]+[a-zA-Z]*|-XX:InitialHeapSize=[0-9]+' | head -1)
XMX=$(echo "$JAVA_ARGS" | grep -oE -- '-Xmx[0-9]+[a-zA-Z]*|-XX:MaxHeapSize=[0-9]+' | head -1)
CONF_PAGECACHE=$(docker exec "$NAME" sh -c "grep -h '^server.memory.pagecache.size' /var/lib/neo4j/conf/neo4j.conf | tail -1")
{
  echo "container=$NAME engine=$ENGINE"
  docker inspect "$NAME" --format 'image={{.Config.Image}} image_id={{.Image}} memory={{.HostConfig.Memory}} memory_swap={{.HostConfig.MemorySwap}} ports={{json .HostConfig.PortBindings}} mounts={{range .Mounts}}{{.Type}}:{{.Destination}} {{end}}'
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
  echo "Neo4j limits not applied as required; see $REPORT" >&2
  exit 8
fi

export NEO4J_URI="bolt://127.0.0.1:$PORT" NEO4J_USERNAME=neo4j NEO4J_PASSWORD="$PW"
export GNN_ENABLED=false MISP_URL=https://127.0.0.1:9 SHUFFLE_WEBHOOK= SHUFFLE_BASE_URL= SHUFFLE_API_KEY=
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
"$@"
RC=$?
{
  echo "after_run memory_peak=$(docker exec "$NAME" cat /sys/fs/cgroup/memory.peak 2>/dev/null) memory_current=$(docker exec "$NAME" cat /sys/fs/cgroup/memory.current 2>/dev/null)"
  echo "after_run memory_stat $(docker exec "$NAME" sh -c "grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat" 2>/dev/null | tr '
' ' ')"
  docker inspect "$NAME" --format 'after_run state={{.State.Status}} oom_killed={{.State.OOMKilled}} restarts={{.RestartCount}}'
} >> "$REPORT"
exit $RC
