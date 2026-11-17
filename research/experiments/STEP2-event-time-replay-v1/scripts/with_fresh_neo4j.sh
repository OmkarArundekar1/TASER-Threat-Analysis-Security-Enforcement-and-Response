#!/usr/bin/env bash
# usage: with_fresh_neo4j.sh <name> <command...>
# Starts a brand-new disposable neo4j:5-community (no volume, port 17690),
# runs the command with NEO4J_* pointing at it, then removes the container.
# Never touches the neo4j-soc container or its data.
set -u
NAME="cyukti-s2-$1"; shift
PW=$(head -c 18 /dev/urandom | base64 | tr -dc A-Za-z0-9)
docker run -d --rm --name "$NAME" -p 127.0.0.1:17690:7687 -e NEO4J_AUTH="neo4j/$PW" \
  -e NEO4J_server_memory_heap_max__size=512m -e NEO4J_server_memory_pagecache_size=256m \
  neo4j:5-community >/dev/null || exit 3
for i in $(seq 1 90); do docker logs "$NAME" 2>&1 | grep -q 'Started.' && break; sleep 2; done
export NEO4J_URI=bolt://127.0.0.1:17690 NEO4J_USERNAME=neo4j NEO4J_PASSWORD="$PW"
export GNN_ENABLED=false MISP_URL=https://127.0.0.1:9 PYTHONDONTWRITEBYTECODE=1
"$@"
rc=$?
docker stop "$NAME" >/dev/null
exit $rc
