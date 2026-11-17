# Disposable instance: start and clean up

**Status: the instance used for this report was removed after its logs were kept.** `results/teardown.txt` records that. Nothing needs cleaning up unless you start it again.

## Start

```
cd ~/capstone/research/design/phase_2_6d_marker_validation
env/neo4j_up.sh
```

It refuses to run unless Docker is the expected Docker Desktop daemon, the image is already present, and every name and port is free. It never pulls.

## Clean up

```
env/neo4j_down.sh
```

| It removes | How it's protected |
|---|---|
| Container `cyukti-d6-marker-v1` and its anonymous volume | Refused unless labelled `cyukti.experiment=d6-marker-v1` |
| Named volume `cyukti-d6-marker-v1-data` | The same label check |
| Network `cyukti-d6-marker-v1-net` | The same label check |
| `~/cyukti_diag/d6_marker_v1/state/conn.json`, the credentials | Only this file |

It refuses to run on any daemon but the expected Docker Desktop, and it saves the server's logs to `results/` first. **That overwrites the logs from this report's run,** so copy them first if you want both.

## If the script can't run

Check the label first, then remove only these names, on Docker Desktop:

```
docker inspect cyukti-d6-marker-v1 --format '{{index .Config.Labels "cyukti.experiment"}}'
docker rm -f -v cyukti-d6-marker-v1
docker volume rm cyukti-d6-marker-v1-data
docker network rm cyukti-d6-marker-v1-net
rm -f ~/cyukti_diag/d6_marker_v1/state/conn.json
```

**Never touch anything else.** In particular, not `neo4j-soc`, or any container on Ubuntu's native `dockerd`.

## Left in place

The empty directory `~/cyukti_diag/d6_marker_v1/state/`, and this package.
