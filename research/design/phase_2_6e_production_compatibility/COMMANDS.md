# Exact commands

Run on the WSL host from `~/capstone`, with `PY=~/capstone/soc_env/bin/python` and `PYTHONDONTWRITEBYTECODE=1`. `D` is this folder. Credentials are read by the script from `.env`, and aren't on any command line.

## Neo4j, read-only

```
cd $D && $PY -I -B neo4j_metadata_readonly.py ~/capstone/isfcr/CYUKTI/backend evidence/neo4j_metadata.json
```

The queries are at the top of that script. It runs them in READ access mode, makes one attempt, and refuses to start if any holds a write clause.

## Configuration, with values redacted

```
grep -n "NEO4J\|load_dotenv\|getenv\|environ" isfcr/CYUKTI/backend/config.py
sed -E 's/^([A-Za-z0-9_]+)=.*/\1=<redacted>/' isfcr/CYUKTI/backend/.env
grep -E '^NEO4J_URI=' isfcr/CYUKTI/backend/.env | sed -E 's#//[^@/]*@#//<redacted>@#'
grep -E '^WAZUH_ALERT_FILE=' isfcr/CYUKTI/backend/.env
env | grep -c '^NEO4J_'
```

## Wazuh version and settings

```
dpkg-query -W -f='${Package} ${Version} ${Status}\n' wazuh-manager wazuh-agent
systemctl is-active wazuh-manager
ls -ld /var/ossec /var/ossec/logs /var/ossec/logs/alerts
grep -nE '<(alerts_log|jsonout_output|logall|logall_json|hide_cluster_info|disabled|node_name|node_type|name)>' /var/ossec/etc/ossec.conf | grep -vE 'key|password'
awk '/<cluster>/,/<\/cluster>/' /var/ossec/etc/ossec.conf | grep -E 'disabled|node_type'
grep -a 'wazuh-analysisd: INFO: Started' /var/ossec/logs/ossec.log | cut -c1-19
```

## Alert files, streamed read-only

```
# Rotated files in date order, then the live file. evidence/archive_files_read.txt lists them.
cd $D && $PY -I -B analyze_alert_ids.py evidence/wazuh_archive_analysis.json <the 110 files> /var/ossec/logs/alerts/alerts.json
$PY -I -B analyze_alert_ids.py evidence/wazuh_archive_plus_sample_analysis.json <the same> ~/capstone/isfcr/CYUKTI/data/raw_logs/alerts.json
$PY -B analyze_details.py evidence/wazuh_details.json <the 110 files>
```

## Listener logs, counts only

```
cd isfcr/CYUKTI/backend
for f in logs/prerana_listener.log listener/logs/prerana_listener.log; do
  wc -l < $f; grep -c 'Duplicate raw alert ignored' $f; grep -c 'Queue full' $f; grep -c 'no usable source identity' $f; grep -c '| ERROR |' $f
done
pgrep -fc 'wazuh_listener.py'
stat -c '%n %y' logs/prerana_listener.log listener/logs/prerana_listener.log listener/offset.dat listener/wazuh_listener.py
```

## Focused tests

```
cd $D && $PY -B -m pytest tests -q -p no:cacheprovider
```

They use small synthetic files, and contact nothing.

## Public source

`https://raw.githubusercontent.com/wazuh/wazuh/v4.7.5/src/analysisd/format/to_json.c` and `.../analysisd.c`, read through a fetch agent. The second came back partly as a summary, which the report flags.
