# B3 v2: campaign activity policy study (prepared, no full replay run)

**Status (2026-11-04):** all six full-archive experiments completed. Report: `comparison/REPORT.md`.
**v1 (untouched):** session scratchpad `b3_runs/` and `C:\Users\omkar\cyukti_b3_preserved_20261104\` (checksums in `SHA256SUMS`; `CUR_norestart` complete, `CUR_restart` partial).

## Layout
| Path | Contents |
|---|---|
| `V1_INPUT_SHA256` | archive hash, same as v1; each run refuses a different input |
| `variants/{CUR,A,B}/` | backend copies built by `scripts/build_variants_v2.py`, with no `.env`. **About 320 MB; do not commit.** |
| `scripts/` | build, Neo4j wrapper, runner, detached launcher, smoke and diagnostic tests |
| `runs/` | full-archive runs, one directory each |
| `comparison/` | six-run comparison and report |
| `smoke/` | synthetic mini-run outputs (8 alerts, not archive data), connectivity and duplicate-update results |
| `checks/` | variant diffs, watchdog logs, runner tests, failed earlier attempts |

## Exactly what counts as campaign activity
"Yes" = this alert type extends `last_seen` there. All variants also set it on creation and on reopen.

| Alert type | CUR memory | CUR Neo4j | A memory | A Neo4j | B memory | B Neo4j |
|---|---|---|---|---|---|---|
| Resolved, new | yes | yes | yes | yes | yes | yes |
| UNKNOWN, new | no | yes | yes | yes | no | no |
| Resolved duplicate | no | yes, at buffer flush | yes | yes, at buffer flush | no | no |
| UNKNOWN duplicate | no | **no** | yes | yes | no | no |

**CUR** is current repository behaviour, including an inconsistency found during this preparation: `update_duplicate_event` reaches the Campaign through `MATCH (e)-[:MATCHES]->(t)`, and UNKNOWN events have no such edge, so an UNKNOWN duplicate never updates `Campaign.last_seen` (verified on a real Neo4j: `smoke/dup_update_CUR.json`). Earlier reports said UNKNOWN duplicates do extend it; that was wrong.

**A** = all accepted alerts. Code differences from CUR:
- `CampaignManager.touch()`, and three calls to it in `realtime_socgraph.py`;
- one `last_seen`-only update in `update_duplicate_event`, so counters and risk are unchanged.

A duplicate extends the campaign that owns its fingerprint, which may already be closed; that doesn't reopen it. Resolved duplicates reach Neo4j at buffer flush, so Neo4j can lag memory until then.

**B** = only resolved, non-duplicate alerts. Code difference from CUR: three `c.last_seen` writes removed in `neo4j_client.py` (UNKNOWN events, immediate duplicates, buffered duplicates). `AttackEvent.last_seen` is kept, because the dedup fallback needs it.

CUR differs from the repository only in `replay/harness.py`, which is identical across variants. Scoring, thresholds, config, dedup, event-time and ATT&CK files are byte-identical. See `checks/*.diff`.

## Diagnostics (study harness only)
`<run>/diagnostics.log`: `RUN_START`, `PROGRESS` every 500 alerts, `RESTART_BEGIN`, `RESTART_FLUSH_DONE`, `RESTART_END`, `REPLAY_LOOP_DONE`, `FINAL_FLUSH_DONE`, `RUN_END`. A `faulthandler` watchdog re-armed before every alert (300 s) writes all thread stacks to this file and exits 1 if stuck. Python runs with `-u`.

## Running (only when approved)
```
bash scripts/start_batch_v2.sh CUR_restart            # or several names
```
Experiments: `CUR_restart A_norestart A_restart B_norestart B_restart`. No default list; a failure stops the batch; existing run directories are refused. Each run gets a new container, with `run.log`, `diagnostics.log`, `neo4j_resources.txt`, `preflight.txt`, `exit_status`, plus `batch_status.tsv` and `batch_<stamp>.log` (`batch_exit_status=`). Refuses to start below 1.5 GB available memory.

The launcher waits for the runner's first status line, because closing the launching `wsl.exe` session immediately otherwise kills the child (`checks/runner_diag2`).

## Run log
| Experiment | Status | Exit | Notes |
|---|---|---|---|
| `CUR_restart` | stopped by operator: hung in the wrapper guard (`docker info`, `docker-ai` plugin, `powershell.exe`); intermittent, trigger unknown | none recorded | 0 alerts, no container. Kept unchanged in `runs/CUR_restart/`. |
| `CUR_restart_r2` | COMPLETED | 0 | 9,015/9,015 alerts, restart markers present, view `a967fab5`. See `runs/CUR_restart_r2/status.json`. |
| `CUR_norestart` | COMPLETED | 0 | 9,015/9,015 alerts. View hash `70c7b12b` and `view.json` bytes identical to v1 `CUR_norestart`. Comparison: `checks/CUR_norestart_vs_v1.json`. |
| `A_norestart` | COMPLETED | 0 | 9,015/9,015 alerts, view `440b5bea`. |
| `A_restart` | COMPLETED | 0 | 9,015/9,015 alerts, view `8fa6fb65`. |
| `B_norestart` | COMPLETED | 0 | 9,015/9,015 alerts, view `148c48b0`. |
| `B_restart` | COMPLETED | 0 | 9,015/9,015 alerts, view `148c48b0`. |

Infrastructure validation: `checks/infra_v3_output.txt` (50 passed, 0 failed). Scripts used for the failed attempt: `checks/scripts_at_failed_attempt/`.
