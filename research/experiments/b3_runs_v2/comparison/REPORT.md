# B3 v2: comparative report on campaign activity policies

**Date:** 2026-11-04 · **Code:** `3afe62c` + uncommitted Steps 1–2A · **Input:** 9,015 alerts, sha256 `e958c1cc…` (2026-03-11, 16 hours)
**Data:** `comparison/summary.json`, `comparison/runs.csv`, `comparison/campaign_anomalies.csv`; regenerate with `comparison/compare_six.py`. Policy definitions: `../README.md`.

**Limits on every statement below.** This archive has no ground truth, so nothing here measures detection accuracy, and fewer or more campaigns is neither better nor worse. Replay databases have no ATT&CK corpus, so ATT&CK-dependent values (including `misp_would_publish`) are not realistic. Anomalies are reported as stored; nothing was corrected or dropped.

## 1. Runs

All six: status `COMPLETED`, exit code 0, 9,015/9,015 alerts OK, 0 alert errors, 0 tracebacks, watchdog never fired, container stopped with exit code 0, not OOM-killed, 0 containers and 0 run volumes left.

| Run | Campaigns | UNKNOWN-only | Singletons | Self-loops | Reopens | Operations | No operation | In 2+ operations | Transitions (distinct/total) | View hash |
|---|---|---|---|---|---|---|---|---|---|---|
| CUR_norestart | 116 | 63 | 64 | 81 | 13 | 29 | 63 | 2 | 9 / 41 | `70c7b12b` |
| CUR_restart_r2 | 116 | 63 | 64 | 81 | 13 | 29 | 63 | 2 | 9 / 41 | `a967fab5` |
| A_norestart | 97 | 58 | 59 | 64 | 24 | 24 | 58 | 3 | 9 / 43 | `440b5bea` |
| A_restart | 97 | 58 | 59 | 64 | 24 | 24 | 58 | 3 | 9 / 43 | `8fa6fb65` |
| B_norestart | 116 | 63 | 64 | 81 | 13 | 29 | 63 | 2 | 9 / 41 | `148c48b0` |
| B_restart | 116 | 63 | 64 | 81 | 13 | 29 | 63 | 2 | 9 / 41 | `148c48b0` |

Per-alert paths are the same in all six: 1,201 resolved duplicates, 7,552 UNKNOWN duplicates, 110 resolved new, 152 UNKNOWN new. No policy changes which alerts are duplicates.

## 2. Restart effect (same policy, restart before alert 4,500)

| Policy | Hash equal | Assignments, operations, transitions | Campaign field changes | Lifecycle decisions changed |
|---|---|---|---|---|
| CUR | no | identical | 2 `last_seen` | 1 score (C0028 REOPEN 0.8843 → 0.8883) |
| A | no | identical | 2 `last_seen`, 1 `closed_at` | none |
| B | **yes** | identical | none | none |

No policy's campaign grouping changed across the restart on this archive, with one restart at one point.

- **`last_seen` changes in CUR and A:** duplicate-buffer attribution. The buffer merges per fingerprint and the harness flushes only at the restart and at the end, so repeats are credited to the campaign first buffered (traced for the 19:20:11 alert in `CUR_restart_r2`, which belongs to another campaign). The size of the effect is largely a harness artefact; the live listener flushes every 7 s.
- **`closed_at` change in A (first alert 3394, C0025):** 06:21:25 → 06:21:05. Traces show its only later alerts are two buffered duplicates at 06:19:25, and the restarted run closed it at last pre-restart activity + 120 s. Consistent with `touch()` doing nothing while the campaign is not back in the memory cache, with the buffer not yet in Neo4j. Inferred from outputs, not observed directly.
- **CUR score change:** confined to the restart step (`CUR_norestart` reproduces v1 exactly). Exact mechanism not traced. Both scores are far above the 0.70 reopen threshold.

## 3. Variant effect

**B vs CUR:** assignments, operations and transitions identical in both restart modes. Differences: 31 campaigns' `last_seen`, and 2 REOPEN scores (C0017 0.8623 → 0.8566; C0028 0.8843 → 0.8823). Neither crosses a threshold. On this archive B changes what Neo4j records, not the grouping.

**A vs CUR:**
| Measure | CUR | A |
|---|---|---|
| Campaigns | 116 | 97 |
| Identical campaign memberships | | 85 |
| Operations | 29 | 24 |
| Largest campaign (events) | 14 | 27 |
| `T1548.003→T1078` / `T1078→T1562.001` | 30 / 2 | 31 / 3 |

Adjusted Rand index between CUR and A is 0.968, but one campaign holds most alerts (7,822 of 9,015 in CUR, mostly duplicates), so that figure overstates agreement; treat the membership count as the informative one. It compares the runs with each other, not with any truth.

**Where A's 19 fewer campaigns fall:** `Omkar→Omkar` −9, `pes1ug23cs411-VirtualBox→itself` −8, `192.168.56.106→pes1ug23cs411-VirtualBox` −1, `172.27.8.77→172.27.0.1` −1. So **17 of 19 are self-loop identities** (the `attacker := victim` fallback, a separate finding), not network attacker pairs. Whether these merges are right cannot be determined here.

## 4. Lifecycle transitions and scores

| Decision | CUR | A | B |
|---|---|---|---|
| CREATE | 116 | 97 | 116 |
| CONTINUE | 71 | 62 | 71 |
| CLOSE_EXPIRED | 96 | 74 | 96 |
| CLOSE_BELOW_THRESHOLD | 21 | 35 | 21 |
| FAST_REOPEN | 10 | 22 | 10 |
| REOPEN | 3 | 2 | 3 |
| MAINTENANCE_EXPIRE | 11 | 11 | 11 |

Counts are the same with and without restart. Under A, keeping campaigns alive replaces expiry closures with score-based ones: 22 fewer expiries, 14 more below-threshold closes, 12 more fast reopens. That is the existing split mechanism (feedback item (a)), which A exposes more often.

| Score | CUR | A | B |
|---|---|---|---|
| Scored decisions | 95 | 99 | 95 |
| Below 0.35 | 21 | 35 | 21 |
| Min / median / max | 0.317 / 0.863 / 9.52 | 0.317 / 0.862 / 9.82 | 0.317 / 0.863 / 9.52 |
| Above 1.0 | 29 | 31 | 29 |

Scores above 1 are the known `chain_similarity` mis-normalisation (audit A3), unchanged here.

## 5. Timestamps and tracked anomalies

| Measure | CUR no/restart | A no/restart | B no/restart |
|---|---|---|---|
| Campaigns with `closed_at < last_seen` | 2 / 3 | 6 / 7 | 0 / 0 |
| Expired campaigns closed before Neo4j `last_seen` + 120 s | 30 / 30 of 101 | 2 / 3 of 84 | 0 / 0 of 101 |
| Longest `last_seen − first_seen` (s) | 57,163 / 46,514 | 57,163 / 47,809 | 1,401 / 1,401 |

(Step 2 reported 26/101 and 0 `closed_at < last_seen` because that harness never flushed the buffer. These flushed figures supersede those.)

**The two known campaigns (identified by first alert):**
| Campaign | CUR_norestart | CUR_restart_r2 | A_norestart | A_restart | B both |
|---|---|---|---|---|---|
| C0002 (alert 3): closed / last | 03:29:29 / 19:20:11 | 03:29:29 / 06:10:57 | 03:31:08 / 19:20:11 | 03:31:08 / 06:10:57 | 03:29:29 / 03:27:29 |
| C0045 (alert 8535; C0036 in A): closed / last | 06:45:25 / 18:39:48 | same | 06:46:15 / 18:39:48 | same | 06:45:25 / 06:43:25 |

Restart adds a third in CUR: C0034, last 19:20:11 vs closed 06:27:30, the same 19:20:11 value moved by the buffer. A's further anomalies include gaps of 4–40 s where a duplicate arrives after the campaign closed (C0023, C0082, C0083, C0089). Full list: `campaign_anomalies.csv`.

**B shows 0 because it does not record that activity on the Campaign.** It does not mean the repeats stopped: they still occur, and remain on their AttackEvent. These are recording differences; the underlying issue, duplicates keeping a fingerprint alive past its campaign's closure, exists in every policy.

## 6. Runtime and memory

| Run | Total s | Harness s | Peak MB | In use (anon) MB | File cache MB |
|---|---|---|---|---|---|
| CUR_norestart | 152 | 99.7 | 1,029.7 | 814.6 | 184.5 |
| CUR_restart_r2 | 126 | 89.2 | 991.4 | 826.6 | 149.2 |
| A_norestart | 138 | 95.4 | 1,073.7 | 823.6 | 226.9 |
| A_restart | 137 | 99.1 | 1,073.7 | 816.5 | 240.5 |
| B_norestart | 128 | 91.2 | 1,073.7 | 837.8 | 213.4 |
| B_restart | 129 | 93.6 | 1,073.7 | 822.3 | 204.8 |

Four runs reached the 1,073.7 MB cap on `memory.peak`, which includes reclaimable file cache. Non-reclaimable use never exceeded 838 MB (about 236 MB margin), and none was OOM-killed. With one run per configuration, runtime differences are not evidence of a policy effect.

## 7. Integrity and isolation

- Each new run's `SHA256SUMS` verifies. `SHA256SUMS` file hashes: A_norestart `b1cf019a`, A_restart `841e3ec9`, B_norestart `3cf05c8c`, B_restart `0a8d781b`.
- View hashes recomputed and match. B's two `view.json` files are byte-identical (`00b256fb`).
- Earlier runs unchanged: 37 file checksums verified before and after. v1 and its preservation copy verified.
- Variants and scripts unchanged across the batch (hashes in `checks/`). Monitor recorded no violation (`checks/batch_monitor.log`).
- All runs targeted `bolt://127.0.0.1:17690`. `neo4j-soc` JVM ran continuously and was never connected to. No listener. 110 MISP publications and 110 outbound calls suppressed or blocked per run. Same 11 containers up, Wazuh active, no kernel memory failures.

## 8. Not established

- Which policy is correct, or better at detection: no ground truth.
- Behaviour with other restart points, multiple restarts, crashes (no graceful flush), or a periodic flush.
- Whether results hold with an ATT&CK corpus present, or on any other archive. Each configuration ran once, though `CUR_norestart` reproduced v1 and B reproduced itself across a restart.
