# D2: alert deduplication investigation

Phase 2.6A. **The live listener was not run, and nothing was changed.** Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[V]** verified current behaviour, **[M]** measured evidence, **[P]** proposed behaviour, **[A]** assumption, **[D]** open decision, **[R]** requires production validation, **[L]** known limitation.

## Classification

**Confirmed defect by static evidence and reproducible offline analysis.**

**Yes: this defect can lose an alert before the coordinator could protect the write.** It happens in the reader thread, before `process_alert`, and so before any ledger, journal or writer call. The coordinator would never learn the alert existed.

## The mechanism

[V] The listener's key leaves out what separates these alerts:

1. `listener/utils.py:generate_alert_id` L3-27 hashes `timestamp|rule.id|agent.id|srcip|dstip|location`. It reads `srcip` and `dstip`. Suricata alerts spell them `src_ip` and `dest_ip`, which it never reads, so both are blank.
2. It never reads the Wazuh `id`.
3. `listener/wazuh_listener.py:accept_alert` L125-148 returns `False` if that hash was accepted in the last 600 s.
4. `listener/wazuh_listener.py:AlertFileHandler.read_new_lines` L228-306 sets `last_good_offset` at L288 **before** it checks that result, logs "Duplicate raw alert ignored" with 12 hash characters, and saves the offset at L303-304. The alert won't be read again.

So two real alerts in the same millisecond, from one rule, agent and location, become one.

## The ten questions

| # | Question | Answer | Label |
|---|---|---|---|
| 1 | Exact key | `alert["alert_id"]`, read at `wazuh_listener.py` L132, tested at L142, recorded at L145 | [V] |
| 2 | `alert_id`, Wazuh `id`, or a composite? | `alert_id` only, a SHA-256 of six fields. The Wazuh `id` plays no part | [V] |
| 3 | Does the timestamp participate? | Yes, as the exact raw string. All 70 refused alerts share a timestamp with the accepted one | [V], [M] |
| 4 | 600 s window or identity rule? | **The identity rule causes this.** The window only decides when a hash is forgotten. It runs on wall-clock time at reading, so colliding alerts, which sit close together in the file, nearly always fall inside it | [V] |
| 5 | Can different Wazuh IDs be suppressed? | **Yes:** 70 refused, with 70 distinct IDs, all different from the accepted one | [M] |
| 6 | Can differing contents be discarded? | **Yes.** 0 of the 70 differ only in `id`. See below | [M] |
| 7 | Before any durable event? | **Yes.** It's in the reader thread, while `process_alert` runs later on the worker | [V] |
| 8 | Attributed and unattributed? | **Both.** It comes before `resolve_mitre`, so it can't tell them apart | [V], [M] |
| 9 | Does the cache survive a restart? | **No.** `seen_alerts` is an in-memory `OrderedDict` (L35) | [V] |
| 10 | Can archived replay reproduce it? | **Yes,** with the script below | [M] |

## Offline reproduction

**Script:** `evidence/d2_replay_accept_alert.py`. **Output:** `evidence/d2_replay_output.txt`. Hashes are in `MANIFEST.sha256`.

**Input:** `isfcr/CYUKTI/data/raw_logs/alerts.json`, sha256 `e958c1cc9211036b2e9feaf48990c36cf370d724c1c139183229cb59565b3098`.

**Method:** the script lifts the **real** `accept_alert` and `generate_alert_id` from the source by AST, so the listener module is never imported, its signal handlers aren't installed and its log isn't opened. No database or network is involved. The script reads the sample and prints counts and field names, never alert content.

[M] Results:

| Measure | Result |
|---|---|
| Alerts in | 9,015 |
| Accepted / refused, paced by alert timestamps | 8,945 / **70** |
| Accepted / refused, read as one fast backlog | 8,945 / **70** |
| Groups sharing an `alert_id` | 39; the largest holds 9 alerts |
| Groups adjacent in the file | 34 of 39 |
| Refused with no `data.srcip` | 55 |
| Refused with `data.srcip` present | 15 |

### What the refused alerts contain

| Field that differs from the accepted alert | Refused alerts |
|---|---|
| `id` | 70 |
| `rule.firedtimes` | 70 |
| `data.http.url` | 39 |
| `data.tx_id` | 38 |
| `full_log` | 27 |
| `data.flow.pkts_toclient`, `data.flow.bytes_toclient`, `data.timestamp` | 24 each |
| `data.dest_ip` | 7 |

**`rule.firedtimes` differs on all 70,** which shows Wazuh counted them as separate firings. These are real separate alerts and not replays.

### What was lost

[M] From a **model** of later routing, using the real `extract_iocs` and the same fingerprint and 90 s window, without Neo4j:

| Category | All alerts | After `accept_alert` | Lost |
|---|---|---|---|
| Attributed new event | 110 | 110 | 0 |
| Attributed duplicate | 1,201 | 1,198 | 3 |
| Unattributed new event | 152 | 151 | **1** |
| Unattributed duplicate | 7,552 | 7,486 | 66 |

- **One refused alert would have created an event of its own,** so detection, not just a count, changes.
- **Seven have a different attacker or victim** from the accepted alert.
- Refused rule levels: level 3 (53), level 5 (13), level 6 (2), level 7 (2). That's this sample only. Nothing in the mechanism limits severity.
- Loss rate: 70 of 9,015, about 0.78%.

## Limits of this evidence

- [L] **The listener wasn't run.** The real functions were, without threads, a real queue or a real clock. The stand-in queue is never full.
- [L] **One sample,** one manager, about 16 hours. The rate elsewhere is unknown. It should rise with Suricata traffic in bursts.
- [L] The routing model isn't `process_alert`. It leaves out the database lookup, and it treats the reviewed mappings as empty, which they are today (`mitre_rule_registry.py`). Its numbers for lost events are estimates.
- [L] **[A]** The sample's provenance is undocumented, though its form is consistent with a real `alerts.json`.
- [R] Whether the production listener has logged "Duplicate raw alert ignored" wasn't checked, because `logs/prerana_listener.log` is protected. **Counting those lines is the simplest production check.** Note that a re-read alert would log the same line.

## Other findings from this reading

Reported only; nothing here is changed.

1. [V] **Research replays bypass this defect.** `replay/harness.py:replay` L127-175 calls `process_alert` directly. So offline evaluations include alerts the live listener would discard, and the two paths can differ by this amount. Frozen B3 evidence isn't altered by this finding. How to describe the difference is your call.
2. [V] **A crash can re-deliver alerts.** `offset.dat` is written only after the whole read loop ends (L303-304), so a crash during a large read re-reads alerts already queued, with `seen_alerts` now empty. This corrects Phase 2.5's statement that the offset "has already moved past them": both losses and repeats are possible, depending on the moment.
3. [V] **An interrupted offset write can skip alerts.** `open(offset_file, "w")` truncates first, and an empty file means "start at the end of the log" (L217-218). This is by reading only.
4. [V] A full queue doesn't lose alerts: `Full` keeps the offset, and the alert is read again (L274-280). That's relevant to coordinator latency, because a slow worker would create backlog rather than loss, until log rotation intervenes.

## Remediation contract

[P] **A separate change, not implemented, and independent of the coordinator.** It's the one fix here that can land first, because the coordinator can't cover this.

| Item | Contract |
|---|---|
| Scope | `generate_alert_id` only. `accept_alert`, the TTL, the cache size and offset handling stay as they are |
| New key | With a valid `id` and manager (D1 rule): SHA-256 of a versioned string holding those two, and nothing else |
| Fallback | Without them: today's formula, unchanged, so behaviour on odd input stays as it is. **Its known collision remains on that path,** and it should be counted |
| Kept behaviour | The same line read twice within 600 s is still refused, because it has the same ID |
| Persistence | None needed. `alert_id` never reaches the graph, and the cache starts empty |
| Effects you'd see | About 0.78% more alerts on this sample, and more duplicates and occasional new events. Counts will shift |
| Logging | Hash prefix only, as now |

**Acceptance evidence:**

| # | Test | Pass |
|---|---|---|
| 1 | The offline replay here, on the changed function | 9,015 accepted, 0 refused |
| 2 | Feed the sample twice within the window | Second pass: all 9,015 refused |
| 3 | Alerts with no `id` or manager | Same result as today's formula |
| 4 | Two alerts with equal content and different IDs | Both accepted |
| 5 | The existing `tests/test_listener_alert_acceptance.py` | Still passes. It sets `alert_id` itself, so it doesn't pin the formula |
| 6 | No live listener run | Confirmed |

**Open decision [D]:** whether and when to make this change, because it changes live detection counts. It depends on D1's namespace and identity choice.

## If deferred

The listener keeps dropping distinct alerts, with no way to recover them except from Wazuh's own files. The coordinator wouldn't see or report it.
