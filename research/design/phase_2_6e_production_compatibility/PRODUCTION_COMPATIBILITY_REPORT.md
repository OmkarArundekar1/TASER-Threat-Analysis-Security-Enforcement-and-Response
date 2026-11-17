# Production compatibility report

Phase 2.6E. Read-only inspection of the configured Neo4j and the deployed Wazuh. **No production state changed: no write, schema change, restart or configuration change.** Backend paths are relative to `isfcr/CYUKTI/backend/`.

**Labels:** **[M]** measured from production data or metadata, **[V]** verified from code or configuration, **[A]** assumption, **[N]** not validated.

## Decisions

| Decision | Outcome |
|---|---|
| Production Neo4j version and edition | **5.26.27 Community** |
| D6 environment compatibility | **`MATCHES_TESTED_ENVIRONMENT`,** by version, edition and single-server role. This doesn't prove all production behaviour |
| Wazuh deployed version | **4.7.5-1** |
| Wazuh ID policy | **`ACCEPT_TOLERANT_STRING_POLICY`,** with the safety rules below. Not implemented |
| Manager namespace reliability | **Supported** for this one-manager deployment. A second manager is unevidenced |
| Phase 2.7 readiness | **Ready for implementation planning.** Not for deployment |
| Phase 2.8 readiness | **Blocked** |

## Three findings that change earlier conclusions

| Finding | Earlier figure | Production archive |
|---|---|---|
| **How often the old alert key merged alerts** | 0.78%, one day's sample | **11.8% overall: 80,396 of 681,206. 96.8% on 26 September** |
| **Peak alerts in one second** | 377 (W07U) and 409 (all alerts), one day's sample | **11,566** |
| **Whether an id's clock runs backward** | Not observed | **Yes, by up to 126 s,** and still no repeated id |

**So the alert-loss defect was much larger than the sample suggested, and the fix will raise live load by more than Phase 2.6C estimated.** Any throughput target built on 377 a second needs revisiting. Details are below. Phase 2.6C's files aren't changed.

## Scope and authorization

Performed under your Phase 2.6E authorization, which supersedes my earlier standing rules against contacting this Neo4j and reading `/var/ossec`, only for these read-only checks. Exact commands: `COMMANDS.md`.

| Contact | Detail |
|---|---|
| Neo4j | `bolt://localhost:7687`, 2 runs of a fixed metadata query list, READ access mode |
| Wazuh files | Read with existing `wazuh` group membership: no privilege raised, nothing written |
| Public web | Wazuh's GitHub source at tag v4.7.5, through a fetch agent |
| Ubuntu's Docker daemon | **Not contacted,** as instructed |
| Stop conditions | None triggered |

## Repository preflight

| Check | Result |
|---|---|
| HEAD | `3afe62c4216574a906feaffe87e35ddb084db941`, branch `main`: **as expected** |
| Status | 189 entries: **as expected.** 21,455 with all files listed |
| Phase 2.6D manifest | Verifies, sha256 begins `c80a860f269b0c47` |
| Phases 2.5, 2.6A, 2.6B, 2.6C manifests | All verify |
| Discrepancies | None |

## Workstream A: Neo4j

### Target

| Item | Finding | Source |
|---|---|---|
| Configured URI | `bolt://localhost:7687` | `.env`, through `load_dotenv` at `config.py` L2-6 |
| Environment override | None: 0 `NEO4J_` variables set | Shell environment |
| Database name | Not configured by the application, so the server's default: `neo4j` | `neo4j_client.py` L74, and metadata |
| Credentials | `.env`, names `NEO4J_USERNAME`, `NEO4J_PASSWORD`. Values not recorded | |
| Local, remote or container | Local, on a port published to this host | Configuration |
| Container identity | Documented as `neo4j-soc` on Ubuntu's native `dockerd` in `research/STEP2A_DEDUP_DEADLOCK.md` L64. **Not verified directly:** that daemon wasn't contacted | |

**How the target was told apart from a disposable one:**

- It's the application's own configured target, not merely a listening port.
- The Phase 2.6D instance was removed, and it used port 17710.
- [M] Its labels are the application's graph (`AttackEvent`, `Campaign`, `Technique`, `Operation` and others), with **no `CyuktiDisposableTestDatabase` token.**
- [V] `research/STEP2A_DEDUP_DEADLOCK.md` L63 had already recorded production as 5.26.27.

**[A] "Production" here means the database the application is configured to use.** There's no other deployment record in the repository.

### Metadata

[M] Collected 2026-11-06 07:21:23 UTC. Evidence: `evidence/neo4j_metadata.json`. Every query reported a read or schema-read type. An earlier run differed only by a syntax error in my indexes query, which wrote nothing.

| Item | Value |
|---|---|
| Component | Neo4j Kernel |
| **Version** | **5.26.27** |
| **Edition** | **community** |
| Databases | `neo4j` and `system`, each one primary, online and writable, so **single server** |
| Uniqueness constraints | 10 exist, on `Attacker`, `AttackEvent`, `Campaign`, `Host`, `Technique` and others |
| Indexes | All online, including `AttackEvent.fingerprint` |
| `AppliedWrite` label | **Absent,** and no marker constraint. As required |
| `db.transaction.timeout` | `0s`: **no server-side limit** |
| `db.lock.acquisition.timeout` | `0s`: **a lock wait has no limit of its own** |
| Page cache | 512 MiB. The heap isn't set explicitly |

**A correction to Phase 2.6A,** which said production's constraints were unknown: they're now known, and `AttackEvent.event_id` is unique.

### D6 compatibility: `MATCHES_TESTED_ENVIRONMENT`

| Property | Tested | Production | Match? |
|---|---|---|---|
| Version | 5.26.27 | 5.26.27 | **Yes** |
| Edition | Community | Community | **Yes** |
| Topology | Single server | Single primary | **Yes** |
| Uniqueness constraints | Worked | 10 in use | **Yes, the feature is in use** |
| Heap and page cache | 512 MiB, 256 MiB | Heap unset, 512 MiB | **No,** not material to the mechanism's logic |
| Data | Tens of thousands of nodes at most | Not counted, because only labels and schema were read | **Unknown:** would affect cost, not logic |
| Other constraints | 2 of mine | 10 of production's | **No:** makes rule 7 below matter more |
| Host | Docker Desktop | Native `dockerd` | **No:** affects latency |

**This doesn't prove production behaviour.** No marker or constraint was tried there, as instructed. No D6 experiment **must** be repeated because of the version, because it matched. But these should run **on a disposable copy at production scale, or a rollout rehearsal, before coordinator use:**

| Experiment | Why |
|---|---|
| Constraint creation on this data, then E1 and E5 | Confirms the steps on real schema, with its 10 existing constraints |
| E6 | Lock behaviour with real indexes and volume |
| Performance | Numbers from Phase 2.6D aren't production's |
| Backend read paths with markers present | Never run in Phase 2.6D |

### D6 contract readiness

You've accepted the amended marker `{request_key, applied_at, operation, fingerprint}`. Each safeguard is **required of Phase 2.7, and none is implemented.**

| # | Safeguard | Compatible with this Neo4j? | Evidence |
|---|---|---|---|
| 1 | Unique `request_key` | **Yes** | Production has 10. Tested in E5, E6 |
| 2 | Marker in the mutation's transaction and statement | **Yes** | E1, E2 |
| 3 | Immutable operation and fingerprint | **Application rule.** Community has no property-immutability feature, so nothing in Neo4j enforces it | E9: properties written once, never updated |
| 4 | Fingerprint excludes attempt and server time | Application rule | E9 |
| 5 | Same fingerprint means already applied | Yes | E9 |
| 6 | Different fingerprint means conflict | Yes | E9 |
| 7 | `ConstraintValidationFailed` alone isn't success | **Essential here,** since 10 other constraints can raise the same code | E5 |
| 8 | Four lookup results | Yes | E8 |
| 9 | Any failure is `UNKNOWN` | Yes | E8 |
| 10 | Absence never resolves automatically | Application rule | Phase 2.6D |
| 11 | No marker deletion | Application rule | E10 |
| 12 | Refuse if the constraint is missing | **Yes:** `SHOW CONSTRAINTS` works here | This phase and E10 |
| 13 | Drain or reconcile open requests | Application rule | None open: nothing uses the coordinator yet |

**One thing production's settings make important:** with `db.transaction.timeout` and `db.lock.acquisition.timeout` at `0s`, **only a per-transaction timeout limits a stuck write.** The write helper sets one, at 8 s. The absence conditions in Phase 2.6D rely on it. [V] The existing writers use `session.run` with none, so their transactions have no limit, and one of them could hold locks a marked write waits on. Phase 2.7 must keep marked writes on the helper, and consider that interaction.

## Workstream B: Wazuh

### Version: 4.7.5-1

[M] `dpkg-query`: package `wazuh-manager`, version `4.7.5-1`, status `install ok installed`. `systemctl`: `active`. `/var/ossec/VERSION` doesn't exist.

**Phase 2.6A read the v4.9.0 source.** I've now read v4.7.5's `src/analysisd/format/to_json.c`, and the `id` line is the same:

```c
snprintf(alert_id, 22, "%ld.%ld", (long int)lf->time.tv_sec, __crt_ftell)
```

| Point | Finding |
|---|---|
| `id` written | Only if `lf->time.tv_sec` is non-zero. So **an alert could lack one** |
| `manager.name` | From `gethostname`, with `localhost` as its fallback. Written always |
| Offset source | `alerts.log` when `alerts_log` is on. [V] It's on here, with `jsonout_output`, and the cluster is disabled |
| A length limit | **The buffer limits an id to 21 characters.** Offsets past 10 digits, about 10 GB in a day's log, would be silently truncated, and truncated ids could collide. [M] Longest seen: 20 characters |
| Writer threads | A fetch summary, **not verbatim text,** says a mutex guards the offset. Unconfirmed, so Phase 2.6A's concern stands as a question, answered only by measurement: see below |

### What was read

[M] 110 rotated files, 9 March to 6 November 2026, plus the live `alerts.json`. Evidence: `evidence/wazuh_archive_analysis.json`, `evidence/archive_files_read.txt`.

| Coverage | Finding |
|---|---|
| Alerts | **681,259** |
| Months | 9, from 182 alerts in October to 497,189 in April |
| Rotations | 110 files. Each starts at an offset of 0 |
| Analysis daemon starts | 11, in the current manager log, 13 October to 6 November. Older restarts aren't in a readable log |
| Agents, rules, decoders | 2, 94, 24 |
| Locations | 21 |

### Identity fields

| Check, of 681,259 | Result |
|---|---|
| `id` a JSON string | **681,259** |
| Missing, null, empty, numeric, other form | **0** |
| Matches `seconds.offset` | **681,259** |
| Length | 12 to 20 characters |
| Seconds part equals the alert's timestamp second | **681,259** |
| Timestamp form `YYYY-MM-DDTHH:MM:SS.mmm+0000` | **681,259** |
| `manager.name` a non-empty string | **681,259** |
| Distinct manager names | **1,** in all 9 months. Shown here as hash `e9df13e40857` |
| Names needing percent-encoding | 0 |
| `cluster` block | None |

**Representative ids, one per month's first alert:** `1773062956.0`, `1775049488.0`, `1779890064.0`, `1780640430.0`, `1782876291.0`, `1785660428.0`, `1789200345.0`, `1791867179.0`, `1793725655.0`.

### Uniqueness

| Check | Result |
|---|---|
| Unique manager and id | 681,206 |
| Repeats with an **identical line** | 53: all today's rotated-looking file and the live file, which hold the same alerts |
| **Same id with different content** | **0** |
| Same id under another manager | **0:** there's only one |
| Offset repeated in file order | **0** |
| Offset fell in file order | **0** |
| **Through the busiest second, 11,566 alerts** | No repeat: the measurement answers the shared-offset concern for this data |
| **Id seconds fell** | **15 times, in 6 files, by up to 126 s** |

**The clock result matters.** A clock did step backward here, or events were written out of order, and **the id stayed unique only because the offset kept rising.** Within one file, that protects it. **A step back across a rotation, with the offset reset to 0, could in principle reuse a pair.** It didn't happen in this data.

**Stability:** [M] The 53 ids are unchanged between the two files holding them. [M] **The Phase 2.6A sample is byte-identical to this manager's archive of 11 March: 9,015 of 9,015 lines.** So its provenance, an open question since 2.6A, is settled, and its ids are unchanged after about eight months.

**This is strong, and it isn't proof of uniqueness.** One manager, and restarts only partly visible. No restore or reinstall is known in this period.

### Bad lines

[M] 5 of 681,264 lines: 3 complete lines and 2 unterminated last lines. **None starts with `{`,** so these are fragments, consistent with interrupted writes by Wazuh. 0 are invalid UTF-8.

So [M] **gap 4 hasn't occurred in this archive,** and the reader change in Phase 2.6C handles the rest. An alert torn like this is lost at its source.

### The old key's real effect

[M] Evidence: `evidence/wazuh_details.json`.

| Measure | Value |
|---|---|
| Alerts the old key would merge, of 681,206 | **80,396, 11.8%** |
| Of those, identical apart from `id` and `firedtimes` | **92** |
| Using the `src_ip` spelling | 58,422 |
| With no network field at all | 19,821 |
| Worst day: 26 September | **29,871 of 30,862, 96.8%,** nearly all from the `json` decoder |

**[M] The listener really did this.** Its logs hold 313 and 34 `Duplicate raw alert ignored` lines, in the periods it ran. Those periods are short beside the archive, and a line can't show whether a repeat was real.

**Limit:** the 11.8% counts same-key alerts per file. It's what the old key could merge if the listener had read them all, so it's an upper figure for what was actually lost.

### Peak rate

[M] **The busiest second holds 11,566 alerts,** and 2,087 seconds hold 100 or more. The figure counts alerts by the event second in their ids, so it's a rate of events, not necessarily of arrival.

| Consequence | Detail |
|---|---|
| The 377 figure | Came from one day. **It isn't the production peak** |
| The listener's own limit | At a full cache it managed about 1,360 to 1,780 alerts a second in Phase 2.6C. **A burst like this would backlog in the file** |
| The duplicate cache | 10,000 entries cover **under a second** at that rate, not the intended 600 s |

### Identity contract: `ACCEPT_TOLERANT_STRING_POLICY`

**Every observed id satisfies both the current strict pattern and the proposed tolerant policy,** so the data doesn't force either.

**Why tolerant is the better rule:** the id's form isn't a documented guarantee. With the strict pattern, a Wazuh upgrade changing it would turn off suppression for every alert. Nothing would be lost, and it'd be logged, but a tolerant policy keeps working.

**It's only safe with these limits,** and `analyze_alert_ids.py:tolerant_policy_accepts` and its tests encode them for evaluation:

| Rule | Why |
|---|---|
| Exact `str`, 1 to 64 characters | The buffer gives 21. The margin allows growth, while bounding keys |
| Refuse whitespace and control characters | Accidental damage. The ledger refuses control characters anyway |
| **Refuse `|`** | **It separates the request key, so `1.2|x` could forge an operation** |
| Never parse, normalize or rebuild | Your contract |
| Missing or refused: deliver, count, log, no suppression | Your contract, and Phase 2.6C's behaviour |

**Delimiter safety:**

| Separator | Risk | Status |
|---|---|---|
| `/` between manager and id | Manager is percent-encoded, so the **first** `/` always splits. An id holding `/` is safe | Safe |
| `:` after `wazuh` | Fixed prefix, first occurrence | Safe |
| `|` in the manager | Encoded | Safe |
| **`|` in the id** | **Could forge a request key** | **Must be refused** |

**Not implemented.** [V] The listener still uses the strict pattern at `listener/utils.py` L31. Changing it is a small, separate edit for your authorization, and nothing observed needs it yet.

### Manager namespace: supported

[M] 681,259 of 681,259, one stable value over nine months.

**Limits:** it's the hostname, so a rename or rebuild changes it. A rebuilt manager with the same hostname and a stepped-back clock could reuse a pair. **A second manager is unevidenced,** and two with the same hostname would merge.

## Workstream C: Listener behaviour

| Question | Answer | Source |
|---|---|---|
| Duplicate cache scope | One process's memory | `listener/wazuh_listener.py` L35-36 |
| Expiry | 600 s by wall clock, and at most 10,000 entries, oldest first | `listener/wazuh_listener.py` L33-34, L149-161 |
| Manager absent | No identity: delivered, counted, no suppression | `listener/utils.py` L69-85 |
| Id absent, empty or malformed | The same | `listener/utils.py` L60-68 |
| Does missing identity affect delivery? | **No** | `listener/wazuh_listener.py` L128-162 |
| **Are identity conflicts observable?** | **No.** Same key with different content is refused, logged only as a duplicate with a hash. Content isn't compared | Same. [M] 0 such cases in the archive |
| Restart redelivery | **Expected:** re-read alerts are delivered again, because the cache is memory | Phase 2.6C tests |
| **Validated in production?** | **No.** See below | |

**The fixed listener has never run live.** [M] No listener process is running. Its log's last entry is 4 November, and the fix is dated 6 November. The log holds no `no usable source identity` line. **So production alert-loss remediation isn't demonstrated. The evidence is offline replay, now backed by the real archive's field formats.**

### Known gaps, all still open

| # | Gap | What this phase adds |
|---|---|---|
| 1 | Process crash with up to 1,000 alerts in the memory queue | Bursts here would fill it quickly |
| 2 | Rotation to a file already larger than the old offset | [M] 110 rotations in nine months, and busy days reach about 380,000 alerts |
| 3 | Alerts appended immediately before rotation | The same |
| 4 | Invalid UTF-8 lines | [M] 0 in this archive |
| 5 | Unusable offset file | Unchanged |

Nothing was implemented for these. **One more to add:** conflict observability, above.

## Network-attempt audit

The Phase 2.6D finding stands, unchanged: 27 blocked attempts to `127.0.0.1:1`, 24 from `get_campaign_context_data`, 2 from `get_recommendations`, 1 from `_load_campaign_context`, all read-path fallbacks. **No test was suppressed or changed. No backend test ran in this phase. No new attempt arose:** this phase's only connections are those listed under "Scope".

## Outstanding production validation

| # | Gap | Why it matters |
|---|---|---|
| 1 | Fixed listener never run live | Offline only |
| 2 | Real `process_alert` time, with the fix's extra load | **The fix adds up to 11.8% more alerts, and bursts reach 11,566 a second** |
| 3 | Constraint creation and markers at production scale | Not tried, by instruction |
| 4 | Id reuse after a restore, or a clock step across rotation | Not seen. Not excluded |
| 5 | Restarts before 13 October | Not in a readable log |
| 6 | A second manager | None exists |
| 7 | The id's 10 GB limit | Not approached: the longest id has a 9-digit offset |
| 8 | Writer-thread locking in Wazuh | Source unconfirmed, measured as no repeats |
| 9 | Older listener logs | Counts only. ERROR lines weren't examined |

## What Phase 2.7 planning has to absorb

1. **The 13 safeguards above.**
2. **Marked writes stay on the helper,** because production has no server-side transaction limit.
3. **A real throughput target.** 377 a second isn't the peak. Whether a coordinator must handle 11,566 a second, or a backlog, is **a decision for you,** with the queue and rotation gaps in mind.
4. **The 2.6B and 2.6C open questions:** a confirmed non-commit reusing its record, `authorized_by` and `reason` on later epochs, the attempt arriving at `mark_in_flight`, and the tolerant id policy's implementation.

**Phase 2.7 hasn't started. Phase 2.8 remains blocked.**

## Final decision matrix

| Decision | Required outcome |
|---|---|
| Production Neo4j version/edition | **5.26.27 Community** |
| D6 environment compatibility | **`MATCHES_TESTED_ENVIRONMENT`** |
| Wazuh deployed version | **4.7.5-1** |
| Wazuh ID policy | **`ACCEPT_TOLERANT_STRING_POLICY`** |
| Manager namespace reliability | **Supported** |
| Phase 2.7 readiness | **Ready for implementation planning** |
| Phase 2.8 readiness | **Blocked** |
