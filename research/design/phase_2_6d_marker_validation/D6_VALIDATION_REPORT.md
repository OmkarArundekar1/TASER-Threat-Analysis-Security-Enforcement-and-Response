# D6 validation report

Phase 2.6D: the `AppliedWrite` marker, on a disposable Neo4j. **Nothing in production changed, and no production database was contacted.**

**Labels:** **[M]** measured on the disposable instance, **[V]** verified from code, **[A]** assumption, **[N]** not tested.

## Verdict

**D6 is conditionally validated.** The core mechanism held in every test on Neo4j 5.26.27 Community. The contract **as proposed isn't sufficient,** and needs the changes under "What Phase 2.7 must enforce". It's for you to accept or reject. Phase 2.7 hasn't started.

| Finding | Evidence |
|---|---|
| A marker in the same statement commits or rolls back with the mutation | E1, E2, and a server kill under load: 0 disagreements in 2,339 writes |
| A marker found proves a lost-reply commit | E3: 20 of 20, twice |
| **The proposed marker can't detect a reused key with a different payload** | E9. A fingerprint on the marker, or the intact ledger, does detect it |
| **Marker absence isn't proof of non-commit** | A marker was absent, then appeared, even after the client died |
| **Deleting a marker allows a double apply** | E10 |
| **Without the constraint, duplicates happen** | E6 control: 10 of 10 rounds |

## Scope and environment

| Item | Value |
|---|---|
| Neo4j | **5.26.27 Community,** image `neo4j:5.26.27`, id `sha256:b8b4c30662aa…` |
| Topology | One server. **No cluster** |
| Container | `cyukti-d6-marker-v1`, 2 GiB, 2 CPUs, heap 512 MiB, page cache 256 MiB |
| Data | Its own named volume, `cyukti-d6-marker-v1-data`, plus the image's anonymous volume |
| Daemon | Docker Desktop 4.59.0, id `7879fd3e-b372-45e5-942a-07cc7ca07c63` |
| Target of every experiment | `bolt://127.0.0.1:17710` directly, and `bolt://127.0.0.1:17711` through the fault proxy |
| Credentials | Random, in a mode-700 directory outside the repository, never logged. A test checks the results hold none |
| Client | Python 3.12.3, driver 6.2.0 |
| Host | WSL2 on Windows, reaching the container through Docker Desktop's port forward |

### How isolation was established

Each check runs before any write:

1. **Socket guard:** every connection except 127.0.0.1 on those two ports is refused. Port 7687 can't be reached, and a test shows it.
2. **Container identity:** the expected daemon id, this experiment's label, port mapping and volume.
3. **Database token:** a random token stamped into a database that was **empty when stamped.** Stamping refuses a database holding anything else, and a test shows it.

### Existing Neo4j, and what I couldn't inspect

| Observation | Source |
|---|---|
| Docker Desktop has **no Neo4j container or volume:** its 14 containers are listed in `results/preflight.txt` | [M] |
| A Neo4j process and listeners on 7687 and 7474 exist in Ubuntu, under a separate native `dockerd` | [M] Process and socket tables only |

**I didn't get this environment's container name, volumes or database targets.** A probe of the native Docker socket and of `neo4j-soc` was denied by the permission system, and I didn't retry or work around it. The brief asked for those details, so **this preflight item is only partly met.** Isolation doesn't depend on it: the disposable instance is on a different daemon, port and volume, and the guard refuses that environment's ports. **No connection to that server was attempted.**

## Method

- **Mutation:** `update_duplicate_event`'s statement, with the marker inside its first `MATCH`, as `D6_APPLIED_WRITE_MARKER_DESIGN.md` proposes. `marker_lib.py` holds a copy.
- **Client outcomes:** the **real write helper,** lifted from `neo4j_client.py` by AST with this experiment's driver, so nothing else from production ran. Outcomes are classified by the code a coordinator would use.
- **Faults:** a Bolt-aware proxy acting at exact protocol messages, from `~/cyukti_diag/helper_revalidation_v1/txnlib.py`.
- **Assertions:** every check reads database state directly from the server, never only a client's return value.
- **Repeats:** all E1 to E10 ran twice, the second time on a server that had been killed and restarted: 42 of 42 rows passed in the repeat. Focused tests ran twice: 24 passed each.

## Results

### E1: atomic commit: PASS

[M] 20 of 20 commits showed marker and mutation.

| Case | Result |
|---|---|
| Unattributed event | Event updated. **Campaign untouched: D6's and D13's reading of the stopped query is now confirmed on a real server** |
| Attributed event | Event, Technique and Campaign each updated once, one marker |
| **Missing target** | `COMMITTED`, `nodes_created` 0, **no marker** |
| **Negative control: a marker as its own statement, missing target** | `COMMITTED` **with a marker and no mutation.** This is the false evidence D6 warned about, shown real |

### E2: atomic rollback: PASS

[M] 20 of 20 each, neither remaining: explicit rollback, a failing statement after the marker, a client exception before commit, and through the helper with a failing second statement (`NOT_COMMITTED`).

### E3: lost commit response: PASS

[M] The proxy forwarded COMMIT, saw the server's `SUCCESS`, withheld it, and closed the client's connection.

| | Committed, reply lost (20 of 20) | **Control: COMMIT never reached the server (20 of 20)** |
|---|---|---|
| Client sees | `AMBIGUOUS`, `IncompleteCommit` | **`AMBIGUOUS`, `IncompleteCommit`: identical** |
| Independent lookup | `APPLIED` | Neither appears over 2.5 s |
| Mutation | Present, once | Absent |
| Retry | `ConstraintValidationFailed`, count stays 1 | Not run in this case |

**So the client can't tell these apart, and the marker can.** Repeated in the second run with the same result.

### E4: failure before commit: PASS

[M] Neither marker nor mutation appeared, watched over 1.5 to 2.5 s rather than at one moment.

| Fault | Reps |
|---|---|
| Connection closed instead of RUN | 10 of 10 |
| Connection closed before PULL | 10 of 10 |
| Connection closed instead of COMMIT | 10 of 10 |
| Client process killed by SIGKILL with the transaction open | 5 of 5 |
| Server terminated the transaction, then the held COMMIT arrived | Helper `NOT_COMMITTED`, code `Transaction.Terminated`, neither present |

`SHOW TRANSACTIONS` and `TERMINATE TRANSACTIONS` both work on this Community server.

### E5: duplicate request key: PASS

[M] 20 of 20 through a separate driver and session, plus one from a separate process:

- One marker, one effect, and the marker's `applied_at` unchanged.
- First run: `COMMITTED`, `nodes_created` 1. Second: `NOT_COMMITTED`, attempts 1, `ConstraintValidationFailed`.
- **So the application can tell new from existing.**

**A necessary caution:** the same error code came from a different constraint, with the marker absent and everything rolled back. **The code alone doesn't mean "already applied". The lookup is required.**

### E6: concurrent duplicates: PASS

[M] Database state checked after each round:

| Case | Contenders | Rounds | Commits | Markers | Effects | Rounds wrong |
|---|---|---|---|---|---|---|
| Threads, one key | 16 | 30 | 30 | 30 | 30 | **0** |
| Processes, one key | 4 | 5 | 5 | 5 | 5 | **0** |

Thread race outcomes: 30 `COMMITTED` and 450 `NOT_COMMITTED` with `ConstraintValidationFailed`, each on attempt 1. Processes: 5 and 15.

| Control | Result |
|---|---|
| **The same race with no constraint** | **Duplicate markers and effects in 10 of 10 rounds.** So the constraint is what prevents them |
| Different keys, one event, concurrently | 160 of 160 committed, and the additive count equalled the commits in all 10 rounds |

[M] Those last figures show no lost increment in those runs. They **don't prove** that a read-then-`SET` additive update can never lose one under other interleavings.

### E7: different keys, identical payload: PASS

[M] 10 of 10: both apply, each with its own marker, and the count is 2.

### E8: marker lookup: PASS

[M] In each failure the marker **really existed,** so reading any as absent would have been wrong.

| Condition | Status | Evidence value |
|---|---|---|
| Marker exists | `APPLIED` | **Positive** |
| Marker doesn't exist | `ABSENT` | Only what's under "What absence proves" |
| Connection interrupted | `UNKNOWN`, `ServiceUnavailable` | None |
| Reply lost | `UNKNOWN`, `ServiceUnavailable` | None |
| Stalled past the client's deadline | `UNKNOWN`, `ClientDeadline` | None |
| Server timeout | `UNKNOWN`, `TransactionTimedOutClientConfiguration` | None |
| Query error | `UNKNOWN`, `SyntaxError` | None |
| Authentication failure | `UNKNOWN`, `Unauthorized` | None |
| Nothing listening | `UNKNOWN`, `ServiceUnavailable` | None |
| Server dead | `UNKNOWN` | None |
| More than one marker | `INCONSISTENT` | Damage: see E10 |

**This is a property of how the lookup is written,** so it must be in the coordinator as a contract. A silent stalled read has no driver timeout, so it needs a client-side deadline, as this one has.

### E9: marker and payload mismatch: FAIL for the proposed marker

[M]

| Marker | Same key, different payload | Result |
|---|---|---|
| **Proposed: `{request_key, applied_at}`** | Refused by the constraint, nothing overwritten, lookup `APPLIED` | **The caller can't tell this from a true retry** |
| **Fuller: adds immutable `operation` and `fingerprint`** | Refused, nothing overwritten, stored fingerprint differs | **Conflict detected,** and a true retry is still recognized |
| **The real ledger, on a temporary file** | `IdempotencyConflictError` before Neo4j is reached | Detected, **while the ledger is intact** |

**An additional immutable fingerprint is required for the marker to detect this by itself.** The fingerprint must be computed over the mutation without the marker, and must not hold an attempt number.

### E10: retention: FAIL if markers are deleted

[M]

| Case | Result |
|---|---|
| **Marker deleted, then a retry** | Lookup `ABSENT`, retry `COMMITTED`, **count 2: applied twice** |
| **Constraint missing** | A repeat silently applies twice, two markers, and the lookup says `INCONSISTENT`. `SHOW CONSTRAINTS` can see the absence |
| Restoring the constraint over duplicates | Refused with `ConstraintCreationFailed` until they're removed |

No automatic cleanup was built, as instructed.

### Supporting: a real server kill: PASS

[M] 6 writers, each write with its own event. `docker kill`, then restart on the same volume, then every attempted key compared with its own business effect:

| Client outcome | Marker and effect | Count |
|---|---|---|
| `COMMITTED` | Both present | 2,309 |
| `AMBIGUOUS` | Both present | **5** |
| `AMBIGUOUS` | Neither | **1** |
| `NOT_COMMITTED` | Neither | 24 |

**0 disagreements, 0 lost confirmed commits, 0 confirmed non-commits with a marker.** And the 6 ambiguous ones, which a client can't separate, split both ways, and the marker separated them.

**Limit:** `docker kill` isn't power loss. The OS page cache survives, so this says nothing about fsync on a real power failure.

## Correctness properties

| # | Property | Result | Evidence and conditions |
|---|---|---|---|
| 1 | Atomicity | **PASS** | E1, E2, kill. Only when the marker shares the mutation's `MATCH`: a separate statement fails this, shown |
| 2 | Uniqueness | **PASS, needs the constraint** | E5, E6. The control fails without it |
| 3 | Replay safety | **PASS, needs marker and constraint** | E3, E5. Fails if either is lost (E10). **Doesn't cover a re-delivered alert under a different operation's key,** which is a different request |
| 4 | Ambiguous outcome recovery | **PASS for marker found** | E3, kill. Marker absent gives no such conclusion |
| 5 | Isolation | **PASS** | E6. Real races can't force every interleaving |
| 6 | Conflict safety | **FAIL as proposed.** PASS with a fingerprint, or an intact ledger | E9 |
| 7 | Failure interpretation | **PASS** | E8, kill. As a property of this lookup's code |
| 8 | Retention safety | **FAIL if markers are deleted.** A safe deletion policy is **NOT PROVEN** | E10. Only "never delete" was shown safe |

## What absence proves

**Exactly this:** a read that succeeded found no marker, for that key, on that store, at that moment.

**[M] It isn't proof of non-commit:**

| Situation | Measured |
|---|---|
| A transaction is open, with its COMMIT in flight | `ABSENT`, then the client got `COMMITTED`, then `APPLIED`: 5 of 5 |
| **The client is dead, with its COMMIT still in flight** | `ABSENT` after death, then `APPLIED`: 3 of 3 |
| The target was missing | `COMMITTED`, with no marker |
| The marker was deleted | Applied, with no marker |
| A failed read | Not absence at all |

**[M] One bound did hold:** a COMMIT arriving after the transaction's timeout was rejected, and neither appeared: 5 of 5, with the 7 of 7 from Phase 2.2.1R. The client still sees `AMBIGUOUS`, and **your ruling that this stays `AMBIGUOUS` until separately evidenced stands.**

**Conditions for absence to count as evidence that it didn't apply,** all of them, and a person still decides:

1. The read succeeded.
2. The constraint exists.
3. The owning process is dead.
4. More than the transaction's timeout, plus a margin, has passed since dispatch. The helper's default is 8 s, and earlier phases measured the server acting up to about 1 s late.
5. `SHOW TRANSACTIONS` shows none open that could be it. **It works on Community, but the helper sets no metadata, so a request can't be matched to its transaction.** "None open at all" is the only firm reading today.
6. The same store as at dispatch: no restore or rebuild. [N] Not tested.
7. No marker was ever deleted for this key.
8. **The attempt ran under marker-writing code.** A request from before the marker has none, so absence says nothing.
9. [A] One server. A cluster needs a causally consistent read. [N] Not tested.

## Retention

**What's required:** a marker must be kept for as long as its request could be retried, re-delivered, recovered or replayed. **That includes replaying an archive into the same graph, which has no time limit.**

- **While the ledger is intact,** it refuses a second run, so a marker's loss isn't fatal for that request. But it removes the evidence for any uncertain one, and the second line of defence.
- **So:** keep everything until growth is measured, delete only requests that are terminal in the ledger and past a stated re-delivery horizon, and treat deletion as its own design and approval.
- **Never delete by age alone.**

[M] Rough storage: about 380 bytes a marker. At W07U's sample rate of 11,197 a day, that's about 4.3 MB a day, within the sample's single manager and rate. **Key length is about 80 characters here.**

## Failure cases and limitations

- **One version, one edition, one server.** Your production edition and version are still unverified.
- **No cluster, routing or replica behaviour.**
- **No backup and restore test.**
- **Power loss isn't tested.**
- **Backend read paths with markers present weren't run** (D6's test 11). [M] Phase 2.6A's scan found none that could reach a marker. That was a text scan.
- **The proxy runs on loopback,** so real network partitions aren't covered.
- **Race tests** show contention in those runs.
- **A copy of the statement is used, not the writer.** No production writer was changed.
- **The lifted helper runs with the disposable driver.** The module's other code didn't run.

## Performance observations

**[M] Preliminary, one machine, through Docker Desktop's port forward. These don't establish production throughput.** Outputs: `results/performance_output.txt`, `results/performance_warm_output.txt`. n = 500 unless stated.

| Measure | Median | p95 | Max |
|---|---|---|---|
| Mutation alone, in the helper | 6.052 ms | 9.548 ms | 30.947 ms |
| **Marker plus mutation** | **6.499 ms** | 16.267 ms | 29.552 ms |
| Lookup, marker exists | 2.493 ms | 3.453 ms | 11.624 ms |
| Lookup, marker absent | 2.428 ms | 3.179 ms | 4.599 ms |
| Duplicate refused by the constraint | 2.482 ms | 3.626 ms | 7.996 ms |
| Duplicate detected: refused, then the lookup | 5.442 ms | 6.930 ms | 15.208 ms |
| 2 clients on one key, per client (n=40) | 8.411 ms | 16.470 ms | 18.172 ms |
| 8 clients on one key, per client (n=160) | 15.097 ms | 22.566 ms | 25.656 ms |
| 16 clients on one key, per client (n=320) | 25.651 ms | 46.246 ms | 64.095 ms |

**With about 24,500 markers, warm:** marker plus mutation 7.020 ms, mutation alone 6.595 ms, lookup 2.955 ms. **Measured straight after a restart, the same operations looked about twice as slow (11.871 ms and 5.141 ms, with a 1,080 ms maximum). The warm repeat shows that was a cold server, not the marker count.** The second-run figures are the ones used in the table.

**The marker's own cost is small:** about 0.4 to 0.5 ms at the median.

### What this suggests for a coordinator, without extrapolating

- A marker isn't what would limit throughput. The durable commits around it are.
- [M] Phase 2.6B measured the journal alone at about 9.2 ms for a whole correlated life. The ledger isn't measured on this path. So a synchronous coordinator adds several times the marker's cost.
- [M] Phase 2.6C showed that one synchronous worker needs under about 2.65 ms per alert to sustain 377 a second. **A single round trip here, marked or not, already takes more than that,** so the burst figure can't be met one alert at a time through this setup, and it would queue. That queue is the listener's, and it delays rather than loses, until rotation.
- **I make no throughput claim.** Batching, or a different topology, could change this, and none of it was measured. Real alert processing also does other work.

## Network-attempt audit

**Explained.** Method: only the 16 tests the guard flagged, in an empty network namespace, with `network_audit_plugin.py` adding a stack to each attempt. **Every attempt stayed blocked,** nothing was mocked or allowed, and the backend's tests weren't changed. No broad suite ran.

[M] **27 attempts from 16 tests, all to `127.0.0.1:1`,** matching the earlier count exactly.

| Where it reaches the driver | Attempts | Entered from |
|---|---|---|
| `neo4j_client.py:1785` `get_campaign_context_data` | 24 | GNN topology similarity, via campaign correlation, threat attribution, the SOAR matcher and the investigation loop |
| `recommendation_engine.py:8` `get_recommendations` | 2 | `soar/generator.py:91` |
| `dashboard_api.py:1347` `_load_campaign_context` | 1 | `investigate_campaign` |

**What they are:**

- **127.0.0.1:1 is the guard's own deliberately closed Neo4j URI** (`tests/_production_safety.py`: `CLOSED_PORT = 1`). So the real, module-level driver tried to reach nowhere, by design of the guard.
- **All three are read paths** ([V] by reading each statement) where a test stubbed most Neo4j access and missed one optional lookup.
- **Production handles the failure** by falling back. So those tests pass, and they run the fallback rather than the graph path. That's a real, small coverage gap, and it isn't a safety problem: [V] nothing reached a network.
- **Unrelated to the disposable instance.** They don't use its port, and they ran in a namespace where it can't be reached.

**Recommended separately:** stub those three reads in the tests, or mark the attempts expected. I changed nothing, as instructed.

Evidence: `results/network_audit_summary.txt`, `results/network_audit_stacks.jsonl`, `results/network_audit_pytest.txt`.

## The final questions

1. **Does the marker and mutation commit or roll back atomically in the tested version and edition?** Yes, on 5.26.27 Community with one server, **when the marker is in the mutation's own statement and `MATCH`.** As a separate statement, it can commit without the mutation.
2. **Can a lost commit response be recovered safely with the marker?** Yes when the marker is found: it's direct evidence, and a retry is refused without changing anything. **When it isn't found, no recovery conclusion follows automatically.**
3. **Are concurrent same-key requests prevented from duplicating effects?** Yes, in every measured round, **because of the constraint.** Without it, 10 of 10 rounds duplicated.
4. **Can the system detect a reused key with a different payload?** **Not with the proposed marker.** It can with an immutable fingerprint on the marker, or through the intact ledger.
5. **What does absence prove?** Only that a successful read found none on that store at that moment. See the nine conditions above. It was disproved as proof, even after the client died.
6. **What retention is required?** Markers kept while any retry, re-delivery, recovery or replay of the request is possible. No deletion by age. Only "never delete" was shown safe.
7. **Is D6 ready as a coordinator design contract?** **Conditionally: not as written.** With the changes below, the mechanism has the evidence for one server on this version. **It needs your review and acceptance first.**
8. **What must Phase 2.7 enforce?** The next section.

## What Phase 2.7 must enforce

**Changes to the contract:**

1. **Marker shape:** `{request_key, applied_at, operation, fingerprint}`, all immutable.
2. **The fingerprint** covers the mutation without the marker, and holds no attempt number or server time.

**Rules for the statement:**

3. The marker is created **inside the mutation's first `MATCH`,** in one statement. Each writer needs its own zero-match test.
4. `COMMITTED` with `nodes_created` 0 means **no effect:** report it, and don't treat it as applied.

**Rules for outcomes:**

5. `ConstraintValidationFailed` is **never** read as "already applied" on its code alone. Read the marker, and compare its fingerprint.
6. Lookup gives four results, `APPLIED`, `ABSENT`, `UNKNOWN`, `INCONSISTENT`. **Any failure is `UNKNOWN`,** and it has a client-side deadline.
7. `AMBIGUOUS` with a marker found is evidence for the person resolving. **With none found, it stays uncertain** unless the listed conditions are shown.
8. **Absence never resolves anything automatically.**

**Rules for operation:**

9. The constraint's existence is checked at startup and treated as a hard requirement. **Marker mode refuses to run without it.**
10. No marker deletion without a separate, approved design.
11. Rolling out the marker changes query text and so fingerprints: open requests are drained first, and a request from before the marker has no marker.
12. Production's edition and version are checked first, by you or with your authorization, and this repeated if they differ.

**Still open from D13:** re-delivery across operations, and retried payload overwrite, aren't solved by a marker.

## Reproducing this

See `CLEANUP_INSTRUCTIONS.md`. In short:

```
env/neo4j_up.sh
python -B run_experiments.py
python -B -m pytest tests -q -p no:cacheprovider
python -B perf.py 500 && python -B perf_warm.py
env/neo4j_down.sh
```

Run from this folder with `~/capstone/soc_env/bin/python` and `PYTHONDONTWRITEBYTECODE=1`. Focused tests **fail at import without an instance,** rather than skip, because a skipped validation would read as a passed one.

**The instance used here has been removed.** Rerunning creates a new one, so timings will differ.
