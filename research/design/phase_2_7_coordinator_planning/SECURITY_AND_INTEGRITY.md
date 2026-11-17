# Security and integrity

Planning only. **Nothing here is implemented.** Backend paths are relative to `isfcr/CYUKTI/backend/`. **[F]** fact, **[R]** recommendation, **[U]** unverified.

This report holds no credentials, manager names, tokens or alert payloads.

## Threat model

| Source | Trust | Why |
|---|---|---|
| **Alert content** | **Untrusted** | Attacker-influenced: URLs, log lines, user agents |
| `manager.name`, `id` | Low: written by Wazuh, unvalidated beyond their form | Read from the same file |
| Local files, 0600 | Trusted for accident, **not for tampering by this user** | [F] Checksums have no secret key |
| Neo4j | Trusted, on loopback, with credentials | [F] 2.6E |
| Operator tool | Trusted person | Named in every record |

**[F] Out of scope, as before:** an attacker with this account's access can forge records. Integrity here detects accidents and incoherent forgeries.

## Request keys

| Safeguard | Design | Evidence |
|---|---|---|
| Built in one place | [F] `listener/utils.py:request_key` L96-102. Writers never concatenate strings | Test scans for it |
| Manager encoded | [F] Percent-encoded, so `/`, `|`, `:` can't be forged | 2.6C tests |
| Id strict today | [F] `^[0-9]+\.[0-9]+$`, at most 64 characters | 2.6C |
| **Id under the tolerant policy** | [R] Exact `str`, 1 to 64, **refusing whitespace, control characters and `|`** | 2.6E. [F] `|` could forge a request key's operation |
| Operation | [F] `[A-Za-z0-9_.-]+`, and [R] must be in the registry | |
| Parsing | **Never parse a key.** Keep structured fields beside it. Keys are matched exactly | |
| Length | [F] The ledger's 512-character limit. [F] Observed keys are 59 to 63 characters | 2.6A |
| Same rule in each store | [F] A test compares the journal's and ledger's key rules | 2.6B |
| Normalization | **None** | [F] 2.6B tests |

## Fingerprints

| Safeguard | Design |
|---|---|
| Determinism | [F] Versioned canonical form. [F] 7,552 real W07U payloads gave identical results across two parses (2.6A) |
| Excludes | Attempt number, server time, marker fields, request key |
| Includes | Operation, `return_rows`, ordered statements, exact types |
| Version change | [F] A stored request under another version raises `FingerprintVersionError`, and isn't treated as new |
| A non-deterministic input | **`event_time` under `processing_time_fallback`:** a re-delivered alert differs, and conflicts. Safe, and noisy. [F] 0 in the archive |
| Random ids in a payload | [F] Would conflict. Hence D7, before W01, W05, W06, W16 |

**A fingerprint isn't a secret and isn't authentication.** It detects difference.

## Operation identity

[R] A frozen registry in `neo4j_write_coordinator.py`:

```python
@dataclass(frozen=True)
class RegisteredOperation:
    name: str                      # "duplicate_event.update"
    template_version: int
    unmarked_query: str
    marked_query: str
    parameter_names: frozenset     # exactly these, or refused
    minimum_nodes_created: int
    retry_requires_review: bool
```

| Rule | Test |
|---|---|
| Unknown operation is refused | Unit |
| Parameter names match exactly | Unit |
| `marked_query` holds exactly one marker `CREATE`, after the first `MATCH` | AST and text test on each template |
| No `SET` on `AppliedWrite` anywhere in the backend | Source scan |
| **Function names aren't operation names** | A rename can't change identity |
| No string interpolation of values | Parameters only. Source scan |

## Parameters and untrusted data

| Safeguard | Design | Evidence |
|---|---|---|
| Cypher injection | **Parameters only,** with fixed templates. [F] No alert value reaches query text on this path | Scan |
| Serialization | [F] The journal's exact-type round trip. A value it can't restore is refused | 2.6B |
| Malformed alerts | [F] Non-object JSON and bad lines are counted and passed over | 2.6C |
| **Oversized lines** | [R] A limit of 1 MiB in the reader. Larger goes to `rejected_lines`. [F] Sample maximum is 9,216 bytes | [U] Full archive maximum unknown |
| Non-UTF-8 | [R] Binary reading, per line | Throughput document |
| Deep nesting | [F] The journal refuses over 64 levels | 2.6B |
| Log injection | **Alert values aren't logged.** Type names and hashes only | Test with a canary value |
| Id forgery by a log source | [U] Assumed: `id` is written by Wazuh, not set by a log line. Neither this nor a custom decoder's behaviour has been tested | Review |

## Secrets and logging

| Item | Rule | Enforcement |
|---|---|---|
| Neo4j credentials | From `.env`, as today. Never logged, or passed on a command line | [F] Existing |
| `exception` on `WriteResult` | [F] Its message can quote parameters, so **it's never logged or stored** | Rule 2 in 2.5. Test |
| Stored error detail | Type, code and safe diagnostic only | [F] Stores refuse exception objects |
| Keys | Never logged raw. `key_ref` or `source_ref` | [F] Existing. Canary test |
| Manager names | Reports use a hash. Keys hold them, so stores are sensitive | |
| Journal parameters | **Plaintext, as they must be recoverable** | [F] Documented limit |
| Backlog `raw_line` | **Plaintext alert text** until finished | The same protection as Wazuh's own files |
| Reports | No raw values | Package checks |

## Files

| File | Mode | Rule |
|---|---|---|
| Journal, ledger | 0600 on creation | [F] Tested |
| Backlog | 0600 on creation | [R] The same code pattern |
| Directory | 0700, local, non-network. **Not `/mnt/c`** | [U] WAL and flock haven't been tested on the Windows mount, and SQLite's WAL needs one host's shared memory. [R] Startup refuses network filesystems where detectable |
| Location | One explicit setting. **No default path** | [F] As the stores already do. **OD-10** |
| `.sessions/` | Owner only | [F] Existing |
| Foreign or damaged files | [F] Refused and left unchanged | Tested in 2.3, 2.4 |
| Backups | The same protection | Operator |
| Wazuh files | **Read only,** using the `wazuh` group | [F] 2.6E. Never written |

## Cross-process coordination

| Concern | Design | Evidence |
|---|---|---|
| Claims across processes | [F] `BEGIN IMMEDIATE`. One claim wins | 2.4 tests |
| Next-epoch race | [F] Unique index. One record | 2.6B, 4 processes |
| Marker race | [F] Constraint. One marker, one effect | E6: 16 threads, 4 processes |
| Liveness | [F] flock per session | 2.3 tests |
| **One reader** | [R] An exclusive flock on the backlog, for reader and worker. A second listener refuses to start | New |
| Locks across stores | **None. Never across a Neo4j call** | Rule 1 in 2.5 |
| Store pairing | [R] A random `deployment_id` written to both stores, and to the backlog, at creation. **A mismatch refuses startup.** Catches pointing at the wrong pair | New |
| Fork | [F] The journal refuses use after fork | Tested |

**[U] Restored stores:** a restored pair can look consistent and be out of date. **After any restore, an operator must treat requests since the backup as uncertain.** The `deployment_id` doesn't detect this. A restore marker in the runbook is the mitigation.

## Malformed or conflicting requests

| Case | Result | Audit |
|---|---|---|
| Unknown operation, or wrong parameters | `InvalidRequest`, nothing stored | Log: type only |
| Value not storable | `PayloadNotJournalable`, nothing registered | Log |
| Same key, different payload | `RequestConflict`, nothing runs | [F] Durable conflict row |
| Same source key, different content at intake | `CONFLICT_HELD`. **Delivered to no one until a person looks** | Backlog |
| Marker fingerprint differs | `MARKER_CONFLICT` | Journal diagnostic. Alarm |

**A note on `CONFLICT_HELD`:** your contract says never suppress delivery when identity is **missing or invalid.** This case has a **valid, reused** identity, and holding is visible, not a discard. It's still a choice: **OD-11.** [F] 0 such cases in 681,259.

## Marker tampering and inconsistency

| Case | Detection | Response |
|---|---|---|
| Marker deleted | `ABSENT` with ledger `COMPLETED` | Alarm. Never re-run |
| Marker added by hand | `APPLIED` with no ledger request | Report |
| Marker edited | Fingerprint differs | `MARKER_CONFLICT` |
| Duplicates | `INCONSISTENT` | Marker mode stops |
| Constraint dropped | Startup, and each gate recovery | Refused |
| Missing `operation` or `fingerprint` | Startup sample | Refused |

**Neo4j has no per-label write protection on Community,** and [F] the application's account can write anywhere. So **markers are evidence against accidents, not against a hostile account.**

## Integrity checks

| Check | When | Cost |
|---|---|---|
| Journal `check_integrity`, ledger `check_integrity` | Startup, on demand | [F] Full scan: about 208 ms for 3,000 journal records, so [R] **not on a path per request** |
| Backlog checks | Startup | New |
| Marker and ledger comparison | On demand, on a bounded sample | Read only |
| Constraint | Startup and gate recovery | Small |
