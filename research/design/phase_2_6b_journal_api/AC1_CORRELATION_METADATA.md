# AC-1: correlation metadata

Implemented. Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified by tests, **[M]** measured.

## What's stored

| Field | On | Set | Meaning |
|---|---|---|---|
| `correlation_key` | Record and every history entry | At creation, never changes | The ledger's key, exactly as given |
| `epoch` | Record and every history entry | At creation, never changes | This journal's record number for the request, from 1 |
| `ledger_attempt` | Record and every history entry | When entering `IN_FLIGHT` | The ledger attempt that last started this record |

**These aren't the same as the other counters, and tests pin that:**

| Identity or counter | Source | Scope |
|---|---|---|
| Logical request | `correlation_key`, supplied by the caller | Whole life of the request |
| Epoch | `epoch` | One journal record's life |
| Ledger attempt | `ledger_attempt`, from the ledger | Rises across the request |
| Journal intent | `record_id`, generated | One record |
| Journal-internal attempt | `attempt_count` | Restarts per record |
| Driver tries | `WriteResult.attempts`, not stored here | One helper call |

## Validation

Implemented in `neo4j_write_journal.py:_checked_correlation` L683-688, `neo4j_write_journal.py:_checked_correlation_key` L652-663 and `neo4j_write_journal.py:_checked_counter` L671-676.

| Input | Result |
|---|---|
| Key without epoch, or epoch without key | `TypeError` |
| Key not an exact `str`, subclasses included | `TypeError` |
| Key empty, over 512 characters, with surrounding whitespace, or with control characters or lone surrogates | `ValueError` |
| Epoch or attempt not an exact `int`: `bool`, `float`, `str`, subclasses | `TypeError` |
| Epoch or attempt below 1, or beyond 64 bits | `ValueError` |

[V] **The key rule is the ledger's.** A test passes 16 keys through both `_checked_correlation_key` and the ledger's `_checked_key`, and requires the same verdict.

[V] **The key isn't normalized.** NFC and NFD forms, and different cases, are separate requests. The stored and returned value is identical.

## Lookup

`neo4j_write_journal.py:NeoWriteJournal.find_by_correlation_key` L1233-1238, through `neo4j_write_journal.py:NeoWriteJournal._chain` L1019-1023.

- Returns every record for the key, oldest epoch first, or `()`. An unknown key isn't an error.
- Verifies each row's checksum, as all reads do.
- [V] Uses the index: a test reads the query plan, and requires `USING INDEX journal_records_correlation` with no scan or sort.
- [M] 0.029 ms median, at both 1,000 and 10,000 records. It grows with epochs, not with the journal.

## Uniqueness contract

**At most one record for each `(correlation_key, epoch)`.** Records with no key are never compared.

| Layer | Enforcement |
|---|---|
| Journal | `_admit_epoch`, inside the write transaction: `DuplicateCorrelationError`, which carries the existing `record_id` and changes nothing |
| Database | `CREATE UNIQUE INDEX journal_records_correlation ON journal_records(correlation_key, epoch) WHERE correlation_key IS NOT NULL`. A test proves it refuses a raw insert |

## Durability and integrity

| Protection | How |
|---|---|
| Durable | In the same fsynced transaction as the record |
| In the row checksum | `neo4j_write_journal.py:NeoWriteJournal._row_hash` L975-983 |
| In the history chain | `neo4j_write_journal.py:NeoWriteJournal._event_hash` L986-991 |
| Fixed | Trigger `journal_records_correlation_fixed` refuses any change to key or epoch |
| Rise-only | Trigger `journal_records_attempt_rises` |
| Contents checked | `neo4j_write_journal.py:NeoWriteJournal._correlation_is_sound` L1455-1483 and `neo4j_write_journal.py:NeoWriteJournal._chain_problems` L1486-1515 |

**Why content checks exist as well:** a checksum has no secret, so someone recomputing it isn't caught by it. Tests forge records with every checksum recomputed, and they're detected when incoherent: a key differing between row and history, a later epoch with no authorization, a first epoch claiming one, attempts not rising, a missing epoch, a different intent, or a chain continuing past `FOUND_COMMITTED`.

## Legacy records

[V] A record with no correlation hashes exactly as version 1 did, so old rows verify without being rewritten. A test compares the formula against the frozen previous implementation directly. It takes no attempt arguments: passing one raises `TypeError` and changes nothing.

## Exposure

[V] Keys contain a hostname and alert id, so they stay out of `repr` and errors, which use `key:<12 hex>`. Tested on records, events, errors and logs.

## Reading a correlated record

`get_record`, `list_records`, `get_history` and `find_by_correlation_key` all expose the fields, on writable and read-only instances.
