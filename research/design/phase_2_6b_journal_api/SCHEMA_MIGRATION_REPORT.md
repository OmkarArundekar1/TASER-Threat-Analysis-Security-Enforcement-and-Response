# Schema migration report

Version 1 to 2. Implemented in `neo4j_write_journal.py:NeoWriteJournal._migrate_if_needed` L830-863. Backend paths are relative to `isfcr/CYUKTI/backend/`. **[V]** verified by tests, **[M]** measured, **[A]** assumption.

## Exact schema change

```sql
ALTER TABLE journal_records ADD COLUMN correlation_key TEXT;
ALTER TABLE journal_records ADD COLUMN epoch INTEGER;
ALTER TABLE journal_records ADD COLUMN ledger_attempt INTEGER;
ALTER TABLE journal_history ADD COLUMN correlation_key TEXT;
ALTER TABLE journal_history ADD COLUMN epoch INTEGER;
ALTER TABLE journal_history ADD COLUMN ledger_attempt INTEGER;
CREATE UNIQUE INDEX journal_records_correlation ON journal_records(correlation_key, epoch) WHERE correlation_key IS NOT NULL;
CREATE TRIGGER journal_records_correlation_fixed BEFORE UPDATE OF correlation_key, epoch ON journal_records ...;
CREATE TRIGGER journal_records_attempt_rises BEFORE UPDATE OF ledger_attempt ON journal_records WHEN ... ;
PRAGMA user_version = 2;
```

A new file applies the version 1 statements, then these, in its initial transaction, so new and migrated files have the same schema. A test compares them.

## Behaviour on open

| File | Writable open | Read-only open |
|---|---|---|
| New or empty | Created at version 2 | Refused, as before |
| Version 2 | Opens | Opens |
| **Sound version 1** | **Migrates once, then opens** | **Refused, unchanged:** a read-only open doesn't write |
| Version 1, missing objects | `JournalCorruptionError`, unchanged | Refused |
| Version 1, unexpected columns, including half-migrated by hand | `JournalSchemaError`, unchanged | Refused |
| Version 1, SQLite's quick check fails | `JournalCorruptionError`, unchanged | Refused |
| Any other version | `JournalSchemaError`, unchanged | Refused |
| Foreign or non-SQLite file | Refused before SQLite opens it, as before | The same |

## Why old hashes aren't rewritten

[V] `ADD COLUMN` rewrites no row. And `neo4j_write_journal.py:NeoWriteJournal._row_hash` L975-983 and `neo4j_write_journal.py:NeoWriteJournal._event_hash` L986-991 hash a row or event with no correlation by the version 1 formula. Anything correlated is hashed under a tag.

So: **old rows verify as they are, and later transitions on them extend the same chain.** A test compares every version 1 column of every row and history entry before and after, checksums included, read with neither implementation.

Forging correlation onto an old row changes which formula applies, so the checksum fails. Tested.

## Transactional and restart-safe

| Property | Evidence |
|---|---|
| All or nothing | `BEGIN IMMEDIATE`, with the version change last. SQLite DDL is transactional |
| SQLite failure midway | Injected: no column, index or trigger survives. Version stays 1, and the old code still opens it |
| Non-SQLite failure midway | Injected: the same |
| **Process killed at the COMMIT** | Real SIGKILL: version stays 1, content unchanged, the old code reads it, and the next open migrates |
| Lock held too long | `JournalBusyError`, file unchanged, and the next open works |
| Racing processes | 4 real processes: all report version 2 and all records, content unchanged |
| **The same race, made deterministic** | An open reads version 1, then waits on the lock while another commits the migration. It must see that and not repeat it |
| Reopening | Migrates once, shown by the log. Schema and content are stable |

## Version-1 process still running

[V] A process that opened the file before migration keeps working. It names its columns, so its writes are valid and uncorrelated, and its liveness lock is respected. Tested with the frozen code.

## No path back

[V] **The previous code refuses a version 2 file,** with no change to it. There's no downgrade. To go back, restore a copy taken before the first open. **[A]** No journal file exists in production: nothing imports the module, and a name search of the Linux filesystem found no `*journal*.db` or `*journal*.sqlite*` file, so no migration is expected in practice.

## Damage

[V] A version 1 record damaged before migrating is still reported after it, and other records remain readable. A missing version 2 object or column is refused at open.

## Limits

- [A] **Migration doesn't verify each record's checksum.** Doing so would block opening a partly damaged file. Damage stays detectable afterwards, because nothing is rewritten.
- Power loss during migration wasn't tested.
- [M] Cost, 3,000 records and 9,000 history entries: 52.7 ms including the open's checks. The file grew by 4,096 bytes, one page.

## How old files are produced

`tests/_neo4j_journal_v1_frozen.py` is a byte-identical copy of the journal before this phase, sha256 prefix `c69834f8475dc742`. A test pins that hash. There was no git history to use, because the journal has never been committed.
