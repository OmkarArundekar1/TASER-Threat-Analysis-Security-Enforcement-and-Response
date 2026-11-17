"""Mutation check for the Phase 2.6B journal code. Each mutant is one deliberate bug, applied to a COPY in a temporary overlay,
so the working tree is never touched. The overlay has real copies of the journal and tests/, because a child process
finds its backend from its own path, and symlinks for the rest. A mutant is caught if the focused tests fail.
A no-op control must pass, or the harness itself is broken.
usage (from isfcr/CYUKTI/backend, inside the empty network namespace): python -B mutation_check.py"""
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile

BACKEND = os.getcwd()
SOURCE = open("neo4j_write_journal.py", encoding="utf-8").read()
TESTS = ["tests/test_neo4j_write_journal_correlation.py", "tests/test_neo4j_write_journal_migration.py", "tests/test_neo4j_write_journal_lifecycle.py"]
A = "_admit_epoch"
MUTANTS = [
    ("control: no change", None, None),
    # hashes
    ("row hash ignores correlation", "        correlation = tuple(values[name] for name in _CORRELATION_COLUMNS)\n        if correlation == (None, None, None):\n            return _sha256(*legacy)", "        correlation = (None, None, None)\n        if correlation == (None, None, None):\n            return _sha256(*legacy)"),
    ("event hash ignores correlation", "        correlation = tuple(values[name] for name in _CORRELATION_COLUMNS)\n        if correlation == (None, None, None):\n            return _sha256(previous, *legacy)", "        correlation = (None, None, None)\n        if correlation == (None, None, None):\n            return _sha256(previous, *legacy)"),
    ("legacy rows hashed under the tag", "        if correlation == (None, None, None):\n            return _sha256(*legacy)\n", ""),
    ("legacy events hashed under the tag", "        if correlation == (None, None, None):\n            return _sha256(previous, *legacy)\n", ""),
    # AC-1 validation
    ("bool accepted as a counter", "    if type(value) is not int:\n        raise TypeError(f\"{name} must be an int", "    if not isinstance(value, int):\n        raise TypeError(f\"{name} must be an int"),
    ("zero accepted as a counter", "if not 1 <= value <= _MAX_COUNTER:", "if not 0 <= value <= _MAX_COUNTER:"),
    ("no upper bound on a counter", "if not 1 <= value <= _MAX_COUNTER:", "if not 1 <= value:"),
    ("partial correlation accepted", "    if correlation_key is None or epoch is None:\n        raise TypeError(\"correlation_key and epoch must be given together\")\n", "    if correlation_key is None or epoch is None:\n        return None, None\n"),
    ("key normalized", "        raise ValueError(\"the correlation key must not contain control characters or lone surrogates\")\n    return key", "        raise ValueError(\"the correlation key must not contain control characters or lone surrogates\")\n    return unicodedata.normalize(\"NFC\", key)"),
    ("key lowered", "        raise ValueError(\"the correlation key must not contain control characters or lone surrogates\")\n    return key", "        raise ValueError(\"the correlation key must not contain control characters or lone surrogates\")\n    return key.lower()"),
    ("key limit off by one", "len(key) > _MAX_CORRELATION_KEY_LENGTH", "len(key) > _MAX_CORRELATION_KEY_LENGTH + 1"),
    ("whitespace key accepted", "    if key != key.strip():\n        raise ValueError(\"the correlation key must not have surrounding whitespace\")\n", ""),
    ("control characters accepted", "    if any(unicodedata.category(ch) in (\"Cc\", \"Cs\") for ch in key):\n        raise ValueError(\"the correlation key must not contain", "    if False:\n        raise ValueError(\"the correlation key must not contain"),
    ("str subclass accepted as key", "    if type(key) is not str:\n        raise TypeError(\"the correlation key must be a string\")", "    if not isinstance(key, str):\n        raise TypeError(\"the correlation key must be a string\")"),
    ("unique index not unique", "CREATE UNIQUE INDEX journal_records_correlation", "CREATE INDEX journal_records_correlation"),
    ("correlation trigger removed", "BEGIN SELECT RAISE(ABORT, 'a recorded correlation cannot change'); END", "BEGIN SELECT 1; END"),
    ("attempt trigger removed", "BEGIN SELECT RAISE(ABORT, 'a ledger attempt can only rise'); END", "BEGIN SELECT 1; END"),
    ("key shown in repr", "correlation_key: Optional[str] = field(repr=False)\n    epoch: Optional[int]\n    ledger_attempt: Optional[int]\n    _intent_json", "correlation_key: Optional[str]\n    epoch: Optional[int]\n    ledger_attempt: Optional[int]\n    _intent_json"),
    ("key shown in an event's repr", "    correlation_key: Optional[str] = field(repr=False)\n    epoch: Optional[int]\n    ledger_attempt: Optional[int]\n\n\n@dataclass(frozen=True)\nclass PreflightResult", "    correlation_key: Optional[str]\n    epoch: Optional[int]\n    ledger_attempt: Optional[int]\n\n\n@dataclass(frozen=True)\nclass PreflightResult"),
    ("key quoted in errors", "    return \"key:\" + hashlib.sha256(key.encode(\"utf-8\")).hexdigest()[:12]", "    return key"),
    ("event doesn't carry the key", "\"correlation_key\": key, \"epoch\": values[\"epoch\"], \"ledger_attempt\": updated[\"ledger_attempt\"],", "\"correlation_key\": None, \"epoch\": values[\"epoch\"], \"ledger_attempt\": updated[\"ledger_attempt\"],"),
    ("event doesn't carry the attempt", "\"correlation_key\": key, \"epoch\": values[\"epoch\"], \"ledger_attempt\": updated[\"ledger_attempt\"],", "\"correlation_key\": key, \"epoch\": values[\"epoch\"], \"ledger_attempt\": None,"),
    ("attempt not stored on start", "                updated[\"ledger_attempt\"] = ledger_attempt\n", ""),
    # epoch chain
    ("create skips the epoch rules", "            if key is not None:\n                self._admit_epoch(conn, key, epoch, operation, values[\"intent_sha256\"])\n", ""),
    ("duplicate not named", "                raise DuplicateCorrelationError(f\"{_key_ref(key)} already has a record for epoch {epoch}\", values[\"record_id\"])", "                pass"),
    ("skipped epoch allowed", "        if epoch != highest + 1:", "        if False:"),
    ("previous needn't be resolved", "        if previous[\"state\"] != JournalState.RESOLVED.value:", "        if False:"),
    ("any resolution allows a retry", "        if previous[\"resolution\"] != Resolution.FOUND_NOT_COMMITTED.value:", "        if False:"),
    ("abandoned allows a retry", "        if previous[\"resolution\"] != Resolution.FOUND_NOT_COMMITTED.value:", "        if previous[\"resolution\"] == Resolution.FOUND_COMMITTED.value:"),
    ("intent may change", "        if previous[\"operation\"] != operation or previous[\"intent_sha256\"] != intent_sha256:", "        if previous[\"operation\"] != operation:"),
    ("operation may change", "        if previous[\"operation\"] != operation or previous[\"intent_sha256\"] != intent_sha256:", "        if previous[\"intent_sha256\"] != intent_sha256:"),
    ("no authorization needed", "    return _required_text(authorized_by, \"authorized_by\"), _required_text(reason, \"reason\")", "    return authorized_by, reason"),
    ("authorization allowed on the first epoch", "    if epoch is None or epoch == 1:\n        if authorized_by is not None or reason is not None:\n            raise TypeError(\"only an epoch after the first takes authorized_by and reason\")\n        return None, None\n", "    if epoch is None or epoch == 1:\n        return None, None\n"),
    ("authorizer not recorded", "\"cause\": \"CREATED\", \"at\": now, \"actor\": actor, \"reason\": why,", "\"cause\": \"CREATED\", \"at\": now, \"actor\": None, \"reason\": None,"),
    # AC-3
    ("attempt needn't rise", "        if seen and attempt <= max(seen):", "        if False:"),
    ("equal attempt allowed", "        if seen and attempt <= max(seen):", "        if seen and attempt < max(seen):"),
    ("attempts not compared across epochs", "seen = [values[\"ledger_attempt\"] for values in self._chain(conn, key) if", "seen = [values[\"ledger_attempt\"] for values in self._chain(conn, key)[-1:] if"),
    ("start doesn't check the attempt", "            if key is not None and starting:\n                self._admit_attempt(conn, key, ledger_attempt)\n", ""),
    ("start needn't give an attempt", "            elif starting and ledger_attempt is None:\n                raise TypeError(\"a correlated record must be given ledger_attempt\")\n", "            elif starting and ledger_attempt is None:\n                ledger_attempt = 99\n"),
    ("outcome needn't give an attempt", "                if expected_ledger_attempt is None:\n                    raise TypeError(\"a correlated record must be given expected_ledger_attempt\")\n", "                if expected_ledger_attempt is None:\n                    expected_ledger_attempt = values[\"ledger_attempt\"]\n"),
    ("stale outcome accepted", "                if expected_ledger_attempt != values[\"ledger_attempt\"]:", "                if False:"),
    ("outcome from PENDING reaches the state table", "                if values[\"ledger_attempt\"] is None:\n                    raise InvalidTransitionError(f\"record {record_id}: no attempt has started, so there's no outcome to report\")\n                if expected_ledger_attempt != values[\"ledger_attempt\"]:", "                if values[\"ledger_attempt\"] is not None and expected_ledger_attempt != values[\"ledger_attempt\"]:"),
    ("uncorrelated accepts attempts", "                if ledger_attempt is not None or expected_ledger_attempt is not None:\n                    raise TypeError(\"a record with no correlation takes no ledger attempt\")\n", "                ledger_attempt = expected_ledger_attempt = None\n"),
    ("recovery isn't fenced to what it read", "                                     recovery_json=json.dumps(metadata, sort_keys=True),\n                                     expected_ledger_attempt=values[\"ledger_attempt\"])", "                                     recovery_json=json.dumps(metadata, sort_keys=True),\n                                     expected_ledger_attempt=99 if values[\"ledger_attempt\"] else None)"),
    ("recovery metadata lacks the attempt", "                    metadata[\"ledger_attempt\"] = values[\"ledger_attempt\"]\n", "                    pass\n"),
    ("mark_in_flight drops its attempt check of type", "                                ledger_attempt=_optional_counter(ledger_attempt, \"ledger_attempt\"))\n\n    def mark_committed", "                                ledger_attempt=ledger_attempt)\n\n    def mark_committed"),
    ("outcome attempt type not checked", "        return self._transition(record_id, JournalState.COMMITTED, _Cause.OUTCOME, last_outcome=\"COMMITTED\",\n                                expected_ledger_attempt=_optional_counter(expected_ledger_attempt, \"expected_ledger_attempt\"))", "        return self._transition(record_id, JournalState.COMMITTED, _Cause.OUTCOME, last_outcome=\"COMMITTED\",\n                                expected_ledger_attempt=expected_ledger_attempt)"),
    # AC-2
    ("preflight skips the epoch rules", "                self._admit_epoch(conn, key, checked_epoch, operation, sha)\n", ""),
    ("preflight skips the attempt", "                if ledger_attempt is not None:\n                    self._admit_attempt(conn, key, ledger_attempt)\n", ""),
    ("preflight takes the write lock", "conn.execute(\"BEGIN DEFERRED\")", "conn.execute(\"BEGIN IMMEDIATE\")"),
    ("preflight leaves its transaction open", "            try:\n                yield conn\n            finally:\n                self._rollback(conn)\n", "            yield conn\n"),
    ("preflight consumes an iterator", "        if hasattr(statements, \"__next__\"):\n            return refused(PreflightCode.INVALID_ARGUMENT, \"statements must be a sequence, not a one-shot iterator\")\n", ""),
    ("preflight ignores read-only", "        if self._read_only:\n            return refused(PreflightCode.READ_ONLY, \"this journal instance is read-only\", sha)\n", ""),
    ("preflight skips the payload", "            text, _ = _serialize_intent(statements, return_rows)\n        except UnsupportedParameterError as error:\n            return refused(PreflightCode.UNSUPPORTED_PARAMETER", "            text = \"{}\"\n        except UnsupportedParameterError as error:\n            return refused(PreflightCode.UNSUPPORTED_PARAMETER"),
    ("preflight names no existing record", "return refused(PreflightCode.DUPLICATE_CORRELATION, str(error), sha, error.record_id)", "return refused(PreflightCode.DUPLICATE_CORRELATION, str(error), sha)"),
    ("preflight raises on damage", "        except JournalCorruptionError as error:\n            return refused(PreflightCode.JOURNAL_DAMAGED, str(error), sha)\n", ""),
    ("preflight ignores authorization", "        try:\n            _checked_authorization(checked_epoch, authorized_by, reason)\n        except (TypeError, ValueError) as error:\n            return refused(PreflightCode.AUTHORIZATION_REQUIRED, str(error))\n", ""),
    ("lookup verifies no checksums", "        return [self._verified(row) for row in rows]", "        return [dict(zip(_RECORD_COLUMNS, row)) for row in rows]"),
    ("lookup not by exact key", "WHERE correlation_key = ? ORDER BY epoch\", (key,))", "WHERE correlation_key LIKE ? ORDER BY epoch\", (key,))"),
    # migration
    ("migration skipped", "                else:\n                    self._migrate_if_needed()\n", ""),
    ("read-only migrates", "                self._conn = sqlite3.connect(f\"file:{quote(self._path)}?mode=ro\", uri=True, timeout=busy_timeout,\n                                             isolation_level=None, check_same_thread=False)\n", "                self._conn = sqlite3.connect(self._path, timeout=busy_timeout, isolation_level=None, check_same_thread=False)\n                self._migrate_if_needed()\n"),
    ("version not recorded", "                conn.execute(f\"PRAGMA user_version = {SCHEMA_VERSION}\")\n                logger.info", "                logger.info"),
    ("no recheck under the lock", "            if conn.execute(\"PRAGMA user_version\").fetchone()[0] == _PREVIOUS_SCHEMA_VERSION:\n                for statement in _SCHEMA_V2:", "            if True:\n                for statement in _SCHEMA_V2:"),
    ("migration has no transaction", "        conn.execute(\"PRAGMA synchronous = FULL\")\n        conn.execute(\"BEGIN IMMEDIATE\")\n        try:", "        conn.execute(\"PRAGMA synchronous = FULL\")\n        try:"),
    ("shape not checked before migrating", "            raise JournalSchemaError(\"the journal's tables aren't those of schema version 1. It was left unchanged\")", "            pass"),
    ("objects not checked before migrating", "        missing = _V1_REQUIRED_OBJECTS - present\n        if missing:\n            raise JournalCorruptionError(f\"the journal is missing {len(missing)} of its schema objects: {sorted(missing)}. It was left unchanged\")\n", ""),
    ("quick check skipped before migrating", "        if conn.execute(\"PRAGMA quick_check(1)\").fetchone()[0] != \"ok\":\n            raise JournalCorruptionError(\"SQLite's quick check reports the journal file is damaged. It was left unchanged\")\n", ""),
    ("new files get only version 1 objects", "for statement in _SCHEMA + _SCHEMA_V2:", "for statement in _SCHEMA:"),
    ("v2 objects not required at open", "_REQUIRED_OBJECTS = _V1_REQUIRED_OBJECTS | {\"journal_records_correlation\", \"journal_records_correlation_fixed\", \"journal_records_attempt_rises\"}", "_REQUIRED_OBJECTS = _V1_REQUIRED_OBJECTS"),
    ("v2 columns not required at open", "            if set(wanted) - {row[1] for row in conn.execute(f\"PRAGMA table_info({table})\")}:", "            if False:"),
    # integrity
    ("record contents not checked", "and event[\"to_state\"] == values[\"state\"] and self._correlation_is_sound(values, events)", "and event[\"to_state\"] == values[\"state\"]"),
    ("uncorrelated contents not checked", "            return epoch is None and values[\"ledger_attempt\"] is None \\\n                and all((e[\"correlation_key\"], e[\"epoch\"], e[\"ledger_attempt\"]) == (None, None, None) for e in rows)", "            return True"),
    ("history key not compared", "            if (event[\"correlation_key\"], event[\"epoch\"]) != (key, epoch):\n                return False\n", ""),
    ("first-event authorization not checked", "        if first[\"cause\"] != \"CREATED\" or stated != (epoch > 1) or (not stated and (first[\"actor\"], first[\"reason\"]) != (None, None)):\n            return False\n", ""),
    ("carried attempt not checked", "            elif event[\"ledger_attempt\"] != current or (current is None and event[\"to_state\"] != JournalState.PENDING.value):\n                return False\n", ""),
    ("row attempt not compared to history", "        return values[\"ledger_attempt\"] == current\n", "        return True\n"),
    ("attempt rise in one record not checked", "if type(attempt) is not int or attempt < 1 or (current is not None and attempt <= current):", "if type(attempt) is not int or attempt < 1:"),
    ("chain not checked", "        chain_damage = self._chain_problems(rows, events, set(corrupt))\n", "        chain_damage = []\n"),
    ("gaps in a chain not checked", "            if [values[\"epoch\"] for values in chain] != list(range(1, len(chain) + 1)):\n                damaged.extend(values[\"record_id\"] for values in chain)\n                continue\n", ""),
    ("chain resolution not checked", "                                            or earlier[\"resolution\"] != Resolution.FOUND_NOT_COMMITTED.value\n", ""),
    ("chain intent not checked", "                                            or earlier[\"intent_sha256\"] != values[\"intent_sha256\"]):", "                                            ):"),
    ("chain attempts not checked", "                if started and started[0] <= highest_attempt:\n                    damaged.append(values[\"record_id\"])\n", ""),
]


def overlay(root, source):
    for entry in os.listdir(BACKEND):
        if entry in ("tests", "neo4j_write_journal.py", "__pycache__", ".pytest_cache"):
            continue
        os.symlink(os.path.join(BACKEND, entry), os.path.join(root, entry))
    shutil.copytree(os.path.join(BACKEND, "tests"), os.path.join(root, "tests"), ignore=shutil.ignore_patterns("__pycache__"))
    with open(os.path.join(root, "neo4j_write_journal.py"), "w", encoding="utf-8") as handle:
        handle.write(source)


def run(mutant):
    name, old, new = mutant
    if old is None:
        source = SOURCE
    else:
        assert SOURCE.count(old) >= 1, f"pattern not found: {name}"
        source = SOURCE.replace(old, new, 1)
    root = tempfile.mkdtemp(prefix="cyukti-p26b-mut-")
    try:
        overlay(root, source)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "-p", "no:cacheprovider", "--no-header", "--tb=no"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=400)
            tail = [line for line in result.stdout.splitlines() if line.startswith("FAILED") or line.startswith("ERROR")]
            return name, result.returncode, (tail[0][:110] if tail else result.stdout.strip().splitlines()[-1][:110])
        except subprocess.TimeoutExpired:
            return name, "timeout", "timed out"
    finally:
        shutil.rmtree(root, ignore_errors=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(run, MUTANTS))
caught, survived = 0, []
for name, code, detail in results:
    if name == MUTANTS[0][0]:
        print(f"CONTROL {'passes' if code == 0 else 'FAILS: the harness is broken'}: {detail}")
        continue
    if code == 0:
        survived.append(name)
        print(f"SURVIVED  {name}")
    else:
        caught += 1
        print(f"caught    {name}  <- {detail}")
print(f"\nmutants {len(MUTANTS) - 1}, caught {caught}, survived {len(survived)}")
for name in survived:
    print("  survivor:", name)
