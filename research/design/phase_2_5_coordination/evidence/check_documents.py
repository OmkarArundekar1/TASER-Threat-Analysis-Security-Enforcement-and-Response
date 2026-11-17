"""Read only. Checks the Phase 2.5 documents against the source. usage: check_documents.py DOC_DIR BACKEND"""
import ast
import json
import os
import re
import sys
from pathlib import Path

DOCS, BACKEND = Path(sys.argv[1]), Path(sys.argv[2])
REQUIRED = ["PHASE_2_5_SCOPE_AND_FINDINGS.md", "PRODUCTION_NEO4J_WRITER_INVENTORY.md", "REQUEST_IDENTITY_MAPPING.md",
            "JOURNAL_LEDGER_COORDINATION_CONTRACT.md", "CROSS_COMPONENT_STATE_TRANSITIONS.md", "FAILURE_WINDOW_AND_RECOVERY_ANALYSIS.md",
            "COORDINATOR_INTERFACE_PROPOSAL.md", "PHASE_2_6_IMPLEMENTATION_AND_VALIDATION_PLAN.md", "EVIDENCE_INDEX.md"]
problems, notes = [], []


def symbols(rel):
    tree = ast.parse((BACKEND / rel).read_text(encoding="utf-8", errors="replace"))
    found = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found[prefix + child.name] = (child.lineno, child.end_lineno)
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)
    visit(tree)
    return found


cache = {}
texts = {}
for name in REQUIRED:
    path = DOCS / name
    if not path.exists():
        problems.append(f"missing document {name}")
        continue
    texts[name] = path.read_text(encoding="utf-8")

checked = 0
for name, text in texts.items():
    for match in re.finditer(r"`([\w/\.]+\.py):([\w\.]+)` L(\d+)-(\d+)", text):
        rel, symbol, start, end = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
        checked += 1
        if not (BACKEND / rel).exists():
            problems.append(f"{name}: no file {rel}")
            continue
        table = cache.setdefault(rel, symbols(rel))
        if symbol not in table:
            problems.append(f"{name}: {rel} has no {symbol}")
        elif table[symbol] != (start, end):
            problems.append(f"{name}: {rel}:{symbol} cited L{start}-{end}, actually L{table[symbol][0]}-{table[symbol][1]}")
    # A citation with no range would dodge the check above.
    for match in re.finditer(r"`([\w/\.]+\.py):([A-Za-z_][\w\.]+)`(?! L\d)", text):
        problems.append(f"{name}: citation without a line range: {match.group(0)}")
notes.append(f"symbol citations checked: {checked}")


def lines_of(rel, start, end):
    return "\n".join((BACKEND / rel).read_text(encoding="utf-8").splitlines()[start - 1:end])


journal = lines_of("neo4j_write_journal.py", 115, 123)
ledger = lines_of("neo4j_idempotency_ledger.py", 142, 152)
if "_TRANSITIONS" not in journal or "AMBIGUOUS, _S.RESOLVED" not in journal:
    problems.append("journal transition table isn't at L115-123")
if "_TRANSITIONS" not in ledger or "UNCERTAIN, _S.ABANDONED" not in ledger:
    problems.append("ledger transition table isn't at L142-152")
config = (BACKEND / "config.py").read_text(encoding="utf-8").splitlines()
if config[34].strip() != "ENABLE_DUPLICATE_BUFFER = True" or not config[33].startswith("MAX_BATCH_SIZE = 200"):
    problems.append("config.py L34 or L35 aren't what the inventory says")

inventory = texts.get("PRODUCTION_NEO4J_WRITER_INVENTORY.md", "")
fields = ["Module, function", "Callers", "Logical operation", "Neo4j effects", "Statements, boundary", "Input identity", "Retry behaviour",
          "Idempotency", "Duplicate risk", "Failure handling", "Candidate key", "Identity confidence", "Migration risk"]
sections = re.split(r"\n### (W\d\d): ", inventory)
writers = {}
for index in range(1, len(sections), 2):
    writers[sections[index]] = sections[index + 1].split("\n## ")[0]
expected = [f"W{n:02}" for n in range(1, 21)]
if sorted(writers) != expected:
    problems.append(f"writers in the inventory: {sorted(writers)}")
for writer, body in writers.items():
    for field in fields:
        if f"| {field} |" not in body:
            problems.append(f"{writer} lacks {field}")
tier_rows = re.findall(r"\| (\d|Excluded) \| [^|]+ \| (\d+) \| ([^|]+) \|", inventory)
named = []
for tier, count, listed in tier_rows:
    if tier != "Excluded":
        ids = re.findall(r"W\d\d", listed)
        if len(ids) != int(count):
            problems.append(f"tier {tier}: says {count}, lists {len(ids)}")
        named += ids
if sorted(named) != expected:
    problems.append(f"tiers don't cover each writer once: {sorted(named)}")
identity_rows = re.findall(r"\| (High, given decision D1|Medium:[^|]+|\*\*Unresolved\*\*) \| (\d+) \| ([^|]+) \|", inventory)
named = []
for label, count, listed in identity_rows:
    ids = re.findall(r"W\d\d", listed)
    if len(ids) != int(count):
        problems.append(f"identity row {label[:20]}: says {count}, lists {len(ids)}")
    named += ids
if sorted(named) != expected:
    problems.append(f"identity rows don't cover each writer once: {sorted(named)}")
# Each writer's stated function really exists, and has no write outside what's inventoried.
scan = BACKEND / "neo4j_client.py"
source = scan.read_text(encoding="utf-8")
live = ["create_campaign_db", "reopen_campaign_db", "close_campaign_db", "update_campaign_activity", "create_attack_event", "create_unattributed_attack_event",
        "update_duplicate_event", "batch_update_duplicates", "store_dynamic_risk", "update_campaign_similarity", "update_actor_attribution", "store_prediction",
        "store_cti_confidence", "reopen_operation_db", "create_operation_db", "attach_campaign_to_operation", "update_operation_activity",
        "expire_stale_campaigns_db", "expire_stale_operations_db"]
for function in live + ["close_operation_db", "archive_campaign_db", "store_likely_next"]:
    if f"neo4j_client.py:{function}`" not in inventory:
        problems.append(f"inventory doesn't cite {function}")

matrix = texts.get("CROSS_COMPONENT_STATE_TRANSITIONS.md", "")
for number in range(1, 22):
    if not re.search(rf"^\| {number}[a-z]? \|", matrix, re.M):
        problems.append(f"matrix lacks scenario {number}")
rows = [r for r in re.findall(r"^\| \d+[a-z]? \|.*$", matrix, re.M)]
for row in rows:
    if row.count("|") != 10:
        problems.append(f"matrix row has the wrong column count: {row[:60]}")
notes.append(f"matrix rows: {len(rows)}")

# State diagram against the ledger's real table.
contract = texts.get("JOURNAL_LEDGER_COORDINATION_CONTRACT.md", "")
block = re.search(r"```mermaid\nstateDiagram-v2\n(.*?)```", contract, re.S).group(1)
edges = set(re.findall(r"^\s*(\w+) --> (\w+):", block, re.M))
real = set(re.findall(r"\(_S\.(\w+), _S\.(\w+)\):", ledger))
if edges != real:
    problems.append(f"state diagram differs from the ledger: only in diagram {edges - real}, only in ledger {real - edges}")
notes.append(f"state diagram transitions matched against the ledger: {len(real)}")
journal_states = set(re.findall(r"_S\.(\w+)", journal))
ledger_states = set(re.findall(r"_S\.(\w+)", ledger))
for name, text in (("matrix", matrix), ("contract", contract)):
    for state in re.findall(r"`([A-Z_]{5,})`", text):
        known = journal_states | ledger_states | {"PENDING", "NEW", "DUPLICATE", "FOUND_COMMITTED", "FOUND_NOT_COMMITTED", "FOUND_COMPLETED", "FOUND_NOT_COMPLETED",
                                                  "ABANDONED", "ALREADY_COMPLETED", "IN_PROGRESS_ELSEWHERE", "AWAITING_AUTHORIZATION", "BLOCKED_UNCERTAIN", "BLOCKED_ABANDONED",
                                                  "EXECUTED_COMMITTED", "INCONSISTENT", "TransactionTimedOutClientConfiguration"}
        if state not in known:
            problems.append(f"{name}: unknown state {state}")

for name, text in texts.items():
    if name != "EVIDENCE_INDEX.md" and not re.search(r"\[[VP]\]", text):
        problems.append(f"{name} has no labels")
    blocks = re.findall(r"```mermaid\n(.*?)```", text, re.S)
    for body in blocks:
        first = body.split("\n", 1)[0].strip()
        if first not in ("sequenceDiagram", "flowchart LR", "stateDiagram-v2"):
            problems.append(f"{name}: unknown diagram type {first}")
        if first == "sequenceDiagram":
            declared = set(re.findall(r"participant (\w+)", body))
            used = set(re.findall(r"^\s*(\w+)-+>>", body, re.M)) | set(re.findall(r"-+>>(\w+):", body))
            if used - declared:
                problems.append(f"{name}: undeclared participants {used - declared}")
            if body.count("\n    alt ") + body.count("\n    else ") and body.count("\n    end") != body.count("\n    alt "):
                problems.append(f"{name}: unbalanced alt block")
    notes.append(f"{name}: {len(blocks)} diagrams")
    for line in text.splitlines():
        if re.search(r"exactly[- ]once", line, re.I) and not re.search(r"(no |not |can't|doesn't|denial|isn't|never|without)", line, re.I):
            problems.append(f"{name}: a claim of exactly-once? {line[:90]}")

# Every function holding a Cypher write in the raw scan must be a writer or an exclusion.
scan_file = DOCS / "evidence" / "cypher_scan.txt"
if scan_file.exists():
    current = None
    accounted = 0
    for line in scan_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("### "):
            current = line[4:].split(":")[0]
        match = re.match(r"\s+C\s+\d+\s+(\S+)", line)
        if match:
            function = match.group(1)
            if function == "<module>":
                if os.path.basename(current) not in inventory:
                    problems.append(f"scan: {current} isn't accounted for")
            elif function.split(".")[-1] not in inventory:
                problems.append(f"scan: {current}:{function} isn't accounted for")
            accounted += 1
    notes.append(f"Cypher writes in the raw scan, all accounted for: {accounted}")
else:
    problems.append("no raw scan to compare with")

for note in notes:
    print(note)
if problems:
    print(f"PROBLEMS: {len(problems)}")
    for problem in problems:
        print(" ", problem)
    sys.exit(1)
print("all checks passed")
