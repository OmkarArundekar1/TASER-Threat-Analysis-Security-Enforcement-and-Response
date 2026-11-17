"""Read only. Checks the Phase 2.7 package's citations, cross-references and figures.

usage: python -I -B check_report.py PACKAGE_DIR BACKEND
Exit 0 when nothing is wrong, 1 otherwise.
"""
import ast
import re
import sys
from pathlib import Path

ROOT, BACKEND = Path(sys.argv[1]), Path(sys.argv[2])
DOCS = ["PHASE_2_7_ARCHITECTURE.md", "COORDINATOR_STATE_MACHINE.md", "THROUGHPUT_AND_BACKPRESSURE_DESIGN.md",
        "MARKER_AND_RECOVERY_DESIGN.md", "SECURITY_AND_INTEGRITY.md", "IMPLEMENTATION_WORK_PACKAGES.md",
        "TEST_AND_ACCEPTANCE_MATRIX.md", "OPEN_DECISIONS_AND_BLOCKERS.md", "PHASE_2_7_PLANNING_REPORT.md"]
problems = []
checked = {"function citations": 0, "test citations": 0, "cross references": 0, "model rows": 0, "arithmetic": 0, "transitions": 0}


def text(name):
    path = ROOT / name
    if not path.is_file():
        problems.append(f"missing deliverable {name}")
        return ""
    return path.read_text(encoding="utf-8")


docs = {name: text(name) for name in DOCS}
everything = "\n".join(docs.values())


def functions(rel):
    tree = ast.parse((BACKEND / rel).read_text(encoding="utf-8", errors="replace"))
    found = {}

    def walk(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found[prefix + child.name] = (child.lineno, child.end_lineno)
                walk(child, prefix + child.name + ".")

    walk(tree)
    return found


# 1. `file.py:qualified.name` Lx-y must match the AST exactly.
for name, body in docs.items():
    for rel, qual, start, end in re.findall(r"`([\w/]+\.py):([\w.]+)`\s+L(\d+)-(\d+)", body):
        checked["function citations"] += 1
        table = functions(rel)
        if qual not in table:
            problems.append(f"{name}: {rel}:{qual} does not exist")
        elif table[qual] != (int(start), int(end)):
            problems.append(f"{name}: {rel}:{qual} cited L{start}-{end}, actual L{table[qual][0]}-{table[qual][1]}")

# 2. Named line citations in prose.
fixed = [
    ("neo4j_client.py", "update_duplicate_event", 712),
    ("listener/wazuh_listener.py", 'ALERT_FILE', 26),
    ("config.py", "ALERT_FILE", 10),
    ("neo4j_idempotency_ledger.py", "_TRANSITIONS", 142),
    ("realtime_socgraph.py", "update_duplicate_event", 206),
    ("tests/test_neo4j_write_journal.py", "test_no_production_module_uses_the_journal_yet", 816),
    ("tests/test_neo4j_idempotency_ledger.py", "test_no_production_module_uses_the_ledger_or_journal", 1116),
]
for rel, needle, line in fixed:
    checked["test citations"] += 1
    lines = (BACKEND / rel).read_text(encoding="utf-8", errors="replace").splitlines()
    if needle not in lines[line - 1]:
        problems.append(f"{rel} L{line} does not hold {needle}: the documents cite it there")

# 3. Every cross-reference must be defined where it should be.
defined = {
    "OD": set(re.findall(r"\*\*OD-(\d+)\*\*", docs["OPEN_DECISIONS_AND_BLOCKERS.md"])),
    "BL": set(re.findall(r"\*\*BL-(\d+)\*\*", docs["OPEN_DECISIONS_AND_BLOCKERS.md"])),
    "X": set(re.findall(r"^\| \*\*X(\d+)\*\* \|", docs["OPEN_DECISIONS_AND_BLOCKERS.md"], re.M)),
    "T-": set(re.findall(r"^\| T-(\d+) \|", docs["TEST_AND_ACCEPTANCE_MATRIX.md"], re.M)),
    "WP-": set(re.findall(r"^## WP-(\w+):", docs["IMPLEMENTATION_WORK_PACKAGES.md"], re.M)),
}
patterns = {"OD": r"\bOD-(\d+)\b", "BL": r"\bBL-(\d+)\b", "X": r"\bX(\d+)\b", "T-": r"\bT-(\d+)\b", "WP-": r"\bWP-([A-Z]\d)\b"}
for kind, pattern in patterns.items():
    for name, body in docs.items():
        for ref in re.findall(pattern, body):
            checked["cross references"] += 1
            if ref not in defined[kind]:
                problems.append(f"{name}: {kind}{ref} is referenced and never defined")
for number in sorted(set(str(n) for n in range(1, 27)) - defined["T-"]):
    problems.append(f"required test case T-{number} is missing")

# 4. The backlog table must match the model's output.
model = (ROOT / "evidence/backlog_model_output.txt").read_text(encoding="utf-8")
rows = {5: (6836, 1589, 1662), 10: (6616, 757, 820), 20: (6183, 312, 401), 50: (4893, 99, 156),
        100: (2781, 30, 76), 200: (984, 8, 30), 400: (9, 1, 1)}
throughput = docs["THROUGHPUT_AND_BACKPRESSURE_DESIGN.md"]
for rate, values in rows.items():
    checked["model rows"] += 1
    shown = f"| {rate}/s | {values[0]:,} | {values[1]:,} | {values[2]:,} |"
    if shown not in throughput:
        problems.append(f"throughput table lacks {shown}")
    found = False
    for line in model.splitlines():
        numbers = [int(n) for n in re.findall(r"\d+", line.replace(",", ""))]
        if len(numbers) >= 4 and numbers[0] == rate and tuple(numbers[1:4]) == values:
            found = True
    if not found:
        problems.append(f"model output has no row for {rate}/s with {values}")
for figure in ("379,908", "107,461", "11,566", "2,087", "681,259"):
    if figure not in everything:
        problems.append(f"the package lacks {figure}")
    if figure in ("379,908", "107,461") and figure.replace(",", "") not in model and figure not in model:
        problems.append(f"model output lacks {figure}")

# 5. Arithmetic the documents rely on.
for label, actual, wanted, tolerance in [
    ("largest day mean per second", 379908 / 86400, 4.40, 0.01),
    ("largest day / sample", 379908 / 9015, 42, 0.5),
    ("hours at 20/s", 379908 / 20 / 3600, 5.3, 0.05),
    ("hours at 50/s", 379908 / 50 / 3600, 2.1, 0.05),
    ("target / mean", 50 / (379908 / 86400), 11, 0.5),
    ("capacity / largest day", 1000000 / 379908, 2.6, 0.05),
    ("largest day MB at 1,109 B", 379908 * 1109 / 1e6, 421, 1),
    ("archive MB at 1,109 B", 681259 * 1109 / 1e6, 755.5, 0.5),
    ("share of unattributed duplicates", 7552 / 9015 * 100, 84, 0.5),
    ("alerts a month", 681259 / 9, 76000, 500),
    ("package days", 5 + 2 + 3 + 4 + 3 + 3 + 3 + 4 + 4 + 2 + 3, 36, 0),
]:
    checked["arithmetic"] += 1
    if abs(actual - wanted) > tolerance:
        problems.append(f"{label}: computed {actual}, the documents rely on {wanted}")

# 6. The state design must not rely on a ledger transition that doesn't exist.
ledger = (BACKEND / "neo4j_idempotency_ledger.py").read_text(encoding="utf-8")
actual = set(re.findall(r"\(_S\.(\w+), _S\.(\w+)\): frozenset", ledger))
for pair in [("REGISTERED", "IN_PROGRESS"), ("RETRY_AUTHORIZED", "IN_PROGRESS"), ("IN_PROGRESS", "COMPLETED"),
             ("IN_PROGRESS", "NOT_COMPLETED"), ("IN_PROGRESS", "UNCERTAIN"), ("NOT_COMPLETED", "RETRY_AUTHORIZED"),
             ("UNCERTAIN", "COMPLETED"), ("UNCERTAIN", "NOT_COMPLETED"), ("UNCERTAIN", "ABANDONED")]:
    checked["transitions"] += 1
    if pair not in actual:
        problems.append(f"the design relies on ledger transition {pair}, which doesn't exist")
if ("NOT_COMPLETED", "ABANDONED") in actual:
    problems.append("OD-12 says NOT_COMPLETED to ABANDONED is absent, and it now exists")
if "Conflict --> Abandoned" in docs["COORDINATOR_STATE_MACHINE.md"]:
    problems.append("the state diagram uses Conflict to Abandoned, which the ledger refuses")

# 7. Required content.
for name, needles in {
    "PHASE_2_7_PLANNING_REPORT.md": ["Executive summary", "Existing implementation reused", "Confirmed design decisions",
                                     "Unresolved decisions", "Production compatibility implications",
                                     "Throughput recommendation", "Exact implementation sequence", "Estimated effort",
                                     "Prerequisites for Phase 2.8", "Risks that could invalidate"],
    "COORDINATOR_STATE_MACHINE.md": ["stateDiagram-v2", "Failure at every transition"],
    "PHASE_2_7_ARCHITECTURE.md": ["sequenceDiagram", "Responsibility matrix", "What the caller sees"],
    "THROUGHPUT_AND_BACKPRESSURE_DESIGN.md": ["Approach comparison", "Shutdown and restart", "Rotation", "Monitoring", "The benchmark"],
    "MARKER_AND_RECOVERY_DESIGN.md": ["D13", "Stale payload", "Rehearsal", "Inconsistent marker", "aren't production-scale validation"],
}.items():
    for needle in needles:
        if needle not in docs[name]:
            problems.append(f"{name} lacks {needle!r}")

print("checked: " + ", ".join(f"{k} {v}" for k, v in checked.items()))
for problem in problems:
    print("PROBLEM", problem)
print("problems:", len(problems))
sys.exit(1 if problems else 0)
