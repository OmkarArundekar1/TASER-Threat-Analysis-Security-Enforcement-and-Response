"""Read only. Checks the Phase 2.6B documents against the source and the recorded outputs.
usage: python -I -B check_package.py DOC_DIR BACKEND"""
import ast
import re
import sys
from pathlib import Path

DOCS, BACKEND = Path(sys.argv[1]), Path(sys.argv[2])
REQUIRED = ["PHASE_2_6B_IMPLEMENTATION_REPORT.md", "AC1_CORRELATION_METADATA.md", "AC2_PREFLIGHT_CONTRACT.md", "AC3_ATTEMPT_FENCING.md", "D4_RETRY_LIFECYCLE.md",
            "SCHEMA_MIGRATION_REPORT.md", "TEST_AND_FAULT_RESULTS.md", "PERFORMANCE_RESULTS.md", "COMPATIBILITY_AND_REGRESSION.md", "EVIDENCE_INDEX.md"]
problems, notes = [], []
texts = {}
for name in REQUIRED:
    if (DOCS / name).exists():
        texts[name] = (DOCS / name).read_text(encoding="utf-8")
    else:
        problems.append(f"missing {name}")


def symbols(rel):
    found = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                found[prefix + child.name] = (child.lineno, child.end_lineno)
                visit(child, prefix + child.name + ".")
            else:
                visit(child, prefix)
    visit(ast.parse((BACKEND / rel).read_text(encoding="utf-8", errors="replace")))
    return found


cache, checked = {}, 0
all_tests = "\n".join((BACKEND / "tests" / f).read_text(encoding="utf-8") for f in (
    "test_neo4j_write_journal_correlation.py", "test_neo4j_write_journal_migration.py", "test_neo4j_write_journal_lifecycle.py"))
child = (BACKEND / "tests/_neo4j_journal_lifecycle_child.py").read_text(encoding="utf-8")
named_tests = 0
for name, text in texts.items():
    for match in re.finditer(r"`([\w/\.]+\.py):([\w\.]+)` L(\d+)-(\d+)", text):
        rel, symbol, start, end = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
        checked += 1
        table = cache.setdefault(rel, symbols(rel))
        if table.get(symbol) != (start, end):
            problems.append(f"{name}: {rel}:{symbol} cited L{start}-{end}, actually {table.get(symbol)}")
    for match in re.finditer(r"`([\w/\.]+\.py):([A-Za-z_][\w\.]+)`(?! L\d)", text):
        problems.append(f"{name}: citation with no range: {match.group(0)}")
    for path in re.findall(r"`(evidence/[\w\.]+)`", text):
        if not (DOCS / path).exists():
            problems.append(f"{name}: cites missing {path}")
    for test in re.findall(r"`(test_\w+)`", text):
        named_tests += 1
        if f"def {test}(" not in all_tests:
            problems.append(f"{name}: names a test that doesn't exist: {test}")
    for action in re.findall(r"`(dies_\w+|migrate_dying)`", text):
        if action not in child:
            problems.append(f"{name}: names an unknown child action: {action}")
    for line in text.splitlines():
        if re.search(r"exactly[- ]once", line, re.I) and not re.search(r"(no |not |can't|doesn't|never)", line, re.I):
            problems.append(f"{name}: an exactly-once claim? {line[:80]}")
notes.append(f"symbol citations checked: {checked}; named tests checked: {named_tests}")

for rel, number, expected in [("tests/test_neo4j_write_journal.py", 75, "SCHEMA_VERSION == 2"), ("tests/test_neo4j_write_journal.py", 801, "preflight_intent")]:
    if expected not in (BACKEND / rel).read_text(encoding="utf-8").splitlines()[number - 1]:
        problems.append(f"{rel} L{number} lacks {expected}")


def output(name):
    return (DOCS / "evidence" / name).read_text(encoding="utf-8")


FACTS = [("test_focused_output.txt", "275 passed"), ("test_focused_by_file.txt", "224 passed"), ("test_focused_by_file.txt", "28 passed"), ("test_focused_by_file.txt", "23 passed"),
         ("test_existing_journal_output.txt", "314 passed"), ("test_ledger_output.txt", "346 passed"), ("test_full_suite_output.txt", "2386 passed, 29 skipped"),
         ("mutation_output.txt", "mutants 82, caught 82, survived 0"), ("mutation_output.txt", "CONTROL passes"), ("test_repeat_stability.txt", "51 passed"),
         ("performance_output.txt", "median    9.228 ms"), ("performance_output.txt", "median    2.869 ms"), ("performance_output.txt", "median    3.101 ms"),
         ("performance_output.txt", "median    0.040 ms"), ("performance_output.txt", "median    0.071 ms"), ("performance_output.txt", "median    1.694 ms"),
         ("performance_output.txt", "uncorrelated, 3000 records: 178.1 ms"), ("performance_output.txt", "correlated, 3000 records: 208.1 ms"),
         ("performance_output.txt", "correlated: 10342400 bytes, 3447"), ("performance_output.txt", "uncorrelated: 9457664 bytes, 3153"),
         ("performance_output.txt", "9000 history entries: 52.7 ms"), ("performance_output.txt", "9281536 bytes before, 9285632 bytes after")]
for name, expected in FACTS:
    if expected not in output(name):
        problems.append(f"{name} lacks {expected}")
notes.append(f"recorded figures matched: {len(FACTS)}")
report = texts.get("PHASE_2_6B_IMPLEMENTATION_REPORT.md", "")
for value in ("2,386 passed", "275 passed", "82 of 82", "9.2 ms", "377"):
    if value not in report:
        problems.append(f"report lacks {value}")
for number in ("1 ", "2 ", "3 ", "4 ", "5 ", "6 ", "7 "):
    if f"| {number}" not in texts.get("D4_RETRY_LIFECYCLE.md", ""):
        problems.append(f"D4 lacks window {number}")
for test_name in re.findall(r"def (test_window_\d)", all_tests):
    pass
windows = {m for m in re.findall(r"def test_window_(\d)_", all_tests)}
if windows != set("1234567"):
    problems.append(f"crash window tests present: {sorted(windows)}")
if "journal_v1_frozen" not in texts.get("SCHEMA_MIGRATION_REPORT.md", ""):
    problems.append("migration report doesn't name the frozen copy")

for note in notes:
    print(note)
if problems:
    print(f"PROBLEMS: {len(problems)}")
    for problem in problems:
        print(" ", problem)
    sys.exit(1)
print("all checks passed")
