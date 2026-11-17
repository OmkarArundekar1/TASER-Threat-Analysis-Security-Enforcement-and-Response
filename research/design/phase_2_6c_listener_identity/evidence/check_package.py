"""Read only. Checks the Phase 2.6C documents against the source and the recorded outputs.
usage: python -I -B check_package.py DOC_DIR BACKEND"""
import ast
import re
import sys
from pathlib import Path

DOCS, BACKEND = Path(sys.argv[1]), Path(sys.argv[2])
REQUIRED = ["PHASE_2_6C_IMPLEMENTATION_REPORT.md", "D1_SOURCE_IDENTITY_IMPLEMENTATION.md", "D2_ALERT_DEDUPLICATION_FIX.md", "OFFSET_AND_QUEUE_SAFETY.md", "REPLAY_RESULTS.md",
            "TEST_AND_FAULT_RESULTS.md", "PERFORMANCE_RESULTS.md", "PRODUCTION_VALIDATION_GAPS.md", "EVIDENCE_INDEX.md"]
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


tests = (BACKEND / "tests/test_listener_source_identity.py").read_text(encoding="utf-8")
cache, checked, named = {}, 0, 0
for name, text in texts.items():
    for match in re.finditer(r"`([\w/\.]+\.py):([\w\.]+)` L(\d+)-(\d+)", text):
        rel, symbol, start, end = match.group(1), match.group(2), int(match.group(3)), int(match.group(4))
        checked += 1
        table = cache.setdefault(rel, symbols(rel))
        if table.get(symbol) != (start, end):
            problems.append(f"{name}: {rel}:{symbol} cited L{start}-{end}, actually {table.get(symbol)}")
    for match in re.finditer(r"`([\w/\.]+\.py):([A-Za-z_][\w\.]+)`(?! L\d)", text):
        problems.append(f"{name}: citation with no range: {match.group(0)}")
    for path in re.findall(r"`(evidence/[\w\./]+)`", text):
        if not (DOCS / path).exists():
            problems.append(f"{name}: cites missing {path}")
    for test in re.findall(r"`(test_\w+)`", text):
        if test.endswith("_output") or test.startswith("test_new") or test.startswith("test_full"):
            continue
        named += 1
        if f"def {test}(" not in tests:
            problems.append(f"{name}: names a test that doesn't exist: {test}")
    for line in text.splitlines():
        if re.search(r"exactly[- ]once", line, re.I) and not re.search(r"(no |not |can't|doesn't|never)", line, re.I):
            problems.append(f"{name}: an exactly-once claim? {line[:80]}")
notes.append(f"symbol citations checked: {checked}; named tests checked: {named}")

listener = (BACKEND / "listener/wazuh_listener.py").read_text(encoding="utf-8").splitlines()
if "ALERT_FILE" not in listener[25]:
    problems.append("wazuh_listener.py L26 isn't ALERT_FILE")


def output(name):
    return (DOCS / "evidence" / name).read_text(encoding="utf-8")


FACTS = [("replay_before_fix.txt", "first pass: {'accepted': 8945, 'duplicate': 70}"), ("replay_before_fix.txt", "of the 70 previously refused: preserved 0, distinct ids 70"),
         ("replay_before_fix.txt", "of the 7 with a different attacker or victim: preserved 0"), ("replay_before_fix.txt", "'unattributed duplicate': 7486, 'unattributed new event': 151"),
         ("replay_before_fix.txt", "two manager names: {'accepted': 8945, 'duplicate': 9085}"), ("replay_before_fix.txt", "truncated line ending in a newline: reached process_alert 1 of 2"),
         ("replay_before_fix.txt", "a JSON array: reached process_alert 1 of 2"), ("replay_before_fix.txt", "a JSON number: reached process_alert 1 of 2"),
         ("replay_after_fix.txt", "first pass: {'accepted': 9015, 'status:VALID': 9015}"), ("replay_after_fix.txt", "reached process_alert 9015; distinct Wazuh ids 9015"),
         ("replay_after_fix.txt", "of the 70 previously refused: preserved 70, distinct ids 70"), ("replay_after_fix.txt", "of the 7 with a different attacker or victim: preserved 7"),
         ("replay_after_fix.txt", "second pass, same file, same process: {'duplicate': 9015"), ("replay_after_fix.txt", "reached process_alert in total 9015"),
         ("replay_after_fix.txt", "restart, offset at the end: {}"), ("replay_after_fix.txt", "restart, offset 0: {'accepted': 9015"), ("replay_after_fix.txt", "two manager names: {'accepted': 18030"),
         ("replay_after_fix.txt", "'status:MISSING_ID': 9015"), ("replay_after_fix.txt", "'status:EMPTY_ID': 9015"), ("replay_after_fix.txt", "'status:MALFORMED_ID': 9015"), ("replay_after_fix.txt", "'status:MISSING_MANAGER': 9015"),
         ("replay_after_fix.txt", "after second pass 18030"), ("replay_after_fix.txt", "truncated line ending in a newline: reached process_alert 2 of 2"),
         ("replay_after_fix.txt", "'attributed duplicate': 1201, 'attributed new event': 110, 'unattributed duplicate': 7552, 'unattributed new event': 152"),
         ("test_new_output.txt", "157 passed"), ("test_existing_listener_output.txt", "185 passed"), ("test_journal_ledger_output.txt", "712 passed"), ("test_full_suite_output.txt", "2543 passed, 29 skipped"),
         ("test_repeat_stability.txt", "157 passed"), ("mutation_output.txt", "mutants 54, caught 54, survived 0"), ("mutation_output.txt", "CONTROL passes"),
         ("performance_before_fix.txt", "395,187 alerts a second, 2.53 us"), ("performance_before_fix.txt", "cache  10000: median    527.6 us"), ("performance_before_fix.txt", "first pass 4.612 s, 1,955"),
         ("performance_before_fix.txt", "second pass 4.941 s, 1,824"), ("performance_before_fix.txt", "first pass 2.571 s, 3,506"), ("performance_before_fix.txt", "second pass 4.584 s, 1,967"),
         ("performance_after_fix.txt", "352,769 alerts a second, 2.83 us"), ("performance_after_fix.txt", "cache  10000: median    579.8 us"), ("performance_after_fix.txt", "first pass 5.054 s, 1,784"),
         ("performance_after_fix.txt", "second pass 6.618 s, 1,362"), ("performance_after_fix.txt", "first pass 3.375 s, 2,671"), ("performance_after_fix.txt", "second pass 6.299 s, 1,431"),
         ("performance_after_fix.txt", "source key form: 1640 KiB; earlier 64-character hash form: 1903 KiB"),
         ("performance_after_fix.txt", "service    5 ms: written 5655, processed 5655, deepest queue 999, queue-full refusals 38"), ("performance_after_fix.txt", "behind when writing stopped 2937, finished 31.6"),
         ("performance_after_fix.txt", "service  9.2 ms: written 5655, processed 5655, deepest queue 1000, queue-full refusals 80"), ("performance_after_fix.txt", "behind when writing stopped 4133, finished 55.7")]
for name, expected in FACTS:
    if expected not in output(name):
        problems.append(f"{name} lacks: {expected}")
notes.append(f"recorded figures matched: {len(FACTS)}")
report = texts.get("PHASE_2_6C_IMPLEMENTATION_REPORT.md", "")
for value in ("2,543 passed", "54 of 54", "157", "9,015"):
    if value not in report:
        problems.append(f"report lacks {value}")
for gap in re.findall(r"def (test_known_gap_\w+)", tests):
    if not any(gap in text for text in texts.values()) and "known_gap" not in texts.get("OFFSET_AND_QUEUE_SAFETY.md", ""):
        problems.append(f"known gap not reported: {gap}")
if len(re.findall(r"def test_known_gap_", tests)) != 5:
    problems.append("the count of known-gap tests changed: the safety document lists five")

for note in notes:
    print(note)
if problems:
    print(f"PROBLEMS: {len(problems)}")
    for problem in problems:
        print(" ", problem)
    sys.exit(1)
print("all checks passed")
