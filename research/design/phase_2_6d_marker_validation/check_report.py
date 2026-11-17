"""Read only. Checks the report's figures against the recorded results. usage: python -I -B check_report.py PACKAGE_DIR"""
import glob
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1])
report = (ROOT / "D6_VALIDATION_REPORT.md").read_text(encoding="utf-8")
results = ROOT / "results"
problems = []


def first(pattern):
    return json.loads(Path(sorted(glob.glob(str(results / pattern)))[0]).read_text())


def text(name):
    return (results / name).read_text(encoding="utf-8", errors="replace")


crash = first("crash_*.json")[0]
if crash["attempts"] != 2339 or crash["tally"] != {"COMMITTED marker=1 effect=1": 2309, "AMBIGUOUS marker=1 effect=1": 5, "AMBIGUOUS marker=0 effect=0": 1, "NOT_COMMITTED marker=0 effect=0": 24}:
    problems.append("crash figures differ")
if crash["marker_effect_disagree"] or crash["committed_without_marker"] or crash["not_committed_with_marker"]:
    problems.append("crash disagreements aren't zero")
for run in sorted(glob.glob(str(results / "*_2026*.json"))):
    for row in json.loads(Path(run).read_text()):
        if not all(row["checks"].values()):
            problems.append(f"a failed check in {Path(run).name}: {row['case']}")
if "rows with a failed check: 0" not in text("all_experiments_repeat_output.txt"):
    problems.append("the repeat has failed rows")
if len(re.findall(r"^ok ", text("all_experiments_repeat_output.txt"), re.M)) != 42:
    problems.append("the repeat doesn't have 42 passing rows")
if "24 passed" not in text("focused_tests_output.txt") or "24 passed" not in text("focused_tests_repeat.txt"):
    problems.append("focused tests differ")
performance = text("performance_output.txt")
for line, median, p95, largest in [("mutation alone", "6.052", "9.548", "30.947"), ("marker plus mutation, in the helper", "6.499", "16.267", "29.552"), ("lookup, marker exists  ", "2.493", "3.453", "11.624"),
                                   ("lookup, marker absent  ", "2.428", "3.179", "4.599"), ("duplicate refused", "2.482", "3.626", "7.996"), ("duplicate detected", "5.442", "6.930", "15.208"),
                                   ("2 clients", "8.411", "16.470", "18.172"), ("8 clients", "15.097", "22.566", "25.656"), ("16 clients", "25.651", "46.246", "64.095")]:
    found = [l for l in performance.splitlines() if line in l]
    if not found or not all(v in found[0] for v in (median, p95, largest)):
        problems.append(f"performance row differs: {line}")
    if not all(f"{v} ms" in report for v in (median, p95, largest)):
        problems.append(f"the report lacks a figure for: {line}")
warm = text("performance_warm_output.txt")
for value in ("7.020", "6.595", "2.955"):
    if value not in warm or value not in report:
        problems.append(f"warm figure {value} differs")
for value in ("11.871", "5.141", "1080.364"):
    if value not in performance:
        problems.append(f"cold figure {value} differs")
storage = json.loads((results / "performance.json").read_text())["storage"]
if not 370 <= storage["bytes_per_marker"] <= 390:
    problems.append("storage per marker differs")
audit = text("network_audit_summary.txt")
if "attempts recorded: 27" not in audit or "tests: 16" not in audit or "'127.0.0.1:1': 27" not in audit:
    problems.append("audit figures differ")
environment = json.loads((results / "environment.json").read_text())
if environment["server"]["edition"] != "community" or environment["server"]["versions"] != ["5.26.27"] or environment["environment"]["target"] != "bolt://127.0.0.1:17710":
    problems.append("environment differs")
for matrix in ("Atomicity", "Uniqueness", "Replay safety", "Ambiguous outcome recovery", "Isolation", "Conflict safety", "Failure interpretation", "Retention safety"):
    if not re.search(rf"\| \d \| {matrix} \| \*\*(PASS|FAIL|NOT PROVEN)", report):
        problems.append(f"matrix lacks {matrix}")
for experiment in range(1, 11):
    if f"### E{experiment}:" not in report:
        problems.append(f"report lacks E{experiment}")
for question in range(1, 9):
    if not re.search(rf"^{question}\. \*\*", report, re.M):
        problems.append(f"report lacks answer {question}")
print("PROBLEMS:" if problems else "all checks passed")
for problem in problems:
    print(" ", problem)
sys.exit(1 if problems else 0)
