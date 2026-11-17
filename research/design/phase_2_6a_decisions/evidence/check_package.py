"""Read only. Checks the Phase 2.6A documents against the source and the evidence outputs.
usage: python -I -B check_package.py DOC_DIR BACKEND"""
import ast
import re
import sys
from pathlib import Path

DOCS, BACKEND = Path(sys.argv[1]), Path(sys.argv[2])
REQUIRED = ["PHASE_2_6A_FINDINGS.md", "D1_SOURCE_IDENTITY_DECISION.md", "D2_ALERT_DEDUPLICATION_INVESTIGATION.md", "D3_JOURNAL_LEDGER_API_CONTRACT.md",
            "D4_RETRY_AFTER_RESOLUTION.md", "D6_APPLIED_WRITE_MARKER_DESIGN.md", "D13_FIRST_WRITER_SELECTION.md", "DECISION_REGISTER_D1_D15.md",
            "PHASE_2_6_IMPLEMENTATION_READINESS.md", "EVIDENCE_INDEX.md"]
problems, notes = [], []


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


texts = {}
for name in REQUIRED:
    if (DOCS / name).exists():
        texts[name] = (DOCS / name).read_text(encoding="utf-8")
    else:
        problems.append(f"missing {name}")

cache, checked = {}, 0
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
    for match in re.finditer(r"`([\w/\.]+\.py):([A-Za-z_][\w\.]+)`(?! L\d)", text):
        problems.append(f"{name}: citation with no range: {match.group(0)}")
    for path in re.findall(r"`(evidence/[\w\.]+)`", text):
        if not (DOCS / path).exists():
            problems.append(f"{name}: cites missing {path}")
    for path in re.findall(r"`([A-Z0-9_]+[A-Z0-9_]*\.md)`", text):
        if path not in REQUIRED and path not in ("TRANSACTION_EXPERIMENT_REPORT.md",):
            problems.append(f"{name}: names unknown document {path}")
notes.append(f"symbol citations checked: {checked}")

# Bare line citations whose content is stated.
LINE_FACTS = [("listener/wazuh_listener.py", 26, "ALERT_FILE"), ("listener/wazuh_listener.py", 35, "seen_alerts = OrderedDict()"),
              ("listener/wazuh_listener.py", 132, 'alert["alert_id"]'), ("listener/wazuh_listener.py", 142, "in seen_alerts"),
              ("listener/wazuh_listener.py", 145, "seen_alerts[alert_id]"), ("listener/wazuh_listener.py", 218, "self.offset = current_size"),
              ("listener/wazuh_listener.py", 274, "except Full"), ("listener/wazuh_listener.py", 288, "last_good_offset = f.tell()"),
              ("listener/wazuh_listener.py", 303, 'open(offset_file, "w")'), ("realtime_socgraph.py", 206, "update_duplicate_event("),
              ("realtime_socgraph.py", 341, "_process_unattributed_alert"), ("realtime_socgraph.py", 461, "update_duplicate_event("),
              ("realtime_socgraph.py", 470, "update_duplicate_event("), ("neo4j_client.py", 541, "MERGE (e)-[:MATCHES]->(t)"),
              ("neo4j_client.py", 727, "MATCH (e)-[:MATCHES]->(t:Technique)"), ("neo4j_client.py", 736, "MATCH (c:Campaign"),
              ("neo4j_client.py", 650, "SET"), ("neo4j_client.py", 652, "c.occurrences"), ("config.py", 22, "EVENT_TIME_POLICY"),
              ("config.py", 35, "ENABLE_DUPLICATE_BUFFER = True"), ("graph_feature_engine.py", 54, "(a)-[r1]->(c)"),
              ("graph_feature_engine.py", 57, "(c)-[r4]->(h)"), ("tests/test_neo4j_write_journal.py", 75, "SCHEMA_VERSION"),
              ("tests/test_neo4j_write_journal_faults.py", 390, "_RECORD_COLUMNS")]
for rel, number, expected in LINE_FACTS:
    line = (BACKEND / rel).read_text(encoding="utf-8", errors="replace").splitlines()[number - 1]
    if expected not in line:
        problems.append(f"{rel} L{number} doesn't hold {expected!r}: {line.strip()[:70]}")
notes.append(f"line facts checked: {len(LINE_FACTS)}")

# Only one place creates MATCHES in production.
creators = []
for path in BACKEND.rglob("*.py"):
    if "tests" in path.parts or "__pycache__" in path.parts:
        continue
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8", errors="replace"))):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and not node.value.lstrip().startswith(("Phase", "Creates")):
            for found in re.finditer(r"(MERGE|CREATE)\s*\(\w*(:\w+)?[^)]*\)\s*-\s*\[\s*\w*:MATCHES", node.value):
                creators.append(str(path.relative_to(BACKEND)))
notes.append(f"statements creating MATCHES in production code: {creators}")
if creators != ["neo4j_client.py"]:
    problems.append(f"MATCHES creators: {creators}")

# Numbers in the documents, against the outputs they came from.
d1 = (DOCS / "evidence/d1_wazuh_id_analysis_output.txt").read_text(encoding="utf-8")
d2 = (DOCS / "evidence/d2_replay_output.txt").read_text(encoding="utf-8")
d3 = (DOCS / "evidence/d3_d13_payload_analysis_output.txt").read_text(encoding="utf-8")
EXPECTED = [(d1, "alerts 9015"), (d1, "with id: 9015; distinct: 9015"), (d1, "match seconds.offset: 9015"), (d1, "went down 0 times, stayed equal 0 times, rose 9014 times"),
            (d1, "busiest second: 409"), (d1, "span 58274 s (0.67 days)"), (d1, "cluster.name present: 0"), (d1, "manager.name distinct values: 1 | alerts with manager.name: 9015"),
            (d1, "all fields except id: 9015 distinct"), (d1, "15 groups repeat; 23 alerts"),
            (d2, "paced by alert timestamps: accepted 8945, refused as repeats 70"), (d2, "fast backlog: accepted 8945, refused as repeats 70"),
            (d2, "differ only in Wazuh id: 0; differ in more: 70; distinct Wazuh ids among refused: 70"), (d2, "share a timestamp with the accepted one: 70"),
            (d2, "{True: 15, False: 55}"), (d2, "{3: 53, 5: 13, 6: 2, 7: 2}"), (d2, "'rule.firedtimes': 70"), (d2, "'data.http.url': 39"), (d2, "'data.tx_id': 38"),
            (d2, "'full_log': 27"), (d2, "'data.dest_ip': 7"), (d2, "'attributed duplicate': 1201, 'attributed new event': 110, 'unattributed duplicate': 7552, 'unattributed new event': 152"),
            (d2, "'attributed duplicate': 1198, 'attributed new event': 110, 'unattributed duplicate': 7486, 'unattributed new event': 151"),
            (d2, "differs from the accepted one: 7"), (d2, "34 of 39 groups"), (d2, "one alert_id: 9"), (d2, "would have been a new event rather than a repeat: 1"), (d2, "reviewed rule mappings: 0"),
            (d3, "unattributed duplicates in the model: 7552"), (d3, "identical across parses 7552; refused {}"), (d3, "identical across parses 7552; distinct 7552"),
            (d3, "median 384"), (d3, "min 59, max 63; refused by the ledger's key check: 0"), (d3, "busiest second 377; seconds with 100 or more 27; rate over the span 11197"),
            (d3, "18 of 27"), (d3, "'timezone': 7552"), (d3, "spelled 'None': 0"), (d3, "strict policy refuses: 0"), (d3, "firedtimes present: 9015")]
for text, expected in EXPECTED:
    if expected not in text:
        problems.append(f"not in the evidence output: {expected}")
notes.append(f"measured values matched against outputs: {len(EXPECTED)}")
for name, value in (("D2_ALERT_DEDUPLICATION_INVESTIGATION.md", "Confirmed defect by static evidence and reproducible offline analysis"),
                    ("D13_FIRST_WRITER_SELECTION.md", "7,552"), ("D13_FIRST_WRITER_SELECTION.md", "377"), ("D6_APPLIED_WRITE_MARKER_DESIGN.md", "11,197"),
                    ("D1_SOURCE_IDENTITY_DECISION.md", "59 to 63"), ("D1_SOURCE_IDENTITY_DECISION.md", "23 alerts")):
    if value not in texts.get(name, ""):
        problems.append(f"{name} lacks {value}")

# Required content the brief names.
REQUIRED_TEXT = {
    "D2_ALERT_DEDUPLICATION_INVESTIGATION.md": ["Remediation contract", "before the coordinator could protect"],
    "D6_APPLIED_WRITE_MARKER_DESIGN.md": ["Evidence applied", "Evidence not applied", "Absence of a marker", "Ambiguous query result", "Is it mandatory"],
    "D3_JOURNAL_LEDGER_API_CONTRACT.md": ["## AC-1", "## AC-2", "## AC-3", "Epoch semantics", "Interaction with `RESOLVED`", "doesn't guarantee"],
    "D4_RETRY_AFTER_RESOLUTION.md": ["The ten questions", "Crash windows", "Minimum change"],
    "D13_FIRST_WRITER_SELECTION.md": ["Comparison", "Does the unattributed path stop"],
    "PHASE_2_6_IMPLEMENTATION_READINESS.md": ["Blocks the coordinator", "Blocks only specific writers", "Safely deferred", "production-shaped data", "authorized Neo4j", "isn't ready"],
}
for name, values in REQUIRED_TEXT.items():
    for value in values:
        if value not in texts.get(name, ""):
            problems.append(f"{name} lacks {value!r}")
register = texts.get("DECISION_REGISTER_D1_D15.md", "")
for number in range(1, 16):
    if not re.search(rf"^\| \*\*D{number}\*\* ", register, re.M):
        problems.append(f"register lacks D{number}")
    row = re.search(rf"^\| \*\*D{number}\*\* .*$", register, re.M)
    if row and row.group(0).count("|") != 8:
        problems.append(f"register row D{number} has {row.group(0).count('|')} bars")

for name, text in texts.items():
    if name not in ("EVIDENCE_INDEX.md", "DECISION_REGISTER_D1_D15.md") and not re.search(r"\[[VP]\]", text):
        problems.append(f"{name} has no labels")
    for line in text.splitlines():
        if re.search(r"exactly[- ]once", line, re.I) and not re.search(r"(no |not |can't|doesn't|never|without)", line, re.I):
            problems.append(f"{name}: an exactly-once claim? {line[:80]}")
    for body in re.findall(r"```mermaid\n(.*?)```", text, re.S):
        first = body.split("\n", 1)[0].strip()
        if first not in ("flowchart LR", "flowchart TD", "stateDiagram-v2"):
            problems.append(f"{name}: unknown diagram {first}")
# Readiness diagram names every prerequisite the brief lists.
readiness = texts.get("PHASE_2_6_IMPLEMENTATION_READINESS.md", "")
for item in ("Source identity contract", "Journal correlation API", "Journal preflight", "Journal attempt fencing", "New-intent retry model", "Applied-write marker", "First writer migration", "Integration and failure tests"):
    if item not in readiness:
        problems.append(f"readiness graph lacks {item}")

for note in notes:
    print(note)
if problems:
    print(f"PROBLEMS: {len(problems)}")
    for problem in problems:
        print(" ", problem)
    sys.exit(1)
print("all checks passed")
