"""Read only. For D6 and D13: which production queries could see an unconnected marker node, and every read that
could count nodes without a label. Works on source text, by AST. usage (from isfcr/CYUKTI/backend): python -I d6_d13_static_scan.py"""
import ast
import os
import re

skip = {"tests", "__pycache__", "node_modules"}
LABEL_FREE = re.compile(r"\(\s*\w*\s*\)")
print("=== string constants that start from, or test, a node with no label and no relationship next to it")
hits = 0
for base, dirs, files in os.walk("."):
    dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
    for name in sorted(files):
        if not name.endswith(".py") or name in ("neo4j_write_journal.py", "neo4j_idempotency_ledger.py"):
            continue
        path = os.path.relpath(os.path.join(base, name), ".")
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Constant) and isinstance(node.value, str)):
                continue
            text = node.value
            if "MATCH" not in text and "CALL db." not in text:
                continue
            flat = " ".join(text.split())
            for match in re.finditer(r"(MATCH|WHERE EXISTS|MERGE)\s*\(\s*\w*\s*\)(?!\s*(-|<))", flat):
                hits += 1
                print(f"  {path}:{node.lineno} ...{flat[max(0, match.start()-10):match.end()+50]}")
            for term in ("CALL db.labels", "db.schema", "apoc.meta", "labels(n)", "count(*) AS total_nodes"):
                if term in flat:
                    hits += 1
                    print(f"  {path}:{node.lineno} uses {term}")
print(f"hits: {hits}")
print("\n=== who reads dedup metadata and fingerprints: the identity W07 depends on")
for rel, pattern in (("realtime_socgraph.py", r"get_metadata|is_duplicate|update_duplicate_event|alert_id|\.get\(\"id\"\)"),
                     ("dedup_engine.py", r"def is_duplicate|def get_metadata|find_recent_duplicate\("),
                     ("neo4j_client.py", r"def find_recent_duplicate|def update_duplicate_event|MATCH \(e\)-\[:MATCHES\]")):
    for number, line in enumerate(open(rel, encoding="utf-8"), 1):
        if re.search(pattern, line):
            print(f"  {rel}:{number}: {line.strip()[:90]}")
