"""Read only. Second sweep and caller trace, by AST."""
import ast
import os
import re
import sys

ROOT = sys.argv[1]
skip = {"tests", "__pycache__", "node_modules"}
sources = {}
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
    for name in files:
        if name.endswith(".py"):
            path = os.path.join(base, name)
            rel = os.path.relpath(path, ROOT)
            try:
                sources[rel] = (open(path, encoding="utf-8", errors="replace").read(),)
                sources[rel] = (sources[rel][0], ast.parse(sources[rel][0]))
            except SyntaxError as error:
                print("SYNTAX", rel, error)

print("=== every module importing the driver, or the client's driver, or GraphDatabase")
for rel, (text, tree) in sorted(sources.items()):
    hits = [l.strip()[:90] for l in text.splitlines() if re.search(r"(from neo4j\b|import neo4j\b|GraphDatabase|from neo4j_client import|import neo4j_client)", l)]
    if hits:
        print(f"{rel}: {' | '.join(hits)[:230]}")

print("\n=== every .run( whose receiver the first scan's filter skipped, and any non-literal first argument to a session run")
for rel, (text, tree) in sorted(sources.items()):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "run":
            receiver = ast.unparse(node.func.value)[:30]
            first = node.args[0] if node.args else None
            literal = isinstance(first, ast.Constant)
            if not re.search(r"session|tx|driver|sess|neo", receiver, re.I):
                print(f"  OTHER {rel}:{node.lineno} {receiver}.run({ast.unparse(first)[:50] if first else ''})")
            elif not literal:
                print(f"  DYN   {rel}:{node.lineno} {receiver}.run({ast.unparse(first)[:60] if first else ''})")

WRITERS = sys.argv[2].split(",")
print("\n=== callers of each writer: file:line in function")
for writer in WRITERS:
    found = []
    for rel, (text, tree) in sorted(sources.items()):
        stack = []
        def visit(node):
            pushed = isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            if pushed:
                stack.append(node.name)
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else None
                if name == writer:
                    found.append(f"{rel}:{node.lineno} {'.'.join(stack) or '<module>'}")
            for child in ast.iter_child_nodes(node):
                visit(child)
            if pushed:
                stack.pop()
        visit(tree)
    print(f"{writer}: {found if found else 'NO CALLERS'}")

print("\n=== entry points: __main__ blocks, and who imports each candidate")
for rel, (text, tree) in sorted(sources.items()):
    if "__main__" in text and re.search(r"(chain|attack_chain|reconstruct|recalcul|mitre_import|import_|sync|listener|realtime|replay|harness)", rel):
        print("  main:", rel)
for module in ("attack_chain_builder", "campaign_reconstruction", "risk_recalculation", "chain_updater", "operation_repository", "dynamic_risk_engine", "cti_confidence_engine", "cti_publisher", "campaign_manager", "operation_manager", "prediction_engine", "realtime_socgraph", "dedup_engine"):
    users = [rel for rel, (text, _) in sources.items() if re.search(rf"(from {module}\b|import {module}\b)", text) and rel != module + ".py"]
    print(f"  imported: {module}: {sorted(users)}")
