"""Read only. Finds every string constant holding a Cypher write and every driver entry point, by AST,
so names don't decide anything. Prints file:line, enclosing function, and the clause keywords."""
import ast
import os
import re
import sys

ROOT = sys.argv[1]
WRITE = re.compile(r"\b(CREATE|MERGE|SET|DELETE|DETACH|REMOVE|DROP)\b")
CYPHER = re.compile(r"\b(MATCH|MERGE|CREATE|UNWIND|WITH|CALL)\b")
ENTRY = {"run", "execute_write", "execute_read", "begin_transaction", "write_transaction", "read_transaction", "execute_query", "session"}
skip = ("tests", "__pycache__", "node_modules", "frontend", "dist", "build")


def enclosing(stack):
    return ".".join(n.name for n in stack if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))) or "<module>"


def walk(node, stack, out):
    for child in ast.iter_child_nodes(node):
        pushed = isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        if pushed:
            stack.append(child)
        if isinstance(child, ast.Constant) and isinstance(child.value, str) and WRITE.search(child.value) and CYPHER.search(child.value) and "\n" in child.value + "\n":
            text = child.value
            looks = bool(re.search(r"(\(\w*:\w+|\bMATCH\s*\(|\bMERGE\s*\(|\bCREATE\s*\(|\bUNWIND\b)", text))
            if looks:
                out["cypher"].append((child.lineno, enclosing(stack), sorted(set(WRITE.findall(text))), " ".join(text.split())[:110]))
        if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute) and child.func.attr in ENTRY:
            receiver = ast.unparse(child.func.value)[:40]
            if child.func.attr != "run" or re.search(r"session|tx|driver|sess|neo", receiver, re.I):
                out["calls"].append((child.lineno, enclosing(stack), f"{receiver}.{child.func.attr}"))
        walk(child, stack, out)
        if pushed:
            stack.pop()


totals = {}
for base, dirs, files in os.walk(ROOT):
    dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
    for name in sorted(files):
        if not name.endswith(".py"):
            continue
        path = os.path.join(base, name)
        rel = os.path.relpath(path, ROOT)
        try:
            tree = ast.parse(open(path, encoding="utf-8", errors="replace").read())
        except SyntaxError as error:
            print("SYNTAX", rel, error)
            continue
        out = {"cypher": [], "calls": []}
        walk(tree, [], out)
        if out["cypher"] or out["calls"]:
            totals[rel] = out
for rel in sorted(totals):
    out = totals[rel]
    print(f"\n### {rel}: {len(out['cypher'])} write-looking cypher, {len(out['calls'])} driver calls")
    for line, func, kinds, text in out["cypher"]:
        print(f"  C {line:>5} {func:<46} {','.join(kinds):<22} {text}")
    for line, func, what in out["calls"]:
        if "session" not in what.split(".")[-1]:
            print(f"  D {line:>5} {func:<46} {what}")
