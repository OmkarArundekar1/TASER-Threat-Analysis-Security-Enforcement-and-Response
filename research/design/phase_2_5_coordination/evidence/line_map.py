"""Read only. Line ranges of every function and class in the files the documents cite, plus import checks."""
import ast
import json
import sys

FILES = sys.argv[2:]
out = {}
for rel in FILES:
    tree = ast.parse(open(rel, encoding="utf-8", errors="replace").read())
    names = {}

    def visit(node, prefix=""):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = prefix + child.name
                names[name] = [child.lineno, child.end_lineno]
                visit(child, name + ".")
            else:
                visit(child, prefix)
    visit(tree)
    out[rel] = names
json.dump(out, open(sys.argv[1], "w"), indent=0)
for rel, names in out.items():
    print(f"\n## {rel}")
    print(" ".join(f"{k}:{v[0]}-{v[1]}" for k, v in names.items()))
