import glob
import importlib
import os
import sys

BACKEND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend")
os.chdir(BACKEND)
sys.path.insert(0, os.getcwd())

EXCLUDE_DIRS = {"external", "frontend", "node_modules", "__pycache__", ".ipynb_checkpoints", "mitredata"}

failures = []
oks = []


def _module_name(py_path: str) -> str:
    rel = os.path.relpath(py_path, BACKEND).replace(os.sep, "/")
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _try_import(mod: str) -> None:
    try:
        importlib.import_module(mod)
        oks.append(mod)
    except Exception as e:
        failures.append((mod, type(e).__name__, str(e)))


# flat modules at backend root
for path in sorted(glob.glob("*.py")):
    _try_import(path[:-3])

# every subpackage (any directory with an __init__.py), recursively
for root, dirs, files in os.walk(BACKEND):
    dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
    if "__init__.py" not in files:
        continue
    for f in files:
        if f.endswith(".py"):
            _try_import(_module_name(os.path.join(root, f)))

print(f"{len(oks)} modules imported OK")
if failures:
    print(f"{len(failures)} import failures:")
    for mod, etype, msg in failures:
        first_line = msg.splitlines()[0] if msg else ""
        print(f"  [{mod}] {etype}: {first_line}")
else:
    print("No import failures.")
