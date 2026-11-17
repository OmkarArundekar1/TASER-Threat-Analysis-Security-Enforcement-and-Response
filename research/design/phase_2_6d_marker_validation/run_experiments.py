"""Runs E1 to E10 and the supporting cases, once, against the disposable instance. Keeps sanitized results and never retries. A failed
check is recorded and reported: it's a result.
usage: python -B run_experiments.py [NAME ...]    Names are in scenarios.ALL. Default: all, in order."""
import json
import sys
import time
from pathlib import Path

import marker_lib as m
import scenarios

RESULTS = Path(__file__).resolve().parent / "results"
RESULTS.mkdir(exist_ok=True)
names = sys.argv[1:] or list(scenarios.ALL)
ctx = m.Context("exp")
info = {"environment": ctx.environment, "server": m.server_identity(ctx.direct), "constraints": m.constraints(ctx.direct)}
(RESULTS / "environment.json").write_text(json.dumps(info, indent=1, default=str))
print(json.dumps({"target": info["environment"]["target"], "container": info["environment"]["container"], "server": info["server"]}))
failed = 0
for name in names:
    started = time.monotonic()
    rows = scenarios.ALL[name](ctx)
    took = round(time.monotonic() - started, 1)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    (RESULTS / f"{name}_{stamp}.json").write_text(json.dumps(rows, indent=1, default=str))
    for row in rows:
        bad = [k for k, v in row["checks"].items() if not v]
        failed += bool(bad)
        extras = {k: v for k, v in row.items() if k not in ("experiment", "case", "checks")}
        print(f"{'ok  ' if not bad else 'FAIL'} {row['experiment']:<8} {row['case'][:70]:<70} {json.dumps(extras, default=str)[:300]}")
        for check in bad:
            print(f"       failed check: {check}")
    print(f"--- {name}: {took} s")
ctx.close()
print(f"rows with a failed check: {failed}")
sys.exit(1 if failed else 0)
