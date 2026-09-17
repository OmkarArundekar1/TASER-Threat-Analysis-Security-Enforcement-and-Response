# Audit logging (`listener/wazuh_listener.py` → `logs/prerana_listener.log`)

## Root cause

`logs/prerana_listener.log` stayed at 0 bytes indefinitely despite the
listener actively logging to stdout with real traffic. Not buffering,
not a permissions problem, not a wrong path.

`wazuh_listener.py` does `from realtime_socgraph import process_alert`
(line 15) before its own logging setup runs. `realtime_socgraph.py`
calls `logging.basicConfig(level=logging.INFO, format=...)` at **its
own** module level, with no `handlers=` argument — Python's default in
that case is one `StreamHandler` attached to the root logger. That
import therefore configures the root logger as a side effect, before
`wazuh_listener.py` reaches its own
`logging.basicConfig(handlers=[FileHandler(LOG_FILE), StreamHandler()])`
call. `logging.basicConfig()` is documented to do nothing once the root
logger already has any handler — so that call silently did nothing. The
`FileHandler` it constructed was never attached to any logger, anywhere.
Console output kept working the whole time (every logger propagates to
root by default, and root already had realtime_socgraph's
`StreamHandler`), which is exactly why this looked like "logging works"
rather than crashing outright.

Confirmed by direct instrumentation (not inference): importing
`realtime_socgraph` alone left root with `[StreamHandler]`; importing
`wazuh_listener` afterward left root **unchanged** — still just the one
`StreamHandler`, and `logging.getLogger("PreranaListener").handlers ==
[]`.

## Fix

`listener/wazuh_listener.py`'s `_configure_logging()` (replacing the
`logging.basicConfig(...)` block) attaches handlers directly to the root
logger, idempotently:

- Adds a `FileHandler(LOG_FILE)` only if no handler already points at
  that resolved path.
- Adds a console `StreamHandler` only if none already exists (excluding
  `FileHandler`, which subclasses `StreamHandler`).
- Configures the root logger (not just the `"PreranaListener"` named
  logger) — matching the original design intent: `prerana_listener.log`
  is meant to hold everything this process logs while it runs (CTI/MISP
  initialization included), the same set of lines the console already
  shows.

No new logging framework, no changes to `realtime_socgraph.py` or any
other module's logging setup, no manual `flush()` calls added —
`logging.FileHandler`/`StreamHandler.emit()` already flush after every
record by default, so once the handler is actually attached, persistence
is immediate and this was never the problem.

## Intended contract

1. Every `logger.info/warning/error/exception(...)` call reachable from
   `listener/wazuh_listener.py`'s import chain, once `_configure_logging()`
   has run, reaches both the console and `LOG_FILE`.
2. Console output is unaffected — same lines, same format
   (`%(asctime)s | %(levelname)s | %(message)s`).
3. Calling `_configure_logging()` more than once in the same process
   (repeated listener startup, tests) attaches each handler at most
   once — never duplicates lines.
4. `start_listener()`'s real shutdown path (`stop_event` set → workers
   joined → `"Listener stopped."` logged) reaches the file like any
   other line; no special shutdown handling was added because none was
   needed once the handler was actually attached.
5. Messages logged during imports that happen *before*
   `_configure_logging()` runs (e.g. a CTI/MISP singleton's own
   `__init__` logging) predate the file handler's existence and are
   console-only — an inherent property of how logging handlers work,
   not something this fix could or should retroactively capture.

## Tests

`backend/tests/test_audit_logging.py` — runs the real, unmodified
listener module in fresh subprocesses (root-logger handler state is
process-global, so only a fresh interpreter gives a clean answer), each
pointed at an isolated `tmp_path`, never the developer's cwd or the real
log file:

- `test_audit_event_is_actually_written_to_the_log_file` — basic
  persistence, real file read back.
- `test_stdout_logging_still_works_alongside_file_logging` — console
  output preserved.
- `test_repeated_configure_logging_does_not_duplicate_handlers_or_lines`
  — 4 extra `_configure_logging()` calls, handler count and line count
  both stay at 1.
- `test_reimporting_via_fresh_interpreter_does_not_duplicate_prior_content`
  — two independent processes appending to the same file.
- `test_events_persist_across_the_real_start_listener_shutdown_lifecycle`
  — drives the real `start_listener()` through `stop_event`-based
  shutdown; confirms `"Listener stopped."` itself reaches the file.
- `test_real_listener_startup_path_produces_audit_events_that_reach_the_file`
  — the real startup lifecycle end to end.

All 6 were verified to **fail** against the pre-fix code (via `git
stash` on `wazuh_listener.py` alone) before verifying they pass against
the fix — confirming they exercise the actual regression, not a
vacuously-true assertion.

Run: `cd backend && python -m pytest tests/test_audit_logging.py -q`
(full suite: `python -m pytest tests/ -q`, 199 tests).

Real, non-mocked verification beyond the test suite: `logs/` and
`listener/offset.dat` are both gitignored, so `logs/prerana_listener.log`
is a local runtime artifact, not a committed file. It was manually
regenerated during this fix (`python listener/wazuh_listener.py` run
directly, killed after ~2.5s) and inspected with `cat`/`wc -l` — 10 real
lines, matching stdout, confirming the fix end to end outside the test
harness too.
