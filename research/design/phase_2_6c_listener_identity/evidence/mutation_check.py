"""Mutation check for the Phase 2.6C listener code. Each mutant is one deliberate bug, applied to a COPY in a temporary overlay, so the
working tree is never touched. The overlay has real copies of listener/ and tests/, and symlinks for the rest. A mutant is caught if
the listener tests fail. A no-op control must pass, or the harness itself is broken.
usage (from isfcr/CYUKTI/backend, inside the empty network namespace): python -B mutation_check.py"""
import concurrent.futures
import os
import shutil
import subprocess
import sys
import tempfile

BACKEND = os.getcwd()
TESTS = ["tests/test_listener_source_identity.py", "tests/test_listener_alert_acceptance.py"]
U, L = "listener/utils.py", "listener/wazuh_listener.py"
SOURCES = {path: open(path, encoding="utf-8", newline="").read() for path in (U, L)}
CONTROL = "the no-op control"
MUTANTS = [
    (CONTROL, None, None, None),
    # identity
    ("id pattern not anchored", U, "_WAZUH_ID.fullmatch(source_id)", "_WAZUH_ID.match(source_id)"),
    ("non-ASCII digits accepted", U, 'r"[0-9]+\\.[0-9]+"', 'r"\\d+\\.\\d+"'),
    ("id length not limited", U, "len(source_id) > _MAX_ID_LENGTH or ", ""),
    ("non-string id coerced", U, "    if type(source_id) is not str:\n        return SourceIdentity(MALFORMED_ID)\n", "    source_id = str(source_id)\n"),
    ("empty id treated as malformed", U, "return SourceIdentity(EMPTY_ID)", "return SourceIdentity(MALFORMED_ID)"),
    ("id normalized", U, "return SourceIdentity(VALID, manager, source_id)", "return SourceIdentity(VALID, manager, source_id.lstrip('0'))"),
    ("manager defaulted", U, '    if manager_block is None:\n        return SourceIdentity(MISSING_MANAGER)\n', '    if manager_block is None:\n        return SourceIdentity(VALID, "local", source_id)\n'),
    ("manager taken from the agent", U, 'manager = manager_block.get("name")', 'manager = manager_block.get("name") or alert.get("agent", {}).get("name")'),
    ("missing manager name gives a valid identity", U, '    if manager is None or manager == "":\n        return SourceIdentity(MISSING_MANAGER)\n', '    if manager is None or manager == "":\n        return SourceIdentity(VALID, "unknown", source_id)\n'),
    ("manager block type not checked", U, "    if type(manager_block) is not dict:\n        return SourceIdentity(MALFORMED_MANAGER)\n", ""),
    ("manager name type not checked", U, "    if type(manager) is not str:\n        return SourceIdentity(MALFORMED_MANAGER)\n", "    manager = str(manager)\n"),
    ("lone surrogate raises", U, "    except UnicodeEncodeError:  # a lone surrogate, which JSON can carry\n        return SourceIdentity(MALFORMED_MANAGER)\n", "    except ZeroDivisionError:\n        return SourceIdentity(MALFORMED_MANAGER)\n"),
    ("manager length not limited", U, "    if len(encoded) > _MAX_ENCODED_MANAGER_LENGTH:\n        return SourceIdentity(MALFORMED_MANAGER)\n", ""),
    ("manager not encoded in the key", U, "{quote(self.manager, safe='-._~')}", "{self.manager}"),
    ("namespace dropped from the key", U, 'f"{SOURCE_KIND}:{quote(self.manager, safe=\'-._~\')}/{self.source_id}"', 'f"{SOURCE_KIND}:/{self.source_id}"'),
    ("manager name rewritten when stored", U, "return SourceIdentity(VALID, manager, source_id)", "return SourceIdentity(VALID, encoded, source_id)"),
    ("alert_id falls back to content", U, 'alert["alert_id"] = identity.source_key', 'alert["alert_id"] = identity.source_key or content_fingerprint(alert)'),
    ("identity not put on the alert", U, '    alert["source_identity"] = identity.as_dict()\n', ""),
    ("request key loses the version", U, 'return f"{KEY_VERSION}|{source_key}|{operation}"', 'return f"{source_key}|{operation}"'),
    ("operation not validated", U, "or not _OPERATION.fullmatch(operation)", ""),
    ("source key not validated", U, '    if type(source_key) is not str or not source_key.startswith(f"{SOURCE_KIND}:"):\n        raise ValueError("a request key needs a source key")\n', ""),
    ("source_ref logs the key", U, 'return hashlib.sha256(key.encode("utf-8")).hexdigest()[:12] if key else "no-identity"', 'return key or "no-identity"'),
    # aliases
    ("srcip beats src_ip", U, 'source = (data.get("src_ip"), data.get("srcip"),', 'source = (data.get("srcip"), data.get("src_ip"),'),
    ("dstip beats dest_ip", U, 'destination = (data.get("dest_ip"), data.get("dstip"),', 'destination = (data.get("dstip"), data.get("dest_ip"),'),
    ("top level beats data", U, 'source = (data.get("src_ip"), data.get("srcip"), alert.get("src_ip"), alert.get("srcip"))', 'source = (alert.get("src_ip"), alert.get("srcip"), data.get("src_ip"), data.get("srcip"))'),
    ("conflict not detected", U, "conflict = any(len({str(value) for value in group if value}) > 1 for group in (source, destination))", "conflict = False"),
    ("odd data block raises", U, "    data = data if isinstance(data, dict) else {}\n", ""),
    ("fingerprint omits the network fields", U, 'source or "", destination or "", alert.get("location", "")))', 'alert.get("location", "")))'),
    ("odd rule block raises", U, 'rule.get("id", "") if isinstance(rule, dict) else ""', 'rule.get("id", "")'),
    # acceptance
    ("no identity is refused", L, '    if alert_id is None:\n        alert_queue.put_nowait(alert)\n        return True\n', '    if alert_id is None:\n        return False\n'),
    ("no identity is recorded", L, '    if alert_id is None:\n        alert_queue.put_nowait(alert)\n        return True\n', '    if alert_id is None:\n        alert_id = content_fingerprint(alert)\n'),
    ("no identity skips the queue's failure", L, '    if alert_id is None:\n        alert_queue.put_nowait(alert)\n        return True\n', '    if alert_id is None:\n        try:\n            alert_queue.put_nowait(alert)\n        except Exception:\n            pass\n        return True\n'),
    ("recorded before queued", L, "        alert_queue.put_nowait(alert)\n        seen_alerts[alert_id] = now\n", "        seen_alerts[alert_id] = now\n        alert_queue.put_nowait(alert)\n"),
    ("repeat never refused", L, "        if alert_id in seen_alerts:\n            return False\n", ""),
    ("expiry removed", L, "            if now - ts > ALERT_CACHE_TTL\n", "            if False\n"),
    ("cache limit removed", L, "        while len(seen_alerts) > MAX_ALERT_CACHE:\n            seen_alerts.popitem(last=False)\n", ""),
    # reader
    ("identity not assigned in the reader", L, "                    identity = assign_source_identity(alert)\n", "                    identity = sys.modules['utils'].extract_source_identity(alert)\n"),
    ("unavailable identity not noted", L, "                    if not identity.valid:\n                        _note_identity_unavailable(alert, identity)\n", ""),
    ("conflict not noted", L, "                    _note_conflicting_endpoints(alert)\n", ""),
    ("every unavailable alert logged", L, "    if count == 1 or count % 1000 == 0:\n", "    if True:\n"),
    ("unavailable not counted", L, "    identity_unavailable[identity.status] += 1\n", "    identity_unavailable[identity.status] = 1\n"),
    ("non-object line wedges the reader", L, "                    if not isinstance(alert, dict):\n", "                    if alert is None:\n"),
    ("malformed line waits forever", L, '                        unreadable_lines["malformed" if alert is None else "not_an_object"] += 1\n                        logger.warning("Skipping malformed JSON.")\n                        last_good_offset = f.tell()\n                        continue\n', '                        break\n'),
    ("malformed line not counted", L, '                        unreadable_lines["malformed" if alert is None else "not_an_object"] += 1\n', ""),
    ("partial line treated as complete", L, '                    if not line.endswith("\\n"):\n', '                    if False:\n'),
    ("offset passes a refused queue insertion", L, '                            source_ref(alert)\n                        )\n                        break\n                    except Exception:', '                            source_ref(alert)\n                        )\n                        last_good_offset = f.tell()\n                        break\n                    except Exception:'),
    ("offset passes a failed insertion", L, '                            "offset kept, will re-read at the next poll.",\n                            source_ref(alert)\n                        )\n                        break\n                    last_good_offset', '                            "offset kept, will re-read at the next poll.",\n                            source_ref(alert)\n                        )\n                        last_good_offset = f.tell()\n                        break\n                    last_good_offset'),
    ("offset doesn't move past a duplicate", L, "                    last_good_offset = f.tell()\n                    if not accepted:", "                    if accepted:\n                        last_good_offset = f.tell()\n                    if not accepted:"),
    # offset
    ("offset written in place", L, '    temporary = path + ".tmp"\n    with open(temporary, "w") as handle:\n        handle.write(str(offset))\n        handle.flush()\n        os.fsync(handle.fileno())\n    os.replace(temporary, path)\n', '    with open(path, "w") as handle:\n        handle.write(str(offset))\n'),
    ("offset rewritten every poll", L, "                if self.offset != self._saved_offset:\n", "                if True:\n"),
    ("failed save remembered as saved", L, "                    _save_offset(offset_file, self.offset)\n                    self._saved_offset = self.offset\n", "                    self._saved_offset = self.offset\n                    _save_offset(offset_file, self.offset)\n"),
    ("first start not saved", L, "                _save_offset(offset_file, self.offset)\n                self._saved_offset = self.offset\n            except OSError:", "                self._saved_offset = None\n            except OSError:"),
    ("failed first save stops the start", L, "            except OSError:\n                logger.exception(\"Could not save the initial offset.\")\n", "            except ZeroDivisionError:\n                pass\n"),
    ("rotation not detected", L, "            if self.offset > current_size:\n                logger.warning(\n                    \"Offset beyond file size. Assuming log rotation.\"\n                )\n                self.offset = 0\n", ""),
]


def overlay(root, path, source):
    for entry in os.listdir(BACKEND):
        if entry in ("tests", "listener", "__pycache__", ".pytest_cache"):
            continue
        os.symlink(os.path.join(BACKEND, entry), os.path.join(root, entry))
    shutil.copytree(os.path.join(BACKEND, "tests"), os.path.join(root, "tests"), ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(os.path.join(BACKEND, "listener"), os.path.join(root, "listener"), ignore=shutil.ignore_patterns("__pycache__", "logs", "offset.dat", "misp_cache.json"))
    if path:
        with open(os.path.join(root, path), "w", encoding="utf-8", newline="") as handle:
            handle.write(source)


def run(mutant):
    name, path, old, new = mutant
    source = None
    if path:
        source = SOURCES[path]
        if "\r\n" in source:
            old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
        assert source.count(old) >= 1, f"pattern not found: {name}"
        source = source.replace(old, new, 1)
    root = tempfile.mkdtemp(prefix="cyukti-p26c-mut-")
    try:
        overlay(root, path, source)
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            result = subprocess.run([sys.executable, "-m", "pytest", *TESTS, "-x", "-q", "-p", "no:cacheprovider", "--no-header", "--tb=no"], cwd=root, env=env,
                                    capture_output=True, text=True, timeout=400)
            tail = [line for line in result.stdout.splitlines() if line.startswith("FAILED") or line.startswith("ERROR")]
            return name, result.returncode, (tail[0][:110] if tail else result.stdout.strip().splitlines()[-1][:110])
        except subprocess.TimeoutExpired:
            return name, "timeout", "timed out"
    finally:
        shutil.rmtree(root, ignore_errors=True)


for name, path, old, _ in MUTANTS:
    if path:
        pattern = old.replace("\n", "\r\n") if "\r\n" in SOURCES[path] else old
        assert SOURCES[path].count(pattern) >= 1, f"pattern not found: {name}"

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(run, MUTANTS))
caught, survived = 0, []
for name, code, detail in results:
    if name == CONTROL:
        print(f"CONTROL {'passes' if code == 0 else 'FAILS: the harness is broken'}: {detail}")
        continue
    if code == 0:
        survived.append(name)
        print(f"SURVIVED  {name}")
    else:
        caught += 1
        print(f"caught    {name}  <- {detail}")
print(f"\nmutants {len(MUTANTS) - 1}, caught {caught}, survived {len(survived)}")
for name in survived:
    print("  survivor:", name)
