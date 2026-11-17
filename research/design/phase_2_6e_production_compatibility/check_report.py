"""Read only. Checks the report's figures and citations against the evidence and source. usage: python -I -B check_report.py PACKAGE_DIR BACKEND"""
import json
import re
import sys
from pathlib import Path

ROOT, BACKEND = Path(sys.argv[1]), Path(sys.argv[2])
report = (ROOT / "PRODUCTION_COMPATIBILITY_REPORT.md").read_text(encoding="utf-8")
evidence = ROOT / "evidence"
neo = json.loads((evidence / "neo4j_metadata.json").read_text())
archive = json.loads((evidence / "wazuh_archive_analysis.json").read_text())
plus = json.loads((evidence / "wazuh_archive_plus_sample_analysis.json").read_text())
details = json.loads((evidence / "wazuh_details.json").read_text())
problems = []


def expect(name, actual, wanted):
    if actual != wanted:
        problems.append(f"{name}: evidence has {actual}, the report relies on {wanted}")


expect("component", neo["results"]["components"][0], {"name": "Neo4j Kernel", "versions": ["5.26.27"], "edition": "community"})
expect("errors", neo["errors"], {})
expect("query types", set(neo["query_types"].values()) <= {"r", "s"}, True)
expect("target", neo["configured_target"], "bolt://localhost:7687")
expect("collected", neo["collected_utc"], "2026-11-06T07:21:23+00:00")
expect("constraints", len(neo["results"]["constraints"]), 10)
labels = {r["label"] for r in neo["results"]["labels"]}
expect("no marker label", "AppliedWrite" in labels, False)
expect("no disposable token", "CyuktiDisposableTestDatabase" in labels, False)
settings = {r["name"]: r["value"] for r in neo["results"]["settings"]}
expect("transaction timeout", settings["db.transaction.timeout"], "0s")
expect("lock timeout", settings["db.lock.acquisition.timeout"], "0s")
t = archive["totals"]
expect("alerts", t["alerts"], 681259)
expect("kinds", archive["id_kinds"], {"string_seconds_dot_offset": 681259})
expect("managers", archive["manager_kinds"], {"name_string": 681259})
expect("one manager", list(archive["managers_by_short_hash"]), ["e9df13e40857"])
expect("unique", archive["unique_manager_and_id"], 681206)
expect("identical repeats", archive["repeats_identical_line"], 53)
expect("collisions", t.get("duplicates_different_content", 0), 0)
expect("offset fell", t["offset_fell"], 0)
expect("offset repeated", t["offset_repeated"], 0)
expect("seconds fell", t["seconds_fell"], 15)
expect("not utf8", t["not_utf8"], 0)
expect("bad lines", (t["malformed_json"], t["partial_final_line"], t["lines"]), (3, 2, 681264))
expect("seconds match", archive["id_seconds_minus_timestamp_clamped"], {"0": 681259})
expect("tolerant", t["tolerant_policy_accepts"], 681259)
expect("timestamp form", t["timestamp_expected_form"], 681259)
expect("no cluster", t.get("has_cluster_block", 0), 0)
expect("lengths", (min(map(int, archive["id_lengths"])), max(map(int, archive["id_lengths"]))), (12, 20))
expect("coverage", (archive["distinct_agents"], archive["distinct_rules"], archive["decoder_count"], archive["distinct_locations"]), (2, 94, 24, 21))
expect("files", len(archive["files"]), 111)
expect("months", (archive["months"]["2026-10"], archive["months"]["2026-04"], len(archive["months"])), (182, 497189, 9))
expect("all files start at 0", all((f["first_id"] or ".0").endswith(".0") for f in archive["files"]), True)
expect("sample copy", plus["repeats_identical_line_where"].get("2026/Mar/ossec-alerts-11.json and data/raw_logs/alerts.json"), 9015)
expect("sample collisions", plus["totals"].get("duplicates_different_content", 0), 0)
d = details["totals"]
expect("merged", (d["alerts"], d["merged"], d["merged_identical_content_apart_from_id"], d["merged_uses_src_ip_spelling"], d["merged_has_no_network_field"]), (681206, 80396, 92, 58422, 19821))
expect("peak", (details["busiest_second_anywhere"], details["seconds_with_100_or_more"], details["largest_id_seconds_drop"]), (11566, 2087, 126))
top = details["top_merging_days"][0]
expect("worst day", (top["file"], top["merged"], top["alerts"]), ("2026/Sep/ossec-alerts-26.json", 29871, 30862))
bad = [b for rows in details["bad_lines"].values() for b in rows]
expect("no bad line starts with a brace", any(b["starts_with_brace"] for b in bad), False)
expect("starts", len((evidence / "analysisd_starts_in_current_log.txt").read_text().splitlines()), 11)
logs = (evidence / "listener_log_counts.txt").read_text()
expect("listener counts", ("duplicate ignored 313" in logs, "duplicate ignored 34" in logs), (True, True))

for value in ("681,259", "681,206", "80,396", "11,566", "126 s", "96.8%", "11.8%", "29,871 of 30,862", "9,015 of 9,015", "5.26.27", "4.7.5-1"):
    if value not in report:
        problems.append(f"the report lacks {value}")
for classification in ("`MATCHES_TESTED_ENVIRONMENT`", "`ACCEPT_TOLERANT_STRING_POLICY`", "Ready for implementation planning", "**Blocked**"):
    if classification not in report:
        problems.append(f"the report lacks {classification}")


def lines_of(rel):
    return (BACKEND / rel).read_text(encoding="utf-8", errors="replace").splitlines()


listener, utils = lines_of("listener/wazuh_listener.py"), lines_of("listener/utils.py")
for lines, number, text in ((listener, 33, "MAX_ALERT_CACHE"), (listener, 34, "ALERT_CACHE_TTL"), (listener, 35, "seen_alerts = OrderedDict"), (listener, 128, "def accept_alert"), (utils, 31, "_WAZUH_ID = re.compile"), (lines_of("neo4j_client.py"), 74, "driver = GraphDatabase.driver"), (lines_of("config.py"), 6, "NEO4J_URI")):
    if text not in lines[number - 1]:
        problems.append(f"line {number} doesn't hold {text}: {lines[number - 1][:60]}")
for number, wanted in ((61, "source_id is None"), (69, "manager_block = "), (85, "SourceIdentity(VALID")):
    if wanted not in utils[number - 1]:
        problems.append(f"utils.py L{number} doesn't hold {wanted}")
for path in re.findall(r"`(evidence/[\w\.]+)`", report):
    if not (ROOT / path).exists():
        problems.append(f"cites missing {path}")
print("PROBLEMS:" if problems else "all checks passed")
for problem in problems:
    print(" ", problem)
sys.exit(1 if problems else 0)
