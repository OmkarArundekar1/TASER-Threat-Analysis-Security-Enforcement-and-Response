# CYUKTI MITRE ATT&CK Mapping

## Real-world validation: a live Nmap scan (UX-redesign phase)

A real `nmap` scan from Kali (`192.168.56.106`) against `192.168.56.105` was inspected end-to-end this phase. Wazuh's custom rule `100500` (existing since before this session, correctly configured, native-tier) tagged the Nmap Scripting Engine's distinctive HTTP User-Agent with `T1595` (Active Scanning / Reconnaissance) — a real, defensible, non-fabricated mapping, confirmed flowing all the way through to a real Neo4j `AttackEvent` and campaign. Full narrative, including a correction to an initial assumption about how CYUKTI classified it, is in `INCIDENT_VIEW.md`'s "Real-world validation" section.

## Live activation status (this phase)

Neo4j was found down at the start of this phase and was started (`docker start neo4j-soc`, no privileged access needed — the account is in the `docker` group). With Neo4j and the CYUKTI backend/listener running live, the Wazuh manager itself was confirmed **already running** (started 04:58 UTC), producing real alerts. The rule fixes below (committed to `backend/wazuh_rules/local_rules.xml` in a prior phase and copied to the live `/var/ossec/etc/rules/local_rules.xml`) predate that manager start, so **they are still not active in the live manager's in-memory ruleset** — confirmed by comparing the rule file's mtime (05:09:59) against the manager's `ActiveEnterTimestamp` (04:58:44). Activating them requires `sudo /var/ossec/bin/wazuh-control restart`, which requires an interactive password this environment does not provide non-interactively (`sudo -n`/`sudo -ln` both refused). **Not attempted further; not claimed as done.**
*Planned:* have the operator run the manager restart with interactive sudo access in a maintenance window, then re-verify the rule fixes against live alert traffic.

## New finding this phase: a real STIX corpus data-quality issue

While building `enrich_technique_metadata()`'s live `/api/incidents/<id>/overview` integration, spot-checking real technique data revealed that **268 of 858** imported ATT&CK techniques have a non-standard `kill_chain_phases` value — e.g. T1562.001 ("Disable or Modify Tools", genuinely a Defense Evasion technique) and T1055 ("Process Injection", genuinely Defense Evasion + Privilege Escalation) both carry `"stealth"` instead of `"defense-evasion"` in their stored `kill_chain_phases`. Other technique carry `"defense-impairment"`, also not a real ATT&CK Enterprise tactic.

Traced to the **root**: this is not an import bug in `mitre_import/mapper.py` (which correctly extracts every `phase_name` under `kill_chain_name: "mitre-attack"`, with no filtering logic that could be at fault) — the *raw, vendored* `backend/mitredata/attack-stix-data/enterprise-attack/enterprise-attack.json` file itself genuinely contains `{"kill_chain_name": "mitre-attack", "phase_name": "stealth"}` for T1562.001. That file is gitignored ("vendored external reference data — re-fetch, don't commit") and was not re-verified against MITRE's own GitHub source (`mitre-attack/attack-stix-data`) this phase.

**Not silently corrected** — guessing a mapping from `"stealth"` → `"defense-evasion"` for 268 techniques without confirming the full scope/pattern of the discrepancy would itself be a fabrication. Flagged for the user: re-fetch the official bundle and diff, or confirm whether this vendored copy was intentionally customized for this lab. Until then, any `tactic` field surfaced via `enrich_technique_metadata()` (the Incident View header, `/api/incidents/<id>/overview`) should be read with this caveat — `mitre_id` and `technique_name` are unaffected and were spot-checked as correct throughout.
*Planned:* re-fetch the official STIX bundle from `mitre-attack/attack-stix-data`, diff it against the vendored copy to confirm the scope of the discrepancy, and correct the affected `kill_chain_phases` values (or confirm the vendored copy was intentionally customized).

## Precedence (unchanged from Phase 20, `mitre_resolver.py`)

1. `NATIVE_WAZUH` — `rule.mitre.id`, exactly as Wazuh (or a custom rule) supplies it. Confidence `CONFIRMED`.
2. `REVIEWED_RULE_MAPPING` — `mitre_rule_registry.REVIEWED_RULE_MAPPINGS`, keyed by Wazuh rule ID. Confidence `REVIEWED`.
3. `DETERMINISTIC_INFERENCE` — explicit structural rules only, no fuzzy/semantic matching. Confidence `CANDIDATE`. Ships empty; add a rule only when independently defensible.
4. `AMBIGUOUS` — a deterministic-inference rule matched multiple equally-plausible techniques; never arbitrarily resolved to one.
5. `UNKNOWN` — no defensible mapping. `technique_ids=()`. This is a first-class, valid, and *preferred* outcome over a fabricated guess.

`REVIEWED_RULE_MAPPING` and `DETERMINISTIC_INFERENCE` results are validated against the real imported ATT&CK STIX corpus in Neo4j (a technique that's revoked/deprecated/nonexistent downgrades the whole result to `UNKNOWN`). `NATIVE_WAZUH` is never second-guessed this way — a native Wazuh mapping is authoritative.

## Data model (added this phase)

`mitre_resolver.enrich_technique_metadata(resolution)` returns, per technique_id:

```json
{
  "mitre_id": "T1110.001",
  "technique_name": "Password Guessing",
  "tactic": ["Credential Access"],
  "mapping_source": "NATIVE_WAZUH",
  "mapping_confidence": "CONFIRMED",
  "mapping_reason": "Wazuh rule.mitre.id"
}
```

`technique_name`/`tactic` come from the real Neo4j `Technique` node (`name`, `kill_chain_phases`) — `None`/`[]` if the technique can't be looked up (Neo4j down, not in the corpus), never a fabricated placeholder. Purely a display/evaluation enrichment; it never gates or reinterprets `resolve_mitre()`'s own result.

## Local Wazuh rule audit and fix (live-verified, 2026-09-25)

Auditing `/var/ossec/etc/rules/local_rules.xml` (this environment's actual, live custom rule file — a real Wazuh manager is installed and running here) found a genuine, pre-existing configuration bug: **six rule IDs were each defined twice** (`100001`, `100003`, `100004`, `100005`, `100500`, `100501`). Wazuh only honors the first definition of a duplicated ID and silently discards the rest — confirmed live via `wazuh-logtest`'s explicit `Rule ID '...' is duplicated` warnings. This meant six rules the operator clearly intended to be active had never fired:

| Original ID | New ID | Rule | Fix |
|---|---|---|---|
| 100001 (2nd) | 100510 | SSH "Multiple Failed Login Attempts" (if_sid 5716) | Renumbered + added `<mitre><id>T1110</id></mitre>` |
| 100003 (2nd) | 100511 | "Suspicious Process" (wget\|curl) | Renumbered + added T1105 (Ingress Tool Transfer) |
| 100004 (2nd) | 100512 | `[INVENTORY_ABUSE]` | Renumbered only — deliberately left unmapped (see below) |
| 100005 (2nd) | 100513 | "SLOW endpoint" DoS | Renumbered + added T1499.002 (Service Exhaustion Flood) |
| 100500 (2nd) | 100514 | Ignore dpkg installed (if_sid 2902) | Renumbered only (suppression rule, no MITRE needed) |
| 100501 (2nd) | 100515 | Ignore dpkg configured (if_sid 2904) | Renumbered only |

The last two are notable: they were meant to suppress dpkg noise, but being dead meant dpkg install/config events have been reaching the pipeline unfiltered this whole time — which is exactly why rule IDs 2902/2904 appear in `mitre_rule_registry.py`'s own docstring as "currently-unmapped lab rules." Fixing the collision (not adding a mapping — these are legitimately not attacker behavior) resolves that.

Also added `<mitre>` blocks to already-unique, already-firing rules where defensible: `210001` (BOT_ATTACK → T1498.001, Direct Network Flood), `210010`/`210011`/`100100`/`100200` (LOGIN_FAIL/LOGIN_ATTACK → T1110), `210013` (SLOW endpoint → T1499.002, duplicate content of 100513).

**Deliberately left unmapped** (documented, not silently skipped):
- `210020` / bare `sudo` match — already flagged in `mitre_rule_registry.py`'s own docstring as needing independent review; matching the literal word "sudo" is extremely overbroad (fires on any invocation, legitimate or not) with no behavioral specificity.
- `210012` / `100300` / `100512` (`INVENTORY_ABUSE`, "INVENTORY accessed") — synthetic labels from this lab's traffic generator with no independently verifiable behavioral definition. Forcing a technique onto them would fabricate ground truth.

**Live verification** (via `wazuh-logtest`, no manager restart required for this tool since it re-reads the rule file fresh each run):
- wget/curl line → rule `100511`, `mitre.id: ['T1105']`, `mitre.tactic: ['Command and Control']` ✓ (Wazuh's own bundled ATT&CK reference data independently confirms the technique is real and correctly categorized)
- SSH failure (non-invalid-user) → rule `100510`, `mitre.id: ['T1110']` ✓
- "SLOW endpoint" → rule `210013`, `mitre.id: ['T1499.002']`, tactic `Impact` ✓
- "BOT_ATTACK" → rule `210001`, `mitre.id: ['T1498.001']`, tactic `Impact` ✓
- "LOGIN_ATTACK" → rule `210011`, `mitre.id: ['T1110']` ✓
- "INVENTORY_ABUSE" → rule `210012`, no mitre tag (as intended) ✓
- bare `sudo` line → actually resolved to a *more specific native* Wazuh rule (`5403`, already carrying `T1548.003`) before ever reaching `210020` — a real, useful finding: `210020`'s bare-"sudo" match rarely fires in practice for realistic log lines, since native rules with narrower, more specific match conditions win first.
- No collision warnings remain after the fix (previously six).

**Not verified**: the dpkg-suppression fix's live effect (my synthetic test log line didn't match Wazuh's real dpkg decoder format — a test-input problem, not a rule-logic problem; the underlying mechanism, a level=0 rule matching an `if_sid`, is standard Wazuh behavior).
*Planned:* construct a synthetic log line matching Wazuh's real dpkg decoder format and re-verify the suppression fix once one is available.

**Also not done**: fixing rule `100510`'s description ("Multiple Failed Login Attempts") not matching its actual match logic (it has no `frequency`/`timeframe` repetition threshold — fires on any single `if_sid=5716` event) — out of scope for a MITRE-mapping pass; flagged here rather than silently rewritten, since changing detection thresholds is a different kind of change with different risk.
*Planned:* add a `frequency`/`timeframe` repetition threshold to rule `100510` (or correct its description) in a dedicated detection-logic pass.

**A manager restart (`sudo /var/ossec/bin/wazuh-control restart`) is required for the live alert-processing pipeline to load these changes** — `wazuh-logtest` re-reads the file fresh per invocation and was used for validation, but the running `wazuh-analysisd` daemon caches its ruleset in memory until restarted. This was not done by the agent (requires an interactive sudo password); the operator needs to run it themselves for real alerts to reflect these fixes.
*Planned:* have the operator run the manager restart in a maintenance window and confirm the fixes are reflected in live alert processing.

A backup of the pre-fix file and the curated new version are both committed at `backend/wazuh_rules/` for review/history (the live file itself, at `/var/ossec/etc/rules/`, is outside this git repository).
