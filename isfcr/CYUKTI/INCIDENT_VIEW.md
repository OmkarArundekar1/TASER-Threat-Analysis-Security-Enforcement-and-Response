# CYUKTI Incident View

## What this is

The single, primary analyst-facing page (`frontend/src/components/IncidentView.tsx`, reachable via the "Incidents" top-nav button) that answers the full chain in one place: *what happened, why it matters, what CYUKTI thinks it matches historically, whether it's a real threat, what the evidence says, and what should happen next.*

## UX redesign (this phase)

The original version (2 phases ago) rendered all 9 sections permanently, full-size, simultaneously — functionally complete but cluttered. This phase restructured it around an explicit information hierarchy (Level 1 "what happened" → Level 6 "what happened after the response"), collapsing secondary detail behind clicks rather than showing it all at once:

- **Compact header**: one line of status badges (threat classification, severity, confidence %, MITRE ID) instead of a full data table. Raw IDs/JSON moved behind a "Technical details" expandable, not shown by default.
- **Attack Story timeline**: `DETECTED → NETWORK ACTIVITY → MITRE → CAMPAIGN → INVESTIGATION → ASSESSMENT → RESPONSE` as clickable nodes; clicking a node expands only that step's real detail inline, instead of one giant card per step.
- **Current Assessment**: a new, visually dominant panel — the classification, a plain-language narrative, and the six qualification checks with PASS/FAIL/**UNKNOWN** + the real reason text, generated entirely from `overview.threat_qualification` (never hardcoded — verified against three different real campaigns with three different classifications during this phase's investigation, see below).
- **Evidence Drawer**: investigation/evidence is no longer permanently rendered. An "Open Investigation" button opens a slide-over drawer containing the existing `<EvidenceInvestigation />` component (reused verbatim, not duplicated) — closes on backdrop click or the X button.
- **Campaign Selection**: same tree (current incident → top 4 candidates → selected), now with a click-to-compare interaction — clicking any candidate (selected or not) opens its full five-signal breakdown, explicitly labeled "similarity signals, not probabilities."
- **Response / Shuffle / MISP**: condensed into compact, single-purpose cards (playbook name + status + action count; MISP publication status + one-line reason) instead of large panels, per-action SAFE/DESTRUCTIVE and AUTOMATIC/APPROVAL badges kept since they're safety-critical.
- **Playbook Memory**: unchanged content ("No previous playbook found..." honest empty state), same compact card treatment.

## Navigation simplification (this phase)

`TopNavBar.tsx` reduced from 8 always-visible top-level items to 4 primary ones (**Overview, Incidents, Response, Threat Intel**) plus a **More ▾** dropdown (GNN Intelligence, Prediction, System Health, Audit Log) — click-outside-to-close, keyboard-reachable buttons. Engineering/research surfaces are still one click away, just no longer competing for top-level attention.

## Real-world validation: the Nmap scan investigation (this phase)

Before touching any UI, the actual Wazuh alert from a real `nmap` scan (Kali `192.168.56.106` → target `192.168.56.105`) was inspected end-to-end, per this phase's explicit instruction not to change the threat model to make an attack "look" more or less malicious than the evidence supports:

1. **Wazuh detected it**: Suricata's `ET SCAN Possible Nmap User-Agent Observed` signature fired on the Nmap Scripting Engine's distinctive HTTP User-Agent string (`Mozilla/5.0 (compatible; Nmap Scripting Engine...)`), 25 times.
2. **A real MITRE ID was attached**: Wazuh custom rule `100500` (already existing, native-tier, correctly configured since before this session) tags it `T1595` / Active Scanning / Reconnaissance — genuinely defensible, not fabricated this phase.
3. **CYUKTI ingested it**: confirmed live in Neo4j — real `AttackEvent` nodes with `attack_id: T1595`, `first_seen` matching the scan's real timestamp to the second.
4. **It became real campaigns**: e.g. `CAMP_1B8E9033` (attacker `192.168.56.106`, victim `192.168.56.105`, technique `T1595`).
5. **Threat qualification was based on real evidence, not absence of evidence** — and here is the important correction to this phase's own starting assumption: `GET /api/incidents/CAMP_1B8E9033/overview` shows this campaign is **already classified `QUALIFIED_THREAT`** (CTI confidence score 49.08, `may_publish_to_misp: true`), not "non-qualified" as originally described. Severity is separately `LOW` (raw TPS 250 of a 1500 ceiling) — **both are correct and both are real**: this is genuinely low-damage-potential reconnaissance that CYUKTI is nonetheless confidently classifying as real threat activity, which is exactly the kind of nuance a good SOC tool should surface rather than collapse into one number. The most likely explanation for the original observation: the *old* dashboard pages (`SecurityOverview`, `CampaignIntelligence`) prominently show only the severity/risk badge and never surfaced `threat_qualification.classification` anywhere — so a `LOW`-severity badge alone reasonably reads as "not flagged," even though a separate, correct `QUALIFIED_THREAT` verdict existed the whole time. The redesigned Incident View's header now shows **both** badges side by side for exactly this reason.

No MITRE ID was invented or adjusted to make this finding — `T1595` was already correctly attached before this session began; this phase only *verified and surfaced* it correctly.

Other rules fired for the same scan (`31101`/`31121` web-server error codes, `86601` protocol-detect anomaly) correctly carry **no** MITRE mapping — matching text like an HTTP 400 or a one-directional protocol detection has no defensible, specific ATT&CK technique on its own, and none was assigned.

## Tests

`IncidentView.test.tsx` (8, covering the redesigned header, Current Assessment, UNKNOWN-state handling, campaign-selection compare panel, evidence drawer open/close, and attack-story expansion), `test_dashboard_api_incident_views.py` (7, overview + response-plan, unchanged this phase — no backend logic was modified).
