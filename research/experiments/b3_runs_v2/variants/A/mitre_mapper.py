"""
mitre_mapper.py
==================
Technique-ID -> CYUKTI risk stage. Consumed by realtime_socgraph.py to
compute per-event tps (via config.TPS_MAP), and must therefore only ever
use stage names that are actual TPS_MAP keys — a technique mapped to
anything else silently contributes tps=0, indistinguishable from an
unmapped technique.

Every entry here is grounded in the technique's real ATT&CK tactic(s)
(mitre_import.parser against the actual STIX corpus already imported
into this project — see scripts/build_stage_mapping.py), not inferred
from the technique ID or chosen for a particular class balance.

Several techniques have more than one real ATT&CK tactic (e.g. T1078
Valid Accounts: defense-evasion, persistence, privilege-escalation,
initial-access). Where a technique's tactics include more than one that
has a TPS_MAP entry, the existing pre-audit choice is preserved rather
than silently changed (T1078 keeps "Initial Access", even though
"Privilege Escalation" is also a real, valid tactic for it and carries
a higher weight) — changing it would be a risk-model policy decision,
not a data-correctness fix, and is explicitly out of scope here.

TPS_MAP currently only covers 6 of MITRE's 14 Enterprise tactics
(Reconnaissance, Credential Access, Initial Access, Privilege
Escalation, Lateral Movement, Exfiltration). Four of the techniques
audited here have real ATT&CK tactics with NO corresponding TPS_MAP
entry at all (Discovery, Execution, Collection, Defense Evasion) — per
the audit's explicit instruction not to invent a new TPS_MAP stage
without architectural justification, these are left unmapped and
therefore correctly (not accidentally) fall through to "Unknown":

    T1057      Process Discovery                 -> discovery        (no TPS_MAP entry)
    T1059      Command and Scripting Interpreter  -> execution        (no TPS_MAP entry)
    T1059.007  JavaScript (subtechnique of T1059) -> execution        (no TPS_MAP entry)
    T1114      Email Collection                   -> collection       (no TPS_MAP entry)
    T1562.001  Disable or Modify Tools            -> defense-evasion  (no TPS_MAP entry)

Expanding TPS_MAP's vocabulary to cover these tactics is a legitimate
future risk-model decision; this file does not make that decision.
"""

MITRE_TO_STAGE = {
    "T1595": "Reconnaissance",              # tactics: reconnaissance
    "T1595.002": "Reconnaissance",          # subtechnique of T1595 — same tactic, kept consistent
    "T1110": "Credential Access",           # tactics: credential-access
    "T1110.001": "Credential Access",       # subtechnique of T1110 — same tactic, kept consistent
    "T1053.003": "Privilege Escalation",    # tactics: execution, persistence, privilege-escalation
                                             #   (fixed — was the invalid, non-TPS_MAP string "Cron abuse")
    "T1078": "Initial Access",              # tactics: defense-evasion, persistence, privilege-escalation,
                                             #   initial-access — pre-existing mapping preserved (see module docstring)
    "T1068": "Privilege Escalation",
    "T1021": "Lateral Movement",
    "T1021.004": "Lateral Movement",        # subtechnique of T1021 — same tactic, kept consistent
    "T1041": "Exfiltration",
    "T1055": "Privilege Escalation",        # tactics: defense-evasion, privilege-escalation
    "T1190": "Initial Access",              # tactics: initial-access
    "T1210": "Lateral Movement",            # tactics: lateral-movement
}
