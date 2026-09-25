import { useEffect, useState, type ReactNode } from 'react';
import {
  ShieldAlert, ShieldCheck, ShieldOff, Clock, GitBranch, Target, Bot, BookOpen,
  CheckCircle2, XCircle, HelpCircle, Award, ChevronDown, ChevronUp,
} from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import { EvidenceInvestigation } from './EvidenceInvestigation';
import type {
  IncidentOverview, IncidentResponsePlan, SoarRecommendationsResponse, RagSearchResponse,
} from '../types';

const SEVERITY_COLOR: Record<string, string> = {
  CRITICAL: 'bg-red-500/10 text-red-400 border-red-500/30',
  HIGH: 'bg-red-500/10 text-red-400 border-red-500/30',
  MEDIUM: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  LOW: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/30',
};

const CLASSIFICATION_COLOR: Record<string, string> = {
  QUALIFIED_THREAT: 'bg-red-500/10 text-red-400 border-red-500/30',
  SUSPICIOUS: 'bg-amber-500/10 text-amber-400 border-amber-500/30',
  NOT_THREAT: 'bg-slate-800/50 text-slate-400 border-slate-700',
};

function Section({ title, icon: Icon, children, defaultOpen = true }: {
  title: string; icon: typeof ShieldAlert; children: ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="glass-card overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center gap-2 px-4 py-3 border-b border-[#1e2d4a] hover:bg-white/5 transition-colors"
      >
        <Icon className="w-4 h-4 text-cyan-400" />
        <span className="text-sm font-bold text-white">{title}</span>
        <span className="ml-auto text-slate-500">{open ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}</span>
      </button>
      {open && <div className="p-4">{children}</div>}
    </div>
  );
}

function Badge({ label, className }: { label: string; className: string }) {
  return <span className={`text-xs font-bold px-2.5 py-1 rounded-md border ${className}`}>{label}</span>;
}

// ---------------------------------------------------------------- A. Header

function IncidentHeader({ overview }: { overview: IncidentOverview }) {
  const { campaign, severity, threat_qualification, operation_id } = overview;
  return (
    <div className="glass-card p-4">
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <Badge label={severity.label} className={SEVERITY_COLOR[severity.label] || SEVERITY_COLOR.LOW} />
        <Badge
          label={threat_qualification.classification.replace('_', ' ')}
          className={CLASSIFICATION_COLOR[threat_qualification.classification]}
        />
        <span className="font-mono text-xs text-cyan-400">{campaign.campaign_id}</span>
        {operation_id && <span className="font-mono text-xs text-fuchsia-400">→ {operation_id}</span>}
        <span className="ml-auto text-[11px] text-slate-500 flex items-center gap-1">
          <Clock className="w-3 h-3" /> {campaign.last_seen ? new Date(campaign.last_seen).toLocaleString() : 'unknown time'}
        </span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
        <div>
          <div className="text-slate-500 uppercase text-[10px] mb-0.5">Attacker</div>
          <div className="font-mono text-red-400">{campaign.attacker_ip}</div>
        </div>
        <div>
          <div className="text-slate-500 uppercase text-[10px] mb-0.5">Victim</div>
          <div className="font-mono text-slate-300">{campaign.victim_ip}</div>
        </div>
        <div>
          <div className="text-slate-500 uppercase text-[10px] mb-0.5">MITRE Technique(s)</div>
          <div className="font-mono text-slate-300">{campaign.techniques.join(', ') || 'none resolved'}</div>
        </div>
        <div>
          <div className="text-slate-500 uppercase text-[10px] mb-0.5">Status</div>
          <div className="text-slate-300">{campaign.status}</div>
        </div>
      </div>
      {overview.mitre.length > 0 && (
        <div className="mt-3 pt-3 border-t border-[#1e2d4a] flex flex-wrap gap-2">
          {overview.mitre.map((m) => (
            <div key={m.mitre_id} className="text-[11px] bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1">
              <span className="font-mono text-cyan-400">{m.mitre_id}</span>
              {m.technique_name && <span className="text-slate-400"> — {m.technique_name}</span>}
              {m.tactic.length > 0 && <span className="text-slate-500"> ({m.tactic.join(', ')})</span>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- B. Timeline

function AttackTimeline({ campaignId }: { campaignId: string }) {
  const [events, setEvents] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.campaignTimeline(campaignId)
      .then((r) => setEvents(r.events || []))
      .catch(() => setEvents([]))
      .finally(() => setLoading(false));
  }, [campaignId]);

  if (loading) return <div className="text-xs text-slate-500 animate-pulse">Loading timeline...</div>;
  if (events.length === 0) return <div className="text-xs text-slate-500">No timestamped events recorded for this campaign yet.</div>;

  return (
    <div className="space-y-2">
      {events.map((e, i) => (
        <div key={e.id || i} className="flex items-start gap-3">
          <div className="flex flex-col items-center pt-1">
            <div className="w-2 h-2 rounded-full bg-cyan-400" />
            {i < events.length - 1 && <div className="w-px flex-1 bg-[#1e2d4a] min-h-[20px]" />}
          </div>
          <div className="flex-1 pb-2">
            <div className="text-xs text-slate-300">
              <span className="font-mono text-cyan-400">{e.technique_id}</span>
              {e.technique_name && <span className="text-slate-400"> — {e.technique_name}</span>}
            </div>
            <div className="text-[10px] text-slate-500">
              {e.first_seen ? new Date(e.first_seen).toLocaleString() : 'unknown time'}
              {e.occurrences > 1 && ` · ${e.occurrences} occurrences`}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------- C. Campaign selection tree

function SelectionTree({ selection }: { selection: IncidentOverview['campaign_selection'] }) {
  const top = selection.ranked_candidates.slice(0, 4);
  if (top.length === 0) {
    return <div className="text-xs text-slate-500">No comparable historical campaigns found (no shared technique or attacker identity).</div>;
  }
  return (
    <div>
      <div className="flex flex-col items-center mb-4">
        <div className="glass-card px-4 py-2 border border-cyan-500/30 text-xs font-bold text-cyan-300">CURRENT INCIDENT</div>
        <div className="w-px h-4 bg-[#1e2d4a]" />
        <div className="flex gap-1 w-full justify-center">
          {top.map((_, i) => <div key={i} className="h-px bg-[#1e2d4a]" style={{ width: `${80 / top.length}%` }} />)}
        </div>
      </div>
      <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${top.length}, minmax(0, 1fr))` }}>
        {top.map((c) => {
          const isSelected = c.campaign_id === selection.selected?.campaign_id;
          return (
            <div key={c.campaign_id} className={`glass-card p-2.5 text-center ${isSelected ? 'border border-cyan-500/50' : ''}`}>
              {isSelected && <Award className="w-3.5 h-3.5 text-cyan-400 mx-auto mb-1" />}
              <div className="font-mono text-[11px] text-cyan-400 truncate">{c.campaign_id}</div>
              <div className="text-lg font-mono font-bold text-white">{(c.composite_score * 100).toFixed(0)}%</div>
              {isSelected && <div className="text-[9px] font-bold text-cyan-400 uppercase mt-0.5">Selected</div>}
            </div>
          );
        })}
      </div>
      <p className="text-xs text-slate-400 mt-4 px-1">{selection.explanation}</p>
      {selection.selected && (
        <div className="grid grid-cols-5 gap-1 mt-3">
          {Object.entries(selection.selected.signals).map(([key, value]) => (
            <div key={key} className="text-center">
              <div className="text-[9px] text-slate-500 uppercase">{key.replace('_similarity', '')}</div>
              <div className="text-[11px] font-mono text-slate-300">{value === null ? '—' : `${(value * 100).toFixed(0)}%`}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- D. Threat qualification checklist

function ThreatQualificationChecklist({ tq }: { tq: IncidentOverview['threat_qualification'] }) {
  return (
    <div className="space-y-2">
      <div className="text-xs text-slate-400">{tq.reason}</div>
      {tq.checks.map((check) => {
        const isUnknownCase = check.name === 'threat_classification_qualified' && tq.cti_score === null;
        const Icon = isUnknownCase ? HelpCircle : check.passed ? CheckCircle2 : XCircle;
        const color = isUnknownCase ? 'text-slate-500' : check.passed ? 'text-emerald-400' : 'text-red-400';
        const label = isUnknownCase ? 'UNKNOWN' : check.passed ? 'PASS' : 'FAIL';
        return (
          <div key={check.name} className="flex items-center gap-2 text-xs bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2.5 py-1.5">
            <Icon className={`w-3.5 h-3.5 shrink-0 ${color}`} />
            <span className="text-slate-300 flex-1">{check.name.replace(/_/g, ' ')}</span>
            <span className="text-slate-500 text-[11px] flex-[2] truncate" title={check.detail}>{check.detail}</span>
            <span className={`font-bold ${color}`}>{label}</span>
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- F. Multi-RAG

function MultiRagSection({ campaignId, query }: { campaignId: string; query: string | null }) {
  const [data, setData] = useState<RagSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.ragSearch({ campaignId, query: query || undefined, topK: 3 })
      .then(setData)
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [campaignId, query]);

  if (loading) return <div className="text-xs text-slate-500 animate-pulse">Querying MITRE, historical campaigns, and GNN topology...</div>;
  if (!data) return <div className="text-xs text-slate-500">Multi-RAG query unavailable.</div>;

  return (
    <div className="grid md:grid-cols-3 gap-3">
      {Object.entries(data.sources).map(([source, result]) => {
        const items = Array.isArray(result) ? result : (result as any).results || [];
        const unavailable = !Array.isArray(result) && (result as any).gnn_available === false;
        return (
          <div key={source} className="space-y-1.5">
            <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">{source.replace(/_/g, ' ')}</div>
            {unavailable ? (
              <div className="text-[11px] text-slate-500">GNN disabled — no topology results.</div>
            ) : items.length === 0 ? (
              <div className="text-[11px] text-slate-500">No results.</div>
            ) : (
              items.slice(0, 3).map((item: any, i: number) => (
                <div key={i} className="text-[11px] bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1.5">
                  <div className="text-slate-300 truncate">{item.source_id}</div>
                  <div className="text-slate-500">relevance {(item.relevance * 100).toFixed(0)}% · {item.provenance}</div>
                </div>
              ))
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- G. Response plan

function ResponsePlanSection({ campaignId }: { campaignId: string }) {
  const [plan, setPlan] = useState<IncidentResponsePlan | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.incidentResponsePlan(campaignId).then(setPlan).catch(() => setPlan(null)).finally(() => setLoading(false));
  }, [campaignId]);

  if (loading) return <div className="text-xs text-slate-500 animate-pulse">Generating response plan...</div>;
  if (!plan) return <div className="text-xs text-slate-500">Response plan unavailable.</div>;

  return (
    <div className="space-y-3">
      <p className="text-xs text-slate-300">{plan.threat_summary}</p>
      <div className="space-y-1">
        {plan.why_threat.map((w, i) => <div key={i} className="text-[11px] text-slate-500">{w}</div>)}
      </div>
      {plan.selected_historical_campaign && (
        <div className="text-[11px] text-slate-400 border-l-2 border-cyan-500/40 pl-2">
          Historical match: <span className="font-mono text-cyan-400">{plan.selected_historical_campaign}</span> ({plan.selection_confidence} confidence) — {plan.why_selected}
        </div>
      )}
      <div className="flex items-center gap-2">
        <span className="text-[10px] text-slate-500 uppercase">MISP:</span>
        <Badge
          label={plan.misp_status}
          className={plan.misp_status === 'READY' ? SEVERITY_COLOR.LOW : plan.misp_status === 'BLOCKED' ? SEVERITY_COLOR.HIGH : CLASSIFICATION_COLOR.NOT_THREAT}
        />
        <span className="text-[11px] text-slate-500">{plan.misp_reason}</span>
      </div>
      {plan.playbook && (
        <div className="space-y-1.5 pt-2 border-t border-[#1e2d4a]">
          {plan.playbook.actions.map((a) => (
            <div key={a.action_id} className="flex items-center justify-between text-[11px] bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1.5">
              <span className="text-slate-300">{a.order}. {a.name}</span>
              <div className="flex gap-1.5 shrink-0">
                <Badge label={a.destructive ? 'DESTRUCTIVE' : 'SAFE'} className={a.destructive ? SEVERITY_COLOR.HIGH : SEVERITY_COLOR.LOW} />
                <Badge label={a.requires_approval ? 'APPROVAL' : 'AUTO'} className={a.requires_approval ? SEVERITY_COLOR.MEDIUM : CLASSIFICATION_COLOR.NOT_THREAT} />
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- H/I. Shuffle + Playbook memory

function ShuffleAndMemorySection({ overview, campaignId }: { overview: IncidentOverview; campaignId: string }) {
  const [recommendations, setRecommendations] = useState<SoarRecommendationsResponse | null>(null);

  useEffect(() => {
    api.soarRecommendations(campaignId).then(setRecommendations).catch(() => setRecommendations(null));
  }, [campaignId]);

  const executions = overview.soar.executions;
  const activeExecution = executions.find((e) => e.status === 'running' || e.status === 'pending_approval');

  return (
    <div className="space-y-4">
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">Shuffle Execution</div>
        {activeExecution ? (
          <div className="text-xs text-slate-300">
            Execution <span className="font-mono text-cyan-400">{activeExecution.execution_id}</span> — status {activeExecution.status}
          </div>
        ) : executions.length > 0 ? (
          <div className="text-xs text-slate-400">{executions.length} past execution(s) recorded for this campaign.</div>
        ) : (
          <div className="text-xs text-slate-500">Execution unavailable — recommendation generated locally.</div>
        )}
      </div>
      <div>
        <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400 mb-1.5">How CYUKTI will handle similar incidents in the future</div>
        {!recommendations || recommendations.historical_matches.length === 0 ? (
          <div className="text-xs text-slate-500">No previous playbook found. CYUKTI generated a new recommendation.</div>
        ) : (
          recommendations.historical_matches.slice(0, 2).map((m) => (
            <div key={m.playbook_id} className="text-[11px] bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1.5 mb-1.5">
              <div className="text-slate-300">{m.playbook_name} <span className="text-slate-500">from {m.source_campaign_id}</span></div>
              <div className="text-slate-500">{m.recommendation_reason}</div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Main page

export function IncidentView() {
  const { campaigns, selectedCampaign, selectCampaign } = useDashboard();
  const [overview, setOverview] = useState<IncidentOverview | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedCampaign) {
      setOverview(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.incidentOverview(selectedCampaign)
      .then(setOverview)
      .catch((err) => setError(err.message || 'Failed to load incident overview'))
      .finally(() => setLoading(false));
  }, [selectedCampaign]);

  const ragQuery = overview?.mitre[0]?.technique_name || null;

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-cyan-500/10 border border-cyan-500/30">
            <ShieldAlert className="w-6 h-6 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Incident View</h1>
            <p className="text-xs text-slate-500 mt-0.5">Detect → understand → correlate → qualify → investigate → respond, in one place.</p>
          </div>
        </div>

        <div className="glass-card p-3 flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider shrink-0">Incident</label>
          <select
            value={selectedCampaign || ''}
            onChange={(e) => selectCampaign(e.target.value || null)}
            className="flex-1 bg-[#0a0e17] text-sm text-slate-200 font-mono border border-[#1e2d4a] rounded px-2 py-1.5 outline-none focus:border-cyan-500"
          >
            <option value="">Select an incident/campaign...</option>
            {campaigns.map((c) => (
              <option key={c.campaign_id} value={c.campaign_id}>{c.campaign_label || c.campaign_id}</option>
            ))}
          </select>
        </div>

        {!selectedCampaign ? (
          <div className="glass-card p-8 text-center text-slate-500 text-sm">Select an incident above to see its full story.</div>
        ) : loading ? (
          <div className="glass-card p-8 text-center text-cyan-400 text-sm animate-pulse">Loading incident overview...</div>
        ) : error ? (
          <div className="glass-card p-8 text-center text-red-400 text-sm">{error}</div>
        ) : overview ? (
          <>
            <IncidentHeader overview={overview} />
            <Section title="Attack Story / Timeline" icon={Clock}>
              <AttackTimeline campaignId={selectedCampaign} />
            </Section>
            <Section title="Campaign Selection" icon={GitBranch}>
              <SelectionTree selection={overview.campaign_selection} />
            </Section>
            <Section title="Threat Qualification" icon={ShieldCheck}>
              <ThreatQualificationChecklist tq={overview.threat_qualification} />
            </Section>
            <Section title="Investigation / Evidence" icon={Target} defaultOpen={false}>
              <EvidenceInvestigation />
            </Section>
            <Section title="Multi-RAG" icon={BookOpen} defaultOpen={false}>
              <MultiRagSection campaignId={selectedCampaign} query={ragQuery} />
            </Section>
            <Section title="Response Plan" icon={ShieldOff}>
              <ResponsePlanSection campaignId={selectedCampaign} />
            </Section>
            <Section title="Shuffle & Playbook Memory" icon={Bot}>
              <ShuffleAndMemorySection overview={overview} campaignId={selectedCampaign} />
            </Section>
          </>
        ) : null}
      </div>
    </div>
  );
}
