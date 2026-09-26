import { useEffect, useState, type ReactNode } from 'react';
import {
  ShieldAlert, Clock, GitBranch, X,
  CheckCircle2, XCircle, HelpCircle, Award, ChevronRight, ChevronDown,
  Radar, Fingerprint, Layers, SearchCheck, Gauge, Send, FileJson,
} from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import { EvidenceInvestigation } from './EvidenceInvestigation';
import { CampaignId } from './CampaignId';
import type {
  IncidentOverview, IncidentResponsePlan, SoarRecommendationsResponse, RagSearchResponse,
  CampaignCandidate,
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

const CLASSIFICATION_LABEL: Record<string, string> = {
  QUALIFIED_THREAT: 'QUALIFIED THREAT',
  SUSPICIOUS: 'SUSPICIOUS ACTIVITY',
  NOT_THREAT: 'DETECTED ACTIVITY',
};

function Badge({ label, className }: { label: string; className: string }) {
  return <span className={`text-[11px] font-bold px-2.5 py-1 rounded-md border whitespace-nowrap ${className}`}>{label}</span>;
}

// ---------------------------------------------------------------- shared: drawer

function Drawer({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: ReactNode }) {
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-[100] flex justify-end" role="dialog" aria-modal="true" aria-label={title}>
      <div className="absolute inset-0 bg-black/60" onClick={onClose} />
      <div className="relative w-full max-w-2xl h-full bg-[#0a0e17] border-l border-[#1e2d4a] overflow-y-auto custom-scrollbar shadow-2xl animate-slide-in">
        <div className="sticky top-0 flex items-center justify-between px-4 py-3 border-b border-[#1e2d4a] bg-[#0a0e17]/95 backdrop-blur z-10">
          <h2 className="text-sm font-bold text-white">{title}</h2>
          <button onClick={onClose} aria-label="Close" className="text-slate-500 hover:text-white p-1 rounded hover:bg-white/5">
            <X className="w-4 h-4" />
          </button>
        </div>
        <div className="p-4">{children}</div>
      </div>
    </div>
  );
}

function ExpandableTechnicalDetails({ data }: { data: unknown }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-2">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[10px] text-slate-500 hover:text-slate-300 uppercase tracking-wider font-semibold"
      >
        {open ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />} <FileJson className="w-3 h-3" /> Technical details
      </button>
      {open && (
        <pre className="mt-1.5 text-[10px] text-slate-400 bg-[#060a13] border border-[#1e2d4a] rounded p-2 overflow-x-auto max-h-64 overflow-y-auto">
          {JSON.stringify(data, null, 2)}
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- A. Compact header

function IncidentHeader({ overview }: { overview: IncidentOverview }) {
  const { campaign, severity, threat_qualification, operation_id } = overview;
  const primaryTechnique = overview.mitre[0];
  return (
    <div className="glass-card p-4">
      {/* PRIMARY: the canonical campaign ID, then concise state/severity right below it */}
      <CampaignId id={campaign.campaign_id} size="lg" className="block mb-1.5" />
      <div className="flex flex-wrap items-center gap-2 mb-3">
        <Badge label={`${severity.label} SEVERITY`} className={SEVERITY_COLOR[severity.label] || SEVERITY_COLOR.LOW} />
        <Badge label={CLASSIFICATION_LABEL[threat_qualification.classification]} className={CLASSIFICATION_COLOR[threat_qualification.classification]} />
        {threat_qualification.cti_score !== null && (
          <span className="text-[11px] text-slate-400">Confidence {threat_qualification.cti_score.toFixed(0)}%</span>
        )}
        {primaryTechnique && (
          <span className="font-mono text-[11px] text-cyan-400 ml-auto">{primaryTechnique.mitre_id}</span>
        )}
      </div>

      {/* SECONDARY: what it is */}
      <h1 className="text-base font-semibold text-white mb-2">
        {primaryTechnique?.technique_name || campaign.last_technique || 'Unclassified Activity'}
      </h1>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-slate-400">
        <span className="font-mono text-red-400">{campaign.attacker_ip}</span>
        <ChevronRight className="w-3 h-3 text-slate-600" />
        <span className="font-mono text-slate-300">{campaign.victim_ip}</span>
        <span className="flex items-center gap-1 text-slate-500">
          <Clock className="w-3 h-3" /> {campaign.last_seen ? new Date(campaign.last_seen).toLocaleString() : 'unknown time'}
        </span>
        {operation_id && <span className="font-mono text-fuchsia-400">Operation {operation_id}</span>}
      </div>

      {/* TERTIARY: raw/technical details, collapsed by default */}
      <ExpandableTechnicalDetails data={{ campaign_id: campaign.campaign_id, operation_id, mitre: overview.mitre, risk_score: campaign.risk_score }} />
    </div>
  );
}

// ---------------------------------------------------------------- B. Attack story timeline

const STORY_STEPS = [
  { id: 'detected', label: 'Detected', icon: Radar },
  { id: 'network', label: 'Network Activity', icon: Layers },
  { id: 'mitre', label: 'MITRE', icon: Fingerprint },
  { id: 'campaign', label: 'Campaign', icon: GitBranch },
  { id: 'investigation', label: 'Investigation', icon: SearchCheck },
  { id: 'assessment', label: 'Assessment', icon: Gauge },
  { id: 'response', label: 'Response', icon: Send },
] as const;

function AttackStoryTimeline({
  overview, campaignId, onStepClick,
}: { overview: IncidentOverview; campaignId: string; onStepClick: (step: string) => void }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [events, setEvents] = useState<any[]>([]);

  useEffect(() => {
    api.campaignTimeline(campaignId).then((r) => setEvents(r.events || [])).catch(() => setEvents([]));
  }, [campaignId]);

  const handleClick = (id: string) => {
    setExpanded(expanded === id ? null : id);
    onStepClick(id);
  };

  const detailFor = (id: string): ReactNode => {
    switch (id) {
      case 'detected':
        return <p>First observed {overview.campaign.first_seen ? new Date(overview.campaign.first_seen).toLocaleString() : 'at an unrecorded time'}.</p>;
      case 'network':
        return events.length === 0
          ? <p>No timestamped network events recorded.</p>
          : <div className="space-y-1">{events.map((e, i) => (
              <div key={e.id || i}>
                <span className="font-mono text-cyan-400">{e.technique_id || '—'}</span> {e.technique_name && `— ${e.technique_name}`}
                <span className="text-slate-500"> · {e.first_seen ? new Date(e.first_seen).toLocaleTimeString() : ''}</span>
              </div>
            ))}</div>;
      case 'mitre':
        return overview.mitre.length === 0
          ? <p>No defensible MITRE mapping for this activity.</p>
          : <div className="space-y-1">{overview.mitre.map((m) => (
              <div key={m.mitre_id}>
                <span className="font-mono text-cyan-400">{m.mitre_id}</span> — {m.technique_name || 'unnamed'} ({m.tactic.join(', ') || 'tactic unknown'})
              </div>
            ))}</div>;
      case 'campaign':
        return <p>Correlated into campaign <span className="font-mono text-cyan-400">{overview.campaign.campaign_id}</span>{overview.operation_id && <> under operation <span className="font-mono text-fuchsia-400">{overview.operation_id}</span></>}.</p>;
      case 'investigation':
        return <p>Open the Investigation panel below to run evidence-aware investigation for this incident.</p>;
      case 'assessment':
        return <p>{overview.threat_qualification.reason}</p>;
      case 'response':
        return <p>See the Response section below for the recommended playbook and next actions.</p>;
      default:
        return null;
    }
  };

  return (
    <div className="glass-card p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Attack Story</h2>
      <div className="flex items-center overflow-x-auto custom-scrollbar pb-1">
        {STORY_STEPS.map((step, i) => {
          const Icon = step.icon;
          const isOpen = expanded === step.id;
          return (
            <div key={step.id} className="flex items-center shrink-0">
              <button
                onClick={() => handleClick(step.id)}
                className={`flex flex-col items-center gap-1 px-3 py-1.5 rounded-md transition-colors ${
                  isOpen ? 'bg-cyan-500/10 text-cyan-300' : 'text-slate-400 hover:text-slate-200 hover:bg-white/5'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-[10px] font-semibold whitespace-nowrap">{step.label}</span>
              </button>
              {i < STORY_STEPS.length - 1 && <ChevronRight className="w-3.5 h-3.5 text-slate-700 shrink-0" />}
            </div>
          );
        })}
      </div>
      {expanded && (
        <div className="mt-3 pt-3 border-t border-[#1e2d4a] text-xs text-slate-300">
          {detailFor(expanded)}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- C. Current Assessment (dominant panel)

function CurrentAssessment({ overview }: { overview: IncidentOverview }) {
  const tq = overview.threat_qualification;
  const narrative = tq.classification === 'QUALIFIED_THREAT'
    ? 'Current evidence is sufficient to treat this as a qualified threat.'
    : tq.classification === 'SUSPICIOUS'
    ? 'Suspicious activity was detected, but current evidence does not establish a qualified threat.'
    : 'Activity was detected; current evidence does not indicate malicious intent.';

  return (
    <div className="glass-card p-5 border-2 border-[#1e2d4a]">
      <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Current Assessment</h2>
      <div className={`inline-block text-sm font-bold px-3 py-1 rounded-md border mb-3 ${CLASSIFICATION_COLOR[tq.classification]}`}>
        {CLASSIFICATION_LABEL[tq.classification]}
      </div>
      <p className="text-sm text-slate-300 mb-4">{narrative}</p>
      <div className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1.5">Why?</div>
      <div className="space-y-1 mb-4">
        {tq.checks.map((check) => {
          const isUnknown = check.name === 'threat_classification_qualified' && tq.cti_score === null;
          const Icon = isUnknown ? HelpCircle : check.passed ? CheckCircle2 : XCircle;
          const color = isUnknown ? 'text-slate-500' : check.passed ? 'text-emerald-400' : 'text-red-400';
          const label = isUnknown ? 'UNKNOWN' : check.passed ? 'PASS' : 'FAIL';
          return (
            <div key={check.name} className="flex items-start gap-2 text-xs">
              <Icon className={`w-3.5 h-3.5 shrink-0 mt-0.5 ${color}`} />
              <span className="text-slate-300 flex-1">{check.detail}</span>
              <span className={`font-bold shrink-0 ${color}`}>{label}</span>
            </div>
          );
        })}
      </div>
      <div className="flex items-center gap-2 pt-3 border-t border-[#1e2d4a]">
        <span className="text-[10px] text-slate-500 uppercase font-semibold">MISP publication:</span>
        <Badge
          label={tq.may_publish_to_misp ? 'READY' : 'BLOCKED'}
          className={tq.may_publish_to_misp ? SEVERITY_COLOR.LOW : SEVERITY_COLOR.HIGH}
        />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Evidence summary + drawer trigger

function EvidenceSummaryCard({ campaignId }: { campaignId: string }) {
  const [drawerOpen, setDrawerOpen] = useState(false);
  return (
    <div className="glass-card p-4 flex items-center justify-between">
      <div>
        <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-1">Investigation &amp; Evidence</h2>
        <p className="text-xs text-slate-500">Run evidence-aware investigation and inspect collected evidence by source.</p>
      </div>
      <button
        onClick={() => setDrawerOpen(true)}
        className="flex items-center gap-1.5 text-xs font-semibold px-3 py-1.5 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/30 hover:bg-indigo-500/20 shrink-0"
      >
        <SearchCheck className="w-3.5 h-3.5" /> Open Investigation
      </button>
      <Drawer open={drawerOpen} onClose={() => setDrawerOpen(false)} title={`Investigation — ${campaignId}`}>
        <EvidenceInvestigation />
      </Drawer>
    </div>
  );
}

// ---------------------------------------------------------------- C-tree. Campaign selection with compare panel

function CandidateCompare({ candidate, onClose }: { candidate: CampaignCandidate; onClose: () => void }) {
  return (
    <div className="glass-card p-3 mt-3 border border-cyan-500/30">
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-xs text-cyan-400">{candidate.campaign_id}</span>
        <button onClick={onClose} className="text-slate-500 hover:text-white"><X className="w-3.5 h-3.5" /></button>
      </div>
      <div className="grid grid-cols-5 gap-2">
        {Object.entries(candidate.signals).map(([key, value]) => (
          <div key={key} className="text-center">
            <div className="text-[9px] text-slate-500 uppercase mb-1">{key.replace('_similarity', '')}</div>
            <div className="text-sm font-mono font-bold text-slate-200">{value === null ? '—' : `${(value * 100).toFixed(0)}%`}</div>
          </div>
        ))}
      </div>
      <p className="text-[10px] text-slate-500 mt-2">Values are similarity signals, not probabilities.</p>
    </div>
  );
}

function SelectionTree({ selection }: { selection: IncidentOverview['campaign_selection'] }) {
  const [compareId, setCompareId] = useState<string | null>(null);
  const top = selection.ranked_candidates.slice(0, 4);
  if (top.length === 0) {
    return <div className="text-xs text-slate-500">No comparable historical campaigns found (no shared technique or attacker identity).</div>;
  }
  const compareCandidate = top.find((c) => c.campaign_id === compareId) || null;

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
            <button
              key={c.campaign_id}
              onClick={() => setCompareId(compareId === c.campaign_id ? null : c.campaign_id)}
              className={`glass-card p-2.5 text-center transition-colors hover:border-cyan-500/40 border ${isSelected ? 'border-cyan-500/50' : 'border-transparent'}`}
            >
              {isSelected && <Award className="w-3.5 h-3.5 text-cyan-400 mx-auto mb-1" />}
              <div className="font-mono text-[11px] text-cyan-400 truncate">{c.campaign_id}</div>
              <div className="text-lg font-mono font-bold text-white">{(c.composite_score * 100).toFixed(0)}%</div>
              {isSelected && <div className="text-[9px] font-bold text-cyan-400 uppercase mt-0.5">Selected</div>}
            </button>
          );
        })}
      </div>
      <p className="text-xs text-slate-400 mt-4 px-1">
        <span className="font-semibold text-slate-300">Why this campaign? </span>{selection.explanation}
      </p>
      {compareCandidate && <CandidateCompare candidate={compareCandidate} onClose={() => setCompareId(null)} />}
      {!compareCandidate && <p className="text-[10px] text-slate-600 mt-2">Click a candidate to compare its individual signals.</p>}
    </div>
  );
}

// ---------------------------------------------------------------- F. Multi-RAG (compact)

function MultiRagSection({ campaignId, query }: { campaignId: string; query: string | null }) {
  const [data, setData] = useState<RagSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.ragSearch({ campaignId, query: query || undefined, topK: 3 })
      .then(setData).catch(() => setData(null)).finally(() => setLoading(false));
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
              items.slice(0, 2).map((item: any, i: number) => (
                <div key={i} className="text-[11px] bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1.5">
                  <div className="text-slate-300 truncate">{item.source_id}</div>
                  <div className="text-slate-500">relevance {(item.relevance * 100).toFixed(0)}%</div>
                </div>
              ))
            )}
          </div>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------- G/H/I. Response, Shuffle, MISP, Memory

function ResponseSection({ campaignId }: { campaignId: string }) {
  const [plan, setPlan] = useState<IncidentResponsePlan | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    api.incidentResponsePlan(campaignId).then(setPlan).catch(() => setPlan(null)).finally(() => setLoading(false));
  }, [campaignId]);

  if (loading) return <div className="glass-card p-4 text-xs text-slate-500 animate-pulse">Generating response plan...</div>;
  if (!plan) return <div className="glass-card p-4 text-xs text-slate-500">Response plan unavailable.</div>;

  return (
    <div className="glass-card p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Recommended Response</h2>
      {plan.selected_historical_campaign && (
        <div className="text-[11px] text-slate-400 border-l-2 border-cyan-500/40 pl-2 mb-3">
          Based on historical match <span className="font-mono text-cyan-400">{plan.selected_historical_campaign}</span> ({plan.selection_confidence} confidence)
        </div>
      )}
      {plan.playbook ? (
        <div className="space-y-1.5">
          {plan.playbook.actions.map((a) => (
            <div key={a.action_id} className="flex items-center justify-between text-xs bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2.5 py-2">
              <span className="text-slate-300">{a.order}. {a.name}</span>
              <div className="flex gap-1.5 shrink-0">
                <Badge label={a.destructive ? 'DESTRUCTIVE' : 'SAFE'} className={a.destructive ? SEVERITY_COLOR.HIGH : SEVERITY_COLOR.LOW} />
                <Badge label={a.requires_approval ? 'APPROVAL' : 'AUTOMATIC'} className={a.requires_approval ? SEVERITY_COLOR.MEDIUM : CLASSIFICATION_COLOR.NOT_THREAT} />
              </div>
            </div>
          ))}
        </div>
      ) : (
        <p className="text-xs text-slate-500">No playbook generated for this incident.</p>
      )}
    </div>
  );
}

function ShuffleStatus({ overview }: { overview: IncidentOverview }) {
  const executions = overview.soar.executions;
  const active = executions.find((e) => e.status === 'running' || e.status === 'pending_approval');
  const playbook = overview.soar.playbooks[0];

  return (
    <div className="glass-card p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Response Execution</h2>
      <div className="text-xs text-slate-300 mb-1">Playbook: <span className="text-slate-400">{playbook?.name || 'Not yet generated'}</span></div>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-slate-300">Status:</span>
        <Badge
          label={active ? active.status.replace('_', ' ').toUpperCase() : executions.length > 0 ? 'COMPLETED' : 'RECOMMENDED'}
          className={active ? SEVERITY_COLOR.MEDIUM : CLASSIFICATION_COLOR.NOT_THREAT}
        />
      </div>
      <div className="text-xs text-slate-500">Actions: {playbook?.actions.length ?? '—'}</div>
      {executions.length === 0 && (
        <p className="text-[11px] text-slate-500 mt-2">Execution unavailable — recommendation remains available locally.</p>
      )}
    </div>
  );
}

function MispStatus({ overview }: { overview: IncidentOverview }) {
  const tq = overview.threat_qualification;
  return (
    <div className="glass-card p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">Threat Intelligence</h2>
      <div className="flex items-center gap-2 mb-1">
        <span className="text-xs text-slate-300">MISP publication:</span>
        <Badge label={tq.may_publish_to_misp ? 'READY' : 'BLOCKED'} className={tq.may_publish_to_misp ? SEVERITY_COLOR.LOW : SEVERITY_COLOR.HIGH} />
      </div>
      <p className="text-[11px] text-slate-500">{tq.reason}</p>
    </div>
  );
}

function PlaybookMemorySection({ campaignId }: { campaignId: string }) {
  const [recommendations, setRecommendations] = useState<SoarRecommendationsResponse | null>(null);

  useEffect(() => {
    api.soarRecommendations(campaignId).then(setRecommendations).catch(() => setRecommendations(null));
  }, [campaignId]);

  return (
    <div className="glass-card p-4">
      <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-2">How CYUKTI Handles This Next Time</h2>
      {!recommendations || recommendations.historical_matches.length === 0 ? (
        <p className="text-xs text-slate-500">No previous playbook found. CYUKTI generated a new recommendation.</p>
      ) : (
        recommendations.historical_matches.slice(0, 2).map((m) => (
          <div key={m.playbook_id} className="text-[11px] bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1.5 mb-1.5">
            <div className="text-slate-300">{m.playbook_name} <span className="text-slate-500">from {m.source_campaign_id}</span></div>
            <div className="text-slate-500">{m.recommendation_reason}</div>
          </div>
        ))
      )}
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
            <h1 className="text-xl font-bold text-white">Incidents</h1>
            <p className="text-xs text-slate-500 mt-0.5">What happened, why it matters, and what to do next — in one place.</p>
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
              <option key={c.campaign_id} value={c.campaign_id}>{c.campaign_id}</option>
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
            <AttackStoryTimeline overview={overview} campaignId={selectedCampaign} onStepClick={() => {}} />
            <CurrentAssessment overview={overview} />
            <EvidenceSummaryCard campaignId={selectedCampaign} />

            <div className="glass-card p-4">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
                <GitBranch className="w-3.5 h-3.5" /> Campaign Selection
              </h2>
              <SelectionTree selection={overview.campaign_selection} />
            </div>

            <div className="glass-card p-4">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 mb-3">Multi-RAG</h2>
              <MultiRagSection campaignId={selectedCampaign} query={ragQuery} />
            </div>

            <ResponseSection campaignId={selectedCampaign} />

            <div className="grid md:grid-cols-2 gap-4">
              <ShuffleStatus overview={overview} />
              <MispStatus overview={overview} />
            </div>

            <PlaybookMemorySection campaignId={selectedCampaign} />
          </>
        ) : null}
      </div>
    </div>
  );
}
