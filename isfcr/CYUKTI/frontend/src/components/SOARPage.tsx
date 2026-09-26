import { useEffect, useState } from 'react';
import {
  Bot, ShieldCheck, ShieldAlert, ShieldOff, BookOpen, Activity, History as HistoryIcon,
  BarChart3, CheckCircle2, XCircle, RefreshCw, Sparkles, Radio, AlertTriangle,
} from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import type {
  HistoricalPlaybookMatch, Playbook, PlaybookAction, PlaybookEffectiveness,
  PlaybookExecution, SoarStatus, ResponseStateResponse,
} from '../types';

type SoarTab = 'recommendations' | 'library' | 'active' | 'history' | 'effectiveness' | 'response';

const TABS: { id: SoarTab; label: string; icon: typeof BookOpen }[] = [
  { id: 'recommendations', label: 'Recommendations', icon: Sparkles },
  { id: 'library', label: 'Library', icon: BookOpen },
  { id: 'active', label: 'Active Executions', icon: Activity },
  { id: 'history', label: 'History', icon: HistoryIcon },
  { id: 'effectiveness', label: 'Effectiveness', icon: BarChart3 },
  { id: 'response', label: 'Active Response', icon: Radio },
];

// Every state the backend's ResponseState enum / audit event types can
// report, mapped to a color -- deliberately NO color/label implies
// "blocked" or "safe" unless the state is exactly VERIFIED. An
// unrecognized or absent state renders as neutral gray, never green.
const RESPONSE_STATE_STYLE: Record<string, { label: string; className: string }> = {
  NO_RESPONSE_ACTIVITY: { label: 'No response activity', className: 'text-slate-500 border-slate-700 bg-slate-800/30' },
  CONTAINMENT_REQUESTED: { label: 'Containment requested', className: 'text-amber-400 border-amber-500/40 bg-amber-500/10' },
  CONTAINMENT_EXECUTED: { label: 'Containment executed (not yet verified)', className: 'text-amber-400 border-amber-500/40 bg-amber-500/10' },
  CONTAINMENT_FAILED: { label: 'Containment FAILED', className: 'text-red-400 border-red-500/40 bg-red-500/10' },
  CONTAINMENT_NOT_VERIFIED: { label: 'Containment NOT verified', className: 'text-red-400 border-red-500/40 bg-red-500/10' },
  CONTAINMENT_VERIFIED: { label: 'CONTAINMENT VERIFIED', className: 'text-emerald-400 border-emerald-500/40 bg-emerald-500/10' },
  ROLLBACK_REQUESTED: { label: 'Rollback requested', className: 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10' },
  ROLLBACK_VERIFIED: { label: 'Rolled back (verified)', className: 'text-cyan-400 border-cyan-500/40 bg-cyan-500/10' },
  PLAYBOOK_APPROVAL_REQUESTED: { label: 'Awaiting analyst approval', className: 'text-amber-400 border-amber-500/40 bg-amber-500/10' },
  PLAYBOOK_REJECTED: { label: 'Rejected by analyst', className: 'text-slate-400 border-slate-600 bg-slate-800/30' },
  UNKNOWN_EVENT_TYPE: { label: 'Unrecognized event', className: 'text-slate-400 border-slate-600 bg-slate-800/30' },
};

function ResponseStateBadge({ state }: { state: string }) {
  const style = RESPONSE_STATE_STYLE[state] ?? { label: state, className: 'text-slate-400 border-slate-600 bg-slate-800/30' };
  return (
    <span className={`inline-flex items-center gap-1.5 text-xs font-bold uppercase px-3 py-1.5 rounded border ${style.className}`}>
      {state === 'CONTAINMENT_VERIFIED' ? <ShieldCheck className="w-3.5 h-3.5" /> : <AlertTriangle className="w-3.5 h-3.5" />}
      {style.label}
    </span>
  );
}

function ActionBadge({ action }: { action: PlaybookAction }) {
  if (action.destructive) {
    return <span className="flex items-center gap-1 text-[10px] font-bold text-red-400"><ShieldOff className="w-3 h-3" /> DESTRUCTIVE</span>;
  }
  if (action.requires_approval) {
    return <span className="flex items-center gap-1 text-[10px] font-bold text-amber-400"><ShieldAlert className="w-3 h-3" /> APPROVAL</span>;
  }
  return <span className="flex items-center gap-1 text-[10px] font-bold text-emerald-400"><ShieldCheck className="w-3 h-3" /> SAFE</span>;
}

function PlaybookCard({
  playbook, onExecute, executing,
}: { playbook: Playbook; onExecute?: (playbook: Playbook) => void; executing?: boolean }) {
  return (
    <div className="glass-card p-4 space-y-3">
      <div className="flex items-center justify-between">
        <div>
          <div className="font-semibold text-white text-sm">{playbook.name}</div>
          <div className="text-[11px] text-slate-500">{playbook.description}</div>
        </div>
        <span className="text-[10px] font-bold px-2 py-0.5 rounded uppercase bg-slate-800/50 text-slate-400">
          {playbook.execution_policy.replace('_', ' ')}
        </span>
      </div>
      <div className="space-y-1.5">
        {playbook.actions.map((a) => (
          <div key={a.action_id} className="flex items-center justify-between bg-[#0a0e17]/60 border border-[#1e2d4a] rounded px-2 py-1.5">
            <div className="min-w-0">
              <div className="text-xs text-slate-200 truncate">{`${a.order}. ${a.name}`}</div>
              <div className="text-[10px] text-slate-500 truncate" title={a.reason}>{a.reason}</div>
            </div>
            <ActionBadge action={a} />
          </div>
        ))}
      </div>
      {onExecute && playbook.execution_policy !== 'recommend_only' && (
        <button
          onClick={() => onExecute(playbook)}
          disabled={executing}
          className="w-full text-xs font-semibold py-1.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 hover:bg-indigo-500/30 disabled:opacity-50"
        >
          {executing ? 'Executing...' : 'Execute'}
        </button>
      )}
    </div>
  );
}

function HistoricalMatchCard({
  match, onAdapt,
}: { match: HistoricalPlaybookMatch; onAdapt: (match: HistoricalPlaybookMatch) => void }) {
  return (
    <div className="glass-card p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-cyan-400">{match.playbook_name}</span>
        <span className="text-[10px] text-slate-500">from {match.source_campaign_id}</span>
      </div>
      <p className="text-[11px] text-slate-400">{match.recommendation_reason}</p>
      <div className="flex gap-3 text-[10px] text-slate-500">
        <span>Technique: {(match.technique_similarity * 100).toFixed(0)}%</span>
        {match.topology_similarity !== null && <span>Topology: {(match.topology_similarity * 100).toFixed(0)}%</span>}
        <span>{match.historical_executions} exec(s)</span>
      </div>
      <button
        onClick={() => onAdapt(match)}
        className="w-full text-xs font-semibold py-1.5 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 hover:bg-cyan-500/20"
      >
        Adapt &amp; Use
      </button>
    </div>
  );
}

const STATUS_COLOR: Record<string, string> = {
  pending: 'text-slate-400', pending_approval: 'text-amber-400', rejected: 'text-red-400',
  running: 'text-cyan-400', success: 'text-emerald-400', failed: 'text-red-400',
  timeout: 'text-red-400', cancelled: 'text-slate-500',
};

function ExecutionRow({
  execution, onApprove, onReject, onPoll,
}: {
  execution: PlaybookExecution;
  onApprove?: (id: string) => void;
  onReject?: (id: string) => void;
  onPoll?: (id: string) => void;
}) {
  return (
    <div className="glass-card p-3 space-y-2">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-slate-300">{execution.execution_id}</span>
        <span className={`text-[10px] font-bold uppercase ${STATUS_COLOR[execution.status] || 'text-slate-400'}`}>
          {execution.status.replace('_', ' ')}
        </span>
      </div>
      <div className="text-[11px] text-slate-500">Campaign: {execution.campaign_id} · Playbook: {execution.playbook_id}</div>
      {execution.rejection_reason && <div className="text-[11px] text-red-400">Reason: {execution.rejection_reason}</div>}
      <div className="flex gap-2">
        {execution.status === 'pending_approval' && onApprove && (
          <button onClick={() => onApprove(execution.execution_id)} className="flex items-center gap-1 text-[11px] font-semibold px-2 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
            <CheckCircle2 className="w-3 h-3" /> Approve
          </button>
        )}
        {execution.status === 'pending_approval' && onReject && (
          <button onClick={() => onReject(execution.execution_id)} className="flex items-center gap-1 text-[11px] font-semibold px-2 py-1 rounded bg-red-500/10 text-red-400 border border-red-500/30">
            <XCircle className="w-3 h-3" /> Reject
          </button>
        )}
        {execution.status === 'running' && onPoll && (
          <button onClick={() => onPoll(execution.execution_id)} className="flex items-center gap-1 text-[11px] font-semibold px-2 py-1 rounded bg-slate-800/50 text-slate-300 border border-slate-700">
            <RefreshCw className="w-3 h-3" /> Check status
          </button>
        )}
      </div>
    </div>
  );
}

export function SOARPage() {
  const { campaigns, selectedCampaign, selectCampaign } = useDashboard();
  const [tab, setTab] = useState<SoarTab>('recommendations');
  const [status, setStatus] = useState<SoarStatus | null>(null);
  const [playbooks, setPlaybooks] = useState<Playbook[]>([]);
  const [executions, setExecutions] = useState<PlaybookExecution[]>([]);
  const [effectiveness, setEffectiveness] = useState<PlaybookEffectiveness[]>([]);
  const [candidate, setCandidate] = useState<Playbook | null>(null);
  const [matches, setMatches] = useState<HistoricalPlaybookMatch[]>([]);
  const [loading, setLoading] = useState(false);
  const [executing, setExecuting] = useState<string | null>(null);
  const [responseState, setResponseState] = useState<ResponseStateResponse | null>(null);
  const [responseStateLoading, setResponseStateLoading] = useState(false);

  useEffect(() => {
    api.soarStatus().then(setStatus).catch(() => setStatus(null));
  }, []);

  const refreshPlaybooks = () => api.listPlaybooks().then((r) => setPlaybooks(r.playbooks)).catch(() => {});
  const refreshExecutions = () => api.listExecutions().then((r) => setExecutions(r.executions)).catch(() => {});
  const refreshEffectiveness = () => api.soarEffectiveness().then((r) => setEffectiveness(r.effectiveness)).catch(() => {});

  useEffect(() => {
    if (tab === 'library') refreshPlaybooks();
    if (tab === 'active' || tab === 'history') refreshExecutions();
    if (tab === 'effectiveness') refreshEffectiveness();
  }, [tab]);

  useEffect(() => {
    // Response's correlation_id == the campaign_id itself, by design --
    // see backend/active_response/correlation.py's reuse strategy.
    if (tab !== 'response' || !selectedCampaign) {
      setResponseState(null);
      return;
    }
    setResponseStateLoading(true);
    api.responseState(selectedCampaign)
      .then(setResponseState)
      .catch(() => setResponseState(null))
      .finally(() => setResponseStateLoading(false));
  }, [tab, selectedCampaign]);

  useEffect(() => {
    if (tab !== 'recommendations' || !selectedCampaign) {
      setCandidate(null);
      setMatches([]);
      return;
    }
    setLoading(true);
    api.soarRecommendations(selectedCampaign)
      .then((r) => {
        setCandidate(r.candidate_playbook);
        setMatches(r.historical_matches);
      })
      .catch(() => { setCandidate(null); setMatches([]); })
      .finally(() => setLoading(false));
  }, [tab, selectedCampaign]);

  const handleGenerateAndExecute = async (playbook: Playbook) => {
    setExecuting(playbook.playbook_id);
    try {
      const saved = await api.generatePlaybook(playbook.source_campaign_id || selectedCampaign || '');
      await api.executePlaybook(saved.playbook_id, saved.source_campaign_id || undefined);
      setTab('active');
    } finally {
      setExecuting(null);
    }
  };

  const handleAdapt = async (match: HistoricalPlaybookMatch) => {
    if (!selectedCampaign) return;
    await api.adaptPlaybook(match.playbook_id, selectedCampaign);
    setTab('library');
    refreshPlaybooks();
  };

  const handleExecuteFromLibrary = async (playbook: Playbook) => {
    setExecuting(playbook.playbook_id);
    try {
      await api.executePlaybook(playbook.playbook_id, playbook.source_campaign_id || undefined);
      setTab('active');
    } finally {
      setExecuting(null);
    }
  };

  const activeExecutions = executions.filter((e) => e.status === 'running' || e.status === 'pending_approval');
  const historicalExecutions = executions.filter((e) => e.status !== 'running' && e.status !== 'pending_approval');

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30">
            <Bot className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">SOAR / Playbooks</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              CYUKTI generates and remembers response playbooks; Shuffle executes them.
            </p>
          </div>
          {status && (
            <div className="ml-auto text-xs font-semibold px-3 py-1.5 rounded-md border bg-slate-800/50 text-slate-400 border-slate-700">
              {status.shuffle_webhook_configured ? 'Shuffle webhook configured' : 'Shuffle not configured'}
            </div>
          )}
        </div>

        <div className="flex bg-[#0a0e17] border border-[#1e2d4a] rounded-md p-1 gap-1 overflow-x-auto">
          {TABS.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              onClick={() => setTab(id)}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-semibold whitespace-nowrap ${
                tab === id ? 'bg-indigo-500/20 text-indigo-300' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className="w-3.5 h-3.5" /> {label}
            </button>
          ))}
        </div>

        {tab === 'recommendations' && (
          <div className="space-y-4">
            <div className="glass-card p-3 flex items-center gap-3">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider shrink-0">Campaign</label>
              <select
                value={selectedCampaign || ''}
                onChange={(e) => selectCampaign(e.target.value || null)}
                className="flex-1 bg-[#0a0e17] text-sm text-slate-200 font-mono border border-[#1e2d4a] rounded px-2 py-1.5 outline-none focus:border-indigo-500"
              >
                <option value="">Select a campaign...</option>
                {campaigns.map((c) => (
                  <option key={c.campaign_id} value={c.campaign_id}>{c.campaign_label || c.campaign_id}</option>
                ))}
              </select>
            </div>

            {!selectedCampaign ? (
              <div className="glass-card p-8 text-center text-slate-500 text-sm">Select a campaign to see a recommended playbook.</div>
            ) : loading ? (
              <div className="glass-card p-8 text-center text-cyan-400 text-sm animate-pulse">Generating recommendation...</div>
            ) : (
              <>
                {matches.length > 0 && (
                  <div className="space-y-2">
                    <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">Historical Playbook Matches</h2>
                    <div className="grid md:grid-cols-2 gap-3">
                      {matches.map((m) => <HistoricalMatchCard key={m.playbook_id} match={m} onAdapt={handleAdapt} />)}
                    </div>
                  </div>
                )}
                {candidate && (
                  <div className="space-y-2">
                    <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">Candidate Playbook (freshly generated)</h2>
                    <PlaybookCard playbook={candidate} onExecute={handleGenerateAndExecute} executing={executing === candidate.playbook_id} />
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {tab === 'library' && (
          <div className="grid md:grid-cols-2 gap-3">
            {playbooks.length === 0 ? (
              <div className="glass-card p-8 text-center text-slate-500 text-sm md:col-span-2">No playbooks generated yet.</div>
            ) : (
              playbooks.map((p) => (
                <PlaybookCard key={p.playbook_id} playbook={p} onExecute={handleExecuteFromLibrary} executing={executing === p.playbook_id} />
              ))
            )}
          </div>
        )}

        {tab === 'active' && (
          <div className="space-y-2">
            {activeExecutions.length === 0 ? (
              <div className="glass-card p-8 text-center text-slate-500 text-sm">No active executions.</div>
            ) : (
              activeExecutions.map((e) => (
                <ExecutionRow
                  key={e.execution_id}
                  execution={e}
                  onApprove={(id) => api.approveExecution(id).then(refreshExecutions)}
                  onReject={(id) => api.rejectExecution(id, 'Rejected by analyst').then(refreshExecutions)}
                  onPoll={(id) => api.pollExecution(id).then(refreshExecutions)}
                />
              ))
            )}
          </div>
        )}

        {tab === 'history' && (
          <div className="space-y-2">
            {historicalExecutions.length === 0 ? (
              <div className="glass-card p-8 text-center text-slate-500 text-sm">No completed executions yet.</div>
            ) : (
              historicalExecutions.map((e) => <ExecutionRow key={e.execution_id} execution={e} />)
            )}
          </div>
        )}

        {tab === 'effectiveness' && (
          <div className="glass-card divide-y divide-[#1e2d4a]">
            {effectiveness.length === 0 ? (
              <div className="p-8 text-center text-slate-500 text-sm">No effectiveness data yet.</div>
            ) : (
              effectiveness.map((e) => (
                <div key={e.playbook_id} className="p-3 flex items-center justify-between">
                  <div>
                    <div className="text-sm text-slate-200">{e.playbook_name}</div>
                    <div className="text-[11px] text-slate-500">
                      {e.executions} execution(s) · {e.analyst_approvals} approved · {e.analyst_rejections} rejected
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-mono text-sm text-white">
                      {e.success_rate !== null ? `${(e.success_rate * 100).toFixed(0)}%` : '—'}
                    </div>
                    <div className="text-[10px] text-slate-500">success rate</div>
                  </div>
                </div>
              ))
            )}
          </div>
        )}

        {tab === 'response' && (
          <div className="space-y-4">
            <div className="glass-card p-3 flex items-center gap-3">
              <label className="text-xs font-semibold text-slate-400 uppercase tracking-wider shrink-0">Campaign</label>
              <select
                value={selectedCampaign || ''}
                onChange={(e) => selectCampaign(e.target.value || null)}
                className="flex-1 bg-[#0a0e17] text-sm text-slate-200 font-mono border border-[#1e2d4a] rounded px-2 py-1.5 outline-none focus:border-indigo-500"
              >
                <option value="">Select a campaign...</option>
                {campaigns.map((c) => (
                  <option key={c.campaign_id} value={c.campaign_id}>{c.campaign_label || c.campaign_id}</option>
                ))}
              </select>
            </div>

            {!selectedCampaign ? (
              <div className="glass-card p-8 text-center text-slate-500 text-sm">Select a campaign to see its active-response lifecycle.</div>
            ) : responseStateLoading ? (
              <div className="glass-card p-8 text-center text-cyan-400 text-sm animate-pulse">Loading response state...</div>
            ) : !responseState ? (
              <div className="glass-card p-8 text-center text-slate-500 text-sm">Could not load response state for this campaign.</div>
            ) : (
              <>
                <div className="glass-card p-4 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-[11px] text-slate-500 font-mono">correlation_id: {responseState.correlation_id}</span>
                    <ResponseStateBadge state={responseState.current_state} />
                  </div>
                  {responseState.current_state === 'NO_RESPONSE_ACTIVITY' && (
                    <p className="text-[11px] text-slate-500">
                      No containment has been requested for this campaign. This is the normal state for the
                      large majority of campaigns -- containment is gated by ResponsePolicyEngine and, in
                      production, requires AUTO_CONTAIN=true plus an eligible QUALIFIED_THREAT classification.
                    </p>
                  )}
                  {(responseState.current_state === 'CONTAINMENT_EXECUTED') && (
                    <p className="text-[11px] text-amber-400/80">
                      A containment action executed but has not yet been independently verified -- this is
                      deliberately never shown as "blocked" until CONTAINMENT_VERIFIED appears.
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <h2 className="text-xs font-semibold text-slate-400 uppercase tracking-wider px-1">
                    Audit trail ({responseState.event_count} event{responseState.event_count === 1 ? '' : 's'})
                  </h2>
                  {responseState.events.length === 0 ? (
                    <div className="glass-card p-6 text-center text-slate-500 text-sm">No response events recorded yet.</div>
                  ) : (
                    <div className="glass-card divide-y divide-[#1e2d4a]">
                      {responseState.events.map((e, i) => (
                        <div key={i} className="p-3 flex items-center justify-between gap-3">
                          <div className="min-w-0">
                            <div className="text-xs font-mono text-slate-200">{e.event_type}</div>
                            <div className="text-[10px] text-slate-500 truncate">{e.timestamp}</div>
                          </div>
                          {typeof e.detail?.reason === 'string' && (
                            <span className="text-[10px] text-slate-500 truncate max-w-xs" title={e.detail.reason}>{e.detail.reason}</span>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
