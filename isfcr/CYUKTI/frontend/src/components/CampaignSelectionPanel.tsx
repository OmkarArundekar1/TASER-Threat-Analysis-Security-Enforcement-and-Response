import { useEffect, useState } from 'react';
import { GitCompareArrows, Award } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import type { CampaignCandidate, CampaignSelectionResponse } from '../types';

const SIGNAL_LABELS: Record<string, string> = {
  topology_similarity: 'Topology',
  technique_similarity: 'Technique',
  temporal_similarity: 'Temporal',
  attacker_similarity: 'Attacker',
  host_similarity: 'Host',
};

const CONFIDENCE_COLOR: Record<string, string> = {
  HIGH: 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10',
  MEDIUM: 'text-amber-400 border-amber-500/30 bg-amber-500/10',
  LOW: 'text-red-400 border-red-500/30 bg-red-500/10',
  NONE: 'text-slate-500 border-slate-700 bg-slate-800/50',
};

function CandidateRow({ candidate, isSelected }: { candidate: CampaignCandidate; isSelected: boolean }) {
  return (
    <div className={`glass-card p-3 ${isSelected ? 'border border-cyan-500/50' : ''}`}>
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          {isSelected && <Award className="w-3.5 h-3.5 text-cyan-400" />}
          <span className="font-mono text-xs text-cyan-400">{candidate.campaign_id}</span>
        </div>
        <span className="font-mono text-sm font-bold text-white">{(candidate.composite_score * 100).toFixed(0)}%</span>
      </div>
      <div className="grid grid-cols-5 gap-1">
        {Object.entries(SIGNAL_LABELS).map(([key, label]) => {
          const value = candidate.signals[key as keyof typeof candidate.signals];
          return (
            <div key={key} className="text-center">
              <div className="text-[9px] text-slate-500 uppercase">{label}</div>
              <div className="text-[11px] font-mono text-slate-300">
                {value === null ? '—' : `${(value * 100).toFixed(0)}%`}
              </div>
            </div>
          );
        })}
      </div>
      {candidate.historical_playbook_success_rate !== null && (
        <div className="mt-1.5 text-[10px] text-slate-500">
          Historical response success: {(candidate.historical_playbook_success_rate * 100).toFixed(0)}%
          {' '}({candidate.historical_playbook_executions} execution(s))
        </div>
      )}
    </div>
  );
}

export function CampaignSelectionPanel() {
  const { selectedCampaign } = useDashboard();
  const [data, setData] = useState<CampaignSelectionResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!selectedCampaign) {
      setData(null);
      return;
    }
    setLoading(true);
    setError(null);
    api.campaignSelection(selectedCampaign)
      .then(setData)
      .catch((err) => setError(err.message || 'Campaign selection failed'))
      .finally(() => setLoading(false));
  }, [selectedCampaign]);

  if (!selectedCampaign) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center gap-2">
        <GitCompareArrows className="w-8 h-8 opacity-20" />
        <p className="text-sm">Select a campaign to see the best-matching historical campaign and why.</p>
      </div>
    );
  }

  if (loading) {
    return <div className="flex-1 flex items-center justify-center text-cyan-400 text-sm animate-pulse">Ranking historical candidates...</div>;
  }

  if (error) {
    return <div className="flex-1 flex items-center justify-center text-red-400 text-sm">{error}</div>;
  }

  if (!data || data.ranked_candidates.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center gap-2">
        <GitCompareArrows className="w-8 h-8 opacity-20" />
        <p className="text-sm">No comparable historical campaigns found (no shared technique or attacker identity).</p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-3 space-y-3">
      <div className={`text-xs px-3 py-2 rounded-md border ${CONFIDENCE_COLOR[data.confidence]}`}>
        <span className="font-bold uppercase">{data.confidence} confidence</span>
        {data.score_gap !== null && <span className="ml-2 text-slate-400">(gap: {(data.score_gap * 100).toFixed(0)}%)</span>}
      </div>
      <p className="text-xs text-slate-400 px-1">{data.explanation}</p>
      <div className="space-y-2">
        {data.ranked_candidates.map((c) => (
          <CandidateRow key={c.campaign_id} candidate={c} isSelected={c.campaign_id === data.selected?.campaign_id} />
        ))}
      </div>
    </div>
  );
}
