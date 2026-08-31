import { useDashboard } from '../context/DashboardContext';
import { ShieldCheck, CheckCircle2 } from 'lucide-react';
import { useState, useEffect } from 'react';
import type { RecommendationItem } from '../types';
import { PanelWrapper } from './PanelWrapper';

function normalizeRecs(recs: RecommendationItem[] | string[]): RecommendationItem[] {
  if (!recs?.length) return [];
  if (typeof recs[0] === 'string') {
    return (recs as string[]).map((r) => ({ recommendation: r, priority: 'MEDIUM', mitre_mitigation: 'N/A', reason: '', traceability: 'MITIGATION_MAP' }));
  }
  return recs as RecommendationItem[];
}

export function RecommendationEngine() {
  const { recommendations, selectedCampaign, predictions, resetKey } = useDashboard();
  const [completed, setCompleted] = useState<Set<string>>(new Set());

  useEffect(() => {
    setCompleted(new Set());
  }, [resetKey]);

  let activeRecs = null;
  if (selectedCampaign) {
    const pred = predictions.find((p) => p.campaign_id === selectedCampaign);
    if (pred?.predicted_technique) {
      activeRecs = recommendations.find((r) => r.technique_id === pred.predicted_technique);
    }
  } else if (predictions[0]?.predicted_technique) {
    activeRecs = recommendations.find((r) => r.technique_id === predictions[0].predicted_technique);
  }
  if (!activeRecs && recommendations.length > 0) activeRecs = recommendations[0];

  const items = activeRecs ? normalizeRecs(activeRecs.recommendations) : [];

  if (!items.length) {
    return (
      <PanelWrapper title="Defensive Playbook" icon={<ShieldCheck className="w-4 h-4 text-emerald-400" />} className="h-full">
        <div className="flex-1 flex items-center justify-center text-slate-500 text-sm p-4 text-center">
          No automated recommendations available for the selected campaign.
        </div>
      </PanelWrapper>
    );
  }

  const activePrediction = selectedCampaign
    ? predictions.find((p) => p.campaign_id === selectedCampaign)
    : predictions[0];

  return (
    <PanelWrapper
      title="Defensive Playbook"
      icon={<ShieldCheck className="w-4 h-4 text-emerald-400" />}
      className="h-full"
      headerExtra={
        <div className="flex items-center gap-1.5 text-[10px] text-slate-400">
          <span className="font-mono text-cyan-400">{activeRecs?.technique_id}</span>
          {activePrediction?.predicted_name && <span className="truncate max-w-[120px]" title={activePrediction.predicted_name}>{activePrediction.predicted_name}</span>}
        </div>
      }
    >
      <div className="flex-1 overflow-auto p-3 bg-[#060a13] custom-scrollbar space-y-2">
        {items.map((rec, idx) => {
          const key = `${rec.recommendation}-${idx}`;
          const isDone = completed.has(key);
          return (
            <div
              key={key}
              onClick={() => {
                const next = new Set(completed);
                if (next.has(key)) next.delete(key);
                else next.add(key);
                setCompleted(next);
              }}
              className={`p-3 rounded-lg border cursor-pointer ${isDone ? 'opacity-60 border-emerald-500/20' : 'border-[#1e2d4a] hover:border-emerald-500/40 bg-[#0c1220]'}`}
            >
              <div className="flex items-start gap-2">
                <CheckCircle2 className={`w-4 h-4 mt-0.5 ${isDone ? 'text-emerald-500' : 'text-slate-500'}`} />
                <div className="flex-1 min-w-0">
                  <div className={`text-sm ${isDone ? 'line-through text-slate-500' : 'text-slate-200'}`}>{rec.recommendation}</div>
                  <div className="mt-2 grid grid-cols-2 gap-1 text-[10px] text-slate-500">
                    <span>Priority: <strong className="text-orange-400">{rec.priority || 'MEDIUM'}</strong></span>
                    <span>MITRE: <strong className="text-emerald-400">{rec.mitre_mitigation || 'N/A'}</strong></span>
                    {rec.reason && <span className="col-span-2">{rec.reason}</span>}
                    {rec.traceability && <span className="col-span-2 font-mono text-slate-600">{rec.traceability}</span>}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </PanelWrapper>
  );
}
