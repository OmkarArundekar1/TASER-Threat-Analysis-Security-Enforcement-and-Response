import { Zap, Target, ArrowRight } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';

/**
 * Full top-level page for next-technique prediction
 * (prediction_engine.py / chain_updater.py's LIKELY_NEXT relationships,
 * served by GET /api/predictions -- already fetched into
 * DashboardContext for the smaller in-panel PredictionPanel.tsx, which
 * only ever showed one prediction at a time and was not mounted
 * anywhere in the app). This page lists every real prediction the
 * backend currently holds, not a single selected one.
 */
export function PredictionIntelligencePage() {
  const { predictions, selectedCampaign, selectCampaign } = useDashboard();

  return (
    <div className="flex-1 overflow-y-auto custom-scrollbar p-4 md:p-6 bg-[#060a13]">
      <div className="max-w-5xl mx-auto space-y-6">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-indigo-500/10 border border-indigo-500/30">
            <Zap className="w-6 h-6 text-indigo-400" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Attack Prediction Intelligence</h1>
            <p className="text-xs text-slate-500 mt-0.5">
              Next-technique prediction from observed NEXT_TECHNIQUE/LIKELY_NEXT transitions
              (prediction_engine.py). Persisted only once the pipeline has seen enough real
              transitions for a given technique -- not a guess.
            </p>
          </div>
          <div className="ml-auto text-xs font-semibold px-3 py-1.5 rounded-md border bg-slate-800/50 text-slate-400 border-slate-700">
            {predictions.length} active prediction{predictions.length === 1 ? '' : 's'}
          </div>
        </div>

        {predictions.length === 0 ? (
          <div className="glass-card p-8 flex flex-col items-center text-center gap-3">
            <Target className="w-10 h-10 text-slate-600" />
            <p className="text-sm font-semibold text-slate-300">No predictions available yet</p>
            <p className="text-xs text-slate-500 max-w-md">
              Predictions are persisted when the ingestion pipeline observes enough
              NEXT_TECHNIQUE transitions for a technique. Verify in Neo4j:{' '}
              <code className="text-cyan-400 bg-[#0a0e17] px-1.5 py-0.5 rounded">
                MATCH (c:Campaign)-[r:LIKELY_NEXT]-&gt;(t) RETURN c,r,t
              </code>
            </p>
          </div>
        ) : (
          <div className="grid md:grid-cols-2 gap-3">
            {predictions.map((p) => {
              const isSelected = p.campaign_id === selectedCampaign;
              return (
                <div
                  key={p.campaign_id}
                  onClick={() => selectCampaign(p.campaign_id)}
                  className={`glass-card p-4 cursor-pointer transition-colors border ${
                    isSelected ? 'border-indigo-500/60' : 'border-transparent hover:border-indigo-500/30'
                  }`}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-mono text-xs text-slate-500 truncate max-w-[160px]">{p.campaign_id}</span>
                    <span
                      className={`text-[10px] font-bold px-2 py-0.5 rounded uppercase ${
                        p.risk_level === 'CRITICAL' || p.risk_level === 'HIGH'
                          ? 'bg-red-500/10 text-red-400'
                          : p.risk_level === 'MEDIUM'
                          ? 'bg-amber-500/10 text-amber-400'
                          : 'bg-emerald-500/10 text-emerald-400'
                      }`}
                    >
                      {p.risk_level}
                    </span>
                  </div>

                  <div className="flex items-center gap-2 mb-3">
                    <div className="flex-1 bg-[#0a0e17]/80 border border-[#1e2d4a] rounded p-2 text-center">
                      <div className="text-[9px] text-slate-500 uppercase">Current</div>
                      <div className="font-mono font-bold text-slate-300 text-sm truncate">{p.current_technique}</div>
                    </div>
                    <ArrowRight className="w-4 h-4 text-indigo-400 flex-none" />
                    <div className="flex-1 bg-indigo-500/10 border border-indigo-500/40 rounded p-2 text-center">
                      <div className="text-[9px] text-indigo-400 uppercase">Predicted</div>
                      <div className="font-mono font-bold text-white text-sm truncate">{p.predicted_technique}</div>
                    </div>
                  </div>

                  <div className="h-1.5 rounded-full bg-[#131c2e] overflow-hidden mb-2">
                    <div
                      className="h-full bg-gradient-to-r from-indigo-600 to-cyan-400 rounded-full"
                      style={{ width: `${Math.max(0, Math.min(100, p.confidence))}%` }}
                    />
                  </div>

                  <div className="flex justify-between text-[10px] text-slate-500">
                    <span>{p.source || 'LIKELY_NEXT'} · {Math.round(p.confidence)}% confidence</span>
                    <span>{p.generated_at ? new Date(p.generated_at).toLocaleTimeString() : '—'}</span>
                  </div>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
