import { useDashboard } from '../context/DashboardContext';
import { Target, Zap, ArrowRight } from 'lucide-react';
import { PanelWrapper } from './PanelWrapper';

export function PredictionPanel() {
  const { predictions, selectedCampaign } = useDashboard();

  const prediction = selectedCampaign
    ? predictions.find((p) => p.campaign_id === selectedCampaign)
    : predictions[0];

  if (!prediction) {
    return (
      <PanelWrapper title="Predicted Next Attack" icon={<Zap className="w-4 h-4 text-cyan-400 opacity-50" />} className="h-full">
        <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
          <Target className="w-8 h-8 mb-3 opacity-20" />
          <p className="text-sm font-semibold text-slate-400">No LIKELY_NEXT relationship found.</p>
          <p className="text-xs mt-2 max-w-[220px]">
            Predictions are persisted when the ingestion pipeline observes enough NEXT_TECHNIQUE transitions.
            Verify in Neo4j: MATCH (c:Campaign)-[r:LIKELY_NEXT]-&gt;(t) RETURN c,r,t
          </p>
        </div>
      </PanelWrapper>
    );
  }

  return (
    <PanelWrapper
      title="AI Prediction Engine"
      icon={<Zap className="w-4 h-4 text-cyan-400" />}
      className="h-full"
      headerExtra={
        <span className="text-[10px] text-cyan-400 mr-1">{prediction.source || 'LIKELY_NEXT'}</span>
      }
    >
      <div className="p-4 flex-1 flex flex-col items-center justify-between min-h-0">
        <div className="flex items-center justify-between w-full mb-2 gap-2">
          <div className="flex-1 bg-[#0a0e17]/80 border border-[#1e2d4a] rounded p-2 text-center">
            <div className="text-[9px] text-slate-500 uppercase">Current</div>
            <div className="font-mono font-bold text-slate-300 text-sm truncate">{prediction.current_technique}</div>
          </div>
          <ArrowRight className="w-4 h-4 text-cyan-400 flex-none" />
          <div className="flex-1 bg-cyan-500/10 border border-cyan-500/40 rounded p-2 text-center">
            <div className="text-[9px] text-cyan-400 uppercase">Predicted</div>
            <div className="font-mono font-bold text-white text-sm truncate">{prediction.predicted_technique}</div>
          </div>
        </div>

        {/* SVG Semi-Circle Gauge */}
        <div className="relative flex-1 flex flex-col items-center justify-center w-full min-h-[100px]">
          <svg viewBox="0 0 100 50" className="w-full h-full max-h-[100px] overflow-visible">
            {/* Background Arc */}
            <path
              d="M 10 50 A 40 40 0 0 1 90 50"
              fill="none"
              stroke="#1e2d4a"
              strokeWidth="8"
              strokeLinecap="round"
            />
            {/* Foreground Arc (Animated) */}
            <path
              d="M 10 50 A 40 40 0 0 1 90 50"
              fill="none"
              stroke="url(#gradient)"
              strokeWidth="8"
              strokeLinecap="round"
              strokeDasharray="125.6" /* 40 * Math.PI */
              strokeDashoffset={125.6 - (125.6 * prediction.confidence) / 100}
              className="transition-all duration-1000 ease-out"
            />
            <defs>
              <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="#3b82f6" />
                <stop offset="50%" stopColor="#8b5cf6" />
                <stop offset="100%" stopColor="#ef4444" />
              </linearGradient>
            </defs>
          </svg>
          <div className="absolute bottom-2 flex flex-col items-center">
            <span className="font-mono text-2xl font-bold text-white leading-none">
              {Math.round(prediction.confidence)}<span className="text-sm text-cyan-400 ml-1">%</span>
            </span>
            <span className="text-[10px] text-slate-500 uppercase tracking-wider mt-1">Confidence</span>
          </div>
        </div>

        <div className="w-full flex justify-between text-[9px] text-slate-500 mt-2 border-t border-[#1e2d4a] pt-2">
          <span>Risk: <span className="text-slate-300 font-semibold">{prediction.risk_level}</span></span>
          <span>{prediction.generated_at ? new Date(prediction.generated_at).toLocaleTimeString() : '—'}</span>
        </div>
      </div>
    </PanelWrapper>
  );
}
