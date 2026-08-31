import { useDashboard } from '../context/DashboardContext';
import { Target, Clock, ArrowLeft, ShieldCheck } from 'lucide-react';
import { useState, useEffect } from 'react';
import { api } from '../services/api';

export function CampaignIntelligence() {
  const { campaigns, selectedCampaign, selectCampaign, predictions, recommendations } = useDashboard();

  const getRiskClass = (risk: string) => {
    switch (risk) {
      case 'CRITICAL': return 'text-red-400 bg-red-500/10 border-red-500/30';
      case 'HIGH': return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
      case 'MEDIUM': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
      case 'LOW': return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500/30';
    }
  };

  const getRiskWidth = (score: number) => `${Math.min(100, Math.max(5, score))}%`;

  const [timeline, setTimeline] = useState<any[]>([]);
  const [loadingTimeline, setLoadingTimeline] = useState(false);

  useEffect(() => {
    if (selectedCampaign) {
      setLoadingTimeline(true);
      api.campaignTimeline(selectedCampaign)
        .then(res => setTimeline(res.events || []))
        .catch(err => console.error("Failed to load timeline", err))
        .finally(() => setLoadingTimeline(false));
    } else {
      setTimeline([]);
    }
  }, [selectedCampaign]);

  if (selectedCampaign) {
    const camp = campaigns.find(c => c.campaign_id === selectedCampaign);
    const pred = predictions.find(p => p.campaign_id === selectedCampaign);
    const predRecs = pred ? recommendations.find(r => r.technique_id === pred.predicted_technique) : null;
    return (
      <div className="glass-card flex flex-col h-full overflow-hidden animate-slide-in">
        <div className="p-4 border-b border-[#1e2d4a] flex items-center justify-between bg-[#131c2e]/50">
          <h2 className="text-sm font-semibold uppercase tracking-wider flex items-center gap-2">
            <button onClick={() => selectCampaign(null)} className="p-1 hover:bg-[#1a2540] rounded mr-1 transition-colors">
              <ArrowLeft className="w-4 h-4 text-slate-400 hover:text-white" />
            </button>
            <Target className="w-4 h-4 text-orange-400" />
            Campaign Investigation
          </h2>
        </div>
        
        <div className="flex-1 overflow-auto p-4 custom-scrollbar flex flex-col gap-4">
          {/* Summary Panel */}
          {camp && (
            <div className="bg-[#0a0e17] border border-[#1e2d4a] rounded p-3">
              <h3 className="text-xs text-slate-400 uppercase tracking-widest mb-2 font-semibold">Summary</h3>
              <div className="flex justify-between items-end mb-3">
                <div className="font-mono text-lg text-orange-400 font-bold">{camp.campaign_label || camp.campaign_id}</div>
                <div className={`px-2 py-0.5 rounded text-[10px] font-bold border ${getRiskClass(camp.risk_level)}`}>
                  {camp.risk_score}/100 · {camp.risk_level}
                </div>
              </div>
              <div className="grid grid-cols-2 gap-2 text-xs font-mono">
                <div className="bg-[#131c2e] p-2 rounded border border-[#1e2d4a]">
                  <div className="text-[9px] text-slate-500 mb-1">Attacker</div>
                  <div className="text-red-400">{camp.attacker_ip}</div>
                </div>
                <div className="bg-[#131c2e] p-2 rounded border border-[#1e2d4a]">
                  <div className="text-[9px] text-slate-500 mb-1">Target</div>
                  <div className="text-blue-400">{camp.victim_ip}</div>
                </div>
              </div>
            </div>
          )}

          {/* Timeline & Attack Chain */}
          <div className="bg-[#0a0e17] border border-[#1e2d4a] rounded p-3 flex-1 flex flex-col min-h-[200px]">
            <h3 className="text-xs text-slate-400 uppercase tracking-widest mb-3 font-semibold flex items-center justify-between">
              <span>Observed Attack Chain</span>
              {loadingTimeline && <span className="text-indigo-400 animate-pulse text-[10px]">Loading...</span>}
            </h3>
            
            <div className="flex-1 overflow-y-auto pr-2 custom-scrollbar">
              <div className="relative border-l border-[#1e2d4a] ml-3 pl-4 space-y-4 py-2">
                {timeline.map((event, idx) => (
                  <div key={idx} className="relative">
                    <div className="absolute -left-[21px] top-1 w-2.5 h-2.5 bg-indigo-500 rounded-full border-2 border-[#0a0e17] shadow-[0_0_8px_rgba(99,102,241,0.6)]"></div>
                    <div className="text-[10px] text-slate-500 font-mono mb-0.5">
                      {new Date(event.first_seen).toLocaleTimeString()} {event.occurrences > 1 && <span className="text-indigo-400 bg-indigo-500/10 px-1 ml-1 rounded">x{event.occurrences}</span>}
                    </div>
                    <div className="bg-[#131c2e] border border-[#1e2d4a] p-2 rounded">
                      <div className="text-xs font-bold text-slate-200 mb-1 truncate" title={event.technique_name}>{event.technique_name}</div>
                      <div className="flex gap-2 text-[10px] font-mono">
                        <span className="text-indigo-400">{event.technique_id}</span>
                        <span className="text-emerald-400">{event.stage}</span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>

          {pred && (
            <div className="bg-cyan-500/10 border border-cyan-500/30 rounded p-3">
              <h3 className="text-xs text-cyan-400 uppercase mb-2 font-semibold">Predicted Next Technique</h3>
              <div className="font-mono text-lg text-white">{pred.predicted_technique}</div>
              <div className="text-[10px] text-slate-400 mt-1">Confidence: {pred.confidence}% · {pred.source || 'LIKELY_NEXT'}</div>
            </div>
          )}

          {predRecs && (
            <div className="bg-emerald-500/10 border border-emerald-500/30 rounded p-3">
              <h3 className="text-xs text-emerald-400 uppercase mb-2 font-semibold flex items-center gap-1"><ShieldCheck className="w-3 h-3" /> Recommendations</h3>
              <ul className="text-xs text-slate-300 space-y-1">
                {(Array.isArray(predRecs.recommendations) ? predRecs.recommendations : []).slice(0, 3).map((r, i) => (
                  <li key={i}>{typeof r === 'string' ? r : r.recommendation}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="glass-card flex flex-col h-full overflow-hidden">
      <div className="p-4 border-b border-[#1e2d4a] flex items-center justify-between bg-[#131c2e]/50">
        <h2 className="text-sm font-semibold uppercase tracking-wider flex items-center gap-2">
          <Target className="w-4 h-4 text-orange-400" />
          Campaign Intelligence
        </h2>
        {selectedCampaign && (
          <button 
            onClick={() => selectCampaign(null)}
            className="text-xs text-slate-400 hover:text-white bg-[#1e2d4a] px-2 py-1 rounded transition-colors"
          >
            Clear Selection
          </button>
        )}
      </div>

      <div className="flex-1 overflow-auto">
        <table className="soc-table w-full">
          <thead>
            <tr>
              <th>Campaign ID</th>
              <th>Scope</th>
              <th>Risk</th>
              <th>Latest TTP</th>
              <th>Predicted Next</th>
            </tr>
          </thead>
          <tbody>
            {campaigns.length === 0 ? (
              <tr>
                <td colSpan={5} className="text-center py-8 text-slate-500 italic">No active campaigns</td>
              </tr>
            ) : (
              campaigns.map((camp) => (
                <tr 
                  key={camp.campaign_id} 
                  onClick={() => selectCampaign(camp.campaign_id)}
                  className={`cursor-pointer transition-all ${
                    selectedCampaign === camp.campaign_id 
                      ? 'bg-indigo-500/10 border-l-2 border-l-indigo-500' 
                      : 'hover:bg-[#1a2540] border-l-2 border-l-transparent'
                  }`}
                >
                  <td>
                    <div className="flex flex-col">
                      <span className="font-mono text-orange-400 font-bold truncate max-w-[150px]" title={camp.campaign_id}>
                        {camp.campaign_label || camp.campaign_id}
                      </span>
                      <span className="text-[9px] text-slate-500 flex items-center gap-1 mt-1">
                        <Clock className="w-2.5 h-2.5" /> 
                        {new Date(camp.last_seen).toLocaleTimeString()}
                        <span className="bg-slate-800 px-1 rounded ml-1">{camp.event_count} events</span>
                      </span>
                    </div>
                  </td>
                  <td>
                    <div className="flex flex-col gap-1 text-[10px] font-mono">
                      <span className="text-red-400 flex items-center gap-1 truncate max-w-[120px]" title={camp.attacker_ip}><span className="w-1.5 h-1.5 rounded-full bg-red-500 flex-shrink-0"></span> {camp.attacker_ip}</span>
                      <span className="text-blue-400 flex items-center gap-1 truncate max-w-[120px]" title={camp.victim_ip}><span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0"></span> {camp.victim_ip}</span>
                    </div>
                  </td>
                  <td>
                    <div className="flex flex-col gap-1 w-20">
                      <div className="flex items-center justify-between text-[10px]">
                        <span className={`px-1.5 rounded border font-bold ${getRiskClass(camp.risk_level)}`}>
                          {camp.risk_level}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-[#1e2d4a] rounded-full overflow-hidden mt-0.5">
                        <div
                          className={`h-full rounded-full ${
                            camp.risk_level === 'CRITICAL' ? 'bg-red-500' :
                            camp.risk_level === 'HIGH' ? 'bg-orange-500' :
                            camp.risk_level === 'MEDIUM' ? 'bg-yellow-500' : 'bg-blue-500'
                          }`}
                          style={{ width: getRiskWidth(camp.risk_score) }}
                        />
                      </div>
                    </div>
                  </td>
                  <td>
                    <span className="px-1.5 py-0.5 bg-[#0a0e17] border border-[#1e2d4a] rounded text-indigo-400 font-mono text-[10px] font-bold shadow-[inset_0_0_10px_rgba(99,102,241,0.1)]">
                      {camp.latest_technique}
                    </span>
                  </td>
                  <td>
                    {camp.predicted_technique ? (
                      <div className="flex flex-col gap-0.5">
                        <span className="px-1.5 py-0.5 bg-cyan-500/10 border border-cyan-500/30 rounded text-cyan-400 font-mono text-[10px] font-bold w-fit">
                          {camp.predicted_technique}
                        </span>
                        {camp.prediction_confidence !== undefined && (
                          <span className="text-[9px] text-slate-500">{camp.prediction_confidence}% conf</span>
                        )}
                      </div>
                    ) : (
                      <span className="text-[10px] text-slate-600">—</span>
                    )}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
