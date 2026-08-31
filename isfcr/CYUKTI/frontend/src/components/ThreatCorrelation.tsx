import { useState, useEffect } from 'react';
import { Network, ShieldAlert } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';

export function ThreatCorrelation() {
  const { selectedCampaign, selectCampaign } = useDashboard();
  const [loading, setLoading] = useState(false);
  const [correlationData, setCorrelationData] = useState<any[]>([]);

  useEffect(() => {
    if (selectedCampaign) {
      setLoading(true);
      api.campaignCorrelation(selectedCampaign)
        .then(res => setCorrelationData(res.similar_campaigns || []))
        .catch(err => console.error("Correlation fetch failed", err))
        .finally(() => setLoading(false));
    } else {
      setCorrelationData([]);
    }
  }, [selectedCampaign]);

  if (!selectedCampaign) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
        <Network className="w-8 h-8 mb-3 opacity-20" />
        <p className="text-sm font-semibold text-slate-400">Select a Campaign</p>
        <p className="text-xs mt-2 max-w-[220px]">
          Select a campaign from the graph or left panel to analyze its similarities with historical campaigns.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden animate-slide-in p-3 bg-[#060a13] custom-scrollbar space-y-3">
      {loading ? (
        <div className="flex items-center justify-center h-32 text-indigo-400 animate-pulse text-xs">
          Calculating Graph Similarity...
        </div>
      ) : correlationData.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-xs">
          <ShieldAlert className="w-6 h-6 mb-2 opacity-50" />
          No similar campaigns found in historical data.
        </div>
      ) : (
        correlationData.map((camp, idx) => (
          <div key={idx} className="bg-[#0a0e17] border border-[#1e2d4a] rounded-lg p-3 hover:border-indigo-500/50 transition-colors">
            <div className="flex justify-between items-center mb-2">
              <div 
                className="font-mono text-orange-400 font-bold text-sm cursor-pointer hover:underline"
                onClick={() => selectCampaign(camp.campaign_id)}
              >
                {camp.campaign_label}
              </div>
              <div className="flex items-center gap-2">
                <div className="text-[10px] text-slate-400 uppercase">Similarity</div>
                <div className="text-lg font-mono font-bold text-white">
                  {camp.similarity_score}%
                </div>
              </div>
            </div>
            
            <div className="space-y-2 mt-3 text-xs">
              {camp.shared_tactics && camp.shared_tactics.length > 0 && (
                <div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Shared Tactics</div>
                  <div className="flex flex-wrap gap-1">
                    {camp.shared_tactics.map((t: string, i: number) => (
                      <span key={i} className="px-1.5 py-0.5 bg-[#131c2e] border border-[#1e2d4a] text-emerald-400 rounded">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              {camp.shared_techniques && camp.shared_techniques.length > 0 && (
                <div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Shared Techniques</div>
                  <div className="flex flex-wrap gap-1">
                    {camp.shared_techniques.map((t: string, i: number) => (
                      <span key={i} className="px-1.5 py-0.5 bg-indigo-500/10 border border-indigo-500/30 text-indigo-400 rounded">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              <div className="flex gap-4 pt-1 mt-2 border-t border-[#1e2d4a]">
                {camp.shared_attackers && camp.shared_attackers.length > 0 && (
                  <div className="flex items-center gap-1 text-[10px]">
                    <span className="text-slate-500">Attacker Match:</span>
                    <span className="text-red-400 font-mono">{camp.shared_attackers[0]}</span>
                  </div>
                )}
                {camp.shared_hosts && camp.shared_hosts.length > 0 && (
                  <div className="flex items-center gap-1 text-[10px]">
                    <span className="text-slate-500">Host Match:</span>
                    <span className="text-blue-400 font-mono">{camp.shared_hosts[0]}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        ))
      )}
    </div>
  );
}
