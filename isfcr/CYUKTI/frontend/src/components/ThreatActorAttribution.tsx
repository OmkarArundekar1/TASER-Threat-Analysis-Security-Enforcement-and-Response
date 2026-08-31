import { useState, useEffect } from 'react';
import { ShieldAlert, Crosshair, Cpu, Bug } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';

export function ThreatActorAttribution() {
  const { selectedCampaign } = useDashboard();
  const [loading, setLoading] = useState(false);
  const [attributionData, setAttributionData] = useState<any[]>([]);

  useEffect(() => {
    if (selectedCampaign) {
      setLoading(true);
      api.campaignAttribution(selectedCampaign)
        .then(res => setAttributionData(res.attribution || []))
        .catch(err => console.error("Attribution fetch failed", err))
        .finally(() => setLoading(false));
    } else {
      setAttributionData([]);
    }
  }, [selectedCampaign]);

  if (!selectedCampaign) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
        <ShieldAlert className="w-8 h-8 mb-3 opacity-20" />
        <p className="text-sm font-semibold text-slate-400">Select a Campaign</p>
        <p className="text-xs mt-2 max-w-[220px]">
          Select a campaign to run MITRE ATT&CK attribution analysis.
        </p>
      </div>
    );
  }

  const getConfidenceClass = (conf: number) => {
    if (conf >= 80) return 'text-red-400 border-red-500/50 bg-red-500/10';
    if (conf >= 50) return 'text-orange-400 border-orange-500/50 bg-orange-500/10';
    return 'text-yellow-400 border-yellow-500/50 bg-yellow-500/10';
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden animate-slide-in p-3 bg-[#060a13] custom-scrollbar space-y-3">
      {loading ? (
        <div className="flex items-center justify-center h-32 text-red-400 animate-pulse text-xs">
          Running Attribution Analysis...
        </div>
      ) : attributionData.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-xs">
          <Crosshair className="w-6 h-6 mb-2 opacity-50" />
          No known threat actors matched these techniques.
        </div>
      ) : (
        attributionData.map((actor, idx) => (
          <div key={idx} className="bg-[#0a0e17] border border-[#1e2d4a] rounded-lg p-3 hover:border-red-500/30 transition-colors relative overflow-hidden">
            <div className="flex justify-between items-start mb-2 relative z-10">
              <div>
                <div className="font-mono text-red-400 font-bold text-base flex items-center gap-2">
                  <Crosshair className="w-4 h-4" />
                  {actor.actor_name}
                </div>
                {actor.description && (
                  <p className="text-[10px] text-slate-400 mt-1 line-clamp-2 max-w-[90%] leading-snug">
                    {actor.description}
                  </p>
                )}
              </div>
              <div className={`px-2 py-1 rounded border ${getConfidenceClass(actor.confidence)} flex flex-col items-end`}>
                <div className="text-[9px] uppercase tracking-wider font-bold mb-0.5 opacity-80">Confidence</div>
                <div className="text-lg font-mono font-bold leading-none">{actor.confidence}%</div>
              </div>
            </div>
            
            <div className="space-y-2 mt-3 text-xs relative z-10">
              {actor.shared_techniques && actor.shared_techniques.length > 0 && (
                <div>
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-1">
                    <Crosshair className="w-3 h-3" /> Matching Techniques
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {actor.shared_techniques.map((t: string, i: number) => (
                      <span key={i} className="px-1.5 py-0.5 bg-[#131c2e] border border-[#1e2d4a] text-indigo-400 rounded text-[10px] font-mono">
                        {t}
                      </span>
                    ))}
                  </div>
                </div>
              )}
              
              <div className="grid grid-cols-2 gap-2 mt-2">
                {actor.malware && actor.malware.length > 0 && (
                  <div className="bg-[#131c2e]/50 border border-[#1e2d4a]/50 p-2 rounded">
                    <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-1">
                      <Bug className="w-3 h-3 text-emerald-400" /> Associated Malware
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {actor.malware.map((m: string, i: number) => (
                        <span key={i} className="text-emerald-400 font-mono text-[10px]">{m}{i < actor.malware.length - 1 ? ',' : ''}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                {actor.tools && actor.tools.length > 0 && (
                  <div className="bg-[#131c2e]/50 border border-[#1e2d4a]/50 p-2 rounded">
                    <div className="text-[9px] text-slate-500 uppercase tracking-wider mb-1 flex items-center gap-1">
                      <Cpu className="w-3 h-3 text-cyan-400" /> Associated Tools
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {actor.tools.map((t: string, i: number) => (
                        <span key={i} className="text-cyan-400 font-mono text-[10px]">{t}{i < actor.tools.length - 1 ? ',' : ''}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
            
            {/* Background decorative gradient */}
            <div className="absolute top-0 right-0 w-32 h-32 bg-red-500/5 rounded-full blur-3xl -mr-10 -mt-10 pointer-events-none"></div>
          </div>
        ))
      )}
    </div>
  );
}
