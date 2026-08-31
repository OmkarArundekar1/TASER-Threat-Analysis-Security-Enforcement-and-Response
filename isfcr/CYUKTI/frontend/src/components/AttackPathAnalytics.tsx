import { useState, useEffect } from 'react';
import { Activity, ArrowRight, ShieldAlert, GitMerge } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';

export function AttackPathAnalytics() {
  const { setPathExplorerOpen } = useDashboard();
  const [loading, setLoading] = useState(true);
  const [paths, setPaths] = useState<any[]>([]);

  useEffect(() => {
    setLoading(true);
    api.analyticsPaths()
      .then(res => setPaths(res.paths || []))
      .catch(err => console.error("Attack Paths fetch failed", err))
      .finally(() => setLoading(false));
  }, []);

  const getRiskClass = (level: string) => {
    switch (level) {
      case 'CRITICAL': return 'bg-red-500/10 text-red-400 border-red-500/30';
      case 'HIGH': return 'bg-orange-500/10 text-orange-400 border-orange-500/30';
      case 'MEDIUM': return 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30';
      default: return 'bg-blue-500/10 text-blue-400 border-blue-500/30';
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden animate-slide-in p-3 bg-[#060a13] custom-scrollbar space-y-3">
      {/* Header action for full-screen mode */}
      <div className="flex justify-between items-center bg-[#0a0e17] p-2 rounded border border-[#1e2d4a]">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Activity className="w-4 h-4 text-orange-400" />
          <span>Top Historical Attack Chains</span>
        </div>
        <button 
          onClick={() => setPathExplorerOpen(true)}
          className="text-[10px] bg-orange-500 hover:bg-orange-600 text-white px-2 py-1 rounded transition-colors flex items-center gap-1 font-semibold"
        >
          <GitMerge className="w-3 h-3" /> OPEN PATH EXPLORER
        </button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-32 text-orange-400 animate-pulse text-xs">
          Analyzing Campaign Paths...
        </div>
      ) : paths.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-xs">
          <ShieldAlert className="w-6 h-6 mb-2 opacity-50" />
          No multi-step attack paths found.
        </div>
      ) : (
        <div className="space-y-3 flex-1 overflow-auto pr-1">
          {paths.map((p, idx) => (
            <div key={idx} className="bg-[#0a0e17] border border-[#1e2d4a] rounded-lg p-3 relative overflow-hidden group">
              <div className="absolute top-0 left-0 w-1 h-full bg-slate-800 group-hover:bg-orange-500 transition-colors"></div>
              
              <div className="flex justify-between items-start mb-3 ml-2">
                <div className="flex items-center gap-3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">
                    Observed <span className="text-white font-bold ml-1">{p.frequency}</span> times
                  </div>
                </div>
                <div className={`px-2 py-0.5 rounded text-[9px] font-bold border ${getRiskClass(p.risk_level)}`}>
                  Risk: {p.risk_score} · {p.risk_level}
                </div>
              </div>
              
              {/* Visual Chain */}
              <div className="flex flex-wrap items-center gap-2 ml-2 mt-2">
                {p.path.map((techId: string, i: number) => (
                  <div key={`${techId}-${i}`} className="flex items-center gap-2">
                    <div 
                      className="px-2 py-1 bg-[#131c2e] border border-[#1e2d4a] rounded text-xs font-mono text-indigo-400 shadow-[inset_0_0_10px_rgba(99,102,241,0.05)] max-w-[150px] truncate"
                      title={p.path_names[i] || techId}
                    >
                      {techId}
                    </div>
                    {i < p.path.length - 1 && (
                      <ArrowRight className="w-3 h-3 text-slate-600 flex-shrink-0" />
                    )}
                  </div>
                ))}
              </div>
              
              {/* Example campaigns */}
              <div className="ml-2 mt-3 pt-2 border-t border-[#1e2d4a]/50 text-[9px] text-slate-500 flex gap-1 flex-wrap">
                <span>Seen in:</span>
                {p.example_campaigns.map((camp: string, i: number) => (
                  <span key={i} className="text-orange-400/80 font-mono">{camp}{i < p.example_campaigns.length - 1 ? ',' : ''}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
