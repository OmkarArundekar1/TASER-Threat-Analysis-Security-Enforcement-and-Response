import { useState, useEffect } from 'react';
import { AlertTriangle, Activity, Database, Crosshair, Server, Bug } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';

export function RiskPropagation() {
  const { selectedCampaign } = useDashboard();
  const [loading, setLoading] = useState(false);
  const [propagationData, setPropagationData] = useState<any[]>([]);

  useEffect(() => {
    if (selectedCampaign) {
      setLoading(true);
      api.riskPropagation(selectedCampaign)
        .then(res => setPropagationData(res.propagation || []))
        .catch(err => console.error("Risk propagation fetch failed", err))
        .finally(() => setLoading(false));
    } else {
      setPropagationData([]);
    }
  }, [selectedCampaign]);

  if (!selectedCampaign) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center text-slate-500 p-4 text-center">
        <AlertTriangle className="w-8 h-8 mb-3 opacity-20" />
        <p className="text-sm font-semibold text-slate-400">Select a Campaign</p>
        <p className="text-xs mt-2 max-w-[220px]">
          Select a campaign to view how risk propagates across the graph to specific hosts.
        </p>
      </div>
    );
  }

  const getIcon = (type: string) => {
    switch (type) {
      case 'Attacker': return <Crosshair className="w-4 h-4 text-red-400" />;
      case 'Campaign': return <Activity className="w-4 h-4 text-orange-400" />;
      case 'Technique': return <Bug className="w-4 h-4 text-indigo-400" />;
      case 'Host': return <Server className="w-4 h-4 text-blue-400" />;
      default: return <Database className="w-4 h-4 text-slate-400" />;
    }
  };

  const getColorClass = (type: string) => {
    switch (type) {
      case 'Attacker': return 'border-red-500/30 bg-red-500/5 hover:border-red-500/60';
      case 'Campaign': return 'border-orange-500/30 bg-orange-500/5 hover:border-orange-500/60';
      case 'Technique': return 'border-indigo-500/30 bg-indigo-500/5 hover:border-indigo-500/60';
      case 'Host': return 'border-blue-500/30 bg-blue-500/5 hover:border-blue-500/60';
      default: return 'border-slate-500/30 bg-slate-500/5 hover:border-slate-500/60';
    }
  };

  const getBarColorClass = (type: string) => {
    switch (type) {
      case 'Attacker': return 'bg-red-500';
      case 'Campaign': return 'bg-orange-500';
      case 'Technique': return 'bg-indigo-500';
      case 'Host': return 'bg-blue-500';
      default: return 'bg-slate-500';
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden animate-slide-in p-3 bg-[#060a13] custom-scrollbar space-y-3">
      {loading ? (
        <div className="flex items-center justify-center h-32 text-emerald-400 animate-pulse text-xs">
          Calculating Risk Vectors...
        </div>
      ) : propagationData.length === 0 ? (
        <div className="flex flex-col items-center justify-center h-32 text-slate-500 text-xs">
          <AlertTriangle className="w-6 h-6 mb-2 opacity-50" />
          No risk propagation data found for this campaign.
        </div>
      ) : (
        <div className="space-y-3 flex-1 overflow-auto pr-1">
          {propagationData.map((node, idx) => (
            <div key={idx} className={`border rounded-lg p-3 transition-colors relative overflow-hidden ${getColorClass(node.node_type)}`}>
              <div className="flex justify-between items-start mb-2">
                <div className="flex items-center gap-2">
                  <div className="p-1.5 bg-[#0a0e17] rounded-md border border-[#1e2d4a]">
                    {getIcon(node.node_type)}
                  </div>
                  <div>
                    <div className="text-[10px] uppercase tracking-wider text-slate-500">{node.node_type}</div>
                    <div className="font-mono font-bold text-slate-200 text-sm truncate max-w-[180px]" title={node.node_name}>
                      {node.node_name}
                    </div>
                  </div>
                </div>
                
                <div className="text-right">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-0.5">Risk Score</div>
                  <div className="font-mono font-bold text-lg leading-none">{node.risk_score}</div>
                </div>
              </div>
              
              <div className="mt-3">
                <div className="flex justify-between text-[10px] text-slate-400 mb-1">
                  <span>Risk Contribution Vector</span>
                  <span className="font-mono">{node.contribution_percent}%</span>
                </div>
                <div className="w-full h-1.5 bg-[#131c2e] rounded-full overflow-hidden">
                  <div 
                    className={`h-full rounded-full ${getBarColorClass(node.node_type)}`}
                    style={{ width: `${Math.max(2, node.contribution_percent)}%` }}
                  />
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
