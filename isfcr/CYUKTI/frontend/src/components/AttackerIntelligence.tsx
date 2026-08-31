import { useDashboard } from '../context/DashboardContext';
import { Skull, AlertTriangle, Activity, Target } from 'lucide-react';
import { PanelWrapper } from './PanelWrapper';

function riskBadgeClass(level?: string) {
  switch (level) {
    case 'CRITICAL': return 'bg-red-500/20 text-red-400 border-red-500/30';
    case 'HIGH': return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
    case 'MEDIUM': return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
    default: return 'bg-blue-500/20 text-blue-400 border-blue-500/30';
  }
}

export function AttackerIntelligence() {
  const { attackers, selectedAttacker, selectAttacker } = useDashboard();

  return (
    <PanelWrapper
      title="Attacker Intel"
      icon={<Skull className="w-4 h-4 text-red-500" />}
      className="h-full"
      headerExtra={
        <span className="text-[10px] text-slate-400 bg-[#0a0e17] px-2 py-0.5 rounded border border-[#1e2d4a] mr-1">
          {attackers.length} Active
        </span>
      }
    >
      <div className="flex-1 overflow-auto bg-[#060a13] p-2 space-y-2 custom-scrollbar">
        {attackers.length > 0 ? (
          attackers.map((attacker) => (
            <div
              key={attacker.attacker_ip}
              onClick={() => selectAttacker(selectedAttacker === attacker.attacker_ip ? null : attacker.attacker_ip)}
              className={`p-3 rounded-lg border transition-all cursor-pointer ${
                selectedAttacker === attacker.attacker_ip
                  ? 'bg-red-500/10 border-red-500/50'
                  : 'bg-[#0c1220] border-[#1e2d4a] hover:border-slate-600'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <div className="font-mono text-sm font-semibold text-slate-200">{attacker.attacker_ip}</div>
                <div className={`flex items-center gap-1 text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded border ${riskBadgeClass(attacker.risk_level)}`}>
                  <AlertTriangle className="w-3 h-3" />
                  {attacker.risk_score}/100
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs text-slate-400 mb-2">
                <div>
                  <span className="text-[9px] uppercase tracking-wider text-slate-500">Campaigns</span>
                  <div className="font-mono text-slate-300">{attacker.campaign_count}</div>
                </div>
                <div>
                  <span className="text-[9px] uppercase tracking-wider text-slate-500">Events</span>
                  <div className="font-mono text-slate-300">{attacker.event_count}</div>
                </div>
                <div>
                  <span className="text-[9px] uppercase tracking-wider text-slate-500">First Seen</span>
                  <div className="text-[10px] text-slate-300">{attacker.first_seen ? new Date(attacker.first_seen).toLocaleDateString() : '—'}</div>
                </div>
                <div>
                  <span className="text-[9px] uppercase tracking-wider text-slate-500">Predicted Next</span>
                  <div className="font-mono text-cyan-400 text-[10px]">{attacker.predicted_technique || '—'}</div>
                </div>
              </div>

              {attacker.techniques_observed && attacker.techniques_observed.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-2">
                  {attacker.techniques_observed.slice(0, 4).map((tech) => (
                    <span key={tech} className="text-[9px] font-mono bg-indigo-500/10 text-indigo-300 px-1.5 py-0.5 rounded border border-indigo-500/20">
                      {tech}
                    </span>
                  ))}
                </div>
              )}

              <div className="flex justify-between items-center text-[10px] text-slate-500 border-t border-[#1e2d4a] pt-2">
                <span className="flex items-center gap-1"><Activity className="w-3 h-3" /> Last Active</span>
                <span>{attacker.last_seen ? new Date(attacker.last_seen).toLocaleString() : '—'}</span>
              </div>
              {attacker.predicted_technique && (
                <div className="flex items-center gap-1 text-[10px] text-cyan-400 mt-1">
                  <Target className="w-3 h-3" /> Next: {attacker.predicted_technique}
                </div>
              )}
            </div>
          ))
        ) : (
          <div className="h-full flex items-center justify-center text-slate-500 text-sm">No active attackers</div>
        )}
      </div>
    </PanelWrapper>
  );
}
