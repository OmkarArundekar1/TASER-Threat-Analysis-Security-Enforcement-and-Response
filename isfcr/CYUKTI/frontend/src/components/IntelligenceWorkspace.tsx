import { useState, useEffect } from 'react';
import { Network, Activity, ShieldAlert, AlertTriangle, FileSearch } from 'lucide-react';
import { PanelWrapper } from './PanelWrapper';
import { ThreatCorrelation } from './ThreatCorrelation';
import { AttackPathAnalytics } from './AttackPathAnalytics';
import { ThreatActorAttribution } from './ThreatActorAttribution';
import { RiskPropagation } from './RiskPropagation';
import { EvidenceInvestigation } from './EvidenceInvestigation';
import { useDashboard } from '../context/DashboardContext';

export function IntelligenceWorkspace() {
  const { resetKey } = useDashboard();
  const [activeTab, setActiveTab] = useState<'correlation' | 'paths' | 'attribution' | 'risk' | 'investigation'>('correlation');

  useEffect(() => {
    setActiveTab('correlation');
  }, [resetKey]);

  return (
    <PanelWrapper title="Intelligence Workspace" className="h-full flex flex-col">
      {/* Tabs Header */}
      <div className="flex bg-[#0a0e17] border-b border-[#1e2d4a] shrink-0">
        <button
          onClick={() => setActiveTab('correlation')}
          className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'correlation' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <Network className="w-3 h-3" /> Correlation
        </button>
        <button
          onClick={() => setActiveTab('paths')}
          className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'paths' ? 'border-orange-500 text-orange-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <Activity className="w-3 h-3" /> Paths
        </button>
        <button
          onClick={() => setActiveTab('attribution')}
          className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'attribution' ? 'border-red-500 text-red-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <ShieldAlert className="w-3 h-3" /> Attribution
        </button>
        <button
          onClick={() => setActiveTab('risk')}
          className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'risk' ? 'border-emerald-500 text-emerald-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <AlertTriangle className="w-3 h-3" /> Risk
        </button>
        <button
          onClick={() => setActiveTab('investigation')}
          className={`flex-1 py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'investigation' ? 'border-indigo-400 text-indigo-300' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <FileSearch className="w-3 h-3" /> Investigate
        </button>
      </div>

      {/* Tab Content Area */}
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {activeTab === 'correlation' && <ThreatCorrelation />}
        {activeTab === 'paths' && <AttackPathAnalytics />}
        {activeTab === 'attribution' && <ThreatActorAttribution />}
        {activeTab === 'risk' && <RiskPropagation />}
        {activeTab === 'investigation' && <EvidenceInvestigation />}
      </div>
    </PanelWrapper>
  );
}
