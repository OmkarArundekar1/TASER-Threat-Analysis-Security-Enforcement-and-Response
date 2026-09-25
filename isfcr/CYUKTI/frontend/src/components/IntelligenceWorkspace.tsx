import { useState, useEffect } from 'react';
import { Network, Activity, ShieldAlert, AlertTriangle, FileSearch, GitBranch, ShieldCheck, GitCompareArrows } from 'lucide-react';
import { PanelWrapper } from './PanelWrapper';
import { ThreatCorrelation } from './ThreatCorrelation';
import { AttackPathAnalytics } from './AttackPathAnalytics';
import { ThreatActorAttribution } from './ThreatActorAttribution';
import { RiskPropagation } from './RiskPropagation';
import { EvidenceInvestigation } from './EvidenceInvestigation';
import { TopologyIntelligence } from './TopologyIntelligence';
import { RecommendationEngine } from './RecommendationEngine';
import { CampaignSelectionPanel } from './CampaignSelectionPanel';
import { useDashboard } from '../context/DashboardContext';

type WorkspaceTab = 'correlation' | 'paths' | 'attribution' | 'risk' | 'topology' | 'selection' | 'playbook' | 'investigation';

export function IntelligenceWorkspace() {
  const { resetKey } = useDashboard();
  const [activeTab, setActiveTab] = useState<WorkspaceTab>('correlation');

  useEffect(() => {
    setActiveTab('correlation');
  }, [resetKey]);

  return (
    <PanelWrapper title="Intelligence Workspace" className="h-full flex flex-col">
      {/* Tabs Header */}
      <div className="flex bg-[#0a0e17] border-b border-[#1e2d4a] shrink-0 overflow-x-auto custom-scrollbar">
        <button
          onClick={() => setActiveTab('correlation')}
          className={`flex-1 min-w-[64px] py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'correlation' ? 'border-indigo-500 text-indigo-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <Network className="w-3 h-3" /> Correlation
        </button>
        <button
          onClick={() => setActiveTab('topology')}
          className={`flex-1 min-w-[64px] py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'topology' ? 'border-cyan-500 text-cyan-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <GitBranch className="w-3 h-3" /> Topology
        </button>
        <button
          onClick={() => setActiveTab('selection')}
          className={`flex-1 min-w-[64px] py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'selection' ? 'border-fuchsia-500 text-fuchsia-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <GitCompareArrows className="w-3 h-3" /> Selection
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
          onClick={() => setActiveTab('playbook')}
          className={`flex-1 min-w-[64px] py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'playbook' ? 'border-emerald-500 text-emerald-400' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <ShieldCheck className="w-3 h-3" /> Playbook
        </button>
        <button
          onClick={() => setActiveTab('investigation')}
          className={`flex-1 min-w-[64px] py-2 text-[10px] font-bold uppercase tracking-wider border-b-2 transition-colors flex items-center justify-center gap-1.5 ${
            activeTab === 'investigation' ? 'border-indigo-400 text-indigo-300' : 'border-transparent text-slate-500 hover:text-slate-300'
          }`}
        >
          <FileSearch className="w-3 h-3" /> Investigate
        </button>
      </div>

      {/* Tab Content Area */}
      <div className="flex-1 min-h-0 overflow-hidden relative">
        {activeTab === 'correlation' && <ThreatCorrelation />}
        {activeTab === 'topology' && <TopologyIntelligence />}
        {activeTab === 'selection' && <CampaignSelectionPanel />}
        {activeTab === 'paths' && <AttackPathAnalytics />}
        {activeTab === 'attribution' && <ThreatActorAttribution />}
        {activeTab === 'risk' && <RiskPropagation />}
        {activeTab === 'playbook' && <RecommendationEngine />}
        {activeTab === 'investigation' && <EvidenceInvestigation />}
      </div>
    </PanelWrapper>
  );
}
