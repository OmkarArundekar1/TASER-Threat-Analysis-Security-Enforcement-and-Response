import { useState } from 'react';
import { DashboardProvider, useDashboard } from './context/DashboardContext';
import { TopNavBar, type TopLevelView } from './components/TopNavBar';
import { SecurityOverview } from './components/SecurityOverview';
import { LiveEventsFeed } from './components/LiveEventsFeed';
import { AttackGraph } from './components/AttackGraph';
import { MitreAttackChain } from './components/MitreAttackChain';
import { CampaignIntelligence } from './components/CampaignIntelligence';
import { QueryConsole } from './components/QueryConsole';
import { AttackerIntelligence } from './components/AttackerIntelligence';
import { IntelligenceWorkspace } from './components/IntelligenceWorkspace';
import { PathExplorer } from './components/PathExplorer';
import { GNNIntelligencePage } from './components/GNNIntelligencePage';
import { PredictionIntelligencePage } from './components/PredictionIntelligencePage';
import { ThreatIntelligencePage } from './components/ThreatIntelligencePage';
import { SystemHealthPage } from './components/SystemHealthPage';
import { AuditLogPage } from './components/AuditLogPage';

function DashboardLayout() {
  const { isPathExplorerOpen } = useDashboard();
  const [activeView, setActiveView] = useState<TopLevelView>('dashboard');

  return (
    <div className="h-screen flex flex-col bg-[#060a13] text-slate-200 overflow-hidden">
      <TopNavBar activeView={activeView} onChangeView={setActiveView} />

      {activeView === 'gnn' ? (
        <GNNIntelligencePage />
      ) : activeView === 'prediction' ? (
        <PredictionIntelligencePage />
      ) : activeView === 'threat-intel' ? (
        <ThreatIntelligencePage />
      ) : activeView === 'system' ? (
        <SystemHealthPage />
      ) : activeView === 'audit' ? (
        <AuditLogPage />
      ) : (
        <>
      {isPathExplorerOpen && <PathExplorer />}

      <main className="flex-1 p-2 md:p-3 flex flex-col gap-2 md:gap-3 overflow-hidden min-h-0">
        <div className="flex-none shrink-0">
          <SecurityOverview />
        </div>

        <div className="flex-1 flex flex-col xl:flex-row gap-2 md:gap-3 min-h-0">
          {/* Left Column - Campaign Summary / Event Context */}
          <div className="w-full xl:w-[25%] flex flex-col gap-2 md:gap-3 min-h-0 overflow-y-auto custom-scrollbar pr-1">
            <div className="flex-[0.4] min-h-[300px] flex flex-col">
              <CampaignIntelligence />
            </div>
            <div className="flex-[0.6] min-h-[400px] flex flex-col">
              <LiveEventsFeed />
            </div>
          </div>

          {/* Center Column - Investigation Graph (primary focus) */}
          <div className="w-full xl:w-[50%] flex flex-col gap-2 md:gap-3 min-h-0 overflow-y-auto custom-scrollbar pr-1">
            <div className="flex-[0.6] min-h-[500px] flex flex-col">
              <AttackGraph />
            </div>
            <div className="flex-[0.15] min-h-[150px] flex flex-col">
              <MitreAttackChain />
            </div>
            <div className="flex-[0.25] min-h-[300px] flex flex-col">
              <QueryConsole />
            </div>
          </div>

          {/* Right Column - Intelligence Workspace */}
          <div className="w-full xl:w-[25%] flex flex-col gap-2 md:gap-3 min-h-0 overflow-y-auto custom-scrollbar pr-1">
            <div className="flex-[0.6] min-h-[400px] flex flex-col">
              <IntelligenceWorkspace />
            </div>
            <div className="flex-[0.4] min-h-[300px] flex flex-col">
              <AttackerIntelligence />
            </div>
          </div>
        </div>
      </main>
        </>
      )}
    </div>
  );
}

function App() {
  return (
    <DashboardProvider>
      <DashboardLayout />
    </DashboardProvider>
  );
}

export default App;
