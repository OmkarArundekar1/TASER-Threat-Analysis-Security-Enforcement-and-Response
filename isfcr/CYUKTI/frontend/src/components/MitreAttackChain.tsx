import { useDashboard } from '../context/DashboardContext';
import { Shield, ArrowRight, Play, Pause } from 'lucide-react';
import { useEffect, useState } from 'react';
import { api } from '../services/api';
import type { AttackChainStep, ChainTransition } from '../types';
import { PanelWrapper } from './PanelWrapper';

export function MitreAttackChain() {
  const { selectedCampaign } = useDashboard();
  const [transitions, setTransitions] = useState<ChainTransition[]>([]);
  const [chain, setChain] = useState<AttackChainStep[]>([]);
  const [loading, setLoading] = useState(true);
  const [replayIndex, setReplayIndex] = useState(-1);
  const [isReplaying, setIsReplaying] = useState(false);

  useEffect(() => {
    let mounted = true;
    const fetchChain = async () => {
      setLoading(true);
      try {
        const data = await api.attackChain(selectedCampaign || undefined);
        if (!mounted) return;
        if (data.chain && data.chain.length > 0) {
          setChain(data.chain);
          setTransitions([]);
        } else if (data.transitions) {
          setTransitions(data.transitions);
          setChain([]);
        }
      } catch (e) {
        console.error('Failed to fetch attack chain', e);
      } finally {
        if (mounted) setLoading(false);
      }
    };
    fetchChain();
    setReplayIndex(-1);
    setIsReplaying(false);
    return () => { mounted = false; };
  }, [selectedCampaign]);

  useEffect(() => {
    if (!isReplaying || chain.length === 0) return;
    const timer = setInterval(() => {
      setReplayIndex((prev) => {
        if (prev >= chain.length - 1) {
          setIsReplaying(false);
          return prev;
        }
        return prev + 1;
      });
    }, 1200);
    return () => clearInterval(timer);
  }, [isReplaying, chain]);

  const startReplay = () => {
    setReplayIndex(0);
    setIsReplaying(true);
  };

  return (
    <PanelWrapper title="Attack Path Learning" icon={<Shield className="w-4 h-4 text-indigo-400" />} className="h-full">
      <div className="p-3 flex-1 flex flex-col min-h-0 overflow-hidden">
        {loading ? (
          <div className="text-center text-slate-500 text-sm flex-1 flex items-center justify-center">Loading attack chains...</div>
        ) : chain.length > 0 ? (
          <div className="flex-1 min-h-0 flex flex-col">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[10px] text-slate-500 uppercase">Observed Chain Replay</span>
              <button
                onClick={() => (isReplaying ? setIsReplaying(false) : startReplay())}
                className="flex items-center gap-1 text-[10px] px-2 py-1 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
              >
                {isReplaying ? <Pause className="w-3 h-3" /> : <Play className="w-3 h-3" />}
                {isReplaying ? 'Stop' : 'Replay Attack Chain'}
              </button>
            </div>
            <div className="flex-1 overflow-x-auto custom-scrollbar">
              <div className="flex items-center gap-2 min-w-max py-2">
                {chain.map((step, idx) => (
                  <div key={`${step.technique_id}-${idx}`} className="flex items-center gap-2">
                    <div className={`px-3 py-2 rounded border text-center min-w-[100px] transition-all ${
                      replayIndex >= idx ? 'border-cyan-500 bg-cyan-500/10 scale-105' : 'border-[#1e2d4a] bg-[#0a0e17]'
                    }`}>
                      <div className="text-[9px] text-slate-500 font-mono">
                        {step.first_seen ? new Date(step.first_seen).toLocaleTimeString() : '—'}
                      </div>
                      <div className="text-[10px] font-bold text-slate-200 truncate max-w-[120px]" title={step.name}>{step.name}</div>
                      <div className="text-[9px] font-mono text-indigo-400">{step.technique_id}</div>
                    </div>
                    {idx < chain.length - 1 && <ArrowRight className="w-4 h-4 text-slate-600 flex-none" />}
                  </div>
                ))}
              </div>
            </div>
          </div>
        ) : transitions.length > 0 ? (
          <div className="space-y-2 overflow-auto custom-scrollbar flex-1">
            {transitions.slice(0, 4).map((t, idx) => (
              <div key={idx} className="flex items-center justify-between bg-[#0a0e17] border border-[#1e2d4a] rounded p-2 text-xs">
                <div className="flex items-center gap-2 font-mono">
                  <span className="text-blue-400">{t.from}</span>
                  <ArrowRight className="w-3 h-3 text-slate-500" />
                  <span className="text-red-400">{t.to}</span>
                </div>
                <span className="text-slate-500">{t.count}x</span>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center text-slate-500 text-sm flex-1 flex items-center justify-center">No attack chains learned yet</div>
        )}
      </div>
    </PanelWrapper>
  );
}
