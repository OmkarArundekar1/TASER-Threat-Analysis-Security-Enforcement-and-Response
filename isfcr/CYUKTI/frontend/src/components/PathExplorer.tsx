import { useState, useEffect, useCallback, useRef } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { X, Globe, Crosshair, Clock, Play, Pause, FastForward, SkipBack, ShieldAlert } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';

export function PathExplorer() {
  const { setPathExplorerOpen, selectedCampaign } = useDashboard();
  const [scope, setScope] = useState<'global' | 'campaign'>('global');
  const [timeframe, setTimeframe] = useState<'24h' | '7d' | '30d' | 'all'>('all');
  const [graphData, setGraphData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [loading, setLoading] = useState(true);
  const [selectedNode, setSelectedNode] = useState<any>(null);
  const fgRef = useRef<any>();
  const [dimensions, setDimensions] = useState({ width: window.innerWidth, height: window.innerHeight });

  // Replay State
  const [isReplaying, setIsReplaying] = useState(false);
  const [replayIndex, setReplayIndex] = useState(-1);
  const [replayPath, setReplayPath] = useState<string[]>([]); // Array of node IDs representing a sequence
  
  // Close handler
  const handleClose = () => {
    setPathExplorerOpen(false);
  };

  useEffect(() => {
    const updateDimensions = () => setDimensions({ width: window.innerWidth, height: window.innerHeight });
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  const fetchGraphData = useCallback(async () => {
    setLoading(true);
    try {
      const campId = scope === 'campaign' && selectedCampaign ? selectedCampaign : undefined;
      const tf = timeframe !== 'all' ? timeframe : undefined;
      const data = await api.pathGraph(campId, tf);
      setGraphData(data);
    } catch (e) {
      console.error("Path Explorer Graph fetch failed", e);
    } finally {
      setLoading(false);
    }
  }, [scope, timeframe, selectedCampaign]);

  useEffect(() => {
    fetchGraphData();
  }, [fetchGraphData]);

  // Replay Logic
  useEffect(() => {
    let timer: any;
    if (isReplaying && replayPath.length > 0) {
      timer = setInterval(() => {
        setReplayIndex(prev => {
          if (prev >= replayPath.length - 1) {
            setIsReplaying(false);
            return prev;
          }
          return prev + 1;
        });
      }, 1500); // 1.5s per step
    }
    return () => clearInterval(timer);
  }, [isReplaying, replayPath]);

  const startReplay = () => {
    // If we have a sequence, let's start it.
    // For a real implementation, we could find the longest chain in the current graph
    // Or if a node is selected, find a path from it.
    // For now, let's just pick a simple traversal or just highlight nodes in order of frequency
    if (graphData.nodes.length === 0) return;
    
    // Sort nodes by frequency as a proxy for a path if we don't have a specific sequence
    // A better approach is to trace NEXT_TECHNIQUE edges
    const startNode = graphData.nodes.reduce((max, node) => (node.frequency > (max?.frequency || 0) ? node : max), null);
    
    if (startNode) {
      // Very basic BFS to build a path
      const path = [startNode.id];
      let current = startNode.id;
      for (let i = 0; i < 5; i++) {
        const nextLink = graphData.links.find(l => l.source.id === current || l.source === current);
        if (nextLink) {
          const nextId = typeof nextLink.target === 'object' ? nextLink.target.id : nextLink.target;
          if (!path.includes(nextId)) {
            path.push(nextId);
            current = nextId;
          } else break;
        } else break;
      }
      setReplayPath(path);
      setReplayIndex(0);
      setIsReplaying(true);
    }
  };

  const drawNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    // Base size depends on frequency or max_tps
    const baseSize = 4 + Math.log10((node.frequency || 1) + 1) * 3;
    const size = baseSize;
    
    // Check if node is part of active replay
    const isReplayActive = replayIndex >= 0 && replayPath.includes(node.id);
    const isCurrentReplay = isReplayActive && replayPath[replayIndex] === node.id;
    
    const isSelected = selectedNode?.id === node.id;

    let color = '#a855f7'; // default technique color
    if (isCurrentReplay) color = '#f97316'; // active replay step
    else if (isReplayActive && replayPath.indexOf(node.id) < replayIndex) color = '#10b981'; // completed replay step
    else if (isSelected) color = '#3b82f6';
    
    // Node Circle
    ctx.shadowColor = color;
    ctx.shadowBlur = isCurrentReplay || isSelected ? 15 : 5;
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();
    
    // Stroke
    ctx.strokeStyle = isSelected ? '#fff' : '#1e2d4a';
    ctx.lineWidth = isSelected ? 2 : 1;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Label
    if (globalScale > 0.8 || isSelected || isCurrentReplay) {
      const label = node.label || node.attack_id;
      const fontSize = (isSelected || isCurrentReplay ? 14 : 10) / globalScale;
      ctx.font = `${fontSize}px Inter`;
      const textWidth = ctx.measureText(label).width;
      
      ctx.fillStyle = 'rgba(12, 18, 32, 0.85)';
      ctx.fillRect(node.x - textWidth / 2 - 2, node.y + size + 2, textWidth + 4, fontSize + 2);
      
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = isCurrentReplay ? '#f97316' : '#e2e8f0';
      ctx.fillText(label, node.x, node.y + size + 2 + fontSize / 2);
    }
  }, [selectedNode, replayIndex, replayPath]);

  const drawEdge = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (!link.source || !link.target) return;
    const start = link.source;
    const end = link.target;

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);

    let color = 'rgba(99, 102, 241, 0.4)';
    let width = 1 / globalScale;
    let isDashed = false;

    if (link.label === 'LIKELY_NEXT') {
      color = 'rgba(239, 68, 68, 0.6)';
      width = 2 / globalScale;
      isDashed = true;
    } else if (link.label === 'NEXT_TECHNIQUE') {
      // Scale thickness by count
      width = Math.min(8, Math.max(1, (link.count || 1) * 0.5)) / globalScale;
    }
    
    // Highlight replay edges
    if (replayIndex > 0 && replayPath.includes(start.id) && replayPath.includes(end.id)) {
      const startIdx = replayPath.indexOf(start.id);
      const endIdx = replayPath.indexOf(end.id);
      if (endIdx === startIdx + 1 && endIdx <= replayIndex) {
        color = 'rgba(249, 115, 22, 0.9)'; // Active trace
        width = 4 / globalScale;
      }
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(isDashed ? [5 / globalScale, 5 / globalScale] : []);
    ctx.stroke();
    ctx.setLineDash([]);
  }, [replayIndex, replayPath]);

  return (
    <div className="fixed inset-0 z-50 bg-[#060a13] flex flex-col font-sans animate-fade-in">
      {/* Top Header */}
      <div className="h-14 bg-[#0a0e17] border-b border-[#1e2d4a] flex items-center justify-between px-4 shrink-0">
        <div className="flex items-center gap-3">
          <div className="bg-orange-500/10 p-2 rounded text-orange-400">
            <Globe className="w-5 h-5" />
          </div>
          <div>
            <h1 className="text-white font-bold tracking-wide">Attack Path Explorer</h1>
            <p className="text-[10px] text-slate-400 uppercase tracking-widest">Global Technique Traversal</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Scope Toggle */}
          <div className="flex bg-[#131c2e] border border-[#1e2d4a] rounded-lg overflow-hidden p-0.5">
            <button
              onClick={() => setScope('global')}
              className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                scope === 'global' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:text-slate-300'
              }`}
            >
              <Globe className="w-3.5 h-3.5" /> Global
            </button>
            <button
              onClick={() => setScope('campaign')}
              disabled={!selectedCampaign}
              className={`px-3 py-1.5 text-xs font-bold rounded-md flex items-center gap-1.5 transition-colors ${
                scope === 'campaign' ? 'bg-indigo-500 text-white' : 'text-slate-400 hover:text-slate-300'
              } ${!selectedCampaign && 'opacity-50 cursor-not-allowed'}`}
            >
              <Crosshair className="w-3.5 h-3.5" /> Campaign
            </button>
          </div>

          {/* Time Filter */}
          <div className="flex items-center gap-2 bg-[#131c2e] border border-[#1e2d4a] rounded-lg px-2 py-1">
            <Clock className="w-3.5 h-3.5 text-slate-400" />
            <select 
              value={timeframe} 
              onChange={e => setTimeframe(e.target.value as any)}
              className="bg-transparent text-xs text-white outline-none border-none cursor-pointer"
            >
              <option value="24h">Last 24 Hours</option>
              <option value="7d">Last 7 Days</option>
              <option value="30d">Last 30 Days</option>
              <option value="all">All Time</option>
            </select>
          </div>

          {/* Replay Controls */}
          <div className="flex bg-[#131c2e] border border-[#1e2d4a] rounded-lg overflow-hidden">
            <button onClick={() => { setReplayIndex(-1); setReplayPath([]); }} className="p-2 text-slate-400 hover:text-white" title="Reset Replay">
              <SkipBack className="w-4 h-4" />
            </button>
            <button 
              onClick={isReplaying ? () => setIsReplaying(false) : startReplay} 
              className={`p-2 transition-colors ${isReplaying ? 'bg-orange-500/20 text-orange-400' : 'text-slate-400 hover:text-white'}`}
            >
              {isReplaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
            </button>
            <button onClick={() => setReplayIndex(prev => prev + 1)} disabled={!isReplaying && replayIndex === -1} className="p-2 text-slate-400 hover:text-white disabled:opacity-50">
              <FastForward className="w-4 h-4" />
            </button>
          </div>

          <button onClick={handleClose} className="p-2 bg-red-500/10 text-red-400 hover:bg-red-500/20 rounded border border-red-500/20 transition-colors ml-4">
            <X className="w-5 h-5" />
          </button>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex-1 flex relative overflow-hidden">
        {/* Graph */}
        <div className="flex-1 bg-[#03060d]">
          {loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-indigo-400 animate-pulse">
              Loading Path Graph...
            </div>
          ) : graphData.nodes.length === 0 ? (
            <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
              <ShieldAlert className="w-12 h-12 mb-4 opacity-20" />
              <p>No technique paths found for the selected filters.</p>
            </div>
          ) : (
            <ForceGraph2D
              ref={fgRef}
              width={dimensions.width - (selectedNode ? 320 : 0)} // Subtract sidebar width if open
              height={dimensions.height - 56}
              graphData={graphData}
              nodeCanvasObject={drawNode}
              linkCanvasObject={drawEdge}
              linkDirectionalArrowLength={4}
              linkDirectionalArrowRelPos={1}
              onNodeClick={(node) => setSelectedNode(node)}
              onBackgroundClick={() => setSelectedNode(null)}
              d3AlphaDecay={0.02}
              d3VelocityDecay={0.3}
              cooldownTicks={100}
              onEngineStop={() => fgRef.current?.zoomToFit(400, 100)}
            />
          )}
        </div>

        {/* Detail Panel */}
        {selectedNode && (
          <div className="w-80 bg-[#0a0e17] border-l border-[#1e2d4a] flex flex-col h-full animate-slide-in-right overflow-y-auto custom-scrollbar">
            <div className="p-4 border-b border-[#1e2d4a] bg-[#0c1220] sticky top-0 z-10">
              <div className="flex justify-between items-start mb-1">
                <span className="px-2 py-0.5 bg-indigo-500/20 text-indigo-400 rounded text-[10px] font-mono font-bold uppercase tracking-wider border border-indigo-500/30">
                  {selectedNode.attack_id}
                </span>
                <button onClick={() => setSelectedNode(null)} className="text-slate-500 hover:text-white">
                  <X className="w-4 h-4" />
                </button>
              </div>
              <h2 className="text-lg font-bold text-white leading-tight mt-2">{selectedNode.label}</h2>
              <div className="text-xs text-slate-400 mt-1 uppercase tracking-widest">{selectedNode.stage}</div>
            </div>
            
            <div className="p-4 space-y-6 flex-1">
              {/* Stats */}
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-[#131c2e] border border-[#1e2d4a] rounded p-3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Historical Freq</div>
                  <div className="text-xl font-mono text-white font-bold">{selectedNode.frequency || 0}</div>
                </div>
                <div className="bg-[#131c2e] border border-[#1e2d4a] rounded p-3">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider mb-1">Max Risk (TPS)</div>
                  <div className="text-xl font-mono text-red-400 font-bold">{selectedNode.max_tps || 0}</div>
                </div>
                <div className="bg-[#131c2e] border border-[#1e2d4a] rounded p-3 col-span-2 flex justify-between items-center">
                  <div className="text-[10px] text-slate-500 uppercase tracking-wider">Campaign Count</div>
                  <div className="text-lg font-mono text-orange-400 font-bold">{selectedNode.campaign_count || 0}</div>
                </div>
              </div>

              {/* Description */}
              {selectedNode.description && (
                <div>
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2">Description</h3>
                  <p className="text-xs text-slate-400 leading-relaxed max-h-32 overflow-y-auto custom-scrollbar pr-2">
                    {selectedNode.description}
                  </p>
                </div>
              )}

              {/* Mitigations */}
              {selectedNode.mitigations && selectedNode.mitigations.length > 0 && (
                <div>
                  <h3 className="text-xs font-bold text-slate-300 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <ShieldAlert className="w-3.5 h-3.5 text-emerald-500" /> Suggested Mitigations
                  </h3>
                  <div className="space-y-2">
                    {selectedNode.mitigations.map((m: any, i: number) => (
                      <div key={i} className="bg-[#131c2e] border border-emerald-500/20 rounded p-2">
                        <div className="text-xs font-bold text-emerald-400 mb-1">{m.name}</div>
                        <p className="text-[10px] text-slate-400 line-clamp-3">{m.description}</p>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
