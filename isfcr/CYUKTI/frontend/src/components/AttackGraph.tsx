import { useCallback, useRef, useState, useEffect } from 'react';
import ForceGraph2D from 'react-force-graph-2d';
import { Maximize2, ZoomIn, ZoomOut, RefreshCw, Layers, Expand, Crosshair, ListTree } from 'lucide-react';
import { useDashboard } from '../context/DashboardContext';
import { api } from '../services/api';
import { PanelWrapper } from './PanelWrapper';


export function AttackGraph() {
  const { graphData, refreshGraph, loading, graphLayer, setGraphLayer, selectTechnique, selectCampaign, resetKey } = useDashboard();
  const fgRef = useRef<any>(null);
  const [dimensions, setDimensions] = useState({ width: 800, height: 400 });
  const containerRef = useRef<HTMLDivElement>(null);

  const [expandedData, setExpandedData] = useState<{ nodes: any[]; links: any[] }>({ nodes: [], links: [] });
  const [expand2Hops, setExpand2Hops] = useState(false);
  const [showFilters, setShowFilters] = useState(false);
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);
  const [selectedEdge, setSelectedEdge] = useState<any>(null);

  useEffect(() => {
    setExpandedData({ nodes: [], links: [] });
    setSelectedEdge(null);
    if (fgRef.current) {
      fgRef.current.zoomToFit(400, 50);
    }
  }, [resetKey]);

  useEffect(() => {
    const updateDimensions = () => {
      if (containerRef.current) {
        setDimensions({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        });
      }
    };
    window.addEventListener('resize', updateDimensions);
    updateDimensions();
    const timer = setTimeout(updateDimensions, 100);
    return () => {
      window.removeEventListener('resize', updateDimensions);
      clearTimeout(timer);
    };
  }, []);



  const handleNodeClick = useCallback(async (node: any) => {
    if (node.type === 'Technique') {
      selectTechnique(node.attack_id || node.id);
    } else if (node.type === 'Campaign') {
      selectCampaign(node.campaign_id || node.id);
    }

    if (node.type === 'Technique' || graphLayer === 'threat_intel') {
      try {
        const depth = expand2Hops ? 2 : 1;
        const data = await api.graphExpand(node.id, depth);
        if (data?.nodes) {
          setExpandedData((prev) => {
            const newNodes = [...prev.nodes];
            const newLinks = [...prev.links];
            data.nodes.forEach((n) => {
              if (!newNodes.find((ex) => ex.id === n.id)) newNodes.push(n);
            });
            data.links.forEach((l) => {
              if (!newLinks.find((ex) => ex.source === l.source && ex.target === l.target)) newLinks.push(l);
            });
            return { nodes: newNodes, links: newLinks };
          });
        }
      } catch (e) {
        console.error('Expansion failed', e);
      }
    }
  }, [expand2Hops, selectTechnique, selectCampaign, graphLayer]);

  const drawNode = useCallback((node: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    const label = node.label || node.type;
    const fontSize = 11 / globalScale;
    ctx.font = `${fontSize}px Inter, monospace`;

    let color = '#94a3b8';
    let size = 5;
    switch (node.type) {
      case 'Attacker': color = '#ef4444'; size = 8; break;
      case 'Campaign': color = '#f97316'; size = 10; break;
      case 'Technique': color = '#a855f7'; size = 6; break;
      case 'Host': color = '#3b82f6'; size = 7; break;
      case 'AttackEvent': color = '#eab308'; size = 4; break;
      case 'ThreatActor': color = '#dc2626'; size = 9; break;
      case 'Malware': color = '#ec4899'; size = 7; break;
      case 'Tool': color = '#8b5cf6'; size = 7; break;
      case 'Mitigation': color = '#10b981'; size = 7; break;
      case 'Tactic': color = '#14b8a6'; size = 8; break;
    }

    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    ctx.beginPath();
    ctx.arc(node.x, node.y, size, 0, 2 * Math.PI, false);
    ctx.fillStyle = color;
    ctx.fill();
    ctx.strokeStyle = '#fff';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (globalScale > 0.6) {
      const textWidth = ctx.measureText(label).width;
      ctx.fillStyle = 'rgba(12, 18, 32, 0.85)';
      ctx.fillRect(node.x - textWidth / 2 - 2, node.y + size + 2, textWidth + 4, fontSize + 2);
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(label, node.x, node.y + size + 2 + fontSize / 2);
    }
  }, []);

  const drawEdge = useCallback((link: any, ctx: CanvasRenderingContext2D, globalScale: number) => {
    if (!link.source || !link.target) return;
    const start = link.source;
    const end = link.target;

    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);

    let color = 'rgba(42, 63, 102, 0.5)';
    let width = 1 / globalScale;
    let isDashed = false;

    if (link.label === 'LIKELY_NEXT') {
      color = 'rgba(239, 68, 68, 0.9)';
      width = 2.5 / globalScale;
      isDashed = true;
    } else if (link.label === 'NEXT_TECHNIQUE') {
      color = 'rgba(99, 102, 241, 0.6)';
      width = Math.max(1, (link.count || 1) * 0.4) / globalScale;
    } else if (link.label === 'SIMILAR_TO') {
      color = 'rgba(245, 158, 11, 0.9)'; // amber/orange
      width = 2.0 / globalScale;
      isDashed = true;
    } else if (link.label === 'RESEMBLES') {
      color = 'rgba(225, 29, 72, 0.9)'; // crimson
      width = 2.0 / globalScale;
      isDashed = true;
    }

    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(isDashed ? [5 / globalScale, 5 / globalScale] : []);
    ctx.stroke();
    ctx.setLineDash([]);

    if (showEdgeLabels && globalScale > 0.8) {
      const midX = (start.x + end.x) / 2;
      const midY = (start.y + end.y) / 2;
      const fontSize = 8 / globalScale;
      ctx.font = `${fontSize}px Inter`;
      let labelText = link.label;
      if (link.confidence) labelText += ` (${Math.round(link.confidence)}%)`;
      
      if (link.label === 'LIKELY_NEXT' || link.label === 'RESEMBLES') ctx.fillStyle = '#ef4444';
      else if (link.label === 'SIMILAR_TO') ctx.fillStyle = '#f59e0b';
      else ctx.fillStyle = '#94a3b8';
      
      ctx.textAlign = 'center';
      ctx.fillText(labelText, midX, midY);
    }
  }, [showEdgeLabels]);

  let mergedNodes = [...(graphData?.nodes || [])];
  let mergedLinks = [...(graphData?.links || [])];
  expandedData.nodes.forEach((en) => {
    if (!mergedNodes.find((n) => n.id === en.id)) mergedNodes.push(en);
  });
  expandedData.links.forEach((el) => {
    if (!mergedLinks.find((l) => l.source === el.source && l.target === el.target)) mergedLinks.push(el);
  });
  const filteredNodes = mergedNodes;
  const filteredNodeIds = new Set(filteredNodes.map((n) => n.id));
  const filteredLinks = mergedLinks.filter((l) => {
    const sourceId = typeof l.source === 'object' ? (l.source as { id: string }).id : l.source;
    const targetId = typeof l.target === 'object' ? (l.target as { id: string }).id : l.target;
    return filteredNodeIds.has(sourceId) && filteredNodeIds.has(targetId);
  });

  // Custom Neo4j-style forces and layering
  useEffect(() => {
    if (fgRef.current && filteredNodes.length > 0) {
      // Repulsion to prevent overlap
      fgRef.current.d3Force('charge').strength(-400);
      fgRef.current.d3Force('link').distance(60);
      
      // Vertical layering force
      // d3-force mutates plain node objects with x/y/vx/vy at runtime;
      // these aren't part of the static GraphNode type, hence the cast.
      type SimNode = { type?: string; y?: number; vy: number };
      const forceY = (alpha: number) => {
        (filteredNodes as unknown as SimNode[]).forEach(node => {
          let targetY = 0;
          if (node.type === 'Attacker') targetY = -150;
          else if (node.type === 'Campaign') targetY = -50;
          else if (node.type === 'Technique') targetY = 50;
          else if (node.type === 'ThreatActor') targetY = 150;
          else return; // host or other

          // Apply a gentle force towards the target Y
          node.vy += (targetY - (node.y || 0)) * alpha * 0.2;
        });
      };
      fgRef.current.d3Force('layerY', forceY);
    }
  }, [filteredNodes]);

  return (
    <PanelWrapper
      title="Neo4j Attack Graph"
      className="h-full"
      headerExtra={
        <div className="flex items-center gap-1 flex-wrap justify-end">
          {loading && <span className="text-[10px] text-indigo-400 animate-pulse mr-1">Syncing</span>}
          <div className="bg-[#0a0e17] rounded flex border border-[#1e2d4a] text-[10px] overflow-hidden">
            <button
              className={`px-2 py-1 ${graphLayer === 'campaign' ? 'bg-indigo-500 text-white' : 'text-slate-400'}`}
              onClick={() => setGraphLayer('campaign')}
            >
              Campaign
            </button>
            <button
              className={`px-2 py-1 ${graphLayer === 'investigation' ? 'bg-indigo-500 text-white' : 'text-slate-400'}`}
              onClick={() => setGraphLayer('investigation')}
            >
              Investigation
            </button>
            <button
              className={`px-2 py-1 ${graphLayer === 'threat_intel' ? 'bg-indigo-500 text-white' : 'text-slate-400'}`}
              onClick={() => setGraphLayer('threat_intel')}
            >
              Threat Intel
            </button>
          </div>
          <label className="flex items-center gap-1 text-[10px] text-slate-300 cursor-pointer">
            <input type="checkbox" checked={expand2Hops} onChange={(e) => setExpand2Hops(e.target.checked)} className="accent-indigo-500" />
            <Expand className="w-3 h-3" /> 2-Hop
          </label>
          <button onClick={() => setShowFilters(!showFilters)} className="p-1 text-slate-400 hover:text-white" title="Options">
            <Layers className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => { setExpandedData({ nodes: [], links: [] }); refreshGraph(); }} className="p-1 text-slate-400 hover:text-white" title="Refresh Data">
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
          <button onClick={() => fgRef.current?.d3ReheatSimulation()} className="p-1 text-slate-400 hover:text-white" title="Re-Layout"><ListTree className="w-3.5 h-3.5" /></button>
          <button onClick={() => fgRef.current?.centerAt(0, 0, 400)} className="p-1 text-slate-400 hover:text-white" title="Center Graph"><Crosshair className="w-3.5 h-3.5" /></button>
          <button onClick={() => fgRef.current?.zoom(fgRef.current.zoom() * 1.5, 400)} className="p-1 text-slate-400 hover:text-white" title="Zoom In"><ZoomIn className="w-3.5 h-3.5" /></button>
          <button onClick={() => fgRef.current?.zoom(fgRef.current.zoom() / 1.5, 400)} className="p-1 text-slate-400 hover:text-white" title="Zoom Out"><ZoomOut className="w-3.5 h-3.5" /></button>
          <button onClick={() => fgRef.current?.zoomToFit(400, 50)} className="p-1 text-slate-400 hover:text-white" title="Fit to Screen"><Maximize2 className="w-3.5 h-3.5" /></button>
        </div>
      }
    >
      {selectedEdge && (
        <div className="absolute top-14 left-4 w-72 bg-[#0c1220] border border-[#1e2d4a] rounded-lg shadow-2xl z-50 overflow-hidden text-xs animate-slide-in">
          <div className="bg-[#131c2e] p-2 flex justify-between items-center border-b border-[#1e2d4a]">
            <span className="font-bold text-slate-200">Relationship Evidence</span>
            <button onClick={() => setSelectedEdge(null)} className="text-slate-400 hover:text-white">&times;</button>
          </div>
          <div className="p-3 space-y-2">
            <div className="flex gap-2">
              <span className="text-slate-500 w-16">Type:</span>
              <span className="font-mono text-indigo-400">{selectedEdge.label}</span>
            </div>
            {selectedEdge.edge_props?.score && (
              <div className="flex gap-2">
                <span className="text-slate-500 w-16">Score:</span>
                <span className="font-mono text-orange-400">{selectedEdge.edge_props.score}%</span>
              </div>
            )}
            {selectedEdge.edge_props?.confidence && (
              <div className="flex gap-2">
                <span className="text-slate-500 w-16">Confidence:</span>
                <span className="font-mono text-red-400">{selectedEdge.edge_props.confidence}%</span>
              </div>
            )}
            {selectedEdge.edge_props?.shared_techniques && (
              <div>
                <div className="text-slate-500 mb-1">Shared Techniques:</div>
                <div className="flex flex-wrap gap-1">
                  {selectedEdge.edge_props.shared_techniques.map((t: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 bg-[#1a2333] border border-[#2a3f5f] rounded font-mono text-[9px] text-slate-300">{t}</span>
                  ))}
                </div>
              </div>
            )}
            {selectedEdge.edge_props?.shared_tactics && (
              <div className="mt-2">
                <div className="text-slate-500 mb-1">Shared Tactics:</div>
                <div className="flex flex-wrap gap-1">
                  {selectedEdge.edge_props.shared_tactics.map((t: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 bg-[#1a2333] border border-[#2a3f5f] rounded font-mono text-[9px] text-emerald-400">{t}</span>
                  ))}
                </div>
              </div>
            )}
            {selectedEdge.edge_props?.matched_techniques && (
              <div>
                <div className="text-slate-500 mb-1">Matched Techniques:</div>
                <div className="flex flex-wrap gap-1">
                  {selectedEdge.edge_props.matched_techniques.map((t: string, i: number) => (
                    <span key={i} className="px-1.5 py-0.5 bg-[#1a2333] border border-[#2a3f5f] rounded font-mono text-[9px] text-slate-300">{t}</span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
      {showFilters && (
        <div className="absolute right-4 top-14 w-44 bg-[#0c1220] border border-[#1e2d4a] rounded-md shadow-xl z-50 p-2 text-xs">
          <label className="flex items-center gap-2 text-slate-300 py-1">
            <input type="checkbox" checked={showEdgeLabels} onChange={(e) => setShowEdgeLabels(e.target.checked)} className="accent-indigo-500" />
            Show edge labels
          </label>
          <button onClick={() => setExpandedData({ nodes: [], links: [] })} className="text-red-400 mt-2">Clear expansions</button>
        </div>
      )}
      {/* Graph Legend */}
      <div className="absolute bottom-4 left-4 bg-[#0c1220]/90 border border-[#1e2d4a] rounded p-2 text-[10px] shadow-lg pointer-events-none z-40 backdrop-blur-sm">
        <div className="font-bold text-slate-300 mb-1 uppercase">Node Types</div>
        <div className="grid grid-cols-2 gap-x-3 gap-y-1">
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#ef4444]"></div> Attacker</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#f97316]"></div> Campaign</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#a855f7]"></div> Technique</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#dc2626]"></div> Threat Actor</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#3b82f6]"></div> Host</div>
          <div className="flex items-center gap-1"><div className="w-2 h-2 rounded-full bg-[#eab308]"></div> AttackEvent</div>
        </div>
        <div className="font-bold text-slate-300 mt-2 mb-1 uppercase">Edges</div>
        <div className="space-y-1">
          <div className="flex items-center gap-1"><div className="w-4 h-0.5 bg-[#ef4444] border-dashed border-t-2"></div> LIKELY_NEXT (Pred)</div>
          <div className="flex items-center gap-1"><div className="w-4 h-0.5 bg-[#f59e0b] border-dashed border-t-2"></div> SIMILAR_TO</div>
          <div className="flex items-center gap-1"><div className="w-4 h-0.5 bg-[#e11d48] border-dashed border-t-2"></div> RESEMBLES</div>
        </div>
      </div>
      <div className="flex-1 bg-[#060a13] min-h-0" ref={containerRef}>
        {filteredNodes.length > 0 ? (
          <ForceGraph2D
            ref={fgRef}
            width={dimensions.width}
            height={dimensions.height}
            graphData={{ nodes: filteredNodes, links: filteredLinks }}
            nodeCanvasObject={drawNode}
            nodeLabel={(n: any) => n.full_label || n.label || n.id}
            onNodeClick={handleNodeClick}
            onLinkClick={(link: any) => {
              if (link.label === 'SIMILAR_TO' || link.label === 'RESEMBLES') {
                setSelectedEdge(link);
              }
            }}
            linkCanvasObjectMode={() => 'after'}
            linkCanvasObject={drawEdge}
            linkDirectionalArrowLength={3}
            linkDirectionalArrowRelPos={1}
            d3AlphaDecay={0.03}
            d3VelocityDecay={0.35}
            cooldownTicks={80}
            onEngineStop={() => fgRef.current?.zoomToFit(400, 40)}
            backgroundColor="#060a13"
          />
        ) : (
          <div className="h-full flex items-center justify-center text-slate-500 text-sm">
            {loading ? 'Loading graph data...' : 'No graph data available'}
          </div>
        )}
      </div>
    </PanelWrapper>
  );
}
