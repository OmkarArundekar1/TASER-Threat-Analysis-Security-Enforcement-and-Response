import { useState, useEffect, type ReactNode } from 'react';
import { ChevronDown, ChevronUp, Maximize2, Minimize2 } from 'lucide-react';

interface PanelWrapperProps {
  title: string;
  icon?: ReactNode;
  children: ReactNode;
  className?: string;
  defaultCollapsed?: boolean;
  headerExtra?: ReactNode;
}

export function PanelWrapper({
  title,
  icon,
  children,
  className = '',
  defaultCollapsed = false,
  headerExtra,
}: PanelWrapperProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && fullscreen) {
        setFullscreen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [fullscreen]);

  return (
    <div
      className={`glass-card flex flex-col overflow-hidden transition-all ${
        fullscreen ? 'fixed inset-4 z-[100]' : `h-full min-h-0 ${className}`
      }`}
    >
      <div className="p-2 border-b border-[#1e2d4a] flex items-center justify-between bg-[#131c2e]/50 flex-none">
        <h2 className="text-xs font-semibold uppercase tracking-wider flex items-center gap-1.5 opacity-90">
          {icon}
          {title}
        </h2>
        <div className="flex items-center gap-1">
          {headerExtra}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-[#1e2d4a] rounded"
            title={collapsed ? 'Expand panel' : 'Collapse panel'}
          >
            {collapsed ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
          </button>
          <button
            onClick={() => setFullscreen(!fullscreen)}
            className="p-1.5 text-slate-400 hover:text-white hover:bg-[#1e2d4a] rounded"
            title={fullscreen ? 'Exit fullscreen' : 'Fullscreen'}
          >
            {fullscreen ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      </div>
      {!collapsed && (
        <div className="flex-1 min-h-0 overflow-hidden flex flex-col">{children}</div>
      )}
    </div>
  );
}
