import { useDashboard } from '../context/DashboardContext';
import { AlertTriangle, Activity, Users, Target, Zap, Server } from 'lucide-react';

export function SecurityOverview() {
  const { overview } = useDashboard();

  const cards = [
    {
      title: 'Total Events',
      value: overview?.total_events?.toLocaleString() || '0',
      icon: AlertTriangle,
      color: 'blue',
    },
    {
      title: 'Active Campaigns',
      value: overview?.active_campaigns || '0',
      icon: Activity,
      color: 'orange',
    },
    {
      title: 'Unique Attackers',
      value: overview?.unique_attackers || '0',
      icon: Users,
      color: 'red',
    },
    {
      title: 'Techniques Detected',
      value: overview?.techniques_detected || '0',
      icon: Target,
      color: 'indigo',
    },
    {
      title: 'Learned Transitions',
      value: overview?.learned_transitions || '0',
      icon: Zap,
      color: 'cyan',
    },
    {
      title: 'Total Hosts',
      value: overview?.total_hosts || '0',
      icon: Server,
      color: 'yellow',
    },
  ];

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'red': return 'text-red-400 bg-red-500/10 border-red-500/30';
      case 'orange': return 'text-orange-400 bg-orange-500/10 border-orange-500/30';
      case 'yellow': return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
      case 'blue': return 'text-blue-400 bg-blue-500/10 border-blue-500/30';
      case 'indigo': return 'text-indigo-400 bg-indigo-500/10 border-indigo-500/30';
      case 'cyan': return 'text-cyan-400 bg-cyan-500/10 border-cyan-500/30';
      default: return 'text-slate-400 bg-slate-500/10 border-slate-500/30';
    }
  };

  return (
    <div className="flex flex-wrap items-center gap-2 md:gap-4">
      {cards.map((card, i) => {
        const Icon = card.icon;
        return (
          <div key={i} className={`flex-1 min-w-[120px] glass-card px-3 py-2 flex items-center justify-between border ${getColorClasses(card.color)}`}>
            <div className="flex items-center gap-2">
              <Icon className="w-4 h-4 opacity-80" />
              <span className="text-[10px] md:text-xs font-semibold uppercase tracking-wider opacity-80">{card.title}</span>
            </div>
            <span className="text-sm md:text-base font-bold font-mono ml-2 text-white">{card.value}</span>
          </div>
        );
      })}
    </div>
  );
}
