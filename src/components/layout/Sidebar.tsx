import React from 'react';
import { 
  LayoutDashboard, 
  Compass, 
  Cpu, 
  Layers, 
  Sparkles, 
  Award, 
  CheckCircle2, 
  BookOpen,
  MapPin,
  Calendar,
  ShieldCheck
} from 'lucide-react';
import { NavigationTab } from '../../types';
import { OceanStateReturn } from '../../state/useOceanState';

interface SidebarProps {
  currentTab: NavigationTab;
  onSelectTab: (tab: NavigationTab) => void;
  location: OceanStateReturn['location'];
}

export const Sidebar: React.FC<SidebarProps> = ({ currentTab, onSelectTab, location }) => {
  const navItems: { id: NavigationTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-4 h-4" /> },
    { id: 'explorer', label: 'Ocean Explorer', icon: <Compass className="w-4 h-4" /> },
    { id: 'reconstruction', label: 'Reconstruction', icon: <Cpu className="w-4 h-4" />, badge: 'Core' },
    { id: 'depth-profile', label: 'Depth Profile', icon: <Layers className="w-4 h-4" /> },
    { id: 'embedding', label: 'Embedding', icon: <Sparkles className="w-4 h-4" />, badge: '128-D' },
    { id: 'benchmarks', label: 'Benchmarks', icon: <Award className="w-4 h-4" /> },
    { id: 'validation', label: 'Validation', icon: <CheckCircle2 className="w-4 h-4" /> },
    { id: 'methodology', label: 'Methodology', icon: <BookOpen className="w-4 h-4" /> },
  ];

  return (
    <aside className="w-64 bg-[#070b12] border-r border-slate-800/80 flex flex-col justify-between flex-shrink-0 select-none">
      {/* Brand & System Identity */}
      <div>
        <div className="p-5 border-b border-slate-800/80">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 via-blue-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-white font-black text-lg">
              🌊
            </div>
            <div>
              <div className="text-xs font-extrabold tracking-wider text-cyan-400 uppercase">
                Satellite Embedding
              </div>
              <div className="text-[13px] font-bold text-white tracking-tight leading-none mt-0.5">
                Ocean AI Platform
              </div>
            </div>
          </div>

          {/* Status Indicator Required */}
          <div className="mt-4 flex items-center justify-between px-2.5 py-1.5 rounded-lg bg-slate-900/90 border border-slate-800 text-[11px]">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-slate-300 font-semibold tracking-wide">PROTOTYPE</span>
            </div>
            <span className="text-cyan-400 font-mono text-[10px] font-bold">LOCAL SIMULATION</span>
          </div>
        </div>

        {/* Navigation Items */}
        <nav className="p-3 space-y-1">
          <div className="px-3 py-1.5 text-[10px] font-bold tracking-widest text-slate-500 uppercase">
            Platform Modules
          </div>
          {navItems.map((item) => {
            const isActive = currentTab === item.id;
            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full flex items-center justify-between px-3.5 py-2.5 rounded-xl text-xs font-medium transition-all duration-150 ${
                  isActive
                    ? 'bg-gradient-to-r from-cyan-500/15 to-blue-500/10 text-cyan-300 border border-cyan-500/30 font-semibold shadow-sm shadow-cyan-500/10'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900/60 border border-transparent'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className={isActive ? 'text-cyan-400' : 'text-slate-500'}>
                    {item.icon}
                  </span>
                  <span>{item.label}</span>
                </div>
                {item.badge && (
                  <span
                    className={`text-[10px] px-1.5 py-0.5 rounded font-mono font-bold ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/30'
                        : 'bg-slate-800 text-slate-400'
                    }`}
                  >
                    {item.badge}
                  </span>
                )}
              </button>
            );
          })}
        </nav>
      </div>

      {/* Selected Spatial State Card */}
      <div className="p-4 border-t border-slate-800/80 space-y-3">
        <div className="p-3 rounded-xl bg-slate-900/70 border border-slate-800/80 text-xs space-y-2">
          <div className="text-[10px] uppercase font-bold text-slate-500 tracking-wider flex items-center gap-1.5">
            <MapPin className="w-3 h-3 text-cyan-400" />
            Active Target Point
          </div>
          <div className="flex justify-between items-center text-slate-300">
            <span className="text-slate-400">Region:</span>
            <span className="font-semibold text-white">{location.region}</span>
          </div>
          <div className="flex justify-between items-center font-mono text-[11px] text-slate-300">
            <span className="text-slate-400">Coords:</span>
            <span className="text-cyan-300 font-bold">{location.gridPoint}</span>
          </div>
          <div className="flex justify-between items-center text-slate-300">
            <span className="text-slate-400">Depth:</span>
            <span className="text-blue-300 font-mono font-bold">{location.depth} m</span>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-[10px] text-slate-500 px-1">
          <ShieldCheck className="w-3.5 h-3.5 text-cyan-500" />
          <span>North Indian Ocean (5–30°N, 45–105°E)</span>
        </div>
      </div>
    </aside>
  );
};
