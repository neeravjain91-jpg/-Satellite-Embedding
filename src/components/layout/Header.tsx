import React from 'react';
import { MapPin, Calendar, Layers, Activity, ChevronDown } from 'lucide-react';
import { NavigationTab, CanonicalDepth } from '../../types';
import { OceanStateReturn } from '../../state/useOceanState';
import { CANONICAL_DEPTHS } from '../../mock/oceanData';

interface HeaderProps {
  currentTab: NavigationTab;
  location: OceanStateReturn['location'];
  onUpdateDepth: (d: CanonicalDepth) => void;
  onUpdateDate: (date: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentTab,
  location,
  onUpdateDepth,
  onUpdateDate
}) => {
  const tabTitles: Record<NavigationTab, { title: string; subtitle: string }> = {
    dashboard: {
      title: 'Platform Overview',
      subtitle: 'Spatiotemporal reconstruction of depth-wise ocean temperature from daily surface observations'
    },
    explorer: {
      title: 'Ocean Surface Explorer',
      subtitle: 'Multisatellite daily predictor space across the North Indian Ocean basin'
    },
    reconstruction: {
      title: 'Subsurface Reconstruction Pipeline',
      subtitle: 'B8 Deep Learning Engine: Conv2D patch encoder → GRU → 128-D bottleneck → 15 depths'
    },
    'depth-profile': {
      title: 'Vertical Temperature Profiles',
      subtitle: '15-depth vertical stratification and thermocline structure comparison'
    },
    embedding: {
      title: '128-D Ocean Latent Embedding',
      subtitle: 'Dense spatiotemporal representation learning and hydrographic cluster topology'
    },
    benchmarks: {
      title: 'Model Benchmark Comparison',
      subtitle: 'Standardized evaluation of B0 through B8 on certified full-year 2020 production partition'
    },
    validation: {
      title: 'Scientific Validation & QA',
      subtitle: 'Depth-wise error decomposition, regional basin breakdown, and seasonal fidelity'
    },
    methodology: {
      title: 'Scientific Methodology',
      subtitle: 'Zero data leakage protocol, 6-day purge boundaries, and architecture specification'
    }
  };

  const activeMeta = tabTitles[currentTab] || { title: 'Ocean AI', subtitle: '' };

  return (
    <header className="bg-[#070b12]/90 backdrop-blur-md border-b border-slate-800/80 px-6 py-3.5 sticky top-0 z-40 flex flex-col md:flex-row md:items-center justify-between gap-3">
      {/* View Title */}
      <div>
        <div className="flex items-center gap-2">
          <span className="text-cyan-400 text-xs font-mono font-semibold uppercase tracking-wider">
            Module //
          </span>
          <h1 className="text-sm md:text-base font-bold text-white tracking-tight">
            {activeMeta.title}
          </h1>
        </div>
        <p className="text-xs text-slate-400 hidden sm:block mt-0.5">
          {activeMeta.subtitle}
        </p>
      </div>

      {/* Global Interactive Context Bar */}
      <div className="flex items-center gap-2.5 flex-wrap">
        {/* Location Badge */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <MapPin className="w-3.5 h-3.5 text-cyan-400 flex-shrink-0" />
          <span className="font-semibold text-white">{location.region}</span>
          <span className="text-slate-500 font-mono">({location.gridPoint})</span>
        </div>

        {/* Date Selector */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <Calendar className="w-3.5 h-3.5 text-blue-400 flex-shrink-0" />
          <select
            value={location.date}
            onChange={(e) => onUpdateDate(e.target.value)}
            className="bg-transparent text-white font-mono text-xs focus:outline-none cursor-pointer"
          >
            <option value="2020-12-15" className="bg-slate-900">15 Dec 2020 (Test)</option>
            <option value="2020-11-20" className="bg-slate-900">20 Nov 2020 (Late Fall)</option>
            <option value="2020-10-15" className="bg-slate-900">15 Oct 2020 (Validation)</option>
            <option value="2020-07-15" className="bg-slate-900">15 Jul 2020 (SW Monsoon)</option>
            <option value="2020-03-15" className="bg-slate-900">15 Mar 2020 (Spring Inter)</option>
          </select>
        </div>

        {/* Depth Selector */}
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
          <Layers className="w-3.5 h-3.5 text-indigo-400 flex-shrink-0" />
          <span className="text-slate-400">Depth:</span>
          <select
            value={location.depth}
            onChange={(e) => onUpdateDepth(Number(e.target.value) as CanonicalDepth)}
            className="bg-transparent text-cyan-300 font-mono text-xs font-bold focus:outline-none cursor-pointer"
          >
            {CANONICAL_DEPTHS.map((d) => (
              <option key={d} value={d} className="bg-slate-900">
                {d} m
              </option>
            ))}
          </select>
        </div>

        {/* Scientific Prototype Badge */}
        <div className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-xl bg-cyan-950/40 border border-cyan-800/60 text-[11px] text-cyan-300 font-semibold">
          <Activity className="w-3.5 h-3.5 text-cyan-400" />
          <span>PROTOTYPE • LOCAL SIMULATION</span>
        </div>
      </div>
    </header>
  );
};
