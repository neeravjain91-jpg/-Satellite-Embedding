import React from 'react';
import { 
  ArrowRight, 
  Radio, 
  Compass
} from 'lucide-react';
import { OceanStateReturn } from '../../state/useOceanState';
import { OceanMap } from '../shared/OceanMap';
import { getDepthProfile } from '../../mock/profiles';

interface DashboardViewProps {
  oceanState: OceanStateReturn;
}

export const DashboardView: React.FC<DashboardViewProps> = ({ oceanState }) => {
  const { location, updateCoordinates, setCurrentTab } = oceanState;

  // Mini preview data
  const profilePreview = getDepthProfile(location.lat, location.lon, location.date);
  const targetPoint = profilePreview.find(p => p.depth === location.depth) || profilePreview[7];

  return (
    <div className="space-y-6">
      {/* Hero Welcome Banner */}
      <div className="relative rounded-3xl bg-gradient-to-r from-[#071120] via-[#09152a] to-[#040813] border border-cyan-900/30 p-6 md:p-8 overflow-hidden shadow-2xl">
        <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/5 rounded-full blur-3xl pointer-events-none"></div>
        <div className="relative z-10 max-w-3xl space-y-3">
          <div className="flex items-center gap-2">
            <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold tracking-wider uppercase bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
              Deep Learning Research Framework
            </span>
            <span className="text-slate-500 text-xs font-mono">• 2020 Production Dataset</span>
          </div>

          <h1 className="text-2xl sm:text-3xl lg:text-4xl font-extrabold text-white tracking-tight leading-tight">
            SATELLITE EMBEDDING
            <span className="block text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-blue-400 to-indigo-300">
              Subsurface Ocean Temperature Reconstruction
            </span>
          </h1>

          <p className="text-slate-300 text-xs sm:text-sm font-normal leading-relaxed max-w-2xl">
            Spatiotemporal reconstruction of depth-wise ocean temperature from daily surface observations across the North Indian Ocean (5°N–30°N, 45°E–105°E).
          </p>

          <div className="pt-2 flex flex-wrap items-center gap-3">
            <button
              onClick={() => setCurrentTab('reconstruction')}
              className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-lg shadow-cyan-500/20 transition flex items-center gap-2 group"
            >
              <span>Open Reconstruction</span>
              <ArrowRight className="w-3.5 h-3.5 group-hover:translate-x-0.5 transition-transform" />
            </button>

            <button
              onClick={() => setCurrentTab('explorer')}
              className="px-4 py-2.5 rounded-xl bg-slate-900/80 hover:bg-slate-800 text-slate-200 font-semibold text-xs border border-slate-700/80 transition flex items-center gap-2"
            >
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              <span>Explore Surface Observations</span>
            </button>
          </div>
        </div>
      </div>

      {/* Primary KPI Metrics Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3.5">
        {/* Active Model */}
        <div className="p-4 rounded-2xl bg-[#070d18] border border-slate-800 space-y-1 relative overflow-hidden">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Active Model</div>
          <div className="text-base font-extrabold text-cyan-300 tracking-tight leading-snug">
            B8 Spatiotemporal
          </div>
          <div className="text-[11px] text-slate-500 font-mono">Conv2D + GRU Embedding</div>
        </div>

        {/* Overall RMSE */}
        <div className="p-4 rounded-2xl bg-[#070d18] border border-slate-800 space-y-1 relative overflow-hidden">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Overall RMSE</div>
          <div className="text-2xl font-extrabold text-white font-mono flex items-baseline gap-1">
            0.9800 <span className="text-xs font-normal text-slate-400">°C</span>
          </div>
          <div className="text-[11px] text-emerald-400 font-medium">Unweighted Column Mean</div>
        </div>

        {/* Surface Variables */}
        <div className="p-4 rounded-2xl bg-[#070d18] border border-slate-800 space-y-1 relative overflow-hidden">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Surface Variables</div>
          <div className="text-2xl font-extrabold text-white font-mono">
            7 <span className="text-xs font-normal text-slate-400">Channels</span>
          </div>
          <div className="text-[11px] text-slate-400">SST, SSS, SSH, UV, Wind</div>
        </div>

        {/* Depth Levels */}
        <div className="p-4 rounded-2xl bg-[#070d18] border border-slate-800 space-y-1 relative overflow-hidden">
          <div className="text-[10px] uppercase font-bold text-slate-400 tracking-wider">Depth Levels</div>
          <div className="text-2xl font-extrabold text-white font-mono">
            15 <span className="text-xs font-normal text-slate-400">Depths</span>
          </div>
          <div className="text-[11px] text-slate-400">0 m to 1000 m Standard</div>
        </div>

        {/* Improvement */}
        <div className="p-4 rounded-2xl bg-[#070d18] border border-cyan-900/50 bg-cyan-950/10 space-y-1 relative overflow-hidden col-span-2 sm:col-span-1">
          <div className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">Improvement</div>
          <div className="text-2xl font-extrabold text-cyan-300 font-mono">
            +22.11%
          </div>
          <div className="text-[11px] text-emerald-400 font-medium">vs B1 Climatology</div>
        </div>
      </div>

      {/* Main Interactive Domain Map & Current Status Panel */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* North Indian Ocean Map Area */}
        <div className="lg:col-span-8 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-white flex items-center gap-2">
                <span>🗺️</span> North Indian Ocean Basin Domain
              </h2>
              <p className="text-xs text-slate-400 mt-0.5">
                5°N–30°N, 45°E–105°E • Click any point on the map to inspect local surface variables and vertical reconstruction
              </p>
            </div>
          </div>

          <OceanMap
            selectedLat={location.lat}
            selectedLon={location.lon}
            onSelectCoordinates={updateCoordinates}
          />
        </div>

        {/* Current Target Status & Reconstruction Card */}
        <div className="lg:col-span-4 space-y-4">
          <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-5 space-y-4 shadow-xl">
            <div className="flex items-center justify-between pb-3 border-b border-slate-800">
              <span className="text-xs font-bold uppercase tracking-wider text-cyan-400 flex items-center gap-1.5">
                <Radio className="w-3.5 h-3.5 text-cyan-400 animate-pulse" />
                Target State Readout
              </span>
              <span className="text-[10px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded">0.25° Grid</span>
            </div>

            {/* Status Rows */}
            <div className="space-y-3 text-xs">
              <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Ocean Region</span>
                <span className="font-semibold text-white">{location.region}</span>
              </div>

              <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Selected Latitude</span>
                <span className="font-mono text-cyan-300 font-bold">{location.lat.toFixed(2)}°N</span>
              </div>

              <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Selected Longitude</span>
                <span className="font-mono text-cyan-300 font-bold">{location.lon.toFixed(2)}°E</span>
              </div>

              <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Observation Date</span>
                <span className="font-mono text-slate-200">{location.date}</span>
              </div>

              <div className="flex justify-between items-center py-1.5 border-b border-slate-800/60">
                <span className="text-slate-400">Selected Depth</span>
                <span className="font-mono text-blue-300 font-bold">{location.depth} m</span>
              </div>
            </div>

            {/* Mini Profile Preview */}
            <div className="pt-2 bg-slate-950/70 p-3.5 rounded-xl border border-slate-800 text-xs space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-[11px] font-bold text-slate-300">Reconstruction at {location.depth}m:</span>
                <span className="text-xs font-mono font-bold text-cyan-300">
                  {targetPoint.reconstructed.toFixed(2)} °C
                </span>
              </div>
              <div className="flex justify-between items-center text-[11px] text-slate-400">
                <span>Reference (GLORYS):</span>
                <span className="font-mono text-slate-200">{targetPoint.reference.toFixed(2)} °C</span>
              </div>
              <div className="flex justify-between items-center text-[11px] text-slate-400">
                <span>Model Residual:</span>
                <span className="font-mono text-emerald-400">{targetPoint.error.toFixed(2)} °C</span>
              </div>
            </div>

            {/* CTA Button */}
            <button
              onClick={() => setCurrentTab('reconstruction')}
              className="w-full py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 hover:from-cyan-400 hover:to-blue-500 text-slate-950 font-bold text-xs shadow-md transition flex items-center justify-center gap-2"
            >
              <span>Execute B8 Pipeline</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Quick Navigation Cards */}
          <div className="grid grid-cols-2 gap-2.5">
            <button
              onClick={() => setCurrentTab('depth-profile')}
              className="p-3 bg-[#070d18] hover:bg-slate-800/70 border border-slate-800 rounded-xl text-left transition group"
            >
              <div className="text-[10px] font-bold text-slate-400 uppercase">View Profiles</div>
              <div className="text-xs font-bold text-white group-hover:text-cyan-300 transition-colors">15-Depth T(z) →</div>
            </button>

            <button
              onClick={() => setCurrentTab('benchmarks')}
              className="p-3 bg-[#070d18] hover:bg-slate-800/70 border border-slate-800 rounded-xl text-left transition group"
            >
              <div className="text-[10px] font-bold text-slate-400 uppercase">Model Benchmarks</div>
              <div className="text-xs font-bold text-white group-hover:text-cyan-300 transition-colors">B0–B8 Suite →</div>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
