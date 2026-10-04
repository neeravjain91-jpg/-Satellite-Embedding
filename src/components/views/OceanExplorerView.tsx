import React from 'react';
import { 
  Thermometer, 
  Droplets, 
  Waves, 
  Wind, 
  Navigation, 
  Compass, 
  Activity, 
  MapPin, 
  Calendar,
  Layers,
  ArrowUpRight
} from 'lucide-react';
import { OceanStateReturn } from '../../state/useOceanState';
import { getSurfaceObservations, CANONICAL_DEPTHS } from '../../mock/oceanData';
import { CanonicalDepth } from '../../types';
import { OceanMap } from '../shared/OceanMap';

interface OceanExplorerViewProps {
  oceanState: OceanStateReturn;
}

export const OceanExplorerView: React.FC<OceanExplorerViewProps> = ({ oceanState }) => {
  const { location, updateCoordinates, updateDate, updateDepth } = oceanState;

  // Retrieve deterministic surface variables
  const variables = getSurfaceObservations(location.lat, location.lon, location.date);

  const getVariableIcon = (id: string) => {
    switch (id) {
      case 'sst': return <Thermometer className="w-4 h-4 text-rose-400" />;
      case 'sss': return <Droplets className="w-4 h-4 text-cyan-400" />;
      case 'ssh': return <Waves className="w-4 h-4 text-blue-400" />;
      case 'current_u':
      case 'current_v': return <Navigation className="w-4 h-4 text-indigo-400" />;
      case 'wind_u':
      case 'wind_v': return <Wind className="w-4 h-4 text-teal-400" />;
      default: return <Activity className="w-4 h-4 text-cyan-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Control Toolbar */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
          <div>
            <h2 className="text-base font-bold text-white flex items-center gap-2">
              <Compass className="w-4 h-4 text-cyan-400" />
              <span>Observation Target Coordinates & Environment Controls</span>
            </h2>
            <p className="text-xs text-slate-400 mt-0.5">
              Select any coordinate within the North Indian Ocean grid to query surface satellite predictors
            </p>
          </div>

          <div className="text-[11px] font-mono text-cyan-400 bg-slate-900 border border-slate-800 px-3 py-1 rounded-lg">
            Deterministic Local Engine
          </div>
        </div>

        {/* Sliders and Selectors Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Latitude */}
          <div className="space-y-1.5 bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">Latitude:</span>
              <span className="font-mono text-cyan-300 font-bold">{location.lat.toFixed(2)}°N</span>
            </div>
            <input
              type="range"
              min="5.0"
              max="30.0"
              step="0.25"
              value={location.lat}
              onChange={(e) => updateCoordinates(parseFloat(e.target.value), location.lon)}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>5.00°N</span>
              <span>Equator to Tropic</span>
              <span>30.00°N</span>
            </div>
          </div>

          {/* Longitude */}
          <div className="space-y-1.5 bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">Longitude:</span>
              <span className="font-mono text-cyan-300 font-bold">{location.lon.toFixed(2)}°E</span>
            </div>
            <input
              type="range"
              min="45.0"
              max="105.0"
              step="0.25"
              value={location.lon}
              onChange={(e) => updateCoordinates(location.lat, parseFloat(e.target.value))}
              className="w-full accent-cyan-400 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>45.00°E</span>
              <span>Horn of Africa to Malacca</span>
              <span>105.00°E</span>
            </div>
          </div>

          {/* Date */}
          <div className="space-y-1.5 bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">Observation Date:</span>
              <span className="font-mono text-white font-bold">{location.date}</span>
            </div>
            <select
              value={location.date}
              onChange={(e) => updateDate(e.target.value)}
              className="w-full bg-slate-900 border border-slate-700 text-xs text-white rounded-lg px-2.5 py-1.5 focus:outline-none cursor-pointer"
            >
              <option value="2020-12-15">2020-12-15 (Official Test Partition)</option>
              <option value="2020-11-20">2020-11-20 (Late Fall Transition)</option>
              <option value="2020-10-15">2020-10-15 (Validation Split)</option>
              <option value="2020-07-15">2020-07-15 (Southwest Summer Monsoon)</option>
              <option value="2020-04-10">2020-04-10 (Pre-Monsoon Heating)</option>
              <option value="2020-01-15">2020-01-15 (Winter Baseline)</option>
            </select>
            <div className="text-[10px] text-slate-500 font-mono">
              Certified Full-Year 2020 Temporal Range
            </div>
          </div>

          {/* Canonical Depth */}
          <div className="space-y-1.5 bg-slate-950 p-3.5 rounded-xl border border-slate-800/80">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-400 font-medium">Target Depth:</span>
              <span className="font-mono text-blue-300 font-bold">{location.depth} m</span>
            </div>
            <select
              value={location.depth}
              onChange={(e) => updateDepth(Number(e.target.value) as CanonicalDepth)}
              className="w-full bg-slate-900 border border-slate-700 text-xs text-cyan-300 font-bold rounded-lg px-2.5 py-1.5 focus:outline-none cursor-pointer"
            >
              {CANONICAL_DEPTHS.map(d => (
                <option key={d} value={d}>
                  {d} meters {d >= 50 && d <= 200 ? '• Thermocline' : d === 0 ? '• Sea Surface' : ''}
                </option>
              ))}
            </select>
            <div className="text-[10px] text-slate-500 font-mono">
              15 Standard Oceanographic Depths
            </div>
          </div>
        </div>
      </div>

      {/* 7 Surface Variables Display Grid */}
      <div>
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>📡</span> 7 Surface Predictor Variables (Satellite Field Inputs)
          </h3>
          <span className="text-xs text-slate-400 font-mono">
            Location: {location.lat.toFixed(2)}°N, {location.lon.toFixed(2)}°E ({location.region})
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {variables.map((v) => (
            <div
              key={v.id}
              className="p-4 rounded-2xl bg-[#070d18] border border-slate-800 hover:border-slate-700 transition space-y-3 shadow-lg"
            >
              {/* Card Header */}
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-2">
                  <div className="p-2 rounded-xl bg-slate-900 border border-slate-800">
                    {getVariableIcon(v.id)}
                  </div>
                  <div>
                    <div className="text-xs font-bold text-white">{v.name}</div>
                    <div className="text-[10px] text-slate-400 font-mono">{v.symbol}</div>
                  </div>
                </div>

                <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-semibold uppercase ${
                  v.status === 'elevated'
                    ? 'bg-amber-950/60 text-amber-400 border border-amber-800/60'
                    : v.status === 'depressed'
                    ? 'bg-blue-950/60 text-blue-400 border border-blue-800/60'
                    : 'bg-slate-900 text-slate-400'
                }`}>
                  {v.status}
                </span>
              </div>

              {/* Value Readout */}
              <div className="flex items-baseline gap-2 pt-1">
                <span className="text-3xl font-extrabold text-white font-mono tracking-tight">
                  {v.value.toFixed(2)}
                </span>
                <span className="text-xs font-semibold text-slate-400 font-mono">
                  {v.unit}
                </span>
              </div>

              {/* 7-Day Trend Sparkline (SVG) */}
              <div className="space-y-1">
                <div className="flex justify-between text-[10px] text-slate-500 font-mono">
                  <span>7-Day Past Trend</span>
                  <span>{v.trend[v.trend.length - 1]} {v.unit}</span>
                </div>
                <div className="h-9 w-full bg-slate-950/80 rounded-lg p-1 border border-slate-800/60 flex items-center">
                  <svg viewBox="0 0 100 24" className="w-full h-full overflow-visible">
                    {(() => {
                      const min = Math.min(...v.trend);
                      const max = Math.max(...v.trend);
                      const range = max - min || 1;
                      const pts = v.trend.map((val, idx) => {
                        const x = (idx / (v.trend.length - 1)) * 96 + 2;
                        const y = 22 - ((val - min) / range) * 18;
                        return `${x},${y}`;
                      }).join(' ');

                      return (
                        <>
                          <polyline
                            points={pts}
                            fill="none"
                            stroke="#06b6d4"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          />
                          {v.trend.map((val, idx) => {
                            const x = (idx / (v.trend.length - 1)) * 96 + 2;
                            const y = 22 - ((val - min) / range) * 18;
                            return (
                              <circle
                                key={idx}
                                cx={x}
                                cy={y}
                                r="1.5"
                                fill="#22d3ee"
                              />
                            );
                          })}
                        </>
                      );
                    })()}
                  </svg>
                </div>
              </div>

              {/* Physical Description */}
              <p className="text-[11px] text-slate-400 leading-relaxed border-t border-slate-800/60 pt-2">
                {v.description}
              </p>
            </div>
          ))}

          {/* 8th Summary Card */}
          <div className="p-4 rounded-2xl bg-gradient-to-br from-cyan-950/20 to-blue-950/20 border border-cyan-800/40 flex flex-col justify-between shadow-lg">
            <div>
              <div className="flex items-center gap-2 text-cyan-400 text-xs font-bold uppercase tracking-wider mb-2">
                <Activity className="w-4 h-4" />
                Input Space Contract
              </div>
              <p className="text-xs text-slate-300 leading-relaxed">
                All 7 surface variables enter B8's time-distributed spatial Conv2D encoder across a 5-day causal window with zero target leakage.
              </p>
            </div>

            <div className="pt-4 border-t border-cyan-900/40 space-y-2 text-xs font-mono">
              <div className="flex justify-between text-slate-400">
                <span>Patch Shape:</span>
                <span className="text-white">3 × 3 grid cells</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Temporal Window:</span>
                <span className="text-white">T = 5 days</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Normalization:</span>
                <span className="text-emerald-400 font-bold">Train-Only Z-Score</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Interactive Map View */}
      <div className="space-y-3 pt-2">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>🗺️</span> Interactive Geographical Grid Picker
        </h3>
        <OceanMap
          selectedLat={location.lat}
          selectedLon={location.lon}
          onSelectCoordinates={updateCoordinates}
        />
      </div>
    </div>
  );
};
