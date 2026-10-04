import React from 'react';
import { Layers, Thermometer, Info, ArrowDown, Activity, Table } from 'lucide-react';
import { OceanStateReturn } from '../../state/useOceanState';
import { getDepthProfile } from '../../mock/profiles';
import { ProfileChart } from '../shared/ProfileChart';
import { CANONICAL_DEPTHS } from '../../mock/oceanData';

interface DepthProfileViewProps {
  oceanState: OceanStateReturn;
}

export const DepthProfileView: React.FC<DepthProfileViewProps> = ({ oceanState }) => {
  const { location, updateDepth } = oceanState;

  const profilePoints = getDepthProfile(location.lat, location.lon, location.date);
  const activePoint = profilePoints.find(p => p.depth === location.depth) || profilePoints[0];

  return (
    <div className="space-y-6">
      {/* Intro Header */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Layers className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              Vertical Temperature Profile Stratification T(z)
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            Reconstructed vertical column potential temperature from 0m at the sea surface down to 1000m abyssal depth at <span className="font-mono text-cyan-300 font-bold">{location.gridPoint}</span> ({location.region}).
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="px-3.5 py-1.5 rounded-xl bg-slate-900 border border-slate-800 text-xs text-slate-300">
            <span className="text-slate-500 font-mono">Date:</span> {location.date}
          </div>
        </div>
      </div>

      {/* Main Profile Chart */}
      <ProfileChart
        points={profilePoints}
        selectedDepth={location.depth}
        onSelectDepth={updateDepth}
      />

      {/* Depth Inspection Summary Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-1">
          <span className="text-slate-400 uppercase text-[10px] font-bold">Selected Level</span>
          <div className="text-2xl font-extrabold text-cyan-300 font-mono">
            {activePoint.depth} <span className="text-xs font-normal text-slate-400">meters</span>
          </div>
          <div className="text-[11px] text-slate-500">
            {activePoint.depth <= 30 ? 'Upper Mixed Layer' : activePoint.depth <= 200 ? 'Main Thermocline' : 'Deep Abyssal Layer'}
          </div>
        </div>

        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-1">
          <span className="text-slate-400 uppercase text-[10px] font-bold">Predicted Temp (B8)</span>
          <div className="text-2xl font-extrabold text-white font-mono">
            {activePoint.reconstructed.toFixed(2)} <span className="text-xs font-normal text-slate-400">°C</span>
          </div>
          <div className="text-[11px] text-cyan-400">Spatiotemporal Estimate</div>
        </div>

        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-1">
          <span className="text-slate-400 uppercase text-[10px] font-bold">Reference Temp (GLORYS)</span>
          <div className="text-2xl font-extrabold text-slate-200 font-mono">
            {activePoint.reference.toFixed(2)} <span className="text-xs font-normal text-slate-400">°C</span>
          </div>
          <div className="text-[11px] text-slate-400">Target Ocean State</div>
        </div>

        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-1">
          <span className="text-slate-400 uppercase text-[10px] font-bold">Absolute Error</span>
          <div className={`text-2xl font-extrabold font-mono ${activePoint.error <= 0.8 ? 'text-emerald-400' : 'text-amber-400'}`}>
            {activePoint.error.toFixed(2)} <span className="text-xs font-normal text-slate-400">°C</span>
          </div>
          <div className="text-[11px] text-slate-400">Residual Magnitude</div>
        </div>
      </div>

      {/* Comprehensive 15-Depth Table */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <Table className="w-4 h-4 text-cyan-400" />
            <span>Depth-Wise Temperature Reconstruction Table (15 Canonical Levels)</span>
          </h3>
          <span className="text-xs text-slate-500 font-mono">Units: °C</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#05080e] text-slate-400 border-b border-slate-800">
              <tr>
                <th className="py-3 px-4">Depth (m)</th>
                <th className="py-3 px-4 text-right">B8 Reconstructed</th>
                <th className="py-3 px-4 text-right">GLORYS Reference</th>
                <th className="py-3 px-4 text-right">Absolute Residual</th>
                <th className="py-3 px-4 text-right">B1 Climatology</th>
                <th className="py-3 px-4 text-right">B2 Ridge</th>
                <th className="py-3 px-4 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {profilePoints.map((pt) => {
                const isSelected = pt.depth === location.depth;
                return (
                  <tr
                    key={pt.depth}
                    onClick={() => updateDepth(pt.depth)}
                    className={`cursor-pointer transition-colors ${
                      isSelected
                        ? 'bg-cyan-950/30 text-white font-bold'
                        : 'hover:bg-slate-900/60 text-slate-300'
                    }`}
                  >
                    <td className="py-2.5 px-4 font-bold text-cyan-300">{pt.depth} m</td>
                    <td className="py-2.5 px-4 text-right text-white font-semibold">{pt.reconstructed.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-right text-slate-300">{pt.reference.toFixed(2)}</td>
                    <td className={`py-2.5 px-4 text-right font-bold ${pt.error <= 0.8 ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {pt.error.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-4 text-right text-amber-300/80">{pt.climatology.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-right text-blue-300/80">{pt.ridge.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-center">
                      <span className={`text-[10px] px-2 py-0.5 rounded ${
                        pt.error <= 0.8
                          ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/40'
                          : 'bg-amber-950 text-amber-400 border border-amber-800/40'
                      }`}>
                        {pt.error <= 0.8 ? 'HIGH ACCURACY' : 'THERMOCLINE'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
