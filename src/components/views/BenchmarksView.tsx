import React, { useState } from 'react';
import { Award, TrendingDown, Layers, Cpu, ArrowUpRight, CheckCircle2, ShieldCheck } from 'lucide-react';
import { BENCHMARK_MODELS } from '../../mock/benchmarks';
import { BenchmarkModel } from '../../types';

export const BenchmarksView: React.FC = () => {
  const [selectedFamily, setSelectedFamily] = useState<string>('All');

  const families = ['All', 'Persistence', 'Baseline', 'Ablation', 'Champion'];

  const filteredModels = selectedFamily === 'All'
    ? BENCHMARK_MODELS
    : BENCHMARK_MODELS.filter(m => m.status === selectedFamily);

  const b1_rmse = 1.2582;

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-2">
        <div className="flex items-center gap-2">
          <Award className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-extrabold text-white tracking-tight uppercase">
            Official ML Benchmark Model Hierarchy
          </h2>
        </div>
        <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
          Comprehensive evaluation of all 10 candidate models on the certified full-year 2020 production test partition (Days 313–365, N=601,550).
        </p>

        {/* Highlight B8 banner with compliant scientific wording */}
        <div className="mt-4 p-4 rounded-xl bg-gradient-to-r from-cyan-950/40 via-blue-950/30 to-slate-900 border border-cyan-500/40 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-3">
            <span className="w-3 h-3 rounded-full bg-cyan-400 animate-pulse"></span>
            <div>
              <div className="font-extrabold text-white text-sm">
                B8 — Best-performing architecture among evaluated internal benchmarks
              </div>
              <div className="text-slate-400 text-xs">
                Spatiotemporal Embedding Model achieves 0.9800 °C overall column-averaged test RMSE (+22.11% improvement over B1 Climatology).
              </div>
            </div>
          </div>

          <div className="flex items-center gap-3 text-xs font-mono">
            <span className="text-cyan-300 font-bold text-base">0.9800 °C</span>
            <span className="text-emerald-400 font-bold bg-emerald-950/80 px-2 py-0.5 rounded border border-emerald-800/40">+22.11%</span>
          </div>
        </div>
      </div>

      {/* Visual Bar Comparison Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart 1: Test RMSE Comparison */}
        <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white">Test RMSE Comparison (°C) — Lower is Better</h3>
            <span className="text-[10px] font-mono text-slate-400">Reference: B1 (1.2582°C)</span>
          </div>

          {/* Bar Chart SVG */}
          <div className="space-y-2 pt-2">
            {BENCHMARK_MODELS.map(m => {
              const maxRmse = 1.8;
              const barWidthPct = (m.rmse / maxRmse) * 100;
              const isWinner = m.id === 'B8';
              const isRef = m.id === 'B1';

              return (
                <div key={m.id} className="space-y-1 text-xs">
                  <div className="flex justify-between items-center text-[11px]">
                    <span className={`font-mono font-bold ${isWinner ? 'text-cyan-300' : isRef ? 'text-amber-400' : 'text-slate-300'}`}>
                      {m.id}: {m.name}
                    </span>
                    <span className="font-mono text-white font-semibold">{m.rmse.toFixed(4)} °C</span>
                  </div>
                  <div className="w-full bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800 flex">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        isWinner
                          ? 'bg-gradient-to-r from-cyan-400 to-blue-500 shadow-sm shadow-cyan-400'
                          : isRef
                          ? 'bg-amber-500'
                          : m.rmse < b1_rmse
                          ? 'bg-blue-600'
                          : 'bg-slate-700'
                      }`}
                      style={{ width: `${barWidthPct}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Chart 2: Relative Improvement vs Climatology */}
        <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-5 shadow-xl space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white">Relative Improvement vs B1 (%) — Higher is Better</h3>
            <span className="text-[10px] font-mono text-emerald-400 font-bold">B8: +22.11%</span>
          </div>

          <div className="space-y-2 pt-2">
            {BENCHMARK_MODELS.map(m => {
              const isWinner = m.id === 'B8';
              const isPositive = m.improvementPct > 0;
              const barWidth = Math.min(100, Math.abs(m.improvementPct) * 2.2);

              return (
                <div key={`imp-${m.id}`} className="space-y-1 text-xs">
                  <div className="flex justify-between items-center text-[11px]">
                    <span className={`font-mono font-bold ${isWinner ? 'text-cyan-300' : 'text-slate-400'}`}>
                      {m.id}
                    </span>
                    <span className={`font-mono font-bold ${isPositive ? 'text-emerald-400' : m.improvementPct === 0 ? 'text-slate-400' : 'text-rose-400'}`}>
                      {m.improvementPct > 0 ? `+${m.improvementPct.toFixed(2)}%` : `${m.improvementPct.toFixed(2)}%`}
                    </span>
                  </div>

                  <div className="w-full bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800 flex items-center">
                    <div
                      className={`h-full transition-all duration-500 rounded-full ${
                        isWinner
                          ? 'bg-cyan-400'
                          : isPositive
                          ? 'bg-emerald-500'
                          : m.improvementPct === 0
                          ? 'bg-slate-700'
                          : 'bg-rose-500/70'
                      }`}
                      style={{ width: `${barWidth}%` }}
                    ></div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* A. Comprehensive Benchmark Table */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl overflow-hidden shadow-xl">
        <div className="p-4 border-b border-slate-800 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <h3 className="text-sm font-bold text-white flex items-center gap-2">
            <span>📋</span> Full ML Model Benchmark Matrix (B0 through B8)
          </h3>

          {/* Family Filter */}
          <div className="flex items-center gap-1.5 flex-wrap">
            {families.map(f => (
              <button
                key={f}
                onClick={() => setSelectedFamily(f)}
                className={`px-2.5 py-1 rounded-lg text-xs font-medium transition ${
                  selectedFamily === f
                    ? 'bg-cyan-500 text-slate-950 font-bold'
                    : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
                }`}
              >
                {f}
              </button>
            ))}
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-[#05080e] text-slate-400 border-b border-slate-800 font-semibold">
              <tr>
                <th className="py-3 px-4">Model ID</th>
                <th className="py-3 px-4">Model Name</th>
                <th className="py-3 px-4">Architecture Specification</th>
                <th className="py-3 px-4">Context Representation</th>
                <th className="py-3 px-4 text-right">Parameters</th>
                <th className="py-3 px-4 text-right">Test RMSE</th>
                <th className="py-3 px-4 text-right">vs B1</th>
                <th className="py-3 px-4 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60">
              {filteredModels.map((m) => {
                const isWinner = m.id === 'B8';
                const isPositive = m.improvementPct > 0;
                return (
                  <tr
                    key={m.id}
                    className={`transition-colors ${
                      isWinner
                        ? 'bg-cyan-950/20 text-white font-bold'
                        : 'hover:bg-slate-900/50 text-slate-300'
                    }`}
                  >
                    <td className={`py-3 px-4 font-extrabold ${isWinner ? 'text-cyan-300 text-sm' : 'text-white'}`}>
                      {m.id}
                    </td>
                    <td className="py-3 px-4 font-sans font-medium text-slate-100">{m.name}</td>
                    <td className="py-3 px-4 text-slate-400 font-sans text-[11px]">{m.architecture}</td>
                    <td className="py-3 px-4 text-slate-400 text-[11px] font-sans">{m.context}</td>
                    <td className="py-3 px-4 text-right text-slate-300">{m.parameters.toLocaleString()}</td>
                    <td className={`py-3 px-4 text-right font-extrabold ${isWinner ? 'text-cyan-300 text-sm' : 'text-slate-100'}`}>
                      {m.rmse.toFixed(4)} °C
                    </td>
                    <td className={`py-3 px-4 text-right font-bold ${isPositive ? 'text-emerald-400' : m.improvementPct === 0 ? 'text-slate-400' : 'text-rose-400'}`}>
                      {m.improvementPct > 0 ? `+${m.improvementPct.toFixed(2)}%` : `${m.improvementPct.toFixed(2)}%`}
                    </td>
                    <td className="py-3 px-4 text-center">
                      <span className={`text-[10px] px-2 py-0.5 rounded font-bold uppercase ${
                        isWinner
                          ? 'bg-cyan-950 text-cyan-300 border border-cyan-700/60'
                          : m.status === 'Baseline'
                          ? 'bg-blue-950 text-blue-300 border border-blue-800/50'
                          : m.status === 'Ablation'
                          ? 'bg-purple-950 text-purple-300 border border-purple-800/50'
                          : 'bg-slate-900 text-slate-400'
                      }`}>
                        {m.status}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* D. Architecture Comparison Cards */}
      <div>
        <h3 className="text-sm font-bold text-white mb-3">Model Paradigm & Structural Highlights</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          {/* B1 Climatology */}
          <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-amber-400 block">Baseline Anchor (B1)</span>
            <div className="text-sm font-bold text-white">Spatial-Depth Climatology</div>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Computes the historical pixel-by-pixel, depth-by-depth mean temperature over the training partition. Serves as the zero-skill anchor that any machine learning model must convincingly beat.
            </p>
            <div className="pt-2 text-slate-300 font-mono text-[11px]">Test RMSE: 1.2582 °C • Params: 0</div>
          </div>

          {/* B4 LightGBM */}
          <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-blue-400 block">Strongest Tabular Model (B4)</span>
            <div className="text-sm font-bold text-white">Gradient Boosting (LightGBM)</div>
            <p className="text-slate-400 text-[11px] leading-relaxed">
              Fits 15 depth-wise boosted decision trees strictly on pointwise surface observations. Achieves substantial non-linear skill (~1.0288 °C), demonstrating the potency of gradient boosting for ocean tabular data.
            </p>
            <div className="pt-2 text-slate-300 font-mono text-[11px]">Test RMSE: 1.0288 °C • Params: 750</div>
          </div>

          {/* B8 Champion */}
          <div className="p-4 bg-cyan-950/20 border border-cyan-500/40 rounded-2xl space-y-2">
            <span className="text-[10px] uppercase font-bold text-cyan-400 block">Benchmark Champion (B8)</span>
            <div className="text-sm font-bold text-white">Spatiotemporal Embedding Model</div>
            <p className="text-slate-300 text-[11px] leading-relaxed">
              Integrates 5-day causal temporal dynamics with local 3×3 spatial convolutions into a 128-dimensional latent bottleneck. Outperforms all pointwise models by recovering the sharp thermocline boundary.
            </p>
            <div className="pt-2 text-cyan-300 font-mono font-bold text-[11px]">Column-Avg Test RMSE: 0.9800 °C • Params: 203,791</div>
          </div>
        </div>
      </div>
    </div>
  );
};
