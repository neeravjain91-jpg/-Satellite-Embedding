import React from 'react';
import { CheckCircle2, ShieldAlert, BarChart3, TrendingDown, Layers, MapPin, Calendar } from 'lucide-react';
import { REGIONAL_METRICS, SEASONAL_METRICS, B8_EXACT_DEPTH_TABLE } from '../../mock/benchmarks';

export const ValidationView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-2">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-extrabold text-white tracking-tight uppercase">
            Scientific Validation & QA Audit
          </h2>
        </div>
        <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
          Depth-wise, basin-wise, and seasonal statistical validation of B8 against GLORYS potential temperature targets on the 2020 production test partition.
        </p>

        {/* Mandatory Scientific Honesty Alert */}
        <div className="mt-3 p-3.5 rounded-xl bg-slate-900 border border-slate-700/80 flex items-start gap-2.5 text-xs text-slate-300">
          <ShieldAlert className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-white uppercase tracking-wider text-[11px] block">
              Column-Averaged Metric Clarification
            </span>
            <span>
              The benchmark metric <span className="font-mono text-cyan-300 font-bold">0.9800 °C</span> represents the <strong className="text-white">Overall column-averaged test RMSE</strong> (unweighted depth mean across all 15 vertical levels). It is not the error at every single depth. Errors vary from ~0.44°C at the sea surface to ~1.81°C in the high-shear thermocline.
            </span>
          </div>
        </div>
      </div>

      {/* Main Validation KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="p-5 rounded-2xl bg-[#070d18] border border-cyan-800/40 bg-cyan-950/10 space-y-1">
          <div className="text-[10px] uppercase font-bold text-cyan-400 tracking-wider">Overall Column-Averaged RMSE</div>
          <div className="text-3xl font-extrabold text-cyan-300 font-mono">
            0.9800 <span className="text-sm font-normal text-slate-400">°C</span>
          </div>
          <div className="text-xs text-slate-400">95% Bootstrap CI: [0.9270, 1.0283] °C</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#070d18] border border-slate-800 space-y-1">
          <div className="text-[10px] uppercase font-bold text-amber-400 tracking-wider">B1 Climatology Benchmark</div>
          <div className="text-3xl font-extrabold text-white font-mono">
            1.2582 <span className="text-sm font-normal text-slate-400">°C</span>
          </div>
          <div className="text-xs text-slate-400">Reference domain baseline</div>
        </div>

        <div className="p-5 rounded-2xl bg-[#070d18] border border-emerald-800/40 bg-emerald-950/10 space-y-1">
          <div className="text-[10px] uppercase font-bold text-emerald-400 tracking-wider">Error Reduction</div>
          <div className="text-3xl font-extrabold text-emerald-300 font-mono">
            +22.11%
          </div>
          <div className="text-xs text-emerald-400 font-medium">Paired 95% CI: [-0.3957, -0.1756] °C</div>
        </div>
      </div>

      {/* Actual B8 Depth-Wise RMSE Curve & Table (Required Values) */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Official B8 Depth-Wise Test Error Distribution</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Exact Test RMSE across all 15 canonical depths from the master evaluation manifest
            </p>
          </div>
          <span className="text-xs font-mono text-cyan-400 bg-slate-900 px-3 py-1 rounded-lg border border-slate-800">
            N = 601,550 test samples
          </span>
        </div>

        {/* Horizontal Bar Visualization of Depth Error */}
        <div className="space-y-2 pt-2">
          {B8_EXACT_DEPTH_TABLE.map((row) => {
            const maxScale = 2.0;
            const barWidth = (row.rmse / maxScale) * 100;
            const isThermocline = row.depth >= 50 && row.depth <= 200;

            return (
              <div key={row.depth} className="grid grid-cols-12 items-center gap-2 text-xs font-mono">
                {/* Depth Label */}
                <div className="col-span-2 text-right pr-2 text-slate-400 font-bold">
                  {row.depth} m
                </div>

                {/* Bar */}
                <div className="col-span-7 bg-slate-950 rounded-full h-3 overflow-hidden border border-slate-800">
                  <div
                    className={`h-full transition-all duration-300 rounded-full ${
                      isThermocline
                        ? 'bg-amber-400'
                        : row.rmse < 0.6
                        ? 'bg-cyan-400'
                        : 'bg-blue-500'
                    }`}
                    style={{ width: `${barWidth}%` }}
                  ></div>
                </div>

                {/* Readout Values */}
                <div className="col-span-3 flex justify-between items-center text-[11px]">
                  <span className={`font-bold ${isThermocline ? 'text-amber-300' : 'text-cyan-300'}`}>
                    {row.rmse.toFixed(4)} °C
                  </span>
                  <span className="text-slate-500 hidden sm:inline">
                    MAE: {row.mae.toFixed(4)}
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        <div className="p-3 bg-slate-950/80 rounded-xl border border-slate-800/80 text-[11px] text-slate-400 flex flex-col sm:flex-row sm:items-center justify-between gap-1">
          <span>🟡 High error in 50–200m reflects rapid vertical temperature gradients across the thermocline.</span>
          <span className="font-mono text-slate-300 font-bold">Surface: 0.4369°C • Peak: 1.8110°C (75m) • 1000m: 0.5959°C</span>
        </div>
      </div>

      {/* Regional & Seasonal Breakdown Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Regional Breakdown */}
        <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <MapPin className="w-4 h-4 text-cyan-400" />
              <span>Regional Basin Validation (Cosine-Weighted RMSE)</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Area-Weighted</span>
          </div>

          <div className="space-y-3">
            {REGIONAL_METRICS.map(r => (
              <div key={r.region} className="p-3.5 bg-slate-950 rounded-xl border border-slate-800/80 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-white text-xs">{r.region}</span>
                  <span className="font-mono text-cyan-300 font-extrabold text-sm">{r.rmse.toFixed(4)} °C</span>
                </div>
                <div className="flex justify-between items-center text-[10px] text-slate-500">
                  <span>{r.description}</span>
                  <span className="font-mono">N = {r.sampleCount.toLocaleString()}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Seasonal Breakdown */}
        <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between pb-2 border-b border-slate-800">
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Calendar className="w-4 h-4 text-blue-400" />
              <span>Seasonal Partition Validation</span>
            </h3>
            <span className="text-[10px] font-mono text-slate-400">Test Splits</span>
          </div>

          <div className="space-y-3">
            {SEASONAL_METRICS.map(s => (
              <div key={s.period} className="p-3.5 bg-slate-950 rounded-xl border border-slate-800/80 space-y-1">
                <div className="flex justify-between items-center">
                  <span className="font-bold text-white text-xs">{s.period}</span>
                  <span className="font-mono text-blue-300 font-extrabold text-sm">{s.rmse.toFixed(4)} °C</span>
                </div>
                <div className="flex justify-between items-center text-[10px] text-slate-500">
                  <span>{s.description}</span>
                  <span className="font-mono text-slate-400">{s.dateRange}</span>
                </div>
              </div>
            ))}

            <div className="p-3.5 bg-slate-950/60 rounded-xl border border-slate-800/60 text-xs text-slate-400">
              <div className="font-bold text-slate-300 mb-1">Partition Protocol Note:</div>
              Certified seasonal benchmark metrics strictly evaluate the frozen 2020 Test partition windows (Late Fall: Nov 09–30 and Early Winter: Dec 01–31). Annual monsoonal cycles (Pre-Monsoon, Southwest Monsoon) represent climatological regimes across the earlier partitions and are not presented with fabricated performance values.
            </div>
          </div>
        </div>
      </div>

      {/* ARGO–GLORYS Reference Consistency Assessment Card */}
      <div className="p-4 rounded-2xl bg-[#070d18] border border-slate-800 flex items-start gap-3.5 text-xs text-slate-300 shadow-xl">
        <Layers className="w-5 h-5 text-cyan-400 flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <span className="font-bold text-white uppercase tracking-wider text-[11px]">
              ARGO–GLORYS Reference Consistency Assessment
            </span>
            <span className="text-[10px] font-mono text-cyan-400 bg-cyan-950/60 border border-cyan-800/60 px-2 py-0.5 rounded">
              Reference Evaluation Tier
            </span>
          </div>
          <p className="text-slate-400 leading-relaxed">
            Reconstruction targets are benchmarked against the <strong>GLORYS numerical ocean reanalysis reference</strong> state estimate. Matchups against N=1,482 collocated in-situ Argo profiling floats evaluate reanalysis reference consistency (RMSE = 4.17 °C, r = 0.963 across the full basin). Because operational Argo profiles are assimilated into GLORYS, this comparison assesses reanalysis reference consistency rather than serving as independent validation of the machine learning model.
          </p>
        </div>
      </div>
    </div>
  );
};
