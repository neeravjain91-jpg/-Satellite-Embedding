import React from 'react';
import { BookOpen, ShieldCheck, Cpu, Database, Calendar, AlertTriangle, Layers, Activity } from 'lucide-react';

export const MethodologyView: React.FC = () => {
  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-2">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-extrabold text-white tracking-tight uppercase">
            Scientific ML Experiment Protocol & Methodology
          </h2>
        </div>
        <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
          Locked experimental methodology ensuring zero data leakage, chronological purge buffer isolation, and reproducible evaluation of deep ocean reconstruction.
        </p>
      </div>

      {/* Chronological Temporal Partition Timeline (Exact Days) */}
      <div className="bg-[#060a12] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div>
            <h3 className="text-sm font-bold text-white flex items-center gap-2">
              <Calendar className="w-4 h-4 text-cyan-400" />
              <span>Exact Chronological Data Split (Full-Year 2020 Leap Year, 366 Days)</span>
            </h3>
            <p className="text-xs text-slate-400 mt-0.5">
              Strict 6-day purge buffers programmatically isolate train, validation, and test partitions against oceanographic auto-correlation.
            </p>
          </div>
          <span className="text-xs font-mono text-emerald-400 bg-emerald-950/80 px-2.5 py-1 rounded border border-emerald-800/40 font-bold">
            0% Overlap
          </span>
        </div>

        {/* Visual Timeline Bar */}
        <div className="space-y-2 pt-2">
          <div className="w-full h-8 bg-slate-950 rounded-xl overflow-hidden border border-slate-800 flex text-[10px] font-mono font-bold select-none">
            {/* Train: 253 days / 366 = 69.1% */}
            <div style={{ width: '69.1%' }} className="h-full bg-blue-600/90 text-white flex items-center justify-center border-r border-slate-900">
              TRAIN (Days 0–252) • 253 Days (69.1%)
            </div>

            {/* Purge 1: 6 days / 366 = 1.6% */}
            <div style={{ width: '1.6%' }} className="h-full bg-amber-500/80 text-black flex items-center justify-center border-r border-slate-900" title="Purge 1 (Days 253-258)">
              P1
            </div>

            {/* Val: 48 days / 366 = 13.1% */}
            <div style={{ width: '13.1%' }} className="h-full bg-indigo-600/90 text-white flex items-center justify-center border-r border-slate-900">
              VAL (Days 259–306) • 48 Days
            </div>

            {/* Purge 2: 6 days / 366 = 1.6% */}
            <div style={{ width: '1.6%' }} className="h-full bg-amber-500/80 text-black flex items-center justify-center border-r border-slate-900" title="Purge 2 (Days 307-312)">
              P2
            </div>

            {/* Test: 53 days / 366 = 14.5% */}
            <div style={{ width: '14.5%' }} className="h-full bg-cyan-500 text-slate-950 flex items-center justify-center font-extrabold">
              TEST (Days 313–365) • 53 Days
            </div>
          </div>

          {/* Partition Details Grid */}
          <div className="grid grid-cols-1 sm:grid-cols-5 gap-3 pt-3 text-xs">
            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 space-y-1">
              <span className="text-blue-400 font-bold block text-[11px]">Train Split</span>
              <div className="font-mono text-white">Days 0–252 (253 days)</div>
              <div className="text-[10px] text-slate-400">Jan 01 – Sep 09, 2020</div>
              <div className="text-[10px] text-slate-500 font-mono">N = 2,871,550 samples</div>
            </div>

            <div className="p-3 bg-amber-950/20 rounded-xl border border-amber-800/40 space-y-1">
              <span className="text-amber-400 font-bold block text-[11px]">Purge Buffer 1</span>
              <div className="font-mono text-white">Days 253–258 (6 days)</div>
              <div className="text-[10px] text-slate-400">Sep 10 – Sep 15, 2020</div>
              <div className="text-[10px] text-rose-400 font-mono">Completely Discarded</div>
            </div>

            <div className="p-3 bg-slate-900/60 rounded-xl border border-slate-800 space-y-1">
              <span className="text-indigo-400 font-bold block text-[11px]">Validation Split</span>
              <div className="font-mono text-white">Days 259–306 (48 days)</div>
              <div className="text-[10px] text-slate-400">Sep 16 – Nov 02, 2020</div>
              <div className="text-[10px] text-slate-500 font-mono">N = 544,800 samples</div>
            </div>

            <div className="p-3 bg-amber-950/20 rounded-xl border border-amber-800/40 space-y-1">
              <span className="text-amber-400 font-bold block text-[11px]">Purge Buffer 2</span>
              <div className="font-mono text-white">Days 307–312 (6 days)</div>
              <div className="text-[10px] text-slate-400">Nov 03 – Nov 08, 2020</div>
              <div className="text-[10px] text-rose-400 font-mono">Completely Discarded</div>
            </div>

            <div className="p-3 bg-cyan-950/20 rounded-xl border border-cyan-800/50 space-y-1">
              <span className="text-cyan-300 font-bold block text-[11px]">Test Split (Frozen)</span>
              <div className="font-mono text-white">Days 313–365 (53 days)</div>
              <div className="text-[10px] text-slate-400">Nov 09 – Dec 31, 2020</div>
              <div className="text-[10px] text-cyan-400 font-mono font-bold">N = 601,550 samples</div>
            </div>
          </div>
        </div>
      </div>

      {/* 8 Structured Methodology Sections Required by User */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs">
        {/* 1. Problem */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">1</span>
            <span>Problem Definition</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Satellite observations are strictly confined to the ocean surface skin layer (top millimeters to meters). In-situ profiling floats (Argo) are sparse in space and time. The scientific goal is reconstructing continuous 3D subsurface temperature fields down to 1000m using surface physics without unconstrained extrapolation.
          </p>
        </div>

        {/* 2. Data */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">2</span>
            <span>Data Sources & Target</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            The target is GLORYS12V1 daily potential temperature (<code className="text-cyan-300 font-mono">thetao</code>) interpolated to 15 canonical depth levels down to 1000m. The GLORYS/ORCA12 model bathymetry defines seafloor depth. Sub-seafloor points are strictly preserved as NaNs and never converted to physical 0°C.
          </p>
        </div>

        {/* 3. Surface Variables */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">3</span>
            <span>7 Surface Variables</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Daily predictors include Sea Surface Temperature (OSTIA), Sea Surface Salinity (Copernicus Multi-Observation SSS), Sea Surface Height (DUACS Altimetry), Zonal & Meridional Currents (OSCAR Geostrophic), and Zonal & Meridional Wind Stress (CCMP V3.1). All standardized strictly with training split statistics.
          </p>
        </div>

        {/* 4. Temporal Context */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">4</span>
            <span>Temporal Context (5-Day Causal)</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Constructed causally as <code className="text-cyan-300 font-mono">[t-4, t-3, t-2, t-1, t]</code>. Windows strictly look backward in time with zero future leakage. Windows at the beginning of validation and test are clamped at the partition boundary to prevent crossing into purge intervals.
          </p>
        </div>

        {/* 5. Spatial Context */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">5</span>
            <span>Spatial Context (3×3 Patch)</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            A 3×3 grid neighborhood (~75 km × 75 km area) around each center pixel is extracted across all 7 channels. This spatial context allows 2D convolutional kernels to extract mesoscale eddy curvature and horizontal thermal gradients.
          </p>
        </div>

        {/* 6. B8 Architecture */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">6</span>
            <span>B8 Architecture Specification</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Conv2D(7→32) + BN + Conv2D(32→64) + BN + AdaptiveAvgPool → 2-layer causal GRU(64→128) → LayerNorm(128) latent bottleneck → Linear(128→64) → ReLU → Linear(64→15). Total parameters: exactly <strong className="text-cyan-300 font-mono">203,791</strong>.
          </p>
        </div>

        {/* 7. Evaluation */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">7</span>
            <span>Evaluation & Statistical Rigor</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Performance is measured via 7-day Block-Bootstrap confidence intervals ($N=1,000$ iterations) and paired difference tests against B1 Climatology to account for temporal autocorrelation. Regional metrics are cosine-latitude area-weighted.
          </p>
        </div>

        {/* 8. Limitations */}
        <div className="p-5 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2.5 shadow-lg">
          <div className="flex items-center gap-2 text-white font-bold text-sm">
            <span className="w-6 h-6 rounded-lg bg-cyan-500/10 text-cyan-400 flex items-center justify-center font-mono text-xs">8</span>
            <span>Scope & Limitations</span>
          </div>
          <p className="text-slate-300 leading-relaxed">
            Evaluation is conducted on GLORYS12V1 ocean reanalysis target fields as an emulation benchmark. Direct physical comparison with in-situ Argo profiling floats represents an independent validation tier outside the GLORYS training manifold.
          </p>
        </div>
      </div>
    </div>
  );
};
