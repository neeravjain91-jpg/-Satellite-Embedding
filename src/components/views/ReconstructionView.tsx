import React from 'react';
import { 
  Play, 
  Cpu, 
  Activity
} from 'lucide-react';
import { OceanStateReturn } from '../../state/useOceanState';
import { RECONSTRUCTION_STAGES, SIMULATION_STEPS } from '../../mock/reconstruction';
import { getDepthProfile } from '../../mock/profiles';
import { ProfileChart } from '../shared/ProfileChart';

interface ReconstructionViewProps {
  oceanState: OceanStateReturn;
}

export const ReconstructionView: React.FC<ReconstructionViewProps> = ({ oceanState }) => {
  const { 
    location, 
    updateDepth, 
    isSimulating, 
    currentStageIndex, 
    simulationComplete, 
    startReconstruction 
  } = oceanState;

  // Retrieve deterministic depth profile for the current location
  const profilePoints = getDepthProfile(location.lat, location.lon, location.date);

  const getStageStatus = (stageIdx: number): 'idle' | 'processing' | 'complete' => {
    if (simulationComplete) return 'complete';
    if (!isSimulating) return 'idle';
    if (currentStageIndex === stageIdx) return 'processing';
    if (currentStageIndex > stageIdx) return 'complete';
    return 'idle';
  };

  return (
    <div className="space-y-6">
      {/* Reconstruction Control Header Banner */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            <h2 className="text-lg font-bold text-white tracking-tight">
              B8 Spatiotemporal Reconstruction Pipeline
            </h2>
          </div>
          <p className="text-xs text-slate-400 mt-1 max-w-2xl leading-relaxed">
            Execute the deep learning computational graph to infer subsurface potential temperature from surface satellite inputs at <span className="font-mono text-cyan-300 font-bold">{location.gridPoint}</span> ({location.region}).
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-3">
          <button
            onClick={startReconstruction}
            disabled={isSimulating}
            className={`px-6 py-3 rounded-xl font-bold text-xs flex items-center gap-2 shadow-lg transition-all ${
              isSimulating
                ? 'bg-cyan-950 text-cyan-400 border border-cyan-800/60 cursor-not-allowed animate-pulse'
                : 'bg-gradient-to-r from-cyan-500 via-blue-500 to-indigo-600 hover:from-cyan-400 hover:to-blue-400 text-slate-950 shadow-cyan-500/25'
            }`}
          >
            {isSimulating ? (
              <>
                <Activity className="w-4 h-4 animate-spin" />
                <span>Simulating Inference...</span>
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-current" />
                <span>Run Reconstruction</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Pipeline Execution State Stepper Card */}
      <div className="bg-[#060a12] border border-slate-800 rounded-2xl p-5 shadow-lg">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800 text-xs">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <span className="font-bold text-slate-300">Execution Stage Tracker:</span>
            <span className="font-mono text-cyan-300 font-bold">
              {isSimulating ? SIMULATION_STEPS[currentStageIndex + 1] : simulationComplete ? 'Complete' : 'Ready to Run'}
            </span>
          </div>

          <span className="text-[11px] font-mono text-slate-500">
            {isSimulating ? `Step ${currentStageIndex + 1} of 8` : simulationComplete ? '8 / 8 Done' : '0 / 8 Standby'}
          </span>
        </div>

        {/* Progress Bar */}
        <div className="w-full bg-slate-900 rounded-full h-2 mt-3 overflow-hidden border border-slate-800">
          <div 
            className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 transition-all duration-300"
            style={{
              width: simulationComplete ? '100%' : isSimulating ? `${((currentStageIndex + 1) / 8) * 100}%` : '0%'
            }}
          ></div>
        </div>

        {/* Success Banner when Complete */}
        {simulationComplete && (
          <div className="mt-4 p-4 rounded-xl bg-cyan-950/30 border border-cyan-800/50 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center font-bold">
                ✓
              </div>
              <div>
                <div className="font-bold text-white uppercase tracking-wider text-[11px]">
                  RECONSTRUCTION SIMULATION COMPLETE
                </div>
                <div className="text-slate-300 text-xs">
                  Model: <span className="text-cyan-300 font-bold">B8 Spatiotemporal Embedding</span> • Mode: <span className="font-mono text-cyan-300 font-bold">PROTOTYPE • LOCAL SIMULATION</span>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1 rounded bg-slate-900 text-slate-300 font-mono text-[11px] border border-slate-800">
                Column-Avg Test RMSE: 0.9800 °C
              </span>
              <span className="px-2.5 py-1 rounded bg-emerald-950/80 text-emerald-400 font-mono text-[11px] border border-emerald-800/50">
                +22.11% vs Clim
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Visual Connected Architecture Pipeline Nodes */}
      <div className="space-y-3">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span>🔗</span> End-to-End Architectural Dataflow
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3.5">
          {RECONSTRUCTION_STAGES.map((stage, idx) => {
            const status = getStageStatus(idx);
            const isProcessing = status === 'processing';
            const isComplete = status === 'complete';

            return (
              <div
                key={stage.id}
                className={`p-4 rounded-2xl border transition-all duration-300 flex flex-col justify-between ${
                  isProcessing
                    ? 'bg-cyan-950/30 border-cyan-500 shadow-lg shadow-cyan-500/10'
                    : isComplete
                    ? 'bg-[#070e1a] border-cyan-900/60'
                    : 'bg-[#070d18] border-slate-800 opacity-90'
                }`}
              >
                <div>
                  {/* Node Header */}
                  <div className="flex items-center justify-between pb-2 border-b border-slate-800/60 text-xs">
                    <span className="font-mono text-[10px] text-slate-500 uppercase font-bold">
                      Stage 0{stage.stepNumber}
                    </span>

                    <span className={`text-[10px] px-2 py-0.5 rounded font-mono font-bold uppercase ${
                      isComplete
                        ? 'bg-emerald-950 text-emerald-400 border border-emerald-800/60'
                        : isProcessing
                        ? 'bg-cyan-950 text-cyan-300 border border-cyan-800 animate-pulse'
                        : 'bg-slate-900 text-slate-500'
                    }`}>
                      {status}
                    </span>
                  </div>

                  {/* Title & Subtitle */}
                  <div className="pt-2">
                    <h4 className="text-sm font-bold text-white tracking-tight">{stage.title}</h4>
                    <div className="text-[11px] font-mono text-cyan-400 font-medium">{stage.subtitle}</div>
                  </div>

                  {/* Description */}
                  <p className="text-[11px] text-slate-400 mt-2 leading-relaxed">
                    {stage.description}
                  </p>
                </div>

                {/* Stage Details Table */}
                <div className="mt-4 pt-3 border-t border-slate-800/60 text-[10px] font-mono space-y-1">
                  {Object.entries(stage.details).map(([k, v]) => (
                    <div key={k} className="flex justify-between text-slate-400">
                      <span>{k}:</span>
                      <span className="text-slate-200">{v}</span>
                    </div>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Resulting Vertical Profile Visualization */}
      <div className="pt-4">
        <ProfileChart
          points={profilePoints}
          selectedDepth={location.depth}
          onSelectDepth={updateDepth}
        />
      </div>
    </div>
  );
};
