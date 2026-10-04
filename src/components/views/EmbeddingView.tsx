import React from 'react';
import { Sparkles, ArrowRight, Layers, Cpu, ShieldCheck, Database } from 'lucide-react';
import { generateMockEmbeddings } from '../../mock/embeddings';
import { EmbeddingPlot } from '../shared/EmbeddingPlot';
import { OceanStateReturn } from '../../state/useOceanState';

interface EmbeddingViewProps {
  oceanState: OceanStateReturn;
}

export const EmbeddingView: React.FC<EmbeddingViewProps> = ({ oceanState }) => {
  const { location } = oceanState;
  const mockEmbeddings = React.useMemo(() => generateMockEmbeddings(140), []);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#070d18] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-2">
        <div className="flex items-center gap-2">
          <Sparkles className="w-5 h-5 text-cyan-400" />
          <h2 className="text-xl font-extrabold text-white tracking-tight uppercase">
            128-D Ocean Latent Embedding
          </h2>
        </div>
        <p className="text-xs text-slate-300 max-w-3xl leading-relaxed">
          The core scientific innovation of B8: mapping surface satellite fields into a continuous 128-dimensional latent bottleneck vector before vertical depth decoding.
        </p>
      </div>

      {/* Visual Architecture Flow Diagram Required by User */}
      <div className="bg-[#060a12] border border-slate-800 rounded-2xl p-6 shadow-xl space-y-4">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <Cpu className="w-4 h-4 text-cyan-400" />
          <span>Representation Learning Architecture</span>
        </h3>

        {/* Flow chain */}
        <div className="flex flex-col lg:flex-row items-center justify-between gap-2 text-center text-xs">
          {/* Step 1 */}
          <div className="w-full lg:w-44 p-3 bg-slate-900 border border-slate-800 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Step 1</span>
            <div className="font-bold text-white mt-0.5">7 Surface Variables</div>
            <div className="text-[10px] text-slate-400 font-mono mt-1">[B, 7] Physical</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-600 hidden lg:block" />

          {/* Step 2 */}
          <div className="w-full lg:w-44 p-3 bg-slate-900 border border-slate-800 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Step 2</span>
            <div className="font-bold text-white mt-0.5">3×3 Spatial Context</div>
            <div className="text-[10px] text-slate-400 font-mono mt-1">75 km Footprint</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-600 hidden lg:block" />

          {/* Step 3 */}
          <div className="w-full lg:w-44 p-3 bg-slate-900 border border-slate-800 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Step 3</span>
            <div className="font-bold text-white mt-0.5">Spatial CNN</div>
            <div className="text-[10px] text-slate-400 font-mono mt-1">Conv2D(7→32→64)</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-600 hidden lg:block" />

          {/* Step 4 */}
          <div className="w-full lg:w-44 p-3 bg-slate-900 border border-slate-800 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Step 4</span>
            <div className="font-bold text-white mt-0.5">5-Day Temporal Context</div>
            <div className="text-[10px] text-slate-400 font-mono mt-1">Causal Window</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-600 hidden lg:block" />

          {/* Step 5 */}
          <div className="w-full lg:w-44 p-3 bg-slate-900 border border-slate-800 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Step 5</span>
            <div className="font-bold text-white mt-0.5">Temporal GRU</div>
            <div className="text-[10px] text-slate-400 font-mono mt-1">2-Layer Recurrence</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-600 hidden lg:block" />

          {/* Step 6 */}
          <div className="w-full lg:w-48 p-3 bg-cyan-950/40 border border-cyan-500/50 rounded-xl shadow-lg shadow-cyan-500/10">
            <span className="text-[10px] uppercase font-bold text-cyan-400 block">Step 6</span>
            <div className="font-bold text-cyan-300 mt-0.5">128-D Latent Vector</div>
            <div className="text-[10px] text-cyan-400 font-mono mt-1">LayerNorm(128)</div>
          </div>

          <ArrowRight className="w-4 h-4 text-slate-600 hidden lg:block" />

          {/* Step 7 */}
          <div className="w-full lg:w-44 p-3 bg-slate-900 border border-slate-800 rounded-xl">
            <span className="text-[10px] uppercase font-bold text-slate-400 block">Step 7</span>
            <div className="font-bold text-white mt-0.5">15 Depth Reconstruction</div>
            <div className="text-[10px] text-slate-400 font-mono mt-1">Linear(128→64→15)</div>
          </div>
        </div>
      </div>

      {/* Main Embedding 2D Projection Plot */}
      <EmbeddingPlot
        points={mockEmbeddings}
        selectedPointId="pt-0"
      />

      {/* Theoretical Significance Section */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2">
          <div className="font-bold text-white flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-cyan-400" />
            Baroclinic Mode Compression
          </div>
          <p className="text-slate-400 leading-relaxed">
            Ocean physics dictates that upper-surface currents, wind stress, and altimetry reflect integrated vertical internal baroclinic wave modes. The 128-D bottleneck compresses these multi-frequency signals into hydrographic coordinates.
          </p>
        </div>

        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2">
          <div className="font-bold text-white flex items-center gap-1.5">
            <Database className="w-4 h-4 text-blue-400" />
            Zero Target Leakage
          </div>
          <p className="text-slate-400 leading-relaxed">
            The embedding is formed exclusively from surface predictors (SST, SSS, SSH, Current U/V, Wind U/V). Subsurface temperature targets are never fed back into the encoder during forward inference.
          </p>
        </div>

        <div className="p-4 bg-[#070d18] border border-slate-800 rounded-2xl space-y-2">
          <div className="font-bold text-white flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
            LayerNorm Stability
          </div>
          <p className="text-slate-400 leading-relaxed">
            Layer normalization at the bottleneck bounds latent vector magnitudes, preventing runaway gradient explosions during temporal backpropagation through the 2-layer GRU sequence.
          </p>
        </div>
      </div>
    </div>
  );
};
