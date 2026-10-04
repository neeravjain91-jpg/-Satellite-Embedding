import React, { useState } from 'react';
import { Sparkles, Info, Eye, Layers } from 'lucide-react';
import { EmbeddingPoint } from '../../types';
import { EMBEDDING_CLUSTERS } from '../../mock/embeddings';

interface EmbeddingPlotProps {
  points: EmbeddingPoint[];
  selectedPointId?: string;
  onSelectPoint?: (pt: EmbeddingPoint) => void;
  className?: string;
}

export const EmbeddingPlot: React.FC<EmbeddingPlotProps> = ({
  points,
  selectedPointId,
  onSelectPoint,
  className = ''
}) => {
  const [activeCluster, setActiveCluster] = useState<string | null>(null);
  const [hoveredPoint, setHoveredPoint] = useState<EmbeddingPoint | null>(null);

  const width = 640;
  const height = 440;
  const padding = 50;

  // Projection bounds: -1.0 to 1.0 on x and y
  const projToX = (x: number) => padding + ((x + 1.0) / 2.0) * (width - 2 * padding);
  const projToY = (y: number) => padding + ((1.0 - y) / 2.0) * (height - 2 * padding);

  const clusterColorMap: Record<string, string> = {
    'Arabian Sea Upwelling': '#06b6d4',
    'Bay of Bengal River Plume': '#10b981',
    'Equatorial Warm Pool Core': '#f59e0b',
    'Winter Mixed Layer Convection': '#8b5cf6',
    'Anticyclonic Warm Eddy': '#ec4899',
    'Cyclonic Cold Eddy': '#3b82f6'
  };

  const filteredPoints = activeCluster
    ? points.filter(p => p.cluster === activeCluster)
    : points;

  return (
    <div className={`bg-[#060a12] border border-slate-800 rounded-2xl p-6 shadow-xl select-none ${className}`}>
      {/* Title & Scientific Disclaimer */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            <h3 className="text-base font-bold text-white tracking-tight">128-D Ocean Latent Space Projection (2D t-SNE / PCA)</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Low-dimensional manifold projection of spatiotemporal representation vectors from the B8 LayerNorm bottleneck
          </p>
        </div>

        {/* Strict Scientific Honesty Badge */}
        <div className="flex flex-col sm:flex-row items-start sm:items-center gap-2">
          <div className="px-3 py-1.5 rounded-xl bg-cyan-950/40 border border-cyan-800/60 text-[11px] text-cyan-300 flex items-center gap-2 font-mono font-semibold">
            <span>PROTOTYPE • LOCAL SIMULATION</span>
          </div>
          <div className="px-3 py-1.5 rounded-xl bg-amber-950/30 border border-amber-800/40 text-[11px] text-amber-300 flex items-center gap-2">
            <Info className="w-3.5 h-3.5 flex-shrink-0" />
            <span>MOCK EMBEDDING PROJECTION</span>
          </div>
        </div>
      </div>

      {/* Cluster Filter Buttons */}
      <div className="flex flex-wrap items-center gap-2 pt-4">
        <button
          onClick={() => setActiveCluster(null)}
          className={`px-3 py-1 rounded-lg text-xs font-medium transition ${
            activeCluster === null
              ? 'bg-slate-700 text-white font-bold'
              : 'bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800'
          }`}
        >
          All Clusters ({points.length})
        </button>
        {EMBEDDING_CLUSTERS.map(c => {
          const isSelected = activeCluster === c.name;
          return (
            <button
              key={c.name}
              onClick={() => setActiveCluster(isSelected ? null : c.name)}
              className={`px-3 py-1 rounded-lg text-xs font-medium flex items-center gap-1.5 transition ${
                isSelected
                  ? 'bg-slate-800 text-white border border-slate-600 font-bold'
                  : 'bg-slate-900/80 text-slate-400 hover:text-slate-200 border border-slate-800'
              }`}
            >
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: c.color }}></span>
              <span>{c.name}</span>
            </button>
          );
        })}
      </div>

      {/* Interactive SVG Projection */}
      <div className="relative mt-4">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto block bg-[#040810] rounded-xl border border-slate-800/80"
        >
          {/* Grid Cross */}
          <line x1={padding} y1={height / 2} x2={width - padding} y2={height / 2} stroke="#1e293b" strokeWidth="1" strokeDasharray="2 2" />
          <line x1={width / 2} y1={padding} x2={width / 2} y2={height - padding} stroke="#1e293b" strokeWidth="1" strokeDasharray="2 2" />

          {/* Axis Labels */}
          <text x={width - padding} y={height / 2 - 8} fill="#475569" fontSize="10" fontFamily="monospace" textAnchor="end">Latent Dimension 1 →</text>
          <text x={width / 2 + 8} y={padding + 10} fill="#475569" fontSize="10" fontFamily="monospace">↑ Latent Dimension 2</text>

          {/* Points */}
          {filteredPoints.map((pt) => {
            const cx = projToX(pt.x);
            const cy = projToY(pt.y);
            const isHovered = hoveredPoint?.id === pt.id;
            const isSelected = selectedPointId === pt.id;
            const color = clusterColorMap[pt.cluster] || '#06b6d4';

            return (
              <g
                key={pt.id}
                className="cursor-pointer"
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
                onClick={() => onSelectPoint && onSelectPoint(pt)}
              >
                {/* Halo if hovered/selected */}
                {(isHovered || isSelected) && (
                  <circle cx={cx} cy={cy} r="10" fill={color} fillOpacity="0.25" className="animate-pulse" />
                )}
                <circle
                  cx={cx}
                  cy={cy}
                  r={isHovered || isSelected ? "5.5" : "3.5"}
                  fill={color}
                  stroke="#0f172a"
                  strokeWidth="1"
                />
              </g>
            );
          })}
        </svg>

        {/* Hover Tooltip Card Overlay */}
        {hoveredPoint && (
          <div className="absolute bottom-4 right-4 bg-slate-900/95 border border-slate-700 p-3 rounded-xl shadow-2xl text-xs space-y-1.5 max-w-xs backdrop-blur-md">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: clusterColorMap[hoveredPoint.cluster] }}></span>
              <span className="font-bold text-white">{hoveredPoint.cluster}</span>
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 font-mono text-[11px] text-slate-300 pt-1 border-t border-slate-800">
              <div><span className="text-slate-500">Location:</span> {hoveredPoint.lat}°N, {hoveredPoint.lon}°E</div>
              <div><span className="text-slate-500">Date:</span> {hoveredPoint.date}</div>
              <div><span className="text-slate-500">SST:</span> {hoveredPoint.sst} °C</div>
              <div><span className="text-slate-500">SSH:</span> {hoveredPoint.ssh} m</div>
              <div className="col-span-2 text-cyan-400 font-bold">
                Thermocline Depth: ~{hoveredPoint.thermoclineDepth} m
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Cluster Descriptive Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2.5 mt-5">
        {EMBEDDING_CLUSTERS.map(c => (
          <div key={c.name} className="p-3 bg-slate-900/60 border border-slate-800 rounded-xl text-xs">
            <div className="flex items-center gap-2 mb-1">
              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: c.color }}></span>
              <span className="font-bold text-white text-[11px]">{c.name}</span>
            </div>
            <p className="text-slate-400 text-[11px] leading-relaxed">{c.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};
