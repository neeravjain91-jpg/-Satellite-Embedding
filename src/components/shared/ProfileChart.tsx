import React, { useState } from 'react';
import { DepthProfilePoint, CanonicalDepth } from '../../types';
import { CANONICAL_DEPTHS } from '../../mock/oceanData';

interface ProfileChartProps {
  points: DepthProfilePoint[];
  selectedDepth: CanonicalDepth;
  onSelectDepth: (depth: CanonicalDepth) => void;
  className?: string;
  showComparisonBaselines?: boolean;
}

export const ProfileChart: React.FC<ProfileChartProps> = ({
  points,
  selectedDepth,
  onSelectDepth,
  className = '',
  showComparisonBaselines = true
}) => {
  const [hoveredPoint, setHoveredPoint] = useState<DepthProfilePoint | null>(null);

  // SVG dimensions
  const width = 560;
  const height = 480;
  const padding = { top: 30, right: 40, bottom: 50, left: 60 };

  // Plotting domain
  // Temp X: 5 °C to 31 °C
  const minTemp = 5.0;
  const maxTemp = 32.0;

  // We map depth non-linearly or piecewise so shallow depths (0-200m) are prominent
  // Given depths: 0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
  // Index-based or log-spaced vertical depth axis gives scientific readability
  const depthToY = (depth: CanonicalDepth) => {
    const idx = CANONICAL_DEPTHS.indexOf(depth);
    const usableHeight = height - padding.top - padding.bottom;
    return padding.top + (idx / (CANONICAL_DEPTHS.length - 1)) * usableHeight;
  };

  const tempToX = (temp: number) => {
    const usableWidth = width - padding.left - padding.right;
    const clamped = Math.max(minTemp, Math.min(maxTemp, temp));
    return padding.left + ((clamped - minTemp) / (maxTemp - minTemp)) * usableWidth;
  };

  // Build SVG path strings
  const buildPath = (key: 'reconstructed' | 'reference' | 'climatology' | 'ridge') => {
    return points.reduce((acc, pt, i) => {
      const x = tempToX(pt[key]);
      const y = depthToY(pt.depth);
      return i === 0 ? `M ${x},${y}` : `${acc} L ${x},${y}`;
    }, '');
  };

  const activePoint = hoveredPoint || points.find(p => p.depth === selectedDepth) || points[0];

  return (
    <div className={`bg-[#060a12] border border-slate-800 rounded-2xl p-5 shadow-xl select-none ${className}`}>
      {/* Chart Top Header & Active Inspection readout */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <h3 className="text-sm font-bold text-white">Subsurface Vertical Temperature Profile T(z)</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Temperature (°C) vs Inverted Depth (0m surface down to 1000m seabed)
          </p>
        </div>

        {/* Selected Depth Badge Card */}
        {activePoint && (
          <div className="flex items-center gap-4 bg-slate-900 border border-slate-800 px-3.5 py-1.5 rounded-xl text-xs">
            <div>
              <span className="text-[10px] uppercase text-slate-400 block font-bold">Inspected Depth</span>
              <span className="font-mono text-cyan-300 font-bold text-sm">{activePoint.depth} m</span>
            </div>
            <div>
              <span className="text-[10px] uppercase text-slate-400 block font-bold">Predicted (B8)</span>
              <span className="font-mono text-white font-bold text-sm">{activePoint.reconstructed.toFixed(2)} °C</span>
            </div>
            <div>
              <span className="text-[10px] uppercase text-slate-400 block font-bold">Reference (GLORYS)</span>
              <span className="font-mono text-slate-300 font-bold text-sm">{activePoint.reference.toFixed(2)} °C</span>
            </div>
            <div>
              <span className="text-[10px] uppercase text-slate-400 block font-bold">Abs Error</span>
              <span className={`font-mono font-bold text-sm ${activePoint.error <= 0.8 ? 'text-emerald-400' : activePoint.error <= 1.5 ? 'text-amber-400' : 'text-rose-400'}`}>
                {activePoint.error.toFixed(2)} °C
              </span>
            </div>
          </div>
        )}
      </div>

      {/* SVG Chart Canvas */}
      <div className="relative w-full overflow-x-auto py-2">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-auto max-w-2xl mx-auto block"
        >
          <defs>
            {/* Thermocline zone background glow (50m - 200m) */}
            <linearGradient id="thermoGlow" x1="0%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#f59e0b" stopOpacity="0.04" />
              <stop offset="50%" stopColor="#f59e0b" stopOpacity="0.08" />
              <stop offset="100%" stopColor="#f59e0b" stopOpacity="0.02" />
            </linearGradient>
          </defs>

          {/* Thermocline Layer Highlight (50m to 200m) */}
          <rect
            x={padding.left}
            y={depthToY(50)}
            width={width - padding.left - padding.right}
            height={depthToY(200) - depthToY(50)}
            fill="url(#thermoGlow)"
          />
          <text
            x={width - padding.right - 10}
            y={depthToY(100)}
            fill="#d97706"
            fontSize="10"
            fontFamily="monospace"
            textAnchor="end"
            opacity="0.8"
          >
            THERMOCLINE ZONE (50–200m)
          </text>

          {/* Temperature X-Axis Grid Lines */}
          {[10, 15, 20, 25, 30].map(t => (
            <g key={`x-grid-${t}`}>
              <line
                x1={tempToX(t)}
                y1={padding.top}
                x2={tempToX(t)}
                y2={height - padding.bottom}
                stroke="#1e293b"
                strokeWidth="1"
                strokeDasharray="2 2"
              />
              <text
                x={tempToX(t)}
                y={height - padding.bottom + 18}
                fill="#64748b"
                fontSize="10"
                fontFamily="monospace"
                textAnchor="middle"
              >
                {t}°C
              </text>
            </g>
          ))}

          {/* Depth Y-Axis Grid Lines & Tick Labels */}
          {CANONICAL_DEPTHS.map((d) => (
            <g key={`y-grid-${d}`}>
              <line
                x1={padding.left}
                y1={depthToY(d)}
                x2={width - padding.right}
                y2={depthToY(d)}
                stroke="#1e293b"
                strokeWidth={d === 50 || d === 200 ? '1' : '0.5'}
                strokeOpacity={d === 50 || d === 200 ? '0.8' : '0.4'}
              />
              <text
                x={padding.left - 12}
                y={depthToY(d) + 3}
                fill={selectedDepth === d ? '#22d3ee' : '#64748b'}
                fontSize="10"
                fontFamily="monospace"
                fontWeight={selectedDepth === d ? 'bold' : 'normal'}
                textAnchor="end"
              >
                {d}m
              </text>
            </g>
          ))}

          {/* B1 Climatology Baseline Curve (Golden Dash) */}
          {showComparisonBaselines && (
            <path
              d={buildPath('climatology')}
              fill="none"
              stroke="#f59e0b"
              strokeWidth="1.5"
              strokeDasharray="4 4"
              opacity="0.75"
            />
          )}

          {/* B2 Ridge Baseline Curve (Blue) */}
          {showComparisonBaselines && (
            <path
              d={buildPath('ridge')}
              fill="none"
              stroke="#3b82f6"
              strokeWidth="1.5"
              strokeDasharray="2 2"
              opacity="0.75"
            />
          )}

          {/* GLORYS Target Reference Curve (White Solid with low opacity) */}
          <path
            d={buildPath('reference')}
            fill="none"
            stroke="#94a3b8"
            strokeWidth="2.5"
            strokeDasharray="6 3"
          />

          {/* B8 Reconstructed Temperature Curve (Bright Cyan Solid) */}
          <path
            d={buildPath('reconstructed')}
            fill="none"
            stroke="#06b6d4"
            strokeWidth="3.2"
          />

          {/* Interactive Depth Nodes */}
          {points.map((pt) => {
            const xRec = tempToX(pt.reconstructed);
            const xRef = tempToX(pt.reference);
            const y = depthToY(pt.depth);
            const isSelected = selectedDepth === pt.depth;
            const isHovered = hoveredPoint?.depth === pt.depth;

            return (
              <g
                key={`node-${pt.depth}`}
                className="cursor-pointer transition-all"
                onClick={() => onSelectDepth(pt.depth)}
                onMouseEnter={() => setHoveredPoint(pt)}
                onMouseLeave={() => setHoveredPoint(null)}
              >
                {/* Reference node */}
                <circle cx={xRef} cy={y} r="3" fill="#94a3b8" />

                {/* Reconstructed node */}
                <circle
                  cx={xRec}
                  cy={y}
                  r={isSelected || isHovered ? "6" : "4"}
                  fill={isSelected ? "#22d3ee" : "#06b6d4"}
                  stroke="#082f49"
                  strokeWidth="1.5"
                />

                {/* Connecting delta error line */}
                {Math.abs(xRec - xRef) > 4 && (
                  <line
                    x1={xRef}
                    y1={y}
                    x2={xRec}
                    y2={y}
                    stroke="#22d3ee"
                    strokeWidth="1"
                    strokeOpacity="0.4"
                  />
                )}
              </g>
            );
          })}
        </svg>
      </div>

      {/* Curve Legend */}
      <div className="flex flex-wrap items-center justify-center gap-5 pt-3 border-t border-slate-800 text-xs">
        <div className="flex items-center gap-2">
          <span className="w-4 h-0.5 bg-cyan-400"></span>
          <span className="text-white font-medium">B8 Spatiotemporal Model (Reconstructed)</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-4 h-0.5 border-t-2 border-dashed border-slate-400"></span>
          <span className="text-slate-400 font-medium">GLORYS Target (Reference)</span>
        </div>
        {showComparisonBaselines && (
          <>
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 border-t-2 border-dashed border-amber-500"></span>
              <span className="text-slate-400 font-medium">B1 Climatology</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-4 h-0.5 border-t-2 border-dotted border-blue-500"></span>
              <span className="text-slate-400 font-medium">B2 Ridge</span>
            </div>
          </>
        )}
      </div>

      {/* Canonical Depth Quick Selector Bar */}
      <div className="mt-4 pt-3 border-t border-slate-800/80">
        <div className="text-[11px] font-semibold text-slate-400 mb-2">Canonical Depth Levels (15 Depths):</div>
        <div className="flex flex-wrap items-center gap-1.5">
          {CANONICAL_DEPTHS.map((d) => (
            <button
              key={`btn-depth-${d}`}
              onClick={() => onSelectDepth(d)}
              className={`px-2 py-1 rounded-lg text-xs font-mono transition ${
                selectedDepth === d
                  ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm shadow-cyan-500/20'
                  : 'bg-slate-900 hover:bg-slate-800 text-slate-400 hover:text-white border border-slate-800'
              }`}
            >
              {d}m
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
