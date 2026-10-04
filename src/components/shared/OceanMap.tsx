import React, { useRef, useState } from 'react';
import { MapPin, Crosshair, ZoomIn, ZoomOut, Compass } from 'lucide-react';
import { STUDY_DOMAIN } from '../../mock/oceanData';

interface OceanMapProps {
  selectedLat: number;
  selectedLon: number;
  onSelectCoordinates: (lat: number, lon: number) => void;
  className?: string;
  showPresets?: boolean;
}

export const OceanMap: React.FC<OceanMapProps> = ({
  selectedLat,
  selectedLon,
  onSelectCoordinates,
  className = '',
  showPresets = true
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [hoveredCoords, setHoveredCoords] = useState<{ lat: number; lon: number } | null>(null);

  // Map coordinates projection:
  // Lon: 45.0 to 105.0 (60 degrees) -> X: 0 to 720
  // Lat: 30.0 to 5.0 (25 degrees, inverted) -> Y: 0 to 360
  const width = 720;
  const height = 360;

  const lonToX = (lon: number) => ((lon - 45.0) / 60.0) * width;
  const latToY = (lat: number) => ((30.0 - lat) / 25.0) * height;

  const xToLon = (x: number) => 45.0 + (x / width) * 60.0;
  const yToLat = (y: number) => 30.0 - (y / height) * 25.0;

  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const scaleX = width / rect.width;
    const scaleY = height / rect.height;

    const rawLon = xToLon(clickX * scaleX);
    const rawLat = yToLat(clickY * scaleY);

    onSelectCoordinates(rawLat, rawLon);
  };

  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!svgRef.current) return;
    const rect = svgRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const scaleX = width / rect.width;
    const scaleY = height / rect.height;

    const rawLon = Math.round(xToLon(clickX * scaleX) * 4) / 4;
    const rawLat = Math.round(yToLat(clickY * scaleY) * 4) / 4;

    setHoveredCoords({
      lat: Math.min(30, Math.max(5, rawLat)),
      lon: Math.min(105, Math.max(45, rawLon))
    });
  };

  const markerX = lonToX(selectedLon);
  const markerY = latToY(selectedLat);

  return (
    <div className={`relative bg-[#060a12] border border-slate-800 rounded-2xl overflow-hidden select-none ${className}`}>
      {/* Map Control Overlay Header */}
      <div className="absolute top-3 left-3 z-20 flex items-center gap-2">
        <div className="px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-700/80 text-[11px] font-mono text-cyan-300 flex items-center gap-1.5 shadow-md">
          <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
          <span>Interactive Grid: 0.25° Resolution</span>
        </div>

        {hoveredCoords && (
          <div className="hidden sm:flex px-2.5 py-1 rounded-lg bg-slate-900/90 border border-slate-700/80 text-[11px] font-mono text-slate-300 shadow-md">
            Cursor: {hoveredCoords.lat.toFixed(2)}°N, {hoveredCoords.lon.toFixed(2)}°E
          </div>
        )}
      </div>

      {/* Preset Basin Selector Pills */}
      {showPresets && (
        <div className="absolute top-3 right-3 z-20 flex items-center gap-1.5 flex-wrap">
          <button
            onClick={() => onSelectCoordinates(15.25, 68.50)}
            className="px-2 py-1 rounded-md text-[10px] font-semibold bg-slate-900/85 hover:bg-cyan-950/80 text-cyan-300 border border-cyan-800/50 transition shadow-sm"
          >
            Arabian Sea
          </button>
          <button
            onClick={() => onSelectCoordinates(16.50, 88.50)}
            className="px-2 py-1 rounded-md text-[10px] font-semibold bg-slate-900/85 hover:bg-blue-950/80 text-blue-300 border border-blue-800/50 transition shadow-sm"
          >
            Bay of Bengal
          </button>
          <button
            onClick={() => onSelectCoordinates(6.50, 78.50)}
            className="px-2 py-1 rounded-md text-[10px] font-semibold bg-slate-900/85 hover:bg-indigo-950/80 text-indigo-300 border border-indigo-800/50 transition shadow-sm"
          >
            Equatorial Jet
          </button>
          <button
            onClick={() => onSelectCoordinates(11.25, 54.50)}
            className="px-2 py-1 rounded-md text-[10px] font-semibold bg-slate-900/85 hover:bg-emerald-950/80 text-emerald-300 border border-emerald-800/50 transition shadow-sm"
          >
            Somali Coast
          </button>
        </div>
      )}

      {/* Interactive SVG Ocean Map */}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${width} ${height}`}
        className="w-full h-auto cursor-crosshair block"
        onClick={handleSvgClick}
        onMouseMove={handleMouseMove}
        onMouseLeave={() => setHoveredCoords(null)}
      >
        <defs>
          {/* Deep ocean background gradient */}
          <linearGradient id="oceanGrad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#081020" />
            <stop offset="50%" stopColor="#06122b" />
            <stop offset="100%" stopColor="#040b19" />
          </linearGradient>

          {/* Landmass gradient */}
          <linearGradient id="landGrad" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#1e293b" />
            <stop offset="100%" stopColor="#0f172a" />
          </linearGradient>

          {/* Marker pulse animation */}
          <filter id="glowMarker" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Ocean Background */}
        <rect width={width} height={height} fill="url(#oceanGrad)" />

        {/* Bathymetric Subtle Depth Rings (Simulated) */}
        <g stroke="#0ea5e9" strokeWidth="0.5" strokeOpacity="0.08" fill="none">
          <circle cx="280" cy="200" r="120" />
          <circle cx="500" cy="200" r="90" />
          <circle cx="380" cy="300" r="180" />
        </g>

        {/* Coordinate Grid Lines (every 5 degrees) */}
        <g stroke="#334155" strokeWidth="0.5" strokeDasharray="3 3" opacity="0.4">
          {/* Longitude Lines */}
          {[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100].map(lon => (
            <line key={`lon-${lon}`} x1={lonToX(lon)} y1={0} x2={lonToX(lon)} y2={height} />
          ))}
          {/* Latitude Lines */}
          {[10, 15, 20, 25].map(lat => (
            <line key={`lat-${lat}`} x1={0} y1={latToY(lat)} x2={width} y2={latToY(lat)} />
          ))}
        </g>

        {/* Grid Label Guides */}
        <g fill="#64748b" fontSize="9" fontFamily="monospace" textAnchor="middle">
          {[50, 60, 70, 80, 90, 100].map(lon => (
            <text key={`txt-lon-${lon}`} x={lonToX(lon)} y={height - 8}>{lon}°E</text>
          ))}
          {[10, 15, 20, 25].map(lat => (
            <text key={`txt-lat-${lat}`} x={20} y={latToY(lat) + 3}>{lat}°N</text>
          ))}
        </g>

        {/* Coastlines & Landmass Polygons (North Indian Ocean Geometry) */}
        <g fill="url(#landGrad)" stroke="#475569" strokeWidth="1" strokeLinejoin="round">
          {/* Arabian Peninsula & Gulf Coast */}
          <polygon points="
            0,0 210,0 205,30 190,50 170,70 185,85 160,110 135,130 90,140 40,165 0,175
          " />

          {/* Pakistan & Northern Western Boundary */}
          <polygon points="
            210,0 300,0 300,45 280,60 250,70 230,80 205,75 190,50 205,30
          " />

          {/* Indian Subcontinent (Main Peninsula) */}
          <polygon points="
            275,65 305,65 315,80 345,95 380,95 400,85 410,70 435,70 450,80 475,85 
            485,110 475,130 450,150 430,185 405,225 385,255 365,290 350,295 340,280
            335,250 325,220 315,190 295,160 280,140 260,135 250,120 265,110 275,90 270,75
          " />

          {/* Sri Lanka */}
          <polygon points="
            375,295 390,290 395,310 385,325 372,315
          " />

          {/* Bangladesh / Myanmar / Southeast Asian Coast */}
          <polygon points="
            485,85 530,85 540,115 545,145 560,170 565,200 580,240 595,280 610,320 625,360 
            720,360 720,0 485,0
          " />

          {/* Andaman & Nicobar Archipelago */}
          <g fill="#334155" stroke="#475569">
            <ellipse cx="570" cy="200" rx="3" ry="12" />
            <ellipse cx="575" cy="235" rx="3" ry="10" />
            <ellipse cx="580" cy="270" rx="3" ry="8" />
          </g>

          {/* Northern Sumatra Tip */}
          <polygon points="
            605,335 630,325 650,345 625,360
          " />
        </g>

        {/* Ocean Basin Geographic Labels */}
        <g fill="#38bdf8" fontFamily="sans-serif" fontWeight="700" letterSpacing="2" opacity="0.35" textAnchor="middle">
          <text x="180" y="210" fontSize="16">ARABIAN SEA</text>
          <text x="510" y="200" fontSize="16">BAY OF BENGAL</text>
          <text x="360" y="340" fontSize="13">EQUATORIAL INDIAN OCEAN</text>
        </g>

        {/* Current Circulation Flow Vectors (Monsoon Circulation Guides) */}
        <g stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.25" fill="none" strokeDasharray="4 2">
          {/* Arabian Sea Anticyclonic Gyre */}
          <path d="M 120,180 Q 200,150 240,210" markerEnd="url(#arrow)" />
          {/* Bay of Bengal Cyclonic Circulation */}
          <path d="M 440,220 Q 520,170 480,250" />
          {/* Equatorial Jet */}
          <path d="M 220,325 Q 360,335 550,325" />
        </g>

        {/* Selected Location Crosshairs */}
        <g stroke="#22d3ee" strokeWidth="1" strokeDasharray="3 3" opacity="0.65">
          <line x1={markerX} y1={0} x2={markerX} y2={height} />
          <line x1={0} y1={markerY} x2={width} y2={markerY} />
        </g>

        {/* Selected Location Marker Node */}
        <g transform={`translate(${markerX}, ${markerY})`}>
          {/* Pulsing radar ring */}
          <circle r="14" fill="#06b6d4" fillOpacity="0.15" className="animate-ping" />
          <circle r="8" fill="#0891b2" fillOpacity="0.4" stroke="#22d3ee" strokeWidth="1.5" />
          <circle r="3.5" fill="#ffffff" filter="url(#glowMarker)" />
        </g>

        {/* Selected Coordinates Tag Pill */}
        <g transform={`translate(${Math.min(width - 130, Math.max(70, markerX))}, ${markerY > 40 ? markerY - 14 : markerY + 24})`}>
          <rect x="-65" y="-12" width="130" height="20" rx="6" fill="#0f172a" stroke="#06b6d4" strokeWidth="1" opacity="0.95" />
          <text x="0" y="2" fill="#22d3ee" fontSize="10" fontFamily="monospace" fontWeight="bold" textAnchor="middle">
            {selectedLat.toFixed(2)}°N, {selectedLon.toFixed(2)}°E
          </text>
        </g>
      </svg>

      {/* Bottom Map Legend / Status Footer */}
      <div className="p-3 bg-slate-900/95 border-t border-slate-800 text-xs flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div className="flex items-center gap-4 text-[11px] text-slate-400">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span>
            <span>Selected Observation Point</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 bg-slate-800 border border-slate-700 rounded-sm"></span>
            <span>Landmass (GEBCO Masked)</span>
          </div>
        </div>

        <div className="text-[11px] font-mono text-cyan-400">
          Click any ocean grid point to update 15-depth reconstruction
        </div>
      </div>
    </div>
  );
};
