import React, { useRef, useState, useCallback, useEffect } from 'react';
import {
  MapPin,
  Crosshair,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Wind,
  Navigation,
  Compass,
  Eye,
  AlertTriangle,
  Info,
  Maximize2
} from 'lucide-react';
import {
  MAP_DOMAIN,
  lonToX,
  latToY,
  xToLon,
  yToLat,
  coordsToSvgPath,
  INDIA_COASTLINE,
  SRI_LANKA_COASTLINE,
  PAKISTAN_COASTLINE,
  ARABIA_COASTLINE,
  IRAN_COASTLINE,
  SOMALIA_COASTLINE,
  SOCOTRA_ISLAND,
  ABD_AL_KURI,
  BANGLADESH_COASTLINE,
  SE_ASIA_COASTLINE,
  SUMATRA_TIP,
  ANDAMAN_ISLANDS,
  LAKSHADWEEP_ATOLLS,
  MALDIVES_ATOLLS,
  CONTINENTAL_SHELF_200M,
  OCEAN_RIDGES,
  OCEAN_BASINS,
  MONSOON_CURRENTS,
  checkLandLocation,
  getEstimatedOceanDepth,
} from './oceanGeography';
import { resolveOceanRegion } from '../../mock/oceanData';

export type MapOverlayLayer = 'none' | 'sst' | 'sss' | 'ssh' | 'recon';
export type GridMode = 'graticule' | 'dense' | 'none';

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
  showPresets = true,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  // Zoom & Pan transformation state
  const [zoom, setZoom] = useState(1.0);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Interactive Layer & Display Controls
  const [activeLayer, setActiveLayer] = useState<MapOverlayLayer>('none');
  const [gridMode, setGridMode] = useState<GridMode>('graticule');
  const [showBathymetry, setShowBathymetry] = useState(true);
  const [showCurrents, setShowCurrents] = useState(false);
  const [showLabels, setShowLabels] = useState(true);

  // Cursor Hover State
  const [hoveredInfo, setHoveredInfo] = useState<{
    lat: number;
    lon: number;
    depth: number;
    region: string;
    isLand: boolean;
    landName?: string;
  } | null>(null);

  // Land Click Warning Toast
  const [landWarning, setLandWarning] = useState<{
    lat: number;
    lon: number;
    name: string;
  } | null>(null);

  // Clear land warning after 3.5 seconds
  useEffect(() => {
    if (!landWarning) return;
    const timer = setTimeout(() => setLandWarning(null), 3500);
    return () => clearTimeout(timer);
  }, [landWarning]);

  // Convert SVG client mouse event into Projected Ocean Coordinates
  const getEventOceanCoords = useCallback(
    (e: React.MouseEvent<SVGSVGElement>) => {
      if (!svgRef.current) return null;
      const rect = svgRef.current.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      // Scale from client pixels to SVG viewBox space
      const svgViewX = (clientX / rect.width) * MAP_DOMAIN.width;
      const svgViewY = (clientY / rect.height) * MAP_DOMAIN.height;

      // Invert pan & zoom transformation
      const rawX = (svgViewX - pan.x) / zoom;
      const rawY = (svgViewY - pan.y) / zoom;

      // Convert to Longitude & Latitude
      const lon = xToLon(rawX);
      const lat = yToLat(rawY);

      return {
        lat: Math.min(MAP_DOMAIN.latMax, Math.max(MAP_DOMAIN.latMin, lat)),
        lon: Math.min(MAP_DOMAIN.lonMax, Math.max(MAP_DOMAIN.lonMin, lon)),
        svgX: rawX,
        svgY: rawY,
      };
    },
    [pan.x, pan.y, zoom]
  );

  // Handle Mouse Click on the Ocean Map
  const handleSvgClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isDragging) return;
    const coords = getEventOceanCoords(e);
    if (!coords) return;

    // Check if the clicked location is masked continental land
    const landCheck = checkLandLocation(coords.lat, coords.lon);
    if (landCheck.isLand) {
      setLandWarning({
        lat: Math.round(coords.lat * 4) / 4,
        lon: Math.round(coords.lon * 4) / 4,
        name: landCheck.name || 'Continental Landmass',
      });
      return;
    }

    // Snap to canonical 0.25° grid
    const snapLat = Math.round(coords.lat * 4) / 4;
    const snapLon = Math.round(coords.lon * 4) / 4;

    onSelectCoordinates(snapLat, snapLon);
    setLandWarning(null);
  };

  // Handle Mouse Move for Real-Time Coordinate Readout
  const handleMouseMove = (e: React.MouseEvent<SVGSVGElement>) => {
    if (isDragging) {
      const dx = e.clientX - dragStart.x;
      const dy = e.clientY - dragStart.y;
      setPan((prev) => {
        const maxPanX = (MAP_DOMAIN.width * (zoom - 1)) / 2;
        const maxPanY = (MAP_DOMAIN.height * (zoom - 1)) / 2;
        return {
          x: Math.min(maxPanX + 150, Math.max(-maxPanX - 150, prev.x + dx)),
          y: Math.min(maxPanY + 100, Math.max(-maxPanY - 100, prev.y + dy)),
        };
      });
      setDragStart({ x: e.clientX, y: e.clientY });
      return;
    }

    const coords = getEventOceanCoords(e);
    if (!coords) return;

    const snapLat = Math.round(coords.lat * 4) / 4;
    const snapLon = Math.round(coords.lon * 4) / 4;
    const landCheck = checkLandLocation(snapLat, snapLon);
    const depth = landCheck.isLand ? 0 : getEstimatedOceanDepth(snapLat, snapLon);
    const region = resolveOceanRegion(snapLat, snapLon);

    setHoveredInfo({
      lat: snapLat,
      lon: snapLon,
      depth,
      region,
      isLand: landCheck.isLand,
      landName: landCheck.name,
    });
  };

  const handleMouseDown = (e: React.MouseEvent<SVGSVGElement>) => {
    if (e.button === 0 && (e.shiftKey || zoom > 1.0)) {
      setIsDragging(true);
      setDragStart({ x: e.clientX, y: e.clientY });
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleWheel = (e: React.WheelEvent<SVGSVGElement>) => {
    e.preventDefault();
    const zoomDelta = e.deltaY > 0 ? -0.2 : 0.2;
    setZoom((prev) => {
      const nextZoom = Math.min(4.0, Math.max(1.0, +(prev + zoomDelta).toFixed(1)));
      if (nextZoom === 1.0) setPan({ x: 0, y: 0 });
      return nextZoom;
    });
  };

  // Zoom Controls
  const zoomIn = () => setZoom((prev) => Math.min(4.0, +(prev + 0.3).toFixed(1)));
  const zoomOut = () =>
    setZoom((prev) => {
      const next = Math.max(1.0, +(prev - 0.3).toFixed(1));
      if (next === 1.0) setPan({ x: 0, y: 0 });
      return next;
    });
  const resetView = () => {
    setZoom(1.0);
    setPan({ x: 0, y: 0 });
  };

  // Focus View on Current Selected Marker
  const focusSelectedTarget = () => {
    const targetX = lonToX(selectedLon);
    const targetY = latToY(selectedLat);
    const targetZoom = 2.0;
    const nextPanX = MAP_DOMAIN.width / 2 - targetX * targetZoom;
    const nextPanY = MAP_DOMAIN.height / 2 - targetY * targetZoom;
    setZoom(targetZoom);
    setPan({ x: nextPanX, y: nextPanY });
  };

  // Selected Marker Projected Coordinates
  const markerX = lonToX(selectedLon);
  const markerY = latToY(selectedLat);
  const selectedRegion = resolveOceanRegion(selectedLat, selectedLon);
  const selectedDepth = getEstimatedOceanDepth(selectedLat, selectedLon);

  return (
    <div
      ref={containerRef}
      className={`relative bg-[#030712] border border-slate-800 rounded-2xl overflow-hidden select-none shadow-2xl flex flex-col ${className}`}
    >
      {/* ==================================================================== */}
      {/* TOP SCIENTIFIC TOOLBAR: Controls, Mode Toggles & Presets            */}
      {/* ==================================================================== */}
      <div className="px-3.5 py-2.5 bg-slate-950/90 border-b border-slate-800/80 backdrop-blur-md flex flex-wrap items-center justify-between gap-2.5 z-20">
        {/* Left Section: Resolution Badge & Layer Selector */}
        <div className="flex items-center gap-2 flex-wrap">
          <div className="px-2.5 py-1 rounded-md bg-cyan-950/50 border border-cyan-800/60 text-[11px] font-mono text-cyan-300 flex items-center gap-1.5 shadow-sm">
            <Crosshair className="w-3.5 h-3.5 text-cyan-400" />
            <span className="font-semibold tracking-wide">0.25° Mesh</span>
            <span className="text-[10px] text-cyan-400/70 border-l border-cyan-800/60 pl-1.5 font-sans">
              24,341 points
            </span>
          </div>

          {/* Layer Selector */}
          <div className="flex items-center rounded-lg bg-slate-900 border border-slate-800 p-0.5 text-[11px]">
            <button
              onClick={() => setActiveLayer('none')}
              className={`px-2 py-1 rounded-md transition font-medium ${
                activeLayer === 'none'
                  ? 'bg-slate-800 text-cyan-300 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              Bathymetry
            </button>
            <button
              onClick={() => setActiveLayer('sst')}
              className={`px-2 py-1 rounded-md transition font-medium flex items-center gap-1 ${
                activeLayer === 'sst'
                  ? 'bg-amber-950/80 text-amber-300 border border-amber-800/60 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>SST</span>
              <span className="text-[9px] font-mono opacity-80">(°C)</span>
            </button>
            <button
              onClick={() => setActiveLayer('sss')}
              className={`px-2 py-1 rounded-md transition font-medium flex items-center gap-1 ${
                activeLayer === 'sss'
                  ? 'bg-emerald-950/80 text-emerald-300 border border-emerald-800/60 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>SSS</span>
              <span className="text-[9px] font-mono opacity-80">(PSU)</span>
            </button>
            <button
              onClick={() => setActiveLayer('ssh')}
              className={`px-2 py-1 rounded-md transition font-medium flex items-center gap-1 ${
                activeLayer === 'ssh'
                  ? 'bg-purple-950/80 text-purple-300 border border-purple-800/60 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>SSH</span>
              <span className="text-[9px] font-mono opacity-80">(m)</span>
            </button>
            <button
              onClick={() => setActiveLayer('recon')}
              className={`px-2 py-1 rounded-md transition font-medium flex items-center gap-1 ${
                activeLayer === 'recon'
                  ? 'bg-cyan-950/90 text-cyan-300 border border-cyan-700/80 shadow-sm'
                  : 'text-slate-400 hover:text-white'
              }`}
            >
              <span>B8 Recon</span>
            </button>
          </div>
        </div>

        {/* Right Section: Toggles & Navigation Presets */}
        <div className="flex items-center gap-2 flex-wrap">
          {/* Quick Basin Presets */}
          {showPresets && (
            <div className="hidden xl:flex items-center gap-1 bg-slate-900/80 border border-slate-800/80 rounded-lg p-0.5">
              <span className="text-[10px] text-slate-500 font-mono px-1.5 uppercase tracking-wider">
                Basin:
              </span>
              <button
                onClick={() => onSelectCoordinates(15.25, 68.5)}
                className="px-2 py-0.5 rounded text-[10px] font-medium text-cyan-300 hover:bg-slate-800 transition"
              >
                Arabian
              </button>
              <button
                onClick={() => onSelectCoordinates(16.5, 88.5)}
                className="px-2 py-0.5 rounded text-[10px] font-medium text-blue-300 hover:bg-slate-800 transition"
              >
                Bay of Bengal
              </button>
              <button
                onClick={() => onSelectCoordinates(6.5, 78.5)}
                className="px-2 py-0.5 rounded text-[10px] font-medium text-indigo-300 hover:bg-slate-800 transition"
              >
                Equatorial
              </button>
              <button
                onClick={() => onSelectCoordinates(11.25, 54.5)}
                className="px-2 py-0.5 rounded text-[10px] font-medium text-emerald-300 hover:bg-slate-800 transition"
              >
                Somali
              </button>
              <button
                onClick={() => onSelectCoordinates(11.5, 93.25)}
                className="px-2 py-0.5 rounded text-[10px] font-medium text-teal-300 hover:bg-slate-800 transition"
              >
                Andaman
              </button>
            </div>
          )}

          {/* View Toggles */}
          <div className="flex items-center gap-1">
            {/* Currents Toggle */}
            <button
              onClick={() => setShowCurrents((v) => !v)}
              title="Toggle Monsoon Surface Circulation Vectors"
              className={`p-1.5 rounded-lg border text-xs transition flex items-center gap-1 ${
                showCurrents
                  ? 'bg-cyan-950/70 border-cyan-800 text-cyan-300'
                  : 'bg-slate-900 border-slate-800 text-slate-400 hover:text-slate-200'
              }`}
            >
              <Wind className="w-3.5 h-3.5" />
              <span className="text-[10px] hidden md:inline">Vectors</span>
            </button>

            {/* Grid Toggle */}
            <button
              onClick={() =>
                setGridMode((m) => (m === 'graticule' ? 'dense' : m === 'dense' ? 'none' : 'graticule'))
              }
              title={`Grid mode: ${gridMode.toUpperCase()}`}
              className={`p-1.5 rounded-lg border text-xs transition flex items-center gap-1 ${
                gridMode !== 'none'
                  ? 'bg-slate-800 border-slate-700 text-cyan-300'
                  : 'bg-slate-900 border-slate-800 text-slate-500'
              }`}
            >
              <Compass className="w-3.5 h-3.5" />
              <span className="text-[10px] font-mono hidden md:inline">
                {gridMode === 'graticule' ? '5° Grid' : gridMode === 'dense' ? '0.25° Mesh' : 'Grid Off'}
              </span>
            </button>

            {/* Labels Toggle */}
            <button
              onClick={() => setShowLabels((v) => !v)}
              title="Toggle Oceanographic Geographical Labels"
              className={`p-1.5 rounded-lg border text-xs transition ${
                showLabels
                  ? 'bg-slate-800 border-slate-700 text-slate-200'
                  : 'bg-slate-900 border-slate-800 text-slate-500'
              }`}
            >
              <span className="text-[10px] font-mono px-0.5">ABC</span>
            </button>
          </div>
        </div>
      </div>

      {/* ==================================================================== */}
      {/* INTERACTIVE OCEANOGRAPHIC MAP CANVAS (SVG)                           */}
      {/* ==================================================================== */}
      <div className="relative w-full overflow-hidden bg-[#020610] cursor-crosshair">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${MAP_DOMAIN.width} ${MAP_DOMAIN.height}`}
          className="w-full h-auto block select-none"
          onClick={handleSvgClick}
          onMouseMove={handleMouseMove}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onWheel={handleWheel}
          onMouseLeave={() => setHoveredInfo(null)}
        >
          <defs>
            {/* Deep Bathymetric Ocean Basin Gradients */}
            <radialGradient id="arabianDeepGrad" cx="30%" cy="40%" r="55%">
              <stop offset="0%" stopColor="#041830" />
              <stop offset="60%" stopColor="#031024" />
              <stop offset="100%" stopColor="#020814" />
            </radialGradient>

            <radialGradient id="bobDeepGrad" cx="70%" cy="45%" r="55%">
              <stop offset="0%" stopColor="#041a35" />
              <stop offset="65%" stopColor="#031024" />
              <stop offset="100%" stopColor="#020814" />
            </radialGradient>

            {/* Continental Landmass Gradient (Scientific Charcoal/Slate) */}
            <linearGradient id="landTopographyGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#141f30" />
              <stop offset="50%" stopColor="#101a28" />
              <stop offset="100%" stopColor="#0a121c" />
            </linearGradient>

            {/* Continental Shelf Gradient (0 - 200m depth isobath rim) */}
            <linearGradient id="shelfGrad" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#0ea5e9" stopOpacity="0.25" />
              <stop offset="100%" stopColor="#0284c7" stopOpacity="0.05" />
            </linearGradient>

            {/* Surface Variable Layer Shaders */}
            {/* SST Thermal Gradient */}
            <linearGradient id="sstFieldGrad" x1="0%" y1="100%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#0284c7" stopOpacity="0.45" /> {/* Cool Somali Upwelling 23°C */}
              <stop offset="25%" stopColor="#06b6d4" stopOpacity="0.5" />
              <stop offset="50%" stopColor="#10b981" stopOpacity="0.45" /> {/* 27°C */}
              <stop offset="75%" stopColor="#f59e0b" stopOpacity="0.5" /> {/* 28.5°C */}
              <stop offset="100%" stopColor="#f43f5e" stopOpacity="0.6" /> {/* Warm Pool 30.2°C */}
            </linearGradient>

            {/* SSS Haline Gradient */}
            <linearGradient id="sssFieldGrad" x1="0%" y1="30%" x2="100%" y2="70%">
              <stop offset="0%" stopColor="#a855f7" stopOpacity="0.55" /> {/* Arabian Hypersaline 36.8 */}
              <stop offset="45%" stopColor="#06b6d4" stopOpacity="0.45" /> {/* 35.0 PSU */}
              <stop offset="85%" stopColor="#10b981" stopOpacity="0.55" /> {/* Fresher BoB Plume 31.5 */}
              <stop offset="100%" stopColor="#eab308" stopOpacity="0.6" /> {/* Ganges Runoff Dilution 29 */}
            </linearGradient>

            {/* SSH Dynamic Altimetry Gradient */}
            <linearGradient id="sshFieldGrad" x1="20%" y1="0%" x2="80%" y2="100%">
              <stop offset="0%" stopColor="#1d4ed8" stopOpacity="0.5" /> {/* Cyclonic Dip -0.15m */}
              <stop offset="50%" stopColor="#475569" stopOpacity="0.25" /> {/* Neutral 0.00m */}
              <stop offset="100%" stopColor="#ec4899" stopOpacity="0.55" /> {/* Anticyclonic High +0.22m */}
            </linearGradient>

            {/* B8 Reconstruction Subsurface Gradient */}
            <linearGradient id="b8ReconGrad" x1="30%" y1="100%" x2="70%" y2="0%">
              <stop offset="0%" stopColor="#0369a1" stopOpacity="0.55" />
              <stop offset="35%" stopColor="#0891b2" stopOpacity="0.5" />
              <stop offset="70%" stopColor="#d97706" stopOpacity="0.5" />
              <stop offset="100%" stopColor="#e11d48" stopOpacity="0.6" />
            </linearGradient>

            {/* Vector Arrow Markers */}
            <marker id="currentArrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="4" markerHeight="4" orient="auto-start-reverse">
              <path d="M 0 1 L 8 5 L 0 9 z" fill="#22d3ee" fillOpacity="0.6" />
            </marker>

            {/* High-Precision Marker Glow */}
            <filter id="sonarGlow" x="-30%" y="-30%" width="160%" height="160%">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feComposite in="SourceGraphic" in2="blur" operator="over" />
            </filter>
          </defs>

          {/* Transformable Map Group for Zoom & Pan */}
          <g transform={`translate(${pan.x}, ${pan.y}) scale(${zoom})`}>
            {/* ============================================================== */}
            {/* 1. DEEP OCEAN BASE LAYER                                       */}
            {/* ============================================================== */}
            <rect width={MAP_DOMAIN.width} height={MAP_DOMAIN.height} fill="#020814" />
            <rect width={MAP_DOMAIN.width} height={MAP_DOMAIN.height} fill="url(#arabianDeepGrad)" />
            <rect width={MAP_DOMAIN.width} height={MAP_DOMAIN.height} fill="url(#bobDeepGrad)" opacity="0.8" />

            {/* ============================================================== */}
            {/* 2. REALISTIC BATHYMETRIC FEATURES (SUBSEA RIDGES & SHELVES)   */}
            {/* ============================================================== */}
            {showBathymetry && (
              <g id="bathymetricFeatures" opacity="0.9">
                {/* Abyssal Plains Subtle Shading */}
                <ellipse cx="280" cy="270" rx="140" ry="110" fill="#010610" opacity="0.7" />
                <ellipse cx="800" cy="280" rx="130" ry="120" fill="#010610" opacity="0.7" />
                <ellipse cx="540" cy="420" rx="280" ry="80" fill="#01050e" opacity="0.8" />

                {/* Continental Shelf 200m Fringe Ribbon */}
                <path
                  d={coordsToSvgPath(CONTINENTAL_SHELF_200M, false)}
                  stroke="#0284c7"
                  strokeWidth="6"
                  strokeOpacity="0.12"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  fill="none"
                />
                <path
                  d={coordsToSvgPath(CONTINENTAL_SHELF_200M, false)}
                  stroke="#38bdf8"
                  strokeWidth="1"
                  strokeOpacity="0.25"
                  strokeDasharray="4 3"
                  fill="none"
                />

                {/* Submarine Ridges & Trenches */}
                {OCEAN_RIDGES.map((ridge) => (
                  <g key={ridge.name} className="pointer-events-none">
                    {/* Ridge Glow Envelope */}
                    <path
                      d={coordsToSvgPath(ridge.coords, false)}
                      stroke="#06b6d4"
                      strokeWidth="5"
                      strokeOpacity="0.1"
                      strokeLinecap="round"
                      fill="none"
                    />
                    {/* Ridge Axis Line */}
                    <path
                      d={coordsToSvgPath(ridge.coords, false)}
                      stroke="#38bdf8"
                      strokeWidth="1.2"
                      strokeDasharray="2 3"
                      strokeOpacity="0.38"
                      fill="none"
                    />
                    {/* Ridge Label */}
                    {showLabels && (
                      <text
                        x={lonToX(ridge.labelPos[0])}
                        y={latToY(ridge.labelPos[1])}
                        fill="#0284c7"
                        fontSize="8.5"
                        fontFamily="monospace"
                        fontWeight="600"
                        letterSpacing="0.8"
                        textAnchor="middle"
                        opacity="0.7"
                      >
                        {ridge.name.toUpperCase()}
                      </text>
                    )}
                  </g>
                ))}
              </g>
            )}

            {/* ============================================================== */}
            {/* 3. ACTIVE SURFACE VARIABLE / RECONSTRUCTION OVERLAY LAYER      */}
            {/* ============================================================== */}
            {activeLayer !== 'none' && (
              <g id="surfaceOverlayLayer" className="transition-opacity duration-300">
                {activeLayer === 'sst' && (
                  <rect
                    width={MAP_DOMAIN.width}
                    height={MAP_DOMAIN.height}
                    fill="url(#sstFieldGrad)"
                    className="mix-blend-screen"
                  />
                )}
                {activeLayer === 'sss' && (
                  <rect
                    width={MAP_DOMAIN.width}
                    height={MAP_DOMAIN.height}
                    fill="url(#sssFieldGrad)"
                    className="mix-blend-screen"
                  />
                )}
                {activeLayer === 'ssh' && (
                  <rect
                    width={MAP_DOMAIN.width}
                    height={MAP_DOMAIN.height}
                    fill="url(#sshFieldGrad)"
                    className="mix-blend-screen"
                  />
                )}
                {activeLayer === 'recon' && (
                  <rect
                    width={MAP_DOMAIN.width}
                    height={MAP_DOMAIN.height}
                    fill="url(#b8ReconGrad)"
                    className="mix-blend-screen"
                  />
                )}
              </g>
            )}

            {/* ============================================================== */}
            {/* 4. MONSOON SURFACE CIRCULATION VECTORS (CURRENTS)              */}
            {/* ============================================================== */}
            {showCurrents && (
              <g id="circulationVectors" stroke="#06b6d4" strokeWidth="1.3" strokeOpacity="0.45" fill="none">
                {MONSOON_CURRENTS.map((c) => (
                  <path
                    key={c.id}
                    d={c.path}
                    markerEnd="url(#currentArrow)"
                    strokeDasharray="6 3"
                    className="animate-pulse"
                  />
                ))}
              </g>
            )}

            {/* ============================================================== */}
            {/* 5. CANONICAL COORDINATE GRATICULE & 0.25° RESEARCH MESH        */}
            {/* ============================================================== */}
            {gridMode !== 'none' && (
              <g id="coordinateGraticule">
                {/* Dense 0.25° Mesh (Fine scientific dot lattice when zoomed) */}
                {gridMode === 'dense' && (
                  <g stroke="#1e293b" strokeWidth="0.3" strokeDasharray="1 3" opacity="0.35">
                    {Array.from({ length: 60 * 4 + 1 }, (_, i) => 45 + i * 0.25)
                      .filter((lon) => lon % 5 !== 0)
                      .map((lon) => (
                        <line
                          key={`sublon-${lon}`}
                          x1={lonToX(lon)}
                          y1={0}
                          x2={lonToX(lon)}
                          y2={MAP_DOMAIN.height}
                        />
                      ))}
                    {Array.from({ length: 25 * 4 + 1 }, (_, i) => 5 + i * 0.25)
                      .filter((lat) => lat % 5 !== 0)
                      .map((lat) => (
                        <line
                          key={`sublat-${lat}`}
                          x1={0}
                          y1={latToY(lat)}
                          x2={MAP_DOMAIN.width}
                          y2={latToY(lat)}
                        />
                      ))}
                  </g>
                )}

                {/* Major 5° Graticule Lines */}
                <g stroke="#334155" strokeWidth="0.6" strokeDasharray="4 3" opacity="0.55">
                  {[50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100].map((lon) => (
                    <line
                      key={`major-lon-${lon}`}
                      x1={lonToX(lon)}
                      y1={0}
                      x2={lonToX(lon)}
                      y2={MAP_DOMAIN.height}
                    />
                  ))}
                  {[10, 15, 20, 25].map((lat) => (
                    <line
                      key={`major-lat-${lat}`}
                      x1={0}
                      y1={latToY(lat)}
                      x2={MAP_DOMAIN.width}
                      y2={latToY(lat)}
                    />
                  ))}
                </g>

                {/* Latitude & Longitude Numeric Scientific Callouts */}
                <g fill="#64748b" fontSize="9.5" fontFamily="monospace" fontWeight="500" textAnchor="middle">
                  {[50, 60, 70, 80, 90, 100].map((lon) => (
                    <text key={`lbl-lon-${lon}`} x={lonToX(lon)} y={MAP_DOMAIN.height - 10}>
                      {lon}°E
                    </text>
                  ))}
                  {[10, 15, 20, 25].map((lat) => (
                    <text key={`lbl-lat-${lat}`} x={24} y={latToY(lat) + 3} textAnchor="start">
                      {lat}°N
                    </text>
                  ))}
                </g>
              </g>
            )}

            {/* ============================================================== */}
            {/* 6. CONTINENTAL LANDMASSES & ISLANDS (GEBCO MASKED LAND)        */}
            {/* ============================================================== */}
            <g
              id="continentalLandmasses"
              fill="url(#landTopographyGrad)"
              stroke="#29394f"
              strokeWidth="1.2"
              strokeLinejoin="round"
              strokeLinecap="round"
            >
              {/* Indian Subcontinent */}
              <path d={coordsToSvgPath(INDIA_COASTLINE)} />

              {/* Sri Lanka */}
              <path d={coordsToSvgPath(SRI_LANKA_COASTLINE)} />

              {/* Pakistan & Makran */}
              <path d={coordsToSvgPath(PAKISTAN_COASTLINE)} />

              {/* Arabian Peninsula */}
              <path d={coordsToSvgPath(ARABIA_COASTLINE)} />

              {/* Iran / Persian Gulf */}
              <path d={coordsToSvgPath(IRAN_COASTLINE)} />

              {/* Horn of Africa (Somalia) */}
              <path d={coordsToSvgPath(SOMALIA_COASTLINE)} />

              {/* Socotra Island & Archipelago */}
              <path d={coordsToSvgPath(SOCOTRA_ISLAND)} />
              <path d={coordsToSvgPath(ABD_AL_KURI)} />

              {/* Bangladesh */}
              <path d={coordsToSvgPath(BANGLADESH_COASTLINE)} />

              {/* Southeast Asia (Myanmar, Thailand, Malacca) */}
              <path d={coordsToSvgPath(SE_ASIA_COASTLINE)} />

              {/* Northern Sumatra Tip */}
              <path d={coordsToSvgPath(SUMATRA_TIP)} />

              {/* Andaman & Nicobar Archipelago */}
              {ANDAMAN_ISLANDS.map((island, idx) => (
                <path key={`andaman-${idx}`} d={coordsToSvgPath(island)} />
              ))}

              {/* Lakshadweep Coral Atolls */}
              <g fill="#334155" stroke="#475569" strokeWidth="0.8">
                {LAKSHADWEEP_ATOLLS.map(([lon, lat], idx) => (
                  <circle
                    key={`laksh-${idx}`}
                    cx={lonToX(lon)}
                    cy={latToY(lat)}
                    r="2.2"
                  />
                ))}
              </g>

              {/* Maldives Atolls (>= 5°N) */}
              <g fill="#334155" stroke="#475569" strokeWidth="0.8">
                {MALDIVES_ATOLLS.map(([lon, lat], idx) => (
                  <circle
                    key={`maldives-${idx}`}
                    cx={lonToX(lon)}
                    cy={latToY(lat)}
                    r="2.2"
                  />
                ))}
              </g>
            </g>

            {/* ============================================================== */}
            {/* 7. OCEANOGRAPHIC BASIN & REGIONAL TYPOGRAPHY                   */}
            {/* ============================================================== */}
            {showLabels && (
              <g id="basinLabels" textAnchor="middle" className="pointer-events-none select-none">
                {OCEAN_BASINS.map((basin) => (
                  <text
                    key={basin.name}
                    x={lonToX(basin.lon)}
                    y={latToY(basin.lat)}
                    fill="#38bdf8"
                    fillOpacity="0.32"
                    fontSize={basin.size}
                    fontFamily="sans-serif"
                    fontWeight={basin.weight}
                    letterSpacing="2.5"
                  >
                    {basin.name}
                  </text>
                ))}
              </g>
            )}

            {/* ============================================================== */}
            {/* 8. STUDY DOMAIN EXACT BOUNDARY RECTANGLE & CORNER CALLOUTS     */}
            {/* ============================================================== */}
            <rect
              x={1}
              y={1}
              width={MAP_DOMAIN.width - 2}
              height={MAP_DOMAIN.height - 2}
              fill="none"
              stroke="#0ea5e9"
              strokeWidth="1.5"
              strokeOpacity="0.4"
            />

            {/* Corner Precision Crosshairs */}
            <g stroke="#38bdf8" strokeWidth="1.2" opacity="0.6">
              {/* NW: 30°N, 45°E */}
              <path d="M 0,14 L 14,14 L 14,0" fill="none" />
              {/* NE: 30°N, 105°E */}
              <path d={`M ${MAP_DOMAIN.width},14 L ${MAP_DOMAIN.width - 14},14 L ${MAP_DOMAIN.width - 14},0`} fill="none" />
              {/* SW: 5°N, 45°E */}
              <path d={`M 0,${MAP_DOMAIN.height - 14} L 14,${MAP_DOMAIN.height - 14} L 14,${MAP_DOMAIN.height}`} fill="none" />
              {/* SE: 5°N, 105°E */}
              <path d={`M ${MAP_DOMAIN.width},${MAP_DOMAIN.height - 14} L ${MAP_DOMAIN.width - 14},${MAP_DOMAIN.height - 14} L ${MAP_DOMAIN.width - 14},${MAP_DOMAIN.height}`} fill="none" />
            </g>

            {/* Corner Geographic Coordinates Badges */}
            <g fill="#0ea5e9" fontSize="9" fontFamily="monospace" fontWeight="bold" opacity="0.65">
              <text x={18} y={16}>30°N, 45°E</text>
              <text x={MAP_DOMAIN.width - 18} y={16} textAnchor="end">30°N, 105°E</text>
              <text x={18} y={MAP_DOMAIN.height - 8}>5°N, 45°E</text>
              <text x={MAP_DOMAIN.width - 18} y={MAP_DOMAIN.height - 8} textAnchor="end">5°N, 105°E</text>
            </g>

            {/* ============================================================== */}
            {/* 9. SELECTED OBSERVATION POINT RETICLE & RADAR PULSE            */}
            {/* ============================================================== */}
            {/* Crosshair guidelines across domain */}
            <g stroke="#06b6d4" strokeWidth="0.8" strokeDasharray="3 3" opacity="0.5">
              <line x1={markerX} y1={0} x2={markerX} y2={MAP_DOMAIN.height} />
              <line x1={0} y1={markerY} x2={MAP_DOMAIN.width} y2={markerY} />
            </g>

            {/* Sonar / Radar Reticle */}
            <g transform={`translate(${markerX}, ${markerY})`} className="pointer-events-none">
              {/* Expanding ping wave */}
              <circle r="18" fill="#06b6d4" fillOpacity="0.12" className="animate-ping" />
              {/* Outer reticle ring */}
              <circle r="9" fill="#0891b2" fillOpacity="0.3" stroke="#22d3ee" strokeWidth="1.5" />
              {/* Inner core */}
              <circle r="3.5" fill="#ffffff" filter="url(#sonarGlow)" />
            </g>

            {/* Floating Coordinate Pill Next to Marker */}
            <g
              transform={`translate(${
                Math.min(MAP_DOMAIN.width - 140, Math.max(75, markerX))
              }, ${markerY > 45 ? markerY - 18 : markerY + 28})`}
              className="pointer-events-none"
            >
              <rect
                x="-65"
                y="-13"
                width="130"
                height="22"
                rx="6"
                fill="#07111e"
                stroke="#06b6d4"
                strokeWidth="1.2"
                opacity="0.95"
              />
              <text
                x="0"
                y="2"
                fill="#22d3ee"
                fontSize="10"
                fontFamily="monospace"
                fontWeight="bold"
                textAnchor="middle"
              >
                {selectedLat.toFixed(2)}°N, {selectedLon.toFixed(2)}°E
              </text>
            </g>
          </g>
        </svg>

        {/* ================================================================== */}
        {/* FLOATING HUD CONTROLS: Zoom Buttons & Reset View                   */}
        {/* ================================================================== */}
        <div className="absolute bottom-4 right-4 z-20 flex flex-col items-center gap-1.5 bg-slate-900/90 border border-slate-800/90 rounded-xl p-1.5 backdrop-blur-md shadow-2xl">
          <button
            onClick={zoomIn}
            title="Zoom In (+)"
            className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          >
            <ZoomIn className="w-4 h-4" />
          </button>
          <div className="text-[10px] font-mono text-cyan-400 font-bold px-1 select-none">
            {zoom.toFixed(1)}×
          </div>
          <button
            onClick={zoomOut}
            title="Zoom Out (-)"
            className="p-1.5 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition"
          >
            <ZoomOut className="w-4 h-4" />
          </button>
          <div className="w-full h-px bg-slate-800 my-0.5" />
          <button
            onClick={focusSelectedTarget}
            title="Focus Target Marker"
            className="p-1.5 rounded-lg text-cyan-400 hover:text-cyan-200 hover:bg-slate-800 transition"
          >
            <Navigation className="w-4 h-4" />
          </button>
          <button
            onClick={resetView}
            title="Reset View Extent"
            className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>

        {/* ================================================================== */}
        {/* LAND CLICK INTERACTION WARNING TOAST                               */}
        {/* ================================================================== */}
        {landWarning && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2.5 px-3.5 py-2 rounded-xl bg-amber-950/95 border border-amber-700/80 text-amber-200 text-xs shadow-2xl backdrop-blur-md animate-in fade-in slide-in-from-top-2">
            <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
            <div>
              <span className="font-semibold">{landWarning.name}</span>
              <span className="text-amber-300/80 font-mono text-[11px] ml-1.5">
                [{landWarning.lat.toFixed(2)}°N, {landWarning.lon.toFixed(2)}°E]
              </span>
              <span className="block text-[10px] text-amber-300/70">
                Land cell masked by GEBCO Topography. Ocean ML framework operates only on marine grid points.
              </span>
            </div>
          </div>
        )}

        {/* ================================================================== */}
        {/* FLOATING SCALAR COLORBAR LEGEND (WHEN A LAYER IS ACTIVE)          */}
        {/* ================================================================== */}
        {activeLayer !== 'none' && (
          <div className="absolute bottom-4 left-4 z-20 bg-slate-950/95 border border-slate-800/90 rounded-xl p-2.5 backdrop-blur-md shadow-2xl text-[11px] font-mono space-y-1.5 min-w-[210px]">
            <div className="flex items-center justify-between text-slate-300 text-[10px] font-semibold uppercase tracking-wider">
              <span>
                {activeLayer === 'sst' && 'SST Thermal Field'}
                {activeLayer === 'sss' && 'SSS Haline Field'}
                {activeLayer === 'ssh' && 'SSH Anomaly Field'}
                {activeLayer === 'recon' && 'B8 Subsurface T(z)'}
              </span>
              <span className="text-cyan-400">
                {activeLayer === 'sst' && '°C'}
                {activeLayer === 'sss' && 'PSU'}
                {activeLayer === 'ssh' && 'm'}
                {activeLayer === 'recon' && '°C'}
              </span>
            </div>

            {/* Colormap bar */}
            <div className="h-2.5 w-full rounded-sm overflow-hidden border border-slate-700/60">
              {activeLayer === 'sst' && (
                <div className="w-full h-full bg-gradient-to-r from-sky-600 via-cyan-400 via-emerald-400 via-amber-400 to-rose-600" />
              )}
              {activeLayer === 'sss' && (
                <div className="w-full h-full bg-gradient-to-r from-purple-600 via-cyan-500 via-emerald-500 to-amber-500" />
              )}
              {activeLayer === 'ssh' && (
                <div className="w-full h-full bg-gradient-to-r from-blue-700 via-slate-500 to-pink-600" />
              )}
              {activeLayer === 'recon' && (
                <div className="w-full h-full bg-gradient-to-r from-sky-700 via-cyan-500 via-amber-500 to-rose-600" />
              )}
            </div>

            {/* Min / Mid / Max Tick Labels */}
            <div className="flex justify-between text-[10px] text-slate-400">
              <span>
                {activeLayer === 'sst' && '22.0'}
                {activeLayer === 'sss' && '28.0'}
                {activeLayer === 'ssh' && '-0.20'}
                {activeLayer === 'recon' && '0.50'}
              </span>
              <span>
                {activeLayer === 'sst' && '27.5'}
                {activeLayer === 'sss' && '34.5'}
                {activeLayer === 'ssh' && '0.00'}
                {activeLayer === 'recon' && '15.0'}
              </span>
              <span>
                {activeLayer === 'sst' && '30.5'}
                {activeLayer === 'sss' && '37.0'}
                {activeLayer === 'ssh' && '+0.25'}
                {activeLayer === 'recon' && '29.5'}
              </span>
            </div>
            <div className="text-[9px] text-slate-500 pt-0.5 border-t border-slate-800/80">
              Prototype Oceanographic Visualization
            </div>
          </div>
        )}
      </div>

      {/* ==================================================================== */}
      {/* BOTTOM SCIENTIFIC STATUS FOOTER & LIVE COORDINATE HUD                */}
      {/* ==================================================================== */}
      <div className="px-3.5 py-2.5 bg-slate-950/95 border-t border-slate-800/90 text-xs flex flex-col md:flex-row md:items-center justify-between gap-2 z-20">
        {/* Left: Active Selection Summary */}
        <div className="flex items-center gap-3 text-slate-300 text-[11px]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 ring-2 ring-cyan-400/30"></span>
            <span className="font-semibold text-white">Target:</span>
            <span className="font-mono text-cyan-300 font-bold">
              {selectedLat.toFixed(2)}°N, {selectedLon.toFixed(2)}°E
            </span>
          </div>

          <div className="hidden sm:flex items-center gap-1 text-slate-400 border-l border-slate-800 pl-3">
            <span>Region:</span>
            <span className="text-white font-medium">{selectedRegion}</span>
          </div>

          <div className="hidden lg:flex items-center gap-1 text-slate-400 border-l border-slate-800 pl-3">
            <span>GEBCO Depth:</span>
            <span className="text-slate-200 font-mono">~{selectedDepth.toLocaleString()} m</span>
          </div>
        </div>

        {/* Right: Live Cursor Coordinate Readout */}
        <div className="flex items-center gap-3 text-[11px] font-mono">
          {hoveredInfo ? (
            <div className="flex items-center gap-2">
              <span className="text-slate-500">CURSOR:</span>
              <span
                className={
                  hoveredInfo.isLand ? 'text-amber-400 font-semibold' : 'text-slate-200'
                }
              >
                {hoveredInfo.lat.toFixed(2)}°N, {hoveredInfo.lon.toFixed(2)}°E
              </span>
              {hoveredInfo.isLand ? (
                <span className="text-[10px] text-amber-500 bg-amber-950/80 px-1.5 py-0.5 rounded border border-amber-800/60">
                  LAND ({hoveredInfo.landName})
                </span>
              ) : (
                <span className="text-[10px] text-cyan-400 bg-cyan-950/60 px-1.5 py-0.5 rounded border border-cyan-800/50">
                  ~{hoveredInfo.depth}m • {hoveredInfo.region}
                </span>
              )}
            </div>
          ) : (
            <div className="text-slate-500 text-[10px]">
              Click any marine coordinate to update 15-depth profile reconstruction
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
