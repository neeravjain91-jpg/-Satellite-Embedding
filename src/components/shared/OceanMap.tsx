import React, { useRef, useState, useEffect, useCallback } from 'react';
import L from 'leaflet';
import {
  MapPin,
  RotateCcw,
  Maximize2,
  Minimize2,
  ZoomIn,
  ZoomOut,
  Layers,
  Calendar,
  Layers3,
  Play,
  Pause,
  ChevronLeft,
  ChevronRight,
  Info,
  Sliders,
  Wind,
  Navigation,
  Check,
  AlertCircle,
  Eye,
  EyeOff
} from 'lucide-react';
import {
  OceanManifest,
  loadOceanManifest,
  loadDayData,
  inspectPoint,
  renderScalarToCanvas,
  extractVectorArrows,
  PointInspectionResult,
  VectorSample
} from '../../services/oceanDataService';

export interface OceanMapProps {
  selectedLat: number;
  selectedLon: number;
  onSelectCoordinates: (lat: number, lon: number) => void;
  className?: string;
  showPresets?: boolean;
}

type BasemapProvider = 'carto-dark' | 'carto-light' | 'esri-ocean' | 'osm';

const BASEMAP_TILES: Record<BasemapProvider, { name: string; url: string; attribution: string; maxZoom: number }> = {
  'carto-dark': {
    name: 'CartoDB Dark Matter',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions" target="_blank" rel="noopener">CARTO</a>',
    maxZoom: 19
  },
  'carto-light': {
    name: 'CartoDB Positron',
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>',
    maxZoom: 19
  },
  'esri-ocean': {
    name: 'Esri World Ocean',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}',
    attribution: 'Tiles &copy; Esri &mdash; Sources: GEBCO, NOAA, CHS, OSU, UNH, CSUMB, DeLorme',
    maxZoom: 13
  },
  'osm': {
    name: 'OpenStreetMap Standard',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 19
  }
};

const OCEAN_PRESETS = [
  { name: 'Central Arabian Sea', lat: 15.0, lon: 65.0, desc: 'High salinity & oligotrophic central basin' },
  { name: 'Bay of Bengal Interior', lat: 15.0, lon: 88.0, desc: 'Ganges-Brahmaputra freshwater plume stratification' },
  { name: 'Somali Upwelling Boundary', lat: 9.0, lon: 52.0, desc: 'SW monsoon cold coastal upwelling' },
  { name: 'Sri Lanka Dome Region', lat: 8.0, lon: 83.5, desc: 'Monsoon cyclonic eddy & thermocline shallowing' },
  { name: 'Equatorial Indian Ocean', lat: 5.0, lon: 77.0, desc: 'Wyrtki jet equatorial waveguide' },
  { name: 'Andaman Sea Basin', lat: 11.5, lon: 94.0, desc: 'Internal solitary waves & basin bathymetry' },
];

export const OceanMap: React.FC<OceanMapProps> = ({
  selectedLat,
  selectedLon,
  onSelectCoordinates,
  className = '',
  showPresets = true,
}) => {
  const mapContainerRef = useRef<HTMLDivElement>(null);
  const leafletMapRef = useRef<L.Map | null>(null);
  const tileLayerRef = useRef<L.TileLayer | null>(null);
  const imageOverlayRef = useRef<L.ImageOverlay | null>(null);
  const boundaryLayerRef = useRef<L.Rectangle | null>(null);
  const vectorLayerGroupRef = useRef<L.LayerGroup | null>(null);
  const markerRef = useRef<L.Marker | null>(null);

  // Scientific data state
  const [manifest, setManifest] = useState<OceanManifest | null>(null);
  const [currentDayData, setCurrentDayData] = useState<Int16Array | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(true);
  const [loadingMsg, setLoadingMsg] = useState<string>('Initializing Ocean GIS...');
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Map configuration state
  const [activeLayer, setActiveLayer] = useState<string>('sst');
  const [activeDepth, setActiveDepth] = useState<number>(0);
  const [activeDate, setActiveDate] = useState<string>('2020-11-09');
  const [basemap, setBasemap] = useState<BasemapProvider>('carto-dark');
  const [layerOpacity, setLayerOpacity] = useState<number>(0.85);
  const [showVectors, setShowVectors] = useState<boolean>(true);
  const [showBoundary, setShowBoundary] = useState<boolean>(true);
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);

  // Inspection & Hover state
  const [hoverCoords, setHoverCoords] = useState<{ lat: number; lon: number } | null>(null);
  const [inspection, setInspection] = useState<PointInspectionResult | null>(null);
  const [showInspector, setShowInspector] = useState<boolean>(true);

  // 1. Initial Load: Load Manifest
  useEffect(() => {
    let mounted = true;
    async function init() {
      try {
        setLoadingMsg('Loading Research Domain Metadata...');
        const m = await loadOceanManifest();
        if (!mounted) return;
        setManifest(m);
        setActiveDate(m.defaultDate);
        setActiveDepth(m.defaultDepth);

        setLoadingMsg(`Loading Daily Scientific Slice (${m.defaultDate})...`);
        const dayArr = await loadDayData(m.defaultDate);
        if (!mounted) return;
        setCurrentDayData(dayArr);
        setIsLoading(false);
      } catch (err: any) {
        if (!mounted) return;
        console.error('Failed to initialize scientific map data:', err);
        setErrorMsg(err.message || 'Error loading scientific dataset.');
        setIsLoading(false);
      }
    }
    init();
    return () => {
      mounted = false;
    };
  }, []);

  // 2. Initialize Leaflet Map Instance
  useEffect(() => {
    if (!mapContainerRef.current || leafletMapRef.current) return;

    // Study Domain Bounds: [5.0, 45.0] to [30.0, 105.0]
    const southWest = L.latLng(5.0, 45.0);
    const northEast = L.latLng(30.0, 105.0);
    const bounds = L.latLngBounds(southWest, northEast);

    const map = L.map(mapContainerRef.current, {
      center: [17.5, 75.0],
      zoom: 4,
      minZoom: 3,
      maxZoom: 9,
      zoomControl: false,
      attributionControl: true,
      maxBounds: L.latLngBounds(L.latLng(-5.0, 30.0), L.latLng(40.0, 120.0)),
      maxBoundsViscosity: 0.8,
    });

    leafletMapRef.current = map;

    // Add Initial Basemap Tiles
    const tileConfig = BASEMAP_TILES['carto-dark'];
    const tileLayer = L.tileLayer(tileConfig.url, {
      attribution: tileConfig.attribution,
      maxZoom: tileConfig.maxZoom,
      subdomains: 'abcd',
    }).addTo(map);
    tileLayerRef.current = tileLayer;

    // Add Scale Bar
    L.control
      .scale({
        imperial: false,
        metric: true,
        position: 'bottomleft',
      })
      .addTo(map);

    // Add Domain Boundary Rectangle
    const boundary = L.rectangle(bounds, {
      color: '#06b6d4',
      weight: 1.5,
      dashArray: '5, 5',
      fill: false,
      interactive: false,
    }).addTo(map);
    boundaryLayerRef.current = boundary;

    // Layer group for vector glyphs
    const vectorGroup = L.layerGroup().addTo(map);
    vectorLayerGroupRef.current = vectorGroup;

    // Fit Initial View to Research Basin Domain
    map.fitBounds(bounds, { padding: [25, 25] });

    // Handle Map Movement / Coordinate Hover
    map.on('mousemove', (e: L.LeafletMouseEvent) => {
      const { lat, lng } = e.latlng;
      setHoverCoords({
        lat: Math.round(lat * 100) / 100,
        lon: Math.round(lng * 100) / 100,
      });
    });

    map.on('mouseout', () => {
      setHoverCoords(null);
    });

    // Handle Map Clicks: Snap to 0.25° grid and select
    map.on('click', (e: L.LeafletMouseEvent) => {
      const snapLat = Math.round(e.latlng.lat * 4) / 4;
      const snapLon = Math.round(e.latlng.lng * 4) / 4;

      // Bound to domain
      const boundedLat = Math.max(5.0, Math.min(30.0, snapLat));
      const boundedLon = Math.max(45.0, Math.min(105.0, snapLon));

      onSelectCoordinates(boundedLat, boundedLon);
    });

    return () => {
      map.remove();
      leafletMapRef.current = null;
    };
  }, [onSelectCoordinates]);

  // 3. Basemap Switcher
  useEffect(() => {
    if (!leafletMapRef.current) return;
    const map = leafletMapRef.current;
    if (tileLayerRef.current) {
      map.removeLayer(tileLayerRef.current);
    }
    const tileConfig = BASEMAP_TILES[basemap];
    const newTiles = L.tileLayer(tileConfig.url, {
      attribution: tileConfig.attribution,
      maxZoom: tileConfig.maxZoom,
      subdomains: 'abcd',
    }).addTo(map);
    tileLayerRef.current = newTiles;
  }, [basemap]);

  // 4. Update Boundary Visibility
  useEffect(() => {
    if (!boundaryLayerRef.current || !leafletMapRef.current) return;
    if (showBoundary) {
      boundaryLayerRef.current.addTo(leafletMapRef.current);
    } else {
      leafletMapRef.current.removeLayer(boundaryLayerRef.current);
    }
  }, [showBoundary]);

  // 5. Update Selected Location Marker
  useEffect(() => {
    if (!leafletMapRef.current) return;
    const map = leafletMapRef.current;

    // Custom pulsing target marker icon
    const customIcon = L.divIcon({
      className: 'ocean-selected-marker',
      html: `
        <div class="relative flex items-center justify-center w-8 h-8">
          <span class="absolute inline-flex w-7 h-7 rounded-full bg-cyan-400 opacity-75 animate-ping"></span>
          <span class="relative inline-flex items-center justify-center w-5 h-5 rounded-full bg-cyan-500 border-2 border-white shadow-lg text-slate-950 font-bold text-[9px]">
            +
          </span>
        </div>
      `,
      iconSize: [32, 32],
      iconAnchor: [16, 16],
    });

    if (markerRef.current) {
      markerRef.current.setLatLng([selectedLat, selectedLon]);
    } else {
      markerRef.current = L.marker([selectedLat, selectedLon], {
        icon: customIcon,
        zIndexOffset: 1000,
      }).addTo(map);
    }
  }, [selectedLat, selectedLon]);

  // 6. Handle Date Changes & Load Scientific Daily Slice
  useEffect(() => {
    let mounted = true;
    async function switchDate() {
      if (!manifest) return;
      try {
        setIsLoading(true);
        setLoadingMsg(`Loading Day Data: ${activeDate}...`);
        const dayArr = await loadDayData(activeDate);
        if (!mounted) return;
        setCurrentDayData(dayArr);
        setIsLoading(false);
      } catch (err: any) {
        if (!mounted) return;
        console.error('Error switching date:', err);
        setErrorMsg(`Failed to load ${activeDate}`);
        setIsLoading(false);
      }
    }
    switchDate();
    return () => {
      mounted = false;
    };
  }, [activeDate, manifest]);

  // 7. Render Scientific Raster Overlay to Leaflet
  useEffect(() => {
    if (!leafletMapRef.current || !manifest || !currentDayData) return;
    const map = leafletMapRef.current;

    if (activeLayer === 'none') {
      if (imageOverlayRef.current) {
        map.removeLayer(imageOverlayRef.current);
        imageOverlayRef.current = null;
      }
      return;
    }

    // Render scalar grid to offscreen canvas
    const canvas = document.createElement('canvas');
    renderScalarToCanvas(canvas, manifest, currentDayData, activeLayer, activeDepth, layerOpacity);
    const dataUrl = canvas.toDataURL();

    const bounds = L.latLngBounds(
      L.latLng(manifest.domain.latMin, manifest.domain.lonMin),
      L.latLng(manifest.domain.latMax, manifest.domain.lonMax)
    );

    if (imageOverlayRef.current) {
      imageOverlayRef.current.setUrl(dataUrl);
      imageOverlayRef.current.setOpacity(layerOpacity);
    } else {
      imageOverlayRef.current = L.imageOverlay(dataUrl, bounds, {
        opacity: layerOpacity,
        interactive: false,
      }).addTo(map);
    }
  }, [activeLayer, activeDepth, layerOpacity, currentDayData, manifest]);

  // 8. Render Vector Glyphs (Currents / Winds)
  useEffect(() => {
    if (!vectorLayerGroupRef.current || !manifest || !currentDayData) return;
    const group = vectorLayerGroupRef.current;
    group.clearLayers();

    const isCurrents = activeLayer === 'currents';
    const isWinds = activeLayer === 'winds';

    if ((!isCurrents && !isWinds) || !showVectors) {
      return;
    }

    const vectorLayerId = isCurrents ? 'currents' : 'winds';
    const vectors = extractVectorArrows(manifest, currentDayData, vectorLayerId, 1.5);

    vectors.forEach((vec: VectorSample) => {
      // Calculate arrow styling
      const strokeColor = isCurrents ? '#38bdf8' : '#facc15';
      const arrowLength = Math.min(22, Math.max(10, vec.speed * (isCurrents ? 14 : 1.2)));

      // SVG Arrow icon rotated to angle
      // Angle: 0 = East, 90 = North. SVG rotation is clockwise from Top (0 = North).
      // Standard SVG transform: rotation in degrees clockwise from East is -angleDeg
      const svgRot = -vec.angleDeg;

      const arrowIcon = L.divIcon({
        className: 'vector-glyph-marker',
        html: `
          <div style="transform: rotate(${svgRot}deg); transform-origin: 12px 12px; width: 24px; height: 24px; pointer-events: none;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
              <line x1="3" y1="12" x2="${3 + arrowLength}" y2="12" stroke="${strokeColor}" stroke-width="2" stroke-linecap="round"/>
              <polyline points="${3 + arrowLength - 5},8 ${3 + arrowLength},12 ${3 + arrowLength - 5},16" stroke="${strokeColor}" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
          </div>
        `,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      });

      L.marker([vec.lat, vec.lon], {
        icon: arrowIcon,
        interactive: false,
      }).addTo(group);
    });
  }, [activeLayer, showVectors, currentDayData, manifest]);

  // 9. Update Point Inspection on selection or date/depth change
  useEffect(() => {
    if (!manifest || !currentDayData) return;
    const result = inspectPoint(
      manifest,
      currentDayData,
      activeDate,
      activeDepth,
      selectedLat,
      selectedLon,
      activeLayer
    );
    setInspection(result);
  }, [manifest, currentDayData, activeDate, activeDepth, selectedLat, selectedLon, activeLayer]);

  // 10. Play / Pause Timeline Animation
  useEffect(() => {
    if (!isPlaying || !manifest) return;
    const dates = manifest.availableDates;
    const interval = setInterval(() => {
      setActiveDate((prev) => {
        const idx = dates.findIndex((d) => d.date === prev);
        const nextIdx = (idx + 1) % dates.length;
        return dates[nextIdx].date;
      });
    }, 1300);
    return () => clearInterval(interval);
  }, [isPlaying, manifest]);

  // Controls Handlers
  const handleResetView = useCallback(() => {
    if (!leafletMapRef.current) return;
    const southWest = L.latLng(5.0, 45.0);
    const northEast = L.latLng(30.0, 105.0);
    leafletMapRef.current.fitBounds(L.latLngBounds(southWest, northEast), { padding: [25, 25] });
  }, []);

  const handleZoomIn = () => leafletMapRef.current?.zoomIn();
  const handleZoomOut = () => leafletMapRef.current?.zoomOut();

  const handleToggleFullscreen = () => {
    setIsFullscreen((prev) => !prev);
    setTimeout(() => {
      leafletMapRef.current?.invalidateSize();
    }, 200);
  };

  const handlePrevDate = () => {
    if (!manifest) return;
    const dates = manifest.availableDates;
    const idx = dates.findIndex((d) => d.date === activeDate);
    if (idx > 0) setActiveDate(dates[idx - 1].date);
    else setActiveDate(dates[dates.length - 1].date);
  };

  const handleNextDate = () => {
    if (!manifest) return;
    const dates = manifest.availableDates;
    const idx = dates.findIndex((d) => d.date === activeDate);
    if (idx < dates.length - 1) setActiveDate(dates[idx + 1].date);
    else setActiveDate(dates[0].date);
  };

  const isLayer3D = manifest?.layers[activeLayer]?.is3D ?? false;
  const currentLayerMeta = manifest?.layers[activeLayer];

  return (
    <div
      className={`relative flex flex-col bg-[#070b14] border border-slate-800 rounded-2xl overflow-hidden shadow-2xl transition-all duration-300 ${
        isFullscreen ? 'fixed inset-0 z-50 rounded-none border-none' : className
      }`}
    >
      {/* =========================================================================
          GIS TOP BAR: RESEARCH BADGE & LAYER TOGGLE BAR
      ========================================================================= */}
      <div className="bg-[#090f1d] border-b border-slate-800 px-4 py-2.5 flex flex-wrap items-center justify-between gap-3 text-xs z-10">
        {/* Left: Study Basin Domain Badge & Active Date */}
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-cyan-950/80 border border-cyan-800/60 text-cyan-300 font-mono text-[11px] font-bold">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse"></span>
            <span>5°N–30°N · 45°E–105°E (0.25° GIS Grid)</span>
          </div>
          <span className="text-slate-400 font-medium hidden md:inline">|</span>
          <span className="text-slate-400 text-[11px] hidden md:inline">
            Historical research dataset · 2020
          </span>
        </div>

        {/* Right: Quick GIS View Actions */}
        <div className="flex items-center gap-2">
          {/* Basemap Switcher */}
          <div className="flex items-center gap-1 bg-slate-900 border border-slate-800 rounded-lg p-0.5">
            {(['carto-dark', 'carto-light', 'esri-ocean', 'osm'] as BasemapProvider[]).map((bm) => (
              <button
                key={bm}
                onClick={() => setBasemap(bm)}
                className={`px-2 py-1 rounded text-[10px] font-medium transition-all ${
                  basemap === bm
                    ? 'bg-cyan-500 text-slate-950 font-bold shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
                title={BASEMAP_TILES[bm].name}
              >
                {bm === 'carto-dark' ? 'Dark' : bm === 'carto-light' ? 'Light' : bm === 'esri-ocean' ? 'Ocean' : 'OSM'}
              </button>
            ))}
          </div>

          {/* Fullscreen Toggle */}
          <button
            onClick={handleToggleFullscreen}
            className="p-1.5 rounded-lg bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
            title={isFullscreen ? 'Exit Fullscreen' : 'Fullscreen Map'}
          >
            {isFullscreen ? <Minimize2 className="w-3.5 h-3.5" /> : <Maximize2 className="w-3.5 h-3.5" />}
          </button>
        </div>
      </div>

      {/* =========================================================================
          INTERACTIVE LAYER SELECTOR STRIP
      ========================================================================= */}
      <div className="bg-[#060a14] border-b border-slate-800/80 px-4 py-2 flex items-center gap-1.5 overflow-x-auto text-xs z-10 no-scrollbar">
        <span className="text-[11px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1 mr-1 shrink-0">
          <Layers className="w-3 h-3 text-cyan-400" />
          Layer:
        </span>

        {[
          { id: 'sst', label: 'SST (OISST)', is3D: false, color: 'text-amber-300' },
          { id: 'sss', label: 'SSS (SMAP)', is3D: false, color: 'text-emerald-300' },
          { id: 'ssh', label: 'SSH (AVISO)', is3D: false, color: 'text-blue-300' },
          { id: 'currents', label: 'Currents (OSCAR)', is3D: false, color: 'text-cyan-300' },
          { id: 'winds', label: 'Winds (CCMP)', is3D: false, color: 'text-yellow-300' },
          { id: 'thetao_ref', label: 'GLORYS Ref θo', is3D: true, color: 'text-indigo-300' },
          { id: 'b2_recon', label: 'B2 Ridge Recon', is3D: true, color: 'text-purple-300' },
          { id: 'b2_residual', label: 'Residual |Recon - Ref|', is3D: true, color: 'text-rose-300' },
          { id: 'none', label: 'None (Basemap)', is3D: false, color: 'text-slate-400' },
        ].map((layer) => {
          const isActive = activeLayer === layer.id;
          return (
            <button
              key={layer.id}
              onClick={() => setActiveLayer(layer.id)}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-medium shrink-0 transition-all flex items-center gap-1.5 ${
                isActive
                  ? 'bg-cyan-500/15 border border-cyan-400/80 text-white font-bold shadow-sm shadow-cyan-950'
                  : 'bg-slate-900/60 border border-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-800/80'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isActive ? 'bg-cyan-400' : 'bg-slate-600'}`}></span>
              <span>{layer.label}</span>
              {layer.is3D && (
                <span className="text-[9px] px-1 py-0.2 rounded bg-slate-800 text-indigo-300 font-mono">3D</span>
              )}
            </button>
          );
        })}
      </div>

      {/* =========================================================================
          MAP WORKSPACE CONTAINER
      ========================================================================= */}
      <div className="relative flex-1 min-h-[500px] h-[600px] w-full bg-[#050912]">
        {/* Leaflet DOM Anchor */}
        <div ref={mapContainerRef} className="absolute inset-0 z-0" />

        {/* Loading Overlay */}
        {isLoading && (
          <div className="absolute inset-0 z-30 bg-[#05080e]/75 backdrop-blur-sm flex flex-col items-center justify-center gap-2">
            <div className="w-8 h-8 rounded-full border-2 border-cyan-400 border-t-transparent animate-spin" />
            <span className="text-xs font-mono text-cyan-300">{loadingMsg}</span>
          </div>
        )}

        {/* Error Alert Overlay */}
        {errorMsg && (
          <div className="absolute top-4 left-4 right-4 z-40 bg-rose-950/90 border border-rose-800 text-rose-200 p-3 rounded-xl text-xs flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-rose-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            <button
              onClick={() => setErrorMsg(null)}
              className="px-2 py-1 rounded bg-rose-900/60 hover:bg-rose-800 text-rose-100 text-[10px]"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* MAP ON-CANVAS FLOATING TOOLBOX (TOP LEFT) */}
        <div className="absolute top-4 left-4 z-20 flex flex-col gap-2">
          {/* Zoom & Reset Controls */}
          <div className="bg-[#070d18]/90 backdrop-blur-md border border-slate-800 rounded-xl p-1 flex flex-col gap-1 shadow-xl">
            <button
              onClick={handleZoomIn}
              className="p-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-2 rounded-lg text-slate-300 hover:text-white hover:bg-slate-800 transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <div className="h-px bg-slate-800 my-0.5" />
            <button
              onClick={handleResetView}
              className="p-2 rounded-lg text-slate-300 hover:text-cyan-400 hover:bg-slate-800 transition-colors"
              title="Reset to Study Basin (5°N–30°N, 45°E–105°E)"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Layer Display Controls: Vectors & Boundary */}
          <div className="bg-[#070d18]/90 backdrop-blur-md border border-slate-800 rounded-xl p-1 flex flex-col gap-1 shadow-xl">
            {(activeLayer === 'currents' || activeLayer === 'winds') && (
              <button
                onClick={() => setShowVectors((p) => !p)}
                className={`p-2 rounded-lg transition-colors ${
                  showVectors ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white hover:bg-slate-800'
                }`}
                title={showVectors ? 'Hide Vector Glyphs' : 'Show Vector Glyphs'}
              >
                <Wind className="w-4 h-4" />
              </button>
            )}
            <button
              onClick={() => setShowBoundary((p) => !p)}
              className={`p-2 rounded-lg transition-colors ${
                showBoundary ? 'bg-cyan-500/20 text-cyan-300' : 'text-slate-400 hover:text-white hover:bg-slate-800'
              }`}
              title={showBoundary ? 'Hide 0.25° Study Boundary' : 'Show 0.25° Study Boundary'}
            >
              <Layers3 className="w-4 h-4" />
            </button>
          </div>
        </div>

        {/* FLOATING POINT INSPECTOR POPUP CARD (TOP RIGHT) */}
        {showInspector && inspection && (
          <div className="absolute top-4 right-4 z-20 w-80 max-w-[calc(100vw-2rem)] bg-[#070d18]/95 backdrop-blur-md border border-slate-800 rounded-2xl p-4 shadow-2xl space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800">
              <div className="flex items-center gap-1.5">
                <MapPin className="w-4 h-4 text-cyan-400" />
                <span className="text-xs font-bold text-white">Grid Point Inspection</span>
              </div>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 text-cyan-300 border border-slate-800">
                {inspection.lat.toFixed(2)}°N, {inspection.lon.toFixed(2)}°E
              </span>
            </div>

            {/* Region & Seafloor Status */}
            <div className="space-y-1.5 text-[11px]">
              <div className="flex justify-between text-slate-400">
                <span>Ocean Sub-basin:</span>
                <span className="text-slate-200 font-medium">{inspection.region}</span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Local Bathymetry:</span>
                <span className="text-slate-200 font-mono font-medium">
                  {inspection.isLand ? 'Landmass' : `${inspection.bathymetryDepth}m bottom`}
                </span>
              </div>
              <div className="flex justify-between text-slate-400">
                <span>Selected Date:</span>
                <span className="text-cyan-300 font-mono">{inspection.date}</span>
              </div>
              {isLayer3D && (
                <div className="flex justify-between text-slate-400">
                  <span>Selected Depth:</span>
                  <span className="text-indigo-300 font-mono font-bold">{inspection.depth}m</span>
                </div>
              )}
            </div>

            {/* Primary Active Value Showcase */}
            <div className="p-2.5 rounded-xl bg-slate-900/80 border border-slate-800 space-y-1">
              <div className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                {inspection.activeLayer.name}
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-2xl font-mono font-black text-white">
                  {inspection.isLand
                    ? 'LAND'
                    : inspection.isBelowSeafloor
                    ? 'BELOW SEAFLOOR'
                    : inspection.activeValue !== null
                    ? `${inspection.activeValue.toFixed(2)}`
                    : 'N/A'}
                </span>
                {!inspection.isLand && !inspection.isBelowSeafloor && inspection.activeValue !== null && (
                  <span className="text-xs font-bold text-cyan-400">{inspection.activeLayer.unit}</span>
                )}
              </div>
              <div className="text-[10px] text-slate-400 leading-tight">
                Source: {inspection.activeLayer.source}
              </div>
            </div>

            {/* Comprehensive Multi-Variable Breakdown Table */}
            {!inspection.isLand && (
              <div className="space-y-1 pt-1 text-[10px] font-mono border-t border-slate-800/80">
                <div className="text-[9px] font-bold text-slate-400 uppercase tracking-wider font-sans mb-1">
                  Co-Located Scientific Readings:
                </div>
                <div className="grid grid-cols-2 gap-1.5">
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">SST:</span>
                    <span className="text-amber-300">{inspection.allVariables.sst?.toFixed(2) ?? 'N/A'}°C</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">SSS:</span>
                    <span className="text-emerald-300">{inspection.allVariables.sss?.toFixed(2) ?? 'N/A'} PSU</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">SSH:</span>
                    <span className="text-blue-300">{inspection.allVariables.ssh?.toFixed(3) ?? 'N/A'} m</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">Current:</span>
                    <span className="text-cyan-300">{inspection.allVariables.currentSpeed?.toFixed(2) ?? 'N/A'} m/s</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">Wind:</span>
                    <span className="text-yellow-300">{inspection.allVariables.windSpeed?.toFixed(1) ?? 'N/A'} m/s</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">GLORYS θo:</span>
                    <span className="text-indigo-300">{inspection.allVariables.refThetao?.toFixed(2) ?? 'N/A'}°C</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">B2 Recon:</span>
                    <span className="text-purple-300">{inspection.allVariables.b2ReconThetao?.toFixed(2) ?? 'N/A'}°C</span>
                  </div>
                  <div className="bg-slate-950/60 p-1.5 rounded border border-slate-800/60 flex justify-between">
                    <span className="text-slate-400">Residual:</span>
                    <span className="text-rose-300">{inspection.allVariables.b2Residual?.toFixed(2) ?? 'N/A'}°C</span>
                  </div>
                </div>
              </div>
            )}

            {/* Scientific Integrity Note */}
            <p className="text-[9px] text-slate-500 leading-tight italic">
              {inspection.dataSourceNotes}
            </p>
          </div>
        )}

        {/* LIVE MOUSE COORDINATE FOOTER READOUT (BOTTOM RIGHT) */}
        <div className="absolute bottom-3 right-3 z-20 pointer-events-none">
          <div className="bg-[#070d18]/90 backdrop-blur-md border border-slate-800 rounded-lg px-2.5 py-1 text-[11px] font-mono text-slate-300 shadow-lg flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            <span>
              {hoverCoords ? `Lat: ${hoverCoords.lat.toFixed(2)}°N  Lon: ${hoverCoords.lon.toFixed(2)}°E` : 'Hover map for coordinates'}
            </span>
          </div>
        </div>
      </div>

      {/* =========================================================================
          GIS BOTTOM CONTROL BAR: DEPTH SELECTOR, DATE TIMELINE, COLORBAR
      ========================================================================= */}
      <div className="bg-[#070d18] border-t border-slate-800 p-4 space-y-4 z-10">
        {/* ROW 1: 15-DEPTH SELECTOR STRIP */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="text-xs font-bold text-white flex items-center gap-1.5">
              <span>🌊</span> Depth Level:
            </span>
            {!isLayer3D && (
              <span className="text-[10px] text-amber-400/90 bg-amber-950/40 border border-amber-800/40 px-2 py-0.5 rounded">
                Surface variable (0m only) • Vertical depths apply to GLORYS θo, B2 Recon, and Residual
              </span>
            )}
          </div>

          {/* Depth Buttons / Pills */}
          <div className="flex items-center gap-1 overflow-x-auto no-scrollbar py-0.5">
            {manifest?.canonicalDepths.map((d) => {
              const isSelected = activeDepth === d;
              return (
                <button
                  key={d}
                  disabled={!isLayer3D}
                  onClick={() => setActiveDepth(d)}
                  className={`px-2 py-1 rounded-md text-[10px] font-mono font-medium transition-all shrink-0 ${
                    !isLayer3D
                      ? 'opacity-40 cursor-not-allowed bg-slate-900 text-slate-500 border border-slate-800/50'
                      : isSelected
                      ? 'bg-gradient-to-r from-cyan-500 to-blue-600 text-slate-950 font-bold shadow-md shadow-cyan-950'
                      : 'bg-slate-900 border border-slate-800 text-slate-300 hover:text-white hover:bg-slate-800'
                  }`}
                >
                  {d}m
                </button>
              );
            })}
          </div>
        </div>

        {/* ROW 2: TIME NAVIGATOR & DATE SLIDER */}
        <div className="bg-[#050912] border border-slate-800/80 rounded-xl p-3 space-y-2">
          <div className="flex flex-wrap items-center justify-between gap-2 text-xs">
            <div className="flex items-center gap-2">
              <Calendar className="w-3.5 h-3.5 text-cyan-400" />
              <span className="font-bold text-white">Date Exploration (2020):</span>
              <span className="font-mono text-cyan-300 font-bold px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/60 text-[11px]">
                {activeDate}
              </span>
              <span className="text-[10px] text-slate-400">
                ({manifest?.availableDates.find((d) => d.date === activeDate)?.split.toUpperCase()} SPLIT)
              </span>
            </div>

            {/* Playback Controls */}
            <div className="flex items-center gap-1">
              <button
                onClick={handlePrevDate}
                className="p-1 rounded-md bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white"
                title="Previous Day"
              >
                <ChevronLeft className="w-3.5 h-3.5" />
              </button>
              <button
                onClick={() => setIsPlaying((p) => !p)}
                className={`px-2.5 py-1 rounded-md text-[11px] font-bold flex items-center gap-1 transition-all ${
                  isPlaying
                    ? 'bg-amber-500 text-slate-950'
                    : 'bg-cyan-500 hover:bg-cyan-400 text-slate-950'
                }`}
                title={isPlaying ? 'Pause Animation' : 'Auto-Play Daily Sequence'}
              >
                {isPlaying ? <Pause className="w-3 h-3 fill-current" /> : <Play className="w-3 h-3 fill-current" />}
                <span>{isPlaying ? 'Pause' : 'Play'}</span>
              </button>
              <button
                onClick={handleNextDate}
                className="p-1 rounded-md bg-slate-900 hover:bg-slate-800 border border-slate-800 text-slate-300 hover:text-white"
                title="Next Day"
              >
                <ChevronRight className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Slider across available dates */}
          <div className="pt-1">
            <input
              type="range"
              min={0}
              max={(manifest?.availableDates.length ?? 1) - 1}
              value={manifest?.availableDates.findIndex((d) => d.date === activeDate) ?? 0}
              onChange={(e) => {
                const idx = parseInt(e.target.value, 10);
                if (manifest?.availableDates[idx]) {
                  setActiveDate(manifest.availableDates[idx].date);
                }
              }}
              className="w-full accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
            />
          </div>
        </div>

        {/* ROW 3: COLORBAR LEGEND & LAYER DESCRIPTION */}
        {currentLayerMeta && activeLayer !== 'none' && (
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pt-1">
            {/* Colorbar Graphic */}
            <div className="space-y-1">
              <div className="flex justify-between items-center text-[10px] text-slate-400 font-mono">
                <span>{currentLayerMeta.min} {currentLayerMeta.unit}</span>
                <span className="font-bold text-slate-300">{currentLayerMeta.name} ({currentLayerMeta.unit})</span>
                <span>{currentLayerMeta.max} {currentLayerMeta.unit}</span>
              </div>
              <div
                className="h-3 w-72 max-w-full rounded-md border border-slate-700/80 shadow-inner"
                style={{
                  background:
                    activeLayer === 'sss'
                      ? 'linear-gradient(to right, rgb(20,10,60), rgb(20,80,140), rgb(20,150,140), rgb(100,200,100), rgb(240,240,120))'
                      : activeLayer === 'ssh'
                      ? 'linear-gradient(to right, rgb(15,65,170), rgb(85,160,230), rgb(240,240,245), rgb(240,140,90), rgb(180,20,30))'
                      : activeLayer === 'b2_residual'
                      ? 'linear-gradient(to right, rgb(245,245,245), rgb(254,224,139), rgb(244,109,67), rgb(213,62,79), rgb(110,10,70))'
                      : activeLayer === 'currents' || activeLayer === 'winds'
                      ? 'linear-gradient(to right, rgb(10,20,50), rgb(15,90,160), rgb(0,180,160), rgb(140,220,80), rgb(250,230,50))'
                      : 'linear-gradient(to right, rgb(4,3,15), rgb(24,40,110), rgb(0,130,180), rgb(60,190,120), rgb(230,200,50), rgb(235,110,30), rgb(180,10,30))',
                }}
              />
            </div>

            {/* Opacity Slider */}
            <div className="flex items-center gap-2 text-xs">
              <Sliders className="w-3.5 h-3.5 text-slate-400" />
              <span className="text-[11px] text-slate-400">Opacity:</span>
              <input
                type="range"
                min={0.3}
                max={1.0}
                step={0.05}
                value={layerOpacity}
                onChange={(e) => setLayerOpacity(parseFloat(e.target.value))}
                className="w-24 accent-cyan-400 cursor-pointer h-1.5 bg-slate-800 rounded-lg appearance-none"
              />
              <span className="text-[10px] font-mono text-cyan-300">{Math.round(layerOpacity * 100)}%</span>
            </div>
          </div>
        )}

        {/* ROW 4: OCEANOGRAPHIC PRESET QUICK JUMP */}
        {showPresets && (
          <div className="pt-2 border-t border-slate-800/80 space-y-1.5">
            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
              <span>📍</span> Oceanographic Landmark Presets:
            </div>
            <div className="flex flex-wrap gap-1.5">
              {OCEAN_PRESETS.map((p) => (
                <button
                  key={p.name}
                  onClick={() => onSelectCoordinates(p.lat, p.lon)}
                  className="px-2.5 py-1 rounded-lg bg-slate-900/80 hover:bg-slate-800 border border-slate-800 text-[11px] text-slate-300 hover:text-cyan-300 transition-all flex items-center gap-1"
                  title={`${p.desc} (${p.lat}°N, ${p.lon}°E)`}
                >
                  <MapPin className="w-2.5 h-2.5 text-cyan-400" />
                  <span>{p.name}</span>
                  <span className="text-[9px] font-mono text-slate-500">
                    ({p.lat}°, {p.lon}°)
                  </span>
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
