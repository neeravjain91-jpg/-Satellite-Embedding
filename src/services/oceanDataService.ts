import {
  COLORMAP_THERMAL,
  COLORMAP_HALINE,
  COLORMAP_BALANCE,
  COLORMAP_RESIDUAL,
  COLORMAP_SPEED,
  interpolateColormap
} from '../utils/scientificColormaps';

export interface LayerMeta {
  id: string;
  name: string;
  unit: string;
  source: string;
  min: number;
  max: number;
  is3D: boolean;
  isVector?: boolean;
  description: string;
  channelIdx?: number;
  channelOffset?: number;
  uChannelIdx?: number;
  vChannelIdx?: number;
  scale?: number;
  computed?: boolean;
}

export interface AvailableDateEntry {
  date: string;
  dayIdx: number;
  split: string;
  label: string;
}

export interface OceanManifest {
  domain: {
    latMin: number;
    latMax: number;
    lonMin: number;
    lonMax: number;
    latStep: number;
    lonStep: number;
    nLat: number;
    nLon: number;
    nOcean: number;
  };
  canonicalDepths: number[];
  availableDates: AvailableDateEntry[];
  defaultDate: string;
  defaultDepth: number;
  scaler: {
    features: string[];
    mean: number[];
    std: number[];
  };
  b2Coefficients: Record<string, {
    intercept: number;
    weights: Record<string, number>;
  }>;
  oceanCoords: {
    y: number[];
    x: number[];
  };
  gridToOcean: number[][];
  deepestValidDepth: number[];
  layers: Record<string, LayerMeta>;
}

export interface PointInspectionResult {
  lat: number;
  lon: number;
  date: string;
  depth: number;
  activeLayer: LayerMeta;
  activeValue: number | null;
  region: string;
  isLand: boolean;
  isBelowSeafloor: boolean;
  bathymetryDepth: number;
  allVariables: {
    sst: number | null;
    sss: number | null;
    ssh: number | null;
    currentU: number | null;
    currentV: number | null;
    currentSpeed: number | null;
    windU: number | null;
    windV: number | null;
    windSpeed: number | null;
    refThetao: number | null;
    b2ReconThetao: number | null;
    b2Residual: number | null;
  };
  dataSourceNotes: string;
}

export interface VectorSample {
  lat: number;
  lon: number;
  u: number;
  v: number;
  speed: number;
  angleDeg: number;
}

// In-memory cache
let manifestCache: OceanManifest | null = null;
const dayDataCache: Map<string, Int16Array> = new Map();

/**
 * Loads the manifest metadata from public/data/ocean_manifest.json
 */
export async function loadOceanManifest(): Promise<OceanManifest> {
  if (manifestCache) return manifestCache;
  const res = await fetch('/data/ocean_manifest.json');
  if (!res.ok) {
    throw new Error(`Failed to load ocean manifest: HTTP ${res.status}`);
  }
  manifestCache = (await res.json()) as OceanManifest;
  return manifestCache;
}

/**
 * Loads the binary daily slice for a specific date from public/data/daily/{date}.bin
 */
export async function loadDayData(date: string): Promise<Int16Array> {
  if (dayDataCache.has(date)) {
    return dayDataCache.get(date)!;
  }
  const res = await fetch(`/data/daily/${date}.bin`);
  if (!res.ok) {
    throw new Error(`Failed to load daily scientific slice for ${date}: HTTP ${res.status}`);
  }
  const buf = await res.arrayBuffer();
  const arr = new Int16Array(buf);
  dayDataCache.set(date, arr);
  return arr;
}

/**
 * Helper to resolve geographical sub-basin name from coordinates
 */
export function getOceanSubRegionName(lat: number, lon: number): string {
  if (lon < 77 && lat >= 10) return 'Arabian Sea (Central/North)';
  if (lon < 77 && lat < 10) return 'Western Equatorial Indian Ocean';
  if (lon >= 77 && lon <= 92 && lat >= 10) return 'Bay of Bengal (Central/North)';
  if (lon > 92 && lat >= 6 && lat <= 16) return 'Andaman Sea';
  if (lon >= 77 && lat < 10) return 'Eastern Equatorial Indian Ocean';
  if (lat >= 22 && lon < 60) return 'Gulf of Oman / Persian Gulf Gateway';
  if (lat < 14 && lon < 52) return 'Gulf of Aden / Somali Coast';
  return 'North Indian Ocean Basin';
}

/**
 * Inspect a specific grid point on the 0.25° grid
 */
export function inspectPoint(
  manifest: OceanManifest,
  dayData: Int16Array,
  date: string,
  depth: number,
  lat: number,
  lon: number,
  layerId: string
): PointInspectionResult {
  const { domain, gridToOcean, deepestValidDepth, canonicalDepths, scaler, b2Coefficients, layers } = manifest;
  const nOcean = domain.nOcean;

  const y = Math.round((lat - domain.latMin) / domain.latStep);
  const x = Math.round((lon - domain.lonMin) / domain.lonStep);

  const snappedLat = domain.latMin + y * domain.latStep;
  const snappedLon = domain.lonMin + x * domain.lonStep;
  const region = getOceanSubRegionName(snappedLat, snappedLon);
  const activeLayer = layers[layerId] || layers['sst'];

  // Out of bounds check
  if (y < 0 || y >= domain.nLat || x < 0 || x >= domain.nLon) {
    return {
      lat: snappedLat,
      lon: snappedLon,
      date,
      depth,
      activeLayer,
      activeValue: null,
      region,
      isLand: true,
      isBelowSeafloor: false,
      bathymetryDepth: 0,
      allVariables: {
        sst: null,
        sss: null,
        ssh: null,
        currentU: null,
        currentV: null,
        currentSpeed: null,
        windU: null,
        windV: null,
        windSpeed: null,
        refThetao: null,
        b2ReconThetao: null,
        b2Residual: null,
      },
      dataSourceNotes: 'Location falls outside the 5°N–30°N, 45°E–105°E research domain.',
    };
  }

  const oceanIdx = gridToOcean[y][x];
  if (oceanIdx === -1) {
    return {
      lat: snappedLat,
      lon: snappedLon,
      date,
      depth,
      activeLayer,
      activeValue: null,
      region,
      isLand: true,
      isBelowSeafloor: false,
      bathymetryDepth: 0,
      allVariables: {
        sst: null,
        sss: null,
        ssh: null,
        currentU: null,
        currentV: null,
        currentSpeed: null,
        windU: null,
        windV: null,
        windSpeed: null,
        refThetao: null,
        b2ReconThetao: null,
        b2Residual: null,
      },
      dataSourceNotes: 'Continental landmass or island (masked in scientific ocean target).',
    };
  }

  const bathymetryDepth = deepestValidDepth[oceanIdx] || 0;
  const isBelowSeafloor = depth > bathymetryDepth;

  // Extract raw channels
  const rawSst = dayData[0 * nOcean + oceanIdx];
  const rawSss = dayData[1 * nOcean + oceanIdx];
  const rawSsh = dayData[2 * nOcean + oceanIdx];
  const rawCurU = dayData[3 * nOcean + oceanIdx];
  const rawCurV = dayData[4 * nOcean + oceanIdx];
  const rawWindU = dayData[5 * nOcean + oceanIdx];
  const rawWindV = dayData[6 * nOcean + oceanIdx];

  const sst = rawSst === -32768 ? null : rawSst / 100.0;
  const sss = rawSss === -32768 ? null : rawSss / 100.0;
  const ssh = rawSsh === -32768 ? null : rawSsh / 1000.0;
  const currentU = rawCurU === -32768 ? null : rawCurU / 1000.0;
  const currentV = rawCurV === -32768 ? null : rawCurV / 1000.0;
  const currentSpeed =
    currentU !== null && currentV !== null ? Math.sqrt(currentU * currentU + currentV * currentV) : null;
  const windU = rawWindU === -32768 ? null : rawWindU / 100.0;
  const windV = rawWindV === -32768 ? null : rawWindV / 100.0;
  const windSpeed =
    windU !== null && windV !== null ? Math.sqrt(windU * windU + windV * windV) : null;

  // Reference thetao
  const depthIdx = canonicalDepths.indexOf(depth);
  let refThetao: number | null = null;
  if (!isBelowSeafloor && depthIdx >= 0) {
    const rawRef = dayData[(7 + depthIdx) * nOcean + oceanIdx];
    refThetao = rawRef === -32768 ? null : rawRef / 100.0;
  }

  // Model B2 Reconstruction
  let b2ReconThetao: number | null = null;
  if (!isBelowSeafloor && sst !== null && sss !== null && ssh !== null && currentU !== null && currentV !== null && windU !== null && windV !== null) {
    const depthKey = `${depth.toFixed(1)}`;
    const b2Coeff = b2Coefficients[depthKey];
    if (b2Coeff) {
      const featVals = [sst, sss, ssh, currentU, currentV, windU, windV];
      let sum = b2Coeff.intercept;
      for (let i = 0; i < 7; i++) {
        const norm = (featVals[i] - scaler.mean[i]) / scaler.std[i];
        const fn = scaler.features[i];
        const w = b2Coeff.weights[fn] || 0;
        sum += norm * w;
      }
      b2ReconThetao = Math.round(sum * 100) / 100;
    }
  }

  // B2 Residual
  let b2Residual: number | null = null;
  if (b2ReconThetao !== null && refThetao !== null) {
    b2Residual = Math.round(Math.abs(b2ReconThetao - refThetao) * 100) / 100;
  }

  // Determine active value
  let activeValue: number | null = null;
  if (layerId === 'sst') activeValue = sst;
  else if (layerId === 'sss') activeValue = sss;
  else if (layerId === 'ssh') activeValue = ssh;
  else if (layerId === 'currents') activeValue = currentSpeed;
  else if (layerId === 'winds') activeValue = windSpeed;
  else if (layerId === 'thetao_ref') activeValue = refThetao;
  else if (layerId === 'b2_recon') activeValue = b2ReconThetao;
  else if (layerId === 'b2_residual') activeValue = b2Residual;

  let notes = `Actual scientific record for ${date}. Bathymetric bottom limit: ${bathymetryDepth}m.`;
  if (isBelowSeafloor) {
    notes = `Depth ${depth}m extends below local seafloor bathymetry (${bathymetryDepth}m). Target masked as NaN in physical oceanography.`;
  }

  return {
    lat: snappedLat,
    lon: snappedLon,
    date,
    depth,
    activeLayer,
    activeValue,
    region,
    isLand: false,
    isBelowSeafloor,
    bathymetryDepth,
    allVariables: {
      sst,
      sss,
      ssh,
      currentU,
      currentV,
      currentSpeed,
      windU,
      windV,
      windSpeed,
      refThetao,
      b2ReconThetao,
      b2Residual,
    },
    dataSourceNotes: notes,
  };
}

/**
 * Renders a scalar field onto an HTML Canvas (241 cols x 101 rows)
 * Canvas py=0 (Top) corresponds to Lat 30.0N (y=100)
 * Canvas py=100 (Bottom) corresponds to Lat 5.0N (y=0)
 */
export function renderScalarToCanvas(
  canvas: HTMLCanvasElement,
  manifest: OceanManifest,
  dayData: Int16Array,
  layerId: string,
  depth: number,
  opacity: number = 0.88
): void {
  const { domain, gridToOcean, deepestValidDepth, canonicalDepths, scaler, b2Coefficients, layers } = manifest;
  const nOcean = domain.nOcean;
  const layer = layers[layerId] || layers['sst'];

  canvas.width = domain.nLon;  // 241
  canvas.height = domain.nLat; // 101

  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const imgData = ctx.createImageData(domain.nLon, domain.nLat);
  const data = imgData.data;

  // Pick colormap
  let colormap = COLORMAP_THERMAL;
  if (layerId === 'sss') colormap = COLORMAP_HALINE;
  else if (layerId === 'ssh') colormap = COLORMAP_BALANCE;
  else if (layerId === 'b2_residual') colormap = COLORMAP_RESIDUAL;
  else if (layerId === 'currents' || layerId === 'winds') colormap = COLORMAP_SPEED;

  const minVal = layer.min;
  const maxVal = layer.max;
  const range = maxVal - minVal || 1;

  const depthIdx = canonicalDepths.indexOf(depth);
  const depthKey = `${depth.toFixed(1)}`;
  const b2Coeff = b2Coefficients[depthKey];

  const alphaByte = Math.round(opacity * 255);

  // Pre-calculate B2 coefficients if needed
  let weightsArr: number[] | null = null;
  let interceptVal = 0;
  if (layerId === 'b2_recon' || layerId === 'b2_residual') {
    if (b2Coeff) {
      interceptVal = b2Coeff.intercept;
      weightsArr = scaler.features.map(f => b2Coeff.weights[f] || 0);
    }
  }

  for (let py = 0; py < domain.nLat; py++) {
    const y = domain.nLat - 1 - py; // Invert latitude: py=0 is North (Lat 30)
    const rowOffset = py * domain.nLon * 4;

    for (let px = 0; px < domain.nLon; px++) {
      const x = px;
      const pixelIdx = rowOffset + px * 4;

      const oceanIdx = gridToOcean[y][x];
      if (oceanIdx === -1) {
        // Transparent on land
        data[pixelIdx + 3] = 0;
        continue;
      }

      // Check seafloor bathymetry if depth-dependent
      if (layer.is3D) {
        const maxD = deepestValidDepth[oceanIdx];
        if (depth > maxD) {
          // Below seafloor: transparent
          data[pixelIdx + 3] = 0;
          continue;
        }
      }

      let val: number | null = null;

      if (layerId === 'sst') {
        const raw = dayData[0 * nOcean + oceanIdx];
        if (raw !== -32768) val = raw / 100.0;
      } else if (layerId === 'sss') {
        const raw = dayData[1 * nOcean + oceanIdx];
        if (raw !== -32768) val = raw / 100.0;
      } else if (layerId === 'ssh') {
        const raw = dayData[2 * nOcean + oceanIdx];
        if (raw !== -32768) val = raw / 1000.0;
      } else if (layerId === 'currents') {
        const rawU = dayData[3 * nOcean + oceanIdx];
        const rawV = dayData[4 * nOcean + oceanIdx];
        if (rawU !== -32768 && rawV !== -32768) {
          const u = rawU / 1000.0;
          const v = rawV / 1000.0;
          val = Math.sqrt(u * u + v * v);
        }
      } else if (layerId === 'winds') {
        const rawU = dayData[5 * nOcean + oceanIdx];
        const rawV = dayData[6 * nOcean + oceanIdx];
        if (rawU !== -32768 && rawV !== -32768) {
          const u = rawU / 100.0;
          const v = rawV / 100.0;
          val = Math.sqrt(u * u + v * v);
        }
      } else if (layerId === 'thetao_ref') {
        if (depthIdx >= 0) {
          const raw = dayData[(7 + depthIdx) * nOcean + oceanIdx];
          if (raw !== -32768) val = raw / 100.0;
        }
      } else if (layerId === 'b2_recon') {
        if (weightsArr) {
          const rawSst = dayData[0 * nOcean + oceanIdx];
          const rawSss = dayData[1 * nOcean + oceanIdx];
          const rawSsh = dayData[2 * nOcean + oceanIdx];
          const rawCurU = dayData[3 * nOcean + oceanIdx];
          const rawCurV = dayData[4 * nOcean + oceanIdx];
          const rawWindU = dayData[5 * nOcean + oceanIdx];
          const rawWindV = dayData[6 * nOcean + oceanIdx];

          if (rawSst !== -32768 && rawSss !== -32768 && rawSsh !== -32768 &&
              rawCurU !== -32768 && rawCurV !== -32768 && rawWindU !== -32768 && rawWindV !== -32768) {
            const f0 = (rawSst / 100.0 - scaler.mean[0]) / scaler.std[0];
            const f1 = (rawSss / 100.0 - scaler.mean[1]) / scaler.std[1];
            const f2 = (rawSsh / 1000.0 - scaler.mean[2]) / scaler.std[2];
            const f3 = (rawCurU / 1000.0 - scaler.mean[3]) / scaler.std[3];
            const f4 = (rawCurV / 1000.0 - scaler.mean[4]) / scaler.std[4];
            const f5 = (rawWindU / 100.0 - scaler.mean[5]) / scaler.std[5];
            const f6 = (rawWindV / 100.0 - scaler.mean[6]) / scaler.std[6];

            val = interceptVal +
              f0 * weightsArr[0] + f1 * weightsArr[1] + f2 * weightsArr[2] +
              f3 * weightsArr[3] + f4 * weightsArr[4] + f5 * weightsArr[5] + f6 * weightsArr[6];
          }
        }
      } else if (layerId === 'b2_residual') {
        if (weightsArr && depthIdx >= 0) {
          const rawRef = dayData[(7 + depthIdx) * nOcean + oceanIdx];
          const rawSst = dayData[0 * nOcean + oceanIdx];
          const rawSss = dayData[1 * nOcean + oceanIdx];
          const rawSsh = dayData[2 * nOcean + oceanIdx];
          const rawCurU = dayData[3 * nOcean + oceanIdx];
          const rawCurV = dayData[4 * nOcean + oceanIdx];
          const rawWindU = dayData[5 * nOcean + oceanIdx];
          const rawWindV = dayData[6 * nOcean + oceanIdx];

          if (rawRef !== -32768 && rawSst !== -32768 && rawSss !== -32768 && rawSsh !== -32768 &&
              rawCurU !== -32768 && rawCurV !== -32768 && rawWindU !== -32768 && rawWindV !== -32768) {
            const ref = rawRef / 100.0;
            const f0 = (rawSst / 100.0 - scaler.mean[0]) / scaler.std[0];
            const f1 = (rawSss / 100.0 - scaler.mean[1]) / scaler.std[1];
            const f2 = (rawSsh / 1000.0 - scaler.mean[2]) / scaler.std[2];
            const f3 = (rawCurU / 1000.0 - scaler.mean[3]) / scaler.std[3];
            const f4 = (rawCurV / 1000.0 - scaler.mean[4]) / scaler.std[4];
            const f5 = (rawWindU / 100.0 - scaler.mean[5]) / scaler.std[5];
            const f6 = (rawWindV / 100.0 - scaler.mean[6]) / scaler.std[6];

            const pred = interceptVal +
              f0 * weightsArr[0] + f1 * weightsArr[1] + f2 * weightsArr[2] +
              f3 * weightsArr[3] + f4 * weightsArr[4] + f5 * weightsArr[5] + f6 * weightsArr[6];

            val = Math.abs(pred - ref);
          }
        }
      }

      if (val === null || isNaN(val)) {
        data[pixelIdx + 3] = 0; // Transparent for missing data
        continue;
      }

      // Map value to colormap
      const t = (val - minVal) / range;
      const [r, g, b] = interpolateColormap(colormap, t);

      data[pixelIdx + 0] = r;
      data[pixelIdx + 1] = g;
      data[pixelIdx + 2] = b;
      data[pixelIdx + 3] = alphaByte;
    }
  }

  ctx.putImageData(imgData, 0, 0);
}

/**
 * Extracts vector arrows for currents or winds at a sampled spatial step
 */
export function extractVectorArrows(
  manifest: OceanManifest,
  dayData: Int16Array,
  layerId: 'currents' | 'winds',
  stepDeg: number = 1.5
): VectorSample[] {
  const { domain, gridToOcean } = manifest;
  const nOcean = domain.nOcean;
  const stepIdx = Math.max(1, Math.round(stepDeg / domain.latStep)); // e.g. 1.5° / 0.25° = 6 cells

  const samples: VectorSample[] = [];

  const uOffset = layerId === 'currents' ? 3 * nOcean : 5 * nOcean;
  const vOffset = layerId === 'currents' ? 4 * nOcean : 6 * nOcean;
  const scale = layerId === 'currents' ? 1000.0 : 100.0;

  for (let y = 2; y < domain.nLat; y += stepIdx) {
    const lat = domain.latMin + y * domain.latStep;
    for (let x = 2; x < domain.nLon; x += stepIdx) {
      const lon = domain.lonMin + x * domain.lonStep;
      const oceanIdx = gridToOcean[y][x];
      if (oceanIdx === -1) continue;

      const rawU = dayData[uOffset + oceanIdx];
      const rawV = dayData[vOffset + oceanIdx];
      if (rawU === -32768 || rawV === -32768) continue;

      const u = rawU / scale;
      const v = rawV / scale;
      const speed = Math.sqrt(u * u + v * v);

      if (speed < 0.05) continue; // Skip negligible vectors

      // Mathematical angle in degrees (0 = East, 90 = North)
      const angleRad = Math.atan2(v, u);
      const angleDeg = (angleRad * 180) / Math.PI;

      samples.push({
        lat,
        lon,
        u,
        v,
        speed,
        angleDeg,
      });
    }
  }

  return samples;
}
