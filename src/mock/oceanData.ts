import { SurfaceVariable, CanonicalDepth } from '../types';

export const STUDY_DOMAIN = {
  name: 'North Indian Ocean',
  latMin: 5.0,
  latMax: 30.0,
  lonMin: 45.0,
  lonMax: 105.0,
  latStep: 0.25,
  lonStep: 0.25,
  totalGridPoints: 101 * 241, // 24,341 points
};

export const CANONICAL_DEPTHS: CanonicalDepth[] = [
  0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000
];

export function resolveOceanRegion(lat: number, lon: number): string {
  if (lat < 10) return 'Equatorial Indian Ocean';
  if (lon < 77.5) {
    if (lat > 22 && lon < 65) return 'Gulf of Oman / N. Arabian Sea';
    return 'Arabian Sea';
  }
  if (lon <= 98) {
    if (lat > 18) return 'Northern Bay of Bengal';
    return 'Bay of Bengal';
  }
  return 'Andaman Sea / Eastern Sector';
}

// Pseudo-random deterministic hash generator
function pseudoHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) / 2147483647;
}

export function getSurfaceObservations(lat: number, lon: number, dateStr: string): SurfaceVariable[] {
  const seed = `${lat.toFixed(2)}_${lon.toFixed(2)}_${dateStr}`;
  const h1 = pseudoHash(seed + '_1');
  const h2 = pseudoHash(seed + '_2');
  const h3 = pseudoHash(seed + '_3');
  const h4 = pseudoHash(seed + '_4');

  const region = resolveOceanRegion(lat, lon);

  // Oceanographic realism:
  // Bay of Bengal is fresher (lower SSS) due to major river runoff (Ganges, Brahmaputra, Irrawaddy)
  // Arabian Sea has high evaporation -> higher SSS
  let baseSss = 34.6;
  if (region.includes('Bay of Bengal')) baseSss = 31.8 - (lat - 10) * 0.15;
  else if (region.includes('Arabian')) baseSss = 35.8 + (lat - 10) * 0.08;

  // SST: Cooler upwelling in Western Arabian Sea and winter northern latitudes
  let baseSst = 28.5;
  if (lat > 20) baseSst -= (lat - 20) * 0.45;
  if (lon < 60 && lat > 10) baseSst -= 1.8; // Western Arabian coastal upwelling effect

  const sstVal = +(baseSst + (h1 - 0.5) * 1.6).toFixed(2);
  const sssVal = +(baseSss + (h2 - 0.5) * 0.8).toFixed(2);
  const sshVal = +((h3 - 0.5) * 0.35 + (region.includes('Bay') ? 0.06 : -0.02)).toFixed(2);
  const curU = +(((h4 - 0.45) * 0.6)).toFixed(2);
  const curV = +(((h1 - 0.55) * 0.55)).toFixed(2);
  const windU = +(1.5 + (h2 - 0.5) * 6.5).toFixed(1);
  const windV = +(0.8 + (h3 - 0.5) * 7.0).toFixed(1);

  const makeTrend = (center: number, spread: number) => {
    return Array.from({ length: 7 }, (_, i) => {
      const dayFactor = Math.sin((i / 6) * Math.PI * 1.5) * (spread * 0.5);
      const noise = (pseudoHash(`${seed}_t_${i}`) - 0.5) * (spread * 0.5);
      return +(center + dayFactor + noise).toFixed(2);
    });
  };

  return [
    {
      id: 'sst',
      name: 'Sea Surface Temperature',
      symbol: 'SST',
      value: sstVal,
      unit: '°C',
      trend: makeTrend(sstVal, 1.2),
      status: sstVal > 29.0 ? 'elevated' : sstVal < 26.0 ? 'depressed' : 'normal',
      description: 'Thermal boundary state driving air-sea heat exchange and stratification'
    },
    {
      id: 'sss',
      name: 'Sea Surface Salinity',
      symbol: 'SSS',
      value: sssVal,
      unit: 'PSU',
      trend: makeTrend(sssVal, 0.4),
      status: sssVal < 32.5 ? 'depressed' : sssVal > 36.0 ? 'elevated' : 'normal',
      description: 'Halocline tracer governing density barrier layers in BoB and Arabian evaporation'
    },
    {
      id: 'ssh',
      name: 'Sea Surface Height',
      symbol: 'SSH',
      value: sshVal,
      unit: 'm',
      trend: makeTrend(sshVal, 0.08),
      status: sshVal > 0.15 ? 'elevated' : sshVal < -0.1 ? 'depressed' : 'normal',
      description: 'Baroclinic dynamic topography directly correlated with thermocline displacement'
    },
    {
      id: 'current_u',
      name: 'Zonal Ocean Current',
      symbol: 'Current U',
      value: curU,
      unit: 'm/s',
      trend: makeTrend(curU, 0.15),
      status: 'normal',
      description: 'East-west geostrophic current vector resolving equatorial jet dynamics'
    },
    {
      id: 'current_v',
      name: 'Meridional Ocean Current',
      symbol: 'Current V',
      value: curV,
      unit: 'm/s',
      trend: makeTrend(curV, 0.15),
      status: 'normal',
      description: 'North-south cross-equatorial advection and western boundary current jets'
    },
    {
      id: 'wind_u',
      name: 'Zonal Wind Stress',
      symbol: 'Wind U',
      value: windU,
      unit: 'm/s',
      trend: makeTrend(windU, 2.5),
      status: Math.abs(windU) > 6.0 ? 'elevated' : 'normal',
      description: 'Atmospheric trade wind forcing driving Ekman pumping and divergence'
    },
    {
      id: 'wind_v',
      name: 'Meridional Wind Stress',
      symbol: 'Wind V',
      value: windV,
      unit: 'm/s',
      trend: makeTrend(windV, 2.5),
      status: Math.abs(windV) > 6.0 ? 'elevated' : 'normal',
      description: 'Monsoon cross-equatorial atmospheric jet driving coastal upwelling'
    }
  ];
}
