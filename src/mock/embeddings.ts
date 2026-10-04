import { EmbeddingPoint } from '../types';

export const EMBEDDING_CLUSTERS = [
  { name: 'Arabian Sea Upwelling', color: '#06b6d4', desc: 'Cold SST, strong winds, shallow thermocline' },
  { name: 'Bay of Bengal River Plume', color: '#10b981', desc: 'Low SSS barrier layer, stable stratification' },
  { name: 'Equatorial Warm Pool Core', color: '#f59e0b', desc: 'High SST, high heat content, deep thermocline' },
  { name: 'Winter Mixed Layer Convection', color: '#8b5cf6', desc: 'Surface cooling, deep vertical mixing' },
  { name: 'Anticyclonic Warm Eddy', color: '#ec4899', desc: 'Positive SSH anomaly, depressed thermocline' },
  { name: 'Cyclonic Cold Eddy', color: '#3b82f6', desc: 'Negative SSH anomaly, doming thermocline' }
];

function pseudoHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) / 2147483647;
}

export function generateMockEmbeddings(count = 120): EmbeddingPoint[] {
  const points: EmbeddingPoint[] = [];

  const clusterCenters = [
    { name: 'Arabian Sea Upwelling', cx: -0.65, cy: 0.45, sst: 25.2, ssh: -0.15, depth: 60 },
    { name: 'Bay of Bengal River Plume', cx: 0.55, cy: 0.55, sst: 29.4, ssh: 0.18, depth: 85 },
    { name: 'Equatorial Warm Pool Core', cx: 0.65, cy: -0.40, sst: 30.1, ssh: 0.22, depth: 140 },
    { name: 'Winter Mixed Layer Convection', cx: -0.30, cy: -0.65, sst: 24.8, ssh: -0.05, depth: 110 },
    { name: 'Anticyclonic Warm Eddy', cx: 0.15, cy: -0.15, sst: 29.1, ssh: 0.28, depth: 165 },
    { name: 'Cyclonic Cold Eddy', cx: -0.25, cy: 0.20, sst: 27.2, ssh: -0.20, depth: 75 }
  ];

  for (let i = 0; i < count; i++) {
    const cluster = clusterCenters[i % clusterCenters.length];
    const seed = `embed_${i}_${cluster.name}`;
    const hx = (pseudoHash(seed + '_x') - 0.5) * 0.32;
    const hy = (pseudoHash(seed + '_y') - 0.5) * 0.32;

    const lat = +(10 + pseudoHash(seed + '_lat') * 16).toFixed(2);
    const lon = +(50 + pseudoHash(seed + '_lon') * 48).toFixed(2);

    points.push({
      id: `pt-${i}`,
      x: +(cluster.cx + hx).toFixed(3),
      y: +(cluster.cy + hy).toFixed(3),
      cluster: cluster.name,
      lat,
      lon,
      sst: +(cluster.sst + (pseudoHash(seed + '_sst') - 0.5) * 1.5).toFixed(1),
      ssh: +(cluster.ssh + (pseudoHash(seed + '_ssh') - 0.5) * 0.08).toFixed(2),
      thermoclineDepth: +(cluster.depth + (pseudoHash(seed + '_d') - 0.5) * 20).toFixed(0),
      date: `2020-12-${String((i % 25) + 1).padStart(2, '0')}`
    });
  }

  return points;
}
