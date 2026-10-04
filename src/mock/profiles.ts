import { DepthProfilePoint, CanonicalDepth } from '../types';
import { CANONICAL_DEPTHS } from './oceanData';

// Historical Climatology Reference Baseline (B1)
const CLIMATOLOGY_MEAN: Record<CanonicalDepth, number> = {
  0: 28.66,
  5: 28.58,
  10: 28.56,
  20: 28.43,
  30: 28.17,
  50: 27.28,
  75: 25.69,
  100: 23.55,
  125: 21.11,
  150: 18.84,
  200: 15.87,
  300: 13.22,
  500: 11.31,
  700: 9.87,
  1000: 7.80
};

// Known Test Error per depth for B8 model
export const B8_TEST_DEPTH_RMSE: Record<CanonicalDepth, number> = {
  0: 0.4369,
  5: 0.4381,
  10: 0.4635,
  20: 0.5962,
  30: 0.8607,
  50: 1.3493,
  75: 1.8110,
  100: 1.7651,
  125: 1.5022,
  150: 1.3650,
  200: 1.1834,
  300: 0.9845,
  500: 0.6870,
  700: 0.6619,
  1000: 0.5959
};

function pseudoHash(str: string): number {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = (hash << 5) - hash + str.charCodeAt(i);
    hash |= 0;
  }
  return Math.abs(hash) / 2147483647;
}

export function getDepthProfile(lat: number, lon: number, dateStr: string): DepthProfilePoint[] {
  const seed = `${lat.toFixed(2)}_${lon.toFixed(2)}_${dateStr}`;
  const h_sst = pseudoHash(seed + '_sst_mod');
  const h_ssh = pseudoHash(seed + '_ssh_mod');

  // Surface anomaly shifts upper 50m
  const surfAnomaly = (h_sst - 0.5) * 2.2;
  // SSH anomaly (warm core / cold core eddy) shifts the thermocline (75m - 200m)
  const eddyThermoclineShift = (h_ssh - 0.5) * 3.5;

  return CANONICAL_DEPTHS.map((depth, idx) => {
    const clim = CLIMATOLOGY_MEAN[depth];
    
    // Physical reference: Depth-decaying surface influence + thermocline anomaly
    let depthFactor = 1.0;
    if (depth <= 30) depthFactor = 1.0 - (depth / 100);
    else if (depth <= 200) depthFactor = 0.4;
    else depthFactor = 0.1;

    let thermoFactor = 0.0;
    if (depth >= 50 && depth <= 200) {
      thermoFactor = Math.sin(((depth - 50) / 150) * Math.PI);
    }

    const ref = +(clim + surfAnomaly * depthFactor + eddyThermoclineShift * thermoFactor).toFixed(2);
    
    // B8 Reconstruction: Very close to reference, with residual scaled to published depth RMSE
    const depthNoise = (pseudoHash(`${seed}_depth_${depth}`) - 0.5) * (B8_TEST_DEPTH_RMSE[depth] * 0.95);
    const reconstructed = +(ref + depthNoise).toFixed(2);
    
    // Ridge baseline (less accurate, smoothed)
    const ridgeNoise = (pseudoHash(`${seed}_ridge_${depth}`) - 0.45) * 1.5;
    const ridge = +(ref + ridgeNoise).toFixed(2);

    const error = +Math.abs(reconstructed - ref).toFixed(2);

    return {
      depth,
      reconstructed,
      reference: ref,
      error,
      climatology: clim,
      ridge
    };
  });
}
