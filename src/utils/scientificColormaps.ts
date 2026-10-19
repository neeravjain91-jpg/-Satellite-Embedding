import React from 'react';

// Colormaps for scientific oceanography
export interface RGB {
  r: number;
  g: number;
  b: number;
}

// Multi-stop piecewise linear interpolation
export function interpolateColormap(stops: { val: number; r: number; g: number; b: number }[], t: number): [number, number, number] {
  const clamped = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i++) {
    const s1 = stops[i];
    const s2 = stops[i + 1];
    if (clamped >= s1.val && clamped <= s2.val) {
      const f = (clamped - s1.val) / (s2.val - s1.val || 1);
      const r = Math.round(s1.r + f * (s2.r - s1.r));
      const g = Math.round(s1.g + f * (s2.g - s1.g));
      const b = Math.round(s1.b + f * (s2.b - s1.b));
      return [r, g, b];
    }
  }
  const last = stops[stops.length - 1];
  return [last.r, last.g, last.b];
}

// cmocean thermal (for potential temperature °C)
export const COLORMAP_THERMAL = [
  { val: 0.00, r: 4,   g: 3,   b: 15 },
  { val: 0.15, r: 24,  g: 40,  b: 110 },
  { val: 0.35, r: 0,   g: 130, b: 180 },
  { val: 0.55, r: 60,  g: 190, b: 120 },
  { val: 0.70, r: 230, g: 200, b: 50 },
  { val: 0.85, r: 235, g: 110, b: 30 },
  { val: 1.00, r: 180, g: 10,  b: 30 }
];

// cmocean haline (for salinity PSU)
export const COLORMAP_HALINE = [
  { val: 0.00, r: 20,  g: 10,  b: 60 },
  { val: 0.25, r: 20,  g: 80,  b: 140 },
  { val: 0.50, r: 20,  g: 150, b: 140 },
  { val: 0.75, r: 100, g: 200, b: 100 },
  { val: 1.00, r: 240, g: 240, b: 120 }
];

// cmocean balance (for SSH anomaly m, centered at zero)
export const COLORMAP_BALANCE = [
  { val: 0.00, r: 15,  g: 65,  b: 170 },
  { val: 0.30, r: 85,  g: 160, b: 230 },
  { val: 0.50, r: 240, g: 240, b: 245 },
  { val: 0.70, r: 240, g: 140, b: 90 },
  { val: 1.00, r: 180, g: 20,  b: 30 }
];

// cmocean amp/matter (for residual error °C)
export const COLORMAP_RESIDUAL = [
  { val: 0.00, r: 245, g: 245, b: 245 },
  { val: 0.20, r: 254, g: 224, b: 139 },
  { val: 0.45, r: 244, g: 109, b: 67 },
  { val: 0.70, r: 213, g: 62,  b: 79 },
  { val: 1.00, r: 110, g: 10,  b: 70 }
];

// cmocean speed (for currents / winds m/s)
export const COLORMAP_SPEED = [
  { val: 0.00, r: 10,  g: 20,  b: 50 },
  { val: 0.25, r: 15,  g: 90,  b: 160 },
  { val: 0.50, r: 0,   g: 180, b: 160 },
  { val: 0.75, r: 140, g: 220, b: 80 },
  { val: 1.00, r: 250, g: 230, b: 50 }
];
