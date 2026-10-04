// High-Precision North Indian Ocean Geography, Bathymetry & Land Detection
// Domain: 5.0°N – 30.0°N, 45.0°E – 105.0°E (Plate Carrée / Equirectangular Projection)

export interface GeoPoint {
  lat: number;
  lon: number;
}

export const MAP_DOMAIN = {
  lonMin: 45.0,
  lonMax: 105.0,
  latMin: 5.0,
  latMax: 30.0,
  width: 1200,
  height: 500,
  pixelsPerDegreeLon: 20.0, // 1200 / 60
  pixelsPerDegreeLat: 20.0, // 500 / 25
};

// Projection conversion functions
export function lonToX(lon: number): number {
  return ((lon - MAP_DOMAIN.lonMin) / (MAP_DOMAIN.lonMax - MAP_DOMAIN.lonMin)) * MAP_DOMAIN.width;
}

export function latToY(lat: number): number {
  return ((MAP_DOMAIN.latMax - lat) / (MAP_DOMAIN.latMax - MAP_DOMAIN.latMin)) * MAP_DOMAIN.height;
}

export function xToLon(x: number): number {
  return MAP_DOMAIN.lonMin + (x / MAP_DOMAIN.width) * (MAP_DOMAIN.lonMax - MAP_DOMAIN.lonMin);
}

export function yToLat(y: number): number {
  return MAP_DOMAIN.latMax - (y / MAP_DOMAIN.height) * (MAP_DOMAIN.latMax - MAP_DOMAIN.latMin);
}

// Convert an array of [lon, lat] pairs into an SVG path 'd' string
export function coordsToSvgPath(coords: [number, number][], close = true): string {
  if (coords.length === 0) return '';
  const points = coords.map(([lon, lat]) => `${lonToX(lon).toFixed(1)},${latToY(lat).toFixed(1)}`);
  return `M ${points.join(' L ')} ${close ? 'Z' : ''}`;
}

// ============================================================================
// CONTINENTAL LANDMASSES & ISLANDS (NORTH INDIAN OCEAN BASIN)
// ============================================================================

// 1. Indian Subcontinent (Mainland Peninsula & Northern Landmass)
export const INDIA_COASTLINE: [number, number][] = [
  // West Coast from Kutch to Kanyakumari
  [68.16, 23.65], // Kori Creek / Sir Creek
  [69.0, 23.0],   // Gulf of Kutch North
  [70.1, 23.0],
  [70.5, 22.9],   // Kandla
  [70.0, 22.5],   // Jamnagar
  [69.05, 22.25], // Dwarka / Okha
  [69.6, 21.6],   // Porbandar
  [70.4, 20.9],   // Veraval / Somnath
  [70.9, 20.7],   // Diu
  [71.4, 20.85],  // Jafrabad
  [71.8, 21.05],  // Mahuva
  [72.2, 21.75],  // Bhavnagar
  [72.6, 22.3],   // Head of Gulf of Khambhat
  [72.6, 21.6],   // Bharuch (Narmada)
  [72.7, 21.1],   // Surat (Tapi)
  [72.8, 20.4],   // Daman / Vapi
  [72.7, 20.0],   // Dahanu
  [72.82, 18.9],  // Mumbai (Colaba)
  [72.87, 18.65], // Alibaug
  [72.95, 18.3],  // Murud
  [73.28, 17.0],  // Ratnagiri
  [73.33, 16.55], // Vijaydurg
  [73.47, 16.05], // Malvan
  [73.8, 15.4],   // Goa (Mormugao)
  [74.12, 14.8],  // Karwar
  [74.4, 14.4],   // Gokarna
  [74.55, 13.98], // Bhatkal
  [74.7, 13.35],  // Malpe / Udupi
  [74.83, 12.87], // Mangalore
  [74.98, 12.5],  // Kasaragod
  [75.36, 11.87], // Kannur
  [75.77, 11.25], // Kozhikode (Calicut)
  [75.92, 10.77], // Ponnani
  [76.26, 9.93],  // Kochi
  [76.33, 9.49],  // Alappuzha
  [76.58, 8.89],  // Kollam
  [76.95, 8.48],  // Thiruvananthapuram
  [77.55, 8.08],  // Kanyakumari (Cape Comorin)

  // East Coast from Kanyakumari to Sundarbans
  [77.7, 8.18],   // Kudankulam
  [78.12, 8.49],  // Tiruchendur
  [78.16, 8.76],  // Thoothukudi (Tuticorin)
  [79.3, 9.28],   // Mandapam / Rameswaram
  [79.0, 9.75],   // Palk Strait (Tondi)
  [79.85, 10.3],  // Point Calimere
  [79.85, 10.76], // Nagapattinam
  [79.84, 10.92], // Karaikal
  [79.77, 11.75], // Cuddalore
  [79.83, 11.93], // Puducherry
  [80.19, 12.62], // Mahabalipuram
  [80.28, 13.08], // Chennai
  [80.32, 13.45], // Pulicat Lake
  [80.15, 14.45], // Nellore
  [80.05, 15.5],  // Ongole
  [81.19, 16.18], // Machilipatnam (Krishna Delta)
  [82.25, 16.95], // Kakinada (Godavari Delta)
  [83.3, 17.69],  // Visakhapatnam
  [84.12, 18.34], // Kalingapatnam
  [84.9, 19.26],  // Gopalpur
  [85.35, 19.68], // Chilika Lake
  [85.83, 19.8],  // Puri
  [86.67, 20.32], // Paradip (Mahanadi Delta)
  [87.02, 21.47], // Balasore / Chandipur
  [87.52, 21.62], // Digha
  [88.18, 21.65], // Sagar Island / Hooghly Estuary
  [89.15, 21.65], // Sundarbans (Raimangal Border)

  // Continental Hinterland (Northward to 30°N)
  [89.15, 24.5],  // West Bengal interior
  [88.2, 26.5],   // Siliguri corridor
  [88.0, 30.0],   // Northern Boundary
  [71.5, 30.0],   // Rajasthan/Punjab Northern Boundary
  [70.5, 27.5],   // Thar Desert
  [68.5, 24.5],   // Northern Rann of Kutch
];

// 2. Sri Lanka (Teardrop Island)
export const SRI_LANKA_COASTLINE: [number, number][] = [
  [80.24, 9.82], // Point Pedro
  [80.0, 9.66],  // Jaffna
  [79.85, 9.0],  // Mannar
  [79.75, 8.23], // Kalpitiya
  [79.84, 7.21], // Negombo
  [79.86, 6.93], // Colombo
  [79.96, 6.58], // Kalutara
  [80.22, 6.03], // Galle
  [80.59, 5.92], // Dondra Head (Southern Tip)
  [81.12, 6.12], // Hambantota
  [81.83, 6.84], // Arugam Bay
  [81.7, 7.71],  // Batticaloa
  [81.23, 8.58], // Trincomalee
  [80.81, 9.27], // Mullaitivu
];

// 3. Pakistan & Makran Coast
export const PAKISTAN_COASTLINE: [number, number][] = [
  [68.16, 23.65], // Sir Creek
  [67.9, 23.8],   // Shah Bandar
  [67.45, 24.15], // Keti Bandar (Indus Delta)
  [67.0, 24.8],   // Karachi
  [66.66, 24.83], // Cape Monze
  [66.5, 25.15],  // Sonmiani Bay
  [65.8, 25.4],   // Hingol
  [64.63, 25.2],  // Ormara
  [63.48, 25.26], // Pasni
  [62.33, 25.12], // Gwadar
  [61.6, 25.18],  // Gwatar Bay (Iran Border)
  // Interior to 30°N
  [61.6, 30.0],   // Iran/Balochistan North
  [68.16, 30.0],  // Indus Valley North
];

// 4. Arabian Peninsula & Persian Gulf Coast (Oman, Yemen, UAE, Saudi Arabia, Iran North)
export const ARABIA_COASTLINE: [number, number][] = [
  [45.0, 12.8],   // Aden (Southern boundary at 45°E)
  [45.7, 13.35],  // Shuqrah
  [48.33, 14.02], // Bir Ali
  [49.12, 14.54], // Mukalla
  [51.25, 15.21], // Sayhut
  [51.67, 15.42], // Qishn
  [52.25, 15.65], // Ras Fartak
  [53.05, 16.65], // Yemen/Oman Border (Hawf)
  [53.95, 16.93], // Raysut
  [54.09, 17.02], // Salalah (Dhofar)
  [54.7, 16.98],  // Mirbat
  [56.55, 18.15], // Ras Sawqirah
  [57.83, 19.0],  // Ras Madrakah
  [58.2, 20.4],   // Gulf of Masirah
  [58.9, 20.65],  // Ras Hilf
  [59.55, 21.84], // Al Ashkharah
  [59.8, 22.54],  // Ras al Hadd
  [58.59, 23.61], // Muscat
  [56.75, 24.36], // Sohar
  [56.35, 25.12], // Fujairah
  [56.45, 26.2],  // Musandam Peninsula (Strait of Hormuz)
  [55.3, 25.3],   // Dubai / UAE
  [54.3, 24.5],   // Abu Dhabi
  [51.5, 25.3],   // Qatar
  [50.1, 26.5],   // Dammam / Bahrain Coast
  [48.5, 29.5],   // Kuwait
  [48.0, 30.0],   // Shatt al-Arab
  [45.0, 30.0],   // NW Corner
];

// 5. Iran Coastline (Makran & Persian Gulf North)
export const IRAN_COASTLINE: [number, number][] = [
  [61.6, 25.18],  // Pakistan Border (Gwatar Bay)
  [60.6, 25.3],   // Chah Bahar
  [57.77, 25.64], // Jask
  [56.3, 27.15],  // Strait of Hormuz (Bandar Abbas)
  [54.88, 26.56], // Bandar Lengeh
  [50.84, 28.98], // Bushehr
  [48.5, 30.0],   // Abadan
  [61.6, 30.0],   // Northeast border
];

// 6. Horn of Africa (Somalia & Puntland)
export const SOMALIA_COASTLINE: [number, number][] = [
  [45.0, 10.8],   // Gulf of Aden entrance at 45°E
  [47.1, 11.0],   // Maydh
  [49.18, 11.28], // Bosaso
  [50.75, 11.97], // Alula
  [51.27, 11.83], // Cape Guardafui (Ras Asir)
  [51.08, 11.28], // Bargal
  [51.41, 10.43], // Ras Hafun (Easternmost Africa)
  [50.8, 9.49],   // Bandarbeyla
  [49.3, 7.0],    // Garacad
  [48.53, 5.35],  // Hobyo
  [48.1, 5.0],    // Southern boundary at 5°N
  [45.0, 5.0],    // SW Corner
  [45.0, 10.8],   // Return to Aden entrance
];

// 7. Socotra Island & Abd al Kuri
export const SOCOTRA_ISLAND: [number, number][] = [
  [53.3, 12.45],
  [53.6, 12.65],
  [54.1, 12.68],
  [54.5, 12.55],
  [54.4, 12.35],
  [53.8, 12.3],
  [53.4, 12.38],
];

export const ABD_AL_KURI: [number, number][] = [
  [52.1, 12.15],
  [52.4, 12.22],
  [52.3, 12.1],
];

// 8. Bangladesh & Eastern Bay Coastline
export const BANGLADESH_COASTLINE: [number, number][] = [
  [89.15, 21.65], // Raimangal River (Sundarbans West)
  [89.6, 21.75],  // Mongla / Pussur
  [90.12, 21.82], // Kuakata
  [90.7, 22.1],   // Bhola Island / Meghna Estuary
  [91.4, 22.4],   // Sandwip / Hatiya
  [91.8, 22.3],   // Chittagong
  [91.98, 21.43], // Cox's Bazar
  [92.35, 20.86], // Teknaf (Naf River)
  // Interior to North
  [92.35, 25.0],  // Chittagong Hill Tracts
  [89.15, 25.0],  // Northern Plain
];

// 9. Myanmar, Thailand & Malay Peninsula
export const SE_ASIA_COASTLINE: [number, number][] = [
  [92.35, 20.86], // Teknaf / Naf River
  [92.9, 20.14],  // Sittwe (Akyab)
  [93.55, 19.42], // Kyaukpyu (Ramree Island)
  [94.3, 18.46],  // Thandwe
  [94.58, 17.58], // Gwa
  [94.19, 16.03], // Cape Negrais
  [95.4, 15.8],   // Ayeyarwady Delta (Bogale)
  [96.3, 16.48],  // Yangon River Mouth
  [97.6, 16.48],  // Gulf of Martaban / Mawlamyine
  [97.85, 15.25], // Ye
  [98.2, 14.08],  // Dawei
  [98.6, 12.45],  // Myeik (Mergui)
  [98.55, 9.98],  // Kawthaung / Ranong
  [98.24, 8.86],  // Khao Lak
  [98.3, 7.95],   // Phuket Island
  [99.0, 7.5],    // Krabi
  [99.8, 6.35],   // Langkawi
  [100.25, 5.4],  // Penang
  [100.5, 5.0],   // Malacca Strait boundary at 5°N
  // Eastern & Northern Inland
  [105.0, 5.0],   // SE Corner
  [105.0, 30.0],  // NE Corner
  [92.35, 30.0],  // North Myanmar
];

// 10. Northern Tip of Sumatra (Indonesia)
export const SUMATRA_TIP: [number, number][] = [
  [95.2, 5.4],
  [95.32, 5.56],  // Banda Aceh
  [95.9, 5.3],
  [97.15, 5.25],  // Lhokseumawe
  [98.0, 5.0],    // Edge at 5°N
  [95.0, 5.0],    // SW Edge
];

// 11. Andaman & Nicobar Archipelago (Detailed Island Chains)
export const ANDAMAN_ISLANDS: [number, number][][] = [
  // North Andaman
  [
    [92.9, 13.55],
    [93.08, 13.6],
    [93.1, 13.1],
    [92.92, 13.05],
  ],
  // Middle Andaman
  [
    [92.85, 12.85],
    [93.0, 12.8],
    [92.95, 12.35],
    [92.75, 12.4],
  ],
  // South Andaman (Port Blair)
  [
    [92.68, 12.1],
    [92.8, 12.0],
    [92.78, 11.5],
    [92.65, 11.55],
  ],
  // Little Andaman
  [
    [92.45, 10.85],
    [92.62, 10.82],
    [92.58, 10.55],
    [92.42, 10.6],
  ],
  // Car Nicobar
  [
    [92.72, 9.25],
    [92.85, 9.22],
    [92.82, 9.1],
    [92.7, 9.12],
  ],
  // Central Nicobar (Camorta & Nancowry)
  [
    [93.45, 8.12],
    [93.58, 8.08],
    [93.52, 7.9],
    [93.38, 7.95],
  ],
  // Great Nicobar (Indira Point region)
  [
    [93.75, 7.22],
    [93.92, 7.15],
    [93.88, 6.75],
    [93.72, 6.8],
  ],
];

// 12. Lakshadweep Archipelago (Coral Atolls)
export const LAKSHADWEEP_ATOLLS: [number, number][] = [
  [72.7, 11.68],  // Chetlat
  [72.18, 11.6],  // Bitra
  [73.0, 11.48],  // Kiltan
  [72.78, 11.23], // Kadmat
  [72.73, 11.12], // Amini
  [72.18, 10.85], // Agatti
  [72.64, 10.56], // Kavaratti (Capital)
  [73.68, 10.82], // Andrott
  [73.65, 10.08], // Kalpeni
  [72.28, 10.08], // Suheli
  [73.05, 8.28],  // Minicoy (Nine Degree Channel)
];

// 13. Maldives Atolls (Northern Atolls >= 5°N)
export const MALDIVES_ATOLLS: [number, number][] = [
  [72.95, 7.05],  // Ihavandhippolhu Atoll
  [73.05, 6.6],   // Haa Dhaalu
  [73.0, 6.2],    // Shaviyani Atoll
  [73.25, 5.85],  // Noonu Atoll
  [72.9, 5.65],   // Raa Atoll
  [72.9, 5.25],   // Baa Atoll
  [73.5, 5.4],    // Lhaviyani Atoll
  [73.5, 5.05],   // North Malé Atoll
];

// ============================================================================
// REALISTIC BATHYMETRIC FEATURES (CONTINENTAL SHELVES, BASINS & RIDGES)
// ============================================================================

// Continental Shelf (0 – 200m depth isobath outline)
// In the North Indian Ocean, the shelf is extremely broad in the Gulf of Khambhat/Kutch,
// the Ganges-Brahmaputra estuary (Swatch of No Ground), and the Gulf of Martaban.
export const CONTINENTAL_SHELF_200M: [number, number][] = [
  // West Coast Shelf (60 - 150 km offshore)
  [67.0, 23.5],
  [68.0, 22.5],
  [68.5, 21.0],
  [70.5, 19.5],
  [71.5, 18.5],
  [72.2, 16.5],
  [73.2, 14.5],
  [74.0, 12.5],
  [75.0, 10.5],
  [76.0, 8.5],
  [77.0, 7.2],  // South of Kanyakumari
  // Around Sri Lanka / Palk Bay shelf
  [79.0, 6.5],
  [81.5, 5.5],
  [82.5, 7.5],
  [82.0, 9.5],
  // East Coast Shelf
  [80.8, 12.5],
  [81.5, 14.5],
  [82.8, 16.0],
  [84.0, 17.5],
  [85.5, 19.0],
  [87.0, 20.0],  // Broad Ganges Shelf
  [89.5, 20.5],  // Swatch of No Ground indent
  [91.5, 20.8],
  // Myanmar Shelf
  [92.5, 19.5],
  [93.8, 16.0],
  [95.5, 14.5],  // Broad Martaban Shelf
  [97.0, 14.0],
  [98.0, 10.0],
  [98.2, 6.0],
];

// Deep Sea Ridges & Features
export const OCEAN_RIDGES = [
  {
    name: 'Carlsberg Ridge',
    type: 'Spreading Mid-Ocean Ridge',
    coords: [
      [57.5, 14.5],
      [60.0, 12.0],
      [63.0, 8.5],
      [66.0, 5.0],
    ] as [number, number][],
    labelPos: [61.0, 10.8] as [number, number],
  },
  {
    name: 'Chagos-Laccadive Ridge',
    type: 'Volcanic Aseismic Ridge',
    coords: [
      [72.5, 13.5],
      [72.8, 10.5],
      [73.0, 7.5],
      [73.2, 5.0],
    ] as [number, number][],
    labelPos: [71.0, 9.2] as [number, number],
  },
  {
    name: 'Ninety East Ridge',
    type: 'Linear Submarine Ridge (90°E)',
    coords: [
      [90.0, 10.5],
      [90.0, 8.0],
      [90.0, 5.0],
    ] as [number, number][],
    labelPos: [90.8, 7.8] as [number, number],
  },
  {
    name: 'Murray Ridge',
    type: 'Tectonic Ridge & Trench',
    coords: [
      [62.5, 24.0],
      [64.5, 22.5],
      [66.0, 21.0],
    ] as [number, number][],
    labelPos: [65.0, 23.2] as [number, number],
  },
  {
    name: 'Sunda / Andaman Trench',
    type: 'Subduction Trench (>4000m)',
    coords: [
      [93.2, 14.0],
      [92.8, 11.5],
      [93.0, 8.5],
      [94.0, 5.5],
    ] as [number, number][],
    labelPos: [94.5, 9.5] as [number, number],
  },
];

// Major Ocean Basin Label Placements
export const OCEAN_BASINS = [
  { name: 'ARABIAN SEA', lat: 16.5, lon: 65.5, size: 16, weight: 'bold' },
  { name: 'BAY OF BENGAL', lat: 15.5, lon: 88.0, size: 16, weight: 'bold' },
  { name: 'EQUATORIAL INDIAN OCEAN', lat: 6.2, lon: 77.0, size: 13, weight: '600' },
  { name: 'GULF OF OMAN', lat: 24.5, lon: 58.5, size: 10, weight: '500' },
  { name: 'GULF OF ADEN', lat: 12.8, lon: 47.5, size: 10, weight: '500' },
  { name: 'SOMALI BASIN', lat: 8.5, lon: 52.5, size: 11, weight: '500' },
  { name: 'LAKSHADWEEP SEA', lat: 10.0, lon: 75.0, size: 10, weight: '500' },
  { name: 'ANDAMAN SEA', lat: 11.5, lon: 95.5, size: 11, weight: '500' },
  { name: 'INDUS DEEP-SEA FAN', lat: 20.5, lon: 64.5, size: 9, weight: '400' },
  { name: 'BENGAL DEEP-SEA FAN', lat: 11.0, lon: 85.0, size: 9, weight: '400' },
];

// Oceanographic Monsoon Surface Currents (Circulation Vectors)
export const MONSOON_CURRENTS = [
  // Somali Current / Findlater Jet
  { id: 'somali-1', path: 'M 70,420 Q 95,350 140,290', name: 'Somali Current (Intense Western Boundary Jet)' },
  { id: 'somali-2', path: 'M 140,290 Q 180,240 230,220', name: 'Great Whirl Off-Shore Retroflection' },
  // West India Coastal Current
  { id: 'wicc', path: 'M 565,220 Q 575,290 615,380', name: 'West India Coastal Current (WICC)' },
  // East India Coastal Current
  { id: 'eicc', path: 'M 705,330 Q 725,270 810,210', name: 'East India Coastal Current (EICC)' },
  // Southwest Monsoon Current / Equatorial Flow
  { id: 'smc', path: 'M 250,380 Q 450,420 700,430', name: 'Monsoon Current (Cross-Basin Jet)' },
  // Bay of Bengal Gyre
  { id: 'bob-gyre', path: 'M 820,280 Q 880,230 850,330', name: 'Bay of Bengal Cyclonic / Anticyclonic Gyre' },
  // Wyrtki Jet along Equator
  { id: 'wyrtki', path: 'M 200,470 Q 600,475 1050,470', name: 'Equatorial Wyrtki Jet' },
];

// ============================================================================
// GEBCO DEPTH ESTIMATION & LAND DETECTION
// ============================================================================

// Point-in-polygon algorithm (Ray casting)
function isPointInPolygon(point: [number, number], polygon: [number, number][]): boolean {
  const [x, y] = point;
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    const intersect = ((yi > y) !== (yj > y)) && (x < ((xj - xi) * (y - yi)) / (yj - yi) + xi);
    if (intersect) inside = !inside;
  }
  return inside;
}

export interface LandCheckResult {
  isLand: boolean;
  name?: string;
}

export function checkLandLocation(lat: number, lon: number): LandCheckResult {
  const pt: [number, number] = [lon, lat];

  // 1. Sri Lanka check
  if (isPointInPolygon(pt, SRI_LANKA_COASTLINE)) {
    return { isLand: true, name: 'Sri Lanka' };
  }

  // 2. Indian Subcontinent check
  if (isPointInPolygon(pt, INDIA_COASTLINE)) {
    return { isLand: true, name: 'Indian Subcontinent' };
  }

  // 3. Pakistan check
  if (isPointInPolygon(pt, PAKISTAN_COASTLINE)) {
    return { isLand: true, name: 'Pakistan / Makran Coast' };
  }

  // 4. Arabian Peninsula check
  if (isPointInPolygon(pt, ARABIA_COASTLINE)) {
    return { isLand: true, name: 'Arabian Peninsula' };
  }

  // 5. Iran check
  if (isPointInPolygon(pt, IRAN_COASTLINE)) {
    return { isLand: true, name: 'Iran / Persian Gulf Coast' };
  }

  // 6. Horn of Africa check
  if (isPointInPolygon(pt, SOMALIA_COASTLINE)) {
    return { isLand: true, name: 'Horn of Africa (Somalia)' };
  }

  // 7. Bangladesh check
  if (isPointInPolygon(pt, BANGLADESH_COASTLINE)) {
    return { isLand: true, name: 'Bangladesh Mainland' };
  }

  // 8. Southeast Asia / Myanmar / Malay Peninsula check
  if (isPointInPolygon(pt, SE_ASIA_COASTLINE)) {
    return { isLand: true, name: 'Myanmar / Southeast Asia' };
  }

  // 9. Northern Sumatra check
  if (isPointInPolygon(pt, SUMATRA_TIP)) {
    return { isLand: true, name: 'Northern Sumatra (Indonesia)' };
  }

  // Additional bounding checks for inland / high-latitude blocks
  if (lat > 24.5 && lon > 68.0 && lon < 88.0) {
    return { isLand: true, name: 'Northern India Mainland' };
  }
  if (lat > 22.0 && lon > 91.5) {
    return { isLand: true, name: 'Myanmar / Indochina' };
  }
  if (lat > 25.0 && lon < 60.0) {
    return { isLand: true, name: 'Arabian Peninsula' };
  }

  return { isLand: false };
}

// Realistic GEBCO-consistent ocean bathymetric depth calculator (in meters)
export function getEstimatedOceanDepth(lat: number, lon: number): number {
  // Check if near continental shelf
  // Arabian Sea deep basin: ~3,500m - 4,400m
  // Bay of Bengal deep basin: ~2,800m - 3,800m
  // Carlsberg ridge: ~1,800m - 2,500m
  // Chagos-Laccadive ridge: ~1,200m - 2,100m
  // Ninety East ridge: ~2,000m - 2,400m
  // Sunda trench: ~4,500m - 5,600m

  // Near coasts:
  if (lat > 20.5 && lon > 69.0 && lon < 73.0) return 65; // Gulf of Khambhat / Kutch
  if (lat > 21.0 && lon > 87.0 && lon < 91.0) return 85; // Ganges delta shelf
  if (lat > 15.0 && lon > 95.0 && lon < 98.0) return 55; // Gulf of Martaban shelf
  if (lat < 10.0 && lon > 79.0 && lon < 80.2) return 40; // Palk Bay / Adam's Bridge

  // Ridges
  const distNinetyEast = Math.abs(lon - 90.0);
  if (distNinetyEast < 0.8 && lat < 12.0) return 2150;

  const distLaccadive = Math.abs(lon - 73.0);
  if (distLaccadive < 1.0 && lat < 14.0) return 1680;

  // Sunda trench
  if (lon > 92.5 && lon < 94.0 && lat < 11.0) return 4850;

  // Deep Arabian Sea
  if (lon < 72.0 && lat < 20.0 && lat > 10.0) return 3620;

  // Deep Bay of Bengal
  if (lon > 83.0 && lon < 92.0 && lat < 18.0 && lat > 8.0) return 3350;

  // Equatorial Indian Ocean
  if (lat < 8.0) return 4100;

  // General default depth
  return 2850;
}
