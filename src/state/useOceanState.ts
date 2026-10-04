import { useState, useCallback } from 'react';
import { NavigationTab, CanonicalDepth, OceanLocation } from '../types';
import { resolveOceanRegion } from '../mock/oceanData';

export function useOceanState() {
  const [currentTab, setCurrentTab] = useState<NavigationTab>('dashboard');

  const [location, setLocation] = useState<OceanLocation>({
    lat: 15.25,
    lon: 72.50,
    date: '2020-12-15',
    depth: 100,
    region: 'Arabian Sea',
    gridPoint: '15.25°N, 72.50°E'
  });

  const updateCoordinates = useCallback((lat: number, lon: number) => {
    // Snap to 0.25 grid
    const clampedLat = Math.min(30.0, Math.max(5.0, Math.round(lat / 0.25) * 0.25));
    const clampedLon = Math.min(105.0, Math.max(45.0, Math.round(lon / 0.25) * 0.25));
    const region = resolveOceanRegion(clampedLat, clampedLon);
    
    setLocation(prev => ({
      ...prev,
      lat: clampedLat,
      lon: clampedLon,
      region,
      gridPoint: `${clampedLat.toFixed(2)}°N, ${clampedLon.toFixed(2)}°E`
    }));
  }, []);

  const updateDate = useCallback((date: string) => {
    setLocation(prev => ({ ...prev, date }));
  }, []);

  const updateDepth = useCallback((depth: CanonicalDepth) => {
    setLocation(prev => ({ ...prev, depth }));
  }, []);

  const applyPresetLocation = useCallback((name: 'arabian' | 'bob' | 'equatorial' | 'somali') => {
    switch (name) {
      case 'arabian':
        updateCoordinates(15.25, 68.50);
        break;
      case 'bob':
        updateCoordinates(16.50, 88.50);
        break;
      case 'equatorial':
        updateCoordinates(6.50, 78.50);
        break;
      case 'somali':
        updateCoordinates(11.25, 54.50);
        break;
    }
  }, [updateCoordinates]);

  // Reconstruction simulation state
  const [isSimulating, setIsSimulating] = useState(false);
  const [currentStageIndex, setCurrentStageIndex] = useState(0);
  const [simulationComplete, setSimulationComplete] = useState(false);

  const startReconstruction = useCallback(() => {
    setIsSimulating(true);
    setSimulationComplete(false);
    setCurrentStageIndex(0);

    const stagesCount = 8;
    const intervalTime = 420; // ~3.5 seconds total simulation

    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step < stagesCount) {
        setCurrentStageIndex(step);
      } else {
        clearInterval(interval);
        setIsSimulating(false);
        setSimulationComplete(true);
      }
    }, intervalTime);
  }, []);

  return {
    currentTab,
    setCurrentTab,
    location,
    updateCoordinates,
    updateDate,
    updateDepth,
    applyPresetLocation,
    isSimulating,
    currentStageIndex,
    simulationComplete,
    startReconstruction
  };
}

export type OceanStateReturn = ReturnType<typeof useOceanState>;
