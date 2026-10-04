import React from 'react';
import { useOceanState } from './state/useOceanState';
import { Sidebar } from './components/layout/Sidebar';
import { Header } from './components/layout/Header';

import { DashboardView } from './components/views/DashboardView';
import { OceanExplorerView } from './components/views/OceanExplorerView';
import { ReconstructionView } from './components/views/ReconstructionView';
import { DepthProfileView } from './components/views/DepthProfileView';
import { EmbeddingView } from './components/views/EmbeddingView';
import { BenchmarksView } from './components/views/BenchmarksView';
import { ValidationView } from './components/views/ValidationView';
import { MethodologyView } from './components/views/MethodologyView';

export function App() {
  const oceanState = useOceanState();
  const { currentTab, setCurrentTab, location, updateDepth, updateDate } = oceanState;

  const renderActiveView = () => {
    switch (currentTab) {
      case 'dashboard':
        return <DashboardView oceanState={oceanState} />;
      case 'explorer':
        return <OceanExplorerView oceanState={oceanState} />;
      case 'reconstruction':
        return <ReconstructionView oceanState={oceanState} />;
      case 'depth-profile':
        return <DepthProfileView oceanState={oceanState} />;
      case 'embedding':
        return <EmbeddingView oceanState={oceanState} />;
      case 'benchmarks':
        return <BenchmarksView />;
      case 'validation':
        return <ValidationView />;
      case 'methodology':
        return <MethodologyView />;
      default:
        return <DashboardView oceanState={oceanState} />;
    }
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-[#05080e] text-slate-100 font-sans antialiased">
      {/* Persistent Left Sidebar */}
      <Sidebar
        currentTab={currentTab}
        onSelectTab={setCurrentTab}
        location={location}
      />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col h-full overflow-hidden">
        {/* Top Header Bar */}
        <Header
          currentTab={currentTab}
          location={location}
          onUpdateDepth={updateDepth}
          onUpdateDate={updateDate}
        />

        {/* Scrollable View Container */}
        <main className="flex-1 overflow-y-auto p-6 md:p-8 custom-scrollbar">
          <div className="max-w-7xl mx-auto pb-12">
            {renderActiveView()}
          </div>
        </main>

        {/* Bottom Technical Status Bar */}
        <footer className="bg-[#070b12] border-t border-slate-800/80 px-6 py-2.5 flex items-center justify-between text-[11px] text-slate-400 select-none flex-shrink-0">
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span className="text-slate-300 font-semibold">Local Prototype Standalone Mode</span>
            </div>
            <span className="text-slate-600">|</span>
            <span>Target: GLORYS Subsurface Temperature (15 Canonical Depths)</span>
          </div>

          <div className="flex items-center gap-4 font-mono text-[10px]">
            <span className="text-slate-500">Domain: 5°N–30°N, 45°E–105°E</span>
            <span className="text-cyan-400">B8 Model Active (0.9800 °C)</span>
          </div>
        </footer>
      </div>
    </div>
  );
}

export default App;
