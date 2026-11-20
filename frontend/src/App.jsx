import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { DashboardPage } from './pages/DashboardPage';
import { FindStationPage } from './pages/FindStationPage';
import { PredictionsPage } from './pages/PredictionsPage';
import { StationAnalysisPage } from './pages/StationAnalysisPage';
import { RecommendationsPage } from './pages/RecommendationsPage';
import { ModelPerformancePage } from './pages/ModelPerformancePage';
import { ProbabilisticAvailabilityPage } from './pages/ProbabilisticAvailabilityPage';
import { AvailabilityCalibrationPage } from './pages/AvailabilityCalibrationPage';
import { getHealth } from './services/api';

export default function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [targetStationId, setTargetStationId] = useState('');
  const [health, setHealth] = useState(null);

  useEffect(() => {
    let mounted = true;
    const check = async () => {
      try {
        const data = await getHealth();
        if (mounted) setHealth(data);
      } catch (e) {
        if (mounted) setHealth({ status: 'offline', error: e.message });
      }
    };
    check();
    const timer = setInterval(check, 45000);
    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  const handleNavigateToStation = (stationId, targetPage = 'stations') => {
    setTargetStationId(stationId);
    setActivePage(targetPage);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleNavigateToPage = (pageId) => {
    setActivePage(pageId);
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  return (
    <div className="flex h-screen overflow-hidden bg-slate-100">
      <Sidebar
        activePage={activePage}
        onSelectPage={(page) => {
          setActivePage(page);
          window.scrollTo({ top: 0, behavior: 'smooth' });
        }}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <Navbar
          activePage={activePage}
          health={health}
          onMenuClick={() => setSidebarOpen(true)}
        />

        <main className="flex-1 overflow-y-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="max-w-7xl mx-auto pb-12">
            {activePage === 'dashboard' && (
              <DashboardPage
                onNavigateToStation={handleNavigateToStation}
                onNavigateToPage={handleNavigateToPage}
              />
            )}
            {activePage === 'find-station' && (
              <FindStationPage
                initialStationId={targetStationId}
                onNavigateToStation={handleNavigateToStation}
              />
            )}
            {activePage === 'predictions' && (
              <PredictionsPage initialStationId={targetStationId} />
            )}
            {activePage === 'stations' && (
              <StationAnalysisPage
                initialStationId={targetStationId}
                onOpenForecastForStation={(stId) => handleNavigateToStation(stId, 'predictions')}
              />
            )}
            {activePage === 'recommendations' && (
              <RecommendationsPage
                onNavigateToStationForecast={(stId) => handleNavigateToStation(stId, 'predictions')}
              />
            )}
            {activePage === 'model-performance' && <ModelPerformancePage />}
            {activePage === 'probabilistic-availability' && (
              <ProbabilisticAvailabilityPage
                initialStationId={targetStationId}
                onNavigateToStation={handleNavigateToStation}
              />
            )}
            {activePage === 'availability-calibration' && (
              <AvailabilityCalibrationPage />
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
