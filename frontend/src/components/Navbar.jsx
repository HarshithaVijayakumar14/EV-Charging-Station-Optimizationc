import React from 'react';
import { Menu, Activity, Database } from 'lucide-react';

const PAGE_TITLES = {
  dashboard: {
    title: 'EV Network Overview',
    subtitle: 'Historical charging utilization, temporal patterns, and station rankings (2011–2020)',
  },
  'find-station': {
    title: 'Find a Station & Availability Assistant',
    subtitle: 'Vehicle-specific battery charging duration, physical port availability forecasting, and smart alternatives',
  },
  predictions: {
    title: 'Demand Forecasting',
    subtitle: 'Leakage-free 1-hour-ahead and 24-hour-ahead XGBoost session arrival predictions',
  },
  stations: {
    title: 'Station Explorer',
    subtitle: 'Granular station telemetry, operating spans, and cold-start site monitoring',
  },
  recommendations: {
    title: 'Optimization & Planning Recommendations',
    subtitle: 'Explainable rule-based operational planning guidance grounded in actual forecasts',
  },
  'model-performance': {
    title: 'Model Training & Evaluation Metrics',
    subtitle: 'Phase 5 chronological validation (2018), main test (2019), and COVID OOD stress test (2020)',
  },
  'probabilistic-availability': {
    title: 'Probabilistic Availability & Queueing Propagation',
    subtitle: 'Calibrated discrete arrival PMFs propagated through Erlang-B queueing with 5–95% uncertainty bounds',
  },
};

export function Navbar({ activePage, health, onMenuClick }) {
  const meta = PAGE_TITLES[activePage] || PAGE_TITLES.dashboard;
  const isHealthy = health?.status === 'healthy';
  const isOffline = health?.status === 'offline';

  return (
    <header className="sticky top-0 z-20 h-16 bg-white/95 backdrop-blur-sm border-b border-slate-200 px-4 sm:px-6 flex items-center justify-between gap-4">
      <div className="flex items-center gap-3 min-w-0">
        <button
          onClick={onMenuClick}
          className="lg:hidden p-2 -ml-2 text-slate-600 hover:bg-slate-100 rounded-lg"
          aria-label="Open menu"
        >
          <Menu className="w-5 h-5" />
        </button>
        <div className="min-w-0">
          <h1 className="text-base sm:text-lg font-bold text-slate-900 truncate">{meta.title}</h1>
          <p className="text-xs text-slate-500 hidden sm:block truncate">{meta.subtitle}</p>
        </div>
      </div>

      <div className="flex items-center gap-2.5 shrink-0">
        <div
          className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border ${
            isHealthy
              ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
              : isOffline
              ? 'bg-rose-50 text-rose-700 border-rose-200'
              : 'bg-amber-50 text-amber-700 border-amber-200'
          }`}
        >
          <Activity className="w-3.5 h-3.5" />
          <span className="hidden md:inline">
            {isHealthy ? 'API & Models Ready' : isOffline ? 'Backend Offline' : 'Connecting to API...'}
          </span>
          <span className="md:hidden">
            {isHealthy ? 'Online' : isOffline ? 'Offline' : 'Connecting'}
          </span>
        </div>

        {health?.database && (
          <div
            className="hidden xl:flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-600 border border-slate-200"
            title="Prediction audit logs saved in SQLite"
          >
            <Database className="w-3.5 h-3.5" />
            <span>{health.database.prediction_history_count} logged forecasts</span>
          </div>
        )}
      </div>
    </header>
  );
}
