import React from 'react';
import {
  LayoutDashboard,
  TrendingUp,
  MapPin,
  Lightbulb,
  BarChart3,
  Zap,
  X,
  Compass,
  CheckCircle2,
  Layers,
} from 'lucide-react';

const NAV_ITEMS = [
  { id: 'dashboard', label: 'Network Overview', icon: LayoutDashboard },
  { id: 'find-station', label: 'Find a Station', icon: Compass },
  { id: 'predictions', label: 'Demand Forecasts', icon: TrendingUp },
  { id: 'stations', label: 'Station Explorer', icon: MapPin },
  { id: 'recommendations', label: 'Optimization Plan', icon: Lightbulb },
  { id: 'model-performance', label: 'Model Evaluation', icon: BarChart3 },
  { id: 'probabilistic-availability', label: 'Probabilistic Availability', icon: Layers },
];

export function Sidebar({ activePage, onSelectPage, isOpen, onClose }) {
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-slate-900/50 backdrop-blur-xs z-30 lg:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={`fixed lg:static inset-y-0 left-0 z-40 w-64 bg-slate-950 text-slate-300 flex flex-col transition-transform duration-200 ease-in-out ${
          isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'
        }`}
      >
        <div className="h-16 px-5 flex items-center justify-between border-b border-slate-800/80">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-lg bg-emerald-500/20 border border-emerald-500/40 flex items-center justify-center">
              <Zap className="w-4 h-4 text-emerald-400" />
            </div>
            <div>
              <span className="block text-sm font-bold text-white leading-tight">EV Demand AI</span>
              <span className="block text-[11px] text-slate-400">Palo Alto Network</span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="lg:hidden p-1 text-slate-400 hover:text-white rounded"
            aria-label="Close sidebar"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        <nav className="flex-1 px-3 py-5 space-y-1 overflow-y-auto">
          <p className="px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
            Analytics &amp; ML
          </p>
          {NAV_ITEMS.map((item) => {
            const Icon = item.icon;
            const active = activePage === item.id;
            return (
              <button
                key={item.id}
                onClick={() => {
                  onSelectPage(item.id);
                  onClose();
                }}
                className={`w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                  active
                    ? 'bg-emerald-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/70'
                }`}
              >
                <Icon className={`w-4 h-4 ${active ? 'text-white' : 'text-slate-400'}`} />
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="p-4 border-t border-slate-800/80">
          <div className="rounded-lg bg-slate-900 p-3 border border-slate-800">
            <p className="text-xs font-semibold text-slate-200">Active Models</p>
            <div className="mt-2 space-y-1.5 text-[11px] text-slate-400">
              <div className="flex justify-between">
                <span>1h Arrival Forecast</span>
                <span className="text-emerald-400 font-medium">XGBoost</span>
              </div>
              <div className="flex justify-between">
                <span>24h Day-Ahead</span>
                <span className="text-emerald-400 font-medium">XGBoost</span>
              </div>
              <div className="flex justify-between">
                <span>Target Metric</span>
                <span className="text-slate-300">Arrivals/hr</span>
              </div>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}
