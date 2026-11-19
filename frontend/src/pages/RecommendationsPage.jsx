import React, { useEffect, useState, useMemo } from 'react';
import {
  Lightbulb,
  AlertCircle,
  TrendingUp,
  Clock,
  Sparkles,
  Layers,
  ArrowRight,
  Filter,
} from 'lucide-react';
import { getRecommendations } from '../services/api';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner, ErrorState } from '../components/LoadingSpinner';

export function RecommendationsPage({ onNavigateToStationForecast }) {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedCategory, setSelectedCategory] = useState('ALL');
  const [selectedPriority, setSelectedPriority] = useState('ALL');

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getRecommendations();
      setData(res);
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Failed to load recommendations.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  const filtered = useMemo(() => {
    if (!data?.recommendations) return [];
    return data.recommendations.filter((r) => {
      if (selectedCategory !== 'ALL' && r.category !== selectedCategory) return false;
      if (selectedPriority !== 'ALL' && r.priority !== selectedPriority) return false;
      return true;
    });
  }, [data, selectedCategory, selectedPriority]);

  if (loading) return <LoadingSpinner message="Evaluating explainable demand planning rules..." />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;
  if (!data) return null;

  const priorityStyles = {
    High: 'bg-rose-50 text-rose-800 border-rose-200',
    Medium: 'bg-amber-50 text-amber-800 border-amber-200',
    Low: 'bg-slate-100 text-slate-700 border-slate-200',
    Info: 'bg-blue-50 text-blue-800 border-blue-200',
  };

  const categories = Object.keys(data.category_counts || {});

  return (
    <div className="space-y-6">
      <DisclaimerBanner text={data.disclaimer} />

      {/* Header and Filter Controls */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <Lightbulb className="w-5 h-5 text-amber-500" />
              <h2 className="text-base font-bold text-slate-900">
                Rule-Based Demand Planning Suggestions ({data.total_recommendations} Total)
              </h2>
            </div>
            <p className="text-xs text-slate-500 mt-1">
              Derived transparently from Phase 5 XGBoost forecasts (1h &amp; 24h) and station historical activity profiles.
            </p>
          </div>

          {/* Category Chips */}
          <div className="flex flex-wrap items-center gap-2">
            <button
              onClick={() => setSelectedCategory('ALL')}
              className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                selectedCategory === 'ALL'
                  ? 'bg-slate-900 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              All Categories ({data.total_recommendations})
            </button>
            {categories.map((cat) => (
              <button
                key={cat}
                onClick={() => setSelectedCategory(cat)}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-colors ${
                  selectedCategory === cat
                    ? 'bg-slate-900 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {cat} ({data.category_counts[cat]})
              </button>
            ))}
          </div>
        </div>

        {/* Priority Filter */}
        <div className="mt-4 pt-3 border-t border-slate-100 flex items-center gap-2 text-xs">
          <Filter className="w-3.5 h-3.5 text-slate-400" />
          <span className="font-semibold text-slate-600">Priority Filter:</span>
          {['ALL', 'High', 'Medium', 'Low', 'Info'].map((p) => (
            <button
              key={p}
              onClick={() => setSelectedPriority(p)}
              className={`px-2.5 py-1 rounded-md text-xs font-medium transition-colors ${
                selectedPriority === p
                  ? 'bg-emerald-600 text-white'
                  : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
            >
              {p}
            </button>
          ))}
          <span className="ml-auto text-slate-400">
            Showing {filtered.length} recommendation{filtered.length === 1 ? '' : 's'}
          </span>
        </div>
      </div>

      {/* Cards List */}
      <div className="space-y-4">
        {filtered.map((item) => (
          <div
            key={item.id}
            className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm hover:border-slate-300 transition-all flex flex-col justify-between"
          >
            <div>
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
                <div className="flex items-center gap-2.5 flex-wrap">
                  <span className="text-sm font-bold text-slate-900">{item.short_name}</span>
                  <span className="text-[11px] font-mono text-slate-400 hidden md:inline">
                    ({item.station_id})
                  </span>
                  <span
                    className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                      priorityStyles[item.priority] || priorityStyles.Info
                    }`}
                  >
                    {item.priority} Priority
                  </span>
                </div>
                <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500 bg-slate-100 px-2.5 py-1 rounded-md self-start">
                  {item.category}
                </span>
              </div>

              {/* Operational Planning Message */}
              <div className="mt-3 text-sm font-semibold text-slate-900 flex items-start gap-2">
                <Sparkles className="w-4 h-4 text-emerald-600 shrink-0 mt-0.5" />
                <span>{item.message}</span>
              </div>

              {/* Explainable Rationale */}
              <p className="mt-2 text-xs text-slate-600 leading-relaxed pl-6 border-l-2 border-slate-200">
                {item.reason}
              </p>

              {/* Quantitative Metrics Pills */}
              {item.metrics && (
                <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
                  {Object.entries(item.metrics).map(([k, v]) => {
                    if (Array.isArray(v) || typeof v === 'object' || v === null) return null;
                    return (
                      <span
                        key={k}
                        className="px-2.5 py-1 rounded-md bg-slate-50 border border-slate-200 text-slate-700"
                      >
                        <span className="text-slate-400 capitalize">
                          {k.replace(/_/g, ' ')}:{' '}
                        </span>
                        <strong>{typeof v === 'number' ? Number(v).toFixed(2) : String(v)}</strong>
                      </span>
                    );
                  })}
                </div>
              )}
            </div>

            {/* Bottom Action Footer */}
            <div className="mt-4 pt-3 border-t border-slate-100 flex items-center justify-between">
              <span className="text-[11px] text-slate-400 italic">
                Label: {item.label} (Non-prescriptive arrival-flow insight)
              </span>
              {item.station_id !== 'SHERMAN GROUP (13 NEW STATIONS)' && (
                <button
                  onClick={() => onNavigateToStationForecast(item.station_id)}
                  className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 flex items-center gap-1"
                >
                  Inspect Forecast <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        ))}

        {filtered.length === 0 && (
          <div className="bg-slate-50 border border-slate-200 rounded-xl p-8 text-center text-slate-500 text-sm">
            No recommendations match your selected filters.
          </div>
        )}
      </div>
    </div>
  );
}
