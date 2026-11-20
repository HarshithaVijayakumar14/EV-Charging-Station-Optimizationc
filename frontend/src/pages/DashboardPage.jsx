import React, { useEffect, useState } from 'react';
import {
  Zap,
  BatteryCharging,
  MapPin,
  Clock,
  ArrowRight,
  TrendingUp,
  Layers,
} from 'lucide-react';
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  Cell,
} from 'recharts';
import {
  getDashboardSummary,
  getHourlyHistory,
  getDailyHistory,
  get1HourPredictions,
} from '../services/api';
import { StatCard } from '../components/StatCard';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner, ErrorState } from '../components/LoadingSpinner';

export function DashboardPage({ onNavigateToStation, onNavigateToPage }) {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summary, setSummary] = useState(null);
  const [hourly, setHourly] = useState([]);
  const [dow, setDow] = useState([]);
  const [monthly, setMonthly] = useState([]);
  const [pred1h, setPred1h] = useState(null);

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumRes, hrRes, dayRes, p1Res] = await Promise.all([
        getDashboardSummary(),
        getHourlyHistory(),
        getDailyHistory({ limit: 365 }),
        get1HourPredictions(),
      ]);
      setSummary(sumRes.summary);
      setHourly(hrRes.hourly || []);
      setDow(hrRes.day_of_week || []);
      setMonthly(dayRes.monthly || []);
      setPred1h(p1Res);
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Failed to load dashboard data.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) return <LoadingSpinner message="Loading Palo Alto EV network telemetry..." />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;
  if (!summary) return null;

  const peakHour = hourly.reduce(
    (best, cur) => (cur.sessions > (best?.sessions || -1) ? cur : best),
    hourly[0]
  );
  const lowHour = hourly.reduce(
    (best, cur) => (cur.sessions < (best?.sessions ?? Infinity) ? cur : best),
    hourly[0]
  );

  return (
    <div className="space-y-6">
      <DisclaimerBanner />

      {/* Quick Access Feature Navigation Ribbon */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-3.5 shadow-xs flex flex-wrap items-center justify-between gap-2.5">
        <span className="text-xs font-bold text-slate-700 flex items-center gap-1.5 uppercase tracking-wider">
          <Zap className="w-3.5 h-3.5 text-emerald-600" />
          Quick Actions:
        </span>
        <div className="flex flex-wrap items-center gap-2">
          <button
            onClick={() => onNavigateToPage('find-station')}
            className="px-3 py-1.5 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-800 text-xs font-bold transition-all border border-emerald-200 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            <MapPin className="w-3.5 h-3.5 text-emerald-600" />
            Find a Station &amp; Availability
          </button>
          <button
            onClick={() => onNavigateToPage('predictions')}
            className="px-3 py-1.5 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-800 text-xs font-bold transition-all border border-blue-200 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            <TrendingUp className="w-3.5 h-3.5 text-blue-600" />
            Demand Forecasts (1h &amp; 24h)
          </button>
          <button
            onClick={() => onNavigateToPage('probabilistic-availability')}
            className="px-3 py-1.5 rounded-lg bg-purple-50 hover:bg-purple-100 text-purple-800 text-xs font-bold transition-all border border-purple-200 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            <Clock className="w-3.5 h-3.5 text-purple-600" />
            Probabilistic Availability (Erlang-B)
          </button>
          <button
            onClick={() => onNavigateToPage('stations')}
            className="px-3 py-1.5 rounded-lg bg-slate-100 hover:bg-slate-200 text-slate-800 text-xs font-bold transition-all border border-slate-300 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            <Layers className="w-3.5 h-3.5 text-slate-600" />
            Station Explorer (46 Sites)
          </button>
          <button
            onClick={() => onNavigateToPage('recommendations')}
            className="px-3 py-1.5 rounded-lg bg-amber-50 hover:bg-amber-100 text-amber-800 text-xs font-bold transition-all border border-amber-200 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            Optimization Plan
          </button>
          <button
            onClick={() => onNavigateToPage('model-performance')}
            className="px-3 py-1.5 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-800 text-xs font-bold transition-all border border-indigo-200 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            Model Evaluation
          </button>
          <button
            onClick={() => onNavigateToPage('availability-calibration')}
            className="px-3 py-1.5 rounded-lg bg-teal-50 hover:bg-teal-100 text-teal-800 text-xs font-bold transition-all border border-teal-200 flex items-center gap-1.5 cursor-pointer shadow-2xs"
          >
            Availability Calibration
          </button>
        </div>
      </div>

      {/* Top KPI Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <StatCard
          title="Total Historical Sessions"
          value={summary.total_sessions.toLocaleString()}
          subtitle={`Cleaned dataset: ${summary.date_range_pt.start.slice(0, 10)} to ${summary.date_range_pt.end.slice(0, 10)}`}
          icon={Zap}
          accent="emerald"
          onClick={() => onNavigateToPage('stations')}
          clickableText="Explore Station Sessions"
        />
        <StatCard
          title="Total Energy Delivered"
          value={`${summary.total_energy_mwh.toLocaleString()} MWh`}
          subtitle={`${summary.total_energy_kwh.toLocaleString()} kWh cumulative across all ports`}
          icon={BatteryCharging}
          accent="blue"
          onClick={() => onNavigateToPage('stations')}
          clickableText="Inspect Energy Usage"
        />
        <StatCard
          title="Charging Stations"
          value={`${summary.station_count} Sites`}
          subtitle={`${summary.historical_training_stations_count} established + ${summary.cold_start_stations_count} cold-start (Dec 2020)`}
          icon={MapPin}
          accent="indigo"
          onClick={() => onNavigateToPage('stations')}
          clickableText="View All 46 Station Details"
        />
        <StatCard
          title="Peak vs. Lowest Demand Hour"
          value={`${peakHour?.label} Peak`}
          subtitle={`${peakHour?.sessions.toLocaleString()} arrivals (${peakHour?.pct_share}%) vs. ${lowHour?.label} (${lowHour?.pct_share}%)`}
          icon={Clock}
          accent="amber"
          onClick={() => onNavigateToPage('predictions')}
          clickableText="Inspect Hourly Forecasts"
        />
      </div>

      {/* Main Charts Row 1: Hourly Pattern + Day of Week */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Hourly Session Arrivals: Weekday vs. Weekend
              </h2>
              <p className="text-xs text-slate-500">
                Average network-wide session arrivals per active day across all 24 clock hours (Pacific Time)
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-emerald-50 text-emerald-700 self-start">
                Morning Commuter Surge (08:00–12:00)
              </span>
              <button
                onClick={() => onNavigateToPage('predictions')}
                className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
              >
                Forecasts &rarr;
              </button>
            </div>
          </div>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={hourly} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                <defs>
                  <linearGradient id="wkdayGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#059669" stopOpacity={0.28} />
                    <stop offset="95%" stopColor="#059669" stopOpacity={0.02} />
                  </linearGradient>
                  <linearGradient id="wkendGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.22} />
                    <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#64748b' }} interval={2} />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: '0.75rem',
                    border: '1px solid #e2e8f0',
                    fontSize: '12px',
                  }}
                />
                <Legend wrapperStyle={{ fontSize: '12px' }} />
                <Area
                  type="monotone"
                  dataKey="weekday_avg_sessions_per_day"
                  name="Weekday Avg Arrivals/Hr"
                  stroke="#059669"
                  strokeWidth={2.2}
                  fill="url(#wkdayGrad)"
                />
                <Area
                  type="monotone"
                  dataKey="weekend_avg_sessions_per_day"
                  name="Weekend Avg Arrivals/Hr"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  fill="url(#wkendGrad)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-1">
            <h2 className="text-base font-bold text-slate-900">Demand by Day of Week</h2>
            <button
              onClick={() => onNavigateToPage('probabilistic-availability')}
              className="text-xs font-semibold text-purple-700 hover:text-purple-800 flex items-center gap-1 cursor-pointer"
            >
              Arrival Curves &rarr;
            </button>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Average daily network sessions (Mon–Fri commuter dominance vs. Sat–Sun)
          </p>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={dow} margin={{ top: 10, right: 10, left: -12, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                <XAxis
                  dataKey="name"
                  tickFormatter={(v) => v.slice(0, 3)}
                  tick={{ fontSize: 11, fill: '#64748b' }}
                />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: '0.75rem',
                    border: '1px solid #e2e8f0',
                    fontSize: '12px',
                  }}
                />
                <Bar dataKey="avg_sessions_per_day" name="Avg Sessions / Day" radius={[6, 6, 0, 0]}>
                  {dow.map((entry, idx) => (
                    <Cell key={idx} fill={entry.is_weekend ? '#94a3b8' : '#10b981'} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Main Charts Row 2: Monthly Multi-Year Trajectory + Level 1 vs Level 2 */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">
                Long-Term Monthly Demand Trajectory (2011–2020)
              </h2>
              <p className="text-xs text-slate-500">
                Growth through 2019 followed by the sharp April 2020 COVID-19 shelter-in-place shock (-87.1%)
              </p>
            </div>
            <button
              onClick={() => onNavigateToPage('model-performance')}
              className="text-xs font-semibold text-indigo-700 hover:text-indigo-800 flex items-center gap-1 cursor-pointer shrink-0"
            >
              Model Shifts &amp; Evals &rarr;
            </button>
          </div>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={monthly} margin={{ top: 10, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis
                  dataKey="year_month"
                  tick={{ fontSize: 11, fill: '#64748b' }}
                  minTickGap={35}
                />
                <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                <Tooltip
                  contentStyle={{
                    borderRadius: '0.75rem',
                    border: '1px solid #e2e8f0',
                    fontSize: '12px',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="sessions"
                  name="Monthly Sessions"
                  stroke="#4f46e5"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Level 1 vs Level 2 Breakdown */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 mb-1">
              <div className="flex items-center gap-2">
                <Layers className="w-4 h-4 text-emerald-600" />
                <h2 className="text-base font-bold text-slate-900">Level 1 vs. Level 2 Usage</h2>
              </div>
              <button
                onClick={() => onNavigateToPage('find-station')}
                className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
              >
                Find Ports &rarr;
              </button>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Cleaned unambiguous port types across all 259,411 sessions
            </p>
            <div className="space-y-4">
              {summary.level_comparison.map((lvl) => (
                <div
                  key={lvl.port_type}
                  className="p-3.5 rounded-lg bg-slate-50 border border-slate-200/70"
                >
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-bold text-slate-900">{lvl.port_type}</span>
                    <span className="text-xs font-semibold text-emerald-700 bg-emerald-100 px-2 py-0.5 rounded-full">
                      {lvl.pct_sessions}% of sessions
                    </span>
                  </div>
                  <div className="w-full h-2 bg-slate-200 rounded-full mt-2 overflow-hidden">
                    <div
                      className={`h-full rounded-full ${
                        lvl.port_type === 'Level 2' ? 'bg-emerald-500' : 'bg-amber-500'
                      }`}
                      style={{ width: `${Math.max(lvl.pct_sessions, 2)}%` }}
                    />
                  </div>
                  <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-slate-600">
                    <div>
                      <span className="block text-[10px] uppercase text-slate-400">Sessions</span>
                      <span className="font-semibold text-slate-800">
                        {lvl.sessions.toLocaleString()}
                      </span>
                    </div>
                    <div>
                      <span className="block text-[10px] uppercase text-slate-400">Avg Energy</span>
                      <span className="font-semibold text-slate-800">{lvl.mean_energy_kwh} kWh</span>
                    </div>
                    <div>
                      <span className="block text-[10px] uppercase text-slate-400">Avg Charge</span>
                      <span className="font-semibold text-slate-800">{lvl.mean_charging_min} min</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="mt-4 pt-3 border-t border-slate-100 text-xs text-slate-500">
            Network average rate: <span className="font-semibold text-slate-700">{summary.avg_sessions_per_station_hour}</span>{' '}
            arrivals/station-hr (<span className="font-semibold text-slate-700">{summary.avg_sessions_per_network_hour}</span> network-wide/hr).
          </div>
        </div>
      </div>

      {/* Bottom Row: Top 10 Stations + Next-Hour Live Forecast Snapshot */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-slate-900">Top 10 Highest-Volume Stations</h2>
              <p className="text-xs text-slate-500">
                Click any station or direct action button to view Availability, Forecasts, or Probabilistic Curves
              </p>
            </div>
            <button
              onClick={() => onNavigateToPage('stations')}
              className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
            >
              View all 46 <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-sm">
              <thead>
                <tr className="border-b border-slate-200 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                  <th className="py-2.5 pr-2">Station</th>
                  <th className="py-2.5 px-2 text-right">Sessions</th>
                  <th className="py-2.5 px-2 text-right">Total Energy</th>
                  <th className="py-2.5 px-2 text-right">Per Day</th>
                  <th className="py-2.5 px-2 text-right">Avg kWh</th>
                  <th className="py-2.5 pl-2 text-right">Direct Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {summary.top_10_stations.map((st, idx) => (
                  <tr
                    key={st.station_id}
                    onClick={() => onNavigateToStation(st.station_id, 'stations')}
                    className="hover:bg-slate-50 cursor-pointer transition-colors"
                  >
                    <td className="py-2.5 pr-2 font-medium text-slate-900 flex items-center gap-2">
                      <span className="w-5 h-5 rounded-full bg-slate-100 text-slate-600 text-[11px] font-bold flex items-center justify-center shrink-0">
                        {idx + 1}
                      </span>
                      <span className="truncate max-w-[170px]" title={st.short_name}>
                        {st.short_name}
                      </span>
                    </td>
                    <td className="py-2.5 px-2 text-right font-semibold text-slate-800">
                      {st.total_sessions.toLocaleString()}
                    </td>
                    <td className="py-2.5 px-2 text-right text-slate-600 text-xs">
                      {st.total_energy_kwh.toLocaleString()} kWh
                    </td>
                    <td className="py-2.5 px-2 text-right text-slate-600 text-xs">
                      {st.sessions_per_active_day}
                    </td>
                    <td className="py-2.5 px-2 text-right text-slate-600 text-xs">{st.mean_energy_kwh}</td>
                    <td className="py-2.5 pl-2 text-right">
                      <div
                        className="flex items-center justify-end gap-1"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <button
                          type="button"
                          onClick={() => onNavigateToStation(st.station_id, 'find-station')}
                          className="px-2 py-0.5 rounded bg-emerald-50 hover:bg-emerald-100 text-emerald-800 text-[11px] font-semibold border border-emerald-200 transition-colors cursor-pointer"
                          title={`Check Availability Assistant for ${st.short_name}`}
                        >
                          Availability
                        </button>
                        <button
                          type="button"
                          onClick={() => onNavigateToStation(st.station_id, 'predictions')}
                          className="px-2 py-0.5 rounded bg-blue-50 hover:bg-blue-100 text-blue-800 text-[11px] font-semibold border border-blue-200 transition-colors cursor-pointer"
                          title={`View Demand Forecast for ${st.short_name}`}
                        >
                          Forecast
                        </button>
                        <button
                          type="button"
                          onClick={() => onNavigateToStation(st.station_id, 'probabilistic-availability')}
                          className="px-2 py-0.5 rounded bg-purple-50 hover:bg-purple-100 text-purple-800 text-[11px] font-semibold border border-purple-200 transition-colors cursor-pointer"
                          title={`View Probabilistic Erlang-B Curve for ${st.short_name}`}
                        >
                          Probabilistic
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Live 1-Hour Prediction Snapshot */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-emerald-600" />
                <h2 className="text-base font-bold text-slate-900">1-Hour Forecast Snapshot</h2>
              </div>
              <span className="text-[11px] font-semibold bg-emerald-50 text-emerald-700 px-2 py-0.5 rounded-full">
                XGBoost
              </span>
            </div>
            <p className="text-xs text-slate-500 mb-4">
              Highest predicted next-hour arrival rates across {pred1h?.forecasted_stations_count || 0} forecastable sites
            </p>

            <div className="space-y-2.5">
              {(pred1h?.predictions || []).slice(0, 6).map((item) => (
                <div
                  key={item.station_id}
                  className="p-3 rounded-lg border border-slate-200/70 hover:border-emerald-300 hover:bg-emerald-50/20 transition-all space-y-2"
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="min-w-0">
                      <p className="text-xs font-bold text-slate-900 truncate">{item.short_name}</p>
                      <p className="text-[11px] text-slate-500">
                        Target: {item.target_timestamp_pt.slice(0, 16)} PT
                      </p>
                    </div>
                    <div className="text-right shrink-0">
                      <span className="block text-sm font-bold text-emerald-700">
                        {item.predicted_sessions.toFixed(2)}
                      </span>
                      <span className="block text-[10px] text-slate-400">
                        (~{item.predicted_sessions_rounded} arr/hr)
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center justify-end gap-1.5 pt-1.5 border-t border-slate-100 text-[11px]">
                    <button
                      type="button"
                      onClick={() => onNavigateToStation(item.station_id, 'find-station')}
                      className="px-2 py-0.5 rounded bg-emerald-50 hover:bg-emerald-100 text-emerald-700 font-semibold cursor-pointer transition-colors"
                    >
                      Availability
                    </button>
                    <button
                      type="button"
                      onClick={() => onNavigateToStation(item.station_id, 'predictions')}
                      className="px-2 py-0.5 rounded bg-blue-50 hover:bg-blue-100 text-blue-700 font-semibold cursor-pointer transition-colors"
                    >
                      Forecast
                    </button>
                    <button
                      type="button"
                      onClick={() => onNavigateToStation(item.station_id, 'probabilistic-availability')}
                      className="px-2 py-0.5 rounded bg-purple-50 hover:bg-purple-100 text-purple-700 font-semibold cursor-pointer transition-colors"
                    >
                      Probabilistic
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-2 mt-4">
            <button
              onClick={() => onNavigateToPage('predictions')}
              className="w-full py-2.5 px-4 bg-slate-900 hover:bg-slate-800 text-white text-xs font-semibold rounded-lg transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
            >
              Open Interactive Demand Forecaster <ArrowRight className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={() => onNavigateToPage('find-station')}
              className="w-full py-2 px-4 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg transition-colors flex items-center justify-center gap-1.5 cursor-pointer"
            >
              Find a Station &amp; Availability Assistant <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
