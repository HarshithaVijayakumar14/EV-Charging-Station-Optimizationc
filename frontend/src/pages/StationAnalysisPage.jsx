import React, { useEffect, useState, useMemo } from 'react';
import {
  Search,
  MapPin,
  Calendar,
  Zap,
  BatteryCharging,
  Clock,
  TrendingUp,
  AlertTriangle,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
} from 'recharts';
import { getStations, getStationDetail } from '../services/api';
import { StationBadge } from '../components/StationBadge';
import { LoadingSpinner, ErrorState } from '../components/LoadingSpinner';

export function StationAnalysisPage({ initialStationId, onOpenForecastForStation }) {
  const [stations, setStations] = useState([]);
  const [selectedId, setSelectedId] = useState(initialStationId || '');
  const [detail, setDetail] = useState(null);
  const [search, setSearch] = useState('');
  const [filterType, setFilterType] = useState('all'); // 'all' | 'established' | 'cold_start'
  const [loadingList, setLoadingList] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (initialStationId) setSelectedId(initialStationId);
  }, [initialStationId]);

  const loadStations = async () => {
    setLoadingList(true);
    setError(null);
    try {
      const res = await getStations();
      const list = res.stations || [];
      setStations(list);
      if (!selectedId && list.length > 0) {
        setSelectedId(initialStationId || list[0].station_id);
      }
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Failed to load stations list.');
    } finally {
      setLoadingList(false);
    }
  };

  useEffect(() => {
    loadStations();
  }, []);

  useEffect(() => {
    if (!selectedId) return;
    let cancelled = false;
    const loadDetail = async () => {
      setLoadingDetail(true);
      try {
        const res = await getStationDetail(selectedId);
        if (!cancelled) setDetail(res.station);
      } catch (err) {
        if (!cancelled) setError(err?.response?.data?.message || err.message);
      } finally {
        if (!cancelled) setLoadingDetail(false);
      }
    };
    loadDetail();
    return () => {
      cancelled = true;
    };
  }, [selectedId]);

  const filteredStations = useMemo(() => {
    return stations.filter((s) => {
      if (filterType === 'established' && s.is_cold_start) return false;
      if (filterType === 'cold_start' && !s.is_cold_start) return false;
      if (search.trim()) {
        const q = search.toLowerCase();
        return (
          s.station_id.toLowerCase().includes(q) ||
          s.short_name.toLowerCase().includes(q) ||
          (s.address && s.address.toLowerCase().includes(q))
        );
      }
      return true;
    });
  }, [stations, search, filterType]);

  if (loadingList) return <LoadingSpinner message="Loading 46 Palo Alto EV stations..." />;
  if (error) return <ErrorState message={error} onRetry={loadStations} />;

  const hourlyProfileData = detail
    ? detail.hourly_mean_sessions.map((meanVal, hour) => ({
        label: `${String(hour).padStart(2, '0')}:00`,
        mean_arrivals: meanVal,
        total_sessions: detail.hourly_total_sessions[hour],
      }))
    : [];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
      {/* Left Directory Column */}
      <div className="lg:col-span-4 bg-white rounded-xl border border-slate-200/80 shadow-sm flex flex-col max-h-[820px]">
        <div className="p-4 border-b border-slate-100 space-y-3">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search station name or address..."
              className="w-full pl-9 pr-3 py-2 text-sm bg-slate-50 border border-slate-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-emerald-500"
            />
          </div>
          <div className="flex items-center gap-1.5">
            {[
              { id: 'all', label: `All (${stations.length})` },
              {
                id: 'established',
                label: `Established (${stations.filter((s) => !s.is_cold_start).length})`,
              },
              {
                id: 'cold_start',
                label: `Cold-Start (${stations.filter((s) => s.is_cold_start).length})`,
              },
            ].map((f) => (
              <button
                key={f.id}
                onClick={() => setFilterType(f.id)}
                className={`flex-1 py-1.5 px-2 text-[11px] font-semibold rounded-md transition-colors ${
                  filterType === f.id
                    ? 'bg-slate-900 text-white'
                    : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                }`}
              >
                {f.label}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto divide-y divide-slate-100">
          {filteredStations.map((st) => {
            const active = st.station_id === selectedId;
            return (
              <button
                key={st.station_id}
                onClick={() => setSelectedId(st.station_id)}
                className={`w-full text-left p-3.5 transition-colors ${
                  active ? 'bg-emerald-50/90 border-l-4 border-emerald-600' : 'hover:bg-slate-50'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <span className="text-sm font-bold text-slate-900 leading-snug">
                    {st.short_name}
                  </span>
                  <StationBadge
                    isColdStart={st.is_cold_start}
                    hasSufficientHistory={st.has_sufficient_history}
                    size="xs"
                  />
                </div>
                <p className="text-xs text-slate-500 mt-1 truncate">{st.address}</p>
                <div className="mt-2 flex items-center justify-between text-[11px] text-slate-600">
                  <span>
                    <strong>{st.total_sessions.toLocaleString()}</strong> sessions
                  </span>
                  <span>
                    <strong>{st.sessions_per_active_day}</strong> / active day
                  </span>
                </div>
              </button>
            );
          })}
          {filteredStations.length === 0 && (
            <p className="p-6 text-center text-sm text-slate-500">No stations match your search.</p>
          )}
        </div>
      </div>

      {/* Right Station Detail Panel */}
      <div className="lg:col-span-8 space-y-5">
        {loadingDetail || !detail ? (
          <LoadingSpinner message="Loading station telemetry..." />
        ) : (
          <>
            {/* Station Header Card */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4">
                <div>
                  <div className="flex items-center gap-2.5 flex-wrap">
                    <h2 className="text-lg font-bold text-slate-900">{detail.short_name}</h2>
                    <StationBadge
                      isColdStart={detail.is_cold_start}
                      hasSufficientHistory={detail.has_sufficient_history}
                    />
                  </div>
                  <p className="text-xs text-slate-500 mt-1 font-mono">{detail.station_id}</p>
                  <div className="mt-2.5 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-slate-600">
                    <span className="flex items-center gap-1">
                      <MapPin className="w-3.5 h-3.5 text-emerald-600" />
                      {detail.address} ({detail.latitude}, {detail.longitude})
                    </span>
                    <span className="flex items-center gap-1">
                      <Calendar className="w-3.5 h-3.5 text-blue-600" />
                      Active span: {detail.first_hour_pt.slice(0, 10)} to {detail.last_hour_pt.slice(0, 10)}
                    </span>
                  </div>
                </div>

                <button
                  onClick={() => onOpenForecastForStation(detail.station_id)}
                  className="px-4 py-2 bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5 shrink-0"
                >
                  <TrendingUp className="w-3.5 h-3.5" />
                  Run Demand Forecast
                </button>
              </div>

              {detail.cold_start_note && (
                <div className="mt-4 p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-start gap-2">
                  <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
                  <span>{detail.cold_start_note}</span>
                </div>
              )}

              {/* Metrics Grid */}
              <div className="mt-5 grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-1">
                    <Zap className="w-3 h-3 text-emerald-600" /> Sessions
                  </span>
                  <p className="mt-1 text-lg font-bold text-slate-900">
                    {detail.total_sessions.toLocaleString()}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    {detail.sessions_per_active_day}/active day
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-1">
                    <BatteryCharging className="w-3 h-3 text-blue-600" /> Energy
                  </span>
                  <p className="mt-1 text-lg font-bold text-slate-900">
                    {detail.total_energy_kwh.toLocaleString()} kWh
                  </p>
                  <p className="text-[11px] text-slate-500">Avg {detail.mean_energy_kwh} kWh/session</p>
                </div>

                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold flex items-center gap-1">
                    <Clock className="w-3 h-3 text-indigo-600" /> Avg Dwell
                  </span>
                  <p className="mt-1 text-lg font-bold text-slate-900">
                    {detail.mean_plugged_min ?? '—'} min
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Active charge: {detail.mean_charging_min ?? '—'} min
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-slate-50 border border-slate-100">
                  <span className="text-[11px] uppercase tracking-wider text-slate-500 font-semibold">
                    Top Peak Hours
                  </span>
                  <p className="mt-1 text-lg font-bold text-slate-900">
                    {detail.peak_hours.map((h) => `${String(h).padStart(2, '0')}h`).join(', ')}
                  </p>
                  <p className="text-[11px] text-slate-500">
                    Ports: {detail.port_types.join(', ')}
                  </p>
                </div>
              </div>
            </div>

            {/* Station 24-Hour Arrival Profile */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
              <h3 className="text-base font-bold text-slate-900">
                Station Hourly Arrival Profile (Mean Sessions / Hour)
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Calculated across all {detail.span_hours.toLocaleString()} observed station-hours (Pacific Time)
              </p>
              <div className="h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={hourlyProfileData} margin={{ top: 5, right: 10, left: -12, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" vertical={false} />
                    <XAxis dataKey="label" tick={{ fontSize: 10, fill: '#64748b' }} interval={1} />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: '0.75rem',
                        border: '1px solid #e2e8f0',
                        fontSize: '12px',
                      }}
                    />
                    <Bar
                      dataKey="mean_arrivals"
                      name="Mean Arrivals / Hour"
                      fill="#10b981"
                      radius={[4, 4, 0, 0]}
                    />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Recent Daily Activity Trend */}
            <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
              <h3 className="text-base font-bold text-slate-900">
                Recent Active-Day Trend (Last {detail.daily_recent.length} Active Days)
              </h3>
              <p className="text-xs text-slate-500 mb-4">
                Daily recorded session arrivals at {detail.short_name}
              </p>
              <div className="h-56">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart
                    data={detail.daily_recent}
                    margin={{ top: 5, right: 10, left: -12, bottom: 0 }}
                  >
                    <defs>
                      <linearGradient id="stDailyGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#6366f1" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#6366f1" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="date"
                      tick={{ fontSize: 10, fill: '#64748b' }}
                      minTickGap={30}
                    />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: '0.75rem',
                        border: '1px solid #e2e8f0',
                        fontSize: '12px',
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="sessions"
                      name="Daily Sessions"
                      stroke="#4f46e5"
                      strokeWidth={2}
                      fill="url(#stDailyGrad)"
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
