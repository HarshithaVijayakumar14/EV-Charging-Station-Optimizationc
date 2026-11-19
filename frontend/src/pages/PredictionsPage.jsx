import React, { useEffect, useState } from 'react';
import {
  TrendingUp,
  Clock,
  CalendarClock,
  AlertTriangle,
  History,
  Sparkles,
  RefreshCw,
} from 'lucide-react';
import {
  ResponsiveContainer,
  ComposedChart,
  Bar,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import {
  getStations,
  getStationPredictions,
  get1HourPredictions,
  get24HourPredictions,
} from '../services/api';
import { StationBadge } from '../components/StationBadge';
import { DisclaimerBanner } from '../components/DisclaimerBanner';
import { LoadingSpinner, ErrorState } from '../components/LoadingSpinner';

export function PredictionsPage({ initialStationId }) {
  const [stations, setStations] = useState([]);
  const [selectedStation, setSelectedStation] = useState(initialStationId || '');
  const [horizonView, setHorizonView] = useState('24h'); // '1h' | '24h'
  const [backtestHours, setBacktestHours] = useState(72);

  const [stationData, setStationData] = useState(null);
  const [network1h, setNetwork1h] = useState(null);
  const [network24h, setNetwork24h] = useState(null);

  const [loadingInit, setLoadingInit] = useState(true);
  const [loadingStation, setLoadingStation] = useState(false);
  const [error, setError] = useState(null);

  // Sync if navigated here with a specific station
  useEffect(() => {
    if (initialStationId) {
      setSelectedStation(initialStationId);
    }
  }, [initialStationId]);

  const loadInitial = async () => {
    setLoadingInit(true);
    setError(null);
    try {
      const [stRes, n1Res, n24Res] = await Promise.all([
        getStations(),
        get1HourPredictions(),
        get24HourPredictions(),
      ]);
      const list = stRes.stations || [];
      setStations(list);
      setNetwork1h(n1Res);
      setNetwork24h(n24Res);
      if (!selectedStation && list.length > 0) {
        setSelectedStation(initialStationId || list[0].station_id);
      }
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Failed to load forecasting endpoints.');
    } finally {
      setLoadingInit(false);
    }
  };

  useEffect(() => {
    loadInitial();
  }, []);

  const loadStationForecast = async (stId, hours) => {
    if (!stId) return;
    setLoadingStation(true);
    try {
      const res = await getStationPredictions(stId, hours);
      setStationData(res);
    } catch (err) {
      setError(err?.response?.data?.message || err.message);
    } finally {
      setLoadingStation(false);
    }
  };

  useEffect(() => {
    if (selectedStation) {
      loadStationForecast(selectedStation, backtestHours);
    }
  }, [selectedStation, backtestHours]);

  if (loadingInit) return <LoadingSpinner message="Initializing XGBoost forecasting engine..." />;
  if (error) return <ErrorState message={error} onRetry={loadInitial} />;

  const p1 = stationData?.prediction_1h;
  const p24 = stationData?.prediction_24h;
  const backtest = stationData?.historical_backtest;
  const hasHistory = stationData?.has_sufficient_history !== false && p1?.status === 'ok';

  return (
    <div className="space-y-6">
      <DisclaimerBanner text="Forecasts predict expected charging session arrivals (sessions_count) per clock hour using pre-trained Phase 5 XGBoost models with strictly causal historical features. Predictions do not represent simultaneous port occupancy or physical station capacity." />

      {/* Control Bar: Station Selector + Horizon Toggle */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4">
          <div className="flex-1 max-w-xl">
            <label
              htmlFor="station-select"
              className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1.5"
            >
              Select Charging Station ({stations.length} Available)
            </label>
            <select
              id="station-select"
              value={selectedStation}
              onChange={(e) => setSelectedStation(e.target.value)}
              className="w-full px-3.5 py-2.5 bg-slate-50 border border-slate-300 rounded-lg text-sm font-medium text-slate-900 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
            >
              <optgroup label="Established Historical Stations (2011–2018 Training Data)">
                {stations
                  .filter((s) => !s.is_cold_start)
                  .map((s) => (
                    <option key={s.station_id} value={s.station_id}>
                      {s.short_name} — {s.total_sessions.toLocaleString()} sessions ({s.sessions_per_active_day}/day)
                    </option>
                  ))}
              </optgroup>
              <optgroup label="Cold-Start Stations (Opened Dec 2020)">
                {stations
                  .filter((s) => s.is_cold_start)
                  .map((s) => (
                    <option key={s.station_id} value={s.station_id}>
                      {s.short_name} — {s.span_hours}h observed {!s.has_sufficient_history ? '(Insufficient History)' : '(Cold-Start)'}
                    </option>
                  ))}
              </optgroup>
            </select>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex rounded-lg bg-slate-100 p-1 border border-slate-200">
              <button
                onClick={() => setHorizonView('1h')}
                className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center gap-1.5 ${
                  horizonView === '1h'
                    ? 'bg-white text-emerald-700 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <Clock className="w-3.5 h-3.5" />
                1-Hour-Ahead
              </button>
              <button
                onClick={() => setHorizonView('24h')}
                className={`px-3.5 py-1.5 rounded-md text-xs font-semibold transition-all flex items-center gap-1.5 ${
                  horizonView === '24h'
                    ? 'bg-white text-emerald-700 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                <CalendarClock className="w-3.5 h-3.5" />
                24-Hour Day-Ahead
              </button>
            </div>

            <button
              onClick={() => loadStationForecast(selectedStation, backtestHours)}
              disabled={loadingStation}
              className="px-3 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-50 text-white text-xs font-semibold rounded-lg transition-colors flex items-center gap-1.5"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${loadingStation ? 'animate-spin' : ''}`} />
              Refresh Forecast
            </button>
          </div>
        </div>

        {/* Station Identity & Cold-Start Note */}
        {stationData && (
          <div className="mt-4 pt-4 border-t border-slate-100 flex flex-wrap items-center justify-between gap-3">
            <div className="flex items-center gap-2.5 flex-wrap">
              <span className="text-sm font-bold text-slate-900">{stationData.station_id}</span>
              <StationBadge
                isColdStart={stationData.is_cold_start}
                hasSufficientHistory={stationData.has_sufficient_history}
              />
            </div>
            {p1?.forecast_origin_pt && (
              <span className="text-xs text-slate-500">
                Forecast Origin (Last Observed Data):{' '}
                <span className="font-semibold text-slate-700">{p1.forecast_origin_pt} PT</span>
              </span>
            )}
          </div>
        )}

        {stationData?.cold_start_note && (
          <div className="mt-3 p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <span>{stationData.cold_start_note}</span>
          </div>
        )}
      </div>

      {/* Station Forecast Body */}
      {loadingStation ? (
        <LoadingSpinner message={`Running XGBoost inference for ${selectedStation}...`} />
      ) : !hasHistory ? (
        <div className="bg-rose-50 border border-rose-200 rounded-xl p-6 text-center">
          <AlertTriangle className="w-8 h-8 text-rose-500 mx-auto mb-2" />
          <h3 className="text-base font-bold text-rose-900">
            Insufficient Historical Data for Lag-Based Prediction
          </h3>
          <p className="mt-1 text-sm text-rose-700 max-w-xl mx-auto">
            {p1?.reason ||
              'This newly opened station has fewer than 24 recorded hours. Rather than fabricating missing history or outputting unreliable zero defaults, predictions are withheld until sufficient operating history is collected.'}
          </p>
        </div>
      ) : (
        <>
          {/* KPI Comparison Cards for Selected Station */}
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-4">
            <div className="bg-gradient-to-br from-emerald-600 to-emerald-700 text-white rounded-xl p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-emerald-100">
                  1-Hour-Ahead Forecast
                </span>
                <Clock className="w-4 h-4 text-emerald-200" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-bold">{p1.predicted_sessions.toFixed(3)}</span>
                <span className="text-xs text-emerald-100">expected arrivals</span>
              </div>
              <p className="mt-1 text-xs text-emerald-100">
                Rounded count: <span className="font-bold">~{p1.predicted_sessions_rounded} session(s)</span>
              </p>
              <p className="mt-2 pt-2 border-t border-emerald-500/40 text-[11px] text-emerald-100">
                Target hour: {p1.target_timestamp_pt.slice(0, 16)} PT
              </p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Historical Same-Hour Baseline
                </span>
                <Sparkles className="w-4 h-4 text-blue-500" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-bold text-slate-900">
                  {p1.historical_hour_avg.toFixed(3)}
                </span>
                <span className="text-xs text-slate-500">avg arrivals at {String(p1.target_hour).padStart(2, '0')}:00</span>
              </div>
              <p className="mt-1 text-xs text-slate-500">
                Overall hourly mean: <span className="font-semibold text-slate-700">{p1.overall_hourly_mean}</span>/hr
              </p>
              <p className="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500">
                Prior plugged-in at hour start: <span className="font-semibold text-slate-700">{p1.prior_plugged_in_vehicles}</span> vehicle(s)
              </p>
            </div>

            <div className="bg-gradient-to-br from-indigo-600 to-indigo-700 text-white rounded-xl p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-indigo-100">
                  24-Hour Day-Ahead Total
                </span>
                <CalendarClock className="w-4 h-4 text-indigo-200" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-bold">{p24.total_predicted_sessions_24h.toFixed(2)}</span>
                <span className="text-xs text-indigo-100">expected arrivals</span>
              </div>
              <p className="mt-1 text-xs text-indigo-100">
                Rounded 24h count: <span className="font-bold">~{p24.total_predicted_sessions_24h_rounded} sessions</span>
              </p>
              <p className="mt-2 pt-2 border-t border-indigo-500/40 text-[11px] text-indigo-100">
                Avg {p24.mean_hourly_predicted_sessions.toFixed(3)} arrivals/hr over next 24h
              </p>
            </div>

            <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                  Predicted 24h Peak Hour
                </span>
                <TrendingUp className="w-4 h-4 text-amber-500" />
              </div>
              <div className="mt-2 flex items-baseline gap-2">
                <span className="text-3xl font-bold text-slate-900">
                  {String(p24.peak_predicted_hour).padStart(2, '0')}:00
                </span>
                <span className="text-xs text-slate-500">Pacific Time</span>
              </div>
              <p className="mt-1 text-xs text-slate-600">
                Peak intensity:{' '}
                <span className="font-bold text-amber-700">
                  {p24.peak_predicted_sessions.toFixed(3)} arrivals/hr
                </span>
              </p>
              <p className="mt-2 pt-2 border-t border-slate-100 text-[11px] text-slate-500">
                Window ends: {p24.forecast_window_end_pt.slice(0, 16)} PT
              </p>
            </div>
          </div>

          {/* Primary Visualization: 24-Hour Day-Ahead Hourly Forecast Curve */}
          <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
              <div>
                <h2 className="text-base font-bold text-slate-900">
                  {horizonView === '24h'
                    ? 'Next 24-Hour Predicted Demand Profile (Day-Ahead XGBoost)'
                    : 'Next 24-Hour Context & 1-Hour-Ahead Target Highlight'}
                </h2>
                <p className="text-xs text-slate-500">
                  Strictly causal day-ahead expected session arrivals vs. station&apos;s historical hourly average
                </p>
              </div>
              <span className="text-xs font-medium px-2.5 py-1 rounded-full bg-indigo-50 text-indigo-700 self-start">
                Model: {p24.model_name}
              </span>
            </div>
            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ComposedChart
                  data={p24.hourly_predictions}
                  margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#64748b' }} />
                  <YAxis tick={{ fontSize: 11, fill: '#64748b' }} />
                  <Tooltip
                    contentStyle={{
                      borderRadius: '0.75rem',
                      border: '1px solid #e2e8f0',
                      fontSize: '12px',
                    }}
                    formatter={(value, name) => [Number(value).toFixed(4), name]}
                    labelFormatter={(lbl, payload) =>
                      payload?.[0]?.payload?.target_timestamp_pt
                        ? `Target: ${payload[0].payload.target_timestamp_pt.slice(0, 16)} PT`
                        : lbl
                    }
                  />
                  <Legend wrapperStyle={{ fontSize: '12px' }} />
                  <Bar
                    dataKey="predicted_sessions"
                    name="24h-Ahead Predicted Arrivals"
                    fill="#4f46e5"
                    radius={[5, 5, 0, 0]}
                    barSize={18}
                  />
                  <Line
                    type="monotone"
                    dataKey="historical_hour_avg"
                    name="Historical Hour Average"
                    stroke="#f59e0b"
                    strokeWidth={2.2}
                    dot={{ r: 3 }}
                  />
                </ComposedChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Historical Actual-vs-Predicted Backtest */}
          {backtest && backtest.status === 'ok' && (
            <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 mb-4">
                <div>
                  <div className="flex items-center gap-2">
                    <History className="w-4 h-4 text-slate-700" />
                    <h2 className="text-base font-bold text-slate-900">
                      Historical Backtest: Actual vs. Predicted Arrivals
                    </h2>
                    <span className="text-[11px] font-semibold px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200">
                      Observed Past Window
                    </span>
                  </div>
                  <p className="text-xs text-slate-500 mt-0.5">
                    Evaluates strictly causal predictions against known historical arrivals over the most recent {backtest.hours_evaluated} hours.
                    Window MAE — 1h Model: <strong>{backtest.window_mae_1h}</strong> | 24h Model:{' '}
                    <strong>{backtest.window_mae_24h}</strong> | Totals: Actual{' '}
                    <strong>{backtest.actual_total}</strong> vs. 1h Pred{' '}
                    <strong>{backtest.predicted_1h_total}</strong>
                  </p>
                </div>

                <div className="inline-flex rounded-lg bg-slate-100 p-0.5 border border-slate-200 shrink-0">
                  {[48, 72, 168].map((h) => (
                    <button
                      key={h}
                      onClick={() => setBacktestHours(h)}
                      className={`px-2.5 py-1 rounded-md text-xs font-semibold ${
                        backtestHours === h
                          ? 'bg-white text-slate-900 shadow-xs'
                          : 'text-slate-500 hover:text-slate-800'
                      }`}
                    >
                      Last {h}h
                    </button>
                  ))}
                </div>
              </div>

              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart
                    data={backtest.records.map((r) => ({
                      ...r,
                      short_ts: r.timestamp_pt.slice(5, 16),
                    }))}
                    margin={{ top: 10, right: 20, left: 0, bottom: 0 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="short_ts"
                      tick={{ fontSize: 10, fill: '#64748b' }}
                      minTickGap={30}
                    />
                    <YAxis tick={{ fontSize: 11, fill: '#64748b' }} allowDecimals={true} />
                    <Tooltip
                      contentStyle={{
                        borderRadius: '0.75rem',
                        border: '1px solid #e2e8f0',
                        fontSize: '12px',
                      }}
                    />
                    <Legend wrapperStyle={{ fontSize: '12px' }} />
                    <Bar
                      dataKey="actual_sessions"
                      name="Actual Recorded Arrivals"
                      fill="#cbd5e1"
                      radius={[3, 3, 0, 0]}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted_1h_sessions"
                      name="1h-Ahead Prediction"
                      stroke="#059669"
                      strokeWidth={2}
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted_24h_sessions"
                      name="24h-Ahead Prediction"
                      stroke="#6366f1"
                      strokeWidth={1.8}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                  </ComposedChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {/* Network-Wide Comparison Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900">
              All Stations Forecast Comparison ({horizonView === '1h' ? '1-Hour-Ahead' : '24-Hour-Ahead'})
            </h2>
            <p className="text-xs text-slate-500">
              Ranked from highest to lowest predicted session arrivals. Click any row to load its detailed charts above.
            </p>
          </div>
          <span className="text-xs text-slate-500">
            Forecastable: <strong>{horizonView === '1h' ? network1h?.forecasted_stations_count : network24h?.forecasted_stations_count}</strong> | Insufficient History:{' '}
            <strong>{horizonView === '1h' ? network1h?.insufficient_history_count : network24h?.insufficient_history_count}</strong>
          </span>
        </div>

        <div className="overflow-x-auto max-h-96 overflow-y-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead className="sticky top-0 bg-white z-10">
              <tr className="border-b border-slate-200 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                <th className="py-2.5 pr-3">Station</th>
                <th className="py-2.5 px-3">Status</th>
                {horizonView === '1h' ? (
                  <>
                    <th className="py-2.5 px-3">Target Hour (PT)</th>
                    <th className="py-2.5 px-3 text-right">Predicted (1h)</th>
                    <th className="py-2.5 px-3 text-right">Hist. Hour Avg</th>
                  </>
                ) : (
                  <>
                    <th className="py-2.5 px-3">Peak Predicted Hour</th>
                    <th className="py-2.5 px-3 text-right">24h Total Predicted</th>
                    <th className="py-2.5 px-3 text-right">Peak Hr Rate</th>
                  </>
                )}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {(horizonView === '1h' ? network1h?.predictions : network24h?.predictions || []).map((row) => {
                const isSelected = row.station_id === selectedStation;
                return (
                  <tr
                    key={row.station_id}
                    onClick={() => setSelectedStation(row.station_id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-emerald-50/80' : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="py-2.5 pr-3 font-medium text-slate-900">
                      {row.short_name}
                    </td>
                    <td className="py-2.5 px-3">
                      <StationBadge isColdStart={row.is_cold_start} size="xs" />
                    </td>
                    {horizonView === '1h' ? (
                      <>
                        <td className="py-2.5 px-3 text-xs text-slate-600">
                          {row.target_timestamp_pt.slice(0, 16)}
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-emerald-700">
                          {row.predicted_sessions.toFixed(4)}{' '}
                          <span className="font-normal text-xs text-slate-400">
                            (~{row.predicted_sessions_rounded})
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right text-slate-600">
                          {row.historical_hour_avg.toFixed(4)}
                        </td>
                      </>
                    ) : (
                      <>
                        <td className="py-2.5 px-3 text-xs text-slate-600">
                          {String(row.peak_predicted_hour).padStart(2, '0')}:00 ({row.peak_predicted_hour_pt.slice(0, 10)})
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold text-indigo-700">
                          {row.total_predicted_sessions_24h.toFixed(2)}{' '}
                          <span className="font-normal text-xs text-slate-400">
                            (~{row.total_predicted_sessions_24h_rounded})
                          </span>
                        </td>
                        <td className="py-2.5 px-3 text-right text-slate-600">
                          {row.peak_predicted_sessions.toFixed(3)}/hr
                        </td>
                      </>
                    )}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
