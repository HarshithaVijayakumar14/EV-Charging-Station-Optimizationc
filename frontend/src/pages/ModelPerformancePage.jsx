import React, { useEffect, useState } from 'react';
import {
  Award,
  BarChart3,
  ShieldCheck,
  AlertTriangle,
  BookOpen,
  Layers,
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts';
import { getModelPerformance } from '../services/api';
import { LoadingSpinner, ErrorState } from '../components/LoadingSpinner';

const MODEL_LABELS = {
  xgboost: 'XGBoost Regressor (Selected)',
  random_forest: 'Random Forest Regressor',
  ridge_regression: 'Ridge Regression',
  linear_regression: 'Linear Regression',
  historical_mean: 'Historical Mean Baseline',
  lag_24h: 'Persistence (Lag 24h)',
  lag_168h: 'Weekly Persistence (Lag 168h)',
  global_constant_mean_ref: 'Global Constant Mean Ref',
};

export function ModelPerformancePage() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [horizon, setHorizon] = useState('1h'); // '1h' | '24h'
  const [splitKey, setSplitKey] = useState('test_2019'); // 'validation_2018' | 'test_2019' | 'ood_2020'

  const loadData = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await getModelPerformance();
      setData(res);
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Failed to load Phase 5 evaluation artifacts.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, []);

  if (loading) return <LoadingSpinner message="Loading Phase 5 chronological model evaluation artifacts..." />;
  if (error) return <ErrorState message={error} onRetry={loadData} />;
  if (!data) return null;

  const section = horizon === '1h' ? data.primary_1h : data.secondary_24h;
  const splitModelsObj = section?.[splitKey] || {};
  const modelRows = Object.entries(splitModelsObj)
    .map(([key, val]) => ({
      key,
      label: MODEL_LABELS[key] || key.replace(/_/g, ' '),
      isWinner: key === 'xgboost',
      overall: val?.overall || val,
      weekday: val?.weekday,
      weekend: val?.weekend,
      historical: val?.historical_stations,
      newSt: val?.new_stations,
    }))
    .filter((r) => r.overall && typeof r.overall.rmse === 'number')
    .sort((a, b) => a.overall.rmse - b.overall.rmse);

  const featureImp = (horizon === '1h' ? data.feature_importance_1h : data.feature_importance_24h) || [];
  const topFeaturesChart = featureImp.slice(0, 12).map((f) => ({
    feature: f.feature,
    rmse_increase: Number((f.perm_rmse_increase_mean * 1000).toFixed(3)), // scaled x1000 for readable axis
    raw_rmse_increase: f.perm_rmse_increase_mean,
    native_pct: f.native_importance_pct,
  }));

  const calibration2019 = data.primary_1h?.hourly_calibration_2019_winner
    ? Object.entries(data.primary_1h.hourly_calibration_2019_winner)
        .map(([hr, v]) => ({
          hour: Number(hr),
          label: `${String(hr).padStart(2, '0')}:00`,
          actual_mean: v.actual_mean,
          pred_mean: v.pred_mean,
        }))
        .sort((a, b) => a.hour - b.hour)
    : [];

  return (
    <div className="space-y-6">
      {/* Winner Summary & Split Discipline */}
      <div className="grid grid-cols-1 xl:grid-cols-3 gap-6">
        <div className="xl:col-span-2 bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-2">
            <Award className="w-5 h-5 text-emerald-600" />
            <h2 className="text-base font-bold text-slate-900">
              Selected Production Models (Phase 5 Chronological Evaluation)
            </h2>
          </div>
          <p className="text-xs text-slate-600 leading-relaxed">
            {data.selected_model.selection_criterion}
          </p>

          <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div className="p-3.5 rounded-lg bg-emerald-50/70 border border-emerald-200">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-emerald-800">
                1-Hour-Ahead Winner
              </span>
              <p className="mt-1 text-lg font-bold text-slate-900 uppercase">
                {data.selected_model.primary_1h_winner}
              </p>
              <p className="text-xs text-emerald-900 mt-1">
                Test 2019 RMSE: <strong>{data.primary_1h.test_2019?.xgboost?.overall?.rmse?.toFixed(5)}</strong> | R²:{' '}
                <strong>{data.primary_1h.test_2019?.xgboost?.overall?.r2?.toFixed(4)}</strong>
              </p>
            </div>
            <div className="p-3.5 rounded-lg bg-indigo-50/70 border border-indigo-200">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-indigo-800">
                24-Hour-Ahead Winner
              </span>
              <p className="mt-1 text-lg font-bold text-slate-900 uppercase">
                {data.selected_model.secondary_24h_winner}
              </p>
              <p className="text-xs text-indigo-900 mt-1">
                Test 2019 RMSE: <strong>{data.secondary_24h.test_2019?.xgboost?.overall?.rmse?.toFixed(5)}</strong> | R²:{' '}
                <strong>{data.secondary_24h.test_2019?.xgboost?.overall?.r2?.toFixed(4)}</strong>
              </p>
            </div>
          </div>

          <div className="mt-4 p-3 rounded-lg bg-amber-50 border border-amber-200 text-xs text-amber-900 flex items-start gap-2">
            <AlertTriangle className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
            <span>
              <strong>2020 OOD Stress Test Note:</strong> {data.selected_model.ood_2020_note}
            </span>
          </div>
        </div>

        {/* Chronological Split Summary */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-3">
            <ShieldCheck className="w-4 h-4 text-emerald-600" />
            <h2 className="text-base font-bold text-slate-900">Anti-Leakage Split Design</h2>
          </div>
          <div className="space-y-2.5 text-xs">
            {Object.entries(data.splits).map(([k, desc]) => (
              <div key={k} className="p-2.5 rounded-lg bg-slate-50 border border-slate-100">
                <span className="block font-bold uppercase text-[10px] tracking-wider text-slate-500">
                  {k.replace(/_/g, ' ')}
                </span>
                <span className="text-slate-800 font-medium">{desc}</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Model Comparison Table */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Layers className="w-4 h-4 text-indigo-600" />
              Actual Saved Benchmark Comparison
            </h2>
            <p className="text-xs text-slate-500">
              Sorted by lowest RMSE (all values read directly from Phase 5 evaluation artifacts)
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <div className="inline-flex rounded-lg bg-slate-100 p-1 border border-slate-200">
              <button
                onClick={() => setHorizon('1h')}
                className={`px-3 py-1 rounded-md text-xs font-semibold ${
                  horizon === '1h' ? 'bg-white text-emerald-700 shadow-xs' : 'text-slate-600'
                }`}
              >
                1-Hour-Ahead
              </button>
              <button
                onClick={() => setHorizon('24h')}
                className={`px-3 py-1 rounded-md text-xs font-semibold ${
                  horizon === '24h' ? 'bg-white text-indigo-700 shadow-xs' : 'text-slate-600'
                }`}
              >
                24-Hour-Ahead
              </button>
            </div>

            <div className="inline-flex rounded-lg bg-slate-100 p-1 border border-slate-200">
              {[
                { id: 'validation_2018', label: 'Val 2018' },
                { id: 'test_2019', label: 'Main Test 2019' },
                { id: 'ood_2020', label: 'OOD 2020' },
              ].map((sp) => (
                <button
                  key={sp.id}
                  onClick={() => setSplitKey(sp.id)}
                  className={`px-3 py-1 rounded-md text-xs font-semibold ${
                    splitKey === sp.id ? 'bg-slate-900 text-white shadow-xs' : 'text-slate-600'
                  }`}
                >
                  {sp.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-sm">
            <thead>
              <tr className="border-b border-slate-200 text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                <th className="py-2.5 pr-3">Model</th>
                <th className="py-2.5 px-3 text-right">RMSE ↓</th>
                <th className="py-2.5 px-3 text-right">MAE ↓</th>
                <th className="py-2.5 px-3 text-right">R² ↑</th>
                <th className="py-2.5 px-3 text-right">Bias %</th>
                <th className="py-2.5 px-3 text-right">Weekday RMSE</th>
                <th className="py-2.5 pl-3 text-right">Weekend RMSE</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {modelRows.map((row) => (
                <tr
                  key={row.key}
                  className={row.isWinner ? 'bg-emerald-50/70 font-semibold' : 'hover:bg-slate-50'}
                >
                  <td className="py-2.5 pr-3 text-slate-900 flex items-center gap-2">
                    {row.label}
                    {row.isWinner && (
                      <span className="text-[10px] font-bold uppercase px-1.5 py-0.5 rounded bg-emerald-600 text-white">
                        Winner
                      </span>
                    )}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-900">
                    {row.overall.rmse.toFixed(5)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-700">
                    {row.overall.mae.toFixed(5)}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-700">
                    {row.overall.r2 !== null && row.overall.r2 !== undefined
                      ? row.overall.r2.toFixed(4)
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                    {row.overall.bias_pct !== null && row.overall.bias_pct !== undefined
                      ? `${row.overall.bias_pct > 0 ? '+' : ''}${row.overall.bias_pct.toFixed(2)}%`
                      : '—'}
                  </td>
                  <td className="py-2.5 px-3 text-right font-mono text-slate-600">
                    {row.weekday?.rmse ? row.weekday.rmse.toFixed(4) : '—'}
                  </td>
                  <td className="py-2.5 pl-3 text-right font-mono text-slate-600">
                    {row.weekend?.rmse ? row.weekend.rmse.toFixed(4) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Charts Row: Feature Importance + 2019 Hourly Calibration */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Feature Permutation Importance */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <div className="flex items-center gap-2 mb-1">
            <BarChart3 className="w-4 h-4 text-emerald-600" />
            <h2 className="text-base font-bold text-slate-900">
              Top 12 Feature Importance ({horizon === '1h' ? '1-Hour' : '24-Hour'} XGBoost)
            </h2>
          </div>
          <p className="text-xs text-slate-500 mb-4">
            Measured by permutation RMSE increase (×1,000) on held-out 2019 Test set
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart
                data={topFeaturesChart}
                layout="vertical"
                margin={{ top: 5, right: 20, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#64748b' }} />
                <YAxis
                  type="category"
                  dataKey="feature"
                  width={175}
                  tick={{ fontSize: 10, fill: '#334155' }}
                />
                <Tooltip
                  contentStyle={{
                    borderRadius: '0.75rem',
                    border: '1px solid #e2e8f0',
                    fontSize: '12px',
                  }}
                  formatter={(v, _name, item) => [
                    `+${item.payload.raw_rmse_increase.toFixed(6)} RMSE (Native: ${item.payload.native_pct}%)`,
                    'Permutation Impact',
                  ]}
                />
                <Bar
                  dataKey="rmse_increase"
                  name="RMSE Increase (×1000)"
                  fill={horizon === '1h' ? '#059669' : '#4f46e5'}
                  radius={[0, 4, 4, 0]}
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Test 2019 Hourly Calibration */}
        <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
          <h2 className="text-base font-bold text-slate-900">
            Test 2019 Hourly Calibration (1-Hour XGBoost)
          </h2>
          <p className="text-xs text-slate-500 mb-4">
            Mean actual vs. mean predicted session arrivals by clock hour across 289,080 unseen 2019 station-hours
          </p>
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={calibration2019} margin={{ top: 10, right: 16, left: -10, bottom: 0 }}>
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
                <Line
                  type="monotone"
                  dataKey="actual_mean"
                  name="Actual 2019 Mean Arrivals/Hr"
                  stroke="#0f172a"
                  strokeWidth={2.5}
                  dot={{ r: 3 }}
                />
                <Line
                  type="monotone"
                  dataKey="pred_mean"
                  name="XGBoost Predicted Mean/Hr"
                  stroke="#10b981"
                  strokeWidth={2}
                  strokeDasharray="5 3"
                  dot={{ r: 2.5 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Plain-Language Metric Glossary */}
      <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm">
        <div className="flex items-center gap-2 mb-3">
          <BookOpen className="w-4 h-4 text-slate-700" />
          <h2 className="text-base font-bold text-slate-900">How to Interpret These Metrics</h2>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs text-slate-600 leading-relaxed">
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-100">
            <h3 className="font-bold text-slate-900 mb-1">RMSE &amp; MAE (Lower is Better)</h3>
            <p>
              <strong>MAE</strong> is the average absolute difference between predicted and actual
              session arrivals in a station-hour. <strong>RMSE</strong> penalizes larger errors during
              peak commuter surges more strongly, making it our primary selection metric.
            </p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-100">
            <h3 className="font-bold text-slate-900 mb-1">Why R² ≈ 0.30–0.32 at Hourly Station Level</h3>
            <p>
              Over 85% of individual station-hours have zero arrivals (sparse Poisson-like counts). Even an
              optimal expected-value predictor faces irreducible arrival randomness at a single charger in a
              single hour, while matching hourly averages with near-zero calibration bias.
            </p>
          </div>
          <div className="p-3.5 rounded-lg bg-slate-50 border border-slate-100">
            <h3 className="font-bold text-slate-900 mb-1">Decimal vs. Rounded Predictions</h3>
            <p>
              An output of <strong>0.68 arrivals/hr</strong> represents the expected Poisson arrival rate.
              Preserving decimal rates allows accurate ranking of high- vs. low-demand sites and unbiased
              24-hour sums, while rounded integers offer quick operational intuition.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
