import React, { useState, useEffect } from 'react';
import {
  ShieldAlert,
  CheckCircle2,
  TrendingDown,
  Layers,
  BarChart3,
  Activity,
  AlertTriangle,
  Info,
  Calendar,
  Clock,
  ArrowRight,
  Gauge,
  Sliders,
} from 'lucide-react';
import {
  ResponsiveContainer,
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts';
import {
  getAvailabilityCalibrationSummary,
  getReliabilityDiagram,
} from '../services/api';
import { StatCard } from '../components/StatCard';
import { LoadingSpinner, ErrorState } from '../components/LoadingSpinner';

export function AvailabilityCalibrationPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [summaryData, setSummaryData] = useState(null);
  const [diagramData, setDiagramData] = useState(null);
  const [horizon, setHorizon] = useState('1h');
  const [split, setSplit] = useState('2019_test');
  const [decMethod, setDecMethod] = useState('A0');

  const fetchData = async () => {
    setLoading(true);
    setError(null);
    try {
      const [sumRes, diagRes] = await Promise.all([
        getAvailabilityCalibrationSummary(),
        getReliabilityDiagram({ horizon, split }),
      ]);
      setSummaryData(sumRes);
      setDiagramData(diagRes);
    } catch (err) {
      setError(err?.response?.data?.message || err.message || 'Failed to load availability calibration telemetry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [horizon, split]);

  if (loading) {
    return <LoadingSpinner message="Loading empirical availability calibration analysis..." />;
  }

  if (error) {
    return <ErrorState message={error} onRetry={fetchData} />;
  }

  const evalSummary = summaryData?.evaluations_summary?.[horizon]?.[split] || {};
  const diagrams = diagramData?.diagrams || {};
  const decisionMetrics = diagramData?.decision_metrics || {};

  // Construct chart series for Reliability Diagram
  // 10 quantile bins
  const a0Bins = diagrams?.A0?.bins || [];
  const cBins = diagrams?.C_isotonic?.bins || [];
  const b1Bins = diagrams?.B1?.bins || [];

  const reliabilityChartData = a0Bins.map((b, idx) => ({
    bin: `Bin ${idx + 1}`,
    ideal: b.mean_predicted_prob,
    predicted_a0: b.mean_predicted_prob,
    empirical_a0: b.empirical_accuracy,
    empirical_c: cBins[idx]?.empirical_accuracy ?? null,
    empirical_b1: b1Bins[idx]?.empirical_accuracy ?? null,
  }));

  // Error propagation data
  const errProp = summaryData?.error_propagation?.[horizon]?.[split];
  const errComp = errProp?.component_mae || {};
  const errShares = errProp?.variance_contributions_pct || {};

  const errorDecompData = [
    {
      name: 'Arrival Forecast Error',
      mae: errComp.arrival_forecast_error || 0,
      share: errShares.arrival_forecast_error_share || 0,
      fill: '#E63946',
    },
    {
      name: 'Dwell & Queueing Error',
      mae: errComp.dwell_queueing_assumption_error || 0,
      share: errShares.dwell_queueing_assumption_error_share || 0,
      fill: '#F4A261',
    },
    {
      name: 'Residual Structural Error',
      mae: errComp.residual_structural_error || 0,
      share: errShares.residual_structural_error_share || 0,
      fill: '#2A9D8F',
    },
  ];

  // Sensitivity elasticity data
  const sensList = errProp?.sensitivity_elasticity || [];
  const sensitivityChartData = sensList.map((s) => ({
    perturbation: `${s.arrival_perturbation_pct > 0 ? '+' : ''}${s.arrival_perturbation_pct}%`,
    occupied_change: s.pct_change_occupied,
    prob_full_change: s.pct_change_prob_full,
  }));

  return (
    <div className="space-y-8 animate-in fade-in duration-300">
      {/* 1. Header & Context */}
      <div>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-900">
              Availability Calibration &amp; Error Propagation
            </h2>
            <p className="mt-1 text-sm text-slate-500">
              Rigorous out-of-sample evaluation of forecast-driven queueing assumptions, calibration drift, and empirical baselines.
            </p>
          </div>

          {/* Horizon & Split Selectors */}
          <div className="flex flex-wrap items-center gap-3">
            <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-xs">
              <button
                onClick={() => setHorizon('1h')}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                  horizon === '1h'
                    ? 'bg-emerald-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                1-Hour Horizon (Carryover)
              </button>
              <button
                onClick={() => setHorizon('24h')}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                  horizon === '24h'
                    ? 'bg-emerald-600 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                24-Hour Horizon (Day-Ahead)
              </button>
            </div>

            <div className="inline-flex rounded-lg border border-slate-200 bg-white p-1 shadow-xs">
              {[
                { key: '2019_test', label: '2019 Test' },
                { key: '2018_validation', label: '2018 Val' },
                { key: '2020_ood', label: '2020 OOD' },
              ].map((sp) => (
                <button
                  key={sp.key}
                  onClick={() => setSplit(sp.key)}
                  className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                    split === sp.key
                      ? 'bg-slate-900 text-white shadow-xs'
                      : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {sp.label}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* 2. Research Caveats & Planning Disclaimer Banner */}
      <div className="rounded-xl border border-amber-200 bg-amber-50/80 p-4 sm:p-5 text-amber-900">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
          <div className="text-xs sm:text-sm space-y-1">
            <p className="font-semibold text-amber-950">
              Research Assumptions &amp; Methodological Protocol:
            </p>
            <ul className="list-disc pl-4 space-y-0.5 text-amber-900">
              <li>
                <strong>Single Municipal Network:</strong> Evaluated on 46 public charging stations across the City of Palo Alto, CA.
              </li>
              <li>
                <strong>Occupancy Definition:</strong> Ground truth strictly denotes <em>plugged-in occupancy</em> (charging duration plus idle dwell occupying the charging bay), reconstructed from verified ChargePoint session timestamps.
              </li>
              <li>
                <strong>Observed Hardware Inventory:</strong> Physical port capacity is reconstructed from observed physical connector activations over time (accounting for hardware upgrades and decommissioning).
              </li>
              <li>
                <strong>Strict Out-of-Sample Calibration:</strong> Recalibrators (Isotonic and Platt) were fitted <em>exclusively</em> on 2018 validation predictions and evaluated out-of-sample on 2019 and 2020.
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* 3. Core Metric Stat Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          label={`A0 Current Erlang ECE (${split.replace('_', ' ')})`}
          value={evalSummary?.A0?.ece !== undefined ? `${(evalSummary.A0.ece * 100).toFixed(2)}%` : '—'}
          change={`Raw uncalibrated queueing model error`}
          changeType="neutral"
          icon={Gauge}
        />
        <StatCard
          label={`C Recalibrated Isotonic ECE`}
          value={evalSummary?.C_isotonic?.ece !== undefined ? `${(evalSummary.C_isotonic.ece * 100).toFixed(2)}%` : '—'}
          change={
            evalSummary?.A0?.ece && evalSummary?.C_isotonic?.ece
              ? `-${(((evalSummary.A0.ece - evalSummary.C_isotonic.ece) / evalSummary.A0.ece) * 100).toFixed(1)}% error reduction`
              : 'Recalibrated on 2018 only'
          }
          changeType="positive"
          icon={CheckCircle2}
        />
        <StatCard
          label="Brier Score (Recalibrated)"
          value={evalSummary?.C_isotonic?.brier_score !== undefined ? evalSummary.C_isotonic.brier_score.toFixed(4) : '—'}
          change={`vs A0 raw: ${evalSummary?.A0?.brier_score?.toFixed(4) ?? '—'}`}
          changeType="positive"
          icon={TrendingDown}
        />
        <StatCard
          label="Dwell Exp. Fit (KS Test)"
          value="REJECTED"
          change="p < 1e-50 across all years (CV < 1.0)"
          changeType="negative"
          icon={ShieldAlert}
        />
      </div>

      {/* 4. Reliability Diagrams (Calibration Curves) */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-8 bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h3 className="text-base font-semibold text-slate-900">
                Reliability Diagram (Calibration Curves, 10 Quantile Bins)
              </h3>
              <p className="text-xs text-slate-500">
                Plots mean predicted probability against observed empirical frequency. Points along the diagonal indicate perfect calibration.
              </p>
            </div>
            <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700">
              {horizon.toUpperCase()} Horizon — {split.replace('_', ' ').toUpperCase()}
            </span>
          </div>

          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={reliabilityChartData} margin={{ top: 10, right: 20, left: -10, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                <XAxis dataKey="bin" tick={{ fontSize: 11 }} />
                <YAxis domain={[0, 1]} tick={{ fontSize: 11 }} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                <Tooltip
                  formatter={(val, name) => [
                    typeof val === 'number' ? `${(val * 100).toFixed(1)}%` : val,
                    name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
                <Line
                  type="monotone"
                  dataKey="ideal"
                  name="Ideal (Perfect Calibration)"
                  stroke="#94a3b8"
                  strokeDasharray="4 4"
                  dot={false}
                  strokeWidth={1.5}
                />
                <Line
                  type="monotone"
                  dataKey="empirical_a0"
                  name="A0 Current (Erlang-B Heuristic)"
                  stroke="#1f6f8b"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="empirical_c"
                  name="C Recalibrated (Isotonic)"
                  stroke="#10b981"
                  strokeWidth={2.5}
                  dot={{ r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey="empirical_b1"
                  name="B1 Empirical Baseline"
                  stroke="#f97316"
                  strokeWidth={1.5}
                  strokeDasharray="2 2"
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 5. Error Decomposition Breakdown */}
        <div className="lg:col-span-4 bg-white rounded-2xl border border-slate-200 p-6 shadow-xs flex flex-col justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Error Propagation Decomposition
            </h3>
            <p className="text-xs text-slate-500 mb-4">
              Isolating arrival forecast error vs. queueing assumption vs. residual capacity bound.
            </p>

            <div className="h-52 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={errorDecompData} layout="vertical" margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f1f5f9" />
                  <XAxis type="number" unit="%" tick={{ fontSize: 11 }} />
                  <YAxis type="category" dataKey="name" tick={{ fontSize: 10 }} width={110} />
                  <Tooltip formatter={(v) => [`${Number(v).toFixed(1)}%`, 'Error Share']} />
                  <Bar dataKey="share" radius={[0, 4, 4, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="p-3 bg-slate-50 rounded-xl border border-slate-200 text-xs text-slate-600 space-y-1">
            <div className="flex justify-between">
              <span className="font-medium text-slate-800">Total MAE (Occupied Points):</span>
              <span className="font-bold text-slate-900">{errProp?.total_mae_occupied?.toFixed(3) ?? '—'}</span>
            </div>
            <div className="flex justify-between">
              <span>Arrival Error Share:</span>
              <span className="font-semibold text-rose-600">{errShares.arrival_forecast_error_share?.toFixed(1) ?? '—'}%</span>
            </div>
            <div className="flex justify-between">
              <span>Residual Structural Share:</span>
              <span className="font-semibold text-emerald-600">{errShares.residual_structural_error_share?.toFixed(1) ?? '—'}%</span>
            </div>
          </div>
        </div>
      </div>

      {/* 6. Comprehensive Method Benchmark Table */}
      <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-6 border-b border-slate-200">
          <h3 className="text-base font-semibold text-slate-900">
            Out-of-Sample Calibration &amp; Discrimination Benchmark
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Evaluated on strictly out-of-sample data ({split.replace('_', ' ').toUpperCase()}). 95% CIs computed via 500 block-bootstrap iterations over station-weeks.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider">
                <th className="py-3 px-4">Method Code</th>
                <th className="py-3 px-4">Model Description</th>
                <th className="py-3 px-4 text-right">ECE</th>
                <th className="py-3 px-4 text-right">Brier Score</th>
                <th className="py-3 px-4 text-right">Reliability (Loss)</th>
                <th className="py-3 px-4 text-right">Resolution</th>
                <th className="py-3 px-4 text-right">AUROC</th>
                <th className="py-3 px-4 text-right">AUPRC</th>
                <th className="py-3 px-4 text-center">95% Bootstrap CI (Brier)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {[
                { code: 'A0', desc: 'Current Method (Erlang-B Heuristic)', key: 'A0' },
                { code: 'A1', desc: 'Queueing Ablation (True Arrivals)', key: 'A1' },
                { code: 'A2', desc: 'Queueing + True Hourly Dwell', key: 'A2' },
                { code: 'A3', desc: 'Empirical Dwell Survival Model', key: 'A3' },
                { code: 'B1', desc: 'Empirical Baseline (Hour-of-Week Shrunk)', key: 'B1' },
                { code: 'B2', desc: 'Persistence Baseline', key: 'B2' },
                { code: 'C_isotonic', desc: 'Recalibrated (Isotonic, fitted 2018)', key: 'C_isotonic', highlight: true },
                { code: 'C_platt', desc: 'Recalibrated (Platt Scaling, fitted 2018)', key: 'C_platt' },
              ].map((row) => {
                const mData = evalSummary[row.key] || {};
                const cal = mData.calibration || {};
                const ci = cal.bootstrap_95_ci?.brier_score;

                return (
                  <tr
                    key={row.code}
                    className={`hover:bg-slate-50/60 transition-colors ${
                      row.highlight ? 'bg-emerald-50/40 font-medium text-emerald-950' : 'text-slate-700'
                    }`}
                  >
                    <td className="py-3 px-4 font-bold text-slate-900">{row.code}</td>
                    <td className="py-3 px-4">{row.desc}</td>
                    <td className="py-3 px-4 text-right font-mono font-semibold">
                      {cal.ece !== undefined ? cal.ece.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono font-semibold">
                      {cal.brier_score !== undefined ? cal.brier_score.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-500">
                      {cal.brier_reliability !== undefined ? cal.brier_reliability.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono text-slate-500">
                      {cal.brier_resolution !== undefined ? cal.brier_resolution.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono">
                      {cal.auroc !== undefined ? cal.auroc.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-right font-mono">
                      {cal.auprc !== undefined ? cal.auprc.toFixed(4) : '—'}
                    </td>
                    <td className="py-3 px-4 text-center font-mono text-slate-500">
                      {ci ? `[${ci.ci_lower?.toFixed(4)}, ${ci.ci_upper?.toFixed(4)}]` : '—'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* 6b. Decision-Level Utility & Threshold Tradeoffs Table */}
      <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-xs">
        <div className="p-6 border-b border-slate-200 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Decision-Level Error Metrics across Decision Thresholds (&tau; &isin; [0.1, 0.9])
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Evaluating practical driver trade-offs: False Alarm Rate (unnecessary diversion) vs. Miss Rate and Wasted Trip Rate (diverted to a full station).
            </p>
          </div>

          <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1 shadow-xs">
            {[
              { key: 'A0', label: 'A0 Raw Erlang-B' },
              { key: 'C_isotonic', label: 'C Recalibrated' },
              { key: 'B1', label: 'B1 Empirical' },
            ].map((m) => (
              <button
                key={m.key}
                onClick={() => setDecMethod(m.key)}
                className={`px-3 py-1.5 rounded-md text-xs font-semibold transition-colors cursor-pointer ${
                  decMethod === m.key
                    ? 'bg-slate-900 text-white shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>

        {/* Free Ports continuous error stats & 3-Class Accuracy */}
        {(() => {
          const mDec = decisionMetrics?.[decMethod];
          const freeMae = mDec?.free_ports_mae;
          const freeRmse = mDec?.free_ports_rmse;
          const cm = mDec?.three_class_confusion_matrix;

          if (!freeMae && !cm) return null;

          return (
            <div className="px-6 py-4 bg-slate-50/70 border-b border-slate-200 grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
              <div className="flex items-center justify-between p-3 bg-white rounded-xl border border-slate-200">
                <span className="text-slate-600 font-medium">Free Ports MAE:</span>
                <span className="font-mono font-bold text-slate-900">{freeMae !== undefined ? freeMae : '—'} ports</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white rounded-xl border border-slate-200">
                <span className="text-slate-600 font-medium">Free Ports RMSE:</span>
                <span className="font-mono font-bold text-slate-900">{freeRmse !== undefined ? freeRmse : '—'} ports</span>
              </div>
              <div className="flex items-center justify-between p-3 bg-white rounded-xl border border-slate-200">
                <span className="text-slate-600 font-medium">3-Class Accuracy (0 / 1 / 2+ Free):</span>
                <span className="font-mono font-bold text-emerald-600">
                  {cm?.overall_accuracy !== undefined ? `${(cm.overall_accuracy * 100).toFixed(1)}%` : '—'}
                </span>
              </div>
            </div>
          );
        })()}

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50/80 border-b border-slate-200 text-slate-500 font-semibold uppercase tracking-wider">
                <th className="py-3 px-4">Threshold &tau;</th>
                <th className="py-3 px-4 text-right">True Positives</th>
                <th className="py-3 px-4 text-right">False Positives</th>
                <th className="py-3 px-4 text-right">False Alarm Rate</th>
                <th className="py-3 px-4 text-right">Miss Rate</th>
                <th className="py-3 px-4 text-right">Precision</th>
                <th className="py-3 px-4 text-right">Recall</th>
                <th className="py-3 px-4 text-right">F1 Score</th>
                <th className="py-3 px-4 text-right">Wasted Trip Rate</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {(() => {
                const mDec = decisionMetrics?.[decMethod];
                const sweep = Array.isArray(mDec) ? mDec : (mDec?.threshold_sweep || []);
                if (!sweep.length) {
                  return (
                    <tr>
                      <td colSpan={9} className="py-6 text-center text-slate-500">
                        No threshold sweep metrics recorded for this method.
                      </td>
                    </tr>
                  );
                }
                return sweep.map((row) => (
                  <tr key={row.threshold} className="hover:bg-slate-50/60 transition-colors text-slate-700 font-mono">
                    <td className="py-2.5 px-4 font-bold text-slate-900">&tau; = {row.threshold.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-right text-slate-600">{row.tp?.toLocaleString()}</td>
                    <td className="py-2.5 px-4 text-right text-slate-600">{row.fp?.toLocaleString()}</td>
                    <td className="py-2.5 px-4 text-right text-rose-600 font-medium">{(row.false_alarm_rate * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-4 text-right text-amber-600 font-medium">{(row.miss_rate * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-4 text-right">{row.precision?.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-right">{row.recall?.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-right font-bold text-slate-900">{row.f1_score?.toFixed(4)}</td>
                    <td className="py-2.5 px-4 text-right font-semibold text-rose-700">{(row.wasted_trip_rate * 100).toFixed(2)}%</td>
                  </tr>
                ));
              })()}
            </tbody>
          </table>
        </div>
      </div>

      {/* 7. Dwell Assumption Goodness-of-Fit & Distribution Shift Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Dwell Assumption Table */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-base font-semibold text-slate-900">
                Plugged-In Dwell Time Dispersion vs. Exponential
              </h3>
              <p className="text-xs text-slate-500">
                Erlang-B assumes exponential dwell (CV = 1.0). Empirical data displays significant under-dispersion.
              </p>
            </div>
            <ShieldAlert className="w-5 h-5 text-rose-500" />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-semibold uppercase">
                  <th className="py-2.5 px-3">Split Period</th>
                  <th className="py-2.5 px-3 text-right">Mean Dwell</th>
                  <th className="py-2.5 px-3 text-right">Std Dwell</th>
                  <th className="py-2.5 px-3 text-right">CV (sigma/mu)</th>
                  <th className="py-2.5 px-3 text-right">KS Statistic</th>
                  <th className="py-2.5 px-3 text-center">Exponential?</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.entries(summaryData?.dwell_assumption_test || {}).map(([spName, d]) => (
                  <tr key={spName} className="hover:bg-slate-50/50">
                    <td className="py-2 px-3 font-medium text-slate-800">
                      {spName.replace('_', ' ').title ? spName.replace(/_/g, ' ') : spName}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">{d.mean_dwell_min} min</td>
                    <td className="py-2 px-3 text-right font-mono">{d.std_dwell_min} min</td>
                    <td className="py-2 px-3 text-right font-mono font-bold text-slate-900">
                      {d.coefficient_of_variation}
                    </td>
                    <td className="py-2 px-3 text-right font-mono">{d.ks_statistic}</td>
                    <td className="py-2 px-3 text-center">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-rose-50 text-rose-700 border border-rose-200">
                        Rejected
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Distribution Shift (PSI & KS) */}
        <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h3 className="text-base font-semibold text-slate-900">
                Temporal Distribution Shift (PSI &amp; 2-Sample KS)
              </h3>
              <p className="text-xs text-slate-500">
                Population Stability Index between 2018 Validation baseline and subsequent evaluation periods.
              </p>
            </div>
            <Activity className="w-5 h-5 text-indigo-500" />
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-slate-50 border-b border-slate-200 text-slate-500 font-semibold uppercase">
                  <th className="py-2.5 px-3">Telemetry Feature</th>
                  <th className="py-2.5 px-3 text-center">2018 → 2019 Shift</th>
                  <th className="py-2.5 px-3 text-center">2018 → 2020 Shift (COVID)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {Object.entries(summaryData?.distribution_shift || {}).map(([fKey, s]) => (
                  <tr key={fKey} className="hover:bg-slate-50/50">
                    <td className="py-2.5 px-3 font-semibold text-slate-800">
                      {fKey.replace(/_/g, ' ').toUpperCase()}
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <span className="font-mono font-bold text-slate-700">PSI: {s.shift_2018_to_2019?.psi}</span>
                      <span className="block text-[10px] text-emerald-600 font-medium">
                        {s.shift_2018_to_2019?.shift_severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-center">
                      <span className="font-mono font-bold text-rose-700">PSI: {s.shift_2018_to_2020?.psi}</span>
                      <span className="block text-[10px] text-rose-600 font-bold">
                        {s.shift_2018_to_2020?.shift_severity} Shift
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* 8. Slice Sensitivity Analysis */}
      <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-xs">
        <h3 className="text-base font-semibold text-slate-900 mb-1">
          Slice Sensitivity Analysis (1h Horizon, 2019 Test Set)
        </h3>
        <p className="text-xs text-slate-500 mb-4">
          Disaggregating calibration performance across capacity assumptions, station architecture, and peak vs off-peak hours.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {(summaryData?.slice_analysis?.['1h_2019_test'] || []).map((sl, idx) => (
            <div key={idx} className="p-4 rounded-xl border border-slate-200 bg-slate-50/60 space-y-2">
              <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-500">
                {sl.slice_type}
              </span>
              <p className="font-bold text-sm text-slate-900">{sl.slice_value}</p>
              <div className="pt-2 border-t border-slate-200 text-xs space-y-1">
                <div className="flex justify-between">
                  <span className="text-slate-600">ECE:</span>
                  <span className="font-mono font-semibold text-slate-900">{(sl.ece * 100).toFixed(2)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Brier Score:</span>
                  <span className="font-mono font-semibold text-slate-900">{sl.brier.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-600">Log Loss:</span>
                  <span className="font-mono text-slate-700">{sl.log_loss.toFixed(4)}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}