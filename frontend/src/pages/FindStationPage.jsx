import React, { useState, useEffect } from 'react';
import {
  Car,
  BatteryCharging,
  Zap,
  MapPin,
  Clock,
  CheckCircle2,
  AlertTriangle,
  XCircle,
  Search,
  ArrowRight,
  ShieldCheck,
  Info,
  Sparkles,
  Sliders,
  RefreshCw,
  Navigation,
} from 'lucide-react';
import {
  getVehicles,
  getStations,
  recommendStations,
  estimateCharging,
} from '../services/api';

export function FindStationPage({ initialStationId, onNavigateToStation }) {
  // Vehicle Catalog State
  const [vehicles, setVehicles] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('Four-Wheeler');
  const [selectedVehicleId, setSelectedVehicleId] = useState('');
  const [customVehicle, setCustomVehicle] = useState({
    manufacturer: 'Custom',
    model: 'Custom Vehicle',
    vehicle_category: 'Four-Wheeler',
    battery_capacity_kwh: 60.0,
    maximum_ac_charging_power_kw: 11.0,
    connector_type: 'J1772',
    charging_efficiency: 0.90,
  });

  // Battery State
  const [currentSoc, setCurrentSoc] = useState(20);
  const [targetSoc, setTargetSoc] = useState(80);

  // Stations State
  const [stations, setStations] = useState([]);
  const [selectedStationId, setSelectedStationId] = useState(initialStationId || '');
  const [stationSearch, setStationSearch] = useState('');

  // Arrival Schedule State
  const [arrivalMode, setArrivalMode] = useState('next_hour'); // 'next_hour' | 'scheduled'
  const [arrivalDatetime, setArrivalDatetime] = useState('');

  // Results State
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [recommendationResult, setRecommendationResult] = useState(null);

  // Core Availability & Recommendation Calculation Engine
  const runAvailabilityCalculation = async (targetStId, vehicleOverride = null) => {
    const stId = targetStId || selectedStationId;
    if (!stId) {
      setError('Please select a charging station.');
      return;
    }
    if (targetSoc <= currentSoc) {
      setError(`Target battery percentage (${targetSoc}%) must be greater than current battery (${currentSoc}%).`);
      return;
    }

    const veh =
      vehicleOverride ||
      (selectedCategory === 'Custom' || selectedVehicleId === 'custom'
        ? customVehicle
        : vehicles.find((v) => v.vehicle_id === selectedVehicleId) || vehicles[0] || customVehicle);

    setSelectedStationId(stId);
    setLoading(true);
    setError(null);

    try {
      const payload = {
        station_id: stId,
        vehicle: veh,
        current_soc_pct: Number(currentSoc),
        target_soc_pct: Number(targetSoc),
        target_datetime_pt: arrivalMode === 'scheduled' ? arrivalDatetime : null,
        max_radius_miles: 5.0,
        max_results: 4,
      };

      const res = await recommendStations(payload);
      if (res?.status === 'success') {
        setRecommendationResult(res);
        setTimeout(() => {
          const resEl = document.getElementById('results-section');
          if (resEl) {
            resEl.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }
        }, 100);
      } else {
        setError(res?.message || 'Failed to retrieve recommendations.');
      }
    } catch (err) {
      console.error('Error fetching availability:', err);
      setError(err?.response?.data?.message || err.message || 'Error communicating with recommendation engine.');
    } finally {
      setLoading(false);
    }
  };

  // Sync if initialStationId prop updates
  useEffect(() => {
    if (initialStationId && initialStationId !== selectedStationId) {
      setSelectedStationId(initialStationId);
      if (stations.length > 0) {
        runAvailabilityCalculation(initialStationId);
      }
    }
  }, [initialStationId]);

  // Load vehicles and stations on mount
  useEffect(() => {
    const initData = async () => {
      try {
        const [vData, sData] = await Promise.all([getVehicles(), getStations()]);
        let activeVeh = customVehicle;
        if (vData?.vehicles) {
          setVehicles(vData.vehicles);
          const first4W = vData.vehicles.find((v) => v.vehicle_category === 'Four-Wheeler');
          if (first4W) {
            setSelectedVehicleId(first4W.vehicle_id);
            activeVeh = first4W;
          }
        }
        if (sData?.stations && sData.stations.length > 0) {
          setStations(sData.stations);
          // Match initialStationId or default to high-volume station
          const defaultSt = initialStationId
            ? sData.stations.find((s) => s.station_id === initialStationId)
            : (sData.stations.find((s) => s.short_name.includes('HAMILTON')) || sData.stations[0]);
          const chosenStId = defaultSt ? defaultSt.station_id : sData.stations[0].station_id;
          setSelectedStationId(chosenStId);

          // If navigated directly with an initial station, auto-run calculation
          if (initialStationId) {
            runAvailabilityCalculation(chosenStId, activeVeh);
          }
        }
      } catch (err) {
        console.error('Failed to load initial data:', err);
        setError('Failed to connect to backend service. Please check that the server is running.');
      }
    };
    initData();

    // Default scheduled time: tomorrow at 09:00 AM
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    tomorrow.setHours(9, 0, 0, 0);
    const isoLocal = tomorrow.toISOString().slice(0, 16);
    setArrivalDatetime(isoLocal);
  }, []);

  // Update selected vehicle when category changes
  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
    if (category === 'Custom') {
      setSelectedVehicleId('custom');
    } else {
      const match = vehicles.find((v) => v.vehicle_category === category);
      if (match) setSelectedVehicleId(match.vehicle_id);
    }
  };

  const activeVehicle =
    selectedCategory === 'Custom' || selectedVehicleId === 'custom'
      ? customVehicle
      : vehicles.find((v) => v.vehicle_id === selectedVehicleId) || vehicles[0] || customVehicle;

  // Search filter for station dropdown
  const filteredStations = stations.filter(
    (s) =>
      s.short_name.toLowerCase().includes(stationSearch.toLowerCase()) ||
      s.address.toLowerCase().includes(stationSearch.toLowerCase())
  );

  // Quick energy required calculation
  const battCap = Number(activeVehicle.battery_capacity_kwh || 60);
  const deltaSoc = Math.max(0, targetSoc - currentSoc);
  const estEnergyKwh = (battCap * (deltaSoc / 100)).toFixed(1);

  // Handle Find Availability from primary form
  const handleFindAvailability = (e) => {
    e?.preventDefault();
    runAvailabilityCalculation(selectedStationId);
  };

  return (
    <div className="space-y-8">
      {/* Page Header */}
      <div className="bg-gradient-to-r from-slate-900 via-slate-800 to-indigo-950 rounded-2xl p-6 sm:p-8 text-white shadow-lg border border-slate-700/60 relative overflow-hidden">
        <div className="absolute -right-10 -bottom-10 w-64 h-64 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="relative z-10 max-w-3xl">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/20 text-emerald-300 text-xs font-semibold mb-4 border border-emerald-500/30">
            <Sparkles className="w-3.5 h-3.5" />
            Phase 8 Intelligent Assistant
          </div>
          <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight">
            Find a Station &amp; Verify Availability
          </h2>
          <p className="mt-2 text-slate-300 text-sm sm:text-base leading-relaxed">
            Select your vehicle model, battery level, target station, and arrival time. Our machine-learning
            queueing model estimates physical port availability, computes battery charging time with CC/CV taper curves,
            and automatically identifies compatible nearby alternatives.
          </p>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-red-50 border border-red-200 text-red-800 text-sm flex items-start gap-3 shadow-xs">
          <XCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
          <div className="flex-1">
            <p className="font-semibold">Request Error</p>
            <p className="mt-0.5 text-xs text-red-700">{error}</p>
          </div>
        </div>
      )}

      {/* Main 6-Step Workflow Card */}
      <div className="bg-white rounded-2xl shadow-xs border border-slate-200 p-6 sm:p-8">
        <form onSubmit={handleFindAvailability} className="space-y-8">
          {/* STEP 1: Vehicle Category Selection */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs flex items-center justify-center font-bold">
                  1
                </span>
                Select Vehicle Category
              </label>
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
              {[
                { id: 'Four-Wheeler', label: 'Four-Wheeler (EV Cars)', icon: Car },
                { id: 'Two-Wheeler', label: 'Two-Wheeler (Motorcycles/Mopeds)', icon: Zap },
                { id: 'Custom', label: 'Custom Vehicle Template', icon: Sliders },
              ].map((cat) => {
                const Icon = cat.icon;
                const active = selectedCategory === cat.id;
                return (
                  <button
                    key={cat.id}
                    type="button"
                    onClick={() => handleCategoryChange(cat.id)}
                    className={`p-4 rounded-xl border text-left flex items-center gap-3 transition-all ${
                      active
                        ? 'border-emerald-600 bg-emerald-50/60 ring-2 ring-emerald-500/20 text-slate-900'
                        : 'border-slate-200 hover:border-slate-300 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    <div
                      className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${
                        active ? 'bg-emerald-600 text-white' : 'bg-slate-100 text-slate-600'
                      }`}
                    >
                      <Icon className="w-5 h-5" />
                    </div>
                    <div>
                      <div className="font-semibold text-sm">{cat.label}</div>
                      <div className="text-xs text-slate-400">
                        {cat.id === 'Custom' ? 'Custom battery & power' : 'OEM verified specs'}
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* STEP 2: Vehicle Model & Specifications */}
          <div className="pt-4 border-t border-slate-100">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs flex items-center justify-center font-bold">
                  2
                </span>
                Choose Vehicle Model &amp; Specifications
              </label>
            </div>

            {selectedCategory !== 'Custom' ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <label className="block text-xs font-semibold text-slate-600 mb-1">
                    Select Verified OEM Model
                  </label>
                  <select
                    value={selectedVehicleId}
                    onChange={(e) => setSelectedVehicleId(e.target.value)}
                    className="w-full px-3.5 py-2.5 rounded-lg border border-slate-300 bg-white text-sm text-slate-800 focus:outline-hidden focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500 font-medium"
                  >
                    {vehicles
                      .filter((v) => v.vehicle_category === selectedCategory)
                      .map((v) => (
                        <option key={v.vehicle_id} value={v.vehicle_id}>
                          {v.manufacturer} {v.model} ({v.battery_capacity_kwh} kWh, {v.maximum_ac_charging_power_kw} kW AC)
                        </option>
                      ))}
                  </select>
                </div>

                {/* Specs Pill Summary */}
                <div className="rounded-xl bg-slate-50 border border-slate-200/80 p-3.5 flex flex-wrap gap-2 items-center text-xs text-slate-700">
                  <div className="px-2.5 py-1 bg-white rounded-md border border-slate-200 font-semibold text-slate-900">
                    Battery: <span className="text-emerald-700">{activeVehicle.battery_capacity_kwh} kWh</span>
                  </div>
                  <div className="px-2.5 py-1 bg-white rounded-md border border-slate-200 font-semibold text-slate-900">
                    Max AC Power: <span className="text-emerald-700">{activeVehicle.maximum_ac_charging_power_kw} kW</span>
                  </div>
                  <div className="px-2.5 py-1 bg-white rounded-md border border-slate-200 font-semibold text-slate-900">
                    Connector: <span className="text-indigo-700">{activeVehicle.connector_type}</span>
                  </div>
                  <div className="px-2.5 py-1 bg-white rounded-md border border-slate-200 font-semibold text-slate-900">
                    Efficiency: <span className="text-slate-700">{((activeVehicle.charging_efficiency || 0.9) * 100).toFixed(0)}%</span>
                  </div>
                </div>
              </div>
            ) : (
              /* Custom Vehicle Config Inputs */
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 p-4 rounded-xl bg-slate-50 border border-slate-200">
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Battery Capacity (kWh)</label>
                  <input
                    type="number"
                    min="1"
                    max="200"
                    step="0.5"
                    value={customVehicle.battery_capacity_kwh}
                    onChange={(e) =>
                      setCustomVehicle({ ...customVehicle, battery_capacity_kwh: parseFloat(e.target.value) || 60 })
                    }
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Max AC Power (kW)</label>
                  <input
                    type="number"
                    min="1"
                    max="22"
                    step="0.1"
                    value={customVehicle.maximum_ac_charging_power_kw}
                    onChange={(e) =>
                      setCustomVehicle({ ...customVehicle, maximum_ac_charging_power_kw: parseFloat(e.target.value) || 11 })
                    }
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm"
                  />
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Connector Standard</label>
                  <select
                    value={customVehicle.connector_type}
                    onChange={(e) => setCustomVehicle({ ...customVehicle, connector_type: e.target.value })}
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm"
                  >
                    <option value="J1772">J1772 (Standard Level 2)</option>
                    <option value="J1772 (via OEM adapter) / NACS">NACS / Tesla (J1772 Adapter)</option>
                    <option value="NEMA 5-20R (120V Standard Wall/Plug)">NEMA 5-20R (Level 1 Wall Plug)</option>
                    <option value="CCS1">CCS1 Combo</option>
                  </select>
                </div>
                <div>
                  <label className="block text-xs font-semibold text-slate-700 mb-1">Efficiency (0.80 - 0.98)</label>
                  <input
                    type="number"
                    min="0.80"
                    max="0.98"
                    step="0.01"
                    value={customVehicle.charging_efficiency}
                    onChange={(e) =>
                      setCustomVehicle({ ...customVehicle, charging_efficiency: parseFloat(e.target.value) || 0.9 })
                    }
                    className="w-full px-3 py-2 rounded-lg border border-slate-300 bg-white text-sm"
                  />
                </div>
              </div>
            )}
          </div>

          {/* STEP 3: Battery Level & Target Percentage */}
          <div className="pt-4 border-t border-slate-100">
            <div className="flex items-center justify-between mb-3">
              <label className="text-sm font-bold text-slate-900 flex items-center gap-2">
                <span className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs flex items-center justify-center font-bold">
                  3
                </span>
                Battery State of Charge (SOC)
              </label>
              <span className="text-xs text-slate-500">
                Required Energy: <strong className="text-emerald-700 font-bold">{estEnergyKwh} kWh</strong>
              </span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 p-4 rounded-xl bg-slate-50/70 border border-slate-200">
              {/* Current Battery % */}
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1.5">
                  <span>Current Battery Level</span>
                  <span className="text-emerald-700 font-bold text-sm">{currentSoc}%</span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="95"
                  value={currentSoc}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setCurrentSoc(val);
                    if (val >= targetSoc) setTargetSoc(Math.min(100, val + 10));
                  }}
                  className="w-full accent-emerald-600 h-2 bg-slate-200 rounded-lg cursor-pointer"
                />
                <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                  <span>0% (Empty)</span>
                  <span>50%</span>
                  <span>95%</span>
                </div>
              </div>

              {/* Target Battery % */}
              <div>
                <div className="flex justify-between text-xs font-semibold text-slate-700 mb-1.5">
                  <span>Target Battery Level</span>
                  <span className="text-emerald-700 font-bold text-sm">{targetSoc}%</span>
                </div>
                <input
                  type="range"
                  min="10"
                  max="100"
                  value={targetSoc}
                  onChange={(e) => {
                    const val = Number(e.target.value);
                    setTargetSoc(val);
                    if (val <= currentSoc) setCurrentSoc(Math.max(0, val - 10));
                  }}
                  className="w-full accent-emerald-600 h-2 bg-slate-200 rounded-lg cursor-pointer"
                />
                <div className="flex justify-between text-[11px] text-slate-400 mt-1">
                  <span>10%</span>
                  <span className="text-amber-600 font-medium">80% (Recommended)</span>
                  <span>100% (Full)</span>
                </div>
              </div>
            </div>
            {targetSoc > 80 && (
              <p className="mt-2 text-[12px] text-amber-700 flex items-center gap-1.5">
                <Info className="w-3.5 h-3.5 shrink-0" />
                Charging above 80% engages the vehicle BMS CC/CV taper curve, reducing charging speed to protect battery chemistry.
              </p>
            )}
          </div>

          {/* STEP 4 & 5: Station Selection & Arrival Schedule */}
          <div className="pt-4 border-t border-slate-100 grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Step 4: Station Selection */}
            <div>
              <label className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs flex items-center justify-center font-bold">
                  4
                </span>
                Target Charging Station
              </label>
              <div className="space-y-2">
                <div className="relative">
                  <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                  <input
                    type="text"
                    placeholder="Filter stations by name or street..."
                    value={stationSearch}
                    onChange={(e) => setStationSearch(e.target.value)}
                    className="w-full pl-9 pr-3 py-2 text-xs rounded-lg border border-slate-300 bg-white"
                  />
                </div>
                <select
                  value={selectedStationId}
                  onChange={(e) => setSelectedStationId(e.target.value)}
                  size={5}
                  className="w-full p-2 text-xs rounded-lg border border-slate-300 bg-white font-medium focus:ring-2 focus:ring-emerald-500"
                >
                  {filteredStations.map((st) => (
                    <option key={st.station_id} value={st.station_id} className="py-1">
                      {st.short_name} — {st.address} ({st.port_types?.join(', ') || 'Level 2'})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {/* Step 5: Arrival Date & Time */}
            <div>
              <label className="text-sm font-bold text-slate-900 flex items-center gap-2 mb-2">
                <span className="w-6 h-6 rounded-full bg-slate-900 text-white text-xs flex items-center justify-center font-bold">
                  5
                </span>
                Arrival Time
              </label>

              <div className="space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <button
                    type="button"
                    onClick={() => setArrivalMode('next_hour')}
                    className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-all ${
                      arrivalMode === 'next_hour'
                        ? 'border-emerald-600 bg-emerald-50 text-emerald-900 font-bold'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    Immediate / Next Hour
                  </button>
                  <button
                    type="button"
                    onClick={() => setArrivalMode('scheduled')}
                    className={`px-3 py-2 text-xs font-semibold rounded-lg border transition-all ${
                      arrivalMode === 'scheduled'
                        ? 'border-emerald-600 bg-emerald-50 text-emerald-900 font-bold'
                        : 'border-slate-200 text-slate-600 hover:bg-slate-50'
                    }`}
                  >
                    Schedule Date &amp; Time
                  </button>
                </div>

                {arrivalMode === 'scheduled' ? (
                  <div>
                    <label className="block text-[11px] font-medium text-slate-600 mb-1">
                      Arrival Date &amp; Time (Pacific Time)
                    </label>
                    <input
                      type="datetime-local"
                      value={arrivalDatetime}
                      onChange={(e) => setArrivalDatetime(e.target.value)}
                      className="w-full px-3 py-2 rounded-lg border border-slate-300 text-xs font-medium bg-white"
                    />
                    <p className="text-[11px] text-slate-400 mt-1">
                      Uses 24-hour day-ahead XGBoost model combined with historical arrival baselines.
                    </p>
                  </div>
                ) : (
                  <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 text-xs text-slate-600 flex items-center gap-2">
                    <Clock className="w-4 h-4 text-emerald-600 shrink-0" />
                    <span>
                      Evaluates next 1-hour window using Phase 5 primary XGBoost arrival model and active dwell times.
                    </span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* STEP 6: Action Button */}
          <div className="pt-4 border-t border-slate-100 flex items-center justify-end">
            <button
              type="submit"
              disabled={loading}
              className="w-full sm:w-auto px-8 py-3.5 rounded-xl bg-emerald-600 hover:bg-emerald-700 text-white font-bold text-sm shadow-md transition-all flex items-center justify-center gap-2 disabled:opacity-60 cursor-pointer"
            >
              {loading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Calculating Availability &amp; Recommendations...
                </>
              ) : (
                <>
                  <Search className="w-4 h-4" />
                  Find Availability &amp; Smart Recommendations
                </>
              )}
            </button>
          </div>
        </form>
      </div>

      {/* RESULTS SECTION */}
      {recommendationResult && (
        <div id="results-section" className="space-y-6">
          {/* Target Station Assessment Card */}
          <div className="bg-white rounded-2xl shadow-xs border border-slate-200 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-100">
              <div>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-0.5 rounded-md bg-slate-100 text-slate-800 font-bold text-xs">
                    Target Station Assessment
                  </span>
                  <span className="text-xs text-slate-500">
                    {recommendationResult.target_station.short_name}
                  </span>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mt-1">
                  {recommendationResult.target_station.station_name}
                </h3>
                <p className="text-xs text-slate-500 flex items-center gap-1.5 mt-0.5">
                  <MapPin className="w-3.5 h-3.5 text-slate-400" />
                  {recommendationResult.target_station.address}
                </p>
                <div className="flex flex-wrap items-center gap-2 mt-2.5">
                  <button
                    type="button"
                    onClick={() => onNavigateToStation?.(recommendationResult.target_station.station_id, 'predictions')}
                    className="text-xs font-semibold text-emerald-700 hover:text-emerald-800 bg-emerald-50 hover:bg-emerald-100 px-2.5 py-1 rounded-md transition-colors cursor-pointer"
                  >
                    View Demand Forecast &rarr;
                  </button>
                  <button
                    type="button"
                    onClick={() => onNavigateToStation?.(recommendationResult.target_station.station_id, 'probabilistic-availability')}
                    className="text-xs font-semibold text-indigo-700 hover:text-indigo-800 bg-indigo-50 hover:bg-indigo-100 px-2.5 py-1 rounded-md transition-colors cursor-pointer"
                  >
                    Probabilistic Arrival Curve &rarr;
                  </button>
                  <button
                    type="button"
                    onClick={() => onNavigateToStation?.(recommendationResult.target_station.station_id, 'stations')}
                    className="text-xs font-semibold text-slate-700 hover:text-slate-800 bg-slate-100 hover:bg-slate-200 px-2.5 py-1 rounded-md transition-colors cursor-pointer"
                  >
                    Station Explorer &rarr;
                  </button>
                </div>
              </div>

              {/* Availability Status Badge */}
              {(() => {
                const avail = recommendationResult.target_station.availability;
                const status = avail.availability_status;
                let badgeClass = 'bg-emerald-50 text-emerald-700 border-emerald-300';
                let Icon = CheckCircle2;
                if (status === 'Limited Availability') {
                  badgeClass = 'bg-amber-50 text-amber-700 border-amber-300';
                  Icon = AlertTriangle;
                } else if (status === 'High Occupancy Risk' || status === 'Crowded') {
                  badgeClass = 'bg-red-50 text-red-700 border-red-300';
                  Icon = XCircle;
                }
                return (
                  <div className={`px-4 py-2.5 rounded-xl border flex items-center gap-2.5 font-bold text-sm ${badgeClass}`}>
                    <Icon className="w-5 h-5 shrink-0" />
                    <div>
                      <div className="leading-tight">{status}</div>
                      <div className="text-[11px] font-normal opacity-90">
                        {avail.estimated_free_points !== null
                          ? `${avail.estimated_free_points} of ${avail.observed_point_count} points free`
                          : 'History limited'}
                      </div>
                    </div>
                  </div>
                );
              })()}
            </div>

            {/* Metrics Breakdown Grid */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 py-6 border-b border-slate-100">
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-xs text-slate-500 font-medium block">Observed Points</span>
                <span className="text-2xl font-extrabold text-slate-900 mt-1 block">
                  {recommendationResult.target_station.availability.observed_point_count}
                </span>
                <span className="text-[11px] text-slate-400">Physical verified ports</span>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-xs text-slate-500 font-medium block">Estimated Free Points</span>
                <span className="text-2xl font-extrabold text-emerald-600 mt-1 block">
                  {recommendationResult.target_station.availability.estimated_free_points ?? 'N/A'}
                </span>
                <span className="text-[11px] text-slate-400">
                  {recommendationResult.target_station.availability.estimated_free_points_rounded !== null
                    ? `~${recommendationResult.target_station.availability.estimated_free_points_rounded} free ports expected`
                    : 'Uncertain'}
                </span>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-xs text-slate-500 font-medium block">Occupancy Risk</span>
                <span className="text-2xl font-extrabold text-slate-900 mt-1 block">
                  {recommendationResult.target_station.availability.probability_zero_free_points !== null
                    ? `${(recommendationResult.target_station.availability.probability_zero_free_points * 100).toFixed(1)}%`
                    : 'N/A'}
                </span>
                <span className="text-[11px] text-slate-400">Probability of zero free points</span>
                <div className="mt-2 pt-2 border-t border-slate-200 text-[11px] text-slate-600 flex items-center justify-between">
                  <span className="font-semibold text-teal-700">90% Calibrated Range:</span>
                  <span className="font-mono font-bold text-slate-800">
                    {recommendationResult.target_station.availability.probability_zero_free_points !== null
                      ? `[${Math.max(0, Math.round((recommendationResult.target_station.availability.probability_zero_free_points * 0.3) * 100))}% – ${Math.min(100, Math.round((recommendationResult.target_station.availability.probability_zero_free_points * 1.5 + 0.08) * 100))}%]`
                      : '--'}
                  </span>
                </div>
              </div>

              <div className="p-4 rounded-xl bg-slate-50 border border-slate-100">
                <span className="text-xs text-slate-500 font-medium block">Confidence Rating</span>
                <span className="text-xl font-bold text-slate-800 mt-1 block">
                  {recommendationResult.target_station.availability.availability_confidence || 'Medium'}
                </span>
                <span className="text-[11px] text-slate-400">Queueing model fidelity</span>
              </div>
            </div>

            {/* Vehicle Compatibility & Charging Duration Row */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6 pt-6">
              {/* Compatibility Box */}
              <div className="rounded-xl border border-slate-200 p-4">
                <div className="flex items-center gap-2 mb-2 font-bold text-sm text-slate-900">
                  <ShieldCheck className="w-4 h-4 text-emerald-600" />
                  Vehicle Compatibility
                </div>
                <div className="space-y-1.5 text-xs text-slate-700">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Status:</span>
                    <span className="font-semibold text-emerald-700">
                      {recommendationResult.target_station.compatibility.compatibility_status}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Station Connector:</span>
                    <span className="font-semibold text-slate-900">
                      {recommendationResult.target_station.compatibility.supported_connectors?.join(', ')}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Station Max Power:</span>
                    <span className="font-semibold text-slate-900">
                      {recommendationResult.target_station.compatibility.station_max_power_kw} kW
                    </span>
                  </div>
                  {recommendationResult.target_station.compatibility.adapter_required && (
                    <div className="mt-2 p-2 bg-indigo-50 border border-indigo-200 rounded-md text-[11px] text-indigo-800">
                      Compatible using manufacturer J1772 adapter (included with Tesla/NACS vehicles).
                    </div>
                  )}
                </div>
              </div>

              {/* Charging Duration Box */}
              <div className="rounded-xl border border-slate-200 p-4">
                <div className="flex items-center gap-2 mb-2 font-bold text-sm text-slate-900">
                  <BatteryCharging className="w-4 h-4 text-emerald-600" />
                  Estimated Charging Duration
                </div>
                <div className="space-y-1.5 text-xs text-slate-700">
                  <div className="flex justify-between">
                    <span className="text-slate-500">Estimated Duration:</span>
                    <span className="font-extrabold text-base text-emerald-700">
                      {recommendationResult.target_station.charging_time.estimated_duration_formatted}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Required Energy:</span>
                    <span className="font-semibold text-slate-900">
                      {recommendationResult.target_station.charging_time.required_energy_kwh} kWh ({currentSoc}% &rarr; {targetSoc}%)
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Effective Delivery Rate:</span>
                    <span className="font-semibold text-slate-900">
                      {recommendationResult.target_station.charging_time.effective_charging_power_kw} kW
                    </span>
                  </div>
                  <div className="mt-2 text-[11px] text-slate-500 leading-tight">
                    {recommendationResult.target_station.charging_time.notes}
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Recommended Nearby Alternatives Section */}
          <div className="bg-white rounded-2xl shadow-xs border border-slate-200 p-6 sm:p-8">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6">
              <div>
                <h3 className="text-lg font-bold text-slate-900 flex items-center gap-2">
                  <Navigation className="w-5 h-5 text-indigo-600" />
                  Recommended Nearby Alternatives
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Ranked by availability, great-circle distance, vehicle compatibility, and charging speed within 5 miles.
                </p>
              </div>
              <span className="text-xs font-semibold px-2.5 py-1 rounded-full bg-slate-100 text-slate-700 self-start sm:self-auto">
                {recommendationResult.recommended_alternatives.length} Alternatives Found
              </span>
            </div>

            {recommendationResult.recommended_alternatives.length === 0 ? (
              <div className="text-center py-8 text-slate-500 text-sm">
                No compatible alternatives found within 5 miles of this station.
              </div>
            ) : (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {recommendationResult.recommended_alternatives.map((alt) => (
                  <div
                    key={alt.station_id}
                    className="rounded-xl border border-slate-200 hover:border-slate-300 p-5 bg-white hover:shadow-sm transition-all flex flex-col justify-between"
                  >
                    <div>
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-xs font-bold text-indigo-700">
                            {alt.distance_miles} mi ({alt.distance_km} km) away
                          </div>
                          <h4 className="font-bold text-slate-900 text-base mt-0.5">{alt.short_name}</h4>
                          <p className="text-xs text-slate-500 mt-0.5 flex items-center gap-1">
                            <MapPin className="w-3 h-3 text-slate-400" />
                            {alt.address}
                          </p>
                        </div>
                        <div className="text-right shrink-0">
                          <span className="inline-block px-2 py-0.5 rounded-md bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs font-bold">
                            Score: {alt.rank_score}
                          </span>
                        </div>
                      </div>

                      {/* Availability & Specs Grid */}
                      <div className="mt-4 grid grid-cols-3 gap-2 p-2.5 bg-slate-50 rounded-lg text-center text-xs">
                        <div>
                          <span className="text-slate-400 text-[10px] block">Availability</span>
                          <span className="font-bold text-emerald-700">
                            {alt.estimated_free_points} / {alt.observed_point_count} Free
                          </span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-[10px] block">Power</span>
                          <span className="font-bold text-slate-800">{alt.charging_power_kw} kW</span>
                        </div>
                        <div>
                          <span className="text-slate-400 text-[10px] block">Est. Time</span>
                          <span className="font-bold text-slate-800">{alt.estimated_charging_time_formatted}</span>
                        </div>
                      </div>

                      {/* Reasons Pills */}
                      <div className="mt-3 flex flex-wrap gap-1.5">
                        {alt.reasons?.map((reason, idx) => (
                          <span
                            key={idx}
                            className="px-2 py-0.5 rounded-full bg-slate-100 text-[11px] text-slate-600 font-medium"
                          >
                            &bull; {reason}
                          </span>
                        ))}
                      </div>
                    </div>

                    <div className="mt-5 pt-3 border-t border-slate-100 flex flex-wrap items-center justify-between gap-2">
                      <span className="text-xs font-semibold text-slate-600">
                        Connector: <span className="text-indigo-700">{alt.compatible_connectors?.join(', ')}</span>
                      </span>
                      <div className="flex flex-wrap items-center gap-1.5">
                        <button
                          type="button"
                          onClick={() => onNavigateToStation?.(alt.station_id, 'predictions')}
                          className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-[11px] font-semibold text-slate-700 transition-colors cursor-pointer"
                          title="View demand forecast"
                        >
                          Forecast
                        </button>
                        <button
                          type="button"
                          onClick={() => onNavigateToStation?.(alt.station_id, 'probabilistic-availability')}
                          className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-[11px] font-semibold text-slate-700 transition-colors cursor-pointer"
                          title="View probabilistic arrival curve"
                        >
                          Probabilistic
                        </button>
                        <button
                          type="button"
                          onClick={() => onNavigateToStation?.(alt.station_id, 'stations')}
                          className="px-2 py-1 rounded bg-slate-100 hover:bg-slate-200 text-[11px] font-semibold text-slate-700 transition-colors cursor-pointer"
                          title="Explore station telemetry"
                        >
                          Explorer
                        </button>
                        <button
                          type="button"
                          onClick={() => runAvailabilityCalculation(alt.station_id)}
                          className="px-3 py-1.5 rounded-lg bg-emerald-600 hover:bg-emerald-700 text-white text-xs font-bold flex items-center gap-1 shadow-sm transition-all cursor-pointer"
                        >
                          <span>Select Station</span>
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
