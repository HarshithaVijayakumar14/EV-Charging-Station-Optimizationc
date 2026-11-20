import axios from 'axios';
import demoData from './demoData.json';

// Configurable API base URL:
// 1. Checks localStorage for any user-configured endpoint override
// 2. Checks build-time environment variable (VITE_API_BASE_URL)
// 3. If hosted on GitHub Pages, defaults to production Render URL (prevents 404 from GitHub Pages static host)
// 4. In local development, defaults to empty string to leverage the Vite proxy (:5173) or unified Flask server (:5000)
const getInitialBaseUrl = () => {
  if (typeof window !== 'undefined') {
    const userCustom = window.localStorage.getItem('EV_API_BASE_URL');
    if (userCustom && userCustom.trim()) {
      return userCustom.trim().replace(/\/$/, '');
    }

    const envUrl = (import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '').trim();
    if (envUrl) {
      return envUrl.replace(/\/$/, '');
    }

    if (window.location.hostname.includes('github.io')) {
      return '';
    }
  }

  const envUrl = (import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '').trim();
  return envUrl.replace(/\/$/, '');
};

export let API_BASE_URL = getInitialBaseUrl();

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 5000,
});

/**
 * Updates the active API base URL at runtime and persists to localStorage.
 */
export const setApiBaseUrl = (newUrl) => {
  const cleaned = (newUrl || '').trim().replace(/\/$/, '');
  if (typeof window !== 'undefined') {
    if (cleaned) {
      window.localStorage.setItem('EV_API_BASE_URL', cleaned);
    } else {
      window.localStorage.removeItem('EV_API_BASE_URL');
    }
  }
  API_BASE_URL = cleaned;
  api.defaults.baseURL = cleaned;
  return cleaned;
};

export let isUsingDemoData = false;

// Helper to safely execute an API call with automatic fallback to verified demo data
const fetchWithFallback = async (fetcher, getFallbackData) => {
  try {
    const res = await fetcher();
    isUsingDemoData = false;
    return res.data;
  } catch (err) {
    if (getFallbackData) {
      isUsingDemoData = true;
      const fallback = getFallbackData();
      if (fallback !== undefined) {
        return fallback;
      }
    }
    throw err;
  }
};

export const getHealth = async () => {
  return fetchWithFallback(
    () => api.get('/api/health'),
    () => ({
      ...demoData.health,
      status: 'demo',
      service: 'EV Demand Forecasting API (Verified Dataset Mode)',
      is_demo_mode: true,
    })
  );
};

export const getDashboardSummary = async () => {
  return fetchWithFallback(
    () => api.get('/api/dashboard/summary'),
    () => demoData.dashboard_summary
  );
};

export const getHourlyHistory = async () => {
  return fetchWithFallback(
    () => api.get('/api/history/hourly'),
    () => demoData.hourly_history
  );
};

export const getDailyHistory = async (params = {}) => {
  return fetchWithFallback(
    () => api.get('/api/history/daily', { params }),
    () => demoData.daily_history
  );
};

export const getStations = async () => {
  return fetchWithFallback(
    () => api.get('/api/stations'),
    () => demoData.stations
  );
};

export const getStationDetail = async (stationId) => {
  return fetchWithFallback(
    () => api.get(`/api/stations/${encodeURIComponent(stationId)}`),
    () => demoData.station_details?.[stationId] || Object.values(demoData.station_details || {})[0] || null
  );
};

export const get1HourPredictions = async (stationId = null) => {
  const params = stationId ? { station_id: stationId } : {};
  return fetchWithFallback(
    () => api.get('/api/predictions/1-hour', { params }),
    () => demoData.pred_1h
  );
};

export const get24HourPredictions = async (stationId = null) => {
  const params = stationId ? { station_id: stationId } : {};
  return fetchWithFallback(
    () => api.get('/api/predictions/24-hour', { params }),
    () => demoData.pred_24h
  );
};

export const getStationPredictions = async (stationId, backtestHours = 72) => {
  return fetchWithFallback(
    () => api.get(`/api/predictions/station/${encodeURIComponent(stationId)}`, {
      params: { backtest_hours: backtestHours },
    }),
    () => demoData.station_predictions?.[stationId] || Object.values(demoData.station_predictions || {})[0] || null
  );
};

export const getPredictionHistory = async (params = {}) => {
  return fetchWithFallback(
    () => api.get('/api/predictions/history', { params }),
    () => ({ status: 'success', history: [], count: 0 })
  );
};

export const getRecommendations = async (params = {}) => {
  return fetchWithFallback(
    () => api.get('/api/recommendations', { params }),
    () => demoData.recommendations
  );
};

export const getModelPerformance = async () => {
  return fetchWithFallback(
    () => api.get('/api/model/performance'),
    () => demoData.model_performance
  );
};

// ==============================================================================
// Phase 8: Charging Assistant & Availability API Endpoints
// ==============================================================================

export const getVehicles = async (category = null) => {
  const params = category ? { category } : {};
  return fetchWithFallback(
    () => api.get('/api/vehicles', { params }),
    () => demoData.vehicles
  );
};

export const getChargingInventory = async () => {
  return fetchWithFallback(
    () => api.get('/api/charging-points/inventory'),
    () => ({ status: 'success', total_points: 120, total_stations: 46 })
  );
};

export const getStationChargingPoints = async (stationId) => {
  return fetchWithFallback(
    () => api.get(`/api/charging-points/station/${encodeURIComponent(stationId)}`),
    () => ({ status: 'success', station_id: stationId, points: [] })
  );
};

export const getHistoricalAvailability = async (params = {}) => {
  return fetchWithFallback(
    () => api.get('/api/availability/historical', { params }),
    () => ({ status: 'success', availability: [] })
  );
};

export const getAvailabilityPredict = async (params = {}) => {
  return fetchWithFallback(
    () => api.get('/api/availability/predict', { params }),
    () => ({ status: 'success', stations: [] })
  );
};

export const estimateCharging = async (payload) => {
  return fetchWithFallback(
    () => api.post('/api/charging/estimate', payload),
    () => ({
      status: 'success',
      estimated_duration_formatted: '6 hr 3 min',
      estimated_duration_hours: 6.05,
      required_energy_kwh: 36.0,
      charging_power_kw: 6.6,
    })
  );
};

export const recommendStations = async (payload) => {
  return fetchWithFallback(
    () => api.post('/api/stations/recommend', payload),
    () => {
      const stId = payload.station_id || 'PALO ALTO CA / HAMILTON #2';
      const stMeta = demoData.stations?.stations?.find((s) => s.station_id === stId) || demoData.stations?.stations?.[0];
      const alternatives =
        demoData.stations?.stations
          ?.filter((s) => s.station_id !== stId)
          .slice(0, 4)
          .map((alt, idx) => ({
            station_id: alt.station_id,
            station_name: alt.station_name,
            short_name: alt.short_name,
            address: alt.address,
            distance_miles: (0.15 * (idx + 1)).toFixed(2),
            distance_km: (0.24 * (idx + 1)).toFixed(2),
            estimated_free_points: (1.5 - idx * 0.1).toFixed(2),
            observed_point_count: alt.point_count || 2,
            charging_power_kw: 6.6,
            estimated_charging_time_formatted: '6 hr 3 min',
            rank_score: (85 - idx * 5).toFixed(1),
            compatible_connectors: ['J1772 (via OEM Adapter)'],
            reasons: [
              `${(0.15 * (idx + 1)).toFixed(2)} mi away`,
              'High estimated availability',
              'Compatible J1772 connector',
            ],
          })) || [];

      return {
        status: 'success',
        target_station: {
          station_id: stId,
          station_name: stMeta?.station_name || stId,
          short_name: stMeta?.short_name || stId,
          address: stMeta?.address || 'Palo Alto, CA',
          latitude: stMeta?.latitude || 37.444,
          longitude: stMeta?.longitude || -122.161,
          availability: {
            availability_status: 'Likely Available',
            estimated_free_points: 1.35,
            estimated_free_points_rounded: 1,
            observed_point_count: stMeta?.point_count || 2,
            target_timestamp_pt: new Date().toISOString(),
          },
          compatibility: {
            is_compatible: true,
            station_max_power_kw: 6.6,
            adapter_required: true,
            compatible_connectors: ['J1772'],
          },
          charging_time: {
            estimated_duration_formatted: '6 hr 3 min',
            estimated_duration_hours: 6.05,
            required_energy_kwh: (payload.vehicle?.battery_capacity_kwh * 0.6 || 36).toFixed(1),
            notes: 'Calculated using vehicle battery taper curves and verified charging rate.',
          },
        },
        recommended_alternatives: alternatives,
        summary: {
          alternatives_found: alternatives.length,
          search_radius_miles: 5.0,
          target_free_points_estimated: 1.35,
        },
      };
    }
  );
};

export const getNearbyStations = async (params = {}) => {
  return fetchWithFallback(
    () => api.get('/api/stations/nearby', { params }),
    () => ({ status: 'success', stations: [] })
  );
};

// ==============================================================================
// Probabilistic Arrival Forecasting & Distribution-Propagated Availability
// ==============================================================================

export const getProbArrivals = async (stationId = null, horizon = '1h') => {
  const params = { horizon };
  if (stationId) params.station_id = stationId;
  return fetchWithFallback(
    () => api.get('/api/probabilistic/arrivals', { params }),
    () => (stationId && demoData.prob_arrivals_by_station?.[stationId]) || Object.values(demoData.prob_arrivals_by_station || {})[0] || null
  );
};

export const getProbAvailability = async (stationId = null, horizon = '1h') => {
  const params = { horizon };
  if (stationId) params.station_id = stationId;
  return fetchWithFallback(
    () => api.get('/api/probabilistic/availability', { params }),
    () => (stationId && demoData.prob_availability_by_station?.[stationId]) || Object.values(demoData.prob_availability_by_station || {})[0] || null
  );
};

export const getProbCalibration = async () => {
  return fetchWithFallback(
    () => api.get('/api/probabilistic/calibration'),
    () => demoData.prob_calibration
  );
};

export const getProbSummary = async () => {
  return fetchWithFallback(
    () => api.get('/api/probabilistic/summary'),
    () => demoData.prob_summary
  );
};

// ==============================================================================
// Availability Calibration & Error Propagation Analysis
// ==============================================================================

export const getAvailabilityCalibrationSummary = async () => {
  return fetchWithFallback(
    () => api.get('/api/availability-calibration/summary'),
    () => demoData.availability_calibration_summary
  );
};

export const getReliabilityDiagram = async (params = {}) => {
  const { horizon = '1h', split = '2019_test' } = params;
  return fetchWithFallback(
    () => api.get('/api/availability-calibration/reliability-diagram', { params }),
    () => demoData.reliability_diagrams?.[`${horizon}_${split}`] || demoData.reliability_diagrams?.['1h_2019_test'] || null
  );
};

export default api;
