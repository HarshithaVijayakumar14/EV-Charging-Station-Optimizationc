import axios from 'axios';

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
      return 'https://ev-charging-station-api.onrender.com';
    }
  }

  const envUrl = (import.meta.env.VITE_API_BASE_URL || import.meta.env.VITE_API_URL || '').trim();
  return envUrl.replace(/\/$/, '');
};

export let API_BASE_URL = getInitialBaseUrl();

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 35000,
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

export const getHealth = async () => {
  const res = await api.get('/api/health');
  return res.data;
};

export const getDashboardSummary = async () => {
  const res = await api.get('/api/dashboard/summary');
  return res.data;
};

export const getHourlyHistory = async () => {
  const res = await api.get('/api/history/hourly');
  return res.data;
};

export const getDailyHistory = async (params = {}) => {
  const res = await api.get('/api/history/daily', { params });
  return res.data;
};

export const getStations = async () => {
  const res = await api.get('/api/stations');
  return res.data;
};

export const getStationDetail = async (stationId) => {
  const res = await api.get(`/api/stations/${encodeURIComponent(stationId)}`);
  return res.data;
};

export const get1HourPredictions = async (stationId = null) => {
  const params = stationId ? { station_id: stationId } : {};
  const res = await api.get('/api/predictions/1-hour', { params });
  return res.data;
};

export const get24HourPredictions = async (stationId = null) => {
  const params = stationId ? { station_id: stationId } : {};
  const res = await api.get('/api/predictions/24-hour', { params });
  return res.data;
};

export const getStationPredictions = async (stationId, backtestHours = 72) => {
  const res = await api.get(`/api/predictions/station/${encodeURIComponent(stationId)}`, {
    params: { backtest_hours: backtestHours },
  });
  return res.data;
};

export const getPredictionHistory = async (params = {}) => {
  const res = await api.get('/api/predictions/history', { params });
  return res.data;
};

export const getRecommendations = async (params = {}) => {
  const res = await api.get('/api/recommendations', { params });
  return res.data;
};

export const getModelPerformance = async () => {
  const res = await api.get('/api/model/performance');
  return res.data;
};

// ==============================================================================
// Phase 8: Charging Assistant & Availability API Endpoints
// ==============================================================================

export const getVehicles = async (category = null) => {
  const params = category ? { category } : {};
  const res = await api.get('/api/vehicles', { params });
  return res.data;
};

export const getChargingInventory = async () => {
  const res = await api.get('/api/charging-points/inventory');
  return res.data;
};

export const getStationChargingPoints = async (stationId) => {
  const res = await api.get(`/api/charging-points/station/${encodeURIComponent(stationId)}`);
  return res.data;
};

export const getHistoricalAvailability = async (params = {}) => {
  const res = await api.get('/api/availability/historical', { params });
  return res.data;
};

export const getAvailabilityPredict = async (params = {}) => {
  const res = await api.get('/api/availability/predict', { params });
  return res.data;
};

export const estimateCharging = async (payload) => {
  const res = await api.post('/api/charging/estimate', payload);
  return res.data;
};

export const recommendStations = async (payload) => {
  const res = await api.post('/api/stations/recommend', payload);
  return res.data;
};

export const getNearbyStations = async (params = {}) => {
  const res = await api.get('/api/stations/nearby', { params });
  return res.data;
};

// ==============================================================================
// Probabilistic Arrival Forecasting & Distribution-Propagated Availability
// ==============================================================================

export const getProbArrivals = async (stationId = null, horizon = '1h') => {
  const params = { horizon };
  if (stationId) params.station_id = stationId;
  const res = await api.get('/api/probabilistic/arrivals', { params });
  return res.data;
};

export const getProbAvailability = async (stationId = null, horizon = '1h') => {
  const params = { horizon };
  if (stationId) params.station_id = stationId;
  const res = await api.get('/api/probabilistic/availability', { params });
  return res.data;
};

export const getProbCalibration = async () => {
  const res = await api.get('/api/probabilistic/calibration');
  return res.data;
};

export const getProbSummary = async () => {
  const res = await api.get('/api/probabilistic/summary');
  return res.data;
};

export default api;
