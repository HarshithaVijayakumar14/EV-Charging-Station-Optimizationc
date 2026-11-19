import React, { useState } from 'react';
import { Loader2, AlertCircle, Server, ExternalLink, Settings, RefreshCw, CheckCircle2 } from 'lucide-react';
import { API_BASE_URL, setApiBaseUrl } from '../services/api';

export function LoadingSpinner({ message = 'Loading live EV network data...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-slate-500">
      <Loader2 className="w-8 h-8 animate-spin text-emerald-600 mb-3" />
      <p className="text-sm font-medium">{message}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  const isGitHubPages = typeof window !== 'undefined' && window.location.hostname.includes('github.io');
  const is404OrNetwork = message && (message.includes('404') || message.includes('Network Error') || message.includes('timeout'));
  const [customUrl, setCustomUrl] = useState(API_BASE_URL || 'https://ev-charging-station-api.onrender.com');
  const [showConfig, setShowConfig] = useState(false);
  const [savedNotice, setSavedNotice] = useState(false);

  const handleSaveUrl = (e) => {
    e?.preventDefault();
    setApiBaseUrl(customUrl);
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 3000);
    if (onRetry) onRetry();
  };

  return (
    <div className="bg-white border border-rose-200 rounded-2xl shadow-sm p-6 sm:p-8 max-w-xl mx-auto my-10 text-center">
      <div className="w-12 h-12 bg-rose-100 rounded-full flex items-center justify-center mx-auto mb-4 text-rose-600">
        <AlertCircle className="w-6 h-6" />
      </div>
      <h3 className="text-lg font-bold text-slate-900">Backend API Connecting or Offline</h3>
      <p className="mt-2 text-sm text-slate-600 leading-relaxed">
        {isGitHubPages && is404OrNetwork
          ? 'GitHub Pages hosts the static React interface. If you deployed your Flask backend on Render free tier, it automatically spins down when idle and takes ~45–60 seconds to wake up on the first request.'
          : (message || 'Unable to connect to the backend API. Please ensure the Flask backend service is online.')}
      </p>

      {/* Target API Endpoint Configuration Box */}
      <div className="mt-5 p-4 bg-slate-50 border border-slate-200 rounded-xl text-left text-xs text-slate-600 space-y-3">
        <div className="flex items-center justify-between">
          <span className="font-semibold text-slate-800 flex items-center gap-1.5">
            <Server className="w-3.5 h-3.5 text-emerald-600" />
            Backend API Endpoint:
          </span>
          <button
            type="button"
            onClick={() => setShowConfig(!showConfig)}
            className="text-[11px] font-medium text-emerald-700 hover:text-emerald-800 flex items-center gap-1 cursor-pointer"
          >
            <Settings className="w-3 h-3" />
            {showConfig ? 'Hide Config' : 'Change URL'}
          </button>
        </div>

        <div className="text-slate-500 break-all font-mono bg-white p-2 rounded border border-slate-200 text-[11px]">
          {API_BASE_URL || '(same-host / relative proxy)'}
        </div>

        {showConfig && (
          <form onSubmit={handleSaveUrl} className="pt-2 border-t border-slate-200 space-y-2">
            <label className="block text-[11px] font-semibold text-slate-700">
              Configure Production Backend URL:
            </label>
            <div className="flex gap-2">
              <input
                type="url"
                value={customUrl}
                onChange={(e) => setCustomUrl(e.target.value)}
                placeholder="https://your-backend.onrender.com"
                className="flex-1 px-2.5 py-1.5 text-xs rounded border border-slate-300 focus:outline-emerald-500 bg-white"
              />
              <button
                type="submit"
                className="px-3 py-1.5 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold rounded text-xs transition-colors cursor-pointer"
              >
                Save &amp; Reconnect
              </button>
            </div>
            {savedNotice && (
              <p className="text-[11px] text-emerald-700 font-medium flex items-center gap-1">
                <CheckCircle2 className="w-3.5 h-3.5" /> URL saved to browser! Retrying request...
              </p>
            )}
          </form>
        )}

        {isGitHubPages && (
          <div className="pt-2 border-t border-slate-200 text-slate-500 text-[11px] space-y-1">
            <p><strong>Note for Render Free Tier:</strong> If the service was asleep, the first request wakes it up. Wait 45–60 seconds and click <strong>Retry Connection</strong> below.</p>
          </div>
        )}
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium rounded-xl shadow-sm transition-colors cursor-pointer flex items-center gap-1.5"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Retry Connection
          </button>
        )}
        {isGitHubPages && (
          <a
            href="https://render.com/deploy?repo=https://github.com/HarshithaVijayakumar14/EV-Charging-Station-Optimizationc"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-5 py-2.5 bg-slate-900 hover:bg-slate-800 text-white text-sm font-medium rounded-xl shadow-sm transition-colors"
          >
            Deploy Backend to Render <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>
    </div>
  );
}
