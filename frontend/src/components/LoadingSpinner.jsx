import React from 'react';
import { Loader2, AlertCircle, Server, ExternalLink } from 'lucide-react';
import { API_BASE_URL } from '../services/api';

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
  const is404OrNetwork = message && (message.includes('404') || message.includes('Network Error'));

  return (
    <div className="bg-white border border-rose-200 rounded-2xl shadow-sm p-6 sm:p-8 max-w-xl mx-auto my-10 text-center">
      <div className="w-12 h-12 bg-rose-100 rounded-full flex items-center justify-center mx-auto mb-4 text-rose-600">
        <AlertCircle className="w-6 h-6" />
      </div>
      <h3 className="text-lg font-bold text-slate-900">Backend API Offline or Unreachable</h3>
      <p className="mt-2 text-sm text-slate-600 leading-relaxed">
        {isGitHubPages && is404OrNetwork
          ? 'GitHub Pages hosts the static React interface, but cannot run Python ML models or Flask APIs directly. A live backend service is required to serve charging data and predictions.'
          : (message || 'Unable to connect to the backend API. Please ensure the Flask backend service is online.')}
      </p>

      {isGitHubPages && (
        <div className="mt-5 p-4 bg-slate-50 border border-slate-200 rounded-xl text-left text-xs text-slate-600 space-y-2">
          <div className="font-semibold text-slate-800 flex items-center gap-1.5">
            <Server className="w-3.5 h-3.5 text-emerald-600" />
            Two Ways to Run the Complete Application:
          </div>
          <div className="pl-4">
            <span className="font-medium text-slate-700">1. Local Full-Stack Server:</span>
            <span className="text-slate-500 block">Run <code className="bg-slate-200 px-1 rounded text-slate-800">python backend/app.py</code> on your computer and open <a href="http://localhost:5000" target="_blank" rel="noreferrer" className="text-emerald-600 underline font-medium">http://localhost:5000</a> to view all live models & predictions.</span>
          </div>
          <div className="pl-4">
            <span className="font-medium text-slate-700">2. Free Cloud Backend (Render):</span>
            <span className="text-slate-500 block">Deploy the backend repository to Render with 1-click and set <code className="bg-slate-200 px-1 rounded text-slate-800">VITE_API_BASE_URL</code> in GitHub repository variables.</span>
          </div>
          {API_BASE_URL && (
            <div className="pt-2 border-t border-slate-200 text-slate-400">
              Target API URL: <code className="text-slate-600">{API_BASE_URL}</code>
            </div>
          )}
        </div>
      )}

      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-5 py-2.5 bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium rounded-xl shadow-sm transition-colors cursor-pointer"
          >
            Retry Connection
          </button>
        )}
        {isGitHubPages && (
          <a
            href="https://render.com/deploy?repo=https://github.com/HarshithaVijayakumar14/EV-Charging-Station-Optimizationc"
            target="_blank"
            rel="noreferrer"
            className="inline-flex items-center gap-1.5 px-5 py-2.5 bg-emerald-600 hover:bg-emerald-700 text-white text-sm font-medium rounded-xl shadow-sm transition-colors"
          >
            Deploy Backend to Render <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>
    </div>
  );
}
