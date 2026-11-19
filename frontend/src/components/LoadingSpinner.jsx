import React from 'react';
import { Loader2, AlertCircle } from 'lucide-react';

export function LoadingSpinner({ message = 'Loading live EV network data...' }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-slate-500">
      <Loader2 className="w-8 h-8 animate-spin text-emerald-600 mb-3" />
      <p className="text-sm font-medium">{message}</p>
    </div>
  );
}

export function ErrorState({ message, onRetry }) {
  return (
    <div className="bg-rose-50 border border-rose-200 rounded-xl p-6 text-center max-w-lg mx-auto my-12">
      <AlertCircle className="w-8 h-8 text-rose-600 mx-auto mb-2" />
      <h3 className="text-base font-semibold text-rose-900">Unable to load data</h3>
      <p className="mt-1 text-sm text-rose-700">{message || 'Make sure the Flask backend server is running on port 5000.'}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="mt-4 px-4 py-2 bg-rose-600 hover:bg-rose-700 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Retry Connection
        </button>
      )}
    </div>
  );
}
