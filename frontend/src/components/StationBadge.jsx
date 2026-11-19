import React from 'react';
import { CheckCircle2, AlertTriangle, Snowflake } from 'lucide-react';

export function StationBadge({ isColdStart, hasSufficientHistory = true, size = 'sm' }) {
  const sizeClass = size === 'xs' ? 'text-[10px] px-1.5 py-0.5' : 'text-xs px-2.5 py-0.5';

  if (!hasSufficientHistory) {
    return (
      <span
        className={`inline-flex items-center gap-1 font-medium rounded-full bg-rose-50 text-rose-700 border border-rose-200 ${sizeClass}`}
        title="Less than 24 observed hours available — insufficient history for lag-based forecasts"
      >
        <AlertTriangle className="w-3 h-3" />
        Insufficient History
      </span>
    );
  }

  if (isColdStart) {
    return (
      <span
        className={`inline-flex items-center gap-1 font-medium rounded-full bg-amber-50 text-amber-800 border border-amber-200 ${sizeClass}`}
        title="Opened Dec 2020 — not present in 2011–2018 training data"
      >
        <Snowflake className="w-3 h-3" />
        Cold-Start (New)
      </span>
    );
  }

  return (
    <span
      className={`inline-flex items-center gap-1 font-medium rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200 ${sizeClass}`}
      title="Present in historical 2011–2018 training data"
    >
      <CheckCircle2 className="w-3 h-3" />
      Established
    </span>
  );
}
