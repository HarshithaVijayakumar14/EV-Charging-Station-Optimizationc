import React from 'react';
import { Info } from 'lucide-react';

export function DisclaimerBanner({ text }) {
  const defaultText =
    'Planning Suggestion Only: The City of Palo Alto dataset measures charging session arrival counts and excludes verified port counts, physical station capacities, queue lengths, or maintenance downtime. Forecasts reflect expected session arrivals, not simultaneous occupancy or confirmed congestion.';

  return (
    <div className="bg-amber-50/90 border border-amber-200/90 rounded-xl p-4 flex items-start gap-3 text-amber-900">
      <Info className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
      <div className="text-xs sm:text-sm leading-relaxed">
        <span className="font-semibold">Methodological &amp; Capacity Notice: </span>
        {text || defaultText}
      </div>
    </div>
  );
}
