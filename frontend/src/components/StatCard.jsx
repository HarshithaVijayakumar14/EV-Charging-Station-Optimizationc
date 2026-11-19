import React from 'react';

export function StatCard({ title, value, subtitle, icon: Icon, trend, accent = 'emerald' }) {
  const accentMap = {
    emerald: 'bg-emerald-50 text-emerald-700 border-emerald-100',
    blue: 'bg-blue-50 text-blue-700 border-blue-100',
    amber: 'bg-amber-50 text-amber-700 border-amber-100',
    indigo: 'bg-indigo-50 text-indigo-700 border-indigo-100',
    slate: 'bg-slate-100 text-slate-700 border-slate-200',
    rose: 'bg-rose-50 text-rose-700 border-rose-100',
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">{title}</p>
          <p className="mt-2 text-2xl font-bold text-slate-900 tracking-tight truncate">{value}</p>
          {subtitle && <p className="mt-1 text-xs text-slate-500 leading-relaxed">{subtitle}</p>}
          {trend && <div className="mt-2">{trend}</div>}
        </div>
        {Icon && (
          <div className={`p-2.5 rounded-lg border shrink-0 ${accentMap[accent] || accentMap.emerald}`}>
            <Icon className="w-5 h-5" />
          </div>
        )}
      </div>
    </div>
  );
}
