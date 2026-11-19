from typing import Dict, List

from backend.database.db import log_recommendations
from backend.services.data_service import data_service
from backend.services.prediction_service import prediction_service

PLANNING_DISCLAIMER = (
    "Planning Suggestion Only: The City of Palo Alto dataset does not include verified station port "
    "capacities, real-time operational port counts, queue lengths, turn-aways, or maintenance logs. "
    "Recommendations reflect historical and predicted session arrival patterns and must not be "
    "interpreted as confirmed congestion, utilization percentages, or definitive infrastructure requirements."
)


class RecommendationService:
    """
    Explainable, rule-based EV charging planning recommendation engine.
    Combines Phase 5 XGBoost predictions (1h & 24h) with verified historical statistics.
    Never invents capacity values or congestion claims.
    """

    def generate_recommendations(self, persist: bool = False) -> Dict:
        pred_1h = prediction_service.predict_all_stations_1h(persist=False)
        pred_24h = prediction_service.predict_all_stations_24h(persist=False)
        stations_list = data_service.get_stations_list()

        p1_map = {r["station_id"]: r for r in pred_1h["predictions"]}
        p24_map = {r["station_id"]: r for r in pred_24h["predictions"]}

        recommendations: List[Dict] = []

        # ------------------------------------------------------------------
        # Rule 1: High Predicted Demand vs. Station's Own Historical Baseline
        # ------------------------------------------------------------------
        for st_id, r1 in p1_map.items():
            if r1["is_cold_start"]:
                continue
            hist_hr = r1["historical_hour_avg"]
            pred_val = r1["predicted_sessions"]
            if pred_val >= 0.25 and hist_hr > 0 and pred_val >= 1.30 * hist_hr:
                ratio = round(pred_val / hist_hr, 2)
                recommendations.append(
                    {
                        "id": f"high_1h_{st_id}",
                        "station_id": st_id,
                        "short_name": r1["short_name"],
                        "category": "High Predicted Demand (1-Hour)",
                        "priority": "High",
                        "label": "Planning Suggestion",
                        "message": "Predicted demand is higher than this station's historical average. Consider monitoring peak-hour operations.",
                        "reason": (
                            f"1-hour-ahead XGBoost predicts {pred_val:.2f} session arrivals at hour {r1['target_hour']:02d}:00 "
                            f"({r1['target_timestamp_pt']}), which is {ratio}x this station's historical average of "
                            f"{hist_hr:.2f} arrivals at that hour."
                        ),
                        "metrics": {
                            "predicted_1h_sessions": pred_val,
                            "historical_hour_avg": hist_hr,
                            "ratio_vs_history": ratio,
                            "target_timestamp_pt": r1["target_timestamp_pt"],
                        },
                    }
                )

        # ------------------------------------------------------------------
        # Rule 2: Day-Ahead Planning — Top 5 Stations by 24h-Ahead Forecast
        # ------------------------------------------------------------------
        established_24h = [r for r in pred_24h["predictions"] if not r["is_cold_start"]]
        top_24h = established_24h[:5]
        for rank, r24 in enumerate(top_24h, 1):
            meta = data_service.get_station_detail(r24["station_id"])
            recommendations.append(
                {
                    "id": f"day_ahead_{r24['station_id']}",
                    "station_id": r24["station_id"],
                    "short_name": r24["short_name"],
                    "category": "Day-Ahead Planning (24-Hour)",
                    "priority": "High" if rank <= 2 else "Medium",
                    "label": "Planning Suggestion",
                    "message": (
                        "Day-ahead forecast places this station among the highest expected arrival volumes "
                        "over the next 24 hours. Consider prioritizing operational readiness during its predicted peak hour."
                    ),
                    "reason": (
                        f"24-hour-ahead XGBoost forecasts {r24['total_predicted_sessions_24h']:.2f} total arrivals "
                        f"(~{r24['total_predicted_sessions_24h_rounded']} sessions; Rank #{rank} network-wide), "
                        f"peaking at {r24['peak_predicted_hour']:02d}:00 ({r24['peak_predicted_sessions']:.2f} arrivals/hr). "
                        f"Historical active-day average is {meta['sessions_per_active_day']:.2f} sessions/day."
                    ),
                    "metrics": {
                        "predicted_24h_total": r24["total_predicted_sessions_24h"],
                        "peak_predicted_hour": r24["peak_predicted_hour"],
                        "peak_predicted_sessions": r24["peak_predicted_sessions"],
                        "historical_sessions_per_active_day": meta["sessions_per_active_day"],
                    },
                }
            )

        # ------------------------------------------------------------------
        # Rule 3: Recurring Peak-Hour Demand Concentration (Historical Pattern)
        # ------------------------------------------------------------------
        for s in stations_list:
            if s["is_cold_start"] or s["active_days"] < 180:
                continue
            meta = data_service.get_station_detail(s["station_id"])
            pk_mean = meta["peak_hour_mean"]
            ov_mean = meta["overall_hourly_mean"]
            if pk_mean >= 0.45 and ov_mean > 0 and (pk_mean / ov_mean) >= 1.9:
                pk_hrs_str = ", ".join(f"{h:02d}:00" for h in meta["peak_hours"])
                recommendations.append(
                    {
                        "id": f"recurring_peak_{s['station_id']}",
                        "station_id": s["station_id"],
                        "short_name": s["short_name"],
                        "category": "Recurring Peak-Hour Demand",
                        "priority": "Medium",
                        "label": "Planning Suggestion",
                        "message": "This station has recurring peak-hour demand. Review whether existing charging capacity is sufficient.",
                        "reason": (
                            f"Historically averages {pk_mean:.2f} arrivals/hr at its top hour ({meta['peak_hours'][0]:02d}:00) "
                            f"— {pk_mean / ov_mean:.1f}x its overall hourly average ({ov_mean:.2f}/hr). "
                            f"Top 3 recurring peak hours: {pk_hrs_str} across {meta['active_days']:,} active days."
                        ),
                        "metrics": {
                            "peak_hours": meta["peak_hours"],
                            "peak_hour_mean": pk_mean,
                            "overall_hourly_mean": ov_mean,
                            "concentration_ratio": round(pk_mean / ov_mean, 2),
                        },
                    }
                )

        # ------------------------------------------------------------------
        # Rule 4: Consistently Low Observed Historical Demand (Established Sites)
        # ------------------------------------------------------------------
        for s in stations_list:
            if s["is_cold_start"] or s["active_days"] < 180:
                continue
            if s["sessions_per_active_day"] < 2.2:
                r24 = p24_map.get(s["station_id"])
                p24_str = (
                    f" 24-hour-ahead forecast is {r24['total_predicted_sessions_24h']:.2f} arrivals."
                    if r24
                    else ""
                )
                recommendations.append(
                    {
                        "id": f"low_activity_{s['station_id']}",
                        "station_id": s["station_id"],
                        "short_name": s["short_name"],
                        "category": "Low Observed Historical Demand",
                        "priority": "Low",
                        "label": "Planning Suggestion",
                        "message": "This station has relatively low observed demand. Review utilization before planning additional infrastructure.",
                        "reason": (
                            f"Observed average is {s['sessions_per_active_day']:.2f} sessions per active day "
                            f"({s['total_sessions']:,} total sessions over {s['active_days']:,} active days; "
                            f"overall hourly mean {s['overall_hourly_mean']:.4f}/hr).{p24_str} "
                            f"Note: low recorded counts do not rule out unobserved downtime or site access limits."
                        ),
                        "metrics": {
                            "sessions_per_active_day": s["sessions_per_active_day"],
                            "total_sessions": s["total_sessions"],
                            "active_days": s["active_days"],
                            "predicted_24h_total": r24["total_predicted_sessions_24h"] if r24 else None,
                        },
                    }
                )

        # ------------------------------------------------------------------
        # Rule 5: Cold-Start Stations (Not in 2011–2018 Training Data)
        # ------------------------------------------------------------------
        cold_stations = [s for s in stations_list if s["is_cold_start"]]
        if cold_stations:
            insuff_cnt = sum(1 for s in cold_stations if not s["has_sufficient_history"])
            recommendations.append(
                {
                    "id": "cold_start_sherman_group",
                    "station_id": "SHERMAN GROUP (13 NEW STATIONS)",
                    "short_name": "Sherman Ave Group (13 New Sites)",
                    "category": "Cold-Start Data Monitoring",
                    "priority": "Info",
                    "label": "Planning Suggestion",
                    "message": (
                        "Newly opened stations were absent from 2011–2018 model training data. "
                        "Monitor initial operating data and retrain models before relying on forecasts for planning."
                    ),
                    "reason": (
                        f"{len(cold_stations)} stations (Sherman Ave, opened Dec 2020) have only 1–8 observed sessions each "
                        f"({insuff_cnt} stations have <24h history and cannot produce lag-based forecasts yet)."
                    ),
                    "metrics": {
                        "cold_start_station_count": len(cold_stations),
                        "insufficient_history_count": insuff_cnt,
                        "station_ids": [s["station_id"] for s in cold_stations],
                    },
                }
            )

        # Sort by priority (High -> Medium -> Low -> Info)
        prio_order = {"High": 0, "Medium": 1, "Low": 2, "Info": 3}
        recommendations.sort(key=lambda x: (prio_order.get(x["priority"], 9), x["station_id"]))

        if persist and recommendations:
            log_recommendations(recommendations)

        category_counts: Dict[str, int] = {}
        for r in recommendations:
            category_counts[r["category"]] = category_counts.get(r["category"], 0) + 1

        return {
            "disclaimer": PLANNING_DISCLAIMER,
            "total_recommendations": len(recommendations),
            "category_counts": category_counts,
            "recommendations": recommendations,
        }


recommendation_service = RecommendationService()
