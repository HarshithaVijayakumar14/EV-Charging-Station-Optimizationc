"""
ml/station_recommender.py
Intelligent nearby station recommendation engine.
When a selected station has high occupancy risk or zero estimated free points,
identifies, filters by vehicle compatibility, estimates future availability,
and ranks nearby alternatives using geodesic distances.
Never invents driving directions or unverified capacities.
"""

import math
from typing import Dict, List, Optional
from backend.services.data_service import data_service
from ml.availability_prediction import availability_service
from ml.charging_time_estimator import calculate_charging_time
from ml.station_compatibility import check_vehicle_station_compatibility


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> Dict[str, float]:
    """
    Computes great-circle distance between two GPS coordinates in miles and kilometers.
    """
    R_km = 6371.0
    R_miles = 3958.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))

    return {
        "miles": round(R_miles * c, 2),
        "km": round(R_km * c, 2),
    }


class StationRecommenderService:
    def recommend_nearby_alternatives(
        self,
        target_station_id: str,
        vehicle: Dict,
        target_iso_pt: Optional[str] = None,
        current_soc_pct: float = 20.0,
        target_soc_pct: float = 80.0,
        max_radius_miles: float = 5.0,
        max_results: int = 4,
    ) -> List[Dict]:
        """
        Finds and ranks compatible nearby alternative stations when target station is crowded.
        """
        target_meta = data_service.get_station_detail(target_station_id)
        if target_meta is None:
            return []

        origin_lat = float(target_meta["latitude"])
        origin_lon = float(target_meta["longitude"])

        all_stations = data_service.get_stations_list()
        candidates: List[Dict] = []

        for st in all_stations:
            st_id = st["station_id"]
            if st_id == target_station_id:
                continue

            # Geodesic distance
            st_lat = float(st["latitude"])
            st_lon = float(st["longitude"])
            dist = haversine_distance(origin_lat, origin_lon, st_lat, st_lon)
            if dist["miles"] > max_radius_miles:
                continue

            # Compatibility check
            compat = check_vehicle_station_compatibility(vehicle, st_id)
            if not compat["is_compatible"]:
                continue

            # Availability estimate at requested time
            if target_iso_pt:
                avail = availability_service.estimate_at_target_datetime(st_id, target_iso_pt)
            else:
                avail = availability_service.estimate_station_availability_1h(st_id)

            est_free = float(avail.get("estimated_free_points") or 0.0)
            p_zero = float(avail.get("probability_zero_free_points") or 0.5)
            obs_pts = int(avail.get("observed_point_count") or 2)

            # Charging time estimate at this station
            st_power = compat["station_max_power_kw"]
            veh_max_power = float(vehicle.get("maximum_ac_charging_power_kw", 6.6))
            batt_cap = float(vehicle.get("battery_capacity_kwh", 60.0))
            eff = float(vehicle.get("charging_efficiency", 0.90))

            chg_time = calculate_charging_time(
                battery_capacity_kwh=batt_cap,
                current_soc_pct=current_soc_pct,
                target_soc_pct=target_soc_pct,
                station_power_kw=st_power,
                vehicle_max_power_kw=veh_max_power,
                efficiency=eff,
            )

            # Composite ranking score:
            # Rewards: higher free points, closer distance, higher power
            # Penalizes: high risk of full station
            free_ratio = min(est_free / max(obs_pts, 1), 1.0)
            dist_factor = max(0.0, 1.0 - (dist["miles"] / max_radius_miles))
            risk_penalty = p_zero

            rank_score = (
                0.45 * free_ratio +
                0.35 * dist_factor +
                0.10 * (st_power / 6.6) -
                0.10 * risk_penalty
            ) * 100.0

            reasons: List[Dict] = []
            if dist["miles"] <= 0.5:
                reasons.append(f"Very close proximity ({dist['miles']} mi / {dist['km']} km away)")
            else:
                reasons.append(f"{dist['miles']} miles away")

            if est_free >= 1.0:
                reasons.append(f"High availability ({est_free:.1f} of {obs_pts} observed points estimated free)")
            elif est_free >= 0.3:
                reasons.append(f"Moderate availability ({est_free:.1f} estimated free)")
            else:
                reasons.append("Limited availability")

            if compat.get("adapter_required"):
                reasons.append("Compatible with OEM J1772 adapter")
            else:
                reasons.append("Directly compatible connector")

            candidates.append(
                {
                    "station_id": st_id,
                    "short_name": st["short_name"],
                    "address": st["address"],
                    "latitude": st_lat,
                    "longitude": st_lon,
                    "distance_miles": dist["miles"],
                    "distance_km": dist["km"],
                    "rank_score": round(rank_score, 1),
                    "estimated_free_points": est_free,
                    "estimated_free_points_rounded": int(round(est_free)),
                    "observed_point_count": obs_pts,
                    "probability_zero_free_points": round(p_zero, 3),
                    "availability_status": avail.get("availability_status", "Available"),
                    "compatibility_status": compat["compatibility_status"],
                    "compatible_connectors": compat["compatible_connectors"],
                    "charging_power_kw": st_power,
                    "estimated_charging_time_formatted": chg_time["estimated_duration_formatted"],
                    "reasons": reasons,
                }
            )

        # Sort descending by rank score
        candidates.sort(key=lambda x: x["rank_score"], reverse=True)
        return candidates[:max_results]


station_recommender = StationRecommenderService()

if __name__ == "__main__":
    test_v = {
        "manufacturer": "Tesla",
        "model": "Model 3",
        "connector_type": "J1772 (via OEM adapter) / NACS",
        "battery_capacity_kwh": 60.0,
        "maximum_ac_charging_power_kw": 11.5,
    }
    recs = station_recommender.recommend_nearby_alternatives(
        target_station_id="PALO ALTO CA / HAMILTON #2",
        vehicle=test_v,
    )
    print(f"Found {len(recs)} recommended alternatives:")
    for r in recs:
        print(f"- {r['short_name']} ({r['distance_miles']} mi): Free={r['estimated_free_points']}/{r['observed_point_count']}, Score={r['rank_score']}")
