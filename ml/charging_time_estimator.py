"""
ml/charging_time_estimator.py
Calculates required energy and estimates charging duration for a given vehicle,
current battery percentage, target percentage, and available station charging power.
Models the CC/CV battery management taper curve above 80% SOC.
"""

from typing import Dict, Optional


def calculate_charging_time(
    battery_capacity_kwh: float,
    current_soc_pct: float,
    target_soc_pct: float,
    station_power_kw: float,
    vehicle_max_power_kw: float,
    efficiency: float = 0.90,
    has_charging_curve: bool = True,
) -> Dict:
    """
    Calculates required energy and estimated charging duration.
    """
    if target_soc_pct <= current_soc_pct:
        return {
            "required_energy_kwh": 0.0,
            "estimated_duration_minutes": 0.0,
            "estimated_duration_formatted": "0 min",
            "effective_charging_power_kw": 0.0,
            "is_tapered": False,
            "notes": "Target state-of-charge is less than or equal to current state-of-charge.",
        }

    # Validate inputs
    soc_start = max(0.0, min(current_soc_pct, 100.0))
    soc_end = max(0.0, min(target_soc_pct, 100.0))
    delta_soc = soc_end - soc_start

    # Net electrical energy required
    required_energy_kwh = round(battery_capacity_kwh * (delta_soc / 100.0), 3)

    # Effective bottleneck charging power
    effective_kw = min(float(station_power_kw), float(vehicle_max_power_kw)) * float(efficiency)
    if effective_kw <= 1e-4:
        effective_kw = 1.0  # Fallback

    # Split duration across bulk phase (<80%) and taper phase (80-100%)
    duration_hours = 0.0
    is_tapered = False

    if has_charging_curve:
        # Phase 1: 0% to min(soc_end, 80%)
        soc_bulk_end = min(soc_end, 80.0)
        if soc_bulk_end > soc_start:
            bulk_delta = soc_bulk_end - soc_start
            bulk_energy = battery_capacity_kwh * (bulk_delta / 100.0)
            duration_hours += bulk_energy / effective_kw

        # Phase 2: 80% to soc_end (taper region where BMS reduces current)
        if soc_end > 80.0:
            is_tapered = True
            taper_start = max(80.0, soc_start)
            taper_delta = soc_end - taper_start
            taper_energy = battery_capacity_kwh * (taper_delta / 100.0)
            # Average power in 80-100% window is approximately 60% of peak due to CC/CV transition
            taper_effective_kw = effective_kw * 0.60
            duration_hours += taper_energy / taper_effective_kw
    else:
        # Linear approximation
        duration_hours = required_energy_kwh / effective_kw

    duration_minutes = round(duration_hours * 60.0, 1)

    # Format into human-readable string
    hrs = int(duration_minutes // 60)
    mins = int(duration_minutes % 60)
    if hrs > 0:
        formatted = f"{hrs} hr {mins} min"
    else:
        formatted = f"{mins} min"

    notes = (
        "Calculated using battery CC/CV curve (includes 80-100% thermal and voltage taper slow-down)."
        if is_tapered
        else "Calculated at constant nominal charging rate with thermal efficiency loss."
    )
    if not has_charging_curve:
        notes = "Approximate linear estimate (detailed vehicle charging curve not available)."

    return {
        "required_energy_kwh": required_energy_kwh,
        "effective_charging_power_kw": round(effective_kw, 2),
        "estimated_duration_hours": round(duration_hours, 2),
        "estimated_duration_minutes": duration_minutes,
        "estimated_duration_formatted": formatted,
        "is_tapered": is_tapered,
        "efficiency_factor": efficiency,
        "notes": notes,
    }


if __name__ == "__main__":
    test_res = calculate_charging_time(
        battery_capacity_kwh=60.0,
        current_soc_pct=20.0,
        target_soc_pct=90.0,
        station_power_kw=6.6,
        vehicle_max_power_kw=11.5,
        efficiency=0.90,
    )
    print("Sample Charging Time Estimation:")
    print(test_res)
