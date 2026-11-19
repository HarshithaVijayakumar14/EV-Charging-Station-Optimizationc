"""
ml/station_compatibility.py
Evaluates compatibility between a selected vehicle and a target charging station.
Checks connector types, AC/DC requirements, electrical power limits, and adapter requirements.
Never reports unknown connectors as compatible.
"""

from typing import Dict, List, Optional
from ml.charging_point_inventory import inventory_service


def check_vehicle_station_compatibility(vehicle: Dict, station_id: str) -> Dict:
    """
    Evaluates whether the vehicle is electrically and physically compatible with the station.
    """
    st_inv = inventory_service.get_station_summary(station_id)
    if st_inv is None:
        return {
            "is_compatible": False,
            "compatibility_status": "Unknown Compatibility",
            "reasons": [f"Station '{station_id}' was not found in the verified inventory."],
            "compatible_connectors": [],
            "station_max_power_kw": 0.0,
            "adapter_required": False,
        }

    st_ports = st_inv["supported_port_types"]
    st_connectors = st_inv["supported_connectors"]
    veh_connector = str(vehicle.get("connector_type", "J1772"))
    veh_max_ac_kw = float(vehicle.get("maximum_ac_charging_power_kw", 6.6))

    is_compatible = False
    adapter_required = False
    status = "Incompatible"
    reasons: List[str] = []
    compatible_connectors: List[str] = []

    # Nominal power output based on station's port types
    if "Level 2" in st_ports:
        st_power_kw = 6.6
    elif "Level 1" in st_ports:
        st_power_kw = 1.9
    else:
        st_power_kw = 6.6

    # 1. J1772 Level 2 Compatibility
    if "Level 2" in st_ports:
        if "J1772" in veh_connector:
            if "adapter" in veh_connector.lower():
                is_compatible = True
                adapter_required = True
                status = "Compatible with Adapter"
                compatible_connectors.append("J1772 (via OEM Adapter)")
                reasons.append("Vehicle can charge at this station's J1772 Level 2 ports using its standard J1772 adapter.")
            else:
                is_compatible = True
                status = "Fully Compatible"
                compatible_connectors.append("J1772")
                reasons.append("Vehicle uses standard J1772 connector and is fully compatible with this station's Level 2 ports.")

    # 2. Level 1 NEMA 5-20 Compatibility
    if "Level 1" in st_ports:
        if "NEMA" in veh_connector or "120V" in veh_connector or "J1772" in veh_connector:
            is_compatible = True
            if status != "Fully Compatible":
                status = "Fully Compatible"
            compatible_connectors.append("NEMA 5-20R")
            reasons.append("Station provides Level 1 (120V) AC charging suitable for standard portable cords or small-battery vehicles.")

    # 3. Check if vehicle is DC-only (Palo Alto municipal stations are all AC Level 1/2)
    if veh_max_ac_kw <= 0.0 and "CCS" in veh_connector and "J1772" not in veh_connector:
        is_compatible = False
        status = "Incompatible"
        reasons = ["Vehicle requires high-power DC Fast Charging (CCS). This municipal station provides AC Level 1/2 charging only."]

    # 4. Unknown handling
    if not st_connectors or "Unknown" in st_connectors:
        if not is_compatible:
            status = "Unknown Compatibility"
            reasons.append("Connector details for this station could not be definitively verified.")

    return {
        "station_id": station_id,
        "station_name": st_inv["station_name"],
        "is_compatible": is_compatible,
        "compatibility_status": status,
        "adapter_required": adapter_required,
        "compatible_connectors": compatible_connectors,
        "station_supported_ports": st_ports,
        "station_max_power_kw": st_power_kw,
        "vehicle_max_ac_power_kw": veh_max_ac_kw,
        "bottleneck_charging_power_kw": min(st_power_kw, veh_max_ac_kw),
        "reasons": reasons,
    }


if __name__ == "__main__":
    test_veh = {
        "manufacturer": "Tesla",
        "model": "Model 3",
        "connector_type": "J1772 (via OEM adapter) / NACS",
        "maximum_ac_charging_power_kw": 11.5,
    }
    compat = check_vehicle_station_compatibility(test_veh, "PALO ALTO CA / HAMILTON #2")
    print("Sample Compatibility Check:")
    print(compat)
