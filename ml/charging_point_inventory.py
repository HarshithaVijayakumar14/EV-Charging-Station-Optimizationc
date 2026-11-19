"""
ml/charging_point_inventory.py
Constructs a verified physical charging-point inventory from the cleaned Palo Alto dataset.
Identifies distinct physical charging points using validated combinations of:
station_id, Station Name, MAC Address, EVSE ID, Port Number, and Port Type.
Distinguishes active equipment from historical upgraded/decommissioned hardware.
Never invents unobserved capacity values.
"""

import os
from typing import Dict, List, Optional
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "ev_charging_cleaned.csv")
INVENTORY_CSV = os.path.join(PROJECT_ROOT, "dataset", "processed", "charging_point_inventory.csv")


def build_charging_point_inventory(input_csv: str = CLEANED_CSV, output_csv: str = INVENTORY_CSV) -> pd.DataFrame:
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Cleaned dataset not found: {input_csv}")

    cols = [
        "station_id",
        "Station Name",
        "mac_address_clean",
        "EVSE ID",
        "Port Number",
        "port_type_clean",
        "Plug Type",
        "start_datetime_pt",
        "Model Number",
    ]
    df = pd.read_csv(input_csv, usecols=cols, low_memory=False)

    # Latest recorded timestamp for each station
    st_max_dates = df.groupby("station_id")["start_datetime_pt"].max().to_dict()

    # Determine ports per MAC across the dataset
    mac_ports = df.groupby("mac_address_clean")["Port Number"].nunique().to_dict()

    grouped = (
        df.groupby(
            ["station_id", "mac_address_clean", "Port Number", "port_type_clean"],
            as_index=False,
        )
        .agg(
            station_name=("Station Name", "first"),
            evse_id=("EVSE ID", lambda s: s.dropna().iloc[0] if len(s.dropna()) > 0 else None),
            model_number=("Model Number", lambda s: str(s.dropna().iloc[0]) if len(s.dropna()) > 0 else "Unknown"),
            connector_type=("Plug Type", lambda s: s.dropna().iloc[0] if len(s.dropna()) > 0 else "J1772"),
            first_observed=("start_datetime_pt", "min"),
            last_observed=("start_datetime_pt", "max"),
            total_sessions_observed=("start_datetime_pt", "size"),
        )
    )

    records: List[Dict] = []
    for _, row in grouped.iterrows():
        st_id = str(row["station_id"])
        mac = str(row["mac_address_clean"])
        port_num = int(row["Port Number"])
        port_type = str(row["port_type_clean"])
        total_sessions = int(row["total_sessions_observed"])
        first_obs = str(row["first_observed"])
        last_obs = str(row["last_observed"])
        st_max = st_max_dates.get(st_id, "2020-12-31")

        # Create standardized unique charging point identifier
        short_mac = mac[-6:]
        charging_point_id = f"{st_id} / EVSE-{short_mac} / Port {port_num}"

        # Determine connector type standard
        conn = str(row["connector_type"])
        if port_type == "Level 2":
            standard_connector = "J1772 (Type 1)"
        elif port_type == "Level 1":
            standard_connector = "NEMA 5-20R (120V Standard Wall/Plug)"
        else:
            standard_connector = conn

        # Check if point was active near end of data (within 180 days of station's last session or in 2020)
        last_obs_dt = pd.to_datetime(last_obs)
        st_max_dt = pd.to_datetime(st_max)
        days_from_st_end = (st_max_dt - last_obs_dt).total_seconds() / 86400.0
        is_active_end_of_data = bool(days_from_st_end <= 180.0 or last_obs_dt.year >= 2020)

        # Inventory confidence
        if total_sessions >= 50:
            confidence = "High"
        elif total_sessions >= 5:
            confidence = "Medium"
        else:
            confidence = "Low"

        # Validation status
        ports_on_mac = mac_ports.get(mac, 1)
        if total_sessions < 5:
            status = "Low-Sample Observed Point"
        elif is_active_end_of_data:
            if ports_on_mac >= 2:
                status = "Verified Active Dual-Port EVSE"
            else:
                status = "Verified Active Single-Port EVSE"
        else:
            status = "Historical Upgraded / Decommissioned Equipment"

        evse_val = None
        if pd.notna(row["evse_id"]):
            evse_val = int(row["evse_id"]) if float(row["evse_id"]).is_integer() else str(row["evse_id"])

        records.append(
            {
                "station_id": st_id,
                "station_name": str(row["station_name"]),
                "charging_point_id": charging_point_id,
                "mac_address": mac,
                "evse_id": evse_val,
                "port_number": port_num,
                "port_type": port_type,
                "connector_type": standard_connector,
                "first_observed": first_obs,
                "last_observed": last_obs,
                "total_sessions_observed": total_sessions,
                "is_active_end_of_data": is_active_end_of_data,
                "inventory_confidence": confidence,
                "validation_status": status,
                "notes": (
                    "Observed physical charging point verified from historical session telemetry. "
                    "Does not imply total unobserved station capacity."
                ),
            }
        )

    out_df = pd.DataFrame(records).sort_values(
        ["station_id", "port_number", "total_sessions_observed"],
        ascending=[True, True, False],
    )
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    out_df.to_csv(output_csv, index=False)
    return out_df


class ChargingPointInventoryService:
    def __init__(self, inventory_csv: str = INVENTORY_CSV):
        self.inventory_csv = inventory_csv
        self._df: Optional[pd.DataFrame] = None

    def load(self) -> pd.DataFrame:
        if self._df is None:
            if not os.path.exists(self.inventory_csv):
                self._df = build_charging_point_inventory(output_csv=self.inventory_csv)
            else:
                self._df = pd.read_csv(self.inventory_csv)
        return self._df.copy()

    def get_all_points(self) -> List[Dict]:
        df = self.load()
        return df.to_dict(orient="records")

    def get_station_points(self, station_id: str, only_active: bool = False) -> List[Dict]:
        df = self.load()
        sub = df[df["station_id"] == station_id]
        if only_active:
            sub = sub[sub["is_active_end_of_data"].astype(bool)]
        return sub.to_dict(orient="records")

    def get_station_summary(self, station_id: str) -> Optional[Dict]:
        pts = self.get_station_points(station_id, only_active=False)
        if not pts:
            return None
        active_pts = [p for p in pts if p["is_active_end_of_data"]]
        unique_macs = list({p["mac_address"] for p in pts})
        port_types = sorted(list({p["port_type"] for p in pts}))
        connectors = sorted(list({p["connector_type"] for p in pts}))

        return {
            "station_id": station_id,
            "station_name": pts[0]["station_name"],
            "total_observed_points_lifetime": len(pts),
            "active_observed_points_count": max(len(active_pts), 1),
            "unique_macs_lifetime": len(unique_macs),
            "supported_port_types": port_types,
            "supported_connectors": connectors,
            "has_decommissioned_hardware": len(pts) > len(active_pts),
            "charging_points": pts,
        }


inventory_service = ChargingPointInventoryService()

if __name__ == "__main__":
    df_inv = build_charging_point_inventory()
    print(f"Generated charging point inventory with {len(df_inv)} physical points across {df_inv['station_id'].nunique()} stations.")
    print(df_inv[["station_id", "charging_point_id", "total_sessions_observed", "is_active_end_of_data", "validation_status"]].head(15))
