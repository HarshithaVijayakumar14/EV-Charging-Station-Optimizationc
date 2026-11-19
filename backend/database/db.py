import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional

from backend.config import DB_PATH


def get_connection() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create application-level SQLite tables for prediction and recommendation history."""
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                station_id TEXT NOT NULL,
                horizon TEXT NOT NULL,
                forecast_origin_pt TEXT NOT NULL,
                target_timestamp_pt TEXT NOT NULL,
                predicted_sessions REAL NOT NULL,
                predicted_sessions_rounded INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                is_cold_start INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_pred_hist_station_horizon
            ON prediction_history (station_id, horizon, created_at_utc DESC)
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recommendation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                station_id TEXT NOT NULL,
                category TEXT NOT NULL,
                priority TEXT NOT NULL,
                recommendation_text TEXT NOT NULL,
                rationale TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS availability_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                vehicle_model TEXT NOT NULL,
                vehicle_category TEXT NOT NULL,
                current_soc_pct REAL NOT NULL,
                target_soc_pct REAL NOT NULL,
                station_id TEXT NOT NULL,
                arrival_time_pt TEXT NOT NULL,
                estimated_free_points REAL,
                probability_zero_free REAL,
                availability_status TEXT NOT NULL,
                recommended_alternatives_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS charging_estimates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at_utc TEXT NOT NULL,
                vehicle_model TEXT NOT NULL,
                battery_capacity_kwh REAL NOT NULL,
                required_energy_kwh REAL NOT NULL,
                station_power_kw REAL NOT NULL,
                estimated_duration_minutes REAL NOT NULL
            )
            """
        )
        conn.commit()


def log_predictions(records: List[Dict]) -> int:
    """Persist a batch of generated prediction records into SQLite."""
    if not records:
        return 0
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (
            now_utc,
            r["station_id"],
            r["horizon"],
            r["forecast_origin_pt"],
            r["target_timestamp_pt"],
            float(r["predicted_sessions"]),
            int(r["predicted_sessions_rounded"]),
            r["model_name"],
            1 if r.get("is_cold_start") else 0,
        )
        for r in records
    ]
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO prediction_history (
                created_at_utc, station_id, horizon, forecast_origin_pt,
                target_timestamp_pt, predicted_sessions, predicted_sessions_rounded,
                model_name, is_cold_start
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def log_recommendations(records: List[Dict]) -> int:
    """Persist generated rule-based planning recommendations into SQLite."""
    if not records:
        return 0
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    rows = [
        (
            now_utc,
            r["station_id"],
            r["category"],
            r["priority"],
            r["message"],
            r["reason"],
        )
        for r in records
    ]
    with get_connection() as conn:
        conn.executemany(
            """
            INSERT INTO recommendation_history (
                created_at_utc, station_id, category, priority,
                recommendation_text, rationale
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
    return len(rows)


def get_recent_prediction_history(
    station_id: Optional[str] = None,
    horizon: Optional[str] = None,
    limit: int = 50,
) -> List[Dict]:
    query = "SELECT * FROM prediction_history WHERE 1=1"
    params: list = []
    if station_id:
        query += " AND station_id = ?"
        params.append(station_id)
    if horizon:
        query += " AND horizon = ?"
        params.append(horizon)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    with get_connection() as conn:
        rows = conn.execute(query, params).fetchall()
    return [dict(r) for r in rows]


def get_db_counts() -> Dict[str, int]:
    with get_connection() as conn:
        pred_cnt = conn.execute("SELECT COUNT(*) FROM prediction_history").fetchone()[0]
        rec_cnt = conn.execute("SELECT COUNT(*) FROM recommendation_history").fetchone()[0]
        avail_cnt = conn.execute("SELECT COUNT(*) FROM availability_requests").fetchone()[0]
        chg_cnt = conn.execute("SELECT COUNT(*) FROM charging_estimates").fetchone()[0]
    return {
        "prediction_history_rows": int(pred_cnt),
        "recommendation_history_rows": int(rec_cnt),
        "availability_requests_rows": int(avail_cnt),
        "charging_estimates_rows": int(chg_cnt),
    }


def log_availability_request(
    vehicle_model: str,
    vehicle_category: str,
    current_soc_pct: float,
    target_soc_pct: float,
    station_id: str,
    arrival_time_pt: str,
    estimated_free_points: Optional[float],
    probability_zero_free: Optional[float],
    availability_status: str,
    recommended_alternatives_count: int = 0,
) -> int:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO availability_requests (
                created_at_utc, vehicle_model, vehicle_category,
                current_soc_pct, target_soc_pct, station_id,
                arrival_time_pt, estimated_free_points, probability_zero_free,
                availability_status, recommended_alternatives_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                vehicle_model,
                vehicle_category,
                float(current_soc_pct),
                float(target_soc_pct),
                station_id,
                arrival_time_pt,
                float(estimated_free_points) if estimated_free_points is not None else None,
                float(probability_zero_free) if probability_zero_free is not None else None,
                availability_status,
                int(recommended_alternatives_count),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0


def log_charging_estimate(
    vehicle_model: str,
    battery_capacity_kwh: float,
    required_energy_kwh: float,
    station_power_kw: float,
    estimated_duration_minutes: float,
) -> int:
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO charging_estimates (
                created_at_utc, vehicle_model, battery_capacity_kwh,
                required_energy_kwh, station_power_kw, estimated_duration_minutes
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                now_utc,
                vehicle_model,
                float(battery_capacity_kwh),
                float(required_energy_kwh),
                float(station_power_kw),
                float(estimated_duration_minutes),
            ),
        )
        conn.commit()
        return cur.lastrowid or 0
