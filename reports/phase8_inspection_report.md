# Phase 8: Part 1 — Equipment & Telemetry Inspection Report
## EV Charging Point Availability and Intelligent Station Recommendation

**Project:** Electric Vehicle Charging Station Optimization – Predicting Demand for EV Stations (Palo Alto, CA)  
**Document:** Phase 8, Part 1 — Pre-Implementation Inspection Report  
**Date:** November 2026  
**Status:** Complete & Verified  

---

## 1. Executive Summary

Before implementing the charging-point inventory, availability reconstruction, vehicle database, compatibility matrix, and recommendation modules in Phase 8, a comprehensive inspection of all datasets, models, scripts, database schemas, and frontend/backend code was conducted.

The primary objective of this inspection is to validate how charging equipment, identifiers, timestamps, and occupancy intervals are represented, and to uncover the physical reality of the charging network without making naive assumptions.

### Key Inspection Discoveries
1. **MAC Address $\ne$ Charging Point:** A MAC address in the ChargePoint network identifies an entire physical EVSE pedestal/bollard, which typically houses **two independent charging ports** (`Port Number 1` and `Port Number 2`). Across 259,411 sessions, `Port Number` is never missing and strictly equals `1` (105,936 sessions; 40.8%) or `2` (153,475 sessions; 59.2%).
2. **Missing Identifiers in Historical Telemetry:** While `station_id`, `MAC Address`, and `Port Number` are present in 100% of records, **`EVSE ID`, `Model Number`, and `System S/N` are missing in 78,947 rows (30.43% of the dataset)**, corresponding to early transaction exports prior to ~2015.
3. **Hardware Lifecycle & Decommissioning:** Across the 10-year timeline (2011–2020), municipal stations underwent hardware replacements and upgrades (e.g. replacing early CT2100 bollards with CT4000 series). Consequently, accumulating all historically observed `(MAC Address, Port Number)` pairs over 10 years yields up to 8 unique pairs at a site like `HAMILTON #2`. However, analysis of active years demonstrates that **in any given year, the station typically operated only 1 dual-port pedestal (2 concurrent ports)**.
4. **Equipment Mobility Across Stations:** Two MAC addresses appear under multiple `station_id` values:
   - `000D6F00009ED39E`: Logged 7,797 sessions at `HAMILTON #2` from 2011 to Jan 2016, and was subsequently relocated/reassigned to `BRYANT #3`, logging 4,475 sessions from Feb 2016 to 2019.
   - `000D6F00009D9A1B`: Logged 4,188 sessions at `BRYANT #1` from 2016 to 2019, with a single session logged at neighboring `BRYANT #2` on July 1, 2020.
5. **Network-Wide Equipment Quantities:**
   - 46 unique `station_id` values.
   - 68 unique clean MAC addresses.
   - 117 unique `(MAC Address, Port Number)` hardware pairs.
   - 120 unique `(station_id, MAC Address, Port Number)` historical deployment tuples.
6. **Occupancy vs. Active Charging:** For availability estimation, a charging point is physically **occupied** from `occupancy_start_utc` until `occupancy_end_utc` (unplug event). Active power draw ceases earlier at `active_charging_end_utc` when the battery reaches 100%, but the physical bay remains blocked to arriving vehicles.

---

## 2. Telemetry & Identifier Representation Matrix

| Attribute / Field | Cleaned Dataset Representation | Missing Count | Value Domain / Distribution | Physical Meaning & Caveats |
| :--- | :--- | :---: | :--- | :--- |
| **`station_id`** | String (e.g. `PALO ALTO CA / HAMILTON #2`) | 0 (0.0%) | 46 unique stations | Logical station location designated by the municipality. |
| **`MAC Address`** | Hex string (e.g. `00:0D:6F:00:00:9E:D3:9E`) | 0 (0.0%) | 83 raw strings / 68 clean 16-char hex | Network interface of the physical EVSE pedestal. One MAC serves 1 or 2 ports. |
| **`EVSE ID`** | Integer float (e.g. `11345.0`) | 78,947 (30.4%) | 51 unique IDs | ChargePoint back-office gateway ID. Missing in pre-2015 records. |
| **`Port Number`** | Integer (`1` or `2`) | 0 (0.0%) | `1`: 105,936 rows; `2`: 153,475 rows | Physical plug position on the dual-port bollard. |
| **`Port Type`** | String (`Level 2`, `Level 1`) | 0 (Cleaned) | `Level 2`: 98.2%; `Level 1`: 1.8% | AC charging power level. Cleaned in Phase 3 with 9 recovered values. |
| **`Plug Type`** | String (`J1772`, `NEMA 5-20R`) | 0 (0.0%) | `J1772`: 254,730; `NEMA 5-20R`: 4,681 | Standard physical connector type. J1772 for L2, NEMA 5-20 for L1. |
| **`Start Date` / UTC** | `start_datetime_utc` (ISO 8601) | 0 (0.0%) | 2011-07-29 to 2020-12-31 | Exact session connection timestamp converted to UTC. |
| **`End Date` / UTC** | `end_datetime_utc` (ISO 8601) | 0 (0.0%) | 2011-07-29 to 2021-01-01 | Raw session disconnection timestamp from network logs. |
| **`occupancy_end_utc`** | ISO 8601 UTC timestamp | 0 (0.0%) | Validated unplug timestamp | Derived in Phase 3 using `Start Date + Total Duration` to bypass batch upload lag. |
| **`active_charging_end_utc`** | ISO 8601 UTC timestamp | 0 (0.0%) | Power cutoff timestamp | Derived using `Start Date + Charging Time`. |
| **`flag_occupancy_interval_unreliable`** | Boolean (0 or 1) | 0 (0.0%) | 349 rows flagged (0.13%) | Flagged for negative durations, zero duration with non-zero energy, or extreme >24h errors. |

---

## 3. Detailed Equipment Mapping Analysis

### 3.1 Pedestal Architecture & Port Numbers
In the ChargePoint commercial network architecture deployed across Palo Alto:
- An **EVSE (Electric Vehicle Supply Equipment)** is a physical charging station pedestal or wall mount.
- Each EVSE communicates via a single ZigBee or cellular radio module assigned a **unique MAC Address**.
- Most municipal charging bollards are **dual-port units**:
  - `Port 1` delivers J1772 Level 2 (or NEMA 5-20 Level 1 on combo units).
  - `Port 2` delivers J1772 Level 2.
- Out of 68 clean MAC addresses:
  - **49 MAC addresses** logged sessions on both Port 1 and Port 2.
  - **19 MAC addresses** logged sessions on only one port (either single-port dedicated units or stations where one port saw zero recorded use).

### 3.2 Equipment Lifecycle & Upgrade Analysis
Examining station `PALO ALTO CA / HAMILTON #2` illustrates why accumulating lifetime unique ports without time awareness is misleading:
- **Lifetime Historical Tuples:** 8 unique `(MAC Address, Port Number)` combinations:
  - `000D6F00009ED39E` (Ports 1 & 2): Active 2011–2016 (7,797 sessions).
  - `0024B10000021331` (Ports 1 & 2): Active 2016–2017 (6,089 sessions).
  - `0024B100000203AD` (Ports 1 & 2): Active 2016–2017 (656 sessions).
  - `0024B1000002965D` (Ports 1 & 2): Active 2017–2020 (9,179 sessions).
- **Physical Reality:** In any single calendar year, Hamilton #2 operated only 1 or 2 active dual-port bollards (peak concurrent occupancy ever observed was 3 vehicles, typically 2). The cumulative count of 8 reflects **four successive hardware generations** installed at the exact same parking bays over a 10-year period.
- **Methodological Rule:** The inventory must record both **lifetime observed charging points** and **currently active observed charging points** (based on whether the charging point was observed active in the most recent operating period).

### 3.3 Hardware Reassignment & Mobility
Two physical EVSE units were reassigned across different station names during city reconfigurations:
1. `000D6F00009ED39E`: Transferred from `HAMILTON #2` (until Jan 2016) to `BRYANT #3` (from Feb 2016 onward).
2. `000D6F00009D9A1B`: Primarily deployed at `BRYANT #1` (2016–2019), with a single transaction logged at `BRYANT #2` on July 1, 2020.
- **Inventory Rule:** A unique physical charging point identifier must be scoped by `(station_id, mac_address, port_number)` to maintain unambiguous station-level integrity.

---

## 4. Availability Reconstruction Methodology

### 4.1 Ground Truth Occupancy Definition
For an observed charging point $p$ at station $s$ during time interval $[t_{\text{start}}, t_{\text{end}}]$:
- **Occupied State:** Point $p$ is marked as **Occupied** if at least one validated session ($flag\_unreliable = 0$) satisfied:
  $$\text{occupancy\_start\_utc} < t_{\text{end}} \quad \text{and} \quad \text{occupancy\_end\_utc} > t_{\text{start}}$$
- **Free State:** Point $p$ is marked as **Free** if the point was actively in commission and no session occupied the point during that interval.
- **Unknown State:** Point $p$ is marked as **Unknown** if the point had unrecorded downtime, missing interval data, or an unreliable occupancy flag.
- **Conservation Law:** For each station-hour with $M$ observed charging points:
  $$\text{occupied\_points} + \text{free\_points} + \text{unknown\_points} = M$$

### 4.2 Data Quality Exclusions
The following records are excluded from occupancy calculations:
- `flag_occupancy_interval_unreliable == 1` (349 records).
- `flag_duration_unreliable == 1` (349 records).
- Overlapping sessions on the *same physical port* that exceed simultaneous physical capacity (flagged and handled gracefully).

---

## 5. Architectural Blueprint for Phase 8 Modules

```
┌────────────────────────────────────────────────────────────────────────┐
│                        PHASE 8 MODULAR ARCHITECTURE                    │
├────────────────────────────────────────────────────────────────────────┤
│ 1. INVENTORY (ml/charging_point_inventory.py)                          │
│    -> dataset/processed/charging_point_inventory.csv                   │
│                                                                        │
│ 2. HISTORICAL AVAILABILITY (ml/historical_availability.py)             │
│    -> dataset/processed/historical_station_availability.csv            │
│                                                                        │
│ 3. FUTURE AVAILABILITY ESTIMATOR (ml/availability_prediction.py)       │
│    -> Combines Phase 5 XGBoost (1h & 24h) + Duration Distributions +   │
│       Current Occupancy + M/M/c-like Queue / Occupancy Simulation      │
│                                                                        │
│ 4. VEHICLE CATALOG & ESTIMATOR (data/vehicle_catalog.csv,              │
│    ml/charging_time_estimator.py, ml/station_compatibility.py)         │
│    -> 2-Wheelers, 4-Wheelers, Custom Vehicles; Charging Curves         │
│                                                                        │
│ 5. NEARBY STATION RECOMMENDER (ml/station_recommender.py)              │
│    -> Distance (Haversine), Compatibility, Predicted Free Points       │
│                                                                        │
│ 6. FLASK REST API & SQLITE LOGGING (backend/routes/*, db.py)           │
│                                                                        │
│ 7. REACT FRONTEND (Find a Charging Station Page, Multi-Step Flow)      │
│                                                                        │
│ 8. VERIFICATION SUITE & REPORT (scripts/test_availability.py, report)  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Conclusion

The inspection confirms that:
1. `station_id` and `Port Number` are 100% complete.
2. `MAC Address` is 100% complete and serves as the hardware EVSE identifier.
3. Charging points are definitively identified by `(station_id, mac_address_clean, Port Number)`.
4. Lifetime observed charging points must be distinguished from concurrent active capacity.
5. All occupancy intervals must use the validated Phase 3 timestamps (`occupancy_start_utc` and `occupancy_end_utc`).

We are now ready to proceed directly to **Part 2: Build Charging-Point Inventory**.
