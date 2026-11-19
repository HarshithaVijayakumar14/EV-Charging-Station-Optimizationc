import json
import os
import re
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

# Paths
WORKSPACE = r"c:\Users\harsh\OneDrive\Desktop\ML"
CSV_PATH = os.path.join(WORKSPACE, "EVChargingStationUsage.csv")
VIS_DIR = os.path.join(WORKSPACE, "reports", "eda_visualizations")
STATS_JSON = r"C:\Users\harsh\.gemini\antigravity\brain\7aee0d63-4143-475b-8d08-cc48808ea73c\scratch\phase2_eda_stats.json"

os.makedirs(VIS_DIR, exist_ok=True)

# Visual theme
sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#FAFBFC",
    "axes.edgecolor": "#CCCCCC",
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "grid.alpha": 0.45,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

PALETTE_PRIMARY = "#1F6F8B"
PALETTE_ACCENT = "#E07A5F"
PALETTE_TEAL = "#2A9D8F"
PALETTE_DARK = "#264653"
PALETTE_GOLD = "#E9C46A"
PALETTE_MUTED = "#8D99AE"


def parse_hms_series(series: pd.Series) -> pd.Series:
    """Parse hh:mm:ss strings (including >24h) into float minutes."""
    parts = series.astype(str).str.strip().str.extract(r"^(-?)(\d+):(\d{1,2}):(\d{1,2})$")
    sign = np.where(parts[0] == "-", -1.0, 1.0)
    h = pd.to_numeric(parts[1], errors="coerce")
    m = pd.to_numeric(parts[2], errors="coerce")
    s = pd.to_numeric(parts[3], errors="coerce")
    return pd.Series(sign * (h * 60.0 + m + s / 60.0), index=series.index)


def us_pacific_offset_hours(dt_utc: pd.Timestamp) -> int:
    """Return Pacific Time offset (-7 for PDT, -8 for PST) for a UTC timestamp (2011-2021 US DST rules)."""
    year = dt_utc.year
    # Second Sunday of March: 02:00 PST = 10:00 UTC
    mar1 = datetime(year, 3, 1)
    first_sun_mar = 1 + (6 - mar1.weekday()) % 7
    dst_start_utc = datetime(year, 3, first_sun_mar + 7, 10, 0)
    # First Sunday of November: 02:00 PDT = 09:00 UTC
    nov1 = datetime(year, 11, 1)
    first_sun_nov = 1 + (6 - nov1.weekday()) % 7
    dst_end_utc = datetime(year, 11, first_sun_nov, 9, 0)
    dt_py = dt_utc.to_pydatetime()
    return -7 if dst_start_utc <= dt_py < dst_end_utc else -8


def main():
    print("Loading raw EVChargingStationUsage.csv (read-only)...", flush=True)
    df = pd.read_csv(CSV_PATH, low_memory=False)
    n_rows = len(df)
    print(f"Loaded {n_rows:,} rows x {df.shape[1]} columns.", flush=True)

    # Parse Start Date into consistent Pacific local time in memory
    # (259,384 rows are already PDT/PST local time; 31 rows are labeled UTC and converted to Pacific local time)
    start_raw = pd.to_datetime(df["Start Date"], errors="coerce")
    is_utc_start = df["Start Time Zone"].astype(str).str.strip() == "UTC"
    start_pacific = start_raw.copy()
    if is_utc_start.any():
        utc_vals = start_raw[is_utc_start]
        offsets = utc_vals.apply(lambda t: timedelta(hours=us_pacific_offset_hours(t)))
        start_pacific.loc[is_utc_start] = utc_vals + offsets

    df["start_pt"] = start_pacific
    df["hour"] = df["start_pt"].dt.hour
    df["dow_num"] = df["start_pt"].dt.dayofweek  # 0=Mon .. 6=Sun
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["dow"] = df["dow_num"].map(dict(enumerate(dow_names)))
    df["is_weekend"] = df["dow_num"] >= 5
    df["year"] = df["start_pt"].dt.year
    df["month"] = df["start_pt"].dt.month
    df["year_month"] = df["start_pt"].dt.to_period("M")
    df["date"] = df["start_pt"].dt.date

    df["total_dur_min"] = parse_hms_series(df["Total Duration (hh:mm:ss)"])
    df["chg_time_min"] = parse_hms_series(df["Charging Time (hh:mm:ss)"])
    df["idle_min"] = df["total_dur_min"] - df["chg_time_min"]
    df["energy_kwh"] = pd.to_numeric(df["Energy (kWh)"], errors="coerce")
    df["implied_kw"] = np.where(df["chg_time_min"] >= 1.0, df["energy_kwh"] / (df["chg_time_min"] / 60.0), np.nan)
    df["port_type_clean"] = df["Port Type"].fillna("Missing")

    stats = {
        "total_rows": int(n_rows),
        "utc_start_converted_count": int(is_utc_start.sum()),
        "date_range": [str(df["start_pt"].min()), str(df["start_pt"].max())],
    }

    # =========================================================================
    # 1. CHARGING SESSIONS BY HOUR OF DAY
    # =========================================================================
    print("Generating 01_sessions_by_hour.png...", flush=True)
    hourly = df.groupby("hour").agg(
        sessions=("energy_kwh", "size"),
        total_kwh=("energy_kwh", "sum"),
        mean_kwh=("energy_kwh", "mean"),
        mean_chg_min=("chg_time_min", "mean"),
    ).reindex(range(24), fill_value=0)
    hourly["pct"] = hourly["sessions"] / n_rows * 100.0

    peak_hours_top3 = hourly["sessions"].nlargest(3)
    low_hours_bot3 = hourly["sessions"].nsmallest(3)
    peak_hr = int(hourly["sessions"].idxmax())
    low_hr = int(hourly["sessions"].idxmin())

    stats["hourly"] = {
        "by_hour": {
            str(h): {
                "sessions": int(row["sessions"]),
                "pct": round(float(row["pct"]), 2),
                "total_kwh": round(float(row["total_kwh"]), 2),
                "mean_kwh": round(float(row["mean_kwh"]), 2),
            }
            for h, row in hourly.iterrows()
        },
        "peak_hour": peak_hr,
        "peak_hour_sessions": int(hourly.loc[peak_hr, "sessions"]),
        "peak_hour_pct": round(float(hourly.loc[peak_hr, "pct"]), 2),
        "top3_hours": {str(int(k)): int(v) for k, v in peak_hours_top3.items()},
        "lowest_hour": low_hr,
        "lowest_hour_sessions": int(hourly.loc[low_hr, "sessions"]),
        "lowest_hour_pct": round(float(hourly.loc[low_hr, "pct"]), 3),
        "bottom3_hours": {str(int(k)): int(v) for k, v in low_hours_bot3.items()},
        "daytime_8_to_18_pct": round(float(hourly.loc[8:18, "pct"].sum()), 2),
        "overnight_0_to_5_pct": round(float(hourly.loc[0:5, "pct"].sum()), 2),
    }

    fig, ax1 = plt.subplots(figsize=(12, 5.5))
    colors_hr = [
        PALETTE_ACCENT if h in peak_hours_top3.index else
        (PALETTE_MUTED if h in low_hours_bot3.index else PALETTE_PRIMARY)
        for h in hourly.index
    ]
    bars = ax1.bar(hourly.index, hourly["sessions"], color=colors_hr, edgecolor="white", linewidth=0.7, width=0.78)
    ax1.set_title("Charging Sessions & Total Energy by Hour of Day (Pacific Local Time, 2011–2020)")
    ax1.set_xlabel("Session Start Hour of Day (0 = Midnight, 12 = Noon)")
    ax1.set_ylabel("Number of Charging Sessions", color=PALETTE_DARK)
    ax1.set_xticks(range(24))
    ax1.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_ylim(0, hourly["sessions"].max() * 1.16)

    # Annotate top 3 peak and lowest hour
    for h in list(peak_hours_top3.index) + [low_hr]:
        val = hourly.loc[h, "sessions"]
        pct_str = f"{hourly.loc[h, 'pct']:.2f}%" if hourly.loc[h, "pct"] < 0.1 else f"{hourly.loc[h, 'pct']:.1f}%"
        ax1.annotate(
            f"{val:,}\n({pct_str})",
            xy=(h, val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold",
            color=PALETTE_ACCENT if h in peak_hours_top3.index else "#555555",
        )

    ax2 = ax1.twinx()
    ax2.plot(hourly.index, hourly["total_kwh"] / 1000.0, color=PALETTE_DARK, marker="o", linewidth=2, markersize=4.5, label="Total Energy (MWh)")
    ax2.set_ylabel("Total Energy Delivered (MWh)", color=PALETTE_DARK)
    ax2.grid(False)
    ax2.set_ylim(0, (hourly["total_kwh"].max() / 1000.0) * 1.2)

    # Legend
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_handles = [
        Patch(facecolor=PALETTE_ACCENT, label="Top 3 Peak Hours (08:00, 09:00, 11:00–12:00)"),
        Patch(facecolor=PALETTE_PRIMARY, label="Regular Hours"),
        Patch(facecolor=PALETTE_MUTED, label="Lowest 3 Demand Hours (02:00–04:00)"),
        Line2D([0], [0], color=PALETTE_DARK, marker="o", lw=2, label="Total Energy Delivered (MWh)"),
    ]
    # Update exact top 3 / bottom 3 labels dynamically
    top3_lbl = ", ".join(f"{h:02d}:00" for h in sorted(peak_hours_top3.index))
    bot3_lbl = ", ".join(f"{h:02d}:00" for h in sorted(low_hours_bot3.index))
    legend_handles[0].set_label(f"Top 3 Peak Hours ({top3_lbl})")
    legend_handles[2].set_label(f"Lowest 3 Demand Hours ({bot3_lbl})")
    ax1.legend(handles=legend_handles, loc="upper right", frameon=True, facecolor="white", fontsize=9)

    fig.text(0.01, -0.04, f"Source: EVChargingStationUsage.csv (n = {n_rows:,} sessions; 31 UTC-labeled start times converted to Pacific local time).", fontsize=8.5, color="#555555")
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "01_sessions_by_hour.png"))
    plt.close(fig)

    # =========================================================================
    # 2. CHARGING SESSIONS BY DAY OF WEEK
    # =========================================================================
    print("Generating 02_sessions_by_day_of_week.png...", flush=True)
    # Count unique active calendar days per weekday in dataset span
    daily_counts = df.groupby(["date", "dow_num"]).agg(
        sessions=("energy_kwh", "size"),
        energy_kwh=("energy_kwh", "sum"),
    ).reset_index()

    dow_agg = df.groupby("dow_num").agg(
        sessions=("energy_kwh", "size"),
        total_kwh=("energy_kwh", "sum"),
        mean_kwh=("energy_kwh", "mean"),
        mean_dur_min=("total_dur_min", "mean"),
        mean_chg_min=("chg_time_min", "mean"),
    ).reindex(range(7))
    dow_agg["dow"] = dow_names
    dow_agg["pct"] = dow_agg["sessions"] / n_rows * 100.0
    dow_daily_mean = daily_counts.groupby("dow_num")["sessions"].mean().reindex(range(7))
    dow_agg["mean_sessions_per_active_day"] = dow_daily_mean

    wkday_df = df[~df["is_weekend"]]
    wkend_df = df[df["is_weekend"]]
    wkday_daily = daily_counts[daily_counts["dow_num"] < 5]["sessions"].mean()
    wkend_daily = daily_counts[daily_counts["dow_num"] >= 5]["sessions"].mean()

    stats["day_of_week"] = {
        "by_day": {
            dow_names[i]: {
                "sessions": int(row["sessions"]),
                "pct": round(float(row["pct"]), 2),
                "total_kwh": round(float(row["total_kwh"]), 2),
                "mean_kwh": round(float(row["mean_kwh"]), 2),
                "mean_dur_min": round(float(row["mean_dur_min"]), 2),
                "mean_chg_min": round(float(row["mean_chg_min"]), 2),
                "mean_sessions_per_active_day": round(float(row["mean_sessions_per_active_day"]), 2),
            }
            for i, row in dow_agg.iterrows()
        },
        "weekday_total_sessions": int(len(wkday_df)),
        "weekday_pct": round(len(wkday_df) / n_rows * 100.0, 2),
        "weekend_total_sessions": int(len(wkend_df)),
        "weekend_pct": round(len(wkend_df) / n_rows * 100.0, 2),
        "weekday_mean_sessions_per_day": round(float(wkday_daily), 2),
        "weekend_mean_sessions_per_day": round(float(wkend_daily), 2),
        "weekday_vs_weekend_daily_ratio": round(float(wkday_daily / wkend_daily), 2),
        "weekday_mean_kwh": round(float(wkday_df["energy_kwh"].mean()), 2),
        "weekend_mean_kwh": round(float(wkend_df["energy_kwh"].mean()), 2),
        "weekday_mean_dur_min": round(float(wkday_df["total_dur_min"].mean()), 2),
        "weekend_mean_dur_min": round(float(wkend_df["total_dur_min"].mean()), 2),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), gridspec_kw={"width_ratios": [1.35, 1]})
    bar_colors = [PALETTE_PRIMARY if i < 5 else PALETTE_ACCENT for i in range(7)]
    x_dow = np.arange(7)
    ax1.bar(x_dow, dow_agg["sessions"], color=bar_colors, edgecolor="white", width=0.68)
    ax1.set_title("Total Charging Sessions by Day of Week")
    ax1.set_ylabel("Total Charging Sessions (2011–2020)")
    ax1.set_xticks(x_dow)
    ax1.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_ylim(0, dow_agg["sessions"].max() * 1.28)
    for i, row in dow_agg.iterrows():
        ax1.text(i, row["sessions"] + 650, f"{int(row['sessions']):,}\n({row['pct']:.1f}%)",
                 ha="center", va="bottom", fontsize=8.8, fontweight="bold",
                 color=PALETTE_DARK if i < 5 else PALETTE_ACCENT)

    ax1.legend(handles=[
        Patch(facecolor=PALETTE_PRIMARY, label=f"Weekdays ({stats['day_of_week']['weekday_pct']:.1f}% of sessions)"),
        Patch(facecolor=PALETTE_ACCENT, label=f"Weekend ({stats['day_of_week']['weekend_pct']:.1f}% of sessions)"),
    ], loc="upper right", frameon=True, facecolor="white", fontsize=9)

    # Right panel: Mean sessions per active day & mean energy per session
    ax2.bar(x_dow, dow_agg["mean_sessions_per_active_day"], color=bar_colors, alpha=0.88, edgecolor="white", width=0.65)
    ax2.axhline(wkday_daily, color=PALETTE_DARK, linestyle="--", linewidth=1.4, label=f"Weekday Avg: {wkday_daily:.1f}/day")
    ax2.axhline(wkend_daily, color=PALETTE_ACCENT, linestyle=":", linewidth=1.6, label=f"Weekend Avg: {wkend_daily:.1f}/day")
    ax2.set_title("Average Daily Sessions by Day of Week")
    ax2.set_ylabel("Mean Sessions per Active Calendar Day")
    ax2.set_xticks(x_dow)
    ax2.set_xticklabels(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
    ax2.set_ylim(0, dow_agg["mean_sessions_per_active_day"].max() * 1.30)
    for i, val in enumerate(dow_agg["mean_sessions_per_active_day"]):
        ax2.text(i, val + 1.5, f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    fig.suptitle("Day-of-Week Charging Demand: Weekday vs. Weekend Contrast", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "02_sessions_by_day_of_week.png"))
    plt.close(fig)

    # =========================================================================
    # 3. CHARGING SESSIONS BY MONTH AND YEAR
    # =========================================================================
    print("Generating 03_sessions_by_month_and_year.png...", flush=True)
    ym = df.groupby("year_month").agg(
        sessions=("energy_kwh", "size"),
        total_kwh=("energy_kwh", "sum"),
        active_stations=("Station Name", "nunique"),
    ).sort_index()
    ym["sessions_per_station"] = ym["sessions"] / ym["active_stations"]
    ym_dates = ym.index.to_timestamp()

    peak_ym = ym["sessions"].idxmax()
    # Pre-COVID complete years (2012-2019) calendar-month seasonality
    df_12_19 = df[(df["year"] >= 2012) & (df["year"] <= 2019)]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    # Monthly average across complete pre-COVID years (2012-2019) vs 2020
    ym_12_19 = df_12_19.groupby(["year", "month"]).agg(
        sessions=("energy_kwh", "size"),
        active_stations=("Station Name", "nunique"),
    ).reset_index()
    ym_12_19["sess_per_station"] = ym_12_19["sessions"] / ym_12_19["active_stations"]
    seasonality_12_19 = ym_12_19.groupby("month").agg(
        mean_sessions=("sessions", "mean"),
        mean_sess_per_station=("sess_per_station", "mean"),
        total_sessions=("sessions", "sum"),
    ).reindex(range(1, 13))

    ym_2019 = df[df["year"] == 2019].groupby("month").size().reindex(range(1, 13), fill_value=0)
    ym_2020 = df[df["year"] == 2020].groupby("month").size().reindex(range(1, 13), fill_value=0)
    all_months_raw = df.groupby("month").size().reindex(range(1, 13), fill_value=0)
    yearly = df.groupby("year").agg(
        sessions=("energy_kwh", "size"),
        total_kwh=("energy_kwh", "sum"),
        active_stations=("Station Name", "nunique"),
    )
    yearly["sessions_per_active_station"] = yearly["sessions"] / yearly["active_stations"]

    # Apr 2020 vs Apr 2019 drop
    apr_drop_pct = (ym_2020.loc[4] - ym_2019.loc[4]) / ym_2019.loc[4] * 100.0
    mar_dec_2020 = int(ym_2020.loc[3:12].sum())
    mar_dec_2019 = int(ym_2019.loc[3:12].sum())
    mar_dec_drop_pct = (mar_dec_2020 - mar_dec_2019) / mar_dec_2019 * 100.0

    # Fee introduction check by month in 2017
    df["fee_num"] = pd.to_numeric(df["Fee"], errors="coerce")
    fee_2017 = df[df["year"] == 2017].groupby("month").apply(
        lambda g: {"sessions": int(len(g)), "pct_paid": round(float((g["fee_num"] > 0).mean() * 100.0), 1)},
        include_groups=False,
    ).to_dict()

    stats["monthly_yearly"] = {
        "peak_year_month": str(peak_ym),
        "peak_year_month_sessions": int(ym.loc[peak_ym, "sessions"]),
        "peak_year_month_active_stations": int(ym.loc[peak_ym, "active_stations"]),
        "fee_rollout_2017": {month_names[m - 1]: v for m, v in sorted(fee_2017.items())},
        "yearly": {
            str(y): {
                "sessions": int(r["sessions"]),
                "total_kwh": round(float(r["total_kwh"]), 2),
                "active_stations": int(r["active_stations"]),
                "sessions_per_active_station": round(float(r["sessions_per_active_station"]), 1),
            }
            for y, r in yearly.iterrows()
        },
        "seasonality_2012_2019": {
            month_names[m - 1]: {
                "mean_monthly_sessions": round(float(r["mean_sessions"]), 1),
                "mean_sessions_per_active_station": round(float(r["mean_sess_per_station"]), 1),
                "total_sessions_2012_2019": int(r["total_sessions"]),
            }
            for m, r in seasonality_12_19.iterrows()
        },
        "all_years_by_month": {month_names[m - 1]: int(v) for m, v in all_months_raw.items()},
        "monthly_2019": {month_names[m - 1]: int(v) for m, v in ym_2019.items()},
        "monthly_2020": {month_names[m - 1]: int(v) for m, v in ym_2020.items()},
        "apr_2020_vs_2019_pct_change": round(float(apr_drop_pct), 1),
        "mar_to_dec_2020_vs_2019_pct_change": round(float(mar_dec_drop_pct), 1),
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.8), gridspec_kw={"height_ratios": [1.15, 1]})

    # Top panel: Full monthly time series + active station count
    ax1.fill_between(ym_dates, ym["sessions"], color=PALETTE_PRIMARY, alpha=0.18)
    ax1.plot(ym_dates, ym["sessions"], color=PALETTE_PRIMARY, linewidth=2.1, label="Monthly Charging Sessions")
    ax1.set_title("Monthly Charging Demand Over Time & Active Station Network Growth (Jul 2011 – Dec 2020)")
    ax1.set_ylabel("Monthly Sessions", color=PALETTE_PRIMARY)
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.set_ylim(0, ym["sessions"].max() * 1.18)

    # Annotate peak month and COVID-19 drop
    ax1.annotate(
        f"Peak: {peak_ym}\n({int(ym.loc[peak_ym, 'sessions']):,} sessions)",
        xy=(peak_ym.to_timestamp(), ym.loc[peak_ym, "sessions"]),
        xytext=(-110, 14), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=PALETTE_DARK),
        fontsize=9, fontweight="bold", color=PALETTE_DARK,
    )
    apr20_ts = pd.Period("2020-04", "M").to_timestamp()
    apr20_val = int(ym.loc[pd.Period("2020-04", "M"), "sessions"])
    ax1.annotate(
        f"COVID-19 Drop\nApr 2020: {apr20_val:,}\n({apr_drop_pct:.1f}% vs Apr 2019)",
        xy=(apr20_ts, apr20_val),
        xytext=(-140, 48), textcoords="offset points",
        arrowprops=dict(arrowstyle="->", color=PALETTE_ACCENT),
        fontsize=9, fontweight="bold", color=PALETTE_ACCENT,
    )

    ax1_r = ax1.twinx()
    ax1_r.step(ym_dates, ym["active_stations"], where="post", color=PALETTE_ACCENT, linewidth=1.8, linestyle="--", label="Active Stations in Month")
    ax1_r.set_ylabel("Active Stations Reporting", color=PALETTE_ACCENT)
    ax1_r.grid(False)
    ax1_r.set_ylim(0, ym["active_stations"].max() * 1.18)

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax1_r.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", frameon=True, facecolor="white", fontsize=9)

    # Bottom panel: Monthly seasonality (2012-2019 mean vs 2019 vs 2020)
    x = np.arange(12)
    w = 0.27
    ax2.bar(x - w, seasonality_12_19["mean_sessions"], width=w, color=PALETTE_TEAL, label="2012–2019 Monthly Mean (Complete Pre-COVID Years)")
    ax2.bar(x, ym_2019.values, width=w, color=PALETTE_PRIMARY, label="2019 (Pre-COVID Peak Network)")
    ax2.bar(x + w, ym_2020.values, width=w, color=PALETTE_ACCENT, label="2020 (COVID-19 Year)")
    ax2.set_title("Calendar-Month Seasonality: 2012–2019 Average vs. 2019 & 2020 Actuals")
    ax2.set_xticks(x)
    ax2.set_xticklabels(month_names)
    ax2.set_ylabel("Monthly Charging Sessions")
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.set_ylim(0, max(ym_2019.max(), ym_2020.max()) * 1.28)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "03_sessions_by_month_and_year.png"))
    plt.close(fig)

    # =========================================================================
    # 4. CHARGING SESSIONS BY STATION
    # =========================================================================
    print("Generating 04_sessions_by_station.png...", flush=True)
    st_agg = df.groupby("Station Name").agg(
        sessions=("energy_kwh", "size"),
        total_kwh=("energy_kwh", "sum"),
        mean_kwh=("energy_kwh", "mean"),
        median_kwh=("energy_kwh", "median"),
        mean_dur_min=("total_dur_min", "mean"),
        first_dt=("start_pt", "min"),
        last_dt=("start_pt", "max"),
        active_days=("date", "nunique"),
    ).sort_values("sessions", ascending=False)
    st_agg["pct_sessions"] = st_agg["sessions"] / n_rows * 100.0
    st_agg["pct_energy"] = st_agg["total_kwh"] / df["energy_kwh"].sum() * 100.0
    st_agg["span_days"] = ((st_agg["last_dt"] - st_agg["first_dt"]).dt.total_seconds() / 86400.0).clip(lower=1.0)
    st_agg["sessions_per_active_day"] = st_agg["sessions"] / st_agg["active_days"]

    st_short = lambda s: s.replace("PALO ALTO CA / ", "")

    top10_sess_share = float(st_agg.head(10)["pct_sessions"].sum())
    top5_sess_share = float(st_agg.head(5)["pct_sessions"].sum())
    top10_en_share = float(st_agg.sort_values("total_kwh", ascending=False).head(10)["pct_energy"].sum())
    established = st_agg[st_agg["sessions"] >= 500]
    top_mean_kwh = established.sort_values("mean_kwh", ascending=False).head(10)
    top_intensity = established.sort_values("sessions_per_active_day", ascending=False).head(10)

    stats["stations"] = {
        "total_unique_stations": int(len(st_agg)),
        "top5_sessions_share_pct": round(top5_sess_share, 2),
        "top10_sessions_share_pct": round(top10_sess_share, 2),
        "top10_energy_share_pct": round(top10_en_share, 2),
        "top10_by_sessions": [
            {
                "station": idx,
                "sessions": int(r["sessions"]),
                "pct_sessions": round(float(r["pct_sessions"]), 2),
                "total_kwh": round(float(r["total_kwh"]), 2),
                "mean_kwh": round(float(r["mean_kwh"]), 2),
                "active_days": int(r["active_days"]),
                "sessions_per_active_day": round(float(r["sessions_per_active_day"]), 2),
                "first_year": int(r["first_dt"].year),
                "last_year": int(r["last_dt"].year),
            }
            for idx, r in st_agg.head(10).iterrows()
        ],
        "top10_by_total_energy": [
            {
                "station": idx,
                "total_kwh": round(float(r["total_kwh"]), 2),
                "pct_energy": round(float(r["pct_energy"]), 2),
                "sessions": int(r["sessions"]),
                "mean_kwh": round(float(r["mean_kwh"]), 2),
            }
            for idx, r in st_agg.sort_values("total_kwh", ascending=False).head(10).iterrows()
        ],
        "top5_by_mean_kwh_established": [
            {
                "station": idx,
                "mean_kwh": round(float(r["mean_kwh"]), 2),
                "median_kwh": round(float(r["median_kwh"]), 2),
                "sessions": int(r["sessions"]),
                "total_kwh": round(float(r["total_kwh"]), 2),
            }
            for idx, r in top_mean_kwh.head(5).iterrows()
        ],
        "top5_by_sessions_per_active_day_established": [
            {
                "station": idx,
                "sessions_per_active_day": round(float(r["sessions_per_active_day"]), 2),
                "active_days": int(r["active_days"]),
                "sessions": int(r["sessions"]),
            }
            for idx, r in top_intensity.head(5).iterrows()
        ],
        "sherman_and_small_stations_count": int((st_agg["sessions"] < 100).sum()),
        "sherman_and_small_stations_total_sessions": int(st_agg[st_agg["sessions"] < 100]["sessions"].sum()),
    }

    # Plot Top 25 stations (covers 99.4% of all sessions) + note on remaining 22
    top25 = st_agg.head(25).iloc[::-1]
    rem_stations = len(st_agg) - 25
    rem_sessions = int(st_agg.iloc[25:]["sessions"].sum())

    fig, ax = plt.subplots(figsize=(12, 8.2))
    colors_st = [PALETTE_ACCENT if i >= 20 else PALETTE_PRIMARY for i in range(len(top25))]
    labels_25 = [st_short(s) for s in top25.index]
    ax.barh(labels_25, top25["sessions"], color=colors_st, edgecolor="white", height=0.76)
    ax.set_title("Charging Sessions by Station — Top 25 Stations (2011–2020)")
    ax.set_xlabel("Total Charging Sessions")
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(0, top25["sessions"].max() * 1.34)

    for idx_pos, (_, row) in enumerate(top25.iterrows()):
        ax.text(
            row["sessions"] + 250, idx_pos,
            f"{int(row['sessions']):,} ({row['pct_sessions']:.1f}% | {row['sessions_per_active_day']:.1f}/active day)",
            va="center", fontsize=8.3, color=PALETTE_DARK,
        )

    ax.legend(handles=[
        Patch(facecolor=PALETTE_ACCENT, label=f"Top 5 Stations ({top5_sess_share:.1f}% of all sessions)"),
        Patch(facecolor=PALETTE_PRIMARY, label="Ranks 6–25"),
    ], loc="lower right", frameon=True, facecolor="white", fontsize=9)
    fig.text(
        0.01, -0.02,
        f"Note: Shows top 25 of 47 raw station names ({100 - rem_sessions/n_rows*100:.2f}% of sessions). Remaining {rem_stations} stations = {rem_sessions:,} sessions (includes 13 Sherman stations opened Dec 2020 with 39 sessions total & 'BRYANT # 1' typo with 13 sessions). Cumulative counts also reflect differing installation years.",
        fontsize=8.2, color="#555555",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "04_sessions_by_station.png"))
    plt.close(fig)

    # =========================================================================
    # 5. ENERGY CONSUMPTION BY STATION
    # =========================================================================
    print("Generating 05_energy_consumption_by_station.png...", flush=True)
    top20_en = st_agg.sort_values("total_kwh", ascending=False).head(20).iloc[::-1]
    top15_mean_en = established.sort_values("mean_kwh", ascending=False).head(15).iloc[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7.6), gridspec_kw={"width_ratios": [1.25, 1]})

    c1 = [PALETTE_ACCENT if i >= 15 else PALETTE_TEAL for i in range(len(top20_en))]
    ax1.barh([st_short(s) for s in top20_en.index], top20_en["total_kwh"] / 1000.0, color=c1, edgecolor="white", height=0.75)
    ax1.set_title("Top 20 Stations by Total Energy Delivered (MWh)")
    ax1.set_xlabel("Total Energy Delivered (MWh = 1,000 kWh)")
    ax1.set_xlim(0, (top20_en["total_kwh"].max() / 1000.0) * 1.24)
    for pos, (_, r) in enumerate(top20_en.iterrows()):
        ax1.text(r["total_kwh"] / 1000.0 + 2.2, pos, f"{r['total_kwh']/1000.0:,.1f} MWh ({r['pct_energy']:.1f}%)", va="center", fontsize=8.3, color=PALETTE_DARK)

    c2 = [PALETTE_DARK if i >= 10 else PALETTE_PRIMARY for i in range(len(top15_mean_en))]
    ax2.barh([st_short(s) for s in top15_mean_en.index], top15_mean_en["mean_kwh"], color=c2, edgecolor="white", height=0.72)
    ax2.axvline(df["energy_kwh"].mean(), color=PALETTE_ACCENT, linestyle="--", linewidth=1.6, label=f"Network Mean ({df['energy_kwh'].mean():.2f} kWh)")
    ax2.set_title("Top 15 Stations by Mean Energy/Session (≥500 Sessions)")
    ax2.set_xlabel("Mean Energy per Session (kWh)")
    ax2.set_xlim(0, top15_mean_en["mean_kwh"].max() * 1.2)
    for pos, (_, r) in enumerate(top15_mean_en.iterrows()):
        ax2.text(r["mean_kwh"] + 0.15, pos, f"{r['mean_kwh']:.2f} kWh", va="center", fontsize=8.5, color=PALETTE_DARK)
    ax2.legend(loc="lower right", frameon=True, facecolor="white", fontsize=9)

    fig.suptitle("Energy Consumption Patterns Across EV Charging Stations (Total Network = 2,216.5 MWh)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "05_energy_consumption_by_station.png"))
    plt.close(fig)

    # =========================================================================
    # 6. LEVEL 1 VS LEVEL 2 CHARGING COMPARISON
    # =========================================================================
    print("Generating 06_level1_vs_level2_comparison.png...", flush=True)
    lvl_df = df[df["Port Type"].isin(["Level 1", "Level 2"])]
    lvl_stats = {}
    for pt in ["Level 2", "Level 1"]:
        g = lvl_df[lvl_df["Port Type"] == pt]
        lvl_stats[pt] = {
            "sessions": int(len(g)),
            "pct_sessions": round(len(g) / n_rows * 100.0, 3),
            "total_kwh": round(float(g["energy_kwh"].sum()), 2),
            "pct_energy": round(float(g["energy_kwh"].sum() / df["energy_kwh"].sum() * 100.0), 3),
            "mean_kwh": round(float(g["energy_kwh"].mean()), 2),
            "median_kwh": round(float(g["energy_kwh"].median()), 2),
            "mean_chg_min": round(float(g["chg_time_min"].mean()), 2),
            "median_chg_min": round(float(g["chg_time_min"].median()), 2),
            "mean_dur_min": round(float(g["total_dur_min"].mean()), 2),
            "median_dur_min": round(float(g["total_dur_min"].median()), 2),
            "mean_kw": round(float(g["implied_kw"].mean()), 2),
            "median_kw": round(float(g["implied_kw"].median()), 2),
        }
    l1_by_year = df[df["Port Type"] == "Level 1"].groupby("year").size().reindex(range(2011, 2021), fill_value=0)
    l2_by_year = df[df["Port Type"] == "Level 2"].groupby("year").size().reindex(range(2011, 2021), fill_value=0)
    l1_share_by_year = (l1_by_year / (l1_by_year + l2_by_year) * 100.0).round(2)

    stats["level_comparison"] = {
        "levels": lvl_stats,
        "missing_port_type_count": int(df["Port Type"].isna().sum()),
        "l1_sessions_by_year": {str(y): int(v) for y, v in l1_by_year.items()},
        "l1_share_pct_by_year": {str(y): float(v) for y, v in l1_share_by_year.items()},
    }

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))

    # Panel A: Share of Sessions & Share of Energy
    ax = axes[0, 0]
    cats = ["Share of Sessions (%)", "Share of Total Energy (%)"]
    l2_vals = [lvl_stats["Level 2"]["pct_sessions"], lvl_stats["Level 2"]["pct_energy"]]
    l1_vals = [lvl_stats["Level 1"]["pct_sessions"], lvl_stats["Level 1"]["pct_energy"]]
    x = np.arange(2)
    ax.bar(x, l2_vals, width=0.5, color=PALETTE_PRIMARY, label=f"Level 2 (n={lvl_stats['Level 2']['sessions']:,})")
    ax.bar(x, l1_vals, width=0.5, bottom=l2_vals, color=PALETTE_ACCENT, label=f"Level 1 (n={lvl_stats['Level 1']['sessions']:,})")
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Percentage (%)")
    ax.set_ylim(0, 112)
    ax.set_title("A. Network Share: Level 2 vs. Level 1")
    for i in range(2):
        ax.text(i, l2_vals[i] / 2, f"Level 2\n{l2_vals[i]:.2f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=10)
        ax.text(i, 102, f"Level 1: {l1_vals[i]:.2f}%", ha="center", va="bottom", color=PALETTE_ACCENT, fontweight="bold", fontsize=9.5)
    ax.legend(loc="lower right", frameon=True, facecolor="white", fontsize=8.5)

    # Panel B: Energy & Power per Session (Mean & Median)
    ax = axes[0, 1]
    metrics = ["Mean Energy\n(kWh)", "Median Energy\n(kWh)", "Median Power\n(kW)"]
    l2_m = [lvl_stats["Level 2"]["mean_kwh"], lvl_stats["Level 2"]["median_kwh"], lvl_stats["Level 2"]["median_kw"]]
    l1_m = [lvl_stats["Level 1"]["mean_kwh"], lvl_stats["Level 1"]["median_kwh"], lvl_stats["Level 1"]["median_kw"]]
    x = np.arange(len(metrics))
    w = 0.34
    ax.bar(x - w/2, l2_m, w, color=PALETTE_PRIMARY, label="Level 2 (J1772)")
    ax.bar(x + w/2, l1_m, w, color=PALETTE_ACCENT, label="Level 1 (NEMA 5-20R)")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value (kWh or kW)")
    ax.set_ylim(0, max(l2_m) * 1.22)
    ax.set_title("B. Energy Delivered & Implied Power per Session")
    for i in range(len(metrics)):
        ax.text(i - w/2, l2_m[i] + 0.18, f"{l2_m[i]:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE_DARK)
        ax.text(i + w/2, l1_m[i] + 0.18, f"{l1_m[i]:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color=PALETTE_ACCENT)
    ax.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.5)

    # Panel C: Charging Time vs Total Duration (Median & Mean, minutes)
    ax = axes[1, 0]
    d_metrics = ["Median\nCharging Time", "Mean\nCharging Time", "Median\nTotal Duration", "Mean\nTotal Duration"]
    l2_d = [lvl_stats["Level 2"]["median_chg_min"], lvl_stats["Level 2"]["mean_chg_min"], lvl_stats["Level 2"]["median_dur_min"], lvl_stats["Level 2"]["mean_dur_min"]]
    l1_d = [lvl_stats["Level 1"]["median_chg_min"], lvl_stats["Level 1"]["mean_chg_min"], lvl_stats["Level 1"]["median_dur_min"], lvl_stats["Level 1"]["mean_dur_min"]]
    x = np.arange(len(d_metrics))
    ax.bar(x - w/2, l2_d, w, color=PALETTE_PRIMARY, label="Level 2")
    ax.bar(x + w/2, l1_d, w, color=PALETTE_ACCENT, label="Level 1")
    ax.set_xticks(x)
    ax.set_xticklabels(d_metrics, fontsize=9.5)
    ax.set_ylabel("Duration (Minutes)")
    ax.set_ylim(0, max(l1_d) * 1.2)
    ax.set_title("C. Session Duration Comparison (Minutes)")
    for i in range(len(d_metrics)):
        ax.text(i - w/2, l2_d[i] + 3, f"{l2_d[i]:.0f}m", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=PALETTE_DARK)
        ax.text(i + w/2, l1_d[i] + 3, f"{l1_d[i]:.0f}m", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color=PALETTE_ACCENT)
    ax.legend(loc="upper left", frameon=True, facecolor="white", fontsize=8.5)

    # Panel D: Historical Decline of Level 1 Share (2011-2020)
    ax = axes[1, 1]
    years_arr = list(range(2011, 2021))
    ax.bar(years_arr, l1_by_year.values, color=PALETTE_ACCENT, alpha=0.82, width=0.6, label="Level 1 Sessions (Count)")
    ax.set_title("D. Level 1 Usage Over Time (Count & % Share)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Level 1 Sessions", color=PALETTE_ACCENT)
    ax.set_xticks(years_arr)
    ax.set_ylim(0, l1_by_year.max() * 1.22)

    ax_r = ax.twinx()
    ax_r.plot(years_arr, l1_share_by_year.values, color=PALETTE_DARK, marker="o", linewidth=2, label="Level 1 Share of All Sessions (%)")
    ax_r.set_ylabel("Level 1 Share (%)", color=PALETTE_DARK)
    ax_r.grid(False)
    ax_r.set_ylim(0, l1_share_by_year.max() * 1.28)
    for y, pct in zip(years_arr, l1_share_by_year.values):
        y_off = 0.55 if y in (2015, 2016) else 0.35
        x_off = 0.12 if y in (2015, 2016) else 0.0
        ax_r.text(
            y + x_off, pct + y_off, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8,
            color=PALETTE_DARK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
        )
    l1h, l1l = ax.get_legend_handles_labels()
    l2h, l2l = ax_r.get_legend_handles_labels()
    ax.legend(l1h + l2h, l1l + l2l, loc="upper right", frameon=True, facecolor="white", fontsize=8.2)

    fig.suptitle("Level 1 vs. Level 2 EV Charging Comparison (Excludes 9 Missing Port Type Rows)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "06_level1_vs_level2_comparison.png"))
    plt.close(fig)

    # =========================================================================
    # 7. CHARGING DURATION DISTRIBUTION
    # =========================================================================
    print("Generating 07_charging_duration_distribution.png...", flush=True)
    td_hr = df["total_dur_min"] / 60.0
    ct_hr = df["chg_time_min"] / 60.0
    idle_min = df["idle_min"]

    buckets = [
        ("< 30 min", (df["chg_time_min"] < 30).sum()),
        ("30–60 min", ((df["chg_time_min"] >= 30) & (df["chg_time_min"] < 60)).sum()),
        ("1–2 hours", ((df["chg_time_min"] >= 60) & (df["chg_time_min"] < 120)).sum()),
        ("2–4 hours", ((df["chg_time_min"] >= 120) & (df["chg_time_min"] < 240)).sum()),
        ("4–8 hours", ((df["chg_time_min"] >= 240) & (df["chg_time_min"] < 480)).sum()),
        ("> 8 hours", (df["chg_time_min"] >= 480).sum()),
    ]
    idle_gt_15m = int((idle_min > 15).sum())
    idle_gt_60m = int((idle_min > 60).sum())
    td_gt_10h = int((td_hr > 10).sum())
    ct_gt_10h = int((ct_hr > 10).sum())

    stats["durations"] = {
        "charging_time_buckets": {k: {"count": int(v), "pct": round(int(v) / n_rows * 100.0, 2)} for k, v in buckets},
        "total_dur_gt_10h": td_gt_10h,
        "chg_time_gt_10h": ct_gt_10h,
        "idle_gt_15min_count": idle_gt_15m,
        "idle_gt_15min_pct": round(idle_gt_15m / n_rows * 100.0, 2),
        "idle_gt_60min_count": idle_gt_60m,
        "idle_gt_60min_pct": round(idle_gt_60m / n_rows * 100.0, 2),
        "idle_negative_anomaly_count": int((idle_min < 0).sum()),
        "total_dur_median_hr": round(float(td_hr.median()), 2),
        "total_dur_mean_hr": round(float(td_hr.mean()), 2),
        "chg_time_median_hr": round(float(ct_hr.median()), 2),
        "chg_time_mean_hr": round(float(ct_hr.mean()), 2),
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6))

    # Left panel: Histogram of Charging Time vs Total Duration (0 to 10 hours)
    bins = np.linspace(0, 10, 61)
    ax1.hist(td_hr[td_hr <= 10], bins=bins, alpha=0.45, color=PALETTE_ACCENT, label=f"Total Plugged-In Duration (Median: {td_hr.median():.2f}h)")
    ax1.hist(ct_hr[ct_hr <= 10], bins=bins, alpha=0.65, color=PALETTE_PRIMARY, label=f"Active Charging Time (Median: {ct_hr.median():.2f}h)")
    ax1.axvline(ct_hr.median(), color=PALETTE_DARK, linestyle="--", linewidth=1.6, label=f"Charging Median ({ct_hr.median():.2f}h)")
    ax1.axvline(td_hr.median(), color="#B5482E", linestyle=":", linewidth=1.8, label=f"Total Dur. Median ({td_hr.median():.2f}h)")
    ax1.set_title("Distribution of Active Charging Time vs. Total Duration (0–10 h)")
    ax1.set_xlabel("Duration (Hours)")
    ax1.set_ylabel("Number of Sessions")
    ax1.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax1.legend(loc="upper right", frameon=True, facecolor="white", fontsize=8.8)
    ax1.text(
        0.97, 0.62,
        f"Tail beyond 10h:\n• Total Duration > 10h: {td_gt_10h:,} ({td_gt_10h/n_rows*100:.2f}%)\n• Charging Time > 10h: {ct_gt_10h:,} ({ct_gt_10h/n_rows*100:.2f}%)\n• Max Total Duration: {td_hr.max():.1f}h",
        transform=ax1.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#CCCCCC"),
    )

    # Right panel: Idle / Overstay Time breakdown + Charging Time Buckets
    b_labels = [b[0] for b in buckets]
    b_vals = [b[1] for b in buckets]
    b_pcts = [v / n_rows * 100.0 for v in b_vals]
    b_colors = [PALETTE_TEAL, PALETTE_PRIMARY, PALETTE_PRIMARY, PALETTE_DARK, PALETTE_ACCENT, PALETTE_MUTED]
    ax2.bar(b_labels, b_vals, color=b_colors, edgecolor="white", width=0.65)
    ax2.set_title("Active Charging Time Breakdown by Duration Bucket")
    ax2.set_xlabel("Active Charging Duration Category")
    ax2.set_ylabel("Number of Sessions")
    ax2.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    ax2.set_ylim(0, max(b_vals) * 1.18)
    for i, (v, p) in enumerate(zip(b_vals, b_pcts)):
        ax2.text(i, v + 1200, f"{v:,}\n({p:.1f}%)", ha="center", va="bottom", fontsize=8.7, fontweight="bold", color=PALETTE_DARK)
    ax2.text(
        0.97, 0.94,
        f"Post-Charge Idle (Overstay):\n• > 15 min idle: {idle_gt_15m:,} ({idle_gt_15m/n_rows*100:.1f}%)\n• > 60 min idle: {idle_gt_60m:,} ({idle_gt_60m/n_rows*100:.1f}%)\n• Anomaly (Chg > Total): {int((idle_min < 0).sum())} rows",
        transform=ax2.transAxes, ha="right", va="top", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF8F0", edgecolor=PALETTE_ACCENT),
    )

    fig.suptitle("EV Charging Duration & Overstay Distribution (n = 259,415 Sessions)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "07_charging_duration_distribution.png"))
    plt.close(fig)

    # =========================================================================
    # 8. HOURLY AND DAILY CHARGING DEMAND PATTERNS (HEATMAP + WEEKDAY VS WEEKEND)
    # =========================================================================
    print("Generating 08_hourly_daily_demand_patterns.png...", flush=True)
    hm = df.pivot_table(index="dow_num", columns="hour", values="energy_kwh", aggfunc="size", fill_value=0)
    hm = hm.reindex(index=range(7), columns=range(24), fill_value=0)
    hm.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Average sessions per active calendar day by hour for Weekday vs Weekend
    n_wkdays = df[~df["is_weekend"]]["date"].nunique()
    n_wkends = df[df["is_weekend"]]["date"].nunique()
    wkday_hr = df[~df["is_weekend"]].groupby("hour").size().reindex(range(24), fill_value=0) / n_wkdays
    wkend_hr = df[df["is_weekend"]].groupby("hour").size().reindex(range(24), fill_value=0) / n_wkends

    # Peak & min cell in 7x24 heatmap
    flat_idx = hm.values.argmax()
    r_max, c_max = divmod(flat_idx, 24)
    flat_min = hm.values.argmin()
    r_min, c_min = divmod(flat_min, 24)

    stats["hourly_daily_patterns"] = {
        "heatmap_peak_cell": {"day": hm.index[r_max], "hour": int(c_max), "sessions": int(hm.iloc[r_max, c_max])},
        "heatmap_min_cell": {"day": hm.index[r_min], "hour": int(c_min), "sessions": int(hm.iloc[r_min, c_min])},
        "weekday_peak_hour": int(wkday_hr.idxmax()),
        "weekday_peak_avg_sessions_per_day": round(float(wkday_hr.max()), 2),
        "weekend_peak_hour": int(wkend_hr.idxmax()),
        "weekend_peak_avg_sessions_per_day": round(float(wkend_hr.max()), 2),
        "active_weekdays_count": int(n_wkdays),
        "active_weekend_days_count": int(n_wkends),
        "weekday_hourly_avg": {str(h): round(float(v), 2) for h, v in wkday_hr.items()},
        "weekend_hourly_avg": {str(h): round(float(v), 2) for h, v in wkend_hr.items()},
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 9.2), gridspec_kw={"height_ratios": [1.15, 1]})

    # Top panel: 7x24 Heatmap
    sns.heatmap(
        hm, ax=ax1, cmap="YlGnBu", linewidths=0.35, linecolor="white",
        cbar_kws={"label": "Total Charging Sessions (2011–2020)"},
    )
    ax1.set_title("A. Heatmap of Charging Sessions by Day of Week and Hour of Day")
    ax1.set_xlabel("Hour of Day (Pacific Local Time)")
    ax1.set_ylabel("Day of Week")
    ax1.set_xticklabels([f"{h:02d}" for h in range(24)], rotation=0)
    ax1.set_yticklabels(hm.index, rotation=0)

    # Bottom panel: Weekday vs Weekend Hourly Demand Curves (normalized per day)
    ax2.plot(wkday_hr.index, wkday_hr.values, color=PALETTE_PRIMARY, marker="o", linewidth=2.4, markersize=5, label=f"Weekday Average (n = {n_wkdays:,} active weekdays)")
    ax2.fill_between(wkday_hr.index, wkday_hr.values, color=PALETTE_PRIMARY, alpha=0.14)
    ax2.plot(wkend_hr.index, wkend_hr.values, color=PALETTE_ACCENT, marker="s", linewidth=2.4, markersize=5, label=f"Weekend Average (n = {n_wkends:,} active weekend days)")
    ax2.fill_between(wkend_hr.index, wkend_hr.values, color=PALETTE_ACCENT, alpha=0.14)

    # Annotate weekday & weekend peaks cleanly above curves
    wd_pk = int(wkday_hr.idxmax())
    we_pk = int(wkend_hr.idxmax())
    ax2.annotate(
        f"Weekday Peak: {wd_pk:02d}:00 ({wkday_hr.loc[wd_pk]:.2f}/day)",
        xy=(wd_pk, wkday_hr.loc[wd_pk]), xytext=(5.5, 7.05),
        arrowprops=dict(arrowstyle="->", color=PALETTE_DARK, lw=1.3),
        fontsize=9, fontweight="bold", color=PALETTE_DARK,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE_PRIMARY, alpha=0.9),
    )
    ax2.annotate(
        f"Weekend Peak: {we_pk:02d}:00 ({wkend_hr.loc[we_pk]:.2f}/day)",
        xy=(we_pk, wkend_hr.loc[we_pk]), xytext=(13.1, 6.95),
        arrowprops=dict(arrowstyle="->", color=PALETTE_ACCENT, lw=1.3),
        fontsize=9, fontweight="bold", color=PALETTE_ACCENT,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=PALETTE_ACCENT, alpha=0.9),
    )

    ax2.set_title("B. Average Hourly Demand Profile: Weekday Commuter Pattern vs. Weekend Midday Pattern")
    ax2.set_xlabel("Hour of Day (Pacific Local Time)")
    ax2.set_ylabel("Mean Sessions per Active Day (Network-Wide)")
    ax2.set_xticks(range(24))
    ax2.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45, ha="right", fontsize=9)
    ax2.set_ylim(0, 8.0)
    ax2.legend(loc="upper right", frameon=True, facecolor="white", fontsize=9.5)

    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, "08_hourly_daily_demand_patterns.png"))
    plt.close(fig)

    with open(STATS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print(f"SUCCESS! Saved all 8 visualizations to {VIS_DIR} and stats to {STATS_JSON}", flush=True)


if __name__ == "__main__":
    main()
