"""End-to-end tour of the spatialdetection API using a plain pandas DataFrame.

No GeoDataFrame setup required: dbscan_clusters, cluster_summary, morans_i,
getis_ord_hotspots, spatiotemporal_hotspots, and province_hotspots/
district_hotspots/subdistrict_hotspots all accept a plain DataFrame with
lon/lat columns and build the geometry internally.

Run with:
    uv run python examples/quickstart.py
"""

from __future__ import annotations

import json
import warnings

import matplotlib

matplotlib.use("Agg")  # headless: this script saves a PNG instead of opening a window

import numpy as np
import pandas as pd

from spatialdetection import (
    cluster_summary,
    dbscan_clusters,
    detect_level,
    detect_point,
    district_hotspots,
    getis_ord_hotspots,
    morans_i,
    plot_hotspots,
    plot_level_map,
    province_ears,
    province_hotspots,
    spatiotemporal_hotspots,
    subdistrict_hotspots,
    time_bin_label,
)

rng = np.random.default_rng(0)


def make_sample_cases() -> pd.DataFrame:
    """Synthetic outbreak-style case data: a tight cluster plus scattered background cases."""
    cluster = pd.DataFrame(
        {
            "lon": rng.normal(100.50, 0.01, size=30),
            "lat": rng.normal(13.75, 0.01, size=30),
            "cases": rng.poisson(8, size=30),
        }
    )
    background = pd.DataFrame(
        {
            "lon": rng.uniform(100.3, 100.7, size=40),
            "lat": rng.uniform(13.6, 13.9, size=40),
            "cases": rng.poisson(1, size=40),
        }
    )
    df = pd.concat([cluster, background], ignore_index=True)
    days = pd.to_datetime("2024-06-01") + pd.to_timedelta(rng.integers(0, 5, size=len(df)), unit="D")
    df["reported_at"] = days
    return df


def make_sample_cases_over_weeks() -> pd.DataFrame:
    """Same tight-cluster-plus-background shape as make_sample_cases, but spread over
    9 weeks (not 5 days) with enough daily volume for day/week/month timeframe binning
    to each have enough points per bin (see spatiotemporal_hotspots's k+1-per-bin floor)."""
    n_days = 63
    cluster = pd.DataFrame(
        {
            "lon": rng.normal(100.50, 0.01, size=n_days * 8),
            "lat": rng.normal(13.75, 0.01, size=n_days * 8),
            "cases": rng.poisson(8, size=n_days * 8),
        }
    )
    background = pd.DataFrame(
        {
            "lon": rng.uniform(100.3, 100.7, size=n_days * 8),
            "lat": rng.uniform(13.6, 13.9, size=n_days * 8),
            "cases": rng.poisson(1, size=n_days * 8),
        }
    )
    df = pd.concat([cluster, background], ignore_index=True)
    days = pd.to_datetime("2024-05-01") + pd.to_timedelta(rng.integers(0, n_days, size=len(df)), unit="D")
    df["reported_at"] = days
    return df


def make_multi_province_outbreak() -> pd.DataFrame:
    """Case points spread across several provinces, with one outbreak province."""
    with open("data/thailand_admin_centroids.json") as f:
        provinces = pd.DataFrame(json.load(f)["provinces"])
    background_codes = ["TH11", "TH12", "TH13", "TH14", "TH15", "TH16", "TH17", "TH18", "TH19"]
    outbreak_code = provinces.loc[provinces["province_en"] == "Chiang Mai", "province_code"].iloc[0]
    chosen = provinces[provinces["province_code"].isin([*background_codes, outbreak_code])]

    points = []
    for _, p in chosen.iterrows():
        n = 60 if p["province_code"] == outbreak_code else rng.integers(2, 8)
        points.append(
            pd.DataFrame({"lon": rng.normal(p["lon"], 0.05, size=n), "lat": rng.normal(p["lat"], 0.05, size=n)})
        )
    return pd.concat(points, ignore_index=True)


def make_multi_province_outbreak_over_time() -> pd.DataFrame:
    """Case points spread nationwide, with a *different* outbreak province active
    each month over a 3-month span -- for demonstrating multi-location,
    multi-level spatiotemporal abnormal-area detection together (the same
    dataset lets you both track the hotspot moving month to month, and drill
    from province down to district for a given month)."""
    with open("data/thailand_admin_centroids.json") as f:
        provinces = pd.DataFrame(json.load(f)["provinces"])
    # Same central-Bangkok background provinces as make_multi_province_outbreak,
    # chosen because they're geographically distant from all three outbreak
    # provinces below -- keeps each month's Getis-Ord neighborhood cleanly
    # separated instead of smoothing the outbreak's z-score into its real
    # geographic neighbors (which also happen to be zero-count here).
    background_codes = ["TH11", "TH12", "TH13", "TH14", "TH15", "TH16", "TH17", "TH18", "TH19"]
    background = provinces[provinces["province_code"].isin(background_codes)]
    month_starts = pd.to_datetime(["2024-05-01", "2024-06-01", "2024-07-01"])
    outbreak_provinces_en = ["Chiang Mai", "Surat Thani", "Khon Kaen"]

    frames = []
    for month_start, outbreak_en in zip(month_starts, outbreak_provinces_en):
        for _, p in background.iterrows():
            n = rng.integers(2, 8)
            frames.append(
                pd.DataFrame(
                    {
                        "lon": rng.normal(p["lon"], 0.05, size=n),
                        "lat": rng.normal(p["lat"], 0.05, size=n),
                        "reported_at": month_start + pd.to_timedelta(rng.integers(0, 28, size=n), unit="D"),
                    }
                )
            )
        outbreak = provinces.loc[provinces["province_en"] == outbreak_en].iloc[0]
        frames.append(
            pd.DataFrame(
                {
                    "lon": rng.normal(outbreak["lon"], 0.05, size=60),
                    "lat": rng.normal(outbreak["lat"], 0.05, size=60),
                    "reported_at": month_start + pd.to_timedelta(rng.integers(0, 28, size=60), unit="D"),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def make_temporal_vs_spatial_outbreak() -> pd.DataFrame:
    """Dense weekly case counts for 10 provinces (avoids the +-inf/sparse-data
    EARS caveat), where one province's own history is much lower than its
    neighbors' -- so a late rise there lands in the same range its neighbors
    sit in *every* week (spatially unremarkable), while being a huge jump
    from its own baseline (temporally extreme). Demonstrates why Getis-Ord
    Gi* and EARS are complementary, not interchangeable: a unit can hide in
    plain sight spatially while still standing out against its own history."""
    with open("data/thailand_admin_centroids.json") as f:
        provinces = pd.DataFrame(json.load(f)["provinces"])
    background_codes = ["TH11", "TH12", "TH13", "TH14", "TH15", "TH16", "TH17", "TH18", "TH19"]
    spike_code = provinces.loc[provinces["province_en"] == "Chiang Mai", "province_code"].iloc[0]
    chosen = provinces[provinces["province_code"].isin([*background_codes, spike_code])]

    weeks = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(10) * 7, unit="D")
    frames = []
    for week in weeks:
        for _, p in chosen.iterrows():
            if p["province_code"] == spike_code:
                n = 20 if week == weeks[-1] else int(rng.integers(3, 6))  # low baseline, late jump
            else:
                n = int(rng.integers(15, 25))  # steady, already-high background every week
            frames.append(
                pd.DataFrame(
                    {
                        "lon": rng.normal(p["lon"], 0.05, size=n),
                        "lat": rng.normal(p["lat"], 0.05, size=n),
                        "reported_at": week,
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def _print_top(result: pd.DataFrame, name_col: str, label: str, value_col: str = "count", n: int = 5) -> None:
    """Print the top-n rows by gi_zscore, with the aggregated value/z-score/p-value/significance flag.

    `value_col` must match whatever was passed to province_hotspots/district_hotspots
    (defaults to "count", the row-count column used when no value_col was given there)."""
    cols = [name_col, value_col, "gi_zscore", "gi_pvalue", "hotspot"]
    print(f"{label} -- top {n} by gi_zscore:")
    print(result.sort_values("gi_zscore", ascending=False).head(n)[cols].to_string(index=False))


def _top_code(result: pd.DataFrame, code_col: str) -> str:
    """Return the P-code of the top gi_zscore row -- used to zoom a finer-level plot
    into whichever unit the coarser level flagged, without hardcoding a P-code."""
    return result.sort_values("gi_zscore", ascending=False).iloc[0][code_col]


def main() -> None:
    df = make_sample_cases()
    print(f"{len(df)} case records (plain DataFrame, no geometry built yet)\n")

    # 1. Density-based cluster detection
    labels = dbscan_clusters(df, eps_km=1.5, min_samples=5)
    summary = cluster_summary(df, labels)
    print("DBSCAN clusters:")
    print(summary[["cluster", "size"]].to_string(index=False), "\n")

    # 2. Spatial autocorrelation / hotspot detection on the "cases" column
    moran = morans_i(df, value_col="cases", k=5, permutations=199)
    print(f"Global Moran's I: {moran.I:.3f} (p={moran.p_sim:.3f})\n")

    hotspots = getis_ord_hotspots(df, value_col="cases", k=5, permutations=199)
    n_hot = (hotspots["hotspot"] == 1).sum()
    print(f"Getis-Ord Gi*: {n_hot} significant hotspot point(s)\n")

    # 3. Detect a Thai admin level and auto-plot it
    result = detect_level("TH10")  # Bangkok province P-code
    print(f"detect_level('TH10') -> level={result.level}, lat={result.lat:.4f}, lon={result.lon:.4f}")
    ax = plot_level_map("TH10")
    ax.figure.savefig("quickstart_map.png", dpi=100)
    print("Saved quickstart_map.png\n")

    # 3b. plot_level_map also takes named health_zone/province/district/
    # subdistrict selectors instead of a raw P-code -- health_zone is the
    # odd one out, since a MoPH health zone is a group of provinces (not a
    # single admin unit), so it plots and zooms to all of them together.
    # color sets the highlighted unit's (or units', for health_zone) fill
    # color; show_labels/label_fontsize/label_color annotate each with its
    # name, sized and colored to taste.
    ax = plot_level_map(
        health_zone=1, color="seagreen", show_labels=True, label_fontsize=7, label_color="white"
    )
    ax.figure.savefig("quickstart_health_zone.png", dpi=100)
    print("Saved quickstart_health_zone.png\n")

    # 4. Reverse-geocode the case locations: which subdistrict is each one in?
    located = detect_point(df)
    print("Reverse-geocoded case counts by subdistrict:")
    print(located.groupby("subdistrict_en")["cases"].sum().sort_values(ascending=False).to_string(), "\n")

    # 5. Province-level hotspot detection: aggregate point data onto every
    # province (zero-count provinces included) and run Getis-Ord Gi* there.
    outbreak_df = make_multi_province_outbreak()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        province_result = province_hotspots(outbreak_df, k=5, permutations=499)
    hot = province_result[province_result["hotspot"] == 1]
    print("Hotspot province(s):")
    print(hot[["province_en", "count", "gi_zscore", "gi_pvalue"]].to_string(index=False))

    # 5b. plot_hotspots takes the same health_zone/province/district
    # selectors to restrict the choropleth to one region and zoom to it --
    # here, zooming into health zone 1 (the outbreak province, Chiang Mai,
    # is in it) instead of showing the whole country. cmap picks the
    # colormap; show_labels/label_fontsize/label_color annotate each
    # province with its name.
    ax = plot_hotspots(
        province_result, health_zone=1, cmap="viridis", show_labels=True, label_fontsize=7, label_color="black"
    )
    ax.figure.savefig("quickstart_hotspots_zone1.png", dpi=100)
    print("\nSaved quickstart_hotspots_zone1.png")

    # 5c. Plot the discrete hotspot/coldspot/not-significant flag instead of
    # the continuous gi_zscore: cmap is ignored for value_col="hotspot" in
    # favor of three named, independently adjustable colors with a matching
    # legend.
    ax = plot_hotspots(province_result, value_col="hotspot", hotspot_color="crimson", coldspot_color="steelblue")
    ax.figure.savefig("quickstart_hotspots_flag.png", dpi=100)
    print("Saved quickstart_hotspots_flag.png\n")

    # 6. Spatiotemporal hotspots, binned by day
    by_day = spatiotemporal_hotspots(
        df, time_col="reported_at", value_col="cases", timeframe="day", k=5, permutations=199
    )
    print("Hotspot points per day:")
    print(by_day.groupby("time_bin")["hotspot"].apply(lambda s: (s == 1).sum()).to_string(), "\n")

    # 6b. Same abnormal-area detection at different temporal granularities:
    # timeframe="day"/"week"/"month" all find the same persistent spatial
    # cluster, just binned differently in time. Finer bins (day) resolve
    # short-lived signals more precisely but need more points per bin to
    # avoid being skipped (see the k+1-per-bin floor); coarser bins (month)
    # need less data density but blur together anything that moves within
    # a bin.
    time_df = make_sample_cases_over_weeks()
    for timeframe in ("day", "week", "month"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # sparse-bin skip notice
            by_timeframe = spatiotemporal_hotspots(
                time_df, time_col="reported_at", value_col="cases", timeframe=timeframe, k=5, permutations=199
            )
        n_bins = by_timeframe["time_bin"].nunique()
        n_hot = (by_timeframe["hotspot"] == 1).sum()
        print(f"timeframe={timeframe!r}: {n_bins} bin(s) covered, {n_hot} hotspot point-observation(s)")
    print()

    # spatiotemporal_hotspots stacks every bin's points into one
    # GeoDataFrame, so filter to a single time_bin before plotting a
    # readable map -- here, the first week's cluster. cmap adjusts color as
    # usual; show_labels has no effect on this point-level scatter (there's
    # no admin name to show -- see the plot_hotspots docstring), so it's
    # left off here.
    by_week = spatiotemporal_hotspots(
        time_df, time_col="reported_at", value_col="cases", timeframe="week", k=5, permutations=199
    )
    first_week = sorted(by_week["time_bin"].unique())[0]
    ax = plot_hotspots(by_week[by_week["time_bin"] == first_week], cmap="plasma")
    ax.set_title(f"Cases: week {first_week} gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_week.png", dpi=100)
    print("Saved quickstart_spatiotemporal_week.png\n")

    # 6c. Timeframe detail, one block per granularity: bin time_df's
    # persistent Bangkok cluster at day/week/month grain, aggregate that
    # bin to province, district, AND subdistrict level, and plot the finer
    # levels (more informative once there's enough data -- see
    # level_hotspots.py's degenerate-significance caveat for sparse finer
    # levels). Each subdistrict plot zooms into whichever district the
    # district-level pass flagged (169 Bangkok-wide subdistrict labels
    # would be unreadable). Kept as three separate blocks rather than a
    # loop so each granularity's point count and result tables are easy to
    # compare side by side.

    # -- Day --
    day_binned = time_bin_label(time_df["reported_at"], timeframe="day")
    first_day = sorted(day_binned.unique())[0]
    day_df = time_df[day_binned == first_day]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        day_province = province_hotspots(day_df, value_col="cases", k=5, permutations=199)
        day_district = district_hotspots(day_df, value_col="cases", k=5, permutations=199)
        day_subdistrict = subdistrict_hotspots(day_df, value_col="cases", k=5, permutations=199)
    _print_top(day_province, "province_en", f"Day {first_day!r} ({len(day_df)} points), provinces", value_col="cases")
    _print_top(day_district, "district_en", f"Day {first_day!r}, districts", value_col="cases")
    _print_top(day_subdistrict, "subdistrict_en", f"Day {first_day!r}, subdistricts", value_col="cases")
    ax = plot_hotspots(
        day_district, province="TH10", cmap="plasma", show_labels=True, label_fontsize=6, label_color="black"
    )
    ax.set_title(f"Cases: {first_day}, Bangkok districts, gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_day_district.png", dpi=100)
    print("Saved quickstart_spatiotemporal_day_district.png")
    # Zoom to whichever district the district-level pass flagged, since 169
    # subdistrict labels nationwide (or even just Bangkok's) would be unreadable.
    ax = plot_hotspots(
        day_subdistrict,
        district=_top_code(day_district, "district_code"),
        cmap="plasma",
        show_labels=True,
        label_fontsize=6,
        label_color="black",
    )
    ax.set_title(f"Cases: {first_day}, subdistrict-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_day_subdistrict.png", dpi=100)
    print("Saved quickstart_spatiotemporal_day_subdistrict.png\n")

    # -- Week --
    week_binned = time_bin_label(time_df["reported_at"], timeframe="week")
    first_week_2 = sorted(week_binned.unique())[0]
    week_df = time_df[week_binned == first_week_2]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        week_province = province_hotspots(week_df, value_col="cases", k=5, permutations=199)
        week_district = district_hotspots(week_df, value_col="cases", k=5, permutations=199)
        week_subdistrict = subdistrict_hotspots(week_df, value_col="cases", k=5, permutations=199)
    _print_top(week_province, "province_en", f"Week {first_week_2!r} ({len(week_df)} points), provinces", value_col="cases")
    _print_top(week_district, "district_en", f"Week {first_week_2!r}, districts", value_col="cases")
    _print_top(week_subdistrict, "subdistrict_en", f"Week {first_week_2!r}, subdistricts", value_col="cases")
    ax = plot_hotspots(
        week_province, province="TH10", cmap="plasma", show_labels=True, label_fontsize=9, label_color="black"
    )
    ax.set_title(f"Cases: week {first_week_2}, province-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_week_province.png", dpi=100)
    print("Saved quickstart_spatiotemporal_week_province.png")
    ax = plot_hotspots(
        week_district, province="TH10", cmap="plasma", show_labels=True, label_fontsize=6, label_color="black"
    )
    ax.set_title(f"Cases: week {first_week_2}, Bangkok districts, gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_week_district.png", dpi=100)
    print("Saved quickstart_spatiotemporal_week_district.png")
    ax = plot_hotspots(
        week_subdistrict,
        district=_top_code(week_district, "district_code"),
        cmap="plasma",
        show_labels=True,
        label_fontsize=6,
        label_color="black",
    )
    ax.set_title(f"Cases: week {first_week_2}, subdistrict-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_week_subdistrict.png", dpi=100)
    print("Saved quickstart_spatiotemporal_week_subdistrict.png\n")

    # -- Month --
    month_binned = time_bin_label(time_df["reported_at"], timeframe="month")
    first_month = sorted(month_binned.unique())[0]
    month_df_bangkok = time_df[month_binned == first_month]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        month_province = province_hotspots(month_df_bangkok, value_col="cases", k=5, permutations=199)
        month_district = district_hotspots(month_df_bangkok, value_col="cases", k=5, permutations=199)
        month_subdistrict = subdistrict_hotspots(month_df_bangkok, value_col="cases", k=5, permutations=199)
    _print_top(month_province, "province_en", f"Month {first_month!r} ({len(month_df_bangkok)} points), provinces", value_col="cases")
    _print_top(month_district, "district_en", f"Month {first_month!r}, districts", value_col="cases")
    _print_top(month_subdistrict, "subdistrict_en", f"Month {first_month!r}, subdistricts", value_col="cases")
    ax = plot_hotspots(
        month_province, province="TH10", cmap="plasma", show_labels=True, label_fontsize=9, label_color="black"
    )
    ax.set_title(f"Cases: {first_month}, province-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_month_province.png", dpi=100)
    print("Saved quickstart_spatiotemporal_month_province.png")
    ax = plot_hotspots(
        month_district, province="TH10", cmap="plasma", show_labels=True, label_fontsize=6, label_color="black"
    )
    ax.set_title(f"Cases: {first_month}, Bangkok districts, gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_month_district.png", dpi=100)
    print("Saved quickstart_spatiotemporal_month_district.png")
    ax = plot_hotspots(
        month_subdistrict,
        district=_top_code(month_district, "district_code"),
        cmap="plasma",
        show_labels=True,
        label_fontsize=6,
        label_color="black",
    )
    ax.set_title(f"Cases: {first_month}, subdistrict-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_month_subdistrict.png", dpi=100)
    print("Saved quickstart_spatiotemporal_month_subdistrict.png\n")

    # 6d. Multi-location detail, one block per outbreak: the outbreak isn't
    # always in the same place or time. moving_df has a *different* outbreak
    # province active each month -- Chiang Mai in May, Surat Thani in June,
    # Khon Kaen in July, against the same central-Bangkok background used in
    # make_multi_province_outbreak (see that function for why: keeps each
    # month's Getis-Ord neighborhood cleanly separated from the outbreak's
    # real, zero-count geographic neighbors). Each block below prints the
    # detailed province, district, and subdistrict tables and plots all
    # three levels, each zoomed to that month's outbreak province (or, for
    # subdistrict, to whichever district within it was flagged).
    moving_df = make_multi_province_outbreak_over_time()
    moving_df["month"] = time_bin_label(moving_df["reported_at"], timeframe="month")

    # -- May: Chiang Mai --
    may_df = moving_df[moving_df["month"] == "2024-05"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        may_province = province_hotspots(may_df, k=5, permutations=199)
        may_district = district_hotspots(may_df, k=5, permutations=199)
        may_subdistrict = subdistrict_hotspots(may_df, k=5, permutations=199)
    _print_top(may_province, "province_en", "May 2024 (expect Chiang Mai), provinces")
    _print_top(may_district, "district_en", "May 2024, Chiang Mai districts")
    _print_top(may_subdistrict, "subdistrict_en", "May 2024, Chiang Mai subdistricts")
    ax = plot_hotspots(
        may_province, province="TH50", cmap="viridis", show_labels=True, label_fontsize=8, label_color="black"
    )
    ax.set_title("Cases: May 2024, Chiang Mai province-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_may_province.png", dpi=100)
    print("Saved quickstart_spatiotemporal_may_province.png")
    ax = plot_hotspots(
        may_district, province="TH50", cmap="inferno", show_labels=True, label_fontsize=6, label_color="white"
    )
    ax.set_title("Cases: May 2024, Chiang Mai districts, gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_may_district.png", dpi=100)
    print("Saved quickstart_spatiotemporal_may_district.png")
    ax = plot_hotspots(
        may_subdistrict,
        district=_top_code(may_district, "district_code"),
        cmap="inferno",
        show_labels=True,
        label_fontsize=6,
        label_color="white",
    )
    ax.set_title("Cases: May 2024, Chiang Mai subdistrict-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_may_subdistrict.png", dpi=100)
    print("Saved quickstart_spatiotemporal_may_subdistrict.png\n")

    # -- June: Surat Thani --
    june_df = moving_df[moving_df["month"] == "2024-06"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        june_province = province_hotspots(june_df, k=5, permutations=199)
        june_district = district_hotspots(june_df, k=5, permutations=199)
        june_subdistrict = subdistrict_hotspots(june_df, k=5, permutations=199)
    _print_top(june_province, "province_en", "June 2024 (expect Surat Thani), provinces")
    _print_top(june_district, "district_en", "June 2024, Surat Thani districts")
    _print_top(june_subdistrict, "subdistrict_en", "June 2024, Surat Thani subdistricts")
    ax = plot_hotspots(
        june_province, province="TH84", cmap="viridis", show_labels=True, label_fontsize=8, label_color="black"
    )
    ax.set_title("Cases: June 2024, Surat Thani province-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_june_province.png", dpi=100)
    print("Saved quickstart_spatiotemporal_june_province.png")
    ax = plot_hotspots(
        june_district, province="TH84", cmap="inferno", show_labels=True, label_fontsize=6, label_color="white"
    )
    ax.set_title("Cases: June 2024, Surat Thani districts, gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_june_district.png", dpi=100)
    print("Saved quickstart_spatiotemporal_june_district.png")
    ax = plot_hotspots(
        june_subdistrict,
        district=_top_code(june_district, "district_code"),
        cmap="inferno",
        show_labels=True,
        label_fontsize=6,
        label_color="white",
    )
    ax.set_title("Cases: June 2024, Surat Thani subdistrict-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_june_subdistrict.png", dpi=100)
    print("Saved quickstart_spatiotemporal_june_subdistrict.png\n")

    # -- July: Khon Kaen --
    july_df = moving_df[moving_df["month"] == "2024-07"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        july_province = province_hotspots(july_df, k=5, permutations=199)
        july_district = district_hotspots(july_df, k=5, permutations=199)
        july_subdistrict = subdistrict_hotspots(july_df, k=5, permutations=199)
    _print_top(july_province, "province_en", "July 2024 (expect Khon Kaen), provinces")
    _print_top(july_district, "district_en", "July 2024, Khon Kaen districts")
    _print_top(july_subdistrict, "subdistrict_en", "July 2024, Khon Kaen subdistricts")
    ax = plot_hotspots(
        july_province, province="TH40", cmap="viridis", show_labels=True, label_fontsize=8, label_color="black"
    )
    ax.set_title("Cases: July 2024, Khon Kaen province-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_july_province.png", dpi=100)
    print("Saved quickstart_spatiotemporal_july_province.png")
    ax = plot_hotspots(
        july_district, province="TH40", cmap="inferno", show_labels=True, label_fontsize=6, label_color="white"
    )
    ax.set_title("Cases: July 2024, Khon Kaen districts, gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_july_district.png", dpi=100)
    print("Saved quickstart_spatiotemporal_july_district.png")
    ax = plot_hotspots(
        july_subdistrict,
        district=_top_code(july_district, "district_code"),
        cmap="inferno",
        show_labels=True,
        label_fontsize=6,
        label_color="white",
    )
    ax.set_title("Cases: July 2024, Khon Kaen subdistrict-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_july_subdistrict.png", dpi=100)
    print("Saved quickstart_spatiotemporal_july_subdistrict.png\n")

    # 7. EARS temporal anomaly detection: spatial vs. temporal are genuinely
    # different questions. temporal_df has Chiang Mai sitting at a low,
    # steady 3-5 cases/week for 9 weeks, then jumping to 20 in week 10 --
    # squarely inside the range its neighbors occupy *every* week, so
    # Getis-Ord Gi* (comparing it to those neighbors, within that one week)
    # finds nothing remarkable. EARS (comparing that same week to Chiang
    # Mai's own prior 9 weeks) flags it immediately.
    temporal_df = make_temporal_vs_spatial_outbreak()
    weeks = sorted(temporal_df["reported_at"].unique())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        spatial_final_week = province_hotspots(temporal_df[temporal_df["reported_at"] == weeks[-1]], k=5, permutations=499)
    cm_spatial = spatial_final_week[spatial_final_week["province_en"] == "Chiang Mai"].iloc[0]
    spatial_rank = int((spatial_final_week["gi_zscore"] > cm_spatial["gi_zscore"]).sum()) + 1
    print(
        f"Spatial (Getis-Ord Gi*, final week only): Chiang Mai count={cm_spatial['count']:.0f}, "
        f"gi_zscore={cm_spatial['gi_zscore']:.2f} (rank {spatial_rank}/{len(spatial_final_week)} "
        "by gi_zscore -- unremarkable next to its neighbors that week)"
    )

    ears_result = province_ears(temporal_df, time_col="reported_at", timeframe="week", baseline_window=7)
    cm_temporal = ears_result[ears_result["province_en"] == "Chiang Mai"].sort_values("time_bin")
    print("\nTemporal (EARS, Chiang Mai's own history by week):")
    print(cm_temporal[["time_bin", "count", "c1", "c2", "c2_alert"]].to_string(index=False))

    # Filter to the spike week before plotting -- province_ears stacks every
    # week's rows into one result, like spatiotemporal_hotspots does.
    last_week_ears = ears_result[ears_result["time_bin"] == cm_temporal["time_bin"].iloc[-1]]
    ax = plot_hotspots(
        last_week_ears,
        value_col="c2",
        health_zone=1,
        cmap="magma",
        show_labels=True,
        label_fontsize=8,
        label_color="white",
    )
    ax.set_title(f"Week {cm_temporal['time_bin'].iloc[-1]}: EARS c2 (vs. each province's own history)")
    ax.figure.savefig("quickstart_ears_c2.png", dpi=100)
    print("\nSaved quickstart_ears_c2.png")


if __name__ == "__main__":
    main()
