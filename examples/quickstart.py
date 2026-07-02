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
    province_hotspots,
    spatiotemporal_hotspots,
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
    print("Saved quickstart_spatiotemporal_week.png")

    # 6c. For a labeled/colored choropleth per time period, aggregate that
    # same week's raw points to province level (province_hotspots) instead
    # of plotting them as unlabeled points -- now show_labels/label_color
    # apply normally, same as any other province_hotspots result.
    first_week_df = time_df[time_bin_label(time_df["reported_at"], timeframe="week") == first_week]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        first_week_province = province_hotspots(first_week_df, value_col="cases", k=5, permutations=199)
    ax = plot_hotspots(
        first_week_province,
        province="TH10",
        cmap="plasma",
        show_labels=True,
        label_fontsize=9,
        label_color="black",
    )
    ax.set_title(f"Cases: week {first_week}, province-level gi_zscore")
    ax.figure.savefig("quickstart_spatiotemporal_week_province.png", dpi=100)
    print("Saved quickstart_spatiotemporal_week_province.png\n")

    # 6d. Multi-location: the outbreak isn't always in the same place. Bin a
    # nationwide dataset by month, aggregate each month's points to province
    # level separately, and watch the hotspot move -- Chiang Mai in May,
    # Surat Thani in June, Khon Kaen in July.
    moving_df = make_multi_province_outbreak_over_time()
    moving_df["month"] = time_bin_label(moving_df["reported_at"], timeframe="month")
    print("Hotspot province by month:")
    for month in sorted(moving_df["month"].unique()):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
            month_result = province_hotspots(moving_df[moving_df["month"] == month], k=5, permutations=199)
        hot = month_result.loc[month_result["hotspot"] == 1, "province_en"].tolist()
        print(f"  {month}: {hot}")
    print()

    # Multi-level: the same per-month subset also works at district grain --
    # drill into each month's outbreak province at the finer level instead
    # of province, one plot per month.
    month_provinces = {
        "2024-05": ("TH50", "Chiang Mai"),
        "2024-06": ("TH84", "Surat Thani"),
        "2024-07": ("TH40", "Khon Kaen"),
    }
    for month, (province_code, province_en) in month_provinces.items():
        month_df = moving_df[moving_df["month"] == month]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            month_district = district_hotspots(month_df, k=5, permutations=199)
        ax = plot_hotspots(
            month_district,
            province=province_code,
            cmap="inferno",
            show_labels=True,
            label_fontsize=6,
            label_color="white",
        )
        ax.set_title(f"Cases: {month}, {province_en} districts, gi_zscore")
        filename = f"quickstart_spatiotemporal_multi_location_{month}.png"
        ax.figure.savefig(filename, dpi=100)
        print(f"Saved {filename}")


if __name__ == "__main__":
    main()
