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
    getis_ord_hotspots,
    morans_i,
    plot_level_map,
    province_hotspots,
    spatiotemporal_hotspots,
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

    # 3. Spatiotemporal hotspots, binned by day
    by_day = spatiotemporal_hotspots(
        df, time_col="reported_at", value_col="cases", timeframe="day", k=5, permutations=199
    )
    print("Hotspot points per day:")
    print(by_day.groupby("time_bin")["hotspot"].apply(lambda s: (s == 1).sum()).to_string(), "\n")

    # 4. Detect a Thai admin level and auto-plot it
    result = detect_level("TH10")  # Bangkok province P-code
    print(f"detect_level('TH10') -> level={result.level}, lat={result.lat:.4f}, lon={result.lon:.4f}")
    ax = plot_level_map("TH10")
    ax.figure.savefig("quickstart_map.png", dpi=100)
    print("Saved quickstart_map.png\n")

    # 5. Reverse-geocode the case locations: which subdistrict is each one in?
    located = detect_point(df)
    print("Reverse-geocoded case counts by subdistrict:")
    print(located.groupby("subdistrict_en")["cases"].sum().sort_values(ascending=False).to_string(), "\n")

    # 6. Province-level hotspot detection: aggregate point data onto every
    # province (zero-count provinces included) and run Getis-Ord Gi* there.
    outbreak_df = make_multi_province_outbreak()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        province_result = province_hotspots(outbreak_df, k=5, permutations=499)
    hot = province_result[province_result["hotspot"] == 1]
    print("Hotspot province(s):")
    print(hot[["province_en", "count", "gi_zscore", "gi_pvalue"]].to_string(index=False))


if __name__ == "__main__":
    main()
