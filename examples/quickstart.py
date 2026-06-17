"""End-to-end tour of the spatialdetection API using a plain pandas DataFrame.

No GeoDataFrame setup required: dbscan_clusters, cluster_summary, morans_i,
getis_ord_hotspots, and spatiotemporal_hotspots all accept a plain DataFrame
with lon/lat columns and build the geometry internally.

Run with:
    uv run python examples/quickstart.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless: this script saves a PNG instead of opening a window

import numpy as np
import pandas as pd

from spatialdetection import (
    cluster_summary,
    dbscan_clusters,
    detect_level,
    getis_ord_hotspots,
    morans_i,
    plot_level_map,
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
    print("Saved quickstart_map.png")


if __name__ == "__main__":
    main()
