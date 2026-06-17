"""Spatiotemporal hotspot detection: run Getis-Ord Gi* independently within time bins."""

from __future__ import annotations

import warnings

import geopandas as gpd
import pandas as pd

from spatialdetection.autocorrelation import getis_ord_hotspots
from spatialdetection.io import _to_geodataframe

_TIMEFRAME_ALIASES = {"day": "D", "week": "W", "month": "M"}


def time_bin_label(timestamps: pd.Series, timeframe: str = "day") -> pd.Series:
    """Label each timestamp with its day/week/month bin (e.g. "2024-03-04", "2024-09", "2024-W10").

    `timeframe` is "day", "week", or "month" (or a raw pandas Period alias).
    """
    freq = _TIMEFRAME_ALIASES.get(timeframe, timeframe)
    return pd.to_datetime(timestamps).dt.to_period(freq).astype(str)


def spatiotemporal_hotspots(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    timeframe: str = "day",
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> gpd.GeoDataFrame:
    """Detect Getis-Ord Gi* hotspots/coldspots independently within each time bin.

    `df` can be a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built point GeoDataFrame. Points are grouped into day/week/month
    bins (see `time_bin_label`) and `getis_ord_hotspots` runs separately per
    bin, so a hotspot in one period can't be diluted or masked by activity in
    another. Returns a GeoDataFrame with `time_bin`, `gi_zscore`, `gi_pvalue`,
    and `hotspot` columns added.

    A bin needs at least k+1 points to build a KNN weights matrix; bins with
    fewer points are skipped (with a `UserWarning` naming the bin).
    """
    gdf = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    out = gdf.copy()
    out["time_bin"] = time_bin_label(out[time_col], timeframe)

    results = []
    for bin_label, bin_gdf in out.groupby("time_bin", sort=True):
        if len(bin_gdf) < k + 1:
            warnings.warn(
                f"time bin {bin_label!r} has {len(bin_gdf)} point(s) (< k+1={k + 1}); skipped",
                stacklevel=2,
            )
            continue
        results.append(
            getis_ord_hotspots(bin_gdf, value_col=value_col, k=k, permutations=permutations, alpha=alpha)
        )

    if not results:
        raise ValueError(f"no time bin had >= k+1={k + 1} points; cannot compute hotspots")

    return gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs=gdf.crs)
