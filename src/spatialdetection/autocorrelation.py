"""Spatial autocorrelation and hotspot (anomaly) detection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from esda.getisord import G_Local
from esda.moran import Moran
from libpysal.weights import KNN

from spatialdetection.io import _to_geodataframe


def _require_no_nulls(gdf: gpd.GeoDataFrame, value_col: str) -> None:
    n_missing = gdf[value_col].isna().sum()
    if n_missing:
        raise ValueError(
            f"{value_col!r} has {n_missing} missing value(s); spatial statistics would silently "
            f"return NaN. Drop or impute them first, e.g. df.dropna(subset=[{value_col!r}])."
        )


def knn_weights(df: pd.DataFrame, k: int = 8, lon_col: str = "lon", lat_col: str = "lat") -> KNN:
    """Build a k-nearest-neighbor spatial weights matrix, row-standardized.

    `df` can be a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built point GeoDataFrame.
    """
    gdf = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"
    return w


def morans_i(
    df: pd.DataFrame,
    value_col: str,
    k: int = 8,
    permutations: int = 999,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> Moran:
    """Global spatial autocorrelation (Moran's I) for a value column.

    `df` can be a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built point GeoDataFrame.
    """
    gdf = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    _require_no_nulls(gdf, value_col)
    w = knn_weights(gdf, k=k)
    return Moran(gdf[value_col].to_numpy(), w, permutations=permutations)


def getis_ord_hotspots(
    df: pd.DataFrame,
    value_col: str,
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> gpd.GeoDataFrame:
    """Local Getis-Ord Gi* hotspot/coldspot detection.

    `df` can be a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built point GeoDataFrame. Returns a GeoDataFrame with added
    columns: `gi_zscore`, `gi_pvalue`, and `hotspot` (1 = significant
    hotspot, -1 = significant coldspot, 0 = not significant).
    """
    gdf = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    _require_no_nulls(gdf, value_col)
    # Binary transform gives Gi* an unambiguous self-weight of 1 on the
    # diagonal; row-standardized ("R") transform leaves it to guess.
    w = KNN.from_dataframe(gdf, k=k)
    g = G_Local(gdf[value_col].to_numpy(), w, transform="B", star=True, permutations=permutations)

    out = gdf.copy()
    out["gi_zscore"] = g.Zs
    out["gi_pvalue"] = g.p_sim
    out["hotspot"] = np.where(
        g.p_sim >= alpha, 0, np.where(g.Zs > 0, 1, -1)
    )
    return out
