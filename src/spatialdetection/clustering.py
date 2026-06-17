"""Density-based spatial cluster detection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from spatialdetection.io import _to_geodataframe

EARTH_RADIUS_KM = 6371.0088


def dbscan_clusters(
    df: pd.DataFrame,
    eps_km: float = 1.0,
    min_samples: int = 5,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> np.ndarray:
    """Cluster points by geographic proximity using DBSCAN with a haversine metric.

    `df` can be a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built point GeoDataFrame in a geographic CRS. Returns an array of
    cluster labels aligned with `df`'s rows; -1 marks noise points.
    """
    gdf = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    if gdf.crs is None or not gdf.crs.is_geographic:
        raise ValueError(
            f"dbscan_clusters needs a geographic (lon/lat degrees) CRS for its haversine "
            f"distance math, got {gdf.crs!r}. Reproject with .to_crs('EPSG:4326') first."
        )
    coords_rad = np.radians(np.column_stack([gdf.geometry.y, gdf.geometry.x]))
    eps_rad = eps_km / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    return model.fit_predict(coords_rad)


def cluster_summary(
    df: pd.DataFrame,
    labels: np.ndarray,
    lon_col: str = "lon",
    lat_col: str = "lat",
) -> gpd.GeoDataFrame:
    """Summarize cluster sizes and centroids, excluding noise (-1).

    `df` can be a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built point GeoDataFrame — same input `dbscan_clusters` took to
    produce `labels`.
    """
    gdf = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    out = gdf.copy()
    out["cluster"] = labels
    clustered = out[out["cluster"] != -1]
    summary = (
        clustered.groupby("cluster")
        .agg(size=("cluster", "size"), centroid=("geometry", lambda s: s.union_all().centroid))
        .reset_index()
    )
    return gpd.GeoDataFrame(summary, geometry="centroid", crs=gdf.crs)
