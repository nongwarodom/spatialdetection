"""Density-based spatial cluster detection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from sklearn.cluster import DBSCAN

EARTH_RADIUS_KM = 6371.0088


def dbscan_clusters(
    gdf: gpd.GeoDataFrame,
    eps_km: float = 1.0,
    min_samples: int = 5,
) -> np.ndarray:
    """Cluster points by geographic proximity using DBSCAN with a haversine metric.

    Points are expected to be in a geographic (lon/lat) CRS. Returns an array of
    cluster labels aligned with `gdf`'s rows; -1 marks noise points.
    """
    coords_rad = np.radians(np.column_stack([gdf.geometry.y, gdf.geometry.x]))
    eps_rad = eps_km / EARTH_RADIUS_KM
    model = DBSCAN(eps=eps_rad, min_samples=min_samples, metric="haversine")
    return model.fit_predict(coords_rad)


def cluster_summary(gdf: gpd.GeoDataFrame, labels: np.ndarray) -> gpd.GeoDataFrame:
    """Summarize cluster sizes and centroids, excluding noise (-1)."""
    out = gdf.copy()
    out["cluster"] = labels
    clustered = out[out["cluster"] != -1]
    summary = (
        clustered.groupby("cluster")
        .agg(size=("cluster", "size"), centroid=("geometry", lambda s: s.union_all().centroid))
        .reset_index()
    )
    return gpd.GeoDataFrame(summary, geometry="centroid", crs=gdf.crs)
