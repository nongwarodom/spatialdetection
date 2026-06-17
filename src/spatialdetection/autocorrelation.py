"""Spatial autocorrelation and hotspot (anomaly) detection."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
from esda.getisord import G_Local
from esda.moran import Moran
from libpysal.weights import KNN


def knn_weights(gdf: gpd.GeoDataFrame, k: int = 8) -> KNN:
    """Build a k-nearest-neighbor spatial weights matrix, row-standardized."""
    w = KNN.from_dataframe(gdf, k=k)
    w.transform = "r"
    return w


def morans_i(gdf: gpd.GeoDataFrame, value_col: str, k: int = 8, permutations: int = 999) -> Moran:
    """Global spatial autocorrelation (Moran's I) for a value column."""
    w = knn_weights(gdf, k=k)
    return Moran(gdf[value_col].to_numpy(), w, permutations=permutations)


def getis_ord_hotspots(
    gdf: gpd.GeoDataFrame,
    value_col: str,
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
) -> gpd.GeoDataFrame:
    """Local Getis-Ord Gi* hotspot/coldspot detection.

    Returns `gdf` with added columns: `gi_zscore`, `gi_pvalue`, and `hotspot`
    (1 = significant hotspot, -1 = significant coldspot, 0 = not significant).
    """
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
