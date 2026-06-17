"""Getis-Ord Gi* hotspot detection aggregated to a Thai admin level.

`province_hotspots`/`district_hotspots`/`subdistrict_hotspots` take
point-level (lat, lon) data, reverse-geocode each point to its admin unit
with `detect_point`, aggregate onto the *full* list of units at that level
(so units with zero observed points are included as zero, not missing --
required for correct Getis-Ord neighborhood z-scores), and run
`getis_ord_hotspots` on the resulting per-unit centroids.

Caveat: finer levels have far more units (77 provinces vs. 928 districts vs.
7,425 subdistricts) competing for the same point data. If most units end up
with zero observed points, the permutation test's reference distribution
becomes degenerate (many identical all-zero neighborhoods), which can yield
artificially extreme p-values rather than a meaningful signal. Prefer
`province_hotspots` or `district_hotspots` unless your point data is dense
enough that most subdistricts have a nonzero count.
"""

from __future__ import annotations

import pandas as pd

from spatialdetection.autocorrelation import getis_ord_hotspots
from spatialdetection.detect import _COLLECTIONS, _LOOKUP_KEYS, _centroids, detect_point


def _level_hotspots(
    df: pd.DataFrame,
    level: str,
    value_col: str | None,
    lon_col: str,
    lat_col: str,
    k: int,
    permutations: int,
    alpha: float,
) -> pd.DataFrame:
    located = detect_point(df, lat_col=lat_col, lon_col=lon_col)
    code_col = _LOOKUP_KEYS[level]
    out_col = value_col if value_col is not None else "count"

    if value_col is None:
        agg = located.groupby(code_col).size().rename(out_col)
    else:
        agg = located.groupby(code_col)[value_col].sum().rename(out_col)

    units = pd.DataFrame(_centroids()[_COLLECTIONS[level]])
    merged = units.merge(agg, on=code_col, how="left").fillna({out_col: 0})

    return getis_ord_hotspots(merged, value_col=out_col, k=k, permutations=permutations, alpha=alpha)


def province_hotspots(
    df: pd.DataFrame,
    value_col: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Province-level Getis-Ord Gi* hotspot detection from point data.

    `df` is point-level data with `lon_col`/`lat_col` columns (e.g. case
    locations). Each point is reverse-geocoded to its province, then
    aggregated -- summed by `value_col` if given, else counted per point --
    onto all 77 provinces. Returns one row per province with the aggregated
    column, `gi_zscore`, `gi_pvalue`, and `hotspot` (1 = significant
    hotspot, -1 = significant coldspot, 0 = not significant).
    """
    return _level_hotspots(df, "province", value_col, lon_col, lat_col, k, permutations, alpha)


def district_hotspots(
    df: pd.DataFrame,
    value_col: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """District-level Getis-Ord Gi* hotspot detection from point data.

    Same as `province_hotspots`, aggregated onto all 928 districts instead.
    """
    return _level_hotspots(df, "district", value_col, lon_col, lat_col, k, permutations, alpha)


def subdistrict_hotspots(
    df: pd.DataFrame,
    value_col: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Subdistrict-level Getis-Ord Gi* hotspot detection from point data.

    Same as `province_hotspots`, aggregated onto all 7,425 subdistricts
    instead.
    """
    return _level_hotspots(df, "subdistrict", value_col, lon_col, lat_col, k, permutations, alpha)
