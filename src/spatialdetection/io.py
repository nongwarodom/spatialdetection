"""Helpers for loading point data into GeoDataFrames."""

from __future__ import annotations

import geopandas as gpd
import pandas as pd


def points_from_dataframe(
    df: pd.DataFrame,
    lon_col: str = "lon",
    lat_col: str = "lat",
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Build a point GeoDataFrame from columns of longitude/latitude values."""
    return gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )


def load_points(path: str, lon_col: str = "lon", lat_col: str = "lat") -> gpd.GeoDataFrame:
    """Load points from a vector file (GeoJSON/Shapefile/...) or a CSV with lon/lat columns."""
    if path.endswith(".csv"):
        return points_from_dataframe(pd.read_csv(path), lon_col=lon_col, lat_col=lat_col)
    return gpd.read_file(path)
