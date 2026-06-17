"""Helpers for loading point data into GeoDataFrames."""

from __future__ import annotations

from pathlib import Path

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


def load_points(path: str | Path, lon_col: str = "lon", lat_col: str = "lat") -> gpd.GeoDataFrame:
    """Load points from a vector file (GeoJSON/Shapefile/...) or a CSV with lon/lat columns."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return points_from_dataframe(pd.read_csv(path), lon_col=lon_col, lat_col=lat_col)
    return gpd.read_file(path)


def _to_geodataframe(df: pd.DataFrame, lon_col: str = "lon", lat_col: str = "lat") -> gpd.GeoDataFrame:
    """Return `df` as a point GeoDataFrame, building geometry from lon/lat columns if needed.

    Accepts either a plain DataFrame with `lon_col`/`lat_col` columns, or an
    already-built GeoDataFrame (returned as-is). Every clustering/
    autocorrelation function in this package routes its input through here
    so callers don't have to call `points_from_dataframe` themselves first.
    """
    if isinstance(df, gpd.GeoDataFrame):
        return df
    return points_from_dataframe(df, lon_col=lon_col, lat_col=lat_col)
