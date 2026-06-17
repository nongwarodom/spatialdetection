"""Auto-plot a map for a detected Thai admin level (province/district/subdistrict/point)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes

from spatialdetection.detect import _LOOKUP_KEYS, detect_level

_RAW = Path(__file__).resolve().parent.parent.parent / "data" / "raw"

_BOUNDARY_PATHS = {
    "province": _RAW / "adm1_province" / "tha_adm1_province.shp",
    "district": _RAW / "adm2_district" / "tha_admbnda_adm2.shp",
    "subdistrict": _RAW / "adm3_subdistrict" / "tha_admbnda_adm3_rtsd_20220121.shp",
}
_CODE_FIELDS = {"province": "ADM1_PCODE", "district": "ADM2_PCODE", "subdistrict": "ADM3_PCODE"}
_NAME_FIELDS = {"province": "ADM1_EN", "district": "ADM2_EN", "subdistrict": "ADM3_EN"}
_PARENT_LEVEL = {"district": "province", "subdistrict": "province"}


@lru_cache(maxsize=None)
def _boundary(level: str) -> gpd.GeoDataFrame:
    return gpd.read_file(_BOUNDARY_PATHS[level])


def plot_level_map(
    value: str | tuple[float, float] | list[float],
    ax: Axes | None = None,
    buffer_deg: float = 0.3,
) -> Axes:
    """Auto-plot a map for `value`, zoomed and styled to its detected admin level."""
    result = detect_level(value)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if result.level == "point":
        _boundary("province").plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.5)
        ax.set_xlim(result.lon - buffer_deg, result.lon + buffer_deg)
        ax.set_ylim(result.lat - buffer_deg, result.lat + buffer_deg)
        ax.scatter([result.lon], [result.lat], color="red", marker="x", s=80, zorder=5)
        ax.set_title(f"Point ({result.lat:.4f}, {result.lon:.4f})")
    else:
        gdf = _boundary(result.level)
        unit = gdf[gdf[_CODE_FIELDS[result.level]] == result.code]
        if unit.empty:
            raise ValueError(f"{result.code!r} not found in {_BOUNDARY_PATHS[result.level].name}")

        parent_level = _PARENT_LEVEL.get(result.level)
        if parent_level:
            _boundary(parent_level).plot(ax=ax, color="whitesmoke", edgecolor="lightgrey", linewidth=0.5)
        gdf.plot(ax=ax, color="whitesmoke", edgecolor="grey", linewidth=0.3)
        unit.plot(ax=ax, color="orange", edgecolor="black", linewidth=1.2)

        minx, miny, maxx, maxy = unit.total_bounds
        pad_x = max((maxx - minx) * 0.5, buffer_deg / 4)
        pad_y = max((maxy - miny) * 0.5, buffer_deg / 4)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        name = unit.iloc[0][_NAME_FIELDS[result.level]]
        ax.set_title(f"{result.level.title()}: {name} ({result.code})")

    ax.set_aspect("equal")
    return ax


def _detect_result_level(result: pd.DataFrame) -> str | None:
    for level in ("subdistrict", "district", "province"):  # finest first: nested code columns overlap
        if _LOOKUP_KEYS[level] in result.columns:
            return level
    return None


def plot_hotspots(result: pd.DataFrame, value_col: str = "gi_zscore", ax: Axes | None = None) -> Axes:
    """Auto-plot Getis-Ord Gi* results, colored by `value_col` on a diverging scale centered at zero.

    Works on the output of `getis_ord_hotspots`/`spatiotemporal_hotspots`
    (point geometries, plotted as-is) or `province_hotspots`/
    `district_hotspots`/`subdistrict_hotspots` (detected by whichever
    `province_code`/`district_code`/`subdistrict_code` column is present,
    then joined back to that level's boundary polygons for a choropleth).

    Defaults to `gi_zscore` rather than the `hotspot` significance flag:
    under skewed counts (one large outbreak among many small/baseline
    values), `hotspot` can omit the real outbreak even though `gi_zscore`
    ranks it correctly -- see `level_hotspots.py`.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    level = _detect_result_level(result)
    if level is not None:
        boundary = _boundary(level)
        # `result` carries its own point geometry (centroids); drop it so the
        # merge doesn't collide with the boundary's polygon geometry column.
        result_attrs = pd.DataFrame(result.drop(columns=result.geometry.name))
        plot_gdf = boundary.merge(result_attrs, left_on=_CODE_FIELDS[level], right_on=_LOOKUP_KEYS[level], how="left")
        plot_gdf = gpd.GeoDataFrame(plot_gdf, geometry=boundary.geometry.name, crs=boundary.crs)
    else:
        plot_gdf = result

    vmax = plot_gdf[value_col].abs().max()
    plot_gdf.plot(
        ax=ax,
        column=value_col,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        legend=True,
        edgecolor="grey",
        linewidth=0.2,
        missing_kwds={"color": "whitesmoke"},
    )

    ax.set_title(f"{level.title() if level else 'Point'}-level {value_col}")
    ax.set_aspect("equal")
    return ax
