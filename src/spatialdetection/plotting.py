"""Auto-plot a map for a detected Thai admin level (province/district/subdistrict/point)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from spatialdetection.detect import detect_level

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
