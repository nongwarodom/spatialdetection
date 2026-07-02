"""Auto-plot a map for a detected Thai admin level (province/district/subdistrict/point)."""

from __future__ import annotations

import warnings
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.patches import Patch

from spatialdetection.detect import _LOOKUP_KEYS, _normalize_and_validate, detect_level, detect_point
from spatialdetection.health_zones import health_zone_province_codes
from spatialdetection.level_hotspots import _CODE_LENGTH

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


def _label_units(ax: Axes, gdf: gpd.GeoDataFrame, name_col: str, fontsize: float, color: str) -> None:
    """Annotate each polygon with its name at a point guaranteed to fall inside it."""
    geom_col = gdf.geometry.name
    for _, row in gdf.iterrows():
        if pd.isna(row[name_col]) or row[geom_col] is None:
            continue
        point = row[geom_col].representative_point()
        ax.annotate(
            str(row[name_col]),
            xy=(point.x, point.y),
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
        )


def _plot_health_zone(
    zone: int,
    ax: Axes,
    buffer_deg: float,
    color: str,
    show_labels: bool,
    label_fontsize: float,
    label_color: str,
) -> Axes:
    codes = set(health_zone_province_codes(zone))
    gdf = _boundary("province")
    in_zone = gdf[gdf[_CODE_FIELDS["province"]].isin(codes)]
    if in_zone.empty:
        raise ValueError(f"no provinces found for health zone {zone!r}")

    gdf.plot(ax=ax, color="whitesmoke", edgecolor="lightgrey", linewidth=0.3)
    in_zone.plot(ax=ax, color=color, edgecolor="black", linewidth=1.0)

    minx, miny, maxx, maxy = in_zone.total_bounds
    pad_x = max((maxx - minx) * 0.1, buffer_deg / 4)
    pad_y = max((maxy - miny) * 0.1, buffer_deg / 4)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)

    if show_labels:
        _label_units(ax, in_zone, _NAME_FIELDS["province"], label_fontsize, label_color)

    ax.set_title(f"Health zone {zone}: {len(in_zone)} provinces")
    ax.set_aspect("equal")
    return ax


def plot_level_map(
    value: str | tuple[float, float] | list[float] | None = None,
    ax: Axes | None = None,
    buffer_deg: float = 0.3,
    health_zone: int | None = None,
    province: str | None = None,
    district: str | None = None,
    subdistrict: str | None = None,
    color: str = "orange",
    show_labels: bool = False,
    label_fontsize: float = 8,
    label_color: str = "black",
) -> Axes:
    """Auto-plot a map for `value`, zoomed and styled to its detected admin level.

    `value` is a P-code string or `(lat, lon)` pair, dispatched by
    `detect_level` (same as before). Pass exactly one of `value`,
    `health_zone` (1-13, a MoPH เขตสุขภาพ grouping of provinces -- see
    `health_zones.py`), `province`, `district`, or `subdistrict` (the latter
    three take a P-code, e.g. `province="TH10"` -- identical to
    `value="TH10"`, just named for readability). `health_zone` is the odd
    one out: it has no single P-code, so it plots every province in the zone
    highlighted together, zoomed to their combined bounds, rather than one
    unit.

    `color` sets the highlighted unit's (or, for `health_zone`, all units
    in the zone's) fill color -- any matplotlib color spec.

    `show_labels=True` annotates each highlighted unit with its name; only
    meaningful for `health_zone` (multiple provinces) since a single
    province/district/subdistrict is already named in the title.
    `label_fontsize` and `label_color` control that annotation's size and
    text color.
    """
    given = [
        (name, v)
        for name, v in [
            ("value", value),
            ("health_zone", health_zone),
            ("province", province),
            ("district", district),
            ("subdistrict", subdistrict),
        ]
        if v is not None
    ]
    if len(given) != 1:
        names = ", ".join(name for name, _ in given) or "none"
        raise ValueError(
            f"pass exactly one of value/health_zone/province/district/subdistrict, got {len(given)}: {names}"
        )
    name, selected = given[0]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    if name == "health_zone":
        return _plot_health_zone(selected, ax, buffer_deg, color, show_labels, label_fontsize, label_color)

    result = detect_level(selected)

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
        unit.plot(ax=ax, color=color, edgecolor="black", linewidth=1.2)

        minx, miny, maxx, maxy = unit.total_bounds
        pad_x = max((maxx - minx) * 0.5, buffer_deg / 4)
        pad_y = max((maxy - miny) * 0.5, buffer_deg / 4)
        ax.set_xlim(minx - pad_x, maxx + pad_x)
        ax.set_ylim(miny - pad_y, maxy + pad_y)

        if show_labels:
            _label_units(ax, unit, _NAME_FIELDS[result.level], label_fontsize, label_color)

        name = unit.iloc[0][_NAME_FIELDS[result.level]]
        ax.set_title(f"{result.level.title()}: {name} ({result.code})")

    ax.set_aspect("equal")
    return ax


def _detect_result_level(result: pd.DataFrame) -> str | None:
    for level in ("subdistrict", "district", "province"):  # finest first: nested code columns overlap
        if _LOOKUP_KEYS[level] in result.columns:
            return level
    return None


def _region_filter(health_zone: int | None, province: str | None, district: str | None) -> tuple[str, tuple[str, ...]] | None:
    """Resolve the health_zone/province/district selector into (filter_level, code_prefixes).

    Returns None if no filter was given. P-codes are nested strings (a
    subdistrict code starts with its district code, which starts with its
    province code), so matching any finer-grained code column against these
    prefixes with `.str.startswith` restricts to the selected region.
    """
    given = [(n, v) for n, v in [("health_zone", health_zone), ("province", province), ("district", district)] if v is not None]
    if len(given) > 1:
        names = ", ".join(name for name, _ in given)
        raise ValueError(f"pass only one of health_zone/province/district, got {len(given)}: {names}")
    if not given:
        return None
    name, value = given[0]
    if name == "health_zone":
        return "province", tuple(health_zone_province_codes(value))
    if name == "province":
        return "province", (_normalize_and_validate(value, "province"),)
    return "district", (_normalize_and_validate(value, "district"),)


def _zoom_to_bounds(ax: Axes, gdf: gpd.GeoDataFrame) -> None:
    minx, miny, maxx, maxy = gdf.total_bounds
    pad_x = max((maxx - minx) * 0.1, 0.05)
    pad_y = max((maxy - miny) * 0.1, 0.05)
    ax.set_xlim(minx - pad_x, maxx + pad_x)
    ax.set_ylim(miny - pad_y, maxy + pad_y)


def plot_hotspots(
    result: pd.DataFrame,
    value_col: str = "gi_zscore",
    ax: Axes | None = None,
    health_zone: int | None = None,
    province: str | None = None,
    district: str | None = None,
    cmap: str = "coolwarm",
    show_labels: bool = False,
    label_fontsize: float = 8,
    label_color: str = "black",
    hotspot_color: str = "red",
    coldspot_color: str = "blue",
    not_significant_color: str = "whitesmoke",
) -> Axes:
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

    Pass at most one of `health_zone` (1-13, a MoPH เขตสุขภาพ grouping of
    provinces -- see `health_zones.py`), `province` (a province P-code, e.g.
    "TH10"), or `district` (a district P-code, e.g. "TH1001") to restrict
    the map to that region and zoom to its bounds. For point-level results
    (no admin code column yet) this reverse-geocodes each point with
    `detect_point` first. A level-aggregated result can only be filtered at
    its own grain or coarser -- e.g. a `district_hotspots` result can be
    filtered by `district` or `province`, but a `province_hotspots` result
    can't be filtered by `district` (there's no per-district row to keep).

    `cmap` is any matplotlib colormap name (default `"coolwarm"`, a
    diverging scale suited to values centered at zero like `gi_zscore`) --
    used for any continuous `value_col`. If `value_col="hotspot"` (the
    discrete significance flag: 1/-1/0), `cmap` is ignored and each unit is
    instead colored by category using `hotspot_color`/`coldspot_color`/
    `not_significant_color`, with a matching legend.

    `show_labels=True` annotates each plotted unit with its name (choropleth
    results only -- point-level results have no admin name to show, and
    emit a warning if `show_labels` is requested for them). `label_fontsize`
    and `label_color` control that annotation's size and text color.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    region = _region_filter(health_zone, province, district)
    level = _detect_result_level(result)

    if level is not None:
        if region is not None and _CODE_LENGTH[level] < _CODE_LENGTH[region[0]]:
            raise ValueError(
                f"{level}-level results are coarser than a {region[0]} filter; rerun with a level at "
                f"least as fine as {region[0]} (e.g. district_hotspots/subdistrict_hotspots) to filter by {region[0]}."
            )
        boundary = _boundary(level)
        # `result` carries its own point geometry (centroids); drop it so the
        # merge doesn't collide with the boundary's polygon geometry column.
        result_attrs = pd.DataFrame(result.drop(columns=result.geometry.name))
        plot_gdf = boundary.merge(result_attrs, left_on=_CODE_FIELDS[level], right_on=_LOOKUP_KEYS[level], how="left")
        plot_gdf = gpd.GeoDataFrame(plot_gdf, geometry=boundary.geometry.name, crs=boundary.crs)
        code_col = _LOOKUP_KEYS[level]
    else:
        plot_gdf = detect_point(result) if region is not None else result
        code_col = _LOOKUP_KEYS[region[0]] if region is not None else None

    if region is not None:
        _, prefixes = region
        codes = plot_gdf[code_col].astype(str).str.upper()
        plot_gdf = plot_gdf[codes.str.startswith(prefixes)]
        if plot_gdf.empty:
            raise ValueError(f"no rows fall within the selected region ({prefixes})")

    if value_col == "hotspot":
        category_colors = {1: hotspot_color, -1: coldspot_color, 0: not_significant_color}
        category_labels = {1: "Hotspot", -1: "Coldspot", 0: "Not significant"}
        plot_gdf.plot(
            ax=ax,
            color=plot_gdf[value_col].map(category_colors),
            edgecolor="grey",
            linewidth=0.2,
        )
        ax.legend(
            handles=[Patch(facecolor=category_colors[c], edgecolor="grey", label=category_labels[c]) for c in (1, -1, 0)],
            loc="lower left",
        )
    else:
        vmax = plot_gdf[value_col].abs().max()
        plot_gdf.plot(
            ax=ax,
            column=value_col,
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
            legend=True,
            edgecolor="grey",
            linewidth=0.2,
            missing_kwds={"color": "whitesmoke"},
        )

    if region is not None:
        _zoom_to_bounds(ax, plot_gdf)

    if show_labels:
        if level is None:
            warnings.warn("show_labels has no effect on point-level plot_hotspots results", stacklevel=2)
        else:
            _label_units(ax, plot_gdf, _NAME_FIELDS[level], label_fontsize, label_color)

    ax.set_title(f"{level.title() if level else 'Point'}-level {value_col}")
    ax.set_aspect("equal")
    return ax
