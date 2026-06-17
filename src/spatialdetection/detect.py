"""Detect the Thai administrative level of an input: P-code, or reverse-geocode lat/lon.

`detect_province`/`detect_district`/`detect_subdistrict`/`detect_level` are
pure lookups against `data/thailand_admin_centroids.json` (code -> location).
`detect_point` is the inverse: it spatially joins a DataFrame of (lat, lon)
points against the subdistrict boundary polygons to resolve which
province/district/subdistrict each point falls inside (location -> code).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd

from spatialdetection.io import _to_geodataframe

_ROOT = Path(__file__).resolve().parent.parent.parent
_CENTROIDS_PATH = _ROOT / "data" / "thailand_admin_centroids.json"
_SUBDISTRICT_BOUNDARY_PATH = _ROOT / "data" / "raw" / "adm3_subdistrict" / "tha_admbnda_adm3_rtsd_20220121.shp"

# Longest pattern first: an 8-digit code also matches a looser 4-digit regex.
_PCODE_PATTERNS = [
    (re.compile(r"^TH\d{6}$"), "subdistrict"),
    (re.compile(r"^TH\d{4}$"), "district"),
    (re.compile(r"^TH\d{2}$"), "province"),
]
_LOOKUP_KEYS = {"province": "province_code", "district": "district_code", "subdistrict": "subdistrict_code"}
_COLLECTIONS = {"province": "provinces", "district": "districts", "subdistrict": "subdistricts"}
_LEVEL_PATTERN = {level: pattern for pattern, level in _PCODE_PATTERNS}

_BOUNDARY_FIELD_RENAME = {
    "ADM1_PCODE": "province_code",
    "ADM1_EN": "province_en",
    "ADM2_PCODE": "district_code",
    "ADM2_EN": "district_en",
    "ADM3_PCODE": "subdistrict_code",
    "ADM3_EN": "subdistrict_en",
}


@dataclass
class LevelResult:
    """Outcome of a detect_province/detect_district/detect_subdistrict/detect_level call."""

    level: str  # "province" | "district" | "subdistrict" | "point"
    code: str | None
    lat: float
    lon: float
    record: dict[str, Any] | None


@lru_cache(maxsize=1)
def _centroids() -> dict[str, Any]:
    with _CENTROIDS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _subdistrict_boundary() -> gpd.GeoDataFrame:
    gdf = gpd.read_file(_SUBDISTRICT_BOUNDARY_PATH)
    return gdf[list(_BOUNDARY_FIELD_RENAME) + ["geometry"]].rename(columns=_BOUNDARY_FIELD_RENAME)


def _lookup(level: str, code: str) -> LevelResult:
    key = _LOOKUP_KEYS[level]
    record = next((r for r in _centroids()[_COLLECTIONS[level]] if r[key] == code), None)
    if record is None:
        raise ValueError(f"{code!r} is not a known {level} P-code in the reference data")
    return LevelResult(level=level, code=code, lat=record["lat"], lon=record["lon"], record=record)


def _normalize_and_validate(code: str, level: str) -> str:
    if not isinstance(code, str):
        raise TypeError(f"{level} P-code must be a str, got {type(code).__name__}")
    code = code.strip().upper()
    pattern = _LEVEL_PATTERN[level]
    if not pattern.match(code):
        expected = {"province": "TH##", "district": "TH####", "subdistrict": "TH######"}[level]
        raise ValueError(f"{code!r} is not a {level} P-code (expected {expected})")
    return code


def detect_province(code: str) -> LevelResult:
    """Look up a province by its P-code (e.g. "TH10")."""
    return _lookup("province", _normalize_and_validate(code, "province"))


def detect_district(code: str) -> LevelResult:
    """Look up a district by its P-code (e.g. "TH1001")."""
    return _lookup("district", _normalize_and_validate(code, "district"))


def detect_subdistrict(code: str) -> LevelResult:
    """Look up a subdistrict by its P-code (e.g. "TH100101")."""
    return _lookup("subdistrict", _normalize_and_validate(code, "subdistrict"))


def detect_point(df: pd.DataFrame, lat_col: str = "lat", lon_col: str = "lon") -> gpd.GeoDataFrame:
    """Reverse-geocode each (lat, lon) row in `df` to its containing Thai admin units.

    `df` can be a plain DataFrame with `lat_col`/`lon_col` columns, or an
    already-built point GeoDataFrame. One spatial join against the
    subdistrict boundary polygons (which already carry their parent
    district/province P-codes as attributes) resolves all three levels at
    once. Points outside every polygon (e.g. outside Thailand) get null
    `*_code`/`*_en` columns.

    Returns `df`'s rows as a GeoDataFrame with `province_code`,
    `province_en`, `district_code`, `district_en`, `subdistrict_code`, and
    `subdistrict_en` columns added.
    """
    points = _to_geodataframe(df, lon_col=lon_col, lat_col=lat_col)
    joined = gpd.sjoin(points, _subdistrict_boundary(), how="left", predicate="within")
    return joined.drop(columns="index_right")


def detect_level(value: str | tuple[float, float] | list[float]) -> LevelResult:
    """Auto-detect whether `value` is a Thai P-code or a raw (lat, lon) point.

    Dispatches to `detect_province`/`detect_district`/`detect_subdistrict`
    for a P-code string (matched by length/format); a `(lat, lon)` pair is
    wrapped as a "point"-level result without reverse-geocoding (use
    `detect_point` for that, on a DataFrame of points).
    """
    if isinstance(value, str):
        code = value.strip().upper()
        for pattern, level in _PCODE_PATTERNS:
            if pattern.match(code):
                return _lookup(level, code)
        raise ValueError(f"{value!r} is not a recognized Thai P-code (expected TH##, TH####, or TH######)")

    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            f"detect_level expects a P-code string or a (lat, lon) pair of length 2, got {value!r}"
        )
    lat, lon = value
    return LevelResult(level="point", code=None, lat=float(lat), lon=float(lon), record=None)
