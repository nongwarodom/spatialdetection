"""Detect the Thai administrative level of an input: P-code or raw lat/lon.

Pure lookup logic against `data/thailand_admin_centroids.json` — no plotting
dependencies, so importing this module doesn't pull in matplotlib.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

_CENTROIDS_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "thailand_admin_centroids.json"

# Longest pattern first: an 8-digit code also matches a looser 4-digit regex.
_PCODE_PATTERNS = [
    (re.compile(r"^TH\d{6}$"), "subdistrict"),
    (re.compile(r"^TH\d{4}$"), "district"),
    (re.compile(r"^TH\d{2}$"), "province"),
]
_LOOKUP_KEYS = {"province": "province_code", "district": "district_code", "subdistrict": "subdistrict_code"}
_COLLECTIONS = {"province": "provinces", "district": "districts", "subdistrict": "subdistricts"}
_LEVEL_PATTERN = {level: pattern for pattern, level in _PCODE_PATTERNS}


@dataclass
class LevelResult:
    """Outcome of a detect_* call."""

    level: str  # "province" | "district" | "subdistrict" | "point"
    code: str | None
    lat: float
    lon: float
    record: dict[str, Any] | None


@lru_cache(maxsize=1)
def _centroids() -> dict[str, Any]:
    with _CENTROIDS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


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


def detect_point(lat: float, lon: float) -> LevelResult:
    """Wrap a raw (lat, lon) pair as a `LevelResult` at "point" level."""
    return LevelResult(level="point", code=None, lat=float(lat), lon=float(lon), record=None)


def detect_level(value: str | tuple[float, float] | list[float]) -> LevelResult:
    """Auto-detect whether `value` is a Thai P-code or a raw (lat, lon) point.

    Dispatches to `detect_province`/`detect_district`/`detect_subdistrict` for
    a P-code string (matched by length/format), or `detect_point` for a
    `(lat, lon)` pair. Prefer the explicit `detect_*` functions when you
    already know what kind of input you have.
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
    return detect_point(lat, lon)
