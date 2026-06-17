"""Getis-Ord Gi* hotspot detection aggregated to a Thai admin level.

`province_hotspots`/`district_hotspots`/`subdistrict_hotspots` aggregate
per-row data onto the *full* list of units at that level (so units with no
observed rows are included as zero, not missing -- required for correct
Getis-Ord neighborhood z-scores), then run `getis_ord_hotspots` on the
resulting per-unit centroids. Two input modes:

- lat/lon point data (default): each row is reverse-geocoded to its admin
  unit with `detect_point`.
- `pcode_col` (or its level-named synonyms `province_col`/`district_col`/
  `subdistrict_col` -- pass whichever reads best, they're identical):
  skips the spatial join entirely. If your data already carries a P-code
  per row (subdistrict, district, or province -- as real line-list data
  often does), the parent codes for a coarser level come from string
  slicing (P-codes are nested), and aggregation is a plain groupby.
  Faster, and works without any lat/lon at all.

Caveat: finer levels have far more units (77 provinces vs. 928 districts vs.
7,425 subdistricts) competing for the same data. If most units end up with
zero observed rows, the permutation test's reference distribution becomes
degenerate (many identical all-zero neighborhoods), which can yield
artificially extreme p-values rather than a meaningful signal. Prefer
`province_hotspots` or `district_hotspots` unless your data is dense enough
that most subdistricts have a nonzero count.
"""

from __future__ import annotations

import pandas as pd

from spatialdetection.autocorrelation import getis_ord_hotspots
from spatialdetection.detect import _COLLECTIONS, _LOOKUP_KEYS, _centroids, detect_point

_CODE_LENGTH = {"province": 4, "district": 6, "subdistrict": 8}
_EXAMPLE_CODE = {"province": "TH10", "district": "TH1001", "subdistrict": "TH100101"}


def _resolve_code_col(
    pcode_col: str | None, province_col: str | None, district_col: str | None, subdistrict_col: str | None
) -> str | None:
    given = [
        (name, val)
        for name, val in [
            ("pcode_col", pcode_col),
            ("province_col", province_col),
            ("district_col", district_col),
            ("subdistrict_col", subdistrict_col),
        ]
        if val is not None
    ]
    if len(given) > 1:
        names = ", ".join(name for name, _ in given)
        raise ValueError(
            f"pass only one of pcode_col/province_col/district_col/subdistrict_col (they're "
            f"synonyms), got {len(given)}: {names}"
        )
    return given[0][1] if given else None


def _aggregate_by_code(codes: pd.Series, values: pd.Series | None, out_col: str, code_col: str) -> pd.Series:
    codes = codes.rename(code_col)  # groupby uses the key's .name for the result index name
    if values is None:
        return codes.groupby(codes).size().rename(out_col)
    return values.groupby(codes).sum().rename(out_col)


def _codes_from_pcode_col(df: pd.DataFrame, pcode_col: str, level: str) -> pd.Series:
    codes = df[pcode_col].astype(str).str.strip().str.upper()
    min_len = _CODE_LENGTH[level]
    too_short = codes.str.len() < min_len
    if too_short.any():
        raise ValueError(
            f"{pcode_col!r} has {int(too_short.sum())} code(s) shorter than a {level} "
            f"P-code (need >= {min_len} chars, e.g. {_EXAMPLE_CODE[level]!r}); "
            f"can't derive {level}-level codes from a coarser input."
        )
    return codes.str[:min_len]


def _level_hotspots(
    df: pd.DataFrame,
    level: str,
    value_col: str | None,
    lon_col: str,
    lat_col: str,
    pcode_col: str | None,
    k: int,
    permutations: int,
    alpha: float,
) -> pd.DataFrame:
    code_col = _LOOKUP_KEYS[level]
    out_col = value_col if value_col is not None else "count"

    if pcode_col is not None:
        codes = _codes_from_pcode_col(df, pcode_col, level)
        agg = _aggregate_by_code(codes, df[value_col] if value_col is not None else None, out_col, code_col)
    else:
        located = detect_point(df, lat_col=lat_col, lon_col=lon_col)
        values = located[value_col] if value_col is not None else None
        agg = _aggregate_by_code(located[code_col], values, out_col, code_col)

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
    pcode_col: str | None = None,
    province_col: str | None = None,
    district_col: str | None = None,
    subdistrict_col: str | None = None,
) -> pd.DataFrame:
    """Province-level Getis-Ord Gi* hotspot detection.

    `df` is either point-level data with `lon_col`/`lat_col` columns (each
    row reverse-geocoded to its province), or -- if one of `pcode_col`/
    `province_col`/`district_col`/`subdistrict_col` is given (identical
    synonyms; pass whichever name matches your column) -- data that
    already carries a P-code per row (rolled up to province via string
    slicing, no lat/lon needed). Rows are aggregated -- summed by
    `value_col` if given, else counted -- onto all 77 provinces. Returns
    one row per province with the aggregated column, `gi_zscore`,
    `gi_pvalue`, and `hotspot` (1 = significant hotspot, -1 = significant
    coldspot, 0 = not significant).
    """
    code_col = _resolve_code_col(pcode_col, province_col, district_col, subdistrict_col)
    return _level_hotspots(df, "province", value_col, lon_col, lat_col, code_col, k, permutations, alpha)


def district_hotspots(
    df: pd.DataFrame,
    value_col: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
    pcode_col: str | None = None,
    province_col: str | None = None,
    district_col: str | None = None,
    subdistrict_col: str | None = None,
) -> pd.DataFrame:
    """District-level Getis-Ord Gi* hotspot detection.

    Same as `province_hotspots`, aggregated onto all 928 districts instead.
    A code passed via `pcode_col`/`province_col`/`district_col`/
    `subdistrict_col` must be at least district-grained (a province-only
    code can't be rolled up to a finer level).
    """
    code_col = _resolve_code_col(pcode_col, province_col, district_col, subdistrict_col)
    return _level_hotspots(df, "district", value_col, lon_col, lat_col, code_col, k, permutations, alpha)


def subdistrict_hotspots(
    df: pd.DataFrame,
    value_col: str | None = None,
    lon_col: str = "lon",
    lat_col: str = "lat",
    k: int = 8,
    permutations: int = 999,
    alpha: float = 0.05,
    pcode_col: str | None = None,
    province_col: str | None = None,
    district_col: str | None = None,
    subdistrict_col: str | None = None,
) -> pd.DataFrame:
    """Subdistrict-level Getis-Ord Gi* hotspot detection.

    Same as `province_hotspots`, aggregated onto all 7,425 subdistricts
    instead. A code passed via `pcode_col`/`province_col`/`district_col`/
    `subdistrict_col` must be subdistrict-grained.
    """
    code_col = _resolve_code_col(pcode_col, province_col, district_col, subdistrict_col)
    return _level_hotspots(df, "subdistrict", value_col, lon_col, lat_col, code_col, k, permutations, alpha)
