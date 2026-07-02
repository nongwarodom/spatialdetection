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

`province_ears`/`district_ears`/`subdistrict_ears` are the temporal
counterpart: same aggregation, but compared against each unit's own history
in prior time bins (EARS C1/C2/C3, see `temporal.py`) rather than its
spatial neighbors within one bin.
"""

from __future__ import annotations

import pandas as pd

from spatialdetection.autocorrelation import getis_ord_hotspots
from spatialdetection.detect import _COLLECTIONS, _LOOKUP_KEYS, _centroids, detect_point
from spatialdetection.io import points_from_dataframe
from spatialdetection.spatiotemporal import time_bin_label
from spatialdetection.temporal import ears_scores

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


def _aggregate_to_level(
    df: pd.DataFrame,
    level: str,
    value_col: str | None,
    lon_col: str,
    lat_col: str,
    pcode_col: str | None,
) -> tuple[pd.DataFrame, str]:
    """Aggregate `df`'s rows onto *every* unit at `level` (zero-filled for units
    with no rows -- required for correct Getis-Ord neighborhood z-scores, and
    for an unbroken per-unit time series when building an EARS panel).

    Returns `(merged, out_col)`: `merged` is a point GeoDataFrame with one
    row per unit at `level` (centroid geometry, so it's plot_hotspots-ready
    without going through getis_ord_hotspots's own lat/lon-to-geometry step)
    and the aggregated `out_col` column ("count", or `value_col` if given).
    """
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
    return points_from_dataframe(merged, lon_col="lon", lat_col="lat"), out_col


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
    merged, out_col = _aggregate_to_level(df, level, value_col, lon_col, lat_col, pcode_col)
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


def _level_ears(
    df: pd.DataFrame,
    level: str,
    time_col: str,
    value_col: str | None,
    timeframe: str,
    lon_col: str,
    lat_col: str,
    pcode_col: str | None,
    baseline_window: int,
) -> pd.DataFrame:
    code_col = _LOOKUP_KEYS[level]
    bins = time_bin_label(df[time_col], timeframe=timeframe)

    panels = []
    for bin_label in sorted(bins.unique()):
        merged, out_col = _aggregate_to_level(df[bins == bin_label], level, value_col, lon_col, lat_col, pcode_col)
        merged["time_bin"] = bin_label
        panels.append(merged)
    panel = pd.concat(panels, ignore_index=True)

    return ears_scores(panel, time_col="time_bin", group_col=code_col, value_col=out_col, baseline_window=baseline_window)


def province_ears(
    df: pd.DataFrame,
    time_col: str,
    value_col: str | None = None,
    timeframe: str = "week",
    lon_col: str = "lon",
    lat_col: str = "lat",
    pcode_col: str | None = None,
    province_col: str | None = None,
    district_col: str | None = None,
    subdistrict_col: str | None = None,
    baseline_window: int = 7,
) -> pd.DataFrame:
    """Province-level EARS temporal-anomaly detection (see `temporal.py`).

    Unlike `province_hotspots` (which flags a province as anomalous relative
    to its *spatial* neighbors within one time period), this flags a
    province as anomalous relative to *its own* history in prior periods --
    a persistent nationwide rise wouldn't stand out spatially (every
    province looks similar to its neighbors) but would stand out here.

    `df`/`value_col`/`pcode_col` etc. work exactly like `province_hotspots`.
    `time_col` names the timestamp column; `timeframe` ("day"/"week"/"month",
    same as `spatiotemporal_hotspots`) bins it, and every province is
    zero-filled in every bin (same reasoning as `province_hotspots`'s
    zero-fill, extended across time so each province's series has no gaps).
    `baseline_window` is how many prior bins each period is compared against
    (EARS's C1/C2 default is 7). Returns one row per (province, time_bin)
    with the aggregated column, `c1`/`c2`/`c3` (z-score-like statistics) and
    `c1_alert`/`c2_alert`/`c3_alert` (booleans) -- see `ears_scores` for what
    each means.
    """
    code_col = _resolve_code_col(pcode_col, province_col, district_col, subdistrict_col)
    return _level_ears(df, "province", time_col, value_col, timeframe, lon_col, lat_col, code_col, baseline_window)


def district_ears(
    df: pd.DataFrame,
    time_col: str,
    value_col: str | None = None,
    timeframe: str = "week",
    lon_col: str = "lon",
    lat_col: str = "lat",
    pcode_col: str | None = None,
    province_col: str | None = None,
    district_col: str | None = None,
    subdistrict_col: str | None = None,
    baseline_window: int = 7,
) -> pd.DataFrame:
    """District-level EARS temporal-anomaly detection. Same as `province_ears`,
    aggregated onto all 928 districts instead -- see that docstring."""
    code_col = _resolve_code_col(pcode_col, province_col, district_col, subdistrict_col)
    return _level_ears(df, "district", time_col, value_col, timeframe, lon_col, lat_col, code_col, baseline_window)


def subdistrict_ears(
    df: pd.DataFrame,
    time_col: str,
    value_col: str | None = None,
    timeframe: str = "week",
    lon_col: str = "lon",
    lat_col: str = "lat",
    pcode_col: str | None = None,
    province_col: str | None = None,
    district_col: str | None = None,
    subdistrict_col: str | None = None,
    baseline_window: int = 7,
) -> pd.DataFrame:
    """Subdistrict-level EARS temporal-anomaly detection. Same as
    `province_ears`, aggregated onto all 7,425 subdistricts instead -- see
    that docstring, and `level_hotspots.py`'s module docstring caveat about
    sparse data at this grain (applies here too: a subdistrict with mostly
    zero periods gives EARS a degenerate, near-zero-variance baseline)."""
    code_col = _resolve_code_col(pcode_col, province_col, district_col, subdistrict_col)
    return _level_ears(
        df, "subdistrict", time_col, value_col, timeframe, lon_col, lat_col, code_col, baseline_window
    )
