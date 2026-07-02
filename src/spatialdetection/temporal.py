"""EARS (Early Aberration Reporting System) temporal anomaly detection.

`getis_ord_hotspots`/`province_hotspots`/`district_hotspots`/`subdistrict_hotspots`
all flag a unit as anomalous relative to its *spatial* neighbors within one
time period. None of them look backward in time -- a unit that's high but
unremarkable next to its neighbors, yet far above its own recent history,
won't show up. This module is the complementary, purely *temporal* check:
each unit's value is compared only to its own past values, independent of
where it is or what its neighbors are doing.

Implements the CDC's C1-MILD/C2-MEDIUM/C3-ULTRA algorithms (Hutwagner et al.
2003, "The bioterrorism preparedness and response Early Aberration Reporting
System (EARS)"):

- C1: baseline = the `baseline_window` periods immediately before the
  current one. Reacts fastest, but an already-building signal creeps into
  its own baseline, damping sensitivity as an outbreak continues.
- C2: baseline = the `baseline_window` periods ending 2 periods before the
  current one (skips the 2 most recent periods as a "guard band"), avoiding
  that self-contamination. This is EARS's standard/default algorithm.
- C3: sums each of the current and prior 2 periods' C2 excess over 1 (0 if
  negative), so it only fires on a *sustained* elevation across 3
  consecutive periods rather than a single spike -- least sensitive, most
  specific.

All three are z-score-like: `(value - baseline_mean) / baseline_std`. A
zero-variance baseline (every prior value identical) makes this ±inf if the
current value differs, or NaN if it doesn't -- both propagate naturally
through the `> threshold` alert comparison.

`spatial_ears_scores` is a variant for when a unit's own history is too thin
to give C1/C2/C3 a meaningful baseline (the ±inf case above, common at finer
admin grains): it pools a unit's `k` nearest spatial neighbors' historical
values into that unit's baseline, alongside its own. The current value being
tested is still purely the unit's own -- this widens the *baseline*, not the
comparison -- so it's a different kind of "spatial" than `getis_ord_hotspots`
(which compares current values across neighbors within one period). It only
helps when the unit's neighbors sit at a comparable rate to its own; see the
exchangeability caveat in `spatial_ears_scores`'s docstring before using it
on units whose neighbors are at a different scale.
"""

from __future__ import annotations

import pandas as pd

from spatialdetection.autocorrelation import knn_weights

C1_ALERT_THRESHOLD = 3.0
C2_ALERT_THRESHOLD = 3.0
C3_ALERT_THRESHOLD = 2.0


def _rolling_c_stat(values: pd.Series, window: int, gap: int) -> pd.Series:
    """(value - baseline_mean) / baseline_std, baseline = the `window` values
    ending `gap` periods before the current one (gap=0 for C1, gap=2 for C2)."""
    baseline = values.shift(gap + 1)
    baseline_mean = baseline.rolling(window).mean()
    baseline_std = baseline.rolling(window).std(ddof=1)
    return (values - baseline_mean) / baseline_std


def _ears_for_one_group(values: pd.Series, baseline_window: int) -> pd.DataFrame:
    c1 = _rolling_c_stat(values, baseline_window, gap=0)
    c2 = _rolling_c_stat(values, baseline_window, gap=2)
    c2_excess = (c2 - 1).clip(lower=0)
    c3 = c2_excess + c2_excess.shift(1) + c2_excess.shift(2)
    return pd.DataFrame({"c1": c1, "c2": c2, "c3": c3})


def ears_scores(
    panel: pd.DataFrame,
    time_col: str,
    group_col: str,
    value_col: str,
    baseline_window: int = 7,
) -> pd.DataFrame:
    """Compute EARS C1/C2/C3 statistics for each `group_col` unit's `value_col`
    time series, independently per unit.

    `panel` must already be a *complete* (time x group) panel: one row per
    unit per time period, zero-filled for periods with no observations (a
    missing period would silently shrink or misalign the baseline window
    instead of correctly reading as "zero that period"). `province_hotspots`/
    `district_hotspots`/`subdistrict_hotspots` already zero-fill every unit
    for a *single* period; `province_ears`/`district_ears`/`subdistrict_ears`
    in `level_hotspots.py` do the same across every period to build this
    panel automatically -- use those instead of calling this directly unless
    you already have a custom panel.

    Adds `c1`/`c2`/`c3` columns (NaN for a unit's first `baseline_window` (+
    2 for c2/c3) periods, before it has enough history) and matching
    `c1_alert`/`c2_alert`/`c3_alert` boolean columns using EARS's standard
    thresholds (`C1_ALERT_THRESHOLD`/`C2_ALERT_THRESHOLD` = 3.0,
    `C3_ALERT_THRESHOLD` = 2.0). C2 is EARS's default/most commonly used
    algorithm; C1 is more reactive but noisier, C3 needs a sustained signal.
    """
    out = panel.sort_values([group_col, time_col]).reset_index(drop=True)
    stat_frames = [
        _ears_for_one_group(group_df[value_col], baseline_window)
        for _, group_df in out.groupby(group_col, sort=False)
    ]
    out = out.join(pd.concat(stat_frames))
    out["c1_alert"] = out["c1"] > C1_ALERT_THRESHOLD
    out["c2_alert"] = out["c2"] > C2_ALERT_THRESHOLD
    out["c3_alert"] = out["c3"] > C3_ALERT_THRESHOLD
    return out


def _pooled_rolling_c_stat(wide: pd.DataFrame, target: str, neighbor_cols: list, window: int, gap: int) -> pd.Series:
    """Same z-score-like statistic as `_rolling_c_stat`, but the baseline mean/std
    is pooled across `target`'s own `window` lagged values AND its k nearest
    neighbors' values over that same lagged window -- `window * (1 + len(neighbor_cols))`
    data points feeding the baseline instead of just `window`. The numerator
    (current value) is still target's own -- only the baseline is spatially widened."""
    cols = [target, *neighbor_cols]
    shifted = wide[cols].shift(gap + 1)
    n = window * len(cols)
    # skipna=False: a row with even one column still short of a full window
    # must propagate NaN, not silently treat that column's contribution as 0.
    pooled_sum = shifted.rolling(window).sum().sum(axis=1, skipna=False)
    pooled_sumsq = (shifted**2).rolling(window).sum().sum(axis=1, skipna=False)
    mean = pooled_sum / n
    variance = (pooled_sumsq - n * mean**2) / (n - 1)
    std = variance.clip(lower=0) ** 0.5
    return (wide[target] - mean) / std


def _spatial_ears_for_one_unit(wide: pd.DataFrame, target: str, neighbor_cols: list, baseline_window: int) -> pd.DataFrame:
    c1 = _pooled_rolling_c_stat(wide, target, neighbor_cols, baseline_window, gap=0)
    c2 = _pooled_rolling_c_stat(wide, target, neighbor_cols, baseline_window, gap=2)
    c2_excess = (c2 - 1).clip(lower=0)
    c3 = c2_excess + c2_excess.shift(1) + c2_excess.shift(2)
    return pd.DataFrame({"c1": c1, "c2": c2, "c3": c3})


def spatial_ears_scores(
    panel: pd.DataFrame,
    time_col: str,
    group_col: str,
    value_col: str,
    lon_col: str = "lon",
    lat_col: str = "lat",
    baseline_window: int = 7,
    k: int = 5,
) -> pd.DataFrame:
    """Spatially-smoothed variant of `ears_scores`: each unit's C1/C2/C3 baseline
    (mean/std) pools its own `baseline_window` lagged values together with its
    `k` nearest spatial neighbors' values over that same lagged window, instead
    of relying on the unit alone. The current value being tested is still the
    unit's own -- only the *baseline* is widened spatially.

    This is a different kind of "spatial" than `getis_ord_hotspots`/
    `province_hotspots`: those compare a unit's *current* value to its
    neighbors' *current* values (spatial standout, one time period). This
    compares a unit's current value to a *history* built from itself and its
    neighbors (a more stable temporal baseline). It directly targets
    `ears_scores`'s sparse-data weak spot: a unit whose own history is too
    thin for a meaningful variance (std=0 -> c1/c2 = +-inf) usually still gets
    a workable baseline once neighbors' history is pooled in -- useful at
    finer grains (district, subdistrict) where a single unit's own counts are
    often mostly zero.

    IMPORTANT assumption: pooling treats the unit and its k neighbors as
    exchangeable -- drawn from the same underlying rate. That's often true
    for adjacent same-grain units (neighboring subdistricts tend to share a
    background rate), but if a unit's own history sits at a genuinely
    different *level* than its neighbors (e.g. a small, quiet subdistrict
    next to a large, busy one), pooling doesn't just lose specificity, it can
    invert the result: a real spike that merely reaches the neighbors'
    everyday level gets averaged away (masked, no alert), while the unit's
    own ordinary periods can look artificially low against the inflated
    pooled mean (spurious low-side signal). It only helps when the unit is
    sparse (own history alone is unusable) *and* its neighbors sit at a
    comparable rate -- it does not fix a sparse unit surrounded by
    differently-scaled neighbors. When in doubt, cross-check against plain
    `ears_scores`/`province_ears` rather than trusting the pooled result alone.

    `panel` has the same requirement as `ears_scores`: a complete (time x
    group) panel, zero-filled for every unit in every period. Every unit must
    also carry a static `lon_col`/`lat_col` (its centroid) for the k-NN
    lookup -- `province_spatial_ears`/`district_spatial_ears`/
    `subdistrict_spatial_ears` in `level_hotspots.py` build this
    automatically and are the intended entry point. `k` must be smaller than
    the number of distinct units in `panel`. Neighbors are unweighted
    (equal contribution regardless of distance within the k-NN set), not
    distance-decayed.
    """
    units = panel.drop_duplicates(subset=[group_col])[[group_col, lon_col, lat_col]].reset_index(drop=True)
    if k >= len(units):
        raise ValueError(f"k={k} neighbors requested but panel only has {len(units)} distinct {group_col!r} units")

    w = knn_weights(units, k=k, lon_col=lon_col, lat_col=lat_col)
    codes = units[group_col].to_numpy()
    neighbor_codes = {codes[i]: [codes[j] for j in w.neighbors[i]] for i in range(len(codes))}

    wide = panel.pivot(index=time_col, columns=group_col, values=value_col).sort_index()

    frames = []
    for code in codes:
        stats = _spatial_ears_for_one_unit(wide, code, neighbor_codes[code], baseline_window)
        stats[group_col] = code
        stats[time_col] = wide.index.to_numpy()
        frames.append(stats)
    stats_long = pd.concat(frames, ignore_index=True)

    out = panel.merge(stats_long, on=[group_col, time_col], how="left")
    out = out.sort_values([group_col, time_col]).reset_index(drop=True)
    out["c1_alert"] = out["c1"] > C1_ALERT_THRESHOLD
    out["c2_alert"] = out["c2"] > C2_ALERT_THRESHOLD
    out["c3_alert"] = out["c3"] > C3_ALERT_THRESHOLD
    return out
