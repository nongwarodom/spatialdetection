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
"""

from __future__ import annotations

import pandas as pd

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
