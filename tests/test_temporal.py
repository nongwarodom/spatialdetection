import numpy as np
import pandas as pd
import pytest

from spatialdetection.temporal import (
    C1_ALERT_THRESHOLD,
    C2_ALERT_THRESHOLD,
    C3_ALERT_THRESHOLD,
    ears_scores,
    spatial_ears_scores,
)


def _panel(values: list[float], unit: str = "A") -> pd.DataFrame:
    return pd.DataFrame({"unit": [unit] * len(values), "t": range(len(values)), "v": values})


def _line_panel(values_by_unit: dict) -> pd.DataFrame:
    """Four units on a line at lon=0,1,2,3 (lat=0) -- with k=1, A's only
    possible neighbor is B and D's only possible neighbor is C, so k-NN
    results are unambiguous for tests. `values_by_unit` maps unit name to
    its per-period value list (all lists must be the same length)."""
    coords = {"A": (0.0, 0.0), "B": (1.0, 0.0), "C": (2.0, 0.0), "D": (3.0, 0.0)}
    frames = []
    for unit, values in values_by_unit.items():
        lon, lat = coords[unit]
        frames.append(
            pd.DataFrame({"unit": unit, "t": range(len(values)), "v": values, "lon": lon, "lat": lat})
        )
    return pd.concat(frames, ignore_index=True)


def test_ears_scores_c1_matches_hand_computed_value():
    # 7-day baseline (mean=10, sample std=sqrt(2)), then a value equal to the mean (c1=0).
    values = [8, 12, 9, 11, 10, 9, 11, 10]
    result = ears_scores(_panel(values), time_col="t", group_col="unit", value_col="v", baseline_window=7)

    baseline = np.array(values[:7])
    expected_c1 = (values[7] - baseline.mean()) / baseline.std(ddof=1)
    assert result.iloc[7]["c1"] == pytest.approx(expected_c1)


def test_ears_scores_c2_uses_two_period_guard_band():
    # Same baseline as above, but c2's window ends 2 periods earlier (indices 0-6
    # for a value at index 9, skipping indices 7 and 8).
    values = [8, 12, 9, 11, 10, 9, 11, 999, 999, 40]
    result = ears_scores(_panel(values), time_col="t", group_col="unit", value_col="v", baseline_window=7)

    baseline = np.array(values[:7])
    expected_c2 = (values[9] - baseline.mean()) / baseline.std(ddof=1)
    assert result.iloc[9]["c2"] == pytest.approx(expected_c2)
    # c1's baseline *would* include the two 999 spikes, so it's a different (much smaller) value.
    assert result.iloc[9]["c1"] != pytest.approx(expected_c2)


def test_ears_scores_flags_nothing_before_enough_baseline_history():
    values = [10] * 6 + [1000]  # spike arrives before a full 7-period baseline exists
    result = ears_scores(_panel(values), time_col="t", group_col="unit", value_col="v", baseline_window=7)

    assert result["c1"].isna().all()
    assert not result["c1_alert"].any()


def test_ears_scores_flags_a_genuine_spike():
    values = [8, 12, 9, 11, 10, 9, 11, 10, 10, 50]
    result = ears_scores(_panel(values), time_col="t", group_col="unit", value_col="v", baseline_window=7)

    assert result.iloc[9]["c1"] > C1_ALERT_THRESHOLD
    assert result.iloc[9]["c1_alert"]
    assert not result.iloc[:9]["c1_alert"].any()


def test_ears_scores_c3_requires_sustained_elevation():
    # A finite-variance baseline (mean=10, std=sqrt(2)) so c2 is a real number,
    # not +-inf -- otherwise this test never exercises actual C3 arithmetic. A
    # single mild bump's c2-excess (~1.12) stays under C3's threshold on its
    # own; three consecutive mild bumps accumulate past it.
    baseline = [8, 12, 9, 11, 10, 9, 11]
    single_bump = baseline + [10, 10, 13, 10, 10, 10]
    sustained = baseline + [10, 10, 13, 13, 13, 10]

    single_result = ears_scores(_panel(single_bump), "t", "unit", "v", baseline_window=7)
    sustained_result = ears_scores(_panel(sustained), "t", "unit", "v", baseline_window=7)

    assert not single_result["c3_alert"].any()
    assert sustained_result.iloc[11]["c3"] > C3_ALERT_THRESHOLD
    assert sustained_result.iloc[11]["c3_alert"]


def test_ears_scores_zero_variance_baseline_alerts_on_any_change():
    # A perfectly flat baseline has std=0; any different value divides by zero,
    # giving +-inf rather than raising -- this is standard EARS behavior, not a bug.
    values = [10] * 7 + [11]
    result = ears_scores(_panel(values), time_col="t", group_col="unit", value_col="v", baseline_window=7)

    assert result.iloc[7]["c1"] == float("inf")
    assert result.iloc[7]["c1_alert"]


def test_ears_scores_groups_are_independent():
    # Unit B's spike must not leak into unit A's statistics.
    a = pd.DataFrame({"unit": "A", "t": range(8), "v": [8, 12, 9, 11, 10, 9, 11, 10]})
    b = pd.DataFrame({"unit": "B", "t": range(8), "v": [10, 10, 10, 10, 10, 10, 10, 999]})
    panel = pd.concat([a, b], ignore_index=True)

    result = ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=7)

    a_last = result[(result["unit"] == "A") & (result["t"] == 7)].iloc[0]
    b_last = result[(result["unit"] == "B") & (result["t"] == 7)].iloc[0]
    assert a_last["c1"] == pytest.approx(0.0)
    assert not a_last["c1_alert"]
    assert b_last["c1_alert"]


def test_ears_scores_alert_thresholds_are_the_documented_ears_defaults():
    assert C1_ALERT_THRESHOLD == 3.0
    assert C2_ALERT_THRESHOLD == 3.0
    assert C3_ALERT_THRESHOLD == 2.0


def test_spatial_ears_scores_c1_matches_hand_computed_pooled_baseline():
    # A's own history is flat (5); its only k=1 neighbor is B, which varies.
    # The pooled baseline should be A's 7 lagged 5s plus B's 7 lagged values.
    a_values = [5] * 9
    b_values = [12, 10, 11, 12, 10, 11, 12, 10, 11]
    panel = _line_panel({"A": a_values, "B": b_values, "C": [8, 9] * 5, "D": [9, 8] * 5})

    result = spatial_ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=7, k=1)
    a_row = result[(result["unit"] == "A") & (result["t"] == 8)].iloc[0]

    pooled = np.array(a_values[1:8] + b_values[1:8])  # gap=0 -> shift(1), window=7 -> indices 1..7
    expected_c1 = (a_values[8] - pooled.mean()) / pooled.std(ddof=1)
    assert a_row["c1"] == pytest.approx(expected_c1)


def test_spatial_ears_scores_avoids_inf_that_plain_ears_scores_gives():
    # A's own history is perfectly flat (std=0) -- plain ears_scores hits the
    # zero-variance +-inf case on any spike. Pooling in a varying neighbor's
    # history should give a finite, real baseline std instead.
    a_values = [5] * 8 + [20]
    b_values = [10, 11, 12, 10, 11, 12, 10, 11, 12]
    panel = _line_panel({"A": a_values, "B": b_values, "C": [8, 9] * 5, "D": [9, 8] * 5})

    plain = ears_scores(panel[panel["unit"] == "A"], time_col="t", group_col="unit", value_col="v", baseline_window=7)
    assert plain.iloc[8]["c1"] == float("inf")

    pooled = spatial_ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=7, k=1)
    a_last = pooled[(pooled["unit"] == "A") & (pooled["t"] == 8)].iloc[0]
    assert np.isfinite(a_last["c1"])
    assert a_last["c1"] > C1_ALERT_THRESHOLD  # still a genuine spike, just no longer +-inf


def test_spatial_ears_scores_still_infinite_when_every_pooled_unit_is_flat():
    # If A and its only neighbor B are both flat, pooling doesn't manufacture
    # variance that isn't there -- still +-inf on a spike, same as plain EARS.
    a_values = [5] * 8 + [20]
    b_values = [5] * 9
    panel = _line_panel({"A": a_values, "B": b_values, "C": [8, 9] * 5, "D": [9, 8] * 5})

    result = spatial_ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=7, k=1)
    a_last = result[(result["unit"] == "A") & (result["t"] == 8)].iloc[0]
    assert a_last["c1"] == float("inf")


def test_spatial_ears_scores_flags_nothing_before_enough_baseline_history():
    values = {u: [10] * 6 + [1000] for u in "ABCD"}
    panel = _line_panel(values)

    result = spatial_ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=7, k=1)
    assert result["c1"].isna().all()
    assert not result["c1_alert"].any()


def test_spatial_ears_scores_raises_when_k_too_large():
    panel = _line_panel({u: [5, 6, 7] for u in "ABCD"})
    with pytest.raises(ValueError, match="k=4"):
        spatial_ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=2, k=4)


def test_spatial_ears_scores_numerator_is_still_the_unit_s_own_current_value():
    # Pooling widens the baseline, not the value being compared -- two units
    # with the same history but different final values should get different
    # c1 (not both averaged into the same pooled comparison value).
    panel = _line_panel(
        {
            "A": [10, 11, 9, 10, 11, 9, 10, 50],
            "B": [10, 11, 9, 10, 11, 9, 10, 10],
            "C": [8, 9] * 4,
            "D": [9, 8] * 4,
        }
    )
    result = spatial_ears_scores(panel, time_col="t", group_col="unit", value_col="v", baseline_window=7, k=1)
    a_last = result[(result["unit"] == "A") & (result["t"] == 7)].iloc[0]
    b_last = result[(result["unit"] == "B") & (result["t"] == 7)].iloc[0]
    assert a_last["c1"] > C1_ALERT_THRESHOLD
    assert b_last["c1"] < C1_ALERT_THRESHOLD
