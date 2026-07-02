import numpy as np
import pandas as pd
import pytest

from spatialdetection.temporal import C1_ALERT_THRESHOLD, C2_ALERT_THRESHOLD, C3_ALERT_THRESHOLD, ears_scores


def _panel(values: list[float], unit: str = "A") -> pd.DataFrame:
    return pd.DataFrame({"unit": [unit] * len(values), "t": range(len(values)), "v": values})


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
