import numpy as np
import pandas as pd
import pytest

from spatialdetection import points_from_dataframe, spatiotemporal_hotspots, time_bin_label

rng = np.random.default_rng(0)


def _grid_with_hotspot(timestamp: str) -> pd.DataFrame:
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    df = pd.DataFrame({"lon": xs.ravel().astype(float), "lat": ys.ravel().astype(float)})
    df["value"] = rng.normal(loc=10.0, scale=1.0, size=len(df))
    corner = (df["lon"] <= 1) & (df["lat"] <= 1)
    df.loc[corner, "value"] = 100.0
    df["timestamp"] = timestamp
    return df


def _three_day_df():
    return pd.concat(
        [_grid_with_hotspot("2024-01-01"), _grid_with_hotspot("2024-01-02"), _grid_with_hotspot("2024-01-03")],
        ignore_index=True,
    )


def _three_day_gdf():
    return points_from_dataframe(_three_day_df())


def test_time_bin_label_day():
    ts = pd.Series(["2024-03-04 08:00", "2024-03-04 20:00", "2024-03-05 00:00"])
    labels = time_bin_label(ts, "day")
    assert list(labels) == ["2024-03-04", "2024-03-04", "2024-03-05"]


def test_time_bin_label_week():
    ts = pd.Series(["2024-03-04", "2024-03-10"])  # both inside ISO week 10
    labels = time_bin_label(ts, "week")
    assert labels.iloc[0] == labels.iloc[1]


def test_time_bin_label_month():
    ts = pd.Series(["2024-03-04", "2024-03-30", "2024-04-01"])
    labels = time_bin_label(ts, "month")
    assert labels.iloc[0] == labels.iloc[1]
    assert labels.iloc[2] != labels.iloc[0]


def test_spatiotemporal_hotspots_per_day():
    gdf = _three_day_gdf()
    result = spatiotemporal_hotspots(gdf, time_col="timestamp", value_col="value", timeframe="day", k=4, permutations=99)

    assert set(result["time_bin"]) == {"2024-01-01", "2024-01-02", "2024-01-03"}
    for bin_label in result["time_bin"].unique():
        bin_result = result[result["time_bin"] == bin_label]
        corner = (bin_result["lon"] <= 1) & (bin_result["lat"] <= 1)
        assert (bin_result.loc[corner, "hotspot"] == 1).any()


def test_spatiotemporal_hotspots_skips_sparse_bins_with_warning():
    sparse = pd.DataFrame(
        {"lon": [0.0, 0.1], "lat": [0.0, 0.1], "value": [1.0, 2.0], "timestamp": ["2024-02-01"] * 2}
    )
    days = pd.concat(
        [_grid_with_hotspot("2024-01-01"), _grid_with_hotspot("2024-01-02"), _grid_with_hotspot("2024-01-03"), sparse],
        ignore_index=True,
    )
    gdf = points_from_dataframe(days)

    with pytest.warns(UserWarning, match="2024-02-01"):
        result = spatiotemporal_hotspots(gdf, time_col="timestamp", value_col="value", timeframe="day", k=4, permutations=99)

    assert "2024-02-01" not in set(result["time_bin"])


def test_spatiotemporal_hotspots_accepts_plain_dataframe():
    df = _three_day_df()  # plain DataFrame, no geometry built yet
    result = spatiotemporal_hotspots(df, time_col="timestamp", value_col="value", timeframe="day", k=4, permutations=99)

    assert set(result["time_bin"]) == {"2024-01-01", "2024-01-02", "2024-01-03"}
