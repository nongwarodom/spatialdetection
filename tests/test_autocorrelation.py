import numpy as np
import pandas as pd
import pytest

from spatialdetection import getis_ord_hotspots, morans_i, points_from_dataframe

rng = np.random.default_rng(0)


def _grid_with_hotspot_df():
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    df = pd.DataFrame({"lon": xs.ravel().astype(float), "lat": ys.ravel().astype(float)})
    df["value"] = rng.normal(loc=10.0, scale=1.0, size=len(df))
    # Inject a tight high-value hotspot in one corner.
    corner = (df["lon"] <= 1) & (df["lat"] <= 1)
    df.loc[corner, "value"] = 100.0
    return df


def _grid_with_hotspot():
    return points_from_dataframe(_grid_with_hotspot_df())


def test_morans_i_runs_and_returns_bounded_statistic():
    gdf = _grid_with_hotspot()
    result = morans_i(gdf, "value", k=4, permutations=99)
    assert -1.0 <= result.I <= 1.0
    assert 0.0 <= result.p_sim <= 1.0


def test_getis_ord_flags_injected_hotspot():
    gdf = _grid_with_hotspot()
    result = getis_ord_hotspots(gdf, "value", k=4, permutations=99)
    corner = (result["lon"] <= 1) & (result["lat"] <= 1)
    assert (result.loc[corner, "hotspot"] == 1).any()


def test_morans_i_rejects_missing_values():
    gdf = _grid_with_hotspot()
    gdf.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="missing value"):
        morans_i(gdf, "value", k=4, permutations=99)


def test_getis_ord_rejects_missing_values():
    gdf = _grid_with_hotspot()
    gdf.loc[0, "value"] = np.nan
    with pytest.raises(ValueError, match="missing value"):
        getis_ord_hotspots(gdf, "value", k=4, permutations=99)


def test_morans_i_accepts_plain_dataframe():
    df = _grid_with_hotspot_df()  # plain DataFrame, no geometry built yet
    result = morans_i(df, "value", k=4, permutations=99)
    assert -1.0 <= result.I <= 1.0


def test_getis_ord_accepts_plain_dataframe():
    df = _grid_with_hotspot_df()
    result = getis_ord_hotspots(df, "value", k=4, permutations=99)
    corner = (result["lon"] <= 1) & (result["lat"] <= 1)
    assert (result.loc[corner, "hotspot"] == 1).any()
