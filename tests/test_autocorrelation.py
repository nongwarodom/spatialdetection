import numpy as np
import pandas as pd

from spatialdetection import getis_ord_hotspots, morans_i, points_from_dataframe

rng = np.random.default_rng(0)


def _grid_with_hotspot():
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    df = pd.DataFrame({"lon": xs.ravel().astype(float), "lat": ys.ravel().astype(float)})
    df["value"] = rng.normal(loc=10.0, scale=1.0, size=len(df))
    # Inject a tight high-value hotspot in one corner.
    corner = (df["lon"] <= 1) & (df["lat"] <= 1)
    df.loc[corner, "value"] = 100.0
    return points_from_dataframe(df)


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
