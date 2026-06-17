import numpy as np
import pandas as pd

from spatialdetection import (
    district_hotspots,
    getis_ord_hotspots,
    plot_hotspots,
    plot_level_map,
    points_from_dataframe,
    province_hotspots,
)

rng = np.random.default_rng(0)


def test_plot_level_map_province():
    ax = plot_level_map("TH10")
    assert "Bangkok" in ax.get_title()


def test_plot_level_map_district():
    ax = plot_level_map("TH1001")
    assert "District" in ax.get_title()


def test_plot_level_map_point():
    ax = plot_level_map((13.7563, 100.5018))
    assert "Point" in ax.get_title()


def _grid_with_hotspot():
    xs, ys = np.meshgrid(np.arange(10), np.arange(10))
    df = pd.DataFrame({"lon": xs.ravel().astype(float), "lat": ys.ravel().astype(float)})
    df["value"] = rng.normal(loc=10.0, scale=1.0, size=len(df))
    corner = (df["lon"] <= 1) & (df["lat"] <= 1)
    df.loc[corner, "value"] = 100.0
    return points_from_dataframe(df)


def test_plot_hotspots_point_level():
    gdf = _grid_with_hotspot()
    result = getis_ord_hotspots(gdf, "value", k=4, permutations=99)

    ax = plot_hotspots(result)

    assert "Point-level gi_zscore" in ax.get_title()


def test_plot_hotspots_province_level_choropleth():
    df = pd.DataFrame(
        {
            "lon": rng.normal(100.50, 0.01, size=20).tolist() + rng.uniform(98, 102, 20).tolist(),
            "lat": rng.normal(13.75, 0.01, size=20).tolist() + rng.uniform(13, 17, 20).tolist(),
        }
    )
    result = province_hotspots(df, k=5, permutations=49)

    ax = plot_hotspots(result)

    assert "Province-level gi_zscore" in ax.get_title()
    # choropleth renders polygons as a single PatchCollection, not point markers
    assert len(ax.collections) == 1
    assert len(ax.collections[0].get_paths()) >= 70  # one path per province


def test_plot_hotspots_district_level_uses_district_code_not_parent_province():
    df = pd.DataFrame({"lon": rng.uniform(98, 102, 30), "lat": rng.uniform(13, 17, 30)})
    result = district_hotspots(df, k=5, permutations=49)

    ax = plot_hotspots(result)

    assert "District-level gi_zscore" in ax.get_title()
