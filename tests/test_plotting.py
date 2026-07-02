import numpy as np
import pandas as pd
import pytest

from spatialdetection import (
    detect_point,
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


def test_plot_level_map_health_zone():
    ax = plot_level_map(health_zone=1)  # 8 northern provinces
    assert "Health zone 1" in ax.get_title()
    assert len(ax.collections[1].get_paths()) == 8  # collections[0] is the whole-country background


def test_plot_level_map_province_keyword_matches_value():
    assert plot_level_map(province="TH10").get_title() == plot_level_map("TH10").get_title()


def test_plot_level_map_district_keyword_matches_value():
    assert plot_level_map(district="TH1001").get_title() == plot_level_map("TH1001").get_title()


def test_plot_level_map_requires_exactly_one_selector():
    with pytest.raises(ValueError, match="pass exactly one of"):
        plot_level_map()

    with pytest.raises(ValueError, match="pass exactly one of"):
        plot_level_map("TH10", province="TH11")


def test_plot_level_map_color_recolors_highlighted_unit():
    ax = plot_level_map("TH10", color="green")

    unit_facecolor = ax.collections[-1].get_facecolor()[0]
    assert tuple(unit_facecolor) == pytest.approx((0.0, 0.501960784313725, 0.0, 1.0))


def test_plot_level_map_labels_off_by_default():
    ax = plot_level_map(health_zone=1)
    assert len(ax.texts) == 0


def test_plot_level_map_show_labels_annotates_each_unit():
    ax = plot_level_map(health_zone=1, show_labels=True, label_fontsize=6)

    assert len(ax.texts) == 8  # one label per province in the zone
    assert ax.texts[0].get_fontsize() == 6
    assert {t.get_text() for t in ax.texts} == {
        "Chiang Mai",
        "Chiang Rai",
        "Phrae",
        "Nan",
        "Phayao",
        "Lampang",
        "Lamphun",
        "Mae Hong Son",
    }


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


def _nationwide_df(n=200):
    return pd.DataFrame({"lon": rng.uniform(97, 105, n), "lat": rng.uniform(6, 20, n)})


def test_plot_hotspots_province_filtered_by_health_zone_zooms_and_restricts():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)
    full_ax = plot_hotspots(result)

    ax = plot_hotspots(result, health_zone=1)  # 8 northern provinces

    assert len(ax.collections[0].get_paths()) == 8
    full_xmin, full_xmax = full_ax.get_xlim()
    xmin, xmax = ax.get_xlim()
    assert (xmax - xmin) < (full_xmax - full_xmin)


def test_plot_hotspots_province_filtered_by_province_code():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)

    ax = plot_hotspots(result, province="TH10")  # Bangkok only

    assert len(ax.collections[0].get_paths()) == 1


def test_plot_hotspots_district_level_filtered_by_province():
    result = district_hotspots(_nationwide_df(), k=5, permutations=49)

    ax = plot_hotspots(result, province="TH10")  # Bangkok's districts only

    n_bangkok_districts = 50
    assert len(ax.collections[0].get_paths()) == n_bangkok_districts


def test_plot_hotspots_point_level_filtered_by_province():
    xs, ys = np.meshgrid(np.linspace(100.3, 100.7, 8), np.linspace(13.6, 13.9, 8))
    df = pd.DataFrame({"lon": xs.ravel(), "lat": ys.ravel()})
    df["value"] = rng.normal(10.0, 1.0, size=len(df))
    gdf = points_from_dataframe(df)
    result = getis_ord_hotspots(gdf, "value", k=4, permutations=49)
    expected_n = int((detect_point(gdf)["province_code"] == "TH10").sum())
    assert 0 < expected_n < len(df)  # sanity check: the grid straddles Bangkok's border

    ax = plot_hotspots(result, province="TH10")

    assert len(ax.collections[0].get_offsets()) == expected_n


def test_plot_hotspots_region_filters_are_mutually_exclusive():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)

    with pytest.raises(ValueError, match="only one of health_zone/province/district"):
        plot_hotspots(result, province="TH10", district="TH1001")


def test_plot_hotspots_district_filter_too_fine_for_province_level_result_raises():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)

    with pytest.raises(ValueError, match="coarser than a district filter"):
        plot_hotspots(result, district="TH1001")


def test_plot_hotspots_cmap_changes_colormap():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)

    ax = plot_hotspots(result, cmap="viridis")

    assert ax.collections[0].get_cmap().name == "viridis"


def test_plot_hotspots_labels_off_by_default():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)

    ax = plot_hotspots(result, health_zone=1)

    assert len(ax.texts) == 0


def test_plot_hotspots_show_labels_annotates_each_plotted_unit():
    result = province_hotspots(_nationwide_df(), k=5, permutations=49)

    ax = plot_hotspots(result, health_zone=1, show_labels=True, label_fontsize=5)

    assert len(ax.texts) == 8
    assert ax.texts[0].get_fontsize() == 5


def test_plot_hotspots_show_labels_warns_for_point_level_results():
    gdf = _grid_with_hotspot()
    result = getis_ord_hotspots(gdf, "value", k=4, permutations=99)

    with pytest.warns(UserWarning, match="show_labels has no effect"):
        ax = plot_hotspots(result, show_labels=True)

    assert len(ax.texts) == 0
