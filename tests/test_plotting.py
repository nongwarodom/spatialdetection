from spatialdetection import plot_level_map


def test_plot_level_map_province():
    ax = plot_level_map("TH10")
    assert "Bangkok" in ax.get_title()


def test_plot_level_map_district():
    ax = plot_level_map("TH1001")
    assert "District" in ax.get_title()


def test_plot_level_map_point():
    ax = plot_level_map((13.7563, 100.5018))
    assert "Point" in ax.get_title()
