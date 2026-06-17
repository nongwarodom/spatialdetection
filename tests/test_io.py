from pathlib import Path

import pandas as pd

from spatialdetection import load_points


def test_load_points_accepts_str_path(tmp_path):
    csv_path = tmp_path / "points.csv"
    pd.DataFrame({"lon": [100.5, 100.6], "lat": [13.7, 13.8]}).to_csv(csv_path, index=False)

    gdf = load_points(str(csv_path))

    assert len(gdf) == 2


def test_load_points_accepts_pathlib_path(tmp_path):
    csv_path: Path = tmp_path / "points.csv"
    pd.DataFrame({"lon": [100.5, 100.6], "lat": [13.7, 13.8]}).to_csv(csv_path, index=False)

    gdf = load_points(csv_path)

    assert len(gdf) == 2
