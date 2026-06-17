import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from spatialdetection import district_hotspots, province_hotspots, subdistrict_hotspots

rng = np.random.default_rng(0)

_CENTROIDS_PATH = Path(__file__).resolve().parent.parent / "data" / "thailand_admin_centroids.json"


def _provinces() -> pd.DataFrame:
    with _CENTROIDS_PATH.open(encoding="utf-8") as f:
        return pd.DataFrame(json.load(f)["provinces"])


def _case_points_with_outbreak(outbreak_province_en: str = "Chiang Mai") -> pd.DataFrame:
    provinces = _provinces()
    background_codes = ["TH11", "TH12", "TH13", "TH14", "TH15", "TH16", "TH17", "TH18", "TH19"]
    outbreak_code = provinces.loc[provinces["province_en"] == outbreak_province_en, "province_code"].iloc[0]
    chosen = provinces[provinces["province_code"].isin([*background_codes, outbreak_code])]

    points = []
    for _, p in chosen.iterrows():
        n = 60 if p["province_code"] == outbreak_code else rng.integers(2, 8)
        points.append(
            pd.DataFrame(
                {
                    "lon": rng.normal(p["lon"], 0.02, size=n),
                    "lat": rng.normal(p["lat"], 0.02, size=n),
                }
            )
        )
    return pd.concat(points, ignore_index=True)


def test_province_hotspots_flags_injected_outbreak():
    case_points = _case_points_with_outbreak()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        result = province_hotspots(case_points, k=5, permutations=199)

    assert len(result) == 77  # every province present, not just ones with cases
    hot = result[result["hotspot"] == 1]
    assert "Chiang Mai" in set(hot["province_en"])


def test_province_hotspots_counts_points_by_default():
    case_points = _case_points_with_outbreak()
    result = province_hotspots(case_points, k=5, permutations=49)

    chiang_mai = result[result["province_en"] == "Chiang Mai"].iloc[0]
    assert chiang_mai["count"] == 60


def test_province_hotspots_sums_value_col_when_given():
    case_points = _case_points_with_outbreak()
    case_points["severity"] = 2  # each point worth 2 instead of 1

    result = province_hotspots(case_points, value_col="severity", k=5, permutations=49)

    chiang_mai = result[result["province_en"] == "Chiang Mai"].iloc[0]
    assert chiang_mai["severity"] == 120


def test_province_hotspots_accepts_custom_column_names():
    case_points = _case_points_with_outbreak().rename(columns={"lon": "x", "lat": "y"})
    result = province_hotspots(case_points, lon_col="x", lat_col="y", k=5, permutations=49)
    assert len(result) == 77


def test_district_hotspots_returns_one_row_per_district():
    case_points = _case_points_with_outbreak()
    result = district_hotspots(case_points, k=5, permutations=49)
    assert len(result) == 928


def test_subdistrict_hotspots_returns_one_row_per_subdistrict():
    case_points = _case_points_with_outbreak()
    result = subdistrict_hotspots(case_points, k=5, permutations=49)
    assert len(result) == 7425


def test_province_hotspots_drops_points_outside_thailand():
    case_points = _case_points_with_outbreak()
    outside = pd.DataFrame({"lon": [0.0], "lat": [0.0]})  # mid-Atlantic
    df = pd.concat([case_points, outside], ignore_index=True)

    result = province_hotspots(df, k=5, permutations=49)

    assert result["count"].sum() == len(case_points)  # the unmatched point isn't silently counted somewhere
