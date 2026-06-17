import pandas as pd
import pytest

from spatialdetection import (
    detect_district,
    detect_level,
    detect_point,
    detect_province,
    detect_subdistrict,
)


def test_detect_province():
    result = detect_province("TH10")
    assert result.level == "province"
    assert result.record["province_en"] == "Bangkok"


def test_detect_district():
    result = detect_district("th1001")  # lowercase + whitespace should normalize
    assert result.level == "district"
    assert result.code == "TH1001"


def test_detect_subdistrict():
    result = detect_subdistrict("TH100101")
    assert result.level == "subdistrict"


def test_detect_point_reverse_geocodes_known_location():
    # Centroid of subdistrict TH100101 (Phraborom Maharatchawang, Phra Nakhon, Bangkok).
    df = pd.DataFrame({"lat": [13.751466582507641], "lon": [100.49223438698446]})

    result = detect_point(df)

    assert result.iloc[0]["subdistrict_code"] == "TH100101"
    assert result.iloc[0]["district_code"] == "TH1001"
    assert result.iloc[0]["province_code"] == "TH10"
    assert result.iloc[0]["province_en"] == "Bangkok"


def test_detect_point_handles_batch_of_rows():
    df = pd.DataFrame(
        {
            "lat": [13.751466582507641, 13.7563],
            "lon": [100.49223438698446, 100.5018],
        }
    )

    result = detect_point(df)

    assert len(result) == 2
    assert result.iloc[0]["province_code"] == "TH10"
    assert result.iloc[1]["province_code"] == "TH10"


def test_detect_point_outside_thailand_is_null():
    df = pd.DataFrame({"lat": [0.0], "lon": [0.0]})  # middle of the Atlantic, not Thailand

    result = detect_point(df)

    assert pd.isna(result.iloc[0]["subdistrict_code"])
    assert pd.isna(result.iloc[0]["province_code"])


def test_detect_point_accepts_custom_column_names():
    df = pd.DataFrame({"y": [13.751466582507641], "x": [100.49223438698446]})

    result = detect_point(df, lat_col="y", lon_col="x")

    assert result.iloc[0]["subdistrict_code"] == "TH100101"


def test_detect_point_overwrites_colliding_input_column_with_warning():
    # gpd.sjoin would otherwise silently rename both copies to
    # province_code_left/province_code_right instead of erroring.
    df = pd.DataFrame(
        {
            "lat": [13.751466582507641],
            "lon": [100.49223438698446],
            "province_code": ["stale-value"],
        }
    )

    with pytest.warns(UserWarning, match="province_code"):
        result = detect_point(df)

    assert result.iloc[0]["province_code"] == "TH10"


def test_detect_province_rejects_wrong_format():
    with pytest.raises(ValueError):
        detect_province("TH1001")  # district-shaped code


def test_detect_unknown_code_raises():
    with pytest.raises(ValueError):
        detect_province("TH99")  # well-formed but not a real province


def test_detect_level_dispatches_by_code_length():
    assert detect_level("TH10").level == "province"
    assert detect_level("TH1001").level == "district"
    assert detect_level("TH100101").level == "subdistrict"


def test_detect_level_dispatches_lat_lon():
    result = detect_level((13.7563, 100.5018))
    assert result.level == "point"


def test_detect_level_invalid_code_raises():
    with pytest.raises(ValueError):
        detect_level("XX99")


def test_detect_province_rejects_non_str_type():
    with pytest.raises(TypeError):
        detect_province(1001)


def test_detect_level_rejects_none():
    with pytest.raises(ValueError):
        detect_level(None)


def test_detect_level_rejects_wrong_length_tuple():
    with pytest.raises(ValueError):
        detect_level((13.0, 100.0, 99.0))
