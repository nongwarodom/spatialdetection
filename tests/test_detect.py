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


def test_detect_point():
    result = detect_point(13.7563, 100.5018)
    assert result.level == "point"
    assert result.lat == pytest.approx(13.7563)
    assert result.lon == pytest.approx(100.5018)
    assert result.code is None


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
