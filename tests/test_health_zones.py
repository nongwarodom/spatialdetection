import pytest

from spatialdetection import HEALTH_ZONE_PROVINCES, health_zone_province_codes
from spatialdetection.detect import _centroids


def test_health_zones_cover_every_province_exactly_once():
    all_provinces = {p["province_en"] for p in _centroids()["provinces"]}
    listed = [name for provinces in HEALTH_ZONE_PROVINCES.values() for name in provinces]

    assert len(listed) == len(all_provinces) == 77
    assert len(listed) == len(set(listed))  # no province listed twice
    assert set(listed) == all_provinces  # no typo'd/missing province name


def test_health_zone_province_codes_returns_p_codes():
    codes = health_zone_province_codes(13)  # Bangkok-only zone

    assert codes == ["TH10"]


def test_health_zone_province_codes_rejects_out_of_range_zone():
    with pytest.raises(ValueError, match="valid health zone"):
        health_zone_province_codes(14)
