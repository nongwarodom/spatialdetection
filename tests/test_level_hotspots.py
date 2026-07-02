import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from spatialdetection import district_ears, district_hotspots, province_ears, province_hotspots, subdistrict_hotspots

rng = np.random.default_rng(0)

_CENTROIDS_PATH = Path(__file__).resolve().parent.parent / "data" / "thailand_admin_centroids.json"


def _provinces() -> pd.DataFrame:
    with _CENTROIDS_PATH.open(encoding="utf-8") as f:
        return pd.DataFrame(json.load(f)["provinces"])


def _subdistricts() -> pd.DataFrame:
    with _CENTROIDS_PATH.open(encoding="utf-8") as f:
        return pd.DataFrame(json.load(f)["subdistricts"])


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


def _case_pcodes_with_outbreak(outbreak_province_en: str = "Chiang Mai") -> pd.DataFrame:
    """Same shape as _case_points_with_outbreak, but pcode-only -- no lat/lon."""
    subdistricts = _subdistricts()
    provinces = _provinces()
    by_province_en = subdistricts.merge(provinces[["province_code", "province_en"]], on="province_code")

    background_en = ["Bangkok", "Nonthaburi", "Pathum Thani", "Samut Prakan", "Nakhon Pathom"]
    rows = []
    for province_en in background_en:
        code = by_province_en[by_province_en["province_en"] == province_en]["subdistrict_code"].iloc[0]
        rows += [code] * int(rng.integers(2, 8))
    outbreak_code = by_province_en[by_province_en["province_en"] == outbreak_province_en]["subdistrict_code"].iloc[0]
    rows += [outbreak_code] * 60
    return pd.DataFrame({"pcode": rows})


def test_province_hotspots_accepts_pcode_col_no_lat_lon():
    df = _case_pcodes_with_outbreak()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = province_hotspots(df, pcode_col="pcode", k=5, permutations=199)

    assert len(result) == 77
    chiang_mai = result[result["province_en"] == "Chiang Mai"].iloc[0]
    assert chiang_mai["count"] == 60


def test_pcode_col_rolls_up_finer_code_to_requested_level():
    subdistricts = _subdistricts()
    provinces = _provinces()
    by_province_en = subdistricts.merge(provinces[["province_code", "province_en"]], on="province_code")
    chiang_mai_subdistrict = by_province_en[by_province_en["province_en"] == "Chiang Mai"]["subdistrict_code"].iloc[0]
    bangkok_subdistrict = by_province_en[by_province_en["province_en"] == "Bangkok"]["subdistrict_code"].iloc[0]
    df = pd.DataFrame({"pcode": [chiang_mai_subdistrict] * 60 + [bangkok_subdistrict] * 5})

    result = province_hotspots(df, pcode_col="pcode", k=5, permutations=49)

    chiang_mai = result[result["province_en"] == "Chiang Mai"].iloc[0]
    assert chiang_mai["count"] == 60


def test_pcode_col_rejects_coarser_code_than_requested_level():
    df = pd.DataFrame({"pcode": ["TH10", "TH50"]})  # province-grained, too coarse for subdistrict
    with pytest.raises(ValueError, match="shorter than a subdistrict"):
        subdistrict_hotspots(df, pcode_col="pcode")


def test_pcode_col_sums_value_col_when_given():
    df = _case_pcodes_with_outbreak()
    df["severity"] = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = province_hotspots(df, value_col="severity", pcode_col="pcode", k=5, permutations=49)

    chiang_mai = result[result["province_en"] == "Chiang Mai"].iloc[0]
    assert chiang_mai["severity"] == 120


def test_subdistrict_col_is_a_synonym_for_pcode_col():
    df = _case_pcodes_with_outbreak().rename(columns={"pcode": "subdistrict_code"})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = province_hotspots(df, subdistrict_col="subdistrict_code", k=5, permutations=49)

    chiang_mai = result[result["province_en"] == "Chiang Mai"].iloc[0]
    assert chiang_mai["count"] == 60


def test_district_col_rolls_up_to_province():
    df = pd.DataFrame({"district_code": ["TH1001"] * 5 + ["TH5705"] * 60})  # Bangkok, Chiang Rai districts

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = province_hotspots(df, district_col="district_code", k=5, permutations=49)

    chiang_rai = result[result["province_en"] == "Chiang Rai"].iloc[0]
    assert chiang_rai["count"] == 60


def test_province_col_works_directly_at_province_level():
    df = pd.DataFrame({"province_code": ["TH10"] * 5 + ["TH57"] * 60})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        result = province_hotspots(df, province_col="province_code", k=5, permutations=49)

    chiang_rai = result[result["province_en"] == "Chiang Rai"].iloc[0]
    assert chiang_rai["count"] == 60


def test_multiple_code_col_synonyms_at_once_raises():
    df = pd.DataFrame({"pcode": ["TH10"]})
    with pytest.raises(ValueError, match="only one of"):
        province_hotspots(df, pcode_col="pcode", subdistrict_col="pcode")


def _weekly_baseline_with_late_spike(spike_province_en: str = "Chiang Mai", n_weeks: int = 10) -> pd.DataFrame:
    """Steady per-province background case volume across every week, then an
    extra burst in `spike_province_en` only in the final week -- for EARS
    tests, where the signal to detect is a rise vs. a province's OWN history,
    not (necessarily) a spatial standout against its neighbors."""
    provinces = _provinces()
    background_codes = ["TH11", "TH12", "TH13", "TH14", "TH15", "TH16", "TH17", "TH18", "TH19"]
    spike_code = provinces.loc[provinces["province_en"] == spike_province_en, "province_code"].iloc[0]
    chosen = provinces[provinces["province_code"].isin([*background_codes, spike_code])]

    weeks = pd.to_datetime("2024-01-01") + pd.to_timedelta(np.arange(n_weeks) * 7, unit="D")
    frames = []
    for week in weeks:
        for _, p in chosen.iterrows():
            n = int(rng.integers(3, 6))
            frames.append(
                pd.DataFrame(
                    {
                        "lon": rng.normal(p["lon"], 0.02, size=n),
                        "lat": rng.normal(p["lat"], 0.02, size=n),
                        "reported_at": week,
                    }
                )
            )
    spike_row = provinces.loc[provinces["province_code"] == spike_code].iloc[0]
    frames.append(
        pd.DataFrame(
            {
                "lon": rng.normal(spike_row["lon"], 0.02, size=60),
                "lat": rng.normal(spike_row["lat"], 0.02, size=60),
                "reported_at": weeks[-1],
            }
        )
    )
    return pd.concat(frames, ignore_index=True)


def test_province_ears_flags_a_late_spike_against_its_own_history():
    df = _weekly_baseline_with_late_spike()
    result = province_ears(df, time_col="reported_at", timeframe="week", baseline_window=7)

    chiang_mai = result[result["province_en"] == "Chiang Mai"].sort_values("time_bin")
    assert chiang_mai["c2_alert"].iloc[-1]  # only the final (spike) week alerts
    assert not chiang_mai["c2_alert"].iloc[:-1].any()


def test_province_ears_does_not_require_k_or_permutations():
    # Purely temporal -- no spatial weights matrix, unlike province_hotspots.
    df = _weekly_baseline_with_late_spike(n_weeks=3)
    result = province_ears(df, time_col="reported_at", timeframe="week", baseline_window=7)
    assert {"c1", "c2", "c3", "c1_alert", "c2_alert", "c3_alert"} <= set(result.columns)


def test_province_ears_zero_fills_every_province_in_every_week():
    df = _weekly_baseline_with_late_spike(n_weeks=3)
    result = province_ears(df, time_col="reported_at", timeframe="week", baseline_window=7)
    assert len(result) == 77 * 3  # every province, every week -- no gaps in any unit's series


def test_district_ears_returns_one_row_per_district_per_week():
    df = _weekly_baseline_with_late_spike(n_weeks=3)
    result = district_ears(df, time_col="reported_at", timeframe="week", baseline_window=7)
    assert len(result) == 928 * 3
