"""Using P-codes directly, aggregating multi-disease data, and auto-plotting anomalies.

Builds a synthetic person-level dataset (pcode, lat, long, date, disease)
spread across multiple provinces over a 30-day window, with at least 30,000
cases per disease. Each row already carries its subdistrict P-code (as real
line-list data often does), which demonstrates:

1. Using P-codes directly: detect_subdistrict/detect_district/
   detect_province resolve a P-code to its name and centroid. Thailand's
   P-code scheme is nested (subdistrict "TH100101" = district "TH1001" +
   2 digits = province "TH10" + 2 more), so the parent district/province
   codes can be derived from a subdistrict P-code by string slicing -- no
   lookup needed -- which is what lets one column roll up to all three
   admin levels.
2. Aggregating that rolled-up pcode data directly with a groupby is the
   same work province_hotspots/district_hotspots/subdistrict_hotspots do
   internally from lat/long via detect_point's spatial join -- which is
   exactly what pcode_col does for you: pass the column you already have
   and skip the spatial join (and the lat/long columns) entirely.
3. plot_hotspots auto-plots a choropleth straight from each hotspot
   function's output, per disease per level.

Run with:
    uv run python examples/pcode_example.py
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: this script saves PNGs instead of opening windows

import numpy as np
import pandas as pd

from spatialdetection import (
    detect_district,
    detect_province,
    detect_subdistrict,
    district_hotspots,
    plot_hotspots,
    province_hotspots,
    subdistrict_hotspots,
)

rng = np.random.default_rng(0)

MIN_PERSONS_PER_DISEASE = 30_000
DAYS = 30
START_DATE = "2024-05-01"
OUTPUT_DIR = Path(__file__).resolve().parent / "output"

# 7 background provinces spread across Thailand's regions, 2 subdistricts each
BACKGROUND_PROVINCES_EN = [
    "Bangkok",
    "Chiang Mai",
    "Khon Kaen",
    "Nakhon Si Thammarat",
    "Songkhla",
    "Nakhon Ratchasima",
    "Chon Buri",
]
SUBDISTRICTS_PER_PROVINCE = 2

# Each disease gets its own outbreak province, not in the background list
# above, so the resulting hotspot maps genuinely differ by disease.
DISEASE_OUTBREAK_PROVINCE_EN = {
    "Dengue": "Chiang Rai",
    "Influenza": "Surat Thani",
}
OUTBREAK_PERSONS = 8_000


def _jittered_persons(subdistrict_row: pd.Series, n: int, start: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pcode": subdistrict_row["subdistrict_code"],
            "lat": rng.normal(subdistrict_row["lat"], 0.01, size=n),
            "long": rng.normal(subdistrict_row["lon"], 0.01, size=n),
            "date": start + pd.to_timedelta(rng.integers(0, DAYS, size=n), unit="D"),
        }
    )


def make_disease_data(outbreak_province_en: str) -> pd.DataFrame:
    """Person-level records for one disease: one row per case, tagged with subdistrict pcode.

    Every province gets a small baseline count, not just the ones below --
    Getis-Ord's permutation test handles "0 vs. 0" neighbor comparisons
    poorly (degenerate reference distribution -> unreliable p-values, see
    level_hotspots.py), so leaving most provinces at a literal zero count
    would make the outbreak signal below get drowned out by that artifact
    rather than detected on its own merits.
    """
    with open("data/thailand_admin_centroids.json") as f:
        centroids = json.load(f)
    subdistricts = pd.DataFrame(centroids["subdistricts"])
    provinces = pd.DataFrame(centroids["provinces"])
    by_province_en = subdistricts.merge(provinces[["province_code", "province_en"]], on="province_code")

    start = pd.to_datetime(START_DATE)
    rows = []

    baseline = subdistricts.groupby("province_code", as_index=False).first()
    for _, s in baseline.iterrows():
        rows.append(_jittered_persons(s, rng.integers(20, 80), start))

    chosen = [
        by_province_en[by_province_en["province_en"] == p].sample(SUBDISTRICTS_PER_PROVINCE, random_state=0)
        for p in BACKGROUND_PROVINCES_EN
    ]
    for _, s in pd.concat(chosen, ignore_index=True).iterrows():
        rows.append(_jittered_persons(s, rng.integers(1500, 2500), start))

    outbreak = by_province_en[by_province_en["province_en"] == outbreak_province_en].sample(1, random_state=1)
    for _, s in outbreak.iterrows():
        rows.append(_jittered_persons(s, OUTBREAK_PERSONS, start))

    df = pd.concat(rows, ignore_index=True)
    assert len(df) >= MIN_PERSONS_PER_DISEASE, f"only {len(df)} rows, need >= {MIN_PERSONS_PER_DISEASE}"
    return df


def make_all_disease_data() -> pd.DataFrame:
    frames = []
    for disease, outbreak_province_en in DISEASE_OUTBREAK_PROVINCE_EN.items():
        disease_df = make_disease_data(outbreak_province_en)
        disease_df["disease"] = disease
        frames.append(disease_df)
    return pd.concat(frames, ignore_index=True)


def compare_pcode_col_vs_latlon(disease_df: pd.DataFrame, disease: str) -> None:
    """pcode_col skips detect_point's spatial join entirely -- confirm it agrees with the lat/long path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        via_pcode = province_hotspots(disease_df, pcode_col="pcode", k=5, permutations=49)
        via_latlon = province_hotspots(disease_df, lon_col="long", lat_col="lat", k=5, permutations=49)

    compare = via_pcode[["province_code", "count"]].merge(
        via_latlon[["province_code", "count"]], on="province_code", suffixes=("_pcode_col", "_latlon")
    )
    matches = int((compare["count_pcode_col"] == compare["count_latlon"]).sum())
    print(
        f"  pcode_col vs lat/long aggregation for {disease}: {matches}/{len(compare)} provinces match exactly "
        f"(any gap is expected: a jittered point can land just outside its source subdistrict's polygon)"
    )


def examine_and_detect_one_disease(disease_df: pd.DataFrame, disease: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # libpysal disconnected-components notice
        results = {
            "province": province_hotspots(disease_df, lon_col="long", lat_col="lat", k=5, permutations=499),
            "district": district_hotspots(disease_df, lon_col="long", lat_col="lat", k=5, permutations=199),
            "subdistrict": subdistrict_hotspots(disease_df, lon_col="long", lat_col="lat", k=5, permutations=99),
        }

    name_col = {"province": "province_en", "district": "district_en", "subdistrict": "subdistrict_en"}
    for level, result in results.items():
        # Ranked by count (not filtered to hotspot==1): at this skew -- one
        # dominant outbreak unit among many small/baseline ones -- the
        # gi_pvalue significance flag is unreliable (see level_hotspots.py),
        # so filtering to "significant" can hide the actual outbreak.
        top = result.sort_values("count", ascending=False).head(3)
        print(f"  Top {level}(s) by case count:")
        print(top[[name_col[level], "count", "gi_zscore", "gi_pvalue", "hotspot"]].to_string(index=False, header=False).replace("\n", "\n    "))

        # Auto-plot straight from the hotspot result, colored by gi_zscore
        # (continuous, not the unreliable hotspot flag -- see plot_hotspots).
        ax = plot_hotspots(result)
        ax.set_title(f"{disease}: {ax.get_title()}")
        out_path = OUTPUT_DIR / f"{disease.lower()}_{level}_hotspots.png"
        ax.figure.savefig(out_path, dpi=100)
        print(f"    -> saved {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    df = make_all_disease_data()

    print(f"{len(df)} person records across {df['disease'].nunique()} diseases and {df['date'].dt.date.nunique()} days")
    print("Examine data: cases per disease")
    print(df.groupby("disease").size().rename("persons").to_string(), "\n")

    # 1. Using P-codes directly: look up a sample pcode's name/centroid
    sample_pcode = df["pcode"].iloc[0]
    sub = detect_subdistrict(sample_pcode)
    print(f"detect_subdistrict({sample_pcode!r}) -> {sub.record['subdistrict_en']}, lat={sub.lat:.4f}, lon={sub.lon:.4f}")

    # Thailand's P-code scheme is nested: derive parent district/province
    # codes from the subdistrict code by string slicing -- no lookup needed.
    district_code = sample_pcode[:6]
    province_code = sample_pcode[:4]
    print(f"  -> district {district_code} = {detect_district(district_code).record['district_en']}")
    print(f"  -> province {province_code} = {detect_province(province_code).record['province_en']}\n")

    # 2. Roll the whole dataset up to district/province pcode the same way,
    # then aggregate directly with groupby -- no spatial join needed since
    # the subdistrict pcode is already known per row.
    df["district_code"] = df["pcode"].str[:6]
    df["province_code"] = df["pcode"].str[:4]
    by_subdistrict = df.groupby("pcode").size()
    by_district = df.groupby("district_code").size()
    by_province = df.groupby("province_code").size()
    print(f"Aggregated directly from pcode: {len(by_subdistrict)} subdistricts, "
          f"{len(by_district)} districts, {len(by_province)} provinces\n")

    # 3. pcode_col does that same rollup-and-aggregate internally -- call
    # province_hotspots/district_hotspots/subdistrict_hotspots directly on
    # the pcode column. Drop lat/long here to prove they're not needed at all.
    pcode_only = df[["pcode"]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        prov_via_pcode = province_hotspots(pcode_only, pcode_col="pcode", k=5, permutations=99)
    print("province_hotspots(df, pcode_col='pcode') -- no lat/long column present at all:")
    print(prov_via_pcode.sort_values("count", ascending=False)[["province_en", "count"]].head(5).to_string(index=False), "\n")

    # province_col/district_col/subdistrict_col are identical synonyms for
    # pcode_col, just named for whichever level matches your column -- pick
    # whichever reads clearest. subdistrict_col here needs no rollup at all
    # (the pcode column is already subdistrict-grained); district_col rolls
    # district_code up to subdistrict_hotspots's parent level, province.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        sub_via_subdistrict_col = subdistrict_hotspots(df[["pcode"]], subdistrict_col="pcode", k=5, permutations=49)
        prov_via_district_col = province_hotspots(df[["district_code"]], district_col="district_code", k=5, permutations=49)
    print("subdistrict_hotspots(df, subdistrict_col='pcode'):")
    print(sub_via_subdistrict_col.sort_values("count", ascending=False)[["subdistrict_en", "count"]].head(3).to_string(index=False))
    print("\nprovince_hotspots(df, district_col='district_code'):")
    print(prov_via_district_col.sort_values("count", ascending=False)[["province_en", "count"]].head(3).to_string(index=False), "\n")

    # 4. Per disease: cross-check pcode_col against lat/long, then detect
    # anomalies at every level and auto-plot each result.
    for disease in DISEASE_OUTBREAK_PROVINCE_EN:
        print(f"=== {disease} ===")
        compare_pcode_col_vs_latlon(df[df["disease"] == disease], disease)
        examine_and_detect_one_disease(df[df["disease"] == disease], disease)
        print()


if __name__ == "__main__":
    main()
