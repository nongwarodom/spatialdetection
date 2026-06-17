"""Build a JSON file of Thai administrative centroids (province/district/subdistrict/village).

Sources (see data/raw/SOURCES.md for provenance):
  - ADM1/ADM2/ADM3 boundary polygons: UN OCHA Thailand COD-AB, Royal Thai Survey
    Department, redistributed via github.com/prasertcbs/thailand_gis. These carry
    official P-codes (ADM1_PCODE/ADM2_PCODE/ADM3_PCODE) and are treated as authoritative.
  - Villages: TH_VILLAGE2012.shp (same repo), a 2012 point dataset with no official
    P-code. Villages are name-joined to their official subdistrict P-code; the join
    is best-effort and match rate is reported, not assumed.

Centroids are computed in a projected CRS (EPSG:32647, UTM zone 47N) so that area-based
centroid math isn't distorted by lat/lon degrees. If the geometric centroid falls
outside its own polygon (common for concave or multi-island shapes), the point is
replaced with `representative_point()`, which is always guaranteed to lie inside.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path

import geopandas as gpd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
OUT = Path(__file__).resolve().parent.parent / "data" / "thailand_admin_centroids.json"
PROJECTED_CRS = "EPSG:32647"  # UTM 47N, standard for Thailand-wide work


def _normalize(name: str | float | None) -> str:
    if not isinstance(name, str):
        return ""
    return unicodedata.normalize("NFC", name).strip()


def _safe_points_in_4326(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """Centroid (or representative_point fallback) computed in a projected CRS, returned in EPSG:4326."""
    proj = gdf.to_crs(PROJECTED_CRS)
    proj["geometry"] = proj.geometry.make_valid()
    centroids = proj.geometry.centroid
    outside = ~proj.geometry.contains(centroids)
    if outside.any():
        centroids.loc[outside] = proj.geometry.loc[outside].representative_point()
    return gpd.GeoSeries(centroids, crs=PROJECTED_CRS).to_crs("EPSG:4326")


def build_provinces() -> list[dict]:
    gdf = gpd.read_file(RAW / "adm1_province" / "tha_adm1_province.shp", encoding="utf-8")
    points = _safe_points_in_4326(gdf)
    return [
        {
            "province_code": row.ADM1_PCODE,
            "province_en": row.ADM1_EN,
            "province_th": _normalize(row.ADM1_TH),
            "lat": pt.y,
            "lon": pt.x,
        }
        for row, pt in zip(gdf.itertuples(), points)
    ]


def build_districts() -> list[dict]:
    gdf = gpd.read_file(RAW / "adm2_district" / "tha_admbnda_adm2.shp", encoding="utf-8")
    points = _safe_points_in_4326(gdf)
    return [
        {
            "district_code": row.ADM2_PCODE,
            "district_en": row.ADM2_EN,
            "district_th": _normalize(row.ADM2_TH),
            "province_code": row.ADM1_PCODE,
            "lat": pt.y,
            "lon": pt.x,
        }
        for row, pt in zip(gdf.itertuples(), points)
    ]


def build_subdistricts() -> tuple[list[dict], gpd.GeoDataFrame]:
    gdf = gpd.read_file(
        RAW / "adm3_subdistrict" / "tha_admbnda_adm3_rtsd_20220121.shp", encoding="utf-8"
    )
    points = _safe_points_in_4326(gdf)
    records = [
        {
            "subdistrict_code": row.ADM3_PCODE,
            "subdistrict_en": row.ADM3_EN,
            "subdistrict_th": _normalize(row.ADM3_TH),
            "district_code": row.ADM2_PCODE,
            "province_code": row.ADM1_PCODE,
            "lat": pt.y,
            "lon": pt.x,
        }
        for row, pt in zip(gdf.itertuples(), points)
    ]
    return records, gdf


def build_villages(adm3_gdf: gpd.GeoDataFrame) -> list[dict]:
    gdf = gpd.read_file(RAW / "village_2012" / "TH_VILLAGE2012.shp", encoding="utf-8")

    # Best-effort name join to the official subdistrict P-code. The source has no
    # official P-code or moo (หมู่ที่) number of its own. District names need two
    # fallbacks: (1) the village source abbreviates every "Mueang <Province>"
    # district to just "เมือง", and (2) several districts that were a "กิ่งอำเภอ"
    # (minor district) in 2012 have since been promoted to full อำเภอ status and
    # renamed without that prefix -- so a 3-way match is tried first, falling back
    # to a 2-way (province, subdistrict) match where that pair is unique in adm3.
    cols = ["ADM3_PCODE", "ADM2_PCODE", "ADM1_PCODE"]
    full_key = adm3_gdf.assign(
        _k=(
            adm3_gdf["ADM1_TH"].map(_normalize)
            + "|"
            + adm3_gdf["ADM2_TH"].map(_normalize)
            + "|"
            + adm3_gdf["ADM3_TH"].map(_normalize)
        )
    ).set_index("_k")[cols]

    prov_sub_counts = adm3_gdf.assign(
        _k2=adm3_gdf["ADM1_TH"].map(_normalize) + "|" + adm3_gdf["ADM3_TH"].map(_normalize)
    )["_k2"].value_counts()
    unique_prov_sub = prov_sub_counts[prov_sub_counts == 1].index
    fallback_key = (
        adm3_gdf.assign(
            _k2=adm3_gdf["ADM1_TH"].map(_normalize) + "|" + adm3_gdf["ADM3_TH"].map(_normalize)
        )
        .set_index("_k2")[cols]
        .loc[unique_prov_sub]
    )

    def amp_variant(prv: str, amp: str) -> str:
        return f"เมือง{prv}" if amp == "เมือง" else amp

    gdf = gdf.assign(
        _k=[
            norm_p + "|" + amp_variant(norm_p, norm_a) + "|" + norm_t
            for norm_p, norm_a, norm_t in zip(
                gdf["PRV_NAME"].map(_normalize),
                gdf["AMP_NAME"].map(_normalize),
                gdf["TAM_NAME"].map(_normalize),
            )
        ],
        _k2=gdf["PRV_NAME"].map(_normalize) + "|" + gdf["TAM_NAME"].map(_normalize),
    )
    joined = gdf.join(full_key, on="_k")
    still_missing = joined["ADM3_PCODE"].isna()
    fallback = joined.loc[still_missing, "_k2"].map(
        lambda k: fallback_key.loc[k] if k in fallback_key.index else None
    )
    for col in cols:
        joined.loc[still_missing, col] = [
            v[col] if v is not None else None for v in fallback
        ]

    matched = joined["ADM3_PCODE"].notna()
    print(f"Village name-join match rate: {matched.sum()}/{len(joined)} ({matched.mean():.1%})")

    records = []
    for row in joined.itertuples():
        pt = row.geometry
        records.append(
            {
                "village_name_th": _normalize(row.NAME),
                "source_village_id": row.MAIN_ID,  # source-internal ID, NOT an official moo number
                "subdistrict_code": row.ADM3_PCODE if isinstance(row.ADM3_PCODE, str) else None,
                "district_code": row.ADM2_PCODE if isinstance(row.ADM2_PCODE, str) else None,
                "province_code": row.ADM1_PCODE if isinstance(row.ADM1_PCODE, str) else None,
                "lat": pt.y,
                "lon": pt.x,
            }
        )
    return records


def main() -> None:
    provinces = build_provinces()
    districts = build_districts()
    subdistricts, adm3_gdf = build_subdistricts()
    villages = build_villages(adm3_gdf)

    output = {
        "metadata": {
            "description": "Thailand administrative centroids: province, district, subdistrict, village",
            "address_code_standard": (
                "UN OCHA / COD-AB P-codes (ADM1_PCODE/ADM2_PCODE/ADM3_PCODE) for "
                "province/district/subdistrict, sourced from Royal Thai Survey "
                "Department boundaries. Villages have NO official P-code or moo "
                "(หมู่ที่) number in the source data; "
                "they carry the source dataset's internal 'source_village_id' and a "
                "best-effort name-joined parent subdistrict_code instead."
            ),
            "sources": {
                "province_district_subdistrict": (
                    "OCHA Thailand COD-AB (Royal Thai Survey Department), redistributed via "
                    "github.com/prasertcbs/thailand_gis; subdistrict layer dated 2022-01-21"
                ),
                "village": (
                    "TH_VILLAGE2012.shp via github.com/prasertcbs/thailand_gis -- 2012 vintage, "
                    "point geometries, no official P-code"
                ),
            },
            "known_gaps": {
                "village_coverage": (
                    f"{len(villages)} villages present, covering "
                    f"{len({v['subdistrict_code'] for v in villages if v['subdistrict_code']})} "
                    f"of {len(subdistricts)} subdistricts. Official current village count is "
                    "~75,000+; this 2012 snapshot is missing villages created/split since then."
                ),
                "centroid_method": (
                    "Polygon centroids computed in EPSG:32647 (UTM 47N), falling back to "
                    "representative_point() when the geometric centroid fell outside the "
                    "polygon. Village points are the source dataset's own point locations, "
                    "not computed centroids."
                ),
            },
            "counts": {
                "provinces": len(provinces),
                "districts": len(districts),
                "subdistricts": len(subdistricts),
                "villages": len(villages),
            },
        },
        "provinces": provinces,
        "districts": districts,
        "subdistricts": subdistricts,
        "villages": villages,
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=None, separators=(",", ":"))

    print(f"Wrote {OUT} ({OUT.stat().st_size / 1_048_576:.1f} MB)")
    print(output["metadata"]["counts"])


if __name__ == "__main__":
    main()
