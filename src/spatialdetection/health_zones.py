"""Thailand Ministry of Public Health's 13 health zones (เขตสุขภาพที่ 1-13).

Each zone groups several provinces for health service planning/administration
(regional health offices rh1.moph.go.th .. rh12.moph.go.th); zone 13 is
Bangkok on its own. Provinces essentially never move between zones, so this
is hardcoded rather than derived from a shapefile like the rest of
`data/thailand_admin_centroids.json`.
"""

from __future__ import annotations

from functools import lru_cache

from spatialdetection.detect import _centroids

HEALTH_ZONE_PROVINCES: dict[int, list[str]] = {
    1: ["Chiang Mai", "Chiang Rai", "Phrae", "Nan", "Phayao", "Lampang", "Lamphun", "Mae Hong Son"],
    2: ["Tak", "Phetchabun", "Phitsanulok", "Uttaradit", "Sukhothai"],
    3: ["Kamphaeng Phet", "Phichit", "Nakhon Sawan", "Chai Nat", "Uthai Thani"],
    4: [
        "Saraburi",
        "Nonthaburi",
        "Lop Buri",
        "Ang Thong",
        "Nakhon Nayok",
        "Sing Buri",
        "Phra Nakhon Si Ayutthaya",
        "Pathum Thani",
    ],
    5: [
        "Phetchaburi",
        "Samut Sakhon",
        "Samut Songkhram",
        "Prachuap Khiri Khan",
        "Suphan Buri",
        "Nakhon Pathom",
        "Ratchaburi",
        "Kanchanaburi",
    ],
    6: ["Sa Kaeo", "Prachin Buri", "Chachoengsao", "Samut Prakan", "Chon Buri", "Chanthaburi", "Rayong", "Trat"],
    7: ["Kalasin", "Khon Kaen", "Maha Sarakham", "Roi Et"],
    8: ["Udon Thani", "Sakon Nakhon", "Nakhon Phanom", "Loei", "Nong Khai", "Nong Bua Lam Phu", "Bueng Kan"],
    9: ["Chaiyaphum", "Nakhon Ratchasima", "Buri Ram", "Surin"],
    10: ["Ubon Ratchathani", "Si Sa Ket", "Yasothon", "Amnat Charoen", "Mukdahan"],
    11: ["Nakhon Si Thammarat", "Surat Thani", "Phuket", "Krabi", "Phangnga", "Ranong", "Chumphon"],
    12: ["Songkhla", "Satun", "Trang", "Phatthalung", "Pattani", "Yala", "Narathiwat"],
    13: ["Bangkok"],
}


@lru_cache(maxsize=1)
def _province_name_to_code() -> dict[str, str]:
    return {p["province_en"]: p["province_code"] for p in _centroids()["provinces"]}


def health_zone_province_codes(zone: int) -> list[str]:
    """Return the province P-codes (e.g. "TH10") belonging to health `zone` (1-13)."""
    if zone not in HEALTH_ZONE_PROVINCES:
        raise ValueError(f"{zone!r} is not a valid health zone (expected an int 1-13)")
    name_to_code = _province_name_to_code()
    return [name_to_code[name] for name in HEALTH_ZONE_PROVINCES[zone]]
