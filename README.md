# spatialdetection

A toolkit for general-purpose spatial clustering and anomaly/hotspot detection
on point data: DBSCAN-based density clustering and spatial autocorrelation
statistics (global Moran's I, local Getis-Ord Gi*).

## Setup

Requires [uv](https://docs.astral.sh/uv/). The project pins Python 3.11 for
geospatial dependency compatibility.

```bash
uv sync
```

Try the end-to-end example (no GeoDataFrame setup needed — see Usage below):

```bash
uv run python examples/quickstart.py
```

For a larger, P-code-centric example (40k+ synthetic person records across
multiple provinces and 30 days, rolled up to subdistrict/district/province):

```bash
uv run python examples/pcode_example.py
```

### Notebook / IDE

`notebooks/quickstart.ipynb` is the same tour as `examples/quickstart.py`,
cell-by-cell with inline plots. To run it:

```bash
uv run python -m ipykernel install --user --name spatialdetection --display-name "spatialdetection"
uv run jupyter lab notebooks/quickstart.ipynb
```

Or open the project folder in VS Code with the Jupyter extension, open the
notebook, and select the **spatialdetection** kernel (registered by the
command above) before running cells. The same kernel also works for ad hoc
`.py` files run via the Jupyter extension's "Run Cell"/interactive window.

## Usage

Every clustering/autocorrelation function takes a plain `pandas.DataFrame`
with `lon`/`lat` columns directly — no need to call `points_from_dataframe`
yourself first. Pass `lon_col`/`lat_col` if your columns are named
differently; pass an already-built `GeoDataFrame` instead and it's used as-is.

```python
import pandas as pd
from spatialdetection import (
    dbscan_clusters,
    cluster_summary,
    morans_i,
    getis_ord_hotspots,
    detect_province,
    detect_district,
    detect_subdistrict,
    detect_point,
    detect_level,
    plot_level_map,
    time_bin_label,
    spatiotemporal_hotspots,
    province_hotspots,
    district_hotspots,
    subdistrict_hotspots,
)

# df is a plain DataFrame with "lon"/"lat" columns (e.g. from pd.read_csv)

# Density-based cluster detection (eps in kilometers)
labels = dbscan_clusters(df, eps_km=1.0, min_samples=5)
summary = cluster_summary(df, labels)

# Spatial autocorrelation / hotspot detection on a value column
moran = morans_i(df, value_col="value")
hotspots = getis_ord_hotspots(df, value_col="value")

# Look up a Thai admin P-code -> location, or auto-dispatch with detect_level
# on a mix of P-codes and (lat, lon) pairs.
province = detect_province("TH10")
district = detect_district("TH1001")
subdistrict = detect_subdistrict("TH100101")
auto = detect_level("TH100101")  # -> same as detect_subdistrict here

# Reverse-geocode: location -> P-code. Takes a DataFrame of (lat, lon) rows
# and spatially resolves each one to its containing province/district/
# subdistrict in a single batch (points outside Thailand get null codes).
located = detect_point(pd.DataFrame({"lat": [13.7563], "lon": [100.5018]}))

# Auto-plot a map zoomed to whatever level detect_level finds
ax = plot_level_map("TH100101")
ax = plot_level_map((13.7563, 100.5018))

# Spatiotemporal hotspot detection: bin points by day/week/month and run
# Getis-Ord Gi* independently within each bin, so a hotspot in one period
# isn't diluted by activity in another.
hotspots_by_week = spatiotemporal_hotspots(
    df, time_col="observed_at", value_col="value", timeframe="week"
)

# Admin-level hotspot detection: reverse-geocode point data onto every
# province/district/subdistrict (zero-count units included, not dropped),
# then run Getis-Ord Gi* on the aggregated counts (or value_col sums).
hot_provinces = province_hotspots(df)
hot_districts = district_hotspots(df)
hot_subdistricts = subdistrict_hotspots(df)  # needs dense data -- see caveat below
```

See `examples/quickstart.py` for a runnable version of this with synthetic
data (DBSCAN, Moran's I, Getis-Ord Gi*, spatiotemporal hotspots, and an
auto-plotted map, all from one plain DataFrame).

## Modules

- `spatialdetection.io` — load points from CSV or vector files into a `GeoDataFrame`.
- `spatialdetection.clustering` — DBSCAN clustering with a haversine metric over lon/lat.
- `spatialdetection.autocorrelation` — Moran's I and Getis-Ord Gi* hotspot/coldspot detection.
- `spatialdetection.detect` — `detect_province`/`detect_district`/
  `detect_subdistrict` resolve a Thai P-code to its location (code ->
  location); `detect_point` is the inverse, reverse-geocoding a DataFrame of
  (lat, lon) rows to their containing province/district/subdistrict
  (location -> code) via a spatial join. `detect_level` auto-dispatches a
  single P-code string or (lat, lon) pair. P-codes are nested strings
  (subdistrict `"TH100101"` = district `"TH1001"` + 2 digits = province
  `"TH10"` + 2 more), so if you already have a subdistrict P-code per row,
  `code[:6]`/`code[:4]` gets you the parent district/province code directly
  — no lookup needed (see `examples/pcode_example.py`).
- `spatialdetection.plotting` — `plot_level_map`, auto-zoomed/styled to
  whatever level `detect_level` finds for its input.
- `spatialdetection.spatiotemporal` — `time_bin_label` (day/week/month
  bin labels for a timestamp column) and `spatiotemporal_hotspots`
  (Getis-Ord Gi* run independently per time bin).
- `spatialdetection.level_hotspots` — `province_hotspots`/`district_hotspots`/
  `subdistrict_hotspots` take point-level (lat, lon) data, reverse-geocode it
  with `detect_point`, aggregate onto *every* unit at that level (so
  zero-count units are included, not dropped — required for correct
  Getis-Ord neighborhood z-scores), and run `getis_ord_hotspots` on the
  result. Finer levels (district, especially subdistrict) need denser point
  data: with mostly-zero counts spread across thousands of units, the
  permutation test's reference distribution degenerates and p-values stop
  being meaningful — prefer `province_hotspots`/`district_hotspots` unless
  your data supports the finer grain.

## Development

```bash
uv run pytest
uv run ruff check .
```

## Reference data: Thailand administrative centroids

`data/thailand_admin_centroids.json` has lat/lon centroids and address codes for
all Thai provinces (77), districts (928), subdistricts (7,425), and villages
(65,213). Generated by `scripts/build_thailand_admin_centroids.py`.

Province/district/subdistrict carry official UN OCHA/COD-AB P-codes and are
considered authoritative. **Villages do not** — the only open village-level
dataset available is a 2012 point snapshot with no official P-code or moo
(หมู่ที่) number; villages here carry that source's internal ID plus a
best-effort name-joined parent `subdistrict_code` (96.2% match rate). See the
`metadata` block in the JSON file for full provenance and known gaps (notably:
65,213 villages vs. Thailand's current ~75,000+, and 2012-era naming).

To regenerate: place the source shapefiles per `data/raw/SOURCES.md`, then run
`uv run python scripts/build_thailand_admin_centroids.py`.
