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

For a larger, P-code-centric example (80k+ synthetic person records across
two diseases, each with 30k+ cases, multiple provinces, and 30 days, rolled
up to subdistrict/district/province and auto-plotted per disease per level):

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
    plot_hotspots,
    time_bin_label,
    spatiotemporal_hotspots,
    province_hotspots,
    district_hotspots,
    subdistrict_hotspots,
    province_ears,
    district_ears,
    subdistrict_ears,
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

# Customize the highlight color, and label each unit with its name
# (mainly useful for health_zone, which highlights several provinces
# at once -- see below). label_color sets the label text's color.
ax = plot_level_map(
    health_zone=1, color="green", show_labels=True, label_fontsize=6, label_color="white"
)

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

# No lat/lon? If your data already carries a P-code per row, pcode_col
# skips the spatial join entirely -- a finer code rolls up to the level
# you call automatically (subdistrict code -> province_hotspots works).
# province_col/district_col/subdistrict_col are identical synonyms, named
# for whichever reads clearest with your column (pass only one).
hot_provinces = province_hotspots(df, pcode_col="subdistrict_code")
hot_provinces = province_hotspots(df, subdistrict_col="subdistrict_code")  # same thing

# Auto-plot any of the above: a choropleth of the matching admin boundaries
# (or a point scatter for raw getis_ord_hotspots/spatiotemporal_hotspots
# output), colored by gi_zscore on a diverging scale centered at zero.
ax = plot_hotspots(hot_provinces)

# Zoom into one region instead of the whole country: pass at most one of
# health_zone (1-13, MoPH's province groupings), province, or district
# (P-code). Restricts the map to that region and zooms to its bounds. A
# result can only be filtered at its own grain or coarser -- e.g. a
# district_hotspots result can be filtered by district or province, but a
# province_hotspots result can't be filtered by district.
ax = plot_hotspots(hot_provinces, health_zone=1)   # zone 1: 8 northern provinces
ax = plot_hotspots(hot_districts, province="TH10")  # Bangkok's districts only

# cmap picks the colormap (any matplotlib name); show_labels annotates
# each plotted unit with its name (choropleth results only); label_color
# sets the label text's color.
ax = plot_hotspots(hot_provinces, cmap="viridis", show_labels=True, label_fontsize=6, label_color="black")

# Plot the discrete hotspot/coldspot/not-significant flag instead of the
# continuous gi_zscore: cmap is ignored for value_col="hotspot" in favor of
# three named colors (with a matching legend), each independently adjustable.
ax = plot_hotspots(hot_provinces, value_col="hotspot", hotspot_color="red", coldspot_color="blue")

# Getis-Ord Gi* (above) only compares a unit to its spatial neighbors within
# one time period -- a province that's high everywhere around it, yet far
# above its OWN recent weeks, won't show up there. province_ears/
# district_ears/subdistrict_ears are the temporal counterpart: same
# aggregation and zero-filling as *_hotspots, but bin by time_col first and
# compare each unit's count each period against its own history (CDC's EARS
# C1/C2/C3 historical-limits algorithms) instead of its neighbors. No k/
# permutations/alpha -- there's no spatial weights matrix here.
weekly_province_ears = province_ears(df, time_col="observed_at", timeframe="week")
alerts = weekly_province_ears[weekly_province_ears["c2_alert"]]  # C2 is EARS's default algorithm

# c1/c2/c3 are NaN for a unit's first baseline_window (+2 for c2/c3) periods,
# before it has enough history to compute a baseline. c2/c3 can be +-inf when
# the baseline is perfectly flat (zero variance) -- this is standard EARS
# behavior, not a bug, but it means EARS on sparse, mostly-zero counts (e.g.
# subdistrict-level or early-outbreak province data) produces alert-flooded,
# uninformative output, same caveat as the finer *_hotspots levels below.
# Prefer coarser levels or a longer baseline_window for thin data.

# province_ears stacks every time bin's rows into one result, like
# spatiotemporal_hotspots does -- filter to a single time_bin before
# plotting a readable map.
latest_bin = weekly_province_ears["time_bin"].max()
ax = plot_hotspots(weekly_province_ears[weekly_province_ears["time_bin"] == latest_bin], value_col="c2")
```

See `examples/quickstart.py` for a runnable version of this with synthetic
data (DBSCAN, Moran's I, Getis-Ord Gi*, spatiotemporal hotspots, EARS
temporal anomaly detection at day/week/month grain, and an auto-plotted
map, all from one plain DataFrame).

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
  whatever level `detect_level` finds for its input; `plot_hotspots`,
  auto-plotted straight from a `getis_ord_hotspots`/`spatiotemporal_hotspots`/
  `province_hotspots`/`district_hotspots`/`subdistrict_hotspots` result
  (choropleth for admin-level results, point scatter otherwise), colored by
  `gi_zscore` rather than the `hotspot` flag since that flag can be
  unreliable under skewed counts (see below). Takes an optional
  `health_zone`/`province`/`district` filter (pass at most one) to restrict
  the map to one region and zoom to it, an optional `cmap` to change the
  colormap, and `show_labels`/`label_fontsize`/`label_color` to annotate
  each plotted unit with its name. Plotting `value_col="hotspot"` (the
  discrete flag) switches to categorical coloring via `hotspot_color`/
  `coldspot_color`/`not_significant_color` instead of `cmap`, with a
  matching legend. `plot_level_map` similarly takes `color` (the
  highlighted unit's fill color) and `show_labels`/`label_fontsize`/
  `label_color`.
- `spatialdetection.health_zones` — `HEALTH_ZONE_PROVINCES` maps Thailand
  Ministry of Public Health's 13 health zones (เขตสุขภาพที่ 1-13, each a
  group of provinces; zone 13 is Bangkok alone) to their province names;
  `health_zone_province_codes` resolves a zone number to its province
  P-codes.
- `spatialdetection.spatiotemporal` — `time_bin_label` (day/week/month
  bin labels for a timestamp column) and `spatiotemporal_hotspots`
  (Getis-Ord Gi* run independently per time bin).
- `spatialdetection.level_hotspots` — `province_hotspots`/`district_hotspots`/
  `subdistrict_hotspots` aggregate onto *every* unit at that level (so
  zero-count units are included, not dropped — required for correct
  Getis-Ord neighborhood z-scores), then run `getis_ord_hotspots` on the
  result. Two input modes: point-level (lat, lon) data, reverse-geocoded
  with `detect_point`; or `pcode_col` (or its identical, level-named
  synonyms `province_col`/`district_col`/`subdistrict_col` — pass only
  one), if your data already carries a P-code per row — skips the spatial
  join entirely, and a finer code (e.g. subdistrict) rolls up to whatever
  level you call via string slicing, no lat/lon needed at all. Finer
  levels (district, especially subdistrict) need denser data: with
  mostly-zero counts spread across thousands of units, the permutation
  test's reference distribution degenerates and p-values stop being
  meaningful — prefer `province_hotspots`/`district_hotspots` unless your
  data supports the finer grain.
- `spatialdetection.temporal` — `ears_scores`, the purely temporal
  counterpart to Getis-Ord Gi*: implements the CDC's EARS C1/C2/C3
  historical-limits algorithms (Hutwagner et al. 2003), comparing each
  unit's value only to its own past values, independent of its neighbors.
  C1's baseline is the `baseline_window` periods immediately before the
  current one (most reactive, but a growing outbreak leaks into its own
  baseline); C2's baseline ends 2 periods earlier, guarding against that
  self-contamination (EARS's standard/default algorithm); C3 sums each of
  the current and prior 2 periods' C2 excess over 1, so it only fires on a
  *sustained* rise across 3 consecutive periods (least sensitive, most
  specific). All three are z-score-like and can be `±inf` when a unit's
  baseline has zero variance — expected EARS behavior, not a bug, but a
  sign the input is too sparse for a meaningful baseline. Takes a
  zero-filled (time × group) panel with no gaps — `province_ears`/
  `district_ears`/`subdistrict_ears` in `level_hotspots` build one
  automatically and are the intended entry point; call `ears_scores`
  directly only if you already have a custom panel.
- `spatialdetection.level_hotspots` also has `province_ears`/
  `district_ears`/`subdistrict_ears`: same point-level/`pcode_col`
  aggregation and zero-filling as `province_hotspots`/`district_hotspots`/
  `subdistrict_hotspots`, but binned by `time_col`/`timeframe`
  (`"day"`/`"week"`/`"month"`, via `spatiotemporal.time_bin_label`) and run
  through `ears_scores` instead of `getis_ord_hotspots` — no `k`/
  `permutations`/`alpha`, since there's no spatial weights matrix to build.
  Output plugs directly into `plot_hotspots` (e.g. `value_col="c2"`). Same
  sparse-data caveat as the finer `*_hotspots` levels: EARS on mostly-zero
  counts alerts on noise (every 0→1 change looks like `c2=inf`), so prefer
  coarser levels or a longer `baseline_window` unless the data is dense.

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
