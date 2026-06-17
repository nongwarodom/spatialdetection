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

## Usage

```python
from spatialdetection import (
    points_from_dataframe,
    dbscan_clusters,
    cluster_summary,
    morans_i,
    getis_ord_hotspots,
)

gdf = points_from_dataframe(df, lon_col="lon", lat_col="lat")

# Density-based cluster detection (eps in kilometers)
labels = dbscan_clusters(gdf, eps_km=1.0, min_samples=5)
summary = cluster_summary(gdf, labels)

# Spatial autocorrelation / hotspot detection on a value column
moran = morans_i(gdf, value_col="value")
hotspots = getis_ord_hotspots(gdf, value_col="value")
```

## Modules

- `spatialdetection.io` — load points from CSV or vector files into a `GeoDataFrame`.
- `spatialdetection.clustering` — DBSCAN clustering with a haversine metric over lon/lat.
- `spatialdetection.autocorrelation` — Moran's I and Getis-Ord Gi* hotspot/coldspot detection.

## Development

```bash
uv run pytest
uv run ruff check .
```
