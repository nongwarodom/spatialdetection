import numpy as np
import pandas as pd
import pytest

from spatialdetection import cluster_summary, dbscan_clusters, points_from_dataframe

rng = np.random.default_rng(0)


def test_dbscan_finds_tight_cluster_and_isolated_noise():
    tight = rng.normal(loc=0.0, scale=0.001, size=(20, 2))  # ~100m spread
    isolated = np.array([[5.0, 5.0]])  # far away, should be noise
    coords = np.vstack([tight, isolated])
    df = pd.DataFrame({"lon": coords[:, 0], "lat": coords[:, 1]})
    gdf = points_from_dataframe(df)

    labels = dbscan_clusters(gdf, eps_km=1.0, min_samples=5)

    assert labels[-1] == -1  # isolated point is noise
    assert (labels[:-1] == labels[0]).all()  # tight points form one cluster
    assert labels[0] != -1


def test_cluster_summary_excludes_noise_and_reports_size():
    tight = rng.normal(loc=0.0, scale=0.001, size=(20, 2))
    isolated = np.array([[5.0, 5.0]])
    coords = np.vstack([tight, isolated])
    df = pd.DataFrame({"lon": coords[:, 0], "lat": coords[:, 1]})
    gdf = points_from_dataframe(df)
    labels = dbscan_clusters(gdf, eps_km=1.0, min_samples=5)

    summary = cluster_summary(gdf, labels)

    assert len(summary) == 1  # only the tight cluster, noise excluded
    assert summary.iloc[0]["size"] == 20
    assert summary.crs == gdf.crs


def test_dbscan_clusters_rejects_projected_crs():
    df = pd.DataFrame({"lon": [100.0, 100.1, 100.2], "lat": [13.0, 13.1, 13.2]})
    gdf = points_from_dataframe(df).to_crs("EPSG:32647")  # UTM meters, not lon/lat degrees

    with pytest.raises(ValueError, match="geographic"):
        dbscan_clusters(gdf, eps_km=50)


def test_dbscan_clusters_accepts_plain_dataframe():
    tight = rng.normal(loc=0.0, scale=0.001, size=(20, 2))
    isolated = np.array([[5.0, 5.0]])
    coords = np.vstack([tight, isolated])
    df = pd.DataFrame({"lon": coords[:, 0], "lat": coords[:, 1]})  # plain DataFrame, no geometry

    labels = dbscan_clusters(df, eps_km=1.0, min_samples=5)

    assert labels[-1] == -1
    assert (labels[:-1] == labels[0]).all()


def test_cluster_summary_accepts_plain_dataframe():
    tight = rng.normal(loc=0.0, scale=0.001, size=(20, 2))
    isolated = np.array([[5.0, 5.0]])
    coords = np.vstack([tight, isolated])
    df = pd.DataFrame({"lon": coords[:, 0], "lat": coords[:, 1]})
    labels = dbscan_clusters(df, eps_km=1.0, min_samples=5)

    summary = cluster_summary(df, labels)

    assert len(summary) == 1
    assert summary.iloc[0]["size"] == 20
