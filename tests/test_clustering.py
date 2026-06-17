import numpy as np
import pandas as pd

from spatialdetection import dbscan_clusters, points_from_dataframe

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
