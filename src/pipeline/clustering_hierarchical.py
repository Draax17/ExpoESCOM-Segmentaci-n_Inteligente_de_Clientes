from __future__ import annotations

import pandas as pd
from sklearn.cluster import AgglomerativeClustering


def fit_hierarchical(df: pd.DataFrame, n_clusters: int, linkage: str = "ward") -> tuple[AgglomerativeClustering, pd.Series]:
    model = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    labels = model.fit_predict(df)
    return model, pd.Series(labels, index=df.index, name="hierarchical_cluster")
