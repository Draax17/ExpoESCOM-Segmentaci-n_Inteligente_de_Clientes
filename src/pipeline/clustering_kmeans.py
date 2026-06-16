from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def fit_kmeans(df: pd.DataFrame, n_clusters: int, random_state: int = 42) -> tuple[KMeans, pd.Series]:
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(df)
    return model, pd.Series(labels, index=df.index, name="kmeans_cluster")


def elbow_report(df: pd.DataFrame, k_range: range, random_state: int = 42) -> pd.DataFrame:
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        model.fit(df)
        records.append({"k": k, "inertia": model.inertia_})
    return pd.DataFrame(records)


def silhouette_report(df: pd.DataFrame, k_range: range, random_state: int = 42) -> pd.DataFrame:
    records = []
    for k in k_range:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = model.fit_predict(df)
        score = silhouette_score(df, labels)
        records.append({"k": k, "silhouette_score": score})
    return pd.DataFrame(records)
