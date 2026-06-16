import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def evaluate_clustering_metrics(X_scaled, labels) -> dict[str, float]:
    return {
        'silhouette': silhouette_score(X_scaled, labels),
        'davies_bouldin': davies_bouldin_score(X_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
    }


def evaluate_kmeans_sweep(X_scaled, k_values=range(2, 11), random_state: int = 42, n_init: int = 10) -> pd.DataFrame:
    rows = []
    for k in k_values:
        model = KMeans(n_clusters=k, random_state=random_state, n_init=n_init)
        labels = model.fit_predict(X_scaled)
        rows.append(
            {
                'k': k,
                'inertia': model.inertia_,
                'silhouette': silhouette_score(X_scaled, labels),
                'davies_bouldin': davies_bouldin_score(X_scaled, labels),
                'calinski_harabasz': calinski_harabasz_score(X_scaled, labels),
            }
        )
    return pd.DataFrame(rows)
