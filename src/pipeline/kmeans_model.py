import pandas as pd
from sklearn.cluster import KMeans

from .config import (
    DEFAULT_CLUSTER_COUNT,
    FEATURE_COLUMNS,
    KMEANS_CLUSTERED_CSV,
    KMEANS_METRICS_CSV,
    KMEANS_PROFILE_CSV,
    KMEANS_SWEEP_CSV,
    KMEANS_SWEEP_PNG,
)
from .metrics import evaluate_clustering_metrics, evaluate_kmeans_sweep
from .visualization import plot_kmeans_sweep


def run_kmeans_workflow(X_scaled, df_clean, n_clusters: int = DEFAULT_CLUSTER_COUNT) -> dict:
    sweep_df = evaluate_kmeans_sweep(X_scaled)
    sweep_df.to_csv(KMEANS_SWEEP_CSV, index=False)
    plot_kmeans_sweep(sweep_df, KMEANS_SWEEP_PNG)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    clustered_df = df_clean.copy()
    clustered_df['Cluster'] = labels
    clustered_df.to_csv(KMEANS_CLUSTERED_CSV, index=False)

    profile_df = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean().round(2)
    profile_df.to_csv(KMEANS_PROFILE_CSV)

    metrics = evaluate_clustering_metrics(X_scaled, labels)
    metrics_df = pd.DataFrame([
        {'model': 'kmeans', 'n_clusters': n_clusters, **metrics},
    ])
    metrics_df.to_csv(KMEANS_METRICS_CSV, index=False)

    return {
        'model': model,
        'labels': labels,
        'clustered_df': clustered_df,
        'profile_df': profile_df,
        'sweep_df': sweep_df,
        'metrics_df': metrics_df,
    }
