import pandas as pd
from sklearn.cluster import AgglomerativeClustering

from .config import (
    DEFAULT_CLUSTER_COUNT,
    FEATURE_COLUMNS,
    HIERARCHICAL_CLUSTERED_CSV,
    HIERARCHICAL_METRICS_CSV,
    HIERARCHICAL_PROFILE_CSV,
    DENDROGRAM_PNG,
)
from .metrics import evaluate_clustering_metrics
from .visualization import plot_hierarchical_dendrogram


def run_hierarchical_workflow(X_scaled, df_clean, n_clusters: int = DEFAULT_CLUSTER_COUNT) -> dict:
    plot_hierarchical_dendrogram(X_scaled, DENDROGRAM_PNG)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_scaled)

    clustered_df = df_clean.copy()
    clustered_df['Cluster'] = labels
    clustered_df.to_csv(HIERARCHICAL_CLUSTERED_CSV, index=False)

    profile_df = clustered_df.groupby('Cluster')[FEATURE_COLUMNS].mean().round(2)
    profile_df.to_csv(HIERARCHICAL_PROFILE_CSV)

    metrics = evaluate_clustering_metrics(X_scaled, labels)
    metrics_df = pd.DataFrame([
        {'model': 'hierarchical', 'n_clusters': n_clusters, **metrics},
    ])
    metrics_df.to_csv(HIERARCHICAL_METRICS_CSV, index=False)

    return {
        'model': model,
        'labels': labels,
        'clustered_df': clustered_df,
        'profile_df': profile_df,
        'metrics_df': metrics_df,
    }
