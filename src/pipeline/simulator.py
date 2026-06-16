from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.config import DEFAULT_RANDOM_STATE
from src.pipeline.clustering_hierarchical import fit_hierarchical
from src.pipeline.clustering_kmeans import elbow_report, fit_kmeans, silhouette_report
from src.pipeline.duplicates import duplicate_summary, remove_duplicates
from src.pipeline.evaluation import best_score_from_report, clustering_metrics, summarize_clusters
from src.pipeline.feature_selection import remove_highly_correlated_features, remove_low_variance_features, select_numeric_features
from src.pipeline.outliers import outlier_mask_iqr, remove_outliers_iqr
from src.pipeline.scaling import scale_features


@dataclass
class PipelineResult:
    raw_data: pd.DataFrame
    cleaned_data: pd.DataFrame
    selected_features: list[str]
    scaled_data: pd.DataFrame
    elbow_table: pd.DataFrame
    silhouette_table: pd.DataFrame
    kmeans_labels: pd.Series
    hierarchical_labels: pd.Series
    kmeans_metrics: dict[str, float]
    hierarchical_metrics: dict[str, float]
    cluster_summary: pd.DataFrame
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class PipelineSimulator:
    scaling_method: str = "standard"
    k_min: int = 2
    k_max: int = 8
    linkage: str = "ward"
    random_state: int = DEFAULT_RANDOM_STATE

    def run(self, df: pd.DataFrame) -> PipelineResult:
        raw_data = df.copy()
        cleaned = remove_duplicates(raw_data)
        numeric_features = select_numeric_features(cleaned)
        outlier_columns = numeric_features.copy()
        no_outliers = remove_outliers_iqr(cleaned, outlier_columns) if outlier_columns else cleaned.copy()
        reduced, selected_after_variance = remove_low_variance_features(no_outliers)
        reduced, selected_features = remove_highly_correlated_features(reduced)

        if not selected_features:
            raise ValueError("No numeric features available for clustering.")

        scaled_data, _ = scale_features(reduced, selected_features, method=self.scaling_method)
        clustering_input = scaled_data[selected_features].copy()

        k_range = range(self.k_min, self.k_max + 1)
        elbow_table = elbow_report(clustering_input, k_range, random_state=self.random_state)
        silhouette_table = silhouette_report(clustering_input, k_range, random_state=self.random_state)
        best_k = int(best_score_from_report(silhouette_table, "silhouette_score")["k"])

        kmeans_model, kmeans_labels = fit_kmeans(clustering_input, best_k, random_state=self.random_state)
        hierarchical_model, hierarchical_labels = fit_hierarchical(clustering_input, best_k, linkage=self.linkage)

        kmeans_metrics = clustering_metrics(clustering_input, kmeans_labels)
        hierarchical_metrics = clustering_metrics(clustering_input, hierarchical_labels)
        cluster_summary = summarize_clusters(clustering_input, kmeans_labels)

        metadata = {
            "duplicate_summary": duplicate_summary(raw_data),
            "outlier_rows": int(outlier_mask_iqr(cleaned, outlier_columns).sum()) if outlier_columns else 0,
            "selected_features_after_variance": selected_after_variance,
            "best_k": best_k,
            "kmeans_inertia": float(kmeans_model.inertia_),
            "hierarchical_model": hierarchical_model.__class__.__name__,
        }

        return PipelineResult(
            raw_data=raw_data,
            cleaned_data=no_outliers,
            selected_features=selected_features,
            scaled_data=scaled_data,
            elbow_table=elbow_table,
            silhouette_table=silhouette_table,
            kmeans_labels=kmeans_labels,
            hierarchical_labels=hierarchical_labels,
            kmeans_metrics=kmeans_metrics,
            hierarchical_metrics=hierarchical_metrics,
            cluster_summary=cluster_summary,
            metadata=metadata,
        )
