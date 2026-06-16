from pathlib import Path

import matplotlib.pyplot as plt

try:
    from scipy.cluster.hierarchy import dendrogram, linkage
except ImportError:  # pragma: no cover - optional dependency
    dendrogram = None
    linkage = None


def create_eda_overview(df, output_path: Path) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Customer Profile Overview', fontsize=16, fontweight='bold')

    axes[0, 0].hist(df['Age'], bins=30, color='steelblue', edgecolor='white', linewidth=0.8)
    axes[0, 0].set_title('Customer Age Distribution', fontweight='bold')
    axes[0, 0].set_xlabel('Age')
    axes[0, 0].set_ylabel('Count')

    axes[0, 1].hist(df['Total_Spent'], bins=30, color='coral', edgecolor='white', linewidth=0.8)
    axes[0, 1].set_title('Total Spending Distribution', fontweight='bold')
    axes[0, 1].set_xlabel('Total Spent ($)')
    axes[0, 1].set_ylabel('Count')

    axes[1, 0].hist(df['Income'], bins=30, color='mediumseagreen', edgecolor='white', linewidth=0.8)
    axes[1, 0].set_title('Income Distribution', fontweight='bold')
    axes[1, 0].set_xlabel('Income ($)')
    axes[1, 0].set_ylabel('Count')

    axes[1, 1].scatter(df['Income'], df['Total_Spent'], alpha=0.4, color='steelblue', s=20)
    axes[1, 1].set_title('Income vs Total Spending', fontweight='bold')
    axes[1, 1].set_xlabel('Income ($)')
    axes[1, 1].set_ylabel('Total Spent ($)')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_kmeans_sweep(sweep_df, output_path: Path) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].plot(sweep_df['k'], sweep_df['silhouette'], marker='o')
    axes[0].set_title('Baseline Silhouette Score')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Silhouette')

    axes[1].plot(sweep_df['k'], sweep_df['davies_bouldin'], marker='o', color='orange')
    axes[1].set_title('Baseline Davies-Bouldin Index')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('DB Index')

    axes[2].plot(sweep_df['k'], sweep_df['calinski_harabasz'], marker='o', color='green')
    axes[2].set_title('Baseline Calinski-Harabasz Score')
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('CH Score')

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path


def plot_hierarchical_dendrogram(X_scaled, output_path: Path, sample_size: int = 200) -> Path | None:
    if linkage is None or dendrogram is None:
        return None

    sample_size = min(len(X_scaled), sample_size)
    if sample_size < 2:
        return None

    sample = X_scaled[:sample_size]
    linkage_matrix = linkage(sample, method='ward')

    fig = plt.figure(figsize=(14, 6))
    dendrogram(linkage_matrix, truncate_mode='lastp', p=20, leaf_rotation=90)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Cluster size')
    plt.ylabel('Distance')
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return output_path
