from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / 'data'
OUTPUT_DIR = PROJECT_ROOT / 'outputs'

RAW_DATA_PATH = DATA_DIR / 'marketing_campaign.csv'
KAGGLE_DATA_PATH = DATA_DIR / 'marketing_campaign_kaggle.csv'

CLEAN_FEATURES_CSV = OUTPUT_DIR / 'marketing_campaign_clean_features.csv'
EDA_OVERVIEW_PNG = OUTPUT_DIR / '01_eda_overview.png'
KMEANS_SWEEP_PNG = OUTPUT_DIR / '02_metricas-codo.png'
DENDROGRAM_PNG = OUTPUT_DIR / '03_hierarchical_dendrogram.png'

KMEANS_CLUSTERED_CSV = OUTPUT_DIR / 'kmeans_clustered_dataset.csv'
KMEANS_PROFILE_CSV = OUTPUT_DIR / 'kmeans_cluster_profile.csv'
KMEANS_SWEEP_CSV = OUTPUT_DIR / 'kmeans_k_sweep_metrics.csv'
KMEANS_METRICS_CSV = OUTPUT_DIR / 'kmeans_metrics.csv'

HIERARCHICAL_CLUSTERED_CSV = OUTPUT_DIR / 'hierarchical_clustered_dataset.csv'
HIERARCHICAL_PROFILE_CSV = OUTPUT_DIR / 'hierarchical_cluster_profile.csv'
HIERARCHICAL_METRICS_CSV = OUTPUT_DIR / 'hierarchical_metrics.csv'

FEATURE_COLUMNS = [
    'Income',
    'Age',
    'Total_Spent',
    'Total_Purchases',
    'Total_Children',
    'Tenure',
    'Recency',
    'NumWebVisitsMonth',
]

DEFAULT_CLUSTER_COUNT = 4
INCOME_OUTLIER_THRESHOLD = 200000
REFERENCE_YEAR = 2026
REFERENCE_DATE = '2026-01-01'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
