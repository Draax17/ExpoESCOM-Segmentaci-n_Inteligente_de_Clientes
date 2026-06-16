import pandas as pd

from .config import (
    CLEAN_FEATURES_CSV,
    FEATURE_COLUMNS,
    INCOME_OUTLIER_THRESHOLD,
    REFERENCE_DATE,
    REFERENCE_YEAR,
)


def preprocess_customer_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    cleaned = dataset.dropna().copy()
    cleaned['Dt_Customer'] = pd.to_datetime(cleaned['Dt_Customer'], dayfirst=True)
    cleaned['Age'] = REFERENCE_YEAR - cleaned['Year_Birth']
    cleaned = cleaned[cleaned['Age'] < 100].copy()

    cleaned['Total_Spent'] = (
        cleaned['MntWines']
        + cleaned['MntFruits']
        + cleaned['MntMeatProducts']
        + cleaned['MntFishProducts']
        + cleaned['MntSweetProducts']
        + cleaned['MntGoldProds']
    )

    cleaned['Total_Purchases'] = (
        cleaned['NumDealsPurchases']
        + cleaned['NumWebPurchases']
        + cleaned['NumCatalogPurchases']
        + cleaned['NumStorePurchases']
    )

    cleaned['Total_Children'] = cleaned['Kidhome'] + cleaned['Teenhome']
    cleaned['Tenure'] = (pd.to_datetime(REFERENCE_DATE) - cleaned['Dt_Customer']).dt.days
    return cleaned


def build_clustering_inputs(cleaned_dataset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_clean = cleaned_dataset[cleaned_dataset['Income'] < INCOME_OUTLIER_THRESHOLD].copy()
    feature_frame = df_clean[FEATURE_COLUMNS].copy()
    return df_clean, feature_frame


def save_clean_feature_dataset(feature_frame: pd.DataFrame, output_path=CLEAN_FEATURES_CSV) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_frame.to_csv(output_path, index=False)
    return str(output_path)
