import pandas as pd

from .config import KAGGLE_DATA_PATH, RAW_DATA_PATH


def resolve_dataset_path() -> str:
    if RAW_DATA_PATH.exists():
        return str(RAW_DATA_PATH)
    if KAGGLE_DATA_PATH.exists():
        return str(KAGGLE_DATA_PATH)
    raise FileNotFoundError(
        'No se encontró el dataset en data/marketing_campaign.csv ni en data/marketing_campaign_kaggle.csv'
    )


def load_raw_dataset() -> tuple[pd.DataFrame, str]:
    dataset_path = resolve_dataset_path()
    dataset = pd.read_csv(dataset_path, sep='\t')
    return dataset, dataset_path
