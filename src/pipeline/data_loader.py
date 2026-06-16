from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.io import read_tabular_file


def load_dataset(path: str | Path) -> pd.DataFrame:
    return read_tabular_file(path)


def generate_demo_customer_dataset(n_samples: int = 300, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    customer_type = rng.integers(0, 4, size=n_samples)

    age = np.clip(rng.normal(35 + customer_type * 5, 8, size=n_samples), 18, 75)
    annual_income = np.clip(rng.normal(45_000 + customer_type * 18_000, 10_000, size=n_samples), 15_000, 150_000)
    spending_score = np.clip(rng.normal(40 + customer_type * 12, 15, size=n_samples), 1, 100)
    frequency = np.clip(rng.normal(6 + customer_type * 3, 2, size=n_samples), 1, 20)
    recency_days = np.clip(rng.normal(60 - customer_type * 10, 18, size=n_samples), 1, 180)

    return pd.DataFrame(
        {
            "customer_id": [f"CUST-{index:04d}" for index in range(1, n_samples + 1)],
            "age": age.round(0).astype(int),
            "annual_income": annual_income.round(0).astype(int),
            "spending_score": spending_score.round(0).astype(int),
            "purchase_frequency": frequency.round(0).astype(int),
            "recency_days": recency_days.round(0).astype(int),
        }
    )
