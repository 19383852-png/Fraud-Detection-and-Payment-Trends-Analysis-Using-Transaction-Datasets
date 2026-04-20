"""Utilities for loading the transaction dataset in a consistent way."""

import pandas as pd
from .config import DEFAULT_DATA_PATH

def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    # Centralising file loading means every script starts from the same validated dataset.
    df = pd.read_csv(path)

    # The project assumes a binary target column called 'Class'
    # where 1 = fraud and 0 = legitimate transaction.
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column (1=fraud, 0=legit).")
    return df
