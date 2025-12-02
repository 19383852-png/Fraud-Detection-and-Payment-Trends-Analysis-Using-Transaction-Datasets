import pandas as pd
from .config import DEFAULT_DATA_PATH

def load_data(path: str = DEFAULT_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Class" not in df.columns:
        raise ValueError("Dataset must contain a 'Class' column (1=fraud, 0=legit).")
    return df
