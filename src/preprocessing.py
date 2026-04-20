"""Preprocessing steps for the baseline fraud-detection models."""

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Only the raw numeric fields need scaling;
# the anonymised V1..V28 columns are already PCA-like and are left unchanged.
COLS_SCALE = ["Time", "Amount"]

def build_preprocessor():
    # Median imputation keeps the pipeline robust if missing values appear later.
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])

    # 'remainder=passthrough' preserves the transformed V-features without re-scaling them.
    return ColumnTransformer([
        ("num", num_pipe, COLS_SCALE)
    ], remainder="passthrough")
