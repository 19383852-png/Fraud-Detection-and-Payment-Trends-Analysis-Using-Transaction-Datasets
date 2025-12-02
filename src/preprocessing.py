from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

COLS_SCALE = ["Time", "Amount"]  # V1..V28 are already PCA-like

def build_preprocessor():
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler())
    ])
    return ColumnTransformer([
        ("num", num_pipe, COLS_SCALE)
    ], remainder="passthrough")
