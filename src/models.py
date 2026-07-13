from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_lr():
    return LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        verbose=1
    )

def make_rf():
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )