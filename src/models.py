"""Factory functions for the baseline models used in the project."""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def make_lr():
    # Logistic Regression is our interpretable baseline.
    # 'class_weight=balanced' compensates for the extreme fraud/legit imbalance.
    return LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        verbose=1
    )

def make_rf():
    # Random Forest captures non-linear fraud patterns that a linear model may miss.
    # Balanced subsampling reweights each tree so rare fraud cases matter during training.
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
        verbose=1
    )
