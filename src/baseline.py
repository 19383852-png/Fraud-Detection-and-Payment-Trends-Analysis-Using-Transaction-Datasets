"""Command-line entry point for the reproducible fraud-detection baseline pipeline."""

import argparse
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from .data_loader import load_data
from .preprocessing import build_preprocessor
from .models import make_lr, make_rf
from .evaluate import evaluate
from .config import RANDOM_STATE, TEST_SIZE

def run(data_path: str, use_smote: bool):
    # Load the dataset through a shared helper so all experiments start identically.
    df = load_data(data_path)

    # Separate predictors from the fraud label.
    X = df.drop(columns=["Class"])
    y = df["Class"]

    # Stratified splitting preserves the original fraud ratio in both train and test sets.
    # This matters because fraud detection is an extreme class-imbalance problem.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Build preprocessing once and reuse it inside each model pipeline.
    pre = build_preprocessor()

    def make_pipe(est, smote=False):
        # Wrapping preprocessing + model in one pipeline prevents data leakage:
        # test data is transformed using statistics learned only from training data.
        if smote:
            # SMOTE is optional here so we can compare the natural class distribution
            # against an oversampled training workflow in later experiments.
            return ImbPipeline([("pre", pre),
                                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.2)),
                                ("clf", est)])
        return Pipeline([("pre", pre), ("clf", est)])

    # Compare a linear baseline (Logistic Regression) with a non-linear ensemble (Random Forest).
    for name, est in [("LogReg", make_lr()), ("RandForest", make_rf())]:
        pipe = make_pipe(est, smote=use_smote)

        # The fit step learns both preprocessing statistics and model parameters on the training set.
        pipe.fit(X_train, y_train)

        # Hard predictions support the class report, while probabilities support ROC-AUC and PR-AUC.
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} (SMOTE={use_smote}) ===")
        evaluate(y_test, y_pred, y_proba)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    # Default path points to the local Kaggle dataset copy stored in /data.
    ap.add_argument("--data", default="./data/creditcard.csv")
    # This flag keeps the CLI simple during demos: --smote true or --smote false.
    ap.add_argument("--smote", default="false", help="true/false")
    args = ap.parse_args()
    run(args.data, args.smote.lower() == "true")
