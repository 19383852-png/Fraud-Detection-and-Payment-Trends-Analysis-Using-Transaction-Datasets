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
    df = load_data(data_path)
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    pre = build_preprocessor()

    def make_pipe(est, smote=False):
        if smote:
            return ImbPipeline([("pre", pre),
                                ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.2)),
                                ("clf", est)])
        return Pipeline([("pre", pre), ("clf", est)])

    for name, est in [("LogReg", make_lr()), ("RandForest", make_rf())]:
        pipe = make_pipe(est, smote=use_smote)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]
        print(f"\n=== {name} (SMOTE={use_smote}) ===")
        evaluate(y_test, y_pred, y_proba)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data/creditcard.csv")
    ap.add_argument("--smote", default="false", help="true/false")
    args = ap.parse_args()
    run(args.data, args.smote.lower() == "true")
