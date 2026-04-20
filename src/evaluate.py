"""Evaluation helpers focused on imbalanced-classification metrics."""

from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def evaluate(y_true, y_pred, y_proba):
    # ROC-AUC measures overall ranking quality across thresholds.
    print("ROC-AUC :", roc_auc_score(y_true, y_proba))

    # PR-AUC is especially important here because fraud is the rare positive class.
    print("PR  AUC:", average_precision_score(y_true, y_proba))

    # The class report gives precision, recall, and F1 for both legit and fraud classes.
    print(classification_report(y_true, y_pred, digits=4))
