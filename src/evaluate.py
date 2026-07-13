from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

def evaluate(y_true, y_pred, y_proba):
    print("ROC-AUC :", roc_auc_score(y_true, y_proba))
    print("PR  AUC:", average_precision_score(y_true, y_proba))
    print(classification_report(y_true, y_pred, digits=4))
