# Presentation Guide

Use the files below in this order during your meeting.

1. `README.md`
Reason: Start with the project goal, dataset, and overall workflow before diving into code.

2. `src/config.py`
Reason: Show that the project is reproducible because the dataset path, random seed, and test split are controlled centrally.

3. `src/data_loader.py`
Reason: Explain that you created a reusable loader and validated that the target column `Class` exists before any modelling starts.

4. `src/preprocessing.py`
Reason: Explain that only `Time` and `Amount` are scaled, while `V1..V28` are left untouched because they are already PCA-like features.

5. `src/models.py`
Reason: Present the two baseline models and explain why class weighting was added for imbalanced fraud detection.

6. `src/baseline.py`
Reason: This is the main pipeline file. Walk through loading data, separating `X` and `y`, stratified splitting, building pipelines, optional SMOTE, training, prediction, and evaluation.

7. `src/evaluate.py`
Reason: Explain why you used ROC-AUC, PR-AUC, precision, recall, and F1-score instead of relying on accuracy alone.

8. `notebooks/02_baseline_models.ipynb`
Reason: Use this notebook if your professor asks about the experiment narrative, timing, reduced Random Forest sample, or intermediate printed outputs.

9. `outputs/metrics/data_split.txt`
Reason: Show the exact train/test sizes and that class imbalance was preserved after stratified splitting.

10. `outputs/metrics/logreg_baseline.txt`
Reason: Use this to explain the Logistic Regression trade-off: very high recall for fraud, but many false positives.

11. `outputs/metrics/rf_baseline.txt`
Reason: Use this to explain why Random Forest gave a better practical balance of precision, recall, and F1-score.

12. `outputs/tables/baseline_results.csv`
Reason: End with a compact comparison table that summarises the baseline findings clearly.

## Short speaking flow

1. Project purpose
2. Reproducible code structure
3. Data loading and validation
4. Preprocessing decisions
5. Model choices
6. Stratified split and pipeline design
7. Evaluation metrics for imbalanced data
8. Final results and model trade-offs
