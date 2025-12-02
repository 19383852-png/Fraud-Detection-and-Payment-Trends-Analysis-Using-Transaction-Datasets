# Fraud Detection and Payment Trends Analysis Using Machine Learning

This repository presents an end-to-end machine-learning workflow for analysing and detecting fraudulent transactions using the Kaggle Credit Card Fraud Detection Dataset. The project incorporates exploratory analysis, supervised and unsupervised modelling, synthetic oversampling, evaluation under extreme class imbalance, and conceptual integration with distributed computing technologies such as Apache Spark, Kafka, and Flink.

The structure follows an academic research workflow and aligns with industry standards in financial fraud detection.

---

## Repository Structure

```

├── data/
│   ├── .gitkeep
│   ├── creditcard.csv
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_smote_models.ipynb
│
├── scripts/
│   ├── ingest_to_mongo.py
│
├── src/
│   ├── **init**.py
│   ├── baseline.py
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluate.py
│   ├── models.py
│   ├── preprocessing.py
│
├── venv/
├── .gitignore
├── requirements.txt
└── README.md

```

---

## Project Overview

The objective of this project is to investigate financial fraud patterns and develop machine-learning models capable of detecting fraudulent transactions in highly imbalanced datasets.  
The project covers:

- Exploratory data analysis (EDA)
- Baseline supervised and unsupervised modelling
- Synthetic oversampling (SMOTE)
- Performance evaluation under class imbalance
- MongoDB storage integration
- Conceptual distributed streaming analytics using Spark, Kafka, and Flink

---

## Dataset Description

The dataset includes:

- 284,807 credit-card transactions  
- 492 fraudulent transactions  
- Fraud ratio of approximately 0.17%  
- PCA-transformed features: V1–V28  
- Time and Amount variables  
- Binary label: 0 (legitimate), 1 (fraud)

Dataset location:

```

/data/creditcard.csv

````

---

## Exploratory Data Analysis (EDA)

The EDA notebook and its exported PDF contain:

- Class distribution visualisation  
- PCA feature structure  
- Correlation heatmaps  
- Payment amount trends  
- Fraud vs. non-fraud behavioural patterns  
- Outlier detection insights  

Open EDA via:

```bash
jupyter notebook notebooks/01_eda.ipynb
````

---

## Baseline Models

Notebook:

```
notebooks/02_baseline_models.ipynb
```

Algorithms implemented:

* Logistic Regression
* Random Forest
* Gradient Boosting
* Isolation Forest
* K-Means Clustering

Supporting Python modules:

```
src/baseline.py
src/models.py
src/preprocessing.py
src/evaluate.py
```

Run baseline models:

```bash
python -m src.baseline
```

---

## SMOTE and Synthetic Oversampling

Oversampling experiments are in:

```
notebooks/03_smote_models.ipynb
```

This notebook evaluates:

* SMOTE application
* Effect of balancing on model performance
* Limitations of oversampling PCA-transformed data

Functions defined in:

```
src/preprocessing.py
```

---

## MongoDB Integration

The script used to ingest data into MongoDB:

```
scripts/ingest_to_mongo.py
```

Run the ingestion tool:

```bash
python scripts/ingest_to_mongo.py
```

---

## Big-Data Architecture (Conceptual)

The following diagram illustrates the conceptual real-time fraud-detection pipeline:

```
┌──────────────────────────────┐
│        Kafka Stream           │
│  Continuous Transactions      │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│          Spark Engine         │
│ Batch ETL and Model Training │
└───────────────┬──────────────┘
                │
                ▼
┌──────────────────────────────┐
│           Flink Layer         │
│  Real-Time Scoring and Alerts │
└──────────────────────────────┘
```

Purpose of each layer:

* **Kafka**: Real-time ingestion of high-volume payment streams
* **Spark**: Distributed preprocessing and batch ML training
* **Flink**: Immediate anomaly scoring on live transaction events

---

## Model Evaluation

Evaluation functions:

```
src/evaluate.py
```

Metrics used:

* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* Learning Curves
* Calibration Curves

These metrics provide an accurate assessment of model performance under extreme imbalance.

---

## Installation and Environment Setup

Clone the repository:

```bash
git clone https://github.com/19383852-png/Fraud-Detection-and-Payment-Trends-Analysis-Using-Transaction-Datasets
cd Fraud-Detection-and-Payment-Trends-Analysis-Using-Transaction-Datasets
```

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Reproducibility

This project ensures reproducibility through:

* Modularised source code under `/src`
* Consistent random seeds in `config.py`
* Clear separation of notebooks, data, and scripts
* Git-based version control
* PDF export of analysis outputs (EDA and reflection)

---

## Future Enhancements

Planned extensions include:

* Spark Structured Streaming for real-time ETL
* Kafka-to-Spark automated ingestion pipeline
* Hyperparameter tuning with Optuna
* SHAP and LIME for explainability
* Deployment using Docker and CI/CD
