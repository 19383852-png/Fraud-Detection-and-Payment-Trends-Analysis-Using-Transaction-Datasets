"""Shared configuration values used across the fraud-detection pipeline."""

# Keeping the dataset path in one place makes notebooks and scripts easier to align.
DEFAULT_DATA_PATH = "./data/creditcard.csv"

# Fixed seeds make our train/test split and model behaviour reproducible for reporting.
RANDOM_STATE = 420

# A 15% holdout set leaves enough data for training while preserving a realistic test set.
TEST_SIZE = 0.15
