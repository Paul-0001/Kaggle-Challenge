'''
Script that prepares the file for sumbimssion for our best CatBoost model.
'''

# Import what needed
from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier

# Set project root
ROOT = Path(__file__).resolve().parents[1]

# Load trained CatBoost model
model = CatBoostClassifier()
model.load_model(str(ROOT / "models/catboost_model.cbm"))

# Load processed and raw test data
test = pd.read_csv(ROOT / "data/processed/test_features.csv")
raw = pd.read_csv(ROOT / "data/raw/test.csv")

# select the same features used during training
features = model.feature_names_
missing = set(features) - set(test.columns)

# Raise valueerror if we have missing features
if missing:
    raise ValueError(f"Missing features:{missing}")

# Copy all the features in X_test
X_test = test[features].copy()

# CatBoost-specific preprocessing. Same as in train_catboost function
categorical = model.get_cat_feature_indices()
for idx in categorical:
    col = features[idx]
    X_test[col] = X_test[col].fillna("Missing").astype(str)

# Replace infinite numerical values
numeric = X_test.select_dtypes(include="number").columns
X_test[numeric] = X_test[numeric].replace([float("inf"), -float("inf")], float("nan"))

# 5. Predict purchase probabilities and labels

# Set optimal threshold we computed before
THRESHOLD = float((ROOT / "models/catboost_threshold.txt").read_text().strip())

# Get the probabilities for the test data
probabilities = model.predict_proba(X_test)[:, 1]

# Get label predictions according to probability and threshold
predictions = (probabilities >= THRESHOLD).astype(int)

# Create Kaggle submission
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "buy": predictions
})

# Validate the sumbmission. Chould be as many predictions as observations in test
assert len(submission) == len(raw)

# Check that each row id is unique
assert submission["row_id"].is_unique

# Check that the row id are the same for submission and test
assert set(submission["row_id"]) == set(raw["row_id"])

# Check that the values in buy are either 1 or 0 like demanded
assert submission["buy"].isin([0, 1]).all()

# Save the submission
submission.to_csv(ROOT / "submission_catboost.csv", index=False)

# Usercheck that the submission file is ready
print(f"Saved {len(submission)} predictions to submission_catboost.csv")
