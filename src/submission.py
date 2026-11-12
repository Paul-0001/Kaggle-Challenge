'''
This script takes the parameters we got in training for a model and predicts the lables for the test set to submit on kaggle
'''

# Import what needed
from pathlib import Path
import joblib
import pandas as pd

# Set root of main project
ROOT = Path(__file__).resolve().parents[1]

# Load trained model params
model = joblib.load(ROOT / "models/logistic_model.joblib")

# Load features
test = pd.read_csv(ROOT / "data/processed/test_features.csv")

# load test data for predictions
raw = pd.read_csv(ROOT / "data/raw/test.csv")

# Use exactly the features expected by the trained pipeline
features = model.feature_names_in_
missing = set(features) - set(test.columns)
if missing:
    raise ValueError(f"Missing features: {missing}")

X_test = test[list(features)].replace([float("inf"), -float("inf")], float("nan"))

# Predict binary labels
THRESHOLD = 0.4
probabilities = model.predict_proba(X_test)[:, 1]
predictions = (probabilities >= THRESHOLD).astype(int)

# Create Kaggle submission
submission = pd.DataFrame({
    "row_id": test["row_id"],
    "buy": predictions.astype(int)
})

# Validate submission
assert len(submission) == len(raw)
assert submission["row_id"].is_unique
assert set(submission["row_id"]) == set(raw["row_id"])
assert submission["buy"].isin([0, 1]).all()

submission.to_csv(ROOT / "submission.csv", index=False)
print(f"Saved {len(submission)} predictions to submission.csv")
