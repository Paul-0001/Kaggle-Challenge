
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "models/logistic_predictions.csv")

y_true = df["y_true"]
y_proba = df["y_proba"]

results = []

for threshold in np.arange(0.1, 0.901, 0.01):
    y_pred = (y_proba >= threshold).astype(int)

    results.append({
        "Threshold": round(threshold, 2),
        "F1": f1_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0)
    })

results = pd.DataFrame(results)

print(results.sort_values("F1", ascending=False).to_string(index=False))
