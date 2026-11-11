'''
Function that evaluates the model's output's in a standardized way so we can properly rank them. Uses criteria from ranking from pdf.
Else we compare appls to pears.
'''

# Import what needed
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,recall_score, f1_score, log_loss)

# Create the function that does the evaluation. Input has to be a df with 3 columns: row_id, y_true and y_proba
def evaluate_models(prediction_files, holdout_path, threshold=0.5):

    # Get the out of sample dataset to evaluate the models
    holdout = pd.read_csv(holdout_path)

    # Get the reference columns
    reference = holdout[["row_id", "buy"]]

    # Raise valueerror if still missing or duplicate values for our reference row id 
    if reference["row_id"].isna().any() or reference["row_id"].duplicated().any():
        raise ValueError("Invalid row_id in holdout")

    # Create empty list for our results
    results = []

    # 
    for name, path in prediction_files.items():

        # Read the dataframe
        df = pd.read_csv(path)

        # Should be in discussed format else raise valueerror
        if set(df.columns) != {"row_id", "y_true", "y_proba"}:
            raise ValueError(f"{name}: invalid columns")

        
        if df["row_id"].isna().any() or df["row_id"].duplicated().any():
            raise ValueError(f"{name}: invalid row_id")

        if len(df) != len(reference) or set(df["row_id"]) != set(reference["row_id"]):
            raise ValueError(f"{name}: predictions do not match holdout")

        df = reference.merge(df, on="row_id", validate="one_to_one")

        if not df["y_true"].equals(df["buy"]):
            raise ValueError(f"{name}: target mismatch")

        if not np.isfinite(df["y_proba"]).all():
            raise ValueError(f"{name}: invalid probabilities")

        if not df["y_proba"].between(0, 1).all():
            raise ValueError(f"{name}: probabilities outside [0,1]")

        y = df["buy"]
        p = df["y_proba"]
        pred = (p >= threshold).astype(int)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y, pred),
            "ROC_AUC": roc_auc_score(y, p),
            "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "Log_Loss": log_loss(y, p, labels=[0, 1])
        })

    return pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    models = root / "models"

    # Add each team member's prediction file here
    files = {
        "Logistic Regression": models / "logistic_predictions.csv",
        # "CatBoost": models / "catboost_predictions.csv",
        # "LightGBM": models / "lightgbm_predictions.csv",
    }

    results = evaluate_models(
        files,
        root / "data/processed/holdout_split.csv"
    )

    print("\n=== MODEL COMPARISON ===")
    print(results.round(4).to_string(index=False))

    results.to_csv(models / "model_comparison.csv", index=False)