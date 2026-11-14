'''
Function that evaluates the model's output's in a standardized way so we can properly rank them. Uses criteria from ranking from pdf.
Else we compare apple to pears.
'''

# Import what needed
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,recall_score, f1_score, log_loss)

# Create the function that does the evaluation. Input has to be a df with 3 columns: row_id, y_true and y_proba
def evaluate_models(prediction_files, holdout_path, thresholds=None):

    # Get the thresholds for the model (F1-opt)
    thresholds = thresholds or {}


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

        # No missing or duplicate values for row id else raise valueerror
        if df["row_id"].isna().any() or df["row_id"].duplicated().any():
            raise ValueError(f"{name}: invalid row_id")

        # Dimensions have to match
        if len(df) != len(reference) or set(df["row_id"]) != set(reference["row_id"]):
            raise ValueError(f"{name}: predictions do not match holdout")

        # Merge both dataframes
        df = reference.merge(df, on="row_id", validate="one_to_one")

        # y-trua values and buy values have to match
        if not df["y_true"].equals(df["buy"]):
            raise ValueError(f"{name}: target mismatch")

        # model probabilitis have to be finite else valueerror
        if not np.isfinite(df["y_proba"]).all():
            raise ValueError(f"{name}: invalid probabilities")

        # model prob have between 0 and 1 included else raise valueerror
        if not df["y_proba"].between(0, 1).all():
            raise ValueError(f"{name}: probabilities outside [0,1]")

        # y takes the target values for out of smaple df
        y = df["buy"]

        # p takes the predictions of the model for the out of sample data
        p = df["y_proba"]

        # if p > 0.5 or whatever threshold set then set label buy else 0
        model_threshold = thresholds.get(name, 0.5)
        pred = (p >= model_threshold).astype(int)

        # Append all the important informations to the results dict for the model
        results.append({

            # What model is it (Logist for example)
            "Model": name,

            # What is the accuracy in the out of smaple data
            "Accuracy": accuracy_score(y, pred),

            # What is our ROC-AUC (Area under the curve)
            "ROC_AUC": roc_auc_score(y, p),

            # What is our precision score ()
            "Precision": precision_score(y, pred, zero_division=0),

            # What is our recall score ()
            "Recall": recall_score(y, pred, zero_division=0),

            # Waht is the F1 score ()
            "F1": f1_score(y, pred, zero_division=0),

            # What is the log loss of the model
            "Log_Loss": log_loss(y, p, labels=[0, 1])
        })

    # Return the results for the model in dict results and sort that dict according to the ROC-AUC values. Best first.
    return pd.DataFrame(results).sort_values("ROC_AUC", ascending=False)

# Function that handles everything
if __name__ == "__main__":

    # Set root of project
    root = Path(__file__).resolve().parents[1]

    # Create subfolder containing the results of our models
    models = root / "models"

    # Add the name for the result file for each model
    files = {
        "Logistic Regression": models / "logistic_predictions.csv", # First one we made and submitted
        "CatBoost": models / "catboost_predictions.csv", # Second
        "LightGBM": models / "lightgbm_predictions.csv", # Third
    }

    # Evaluate the models and save as csv
    results = evaluate_models(
        files,
        root / "data/processed/holdout_split.csv",

        # Dont set uniform thresholds since opt may be different for different models
        thresholds={

            # Was the optimum regarding maximizing our F1 score
            "Logistic Regression": 0.4,

            # Get the optimal threshold for evaluation that was calculated directly in train_catboost.py
            "CatBoost": float((models / "catboost_threshold.txt").read_text())
        }
    )

    # Print the model metrics (For all models we made until then to compare)
    print("\nModel Metrics")
    print(results.round(4).to_string(index=False))

    # Save comparison results in CSV
    results.to_csv(models / "model_comparison.csv", index=False)