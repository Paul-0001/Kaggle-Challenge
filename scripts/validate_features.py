'''
Script that checks the values of features, validates averything, and gives back datasets for training and out of sample we can work with modelspecifically. 
'''

# Import what needed
import numpy as np
import pandas as pd
from scripts.feature_engineering import EDA_COLUMNS
# Columns excluded from model inputs.
# EDA_COLUMNS = the 16 columns I added to scripts/feature_engineering.py (the EDA features + recency).
# They are excluded here so that logistic regression and CatBoost paul implemented, keep training on the 145 candidate features only,
# exactly as for their Kaggle submissions (LR 0.79918, CatBoost 0.83234). LightGBM uses them in src/train_lightGBM.py.
# To let LR and CatBoost use them too, remove "set(EDA_COLUMNS) | " below.
EXCLUDE = set(EDA_COLUMNS) | {"buy", "row_id", "user_id", "music_app_user_id", "concert_id", "artist_id","concert_at", "sale_opened_at", "prediction_at", "travel_search_arrival"}




# Function that validates all our train and out of sample datasets
def validate_features(train, holdout):
    """Validate train/holdout data and return model-ready feature subsets."""

    # Chech in training dataset and out of sample data set
    for name, df in [("train", train), ("holdout", holdout)]:

        # Check that we have target value and user id
        if not {"buy", "user_id"}.issubset(df.columns):
            raise ValueError(f"{name}: missing buy or user_id")

        # Check that all the values in the target column are non empty and that they have a value in 0, 1
        if df["buy"].isna().any() or not df["buy"].isin([0, 1]).all():
            raise ValueError(f"{name}: invalid target values")

        # Check that no user id values are missing
        if df["user_id"].isna().any():
            raise ValueError(f"{name}: missing user_id")

        # Check that the column row id is not missing
        if "row_id" not in df.columns:
            raise ValueError(f"{name}: missing row_id")

        # Check that the row id values are nonempty and there are no duplicates else raise valueerror
        if df["row_id"].isna().any() or df["row_id"].duplicated().any():
            raise ValueError(f"{name}: invalid row_id")

    # Check that no user is in both train and out of sampe dataset (Would be data leakage)
    if set(train["user_id"]) & set(holdout["user_id"]):
        raise ValueError("Data leakage: users overlap in train and out of sample")

    # Check that columns are the same in training and out of sample datasets
    if set(train.columns) != set(holdout.columns):
        raise ValueError("Train and holdout columns differ")

    # Only keep the features that we didnt exclude (No use for them as predictors like user id)
    features = [c for c in train.columns if c not in EXCLUDE]

    # Remove constant or entirely missing features using training data only
    removed = [c for c in features if train[c].nunique(dropna=True) <= 1]

    # keep what we didnt exclude
    features = [c for c in features if c not in removed]

    # Copy the datasets with the features we kept
    X_train = train[features].copy()
    X_holdout = holdout[features].copy()

    # Replace infinity with NaN
    for X in (X_train, X_holdout):

        # Select all the columns of features that contain numerica values to check for infinity
        numeric = X.select_dtypes(include="number").columns

        # Replace all the +- infinite values with NaN
        X[numeric] = X[numeric].replace([np.inf, -np.inf], np.nan)

    # Check supported feature typess
    supported = X_train.select_dtypes(include=["number", "object", "string", "category", "bool"]).columns

    # If we find a datatype that is not supported raise valueerror
    if set(features) != set(supported):
        raise ValueError(f"Unsupported dtypes: {set(features) - set(supported)}")

    # Report missing values (no imputation)
    missing = X_train.isna().mean()

    # How many features is the validated complete dataset 
    print(f"Features retained: {len(features)}")

    # Numer of constant feautures that were removed
    print(f"Constant features removed: {len(removed)}")

    # Featurs that have missing values
    print(f"Features with missing values: {(missing > 0).sum()}")

    # Return the validated training and out of sample datasets, the features we kept and the target values for both data sets
    return X_train, X_holdout, train["buy"].copy(), holdout["buy"].copy(), features


