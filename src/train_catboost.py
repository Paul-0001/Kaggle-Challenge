'''
Script that trains a catBoost model. Here with direct F1 optimization to go up in leaderboard after log regression.
'''

# Import what needed
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import f1_score
from scripts.validate_features import validate_features

# Function that trains aour model
def train_catboost(train, holdout):

    # Simple feature validation to make sure everything is in order
    X_train, X_test, y_train, _, _ = validate_features(train, holdout)

    # Get all the feautres which have object, string or category as datatype. (CatBoost-specific)
    categorical = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()

    # For all the previously defined feautres replace the missing values with Missing (Missing is a new category which could be like EDA missing value feautures)
    for X in (X_train, X_test):
        X[categorical] = X[categorical].fillna("Missing").astype(str)

    # Set up our CatBoost model
    model = CatBoostClassifier(

        # Set the loss-function to be log loss (Binary cross-entropy as seen in lecture for log reg)
        loss_function="Logloss",

        # Get live training information
        verbose=True,

        # For reproducability set seed
        random_seed=42,

        # Only use 1 CPU thred per training (Because of after to avoid errors)
        thread_count=1,

        # During training no files are written, this is done later
        allow_writing_files=False
    )

    # We train on 4 folds an validate on fifth (This is our data structure else data leakage)
    cv = GroupKFold(n_splits=5)

    # Go through many trainings
    search = RandomizedSearchCV(
        model,

        # Define our search space for the iterations
        param_distributions={

            # Depth of the decision trees
            "depth": [4, 5, 6, 7, 8],

            # Parameter that defines how much a new tree changes the prediction
            "learning_rate": [0.01, 0.03, 0.05, 0.1],

            # Define how many trees are created one after another
            "iterations": [300, 500, 800, 1200],

            # Regularization. Similar to L2 regularization as seen in lecture
            "l2_leaf_reg": [1, 3, 5, 10, 20],

            # Parameters that increase the noise when choosing for a new split to make it more random
            "random_strength": [0.1, 1, 3, 5]
        },

        # Choose 30 of the 5*4*4*5*4=1.600 combinations (Random)
        n_iter = 30,

        # Which score we are trying to optimize. Since in leaderboard the F1 matters we opt F1
        scoring = "f1",

        # Group K-foold object is given we defined earlier
        cv = cv,

        # Use all CPU kernels for training (Parallel-> thats why before we only said 1 so they do not create problems)
        n_jobs= -1,

        # Set seed
        random_state = 42,

        # Activate live checks for RandomnizedSearchCV
        verbose =1
    )

    # Get all user id from training dataset (Same user not in train and test!!!!)
    groups = train["user_id"]

    # Start with the real training now that everything is defined
    search.fit(X_train,y_train, groups=groups,cat_features=categorical)

    # Usercheck for best hyperparameters we defined in randomnized search cv
    print("\nBest parameters:", search.best_params_)

    # F1 score for these best hyperparameters and a threshold of 0,5
    print(f"Best CV F1 (threshold 0.5): {search.best_score_:.4f}")

    # make a vector of 0 that will have all the probabilities computed by our model on non-seen data
    oof_probabilities = np.zeros(len(X_train))

    # Go through our 5 fold validation folds
    for train_idx, val_idx in cv.split(X_train, y_train, groups):

        # Clone the best model but untrained
        fold_model = clone(search.best_estimator_)

        # train the model on the respective observations
        fold_model.fit(
            X_train.iloc[train_idx],
            y_train.iloc[train_idx],
            cat_features=categorical
        )

        # Get a probability for each observation
        oof_probabilities[val_idx] = fold_model.predict_proba(X_train.iloc[val_idx])[:, 1]

    # Set all thesholds we want to test
    thresholds = np.arange(0.20, 0.81, 0.01)

    # Get F1 score for the model with the different thresholds
    scores = [f1_score(y_train, (oof_probabilities >= t).astype(int))for t in thresholds]

    # Get the treshold that maximizes our F1 Score
    best_threshold = float(thresholds[np.argmax(scores)])

    # Usercheck for best threshold (After running we see 0,42 which is nearly the same as opt one for log reg which was 0,4. So maybe 0,5 not the best even if standard in literature)
    print(f"Best threshold: {best_threshold:.2f}")

    # F1 score for best hyperparameters and best threshold
    print(f"OOF F1 (optimized threshold): {max(scores):.4f}")

    # Set the best model with the results from before
    best_model = search.best_estimator_

    # Get the probabilities of the best ,odel for the out of same dataset
    probabilities = best_model.predict_proba(X_test)[:, 1]

    # Get the predictions in submission format
    predictions = pd.DataFrame({
        "row_id": holdout["row_id"].values,
        "y_true": holdout["buy"].values,
        "y_proba": probabilities
    })

    # Return the metrics of the best model, its predictions and the best threshold for our out of sample data
    return best_model, predictions, best_threshold

# Coordinates everything
if __name__ == "__main__":

    # Set root path of prject
    root = Path(__file__).resolve().parents[1]

    # Get our train dataset input for the train_catboost function
    train = pd.read_csv(root / "data/processed/train_split.csv")

    # Get our out of sample data for the train_catboost function
    holdout = pd.read_csv(root / "data/processed/holdout_split.csv")

    # Get the output of our train_catboost function
    model, predictions, threshold = train_catboost(train, holdout)

    # Set the folder where the output will be saved
    output = root / "models"

    # If parent folders dont exist create them
    output.mkdir(exist_ok=True)

    # Save the predictions of catboost in CSV
    predictions.to_csv(output / "catboost_predictions.csv", index=False)

    # Save the model metrics 
    model.save_model(str(output / "catboost_model.cbm"))

    # Save optimized threshold
    (output / "catboost_threshold.txt").write_text(str(threshold))

    # Save the feautures that are important for our model in ascending order. So we can check what really matters for further improvements
    importance = pd.Series(model.feature_importances_,index=model.feature_names_).sort_values(ascending=False)

    # Save the important features in csv and save
    importance.rename("importance").to_csv(output / "catboost_feature_importance.csv")

    # Usercheck that training was succesfully finished
    print("\nCatBoost training completed.")
