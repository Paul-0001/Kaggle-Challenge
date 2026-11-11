'''
Script that trains a logistic regression and does feature selection etc.
'''

# Import what needed
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from validate_features import validate_features

# Function that trains the logistic regression
def train_logistic(train, holdout):

    # Common feature validation with function we created before
    X_train, X_test, y_train, _, _ = validate_features(train, holdout)


# Exclude population-learned genre similarities from grouped CV (For now)
    global_similarity = ["app_genre_similarity_max","app_genre_similarity_mean","app_genre_similarity_weighted"]

    X_train = X_train.drop(columns=global_similarity, errors="ignore")
    X_test = X_test.drop(columns=global_similarity, errors="ignore")


    # add all numerical columns from the validated datasets to numeric
    numeric = X_train.select_dtypes(include=["number", "bool"]).columns.tolist()

    # Add all the rest to categorical (Could be education)
    categorical = X_train.columns.difference(numeric).tolist()

    ## Model-specific preprocessing (Cannot generallyse this script for all models so need to do it model specific)

    # Use tool from sklearn to transform our values
    preprocessor = ColumnTransformer([

        # Defina a pipeline with the name num
        ("num", Pipeline([

            # Inputer fills all missing values with the columns median and really important we add an indicator that shows if the value was filled or not. See EDA why.
            ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),

            # Scaler standardizes all the values as we saw in lecture (xi - mü) / std with mü=mean and std=std of the column
            ("scaler", StandardScaler())

        # Add all columns this pipeline has to be applied to (Defined before)
        ]), numeric),

        # Define a new pipeline for the categorical values called cat
        ("cat", Pipeline([

            # We fill each missing categroy values with the value "Missing"
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing", keep_empty_features=True)),

            # Make each category a dummy variable that takes numerical values so the log reg model can use them. Ignore category that rent predefined and set 0.
            ("encoder", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=20))

        # Add all columns this pipeline has to be applied to
        ]), categorical)
    ])

    # Define a bigger pipeline that does the preprocessing, the logistic regression and add a L1 regularization. Good in this case since we have a lot of features and L1 tends to set parameters to 0. See slides.
    pipeline = Pipeline([

        # Apply preporcessing
        ("preprocessing", preprocessor),

        # Apply logistic regressio with L1 regularization
        ("model", LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=3000,random_state=42))])

    # Search for hyperparameters (As Lambda for L1 reg.) Train multiple models with different hyperparameters to find the best one
    search = GridSearchCV(

        # Whole pipeline
        pipeline,

        # Set values for c we want to test (lambda = 1/c). Same as changing our lambda
        param_grid={"model__C": [0.01, 0.1, 1, 10]},

        # Give the models with different hyper parameters score using the area under the curve as seen in lecture to compare them.
        scoring="roc_auc",

        # Make sure all observations from a user are either in training or validation 
        cv=GroupKFold(n_splits=5),

        # How many CPU Kerne can be used (ALL)
        n_jobs=-1
    )

    # Execute the training etc
    search.fit(X_train, y_train, groups=train["user_id"])

    # Find best model with best hyperparameters (Best Pipeline)
    model = search.best_estimator_

    # Get the probabilities for each row for the out of sample dataset
    probabilities = model.predict_proba(X_test)[:, 1]

    # Save the predictions of the model in the form evaluate_models.py needs it to be to compare between models
    predictions = pd.DataFrame({
        "row_id": holdout["row_id"].values,
        "y_true": holdout["buy"].values,
        "y_proba": probabilities
    })

    # get all the names of the features after preprocessing
    names = model.named_steps["preprocessing"].get_feature_names_out()

    # Get all the coefficients from the best model
    coefficients = model.named_steps["model"].coef_[0]

    # Create a df that contains the name of the feature and its given prameter value(coefficient)
    selected = pd.Series(coefficients, index=names)

    # Filter everything out that is really close to 0
    selected = selected[~np.isclose(selected, 0, atol=1e-8)]

    # Give the user the best hyperparameter that was found
    print(f"Best C: {search.best_params_['model__C']}")

    # Number of features that were not set to 0 through L1 reg.
    print(f"Selected features: {len(selected)}/{len(names)}")

    # the pipeline, the out of sample predictions and the nonzero features
    return model, predictions, selected

# Function that manages everything
if __name__ == "__main__":

    # Get the path to the root of the project
    root = Path(__file__).resolve().parents[1]

    # Set path for where the training dataset will be saved
    train = pd.read_csv(root / "data/processed/train_split.csv")

    # Set path where the out of sample dataset will be saved
    holdout = pd.read_csv(root / "data/processed/holdout_split.csv")

    # Do the training an get the results
    model, predictions, selected = train_logistic(train, holdout)

    # Create a new folder where the output of our training will be
    output = root / "models"

    # Create parent folders if they do not already exist
    output.mkdir(exist_ok=True)

    # Save predictions as CSV and save
    predictions.to_csv(output / "logistic_predictions.csv", index=False)

    # Save nonzero features as CSV and save
    selected.rename("coefficient").to_csv(output / "logistic_features.csv")

    # Save the whole model
    joblib.dump(model, output / "logistic_model.joblib")

    # Usercheck that everything worked
    print("Logistic regression training completed.")
