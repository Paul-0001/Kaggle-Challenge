# DSBA Kaggle Challenge — Concert Ticket Purchase Prediction

Foundations of Machine Learning · CentraleSupélec · Deadline: 2026-11-29 23:00

## Objective
Predict binary `buy` for each user–concert opportunity using only information available at the prediction snapshot (2026-10-01 00:00 UTC).

## Project layout
- `data/raw/`: original Kaggle files (not tracked)
- `data/processed/`: generated datasets (not tracked)
- `notebooks/`: EDA and exploratory feature experiments
- `src/` : Logistic Regression, CatBoost and LightGBM Training
- `scripts/`: Scripts to evaluate the models, do the feature engineering, split the data, validate our feature engineering and find an optimal threshold
- `submissions/`: Kaggle submission CSVs
- `main/`: pdf for project explanations, mathematical proofs and official instructions

## Getting started (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Place Kaggle data in `data/raw/` after downloading it. Data loading and model scripts are placeholders until the real schema and competition metric are confirmed.

## Validation plan
Determine whether user-grouped folds are appropriate; use the official competition metric. Fit all learned preprocessing inside cross-validation to prevent leakage. Never use post-snapshot observations.

## Reproducibility
Use a fixed random seed and record model parameters, CV results and library versions. Final submission should be reproducible from provided data and source code.
