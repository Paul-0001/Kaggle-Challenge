# DSBA Kaggle Challenge — Concert Ticket Purchase Prediction

Foundations of Machine Learning · CentraleSupélec · Kaggle team **Paul & Dino** (Paul Gehlen, Constandino Hadjiyannis)

Predict `buy` (1 if the user buys a ticket in the 7-day presale starting 2026-10-01) for each user × concert row,
using only information available at the prediction snapshot (2026-10-01 00:00 UTC). Metric: binary F1.
The 4,000 test users never appear in train, so every validation step splits by user.

## Results

| Model | Features | CV F1 (5-fold GroupKFold) | Cut-off | Kaggle public F1 |
|---|---|---|---|---|
| Logistic regression (L1) | 145 candidates | 0.8041 | 0.42 | 0.79918 |
| CatBoost | 145 candidates | 0.8444 | 0.43 | 0.83234 |
| **LightGBM (final)** | 30 EDA features + 6 selected + 3 recency | **0.8455** | 0.44 | **0.84075** |

The report (`Report_Paul_Dino.pdf` on Gradescope, long version `Report_Paul_Dino.docx`) explains every step.
The mathematical background of the candidate features and models is in `Math.pdf`.

## Setup

Mac / Linux:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Put the three Kaggle files in `data/raw/`: `train.csv`, `test.csv`, `music_app.json`.
All commands below are run from the project root.

## 1. Reproduce the final submission (LightGBM, Kaggle 0.84075)

```bash
python scripts/feature_engineering.py
python src/train_lightGBM.py
```

1. `scripts/feature_engineering.py` (about 9 minutes) builds all features: the 145 candidates and our 16 EDA-driven
   columns. It writes `data/processed/train_features.csv` and `data/processed/test_features.csv`.
2. `src/train_lightGBM.py` (about 10–15 minutes) is the code of `notebooks/03_modeling_constandino.ipynb` without the plots:
   feature selection (screening + forward selection), small-trees LightGBM, the 3 recency features, the F1 cut-off chosen
   by cross-validation, a holdout check, then a final fit on all 6,000 users.
   It writes `submissions/submission_lightGBM.csv`, identical to our best Kaggle submission.
   Set `RUN_SELECTION = False` at the top of the script to reuse the 6 selected features (about 1 minute).

Expected last lines:
```
6 kept: ['app_total_plays', 'app_plays_last_14d', 'app_artist_saved_rate', 'app_total_listening_hours', 'app_artist_recent_share', 'app_artist_play_share']
best F1: 0.8455 at cut-off 0.44
holdout F1: 0.853 | AUC 0.9637
20000 rows | 39 features | predicted buy rate: 0.348
```

## 2. Reproduce the model comparison (logistic regression, CatBoost, LightGBM)

The logistic regression and CatBoost scripts import from `scripts/`, so the project root must be on the Python path.

Mac / Linux:
```bash
python scripts/feature_engineering.py        # skip if already done
python scripts/split_data.py
python src/train_lightGBM.py
PYTHONPATH=. python src/train_logistic_regression.py
PYTHONPATH=. python src/train_catboost.py
python scripts/evaluate_models.py
```

Windows PowerShell:
```powershell
python scripts/feature_engineering.py        # skip if already done
python scripts/split_data.py
python src/train_lightGBM.py
$env:PYTHONPATH = "."
python src/train_logistic_regression.py
python src/train_catboost.py
python scripts/evaluate_models.py
```

- `scripts/split_data.py` splits the training users 80/20 by user (`data/processed/train_split.csv`, `holdout_split.csv`).
  Run it **before** `train_lightGBM.py`, otherwise LightGBM does not save its holdout predictions.
- `src/train_logistic_regression.py` and `src/train_catboost.py` print `OOF F1 (optimized threshold)`, the CV F1 in the table above,
  and save their model and holdout predictions in `models/`.
- `scripts/evaluate_models.py` must run **last**: it compares the three models on the same holdout
  (accuracy, AUC, precision, recall, F1, log-loss) and saves `models/model_comparison.csv`.
- Optional: `scripts/find_optimal_threshold_logistic_regression.py` lists the F1 of logistic regression for every cut-off on the holdout.
- Kaggle files for the two other models: after training, `python submissions/submission_logistic_regression.py` and
  `python submissions/submission_catboost.py` write `submission_logistic_regression.csv` and `submission_catboost.csv` in the project root.

Logistic regression and CatBoost use the 145 candidates only (as for their Kaggle submissions): `EXCLUDE` in
`scripts/validate_features.py` removes our 16 EDA-driven columns (`EDA_COLUMNS`) from their inputs.

## Project layout

- `data/raw/`: the Kaggle files (not tracked)
- `data/processed/`: generated feature files and splits (not tracked)
- `notebooks/`: the analysis, step by step with plots
  - `01_eda_constandino.ipynb`, `01_eda_paul.ipynb`: EDA, cleaning and data-quality checks
  - `02_feature_engineering_constandino.ipynb`: our 30 EDA-driven features
  - `02_feature_engineering_paul.ipynb`: the 145 candidate features
  - `03_modeling_constandino.ipynb`: LightGBM, feature selection, tuning, recency features, final model
- `scripts/`: feature engineering, user-grouped split, feature validation, model evaluation, threshold search
- `src/`: one training script per model (`train_lightGBM.py`, `train_logistic_regression.py`, `train_catboost.py`)
- `submissions/`: Kaggle submission files and the scripts that make the LR and CatBoost submissions
- `models/`: trained models and holdout predictions (created by the training scripts, not tracked)
- `Math.pdf`: derivations behind the candidate features and models
- `DSBA-ML-Kaggle_Challenge.pdf`: official instructions

## Validation and leakage

- Every split is by user (`GroupKFold`, `StratifiedGroupKFold`, `GroupShuffleSplit`), because a user's 5 rows are linked and test users are new.
- Learned preprocessing is fitted inside cross-validation; every choice (features, settings, cut-off) is made on out-of-fold predictions.
- Music-app plays after the prediction cut-off are removed, and no feature uses the target.
- Fixed random seeds (42) everywhere.

## Use of AI

An AI assistant helped with improving LightGBM (screening the candidate features and tuning the model) and with the code
moving our notebook features into `scripts/feature_engineering.py`, so that the whole pipeline can be reproduced from scripts.
We ran and checked all of it (identical values and results). Both are also disclosed in comments in the code.
