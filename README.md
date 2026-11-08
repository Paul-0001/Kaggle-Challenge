# DSBA Kaggle Challenge — Concert Ticket Purchase Prediction

Foundations of Machine Learning · CentraleSupélec · Deadline: 2026-11-29 23:00

## Objective
Predict binary `buy` for each user–concert opportunity using only information available at the prediction snapshot (2026-10-01 00:00 UTC).

## Project layout
- `data/raw/`: original Kaggle files (not tracked)
- `data/processed/`: generated datasets (not tracked)
- `notebooks/`: EDA and exploratory feature experiments
- `src/data/`: loading, cleaning and feature engineering
- `src/preprocessing/`: model-specific sklearn preprocessing pipelines
- `src/models/`: classifiers and parameter grids
- `src/evaluation/`: validation, scoring and comparisons
- `scripts/`: reproducible command-line entry points
- `results/`: scores, plots and tuning summaries
- `submissions/`: Kaggle submission CSVs
- `reports/`: final 3-page report
- `tests/`: automated tests

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

## Team workflow
Create feature branches, open pull requests, review before merging to `main`. Avoid committing Kaggle data, secrets, `.venv` or large generated artifacts.

## Submission checklist
- [ ] Kaggle team submission
- [ ] 3-page PDF report with team members and Kaggle team name
- [ ] ZIP of reproducible code sent as instructed in assignment
