'''
Final LightGBM model -> submissions/submission_lightGBM.csv
Run from the project root, after python scripts/feature_engineering.py:   python src/train_lightGBM.py

Where it comes from: this is the code of notebooks/03_modeling_constandino.ipynb, without the plots
(the plots and the explanation of each step are in the notebook). Same process:
1. Start from my 30 features (notebooks 01 + 02) with LightGBM.
2. Test Paul's features one by one on top of mine (CV log-loss), then add the winners one at a time and keep
   a feature only if it still helps on top of the ones already kept -> 6 kept.
3. Switch to the "small trees" settings and add the 3 recency features.
4. Choose the F1 cut-off by cross-validation, check on the holdout users, train on all 6,000 users, predict the test set.
Only difference with the notebook: there my features and Paul's came from two separate files, here they all come
from the common files made by scripts/feature_engineering.py.

AI disclosure: the idea of step 2 (screening Paul's features with a quicker version of the model, 150 trees at
learning rate 0.1, then forward selection) and its code were suggested by an AI assistant (see notebook 03).
'''

# import necessities
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
RUN_SELECTION = True     # True = test Paul's features one by one (about 15 minutes), False = use the 6 kept below

# my 30 features (notebooks 01 + 02)
my_features = ['age', 'education', 'spending_capacity_band', 'home_city', 'stated_favourite_genre', 'past_purchase_count',
               'first_timer', 'typical_ticket_spend', 'usual_booking_lead_days', 'artist_name', 'ticket_price', 'discount_pct',
               'event_page_visits_7d', 'wishlisted', 'music_app_status', 'visits_missing', 'searched_travel', 'price_tier',
               'price_vs_usual', 'lead_gap', 'distance_km_home_to_venue', 'same_city', 'genre_match', 'travel_to_venue',
               'travel_gap_days', 'travel_date_match', 'artist_saves', 'artist_searches', 'artist_days_since_last', 'artist_share']
# the 3 recency features (end of notebook 03)
rec_cols = ['artist_recent_share', 'artist_plays_last_14d', 'artist_trend']

data = pd.read_csv(ROOT / "data/processed/train_features.csv", low_memory=False)
test = pd.read_csv(ROOT / "data/processed/test_features.csv", low_memory=False)

# same fills as my notebook 02 (the common file keeps the gaps)
for df in [data, test]:
    df["stated_favourite_genre"] = df["stated_favourite_genre"].fillna("unknown")
    df["event_page_visits_7d"] = df["event_page_visits_7d"].fillna(0)
    df["genre_match"] = df["genre_match"].fillna(0)

# hold out 20% of the users: the model never sees them until the final check
dev_idx, hold_idx = next(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(data, data["buy"], data["user_id"]))
train   = data.iloc[dev_idx].reset_index(drop=True)
holdout = data.iloc[hold_idx].reset_index(drop=True)
y_train   = train["buy"]
y_holdout = holdout["buy"]

X_train   = train[my_features].copy()
X_holdout = holdout[my_features].copy()

# every text column -> category (holdout uses the same levels as train)
text_cols = X_train.select_dtypes(exclude="number").columns
for c in text_cols:
    X_train[c] = X_train[c].astype("category")
    X_holdout[c] = pd.Categorical(X_holdout[c], categories=X_train[c].cat.categories)

# the lightgbm model with my first settings
clf = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1)
cv = GroupKFold(n_splits=5)                     # keep all 5 rows of a user in the same group
thresholds = np.arange(0.10, 0.71, 0.01)

if RUN_SELECTION:
    # 1. test Paul's features one by one on top of mine (log-loss: lower = better), with a quicker version of the model
    screen = clone(clf).set_params(n_estimators=150, learning_rate=0.1)

    raw_cols = pd.read_csv(ROOT / "data/raw/train.csv", nrows=1).columns
    paul_feats = [c for c in data.columns if c not in raw_cols and c not in my_features + rec_cols]   # only what Paul created
    paul = train[paul_feats].copy()
    for c in paul_feats:
        if not pd.api.types.is_numeric_dtype(paul[c]):
            paul[c] = paul[c].astype("category")                       # text -> category for LightGBM

    def cv_logloss(X):
        oof = cross_val_predict(screen, X, y_train, groups=train["user_id"], cv=cv, method="predict_proba")[:, 1]
        return log_loss(y_train, oof)

    base_logloss = cv_logloss(X_train)
    gain = {}
    for c in paul_feats:
        X = X_train.copy()
        X[c] = paul[c].values
        gain[c] = base_logloss - cv_logloss(X)                          # > 0 means Paul's feature helps
    gain = pd.Series(gain).sort_values(ascending=False)
    print("base log-loss:", round(base_logloss, 4))
    print(gain.head(10).round(4))

    # 2. go down the ranking: keep a feature only if it still improves log-loss on top of the ones already kept
    candidates = gain[gain > 0.0005].index.tolist()
    selected, best = [], base_logloss
    X_sel = X_train.copy()
    for c in candidates:
        X_try = X_sel.copy()
        X_try[c] = paul[c].values
        score = cv_logloss(X_try)
        if best - score > 0.0005:
            selected.append(c); best = score; X_sel = X_try
            print("keep", c, "-> log-loss", round(score, 4))
    print(len(selected), "kept:", selected)
else:
    selected = ['app_total_plays', 'app_plays_last_14d', 'app_artist_saved_rate', 'app_total_listening_hours',
                'app_artist_recent_share', 'app_artist_play_share']

# 3. my features + the kept Paul features + the 3 recency features, with the small-trees settings
X_final = X_train.copy()
X_hold_final = X_holdout.copy()
for c in selected + rec_cols:
    X_final[c] = train[c].values
    X_hold_final[c] = holdout[c].values

clf = clone(clf).set_params(n_estimators=2800, learning_rate=0.02, num_leaves=4, min_child_samples=87,
                            colsample_bytree=0.57, subsample=0.84, reg_alpha=0.22, min_split_gain=0.17)
oof_rec = cross_val_predict(clf, X_final, y_train, groups=train["user_id"], cv=cv, method="predict_proba")[:, 1]
f1s_rec = [f1_score(y_train, oof_rec >= t) for t in thresholds]
cut_final = thresholds[np.argmax(f1s_rec)]
print("best F1:", round(max(f1s_rec), 4), "at cut-off", round(cut_final, 2))

# 4. final check on the 1,200 holdout users (never used for any decision)
p_final = clone(clf).fit(X_final, y_train).predict_proba(X_hold_final)[:, 1]
print("holdout F1:", round(f1_score(y_holdout, p_final >= cut_final), 4), "| AUC", round(roc_auc_score(y_holdout, p_final), 4))

# 5. final model: train on all 6,000 users, predict the Kaggle test set, save
X_all  = data[X_final.columns].astype({c: "category" for c in text_cols})
X_test = test[X_all.columns].astype({c: X_all[c].dtype for c in text_cols})
model = clone(clf).fit(X_all, data["buy"])
pred  = (model.predict_proba(X_test)[:, 1] >= cut_final).astype(int)
pd.DataFrame({"row_id": test["row_id"], "buy": pred}).to_csv(ROOT / "submissions/submission_lightGBM.csv", index=False)
print(len(pred), "rows |", X_all.shape[1], "features | predicted buy rate:", round(pred.mean(), 3))
