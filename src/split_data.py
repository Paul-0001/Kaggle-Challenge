'''
Split our training data 80-20.
80% for train and 20% for out of sample cross validation
30.000 obs -> 5 concert offers per user -> 6.000 users -> 4.800-1.200 split
'''

# Import what needed
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Function that splits our dataset and groups all observations belonging to one user together. Set seed for reproducability,
def split_by_user(df, test_size=0.20, random_state=42):

    # Set required columns as dict
    required = {"user_id", "buy"}

    # If these required columns are not insde we cannot continue
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    # Check if every observation has a user id
    if df["user_id"].isna().any():
        raise ValueError("Missing user_id values found.")

    # Initialize grouped random split
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size,random_state=random_state)

    # Obtain row indices for each subbset
    train_idx, holdout_idx = next(splitter.split(df, y=df["buy"], groups=df["user_id"]))

    # Create independent datasets
    train = df.iloc[train_idx].copy().reset_index(drop=True)
    holdout = df.iloc[holdout_idx].copy().reset_index(drop=True)

    # Check that no user occurs in both datasets
    train_users = set(train["user_id"])
    holdout_users = set(holdout["user_id"])

    assert train_users.isdisjoint(holdout_users)

    # Usercheck in terminal for training and out of sample datasets.
    print(f"Training: {len(train):,} rows, {len(train_users):,} users")
    print(f"Holdout:  {len(holdout):,} rows, {len(holdout_users):,} users")
    print(f"Training positive rate: {train['buy'].mean():.3f}")
    print(f"Holdout positive rate:  {holdout['buy'].mean():.3f}")

    # Return both datasets
    return train, holdout
