#!/usr/bin/env python3

import argparse

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction import DictVectorizer

from sklearn.model_selection import train_test_split


def debug(X_train, X_val, v):
    print(f"NaN in X_train: {np.isnan(X_train.data).sum()}")
    print(f"NaN in X_val: {np.isnan(X_val.data).sum()}")
    # If you find NaN, see which columns:
    if np.isnan(X_train.data).any():
        nan_cols = np.where(np.isnan(X_train.data).any(axis=0))[0]
        print(f"Columns with NaN: {nan_cols}")
        print(f"Feature names: {[v.feature_names_[i] for i in nan_cols]}")


def main():
    parser = argparse.ArgumentParser(description="Feature elimination analysis")
    parser.add_argument("--train", help="Training CSV file")
    parser.add_argument("--val", help="Validation CSV file")
    parser.add_argument("--target", type=str, default="converted", help="Target column name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed to use")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train)  # keep_default_na=False, na_values=[""])
    df_train = df_train.reset_index(drop=True)
    print(f"train data shape: {df_train.shape}")
    df_val = pd.read_csv(args.val)  # keep_default_na=False, na_values=[""])
    df_val = df_val.reset_index(drop=True)
    print(f"validation data shape: {df_val.shape}")

    y_train = df_train.pop(args.target)
    y_val = df_val.pop(args.target)

    train_dicts = df_train.to_dict(orient="records")
    val_dicts = df_val.to_dict(orient="records")

    v = DictVectorizer()
    X_train = v.fit_transform(train_dicts)
    X_val = v.transform(val_dicts)

    # print(f"Number of features after encoding: {X_train.shape[1]}")
    # print(f"Sample of feature names: {v.feature_names_[:20]}")
    # print(f"X_train sample (first row): {X_train[0][:20]}")

    debug(X_train, X_val, v)
    model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print(f"got {auc=}")


if __name__ == "__main__":
    main()
