#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd
import rich
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split


def q5_meat(df_full_train, target, C):
    kf = KFold(n_splits=5, shuffle=True, random_state=1)
    auc_scores = []
    for train_idx, val_idx in kf.split(df_full_train):
        df_train = df_full_train.iloc[train_idx].copy()
        df_val = df_full_train.iloc[val_idx].copy()

        y_train = df_train.pop(target).values
        y_val = df_val.pop(target).values

        dicts = df_train.to_dict(orient="records")
        dv = DictVectorizer()
        X_train = dv.fit_transform(dicts)

        model = LogisticRegression(solver="liblinear", C=C, max_iter=1000)
        # model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=10_000)
        model.fit(X_train, y_train)

        dicts = df_val.to_dict(orient="records")
        X_val = dv.transform(dicts)
        y_pred = model.predict_proba(X_val)[:, 1]

        auc_scores.append(roc_auc_score(y_val, y_pred))

    return auc_scores


def main():
    parser = argparse.ArgumentParser(description="Q5: 5-Fold CV")
    parser.add_argument("--csv", help="Original CSV file")
    parser.add_argument("--target", type=str, default="converted", help="Target column name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed")
    parser.add_argument("--q5", action="store_true")
    parser.add_argument("--q6", action="store_true")
    args = parser.parse_args()

    if not (args.q5 or args.q6):
        raise "Need at least one of --q5 or --q6 flags"

    df = pd.read_csv(args.csv)
    categorical_features = ["industry", "location", "lead_source", "employment_status"]
    for cat in categorical_features:
        if cat in df.columns:
            df[cat] = df[cat].fillna("NA")
    numerical_features = [col for col in df.columns if col not in categorical_features and col != args.target]
    df[numerical_features] = df[numerical_features].fillna(0.0)

    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=args.seed)

    if args.q5:
        auc_scores = q5_meat(df_full_train, target=args.target, C=1.0)
        print(f"AUC scores: {auc_scores}")
        print(f"Mean: {np.mean(auc_scores):.3f}")
        print(f"Std: {np.std(auc_scores, ddof=1):.3f}")

    if args.q6:
        for C in [0.000001, 0.001, 1]:
            auc_scores = q5_meat(df_full_train, target=args.target, C=C)
            rich.print(f"== FOR {C=}")
            rich.print(f"Mean: {np.mean(auc_scores):.3f}")
            rich.print(f"Std: {np.std(auc_scores, ddof=1):.3f}")


if __name__ == "__main__":
    main()
