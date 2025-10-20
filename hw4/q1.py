#!/usr/bin/env python3

import argparse

import rich
import pandas as pd
from sklearn.metrics import roc_auc_score


def main():
    parser = argparse.ArgumentParser(description="Feature elimination analysis")
    parser.add_argument("--train", help="Training CSV file")
    parser.add_argument("--target", type=str, default="converted", help="Target column name")
    parser.add_argument("--seed", type=int, default=1, help="Random seed to use")
    args = parser.parse_args()

    df_train = pd.read_csv(args.train)
    print(f"train data shape: {df_train.shape}")

    y_train = df_train[args.target]

    print("ROC AUC scores for numerical features:")
    print("=" * 60)

    auc_scores = {}

    for feature in ["lead_score", "number_of_courses_viewed", "interaction_count", "annual_income"]:
        feature_values = df_train[feature]

        auc = roc_auc_score(y_train, feature_values)

        if auc < 0.5:
            feature_values = -df_train[feature]
            auc = roc_auc_score(y_train, feature_values)
            print(f"{feature:30s}: {auc:.4f} (inverted)")
        else:
            print(f"{feature:30s}: {auc:.4f}")

        auc_scores[feature] = auc

    print("=" * 60)
    best_feature = max(auc_scores, key=auc_scores.get)
    rich.print(f"\n[bold]{best_feature=}[/bold] has highest AUC = {auc_scores[best_feature]:.4f}")


if __name__ == "__main__":
    main()
