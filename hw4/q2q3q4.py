#!/usr/bin/env python3

import argparse
from dataclasses import dataclass
from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


@dataclass(frozen=True)
class Score:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self):
        return self.tp / (self.tp + self.fp)

    @property
    def recall(self):
        return self.tp / (self.tp + self.fn)

    @property
    def f1(self):
        p = self.precision
        r = self.recall
        if p + r == 0:
            return 0
        return 2 * (p * r) / (p + r)


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
    parser.add_argument("--q2", action="store_true")
    parser.add_argument("--q3", action="store_true")
    parser.add_argument("--q4", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not (args.q2 or args.q3 or args.q4):
        raise "Need at least one of --q2, --q3 or --q4 flags"

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

    if args.debug:
        print(f"Number of features after encoding: {X_train.shape[1]}")
        print(f"Sample of feature names: {v.feature_names_[:20]}")
        print(f"X_train sample (first row): {X_train[0][:20]}")

        debug(X_train, X_val, v)

    # model = LogisticRegression(solver="liblinear", C=1.0, max_iter=1000)
    model = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
    model.fit(X_train, y_train)

    if args.q2:
        y_pred = model.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)
        print(f"got {auc=}")

    y_pred_proba = model.predict_proba(X_val)[:, 1]

    thresholds = np.arange(0.0, 1.01, 0.01)
    scores = []
    P_hat = y_val == 1
    N_hat = y_val == 0
    for t in thresholds:
        P_pred = y_pred_proba >= t
        N_pred = y_pred_proba < t
        score = Score(
            threshold=t,
            tp=(P_hat & P_pred).sum(),
            fp=(P_pred & N_hat).sum(),
            fn=(N_pred & P_hat).sum(),
            tn=(N_hat & N_pred).sum(),
        )
        scores.append(score)

    if args.q3:
        for s in scores:
            if abs(s.precision - s.recall) < 0.01:
                print(f"Intersection at threshold {s.threshold:.3f}: P={s.precision:.3f}, R={s.recall:.3f}")

        plt.plot([s.threshold for s in scores], [s.precision for s in scores], label="precision")
        plt.plot([s.threshold for s in scores], [s.recall for s in scores], label="recall")
        plt.vlines(0.145, 0, 1, color="grey", linestyle="--", alpha=0.5)
        plt.vlines(0.345, 0, 1, color="grey", linestyle="--", alpha=0.5)
        plt.vlines(0.545, 0, 1, color="grey", linestyle="--", alpha=0.5)
        plt.vlines(0.745, 0, 1, color="grey", linestyle="--", alpha=0.5)

        plt.legend()
        plt.savefig("q3.png")
        print("See q3.png")

    if args.q4:
        f1 = 0
        max_t = 0
        for s in scores:
            if s.f1 > f1:
                f1 = s.f1
                max_t = s.threshold
        print(f"F1={f1} is biggest for threshold of {max_t}")
        plt.plot([s.threshold for s in scores], [s.f1 for s in scores], label="F1")
        plt.vlines(0.14, 0, 1, color="grey", linestyle="--", alpha=0.5)
        plt.vlines(0.34, 0, 1, color="grey", linestyle="--", alpha=0.5)
        plt.vlines(0.54, 0, 1, color="grey", linestyle="--", alpha=0.5)
        plt.vlines(0.74, 0, 1, color="grey", linestyle="--", alpha=0.5)

        plt.legend()
        plt.savefig("q4.png")
        print("See q4.png")


if __name__ == "__main__":
    main()
