import argparse
import sys

import pandas as pd
import rich
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Split data into train/val/test sets")
    parser.add_argument("csv_file", help="Input CSV file")
    parser.add_argument("--seed", type=int, default=1, help="Random seed to use")
    parser.add_argument("--target", type=str, default="converted", help="Target column name")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_file)
    tot_lines = len(df)

    # Handle missing values according to homework instructions
    categorical_features = ["industry", "location", "lead_source", "employment_status"]

    # when doing `pd.read_csv`, 'NA' gets treated as null,
    # so instead I save it as another value
    for cat in categorical_features:
        if cat in df.columns:
            df[cat] = df[cat].fillna("MISSING")

    # For numerical features, replace with 0.0
    numerical_features = [col for col in df.columns if col not in categorical_features and col != args.target]
    df[numerical_features] = df[numerical_features].fillna(0.0)

    y = df[args.target]
    X = df.drop(columns=[args.target])

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=args.seed,
    )

    # 20% is 75% of (100-20)%
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=0.25,
        random_state=args.seed,
    )

    out = {}

    out["train"] = pd.concat([X_train, y_train], axis=1)
    out["val"] = pd.concat([X_val, y_val], axis=1)
    out["test"] = pd.concat([X_test, y_test], axis=1)

    base_name = args.csv_file.replace(".csv", "")

    rich.print(f"60/20/20 Split of {args.csv_file} complete! (seed={args.seed})")
    for tag, df in out.items():
        fname = f"{base_name}_{tag}.csv"
        df.to_csv(fname, index=False)
        rows = df.shape[0]
        rich.print(
            "=" * 30,
            f"{tag}_df saved as {fname}.",
            f"Its shape: {df.shape}",
            f"that's {rows / tot_lines * 100:.2f}% from total rowcount",
            sep="\n  ",
        )


if __name__ == "__main__":
    main()
