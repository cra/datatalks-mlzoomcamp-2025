import argparse

import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder


def main():
    parser = argparse.ArgumentParser(description="Calculate mutual information scores")
    parser.add_argument("train_csv", help="Training CSV file")
    parser.add_argument("--target", type=str, default="converted", help="Target column name")
    args = parser.parse_args()

    df = pd.read_csv(args.train_csv)

    y = df[args.target]

    categories = [
        "industry",
        "location",
        "lead_source",
        "employment_status",
    ]
    X = df.drop(columns=[args.target])[categories].apply(LabelEncoder().fit_transform)

    scores = mutual_info_classif(X, y, discrete_features=True)
    mi_scores = {cat: round(score, 2) for cat, score in zip(categories, scores)}

    w = max(mi_scores, key=mi_scores.get)
    print(f"Max MI score for {w} ({mi_scores[w]})")


if __name__ == "__main__":
    main()
