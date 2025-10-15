import argparse

import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main():
    parser = argparse.ArgumentParser(description="Regularization parameter tuning")
    parser.add_argument("--train", help="Training CSV file")
    parser.add_argument("--val", help="Validation CSV file")
    parser.add_argument("--target", type=str, default="converted", help="Target column name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use")
    args = parser.parse_args()

    train = pd.read_csv(args.train)
    val = pd.read_csv(args.val)

    categories = [
        "industry",
        "location",
        "lead_source",
        "employment_status",
    ]

    y_train = train[args.target]
    y_val = val[args.target]

    numerical = [col for col in train.columns if col not in categories and col != args.target]

    encoder = OneHotEncoder()
    encoder.fit(train[categories])

    X_train_cat = pd.DataFrame(
        encoder.transform(train[categories]).toarray(),
        index=train.index,
        columns=encoder.get_feature_names_out(),
    )

    X_val_cat = pd.DataFrame(
        encoder.transform(val[categories]).toarray(),
        index=val.index,
        columns=encoder.get_feature_names_out(),
    )

    # combine numerical and cat together
    X_train = pd.concat([train[numerical], X_train_cat], axis=1)
    X_val = pd.concat([val[numerical], X_val_cat], axis=1)

    results = {}
    for c in [0.01, 0.1, 1, 10, 100]:
        model = LogisticRegression(solver='liblinear', C=c, max_iter=1000, random_state=args.seed)
        # model = LogisticRegression(solver='lbfgs', C=c, max_iter=1000, random_state=args.seed)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        print(f"Accuracy for {c=}: {accuracy}")
        results[c] = accuracy

    max_acc = max(results.values())
    best_cc = [c for c in results if results[c] == max_acc]
    best_c = min(best_cc)
    print(f"Best C: {best_c} with accuracy {max_acc}")


if __name__ == "__main__":
    main()
