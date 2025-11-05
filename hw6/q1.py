import pandas as pd
import argparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.feature_extraction import DictVectorizer


def main(train_file, target, seed):
    df_train = pd.read_csv(train_file)

    y_train = df_train.pop(target)
    X_rows = df_train.to_dict(orient='records')

    v = DictVectorizer(sparse=True)
    X = v.fit_transform(X_rows)

    model = DecisionTreeRegressor(max_depth=1, random_state=seed)

    model.fit(X, y_train)
    most_important = model.feature_importances_.argmax()
    print(v.feature_names_[most_important])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--target", required=True, type=str, help="target column")
    parser.add_argument("--seed", required=True, type=int, help="random seed to use")
    args = parser.parse_args()

    main(args.train, args.target, args.seed)
