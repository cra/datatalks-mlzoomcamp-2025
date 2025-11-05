import argparse
import math

import pandas as pd
from rich.progress import track
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


def main(train_file, val_file, target, seed):
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)

    y_train = df_train.pop(target)
    X_rows = df_train.to_dict(orient='records')
    y_val = df_val.pop(target)
    X_val = df_val.to_dict(orient='records')

    v = DictVectorizer(sparse=True)

    X_train = v.fit_transform(X_rows)
    X_val = v.transform(X_val)

    bleh = {}
    step = 10
    model = RandomForestRegressor(
        random_state=seed,
        n_jobs=-1,
        warm_start=True,
    )
    for n in track(range(10, 200 + step, step)):
        model.n_estimators = n
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)

        mse = mean_squared_error(y_val, y_pred)
        bleh[n] = round(math.sqrt(mse), 3)
        # print(n, bleh[n])

    print(bleh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val", required=True, help="Path to validation CSV")
    parser.add_argument("--target", required=True, type=str, help="target column")
    parser.add_argument("--seed", required=True, type=int, help="random seed to use")
    args = parser.parse_args()

    main(args.train, args.val, args.target, args.seed)
