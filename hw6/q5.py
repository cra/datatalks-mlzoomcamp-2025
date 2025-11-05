import argparse
import math

import rich
import numpy as np
import pandas as pd
from rich.progress import Progress
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


def main(train_file, target, seed):
    df_train = pd.read_csv(train_file)

    y_train = df_train.pop(target)
    X_rows = df_train.to_dict(orient='records')

    v = DictVectorizer(sparse=True)

    X_train = v.fit_transform(X_rows)

    model = RandomForestRegressor(
        n_estimators=10,
        max_depth=20,
        random_state=seed,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    bleh = dict(zip(v.feature_names_, model.feature_importances_))
    cum_bleh = {
        f: sum(g for k, g in bleh.items() if k.startswith(f))
        for f in [
            "vehicle_weight",
            "horsepower",
            "acceleration",
            "engine_displacement",
        ]
    }
    rich.print(cum_bleh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--target", required=True, type=str, help="target column")
    parser.add_argument("--seed", required=True, type=int, help="random seed to use")
    args = parser.parse_args()

    main(args.train, args.target, args.seed)
