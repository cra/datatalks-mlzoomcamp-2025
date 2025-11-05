import argparse
import math

import numpy as np
import pandas as pd
from rich.progress import Progress
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
    n_step = 10
    d_vars = [10, 15, 20, 25]
    n_vars = list(range(10, 200 + n_step, n_step))
    total = len(d_vars) * len(n_vars)
    with Progress() as progress:
        w = progress.add_task("Computing", total=total)
        for d in [10, 15, 20, 25]:
            model = RandomForestRegressor(
                random_state=seed,
                n_jobs=-1,
                max_depth=d,
                warm_start=True,
            )
            vals = []
            for n in range(10, 200 + n_step, n_step):
                model.n_estimators = n
                model.fit(X_train, y_train)

                y_pred = model.predict(X_val)

                mse = mean_squared_error(y_val, y_pred)
                rmse = math.sqrt(mse)
                vals.append(rmse)
                progress.update(w, advance=1)
                # print(d, n, rmse)
            bleh[d] = np.mean(vals)
            print(f"max_depth={d} yields mean(rmse)={bleh[d]}")

    print("Min rmse would be at max_depth =", min(bleh, key=bleh.get))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val", required=True, help="Path to validation CSV")
    parser.add_argument("--target", required=True, type=str, help="target column")
    parser.add_argument("--seed", required=True, type=int, help="random seed to use")
    args = parser.parse_args()

    main(args.train, args.val, args.target, args.seed)
