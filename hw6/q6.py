import argparse
import math

import rich
import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error


def main(train_file, val_file, target, seed):
    # Load the data
    df_train = pd.read_csv(train_file)
    df_val = pd.read_csv(val_file)

    y_train = df_train.pop(target)
    X_rows = df_train.to_dict(orient='records')
    y_val = df_val.pop(target)
    X_val = df_val.to_dict(orient='records')

    v = DictVectorizer(sparse=True)

    X_train = v.fit_transform(X_rows)
    X_val = v.transform(X_val)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    watchlist = [(dtrain, 'train'), (dval, 'validation')]

    bleh = {}
    for eta in (0.1, 0.3):
        xgb_params = {
            'eta': eta,
            'max_depth': 6,
            'min_child_weight': 1,
            'objective': 'reg:squarederror',
            'nthread': 8,
            'seed': seed,
            'verbosity': 1,
        }

        model = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=100,
            evals=watchlist,
        )

        y_pred = model.predict(dval)

        mse = mean_squared_error(y_val, y_pred)
        rmse = math.sqrt(mse)
        bleh[eta] = rmse
        print(f"{eta=} gives {rmse=}")

    rich.print(bleh)
    print("Minimal RMSE at eta=", min(bleh, key=bleh.get))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True, help="Path to training CSV")
    parser.add_argument("--val", required=True, help="Path to validation CSV")
    parser.add_argument("--target", required=True, type=str, help="target column")
    parser.add_argument("--seed", required=True, type=int, help="random seed to use")
    args = parser.parse_args()

    main(args.train, args.val, args.target, args.seed)
