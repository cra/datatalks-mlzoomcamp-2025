import sys

import numpy as np
import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: python q7.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    masq = df["origin"] == "Asia"
    features = ["vehicle_weight", "model_year"]
    y = [1100, 1300, 800, 900, 1000, 1100, 1200]

    X = df[masq][features][:7].to_numpy()
    XTX_inv = np.linalg.inv(X.T @ X)
    w = XTX_inv @ X.T @ y
    print(sum(w))


if __name__ == "__main__":
    main()
