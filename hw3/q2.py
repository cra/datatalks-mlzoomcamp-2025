import itertools
import sys

import pandas as pd


def main():
    if len(sys.argv) != 2:
        print("Usage: uv python q2.py <csv_file>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    c = 0
    pairs = [
        ["interaction_count", "lead_score"],
        ["number_of_courses_viewed", "lead_score"],
        ["number_of_courses_viewed", "interaction_count"],
        ["annual_income", "interaction_count"],
    ]
    numerical_columns_of_interest = list(set(itertools.chain.from_iterable(pairs)))
    corr_matrix = df[numerical_columns_of_interest].corr()
    win = None
    for p in pairs:
        print(f"= PAIR: {p} =")
        cc = corr_matrix.loc[*p]
        print("  corr=", cc)
        if abs(cc) > abs(c):
            c = cc
            win = p[:]  # copy just in case
    print("BIGGEST corr =", win)


if __name__ == "__main__":
    main()
