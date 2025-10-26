import pickle
import rich
import pathlib
import argparse
import json

RECORD = """
{
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="path_to_model", type=pathlib.Path)

    args = parser.parse_args()

    record = json.loads(RECORD)
    model = pickle.load(args.model.open("rb"))
    rich.print(f"Loaded model from {args.model}\n{model}")
    rich.print("Predicting on record", record)
    rich.print(f"{model.predict_proba(record)[:, 1]=}")


if __name__ == '__main__':
    main()
