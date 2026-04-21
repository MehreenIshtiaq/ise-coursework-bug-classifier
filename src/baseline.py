# baseline classifier for the 5 DL framework issue datasets

import ast
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROJECTS = ["tensorflow", "pytorch", "keras", "mxnet", "caffe"]
N_REPEATS = 30


def safe_list(x):
    # the "Codes" column is stored as a string like "['foo', 'bar']"
    # instead of a real list... no idea why. use literal_eval, not eval,
    # since we definitely don't trust random strings from github issues
    if pd.isna(x) or x == "" or x == "[]":
        return []
    try:
        v = ast.literal_eval(x)
    except Exception:
        # some rows had unterminated quotes - just skip those
        return []
    if isinstance(v, list):
        return v
    return []


if __name__ == "__main__":
    for proj in PROJECTS:
        df = pd.read_csv(DATA_DIR / f"{proj}.csv")
        print(proj, len(df))