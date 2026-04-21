# baseline classifier for the 5 DL framework issue datasets
# (tensorflow, pytorch, keras, mxnet, caffe)
# spec says 30 repeated runs per project with different seeds

from pathlib import Path

import pandas as pd


# data lives one level up, results go into a sibling folder
DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROJECTS = ["tensorflow", "pytorch", "keras", "mxnet", "caffe"]
N_REPEATS = 30  # per assignment spec


if __name__ == "__main__":
    # just checking the csvs load and the row counts look sane
    for proj in PROJECTS:
        df = pd.read_csv(DATA_DIR / f"{proj}.csv")
        print(proj, len(df))