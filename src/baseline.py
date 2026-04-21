# baseline classifier for the 5 DL framework issue datasets

import re
import ast
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROJECTS = ["tensorflow", "pytorch", "keras", "mxnet", "caffe"]
N_REPEATS = 30


def safe_list(x):
    # "Codes" is stored as a stringified list, use literal_eval
    if pd.isna(x) or x == "" or x == "[]":
        return []
    try:
        v = ast.literal_eval(x)
    except Exception:
        return []
    if isinstance(v, list):
        return v
    return []


def clean_text(t):
    # tried nltk.word_tokenize first but it was really slow on the
    # larger datasets (tensorflow especially), falling back to regex

    if pd.isna(t):
        t = ""
    else:
        t = str(t)

    # order matters: kill fenced code blocks BEFORE stripping backticks,
    # otherwise the regex chews up the fence markers and leaves the code
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)

    # urls and @mentions aren't useful features, drop them
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)

    # everything non-alphanumeric -> space, then collapse whitespace
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


if __name__ == "__main__":
    for proj in PROJECTS:
        df = pd.read_csv(DATA_DIR / f"{proj}.csv")
        print(proj, len(df))