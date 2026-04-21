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
    if pd.isna(t):
        t = ""
    else:
        t = str(t)

    # strip fenced code blocks first, then inline backticks, then urls
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def build_text(row):
    # heads up: the column names in this dataset are confusing.
    # "Labels" is actually the issue TITLE (not the classification label -
    # that's in "class"), and "Comments" is the issue BODY. "Codes" holds
    # follow-up comment text. Took me a good hour to realise I was training
    # on the wrong field.
    title = str(row.get("Labels", ""))
    body = str(row.get("Comments", ""))

    parts = [title, body]
    parts.extend(safe_list(row.get("Codes", "[]")))
    return clean_text(" ".join(parts))


if __name__ == "__main__":
    for proj in PROJECTS:
        df = pd.read_csv(DATA_DIR / f"{proj}.csv")
        df["text"] = df.apply(build_text, axis=1)
        
        print(proj, len(df), "sample:", df["text"].iloc[0][:80])