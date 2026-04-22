# proposed method for the bug classifier
# reusing baseline preprocessing so the only thing being compared
# downstream is the classifier, not the text features

import re
import ast
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PROJECTS = ["tensorflow", "pytorch", "keras", "mxnet", "caffe"]
N_REPEATS = 30


# NOTE: these three helpers are copies of the baseline's. Keeping them
# duplicated (rather than importing) so each script runs standalone -
# the marker might want to run them in isolation.
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

    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def build_text(row):
    title = str(row.get("Labels", ""))
    body = str(row.get("Comments", ""))
    parts = [title, body]
    parts.extend(safe_list(row.get("Codes", "[]")))
    return clean_text(" ".join(parts))


if __name__ == "__main__":
    # just making sure the preprocessing runs end to end
    df = pd.read_csv(DATA_DIR / "tensorflow.csv")
    df["text"] = df.apply(build_text, axis=1)
    print(df["text"].iloc[0][:100])