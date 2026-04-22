# proposed method for the bug classifier

import re
import ast
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


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


def build_features():
    # word bigrams catch phrases like "out of memory" or "segmentation fault".
    # char n-grams (3-5, char_wb) catch weird identifier-like tokens that the
    # word tokenizer treats as one chunk - e.g. "cuda_oom", "conv2d", "nan_loss".
    # these subword signals seem to matter a lot for GPU framework bugs.
    word_vec = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        stop_words="english",
        sublinear_tf=True,
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=5000,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


if __name__ == "__main__":
    # quick check: fit feature union and see the combined dim
    df = pd.read_csv(DATA_DIR / "tensorflow.csv")
    df["text"] = df.apply(build_text, axis=1)
    df = df[df["text"].str.len() > 0]

    features = build_features()
    X = features.fit_transform(df["text"].values)
    print("feature matrix shape:", X.shape)