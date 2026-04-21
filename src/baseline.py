# baseline: TF-IDF + Multinomial NB over all 5 projects
# tuned TF-IDF after the smoke test - bigrams + sublinear tf give ~2 points
# better f1, and fit_prior=False helps because the classes are skewed

import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score


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
    # Labels = title, Comments = body, Codes = follow-up comments
    title = str(row.get("Labels", ""))
    body = str(row.get("Comments", ""))
    parts = [title, body]
    parts.extend(safe_list(row.get("Codes", "[]")))
    return clean_text(" ".join(parts))


def run_baseline(csv_path, n_repeats=N_REPEATS):
    df = pd.read_csv(csv_path)
    df["text"] = df.apply(build_text, axis=1)
    df = df[df["text"].str.len() > 0]
    df = df.dropna(subset=["class"])

    X = df["text"].values
    y = df["class"].astype(int).values

    results = []
    for seed in range(n_repeats):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        # unigrams + bigrams; sublinear tf (log-scaled) works better than
        # raw counts when doc lengths vary a lot, which they do here
        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        # fit_prior=False -> uniform class priors; otherwise MNB almost
        # never predicts the minority class on the smaller projects
        clf = MultinomialNB(fit_prior=False)
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        results.append({
            "seed": seed,
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1": f1_score(y_test, preds, zero_division=0),
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    for proj in PROJECTS:
        print(f"\nRunning baseline on {proj}...")
        res = run_baseline(DATA_DIR / f"{proj}.csv")

        # save per-project raw results so we can re-analyse without rerunning
        out_path = RESULTS_DIR / f"baseline_{proj}.csv"
        res.to_csv(out_path, index=False)
        print(f"  f1 mean: {res['f1'].mean():.3f}")