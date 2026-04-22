# proposed method: word+char TFIDF -> SMOTE -> Logistic Regression
# final version with summary table + pred-positive sanity check

import re
import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE


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
    # word bigrams + char n-grams; char_wb because token-internal chars
    # in things like cuda_oom / conv2d carry real signal
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


def run_proposed(csv_path, n_repeats=N_REPEATS):
    df = pd.read_csv(csv_path)
    df["text"] = df.apply(build_text, axis=1)
    df = df[df["text"].str.len() > 0].dropna(subset=["class"])
    X = df["text"].values
    y = df["class"].astype(int).values

    rows = []
    for seed in range(n_repeats):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.3, random_state=seed, stratify=y
        )

        features = build_features()
        X_tr_v = features.fit_transform(X_tr)
        X_te_v = features.transform(X_te)

        # SMOTE k_neighbors guard - minority class can be tiny
        n_pos = int((y_tr == 1).sum())
        k = max(1, min(5, n_pos - 1))
        sm = SMOTE(random_state=seed, k_neighbors=k)
        X_tr_s, y_tr_s = sm.fit_resample(X_tr_v, y_tr)

        clf = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            C=1.0,
            solver="liblinear",
        )
        clf.fit(X_tr_s, y_tr_s)
        pred = clf.predict(X_te_v)

        rows.append({
            "seed": seed,
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred, zero_division=0),
            "f1": f1_score(y_te, pred, zero_division=0),
            # keep this - makes it obvious when the model predicts all-negative
            "n_pred_pos": int((pred == 1).sum()),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    summary = []

    for proj in PROJECTS:
        print(f"\nRunning proposed method on {proj}...")
        results = run_proposed(DATA_DIR / f"{proj}.csv")
        out_path = RESULTS_DIR / f"proposed_{proj}.csv"
        results.to_csv(out_path, index=False)

        p_mean = results["precision"].mean()
        r_mean = results["recall"].mean()
        f_mean = results["f1"].mean()
        f_std = results["f1"].std()
        avg_pred = results["n_pred_pos"].mean()

        print(f"  precision: {p_mean:.3f}")
        print(f"  recall: {r_mean:.3f}")
        print(f"  f1: {f_mean:.3f} (std {f_std:.3f})")
        print(f"  avg predicted positives per run: {avg_pred:.1f}")

        summary.append({
            "project": proj,
            "precision": p_mean,
            "recall": r_mean,
            "f1": f_mean,
            "f1_std": f_std,
        })

    print("\n=== PROPOSED METHOD SUMMARY ===")
    print(pd.DataFrame(summary).to_string(index=False))