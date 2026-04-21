# baseline: TF-IDF + Multinomial NB over all 5 projects
# reports precision / recall / f1 averaged over 30 seeds per project

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
    # Codes column stores stringified python lists, literal_eval them
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

    # order matters - strip code fences before backticks
    t = re.sub(r"```.*?```", " ", t, flags=re.S)
    t = re.sub(r"`[^`]*`", " ", t)
    t = re.sub(r"http\S+", " ", t)
    t = re.sub(r"@\w+", " ", t)
    t = re.sub(r"[^a-zA-Z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t.lower()


def build_text(row):
    # Labels = title, Comments = body, Codes = follow-up comments (yes, really)
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

        vec = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X_train_vec = vec.fit_transform(X_train)
        X_test_vec = vec.transform(X_test)

        clf = MultinomialNB(fit_prior=False)
        clf.fit(X_train_vec, y_train)
        preds = clf.predict(X_test_vec)

        p = precision_score(y_test, preds, zero_division=0)
        r = recall_score(y_test, preds, zero_division=0)
        f = f1_score(y_test, preds, zero_division=0)
        # on mxnet/caffe I had a run where precision=0, recall=0, f1=0
        # because the model predicted all-negative. tracking n_pred_pos
        # makes that obvious in the per-seed output
        n_pos = int((preds == 1).sum())

        results.append({
            "seed": seed,
            "precision": p,
            "recall": r,
            "f1": f,
            "n_pred_pos": n_pos,
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    summary_rows = []

    for proj in PROJECTS:
        print(f"\nRunning baseline on {proj}...")
        res = run_baseline(DATA_DIR / f"{proj}.csv")

        out_path = RESULTS_DIR / f"baseline_{proj}.csv"
        res.to_csv(out_path, index=False)

        p_mean = res["precision"].mean()
        r_mean = res["recall"].mean()
        f_mean = res["f1"].mean()
        f_std = res["f1"].std()
        avg_pred = res["n_pred_pos"].mean()

        print(f"  precision: {p_mean:.3f}")
        print(f"  recall: {r_mean:.3f}")
        print(f"  f1: {f_mean:.3f} (std {f_std:.3f})")
        print(f"  avg predicted positives per run: {avg_pred:.1f}")

        summary_rows.append({
            "project": proj,
            "precision": p_mean,
            "recall": r_mean,
            "f1": f_mean,
            "f1_std": f_std,
        })

    # final summary table - this is what goes into the report
    print("\n=== BASELINE SUMMARY ===")
    print(pd.DataFrame(summary_rows).to_string(index=False))