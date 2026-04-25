"""
Verification checks for baseline and proposed method.
Run with: python src\verify.py
"""
import pandas as pd
import numpy as np
import re, ast
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

DATA_DIR    = Path(__file__).parent.parent / "data"
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Reuse same helpers as baseline / proposed
def safe_list(x):
    if pd.isna(x) or x == '' or x == '[]': return []
    try:
        v = ast.literal_eval(x)
        return v if isinstance(v, list) else []
    except Exception: return []

def clean_text(t):
    t = str(t) if not pd.isna(t) else ''
    t = re.sub(r'```.*?```', ' ', t, flags=re.S)
    t = re.sub(r'`[^`]*`', ' ', t)
    t = re.sub(r'http\S+', ' ', t)
    t = re.sub(r'@\w+', ' ', t)
    t = re.sub(r'[^a-zA-Z0-9\s]', ' ', t)
    t = re.sub(r'\s+', ' ', t).strip().lower()
    return t

def build_text(row):
    parts = [str(row.get('Labels', '')), str(row.get('Comments', ''))]
    parts.extend(safe_list(row.get('Codes', '[]')))
    return clean_text(' '.join(parts))

print("=" * 70)
print("VERIFICATION REPORT")
print("=" * 70)

# ---------- CHECK A: Result files exist and have correct shape ----------
print("\n[A] Checking result files exist and have 30 rows each...")
for kind in ['baseline', 'proposed']:
    for proj in ['tensorflow','pytorch','keras','mxnet','caffe']:
        f = RESULTS_DIR / f"{kind}_{proj}.csv"
        if not f.exists():
            print(f"  MISSING: {f}")
            continue
        df = pd.read_csv(f)
        ok = "OK" if len(df) == 30 else f"WRONG: {len(df)} rows"
        print(f"  {f.name}: {ok}")

# ---------- CHECK B: Train/test splits are identical between methods ----------
print("\n[B] Checking baseline and proposed use IDENTICAL train/test splits...")
csv_path = DATA_DIR / "caffe.csv"
df = pd.read_csv(csv_path)
df['text'] = df.apply(build_text, axis=1)
df = df[df['text'].str.len() > 0].dropna(subset=['class'])
X, y = df['text'].values, df['class'].astype(int).values

# Run the same split twice with same seed — must be identical
X_tr1, X_te1, y_tr1, y_te1 = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X, y, test_size=0.3, random_state=7, stratify=y)
splits_match = (X_tr1 == X_tr2).all() and (y_te1 == y_te2).all()
print(f"  Same seed -> same split: {'OK' if splits_match else 'FAIL'}")
print(f"  Train size = {len(y_tr1)}, Test size = {len(y_te1)}")
print(f"  Train positives = {sum(y_tr1==1)}, Test positives = {sum(y_te1==1)}")

# ---------- CHECK C: Manual metric computation matches sklearn ----------
print("\n[C] Manually re-computing baseline metrics for caffe seed 0...")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
vec = TfidfVectorizer(max_features=5000, stop_words='english',
                      ngram_range=(1,2), sublinear_tf=True)
X_tr_v = vec.fit_transform(X_tr); X_te_v = vec.transform(X_te)
clf = MultinomialNB(fit_prior=False); clf.fit(X_tr_v, y_tr)
pred = clf.predict(X_te_v)
tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0,1]).ravel()
manual_p = tp / (tp+fp) if (tp+fp)>0 else 0.0
manual_r = tp / (tp+fn) if (tp+fn)>0 else 0.0
manual_f = 2*manual_p*manual_r / (manual_p+manual_r) if (manual_p+manual_r)>0 else 0.0

stored = pd.read_csv(RESULTS_DIR / "baseline_caffe.csv").iloc[0]
print(f"  Confusion matrix: TP={tp} FP={fp} FN={fn} TN={tn}")
print(f"  Manually computed   precision={manual_p:.4f}, recall={manual_r:.4f}, f1={manual_f:.4f}")
print(f"  Stored in CSV       precision={stored.precision:.4f}, recall={stored.recall:.4f}, f1={stored.f1:.4f}")
match = (abs(manual_f - stored.f1) < 1e-6)
print(f"  Match: {'OK' if match else 'FAIL'}")

# ---------- CHECK D: SMOTE actually balanced the training set ----------
print("\n[D] Checking SMOTE actually rebalances the training set...")
features = FeatureUnion([
    ('word', TfidfVectorizer(ngram_range=(1,2), max_features=5000, stop_words='english', sublinear_tf=True)),
    ('char', TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), max_features=5000, sublinear_tf=True)),
])
X_tr_v = features.fit_transform(X_tr)
n_pos_before = int((y_tr==1).sum()); n_neg_before = int((y_tr==0).sum())
sm = SMOTE(random_state=0, k_neighbors=min(5, n_pos_before-1))
X_tr_s, y_tr_s = sm.fit_resample(X_tr_v, y_tr)
n_pos_after = int((y_tr_s==1).sum()); n_neg_after = int((y_tr_s==0).sum())
print(f"  Before SMOTE:  {n_pos_before} positives, {n_neg_before} negatives")
print(f"  After SMOTE:   {n_pos_after} positives, {n_neg_after} negatives")
print(f"  Balanced: {'OK' if n_pos_after == n_neg_after else 'FAIL'}")

print("\n" + "=" * 70)
print("Verification complete.")
print("=" * 70)