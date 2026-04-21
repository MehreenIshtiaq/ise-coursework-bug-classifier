import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROJECTS = ['tensorflow', 'pytorch', 'keras', 'mxnet', 'caffe']

for proj in PROJECTS:
    df = pd.read_csv(DATA_DIR / f"{proj}.csv")
    print(f"\n=== {proj.upper()} ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    print(f"Class %: {df['class'].mean()*100:.1f}% positive")