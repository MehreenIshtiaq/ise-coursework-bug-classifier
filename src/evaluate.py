# evaluate baseline vs proposed - stats tests + figures for the report

import numpy as np
import pandas as pd
from pathlib import Path


RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PROJECTS = ["tensorflow", "pytorch", "keras", "mxnet", "caffe"]
METRICS = ["precision", "recall", "f1"]


def vargha_delaney_a12(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    n_pairs = len(x) * len(y)
    wins = sum(1 for xi in x for yi in y if xi > yi)
    ties = sum(1 for xi in x for yi in y if xi == yi)
    return (wins + 0.5 * ties) / n_pairs


def interpret_a12(a):
    if a < 0.5:
        return "worse"
    if a < 0.56:
        return "negligible"
    if a < 0.64:
        return "small"
    if a < 0.71:
        return "medium"
    return "large"


if __name__ == "__main__":
    # first pass: just print means +/- stds side by side so I can
    # eyeball whether the proposed method is actually winning.
    # stat tests + csv export go in once I trust the numbers.
    print("=" * 95)
    print(f"{'Project':<12} {'Metric':<10} {'Baseline (mean±std)':<22} "
          f"{'Proposed (mean±std)':<22}")
    print("=" * 95)

    for proj in PROJECTS:
        base = pd.read_csv(RESULTS_DIR / f"baseline_{proj}.csv")
        prop = pd.read_csv(RESULTS_DIR / f"proposed_{proj}.csv")
        for m in METRICS:
            b = base[m].values
            p = prop[m].values
            print(f"{proj:<12} {m:<10} "
                  f"{b.mean():.3f} ± {b.std():.3f}     "
                  f"{p.mean():.3f} ± {p.std():.3f}")
        print("-" * 95)