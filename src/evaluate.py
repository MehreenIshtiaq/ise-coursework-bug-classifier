# evaluate baseline vs proposed - stats tests + figures for the report
#  with wilcoxon + A12 effect size

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon


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
    print("=" * 95)
    print(f"{'Project':<12} {'Metric':<10} {'Baseline (mean±std)':<22} "
          f"{'Proposed (mean±std)':<22} {'p-value':<10} {'A12':<8} {'Effect'}")
    print("=" * 95)

    table_rows = []
    for proj in PROJECTS:
        base = pd.read_csv(RESULTS_DIR / f"baseline_{proj}.csv")
        prop = pd.read_csv(RESULTS_DIR / f"proposed_{proj}.csv")
        for m in METRICS:
            b, p = base[m].values, prop[m].values

            # wilcoxon signed-rank: paired test across the 30 matched seeds.
            # wraps in try/except because if every diff is zero scipy raises.
            try:
                stat, pval = wilcoxon(p, b, zero_method="wilcox")
            except ValueError:
                pval = float("nan")

            a12 = vargha_delaney_a12(p, b)
            sig = "*" if pval < 0.05 else " "

            print(f"{proj:<12} {m:<10} "
                  f"{b.mean():.3f} ± {b.std():.3f}     "
                  f"{p.mean():.3f} ± {p.std():.3f}     "
                  f"{pval:.4f}{sig}  {a12:.3f}    {interpret_a12(a12)}")

            table_rows.append({
                "project": proj, "metric": m,
                "baseline_mean": b.mean(), "baseline_std": b.std(),
                "proposed_mean": p.mean(), "proposed_std": p.std(),
                "p_value": pval, "a12": a12,
                "effect": interpret_a12(a12),
            })
        print("-" * 95)

    # save for the report - easier than hand-copying numbers out of the console
    summary = pd.DataFrame(table_rows)
    summary.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    print(f"\nSaved summary table to {RESULTS_DIR / 'comparison_summary.csv'}")