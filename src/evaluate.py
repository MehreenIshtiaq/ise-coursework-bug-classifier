# evaluate baseline vs proposed - stats tests + figures for the report
# with wilcoxon + A12 effect size
# with stat tests and F1 boxplot

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RESULTS_DIR = Path(__file__).parent.parent / "results"
FIGURES_DIR = Path(__file__).parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

PROJECTS = ["tensorflow", "pytorch", "keras", "mxnet", "caffe"]
METRICS = ["precision", "recall", "f1"]


def vargha_delaney_a12(x, y):
    """P(random value from x > random value from y). Ties count 0.5."""
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
    # --- comparison table ---
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
            try:
                stat, pval = wilcoxon(p, b, zero_method='wilcox')
            except ValueError:
                pval = float('nan')   # happens if all differences are zero
            a12 = vargha_delaney_a12(p, b)
            sig = "*" if pval < 0.05 else " "
            print(f"{proj:<12} {m:<10} "
                  f"{b.mean():.3f} ± {b.std():.3f}     "
                  f"{p.mean():.3f} ± {p.std():.3f}     "
                  f"{pval:.4f}{sig}  {a12:.3f}    {interpret_a12(a12)}")
            table_rows.append({
                'project': proj, 'metric': m,
                'baseline_mean': b.mean(), 'baseline_std': b.std(),
                'proposed_mean': p.mean(), 'proposed_std': p.std(),
                'p_value': pval, 'a12': a12,
                'effect': interpret_a12(a12),
            })
        print("-" * 95)

    summary = pd.DataFrame(table_rows)
    summary.to_csv(RESULTS_DIR / "comparison_summary.csv", index=False)
    print(f"\nSaved summary table to {RESULTS_DIR / 'comparison_summary.csv'}")

    # --- F1 boxplot ---
    # two boxes per project: baseline (grey) and proposed (blue).
    # positions: project i takes x = i*3 + {0.6, 1.4}, ticks centred at i*3 + 1.0
    fig, ax = plt.subplots(figsize=(10, 5.5))
    positions, data, colors = [], [], []
    for i, proj in enumerate(PROJECTS):
        base = pd.read_csv(RESULTS_DIR / f"baseline_{proj}.csv")['f1'].values
        prop = pd.read_csv(RESULTS_DIR / f"proposed_{proj}.csv")['f1'].values
        positions += [i*3 + 0.6, i*3 + 1.4]
        data      += [base, prop]
        colors    += ['#cccccc', '#4c72b0']

    bp = ax.boxplot(data, positions=positions, widths=0.7, patch_artist=True,
                    medianprops=dict(color='black'))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)

    ax.set_xticks([i*3 + 1.0 for i in range(len(PROJECTS))])
    ax.set_xticklabels([p.capitalize() for p in PROJECTS])
    ax.set_ylabel('F1 score')
    ax.set_title('Baseline (NB + TF-IDF) vs Proposed (LR + n-grams + SMOTE)')
    ax.set_ylim(-0.02, 1.0)
    ax.grid(axis='y', alpha=0.3)

    ax.legend(handles=[Patch(facecolor='#cccccc', label='Baseline'),
                       Patch(facecolor='#4c72b0', label='Proposed')],
              loc='upper right')

    plt.tight_layout()
    out_pdf = FIGURES_DIR / "f1_comparison_boxplot.pdf"
    plt.savefig(out_pdf)
    plt.savefig(FIGURES_DIR / "f1_comparison_boxplot.png", dpi=150)
    print(f"Saved figure to {out_pdf}")