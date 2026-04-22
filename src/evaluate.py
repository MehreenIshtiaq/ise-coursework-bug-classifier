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
    # P(random value from x > random value from y), ties count 0.5.
    # Effect-size counterpart to wilcoxon: 0.5 = no effect, >0.5 = x wins.
    x = np.asarray(x)
    y = np.asarray(y)
    n_pairs = len(x) * len(y)
    wins = sum(1 for xi in x for yi in y if xi > yi)
    ties = sum(1 for xi in x for yi in y if xi == yi)
    return (wins + 0.5 * ties) / n_pairs


def interpret_a12(a):
    # thresholds from Vargha & Delaney (2000) - what the assignment references
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
    # sanity check the helpers before hooking them into the main loop
    print("a12([1,2,3], [0,0,0]) =", vargha_delaney_a12([1, 2, 3], [0, 0, 0]))
    print("interpret_a12(0.85) =", interpret_a12(0.85))