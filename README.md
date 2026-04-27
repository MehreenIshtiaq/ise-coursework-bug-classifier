# ISE Coursework — Bug Report Classification (Task 1)

**Author:** Mehreen Ishtiaq
**Module:** Intelligent Software Engineering (ISE), University of Birmingham
**Task:** Task 1 — Bug Report Classification for Deep Learning Frameworks

This repository contains the implementation, datasets, and supporting documents for my ISE coursework. The project compares a Naive Bayes + TF-IDF baseline against a proposed Logistic Regression pipeline that uses combined word and character n-gram TF-IDF features, SMOTE oversampling, and balanced class weights, evaluated on five deep learning framework bug-report datasets (TensorFlow, PyTorch, Keras, MXNet, Caffe).

## Repository structure

```
.
├── README.md                  # This file
├── requirements.pdf           # Software/hardware requirements (per spec §1.5)
├── manual.pdf                 # User manual (per spec §1.5)
├── replication.pdf            # Replication instructions (per spec §1.5)
├── report.pdf                 # Final coursework report
├── lab1/
│   ├── baseline.py            # Naive Bayes + TF-IDF baseline
│   ├── proposed.py            # Logistic Regression + word/char TF-IDF + SMOTE
│   ├── evaluate.py            # 30-split evaluation, Wilcoxon test, A12 effect size
│   └── utils.py               # Shared text-cleaning and helper functions
├── datasets/
│   ├── tensorflow.csv
│   ├── pytorch.csv
│   ├── keras.csv
│   ├── mxnet.csv
│   └── caffe.csv
├── results/
│   ├── baseline_results.csv
│   ├── proposed_results.csv
│   ├── statistical_tests.csv
│   └── summary_table.csv
├── figures/
│   └── f1_comparison_boxplot.png
└── requirements.txt           # Python dependencies
```

## Problem

Bug-report triage is time-consuming for maintainers of large deep learning frameworks. Performance bugs (slow training, memory blow-ups, GPU under-utilisation) are particularly hard to spot because reporters rarely use the word "performance". This project trains a text classifier to label each bug report as performance-related or not, using only the title and body text.

## Method (one-line summary)

Classical TF-IDF (word 1–2 grams + character 3–5 grams) → SMOTE oversampling on the minority class → Logistic Regression with `class_weight='balanced'`. Compared against a Naive Bayes + word TF-IDF baseline.

## Results

Across all five datasets the proposed method beats the baseline on Precision, Recall, and F1, with Vargha-Delaney A12 effect sizes between 0.948 and 1.000 and Wilcoxon p-values < 0.0001 (30 stratified 70/30 splits per dataset).

| Project    | Baseline F1 | Proposed F1 | A12   |
|------------|-------------|-------------|-------|
| TensorFlow | 0.428       | 0.628       | 1.00  |
| PyTorch    | 0.013       | 0.562       | 1.00  |
| Keras      | 0.042       | 0.465       | 1.00  |
| MXNet      | 0.040       | 0.512       | 1.00  |
| Caffe      | 0.035       | 0.264       | 0.948 |

Full numerical results are in `results/` and the analysis is written up in `report.pdf`.

## Quick start

```bash
# 1. Clone the repo
git clone https://github.com/MehreenIshtiaq/ise-coursework-bug-classifier.git
cd ise-coursework-bug-classifier

# 2. Create a virtual environment (Python 3.10+ recommended)
python -m venv venv
source venv/bin/activate          # on Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Reproduce all results (this takes a few minutes)
python lab1/evaluate.py
```

Outputs will appear in `results/` and the box plot in `figures/`.

## Documents

For full details please see:

- `requirements.pdf` — Python version, libraries, hardware notes.
- `manual.pdf` — How to install, configure, and run the pipeline.
- `replication.pdf` — Exact steps to reproduce every number in the report.
- `report.pdf` — The full coursework report (Introduction, Related Work, Solution, Setup, Experiments, Reflection, Conclusion, Artifact, References).

## Datasets

The five bug-report datasets used in this project were released with the ISE 2024 Lab 1 materials and are stored in `datasets/`. Each CSV has columns: `Title`, `Body`, `class` (1 = performance bug, 0 = otherwise).

## Acknowledgements

Datasets and the original baseline pipeline were provided as part of the University of Birmingham ISE module (Lab 1). All extensions, evaluation code, and analysis in this repository are my own work.

## License

This repository is released for academic assessment purposes only.
