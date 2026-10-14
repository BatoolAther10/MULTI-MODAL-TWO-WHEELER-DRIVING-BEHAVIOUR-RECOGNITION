# TWMRA: Two-Wheeler Model Results Archive

This folder contains the reproducible model artifacts, metrics, logs, figures,
reports, datasets, and Python scripts for the two-wheeler riding-behaviour
recognition experiments.

The task classifies five riding events:

- `BUMP`
- `LEFT`
- `RIGHT`
- `STOP`
- `STRAIGHT`

The model input contains seven sensor features in this order:

```text
Ax, Ay, Az, Gx, Gy, Gz, Speed
```

The configured window is 156 samples with a stride of 78 samples.

## Archive structure

```text
TWMRA/
├── 80-20/
│   ├── data/
│   ├── figures/
│   ├── logs/
│   ├── metrics/
│   ├── models/
│   ├── reports/
│   └── scripts/
└── strict_protocol/
    ├── data/
    ├── figures/
    ├── logs/
    ├── metrics/
    ├── models/
    ├── reports/
    └── scripts/
```

The two protocol folders are intentionally separate. Do not combine their
metrics when making an independent-test claim.

## 1. Clean.csv random 80/20 protocol

The `80-20/` folder contains the completed random-window experiment for all
27 models.

### Protocol

1. Read `data/clean.csv`.
2. Generate 156-sample windows with stride 78.
3. Encode the five classes using the configured class order.
4. Split windows with a stratified random 80/20 split.
5. Use random state `42`.
6. Train each model on 836 windows.
7. Evaluate each model on 209 windows.

The exact protocol metadata is in:

[80-20/logs/protocol.json](80-20/logs/protocol.json)

### Main outputs

- [80-20/metrics/results.csv](80-20/metrics/results.csv)
- [80-20/metrics/classwise.csv](80-20/metrics/classwise.csv)
- [80-20/metrics/confusion_matrices.csv](80-20/metrics/confusion_matrices.csv)
- [80-20/reports/COMPLETE_27_MODEL_REPORT_TOP5_CONCLUSION.docx](80-20/reports/COMPLETE_27_MODEL_REPORT_TOP5_CONCLUSION.docx)
- [80-20/scripts/run_clean_80_20_all_models.py](80-20/scripts/run_clean_80_20_all_models.py)

All 27 rows in `results.csv` have completed accuracy and macro-F1 values.
The best measured 80/20 models were:

| Rank | Model | Accuracy | Macro-F1 | Weighted-F1 |
|---:|---|---:|---:|---:|
| 1 | HistGradientBoosting | 81.34% | 81.53% | 81.20% |
| 2 | LightGBM | 80.38% | 79.99% | 80.34% |
| 3 | XGBoost | 79.90% | 79.28% | 79.92% |
| 4 | Random Forest | 75.60% | 74.59% | 75.32% |
| 5 | Mega | 69.86% | 69.40% | 70.83% |

### 80/20 interpretation

This is a paper-style benchmark, not an independent holdout estimate.
Overlapping windows can place nearby samples in both sides of a random split,
which can make the result optimistic. Use the strict protocol for deployment
and independent-test claims.

## 2. Strict four-file protocol

The `strict_protocol/` folder contains the independent-test workflow.

### Protocol

```text
train.csv
    -> fit candidate models
val.csv
    -> select configurations
train_val_combined.csv
    -> retrain the selected configuration
test.csv
    -> evaluate the saved artifact once
```

The strict split contains:

| Split | Windows |
|---|---:|
| Training | 3,875 |
| Validation | 1,045 |
| Combined retraining | 4,921 |
| Final test | 1,045 |

The workflow metadata is in:

[strict_protocol/logs/comprehensive_workflow.json](strict_protocol/logs/comprehensive_workflow.json)

### Main outputs

- [strict_protocol/metrics/complete_27_model_metrics.csv](strict_protocol/metrics/complete_27_model_metrics.csv)
- [strict_protocol/metrics/comprehensive_metrics.csv](strict_protocol/metrics/comprehensive_metrics.csv)
- [strict_protocol/metrics/comprehensive_classwise.csv](strict_protocol/metrics/comprehensive_classwise.csv)
- [strict_protocol/metrics/comprehensive_robustness.csv](strict_protocol/metrics/comprehensive_robustness.csv)
- [strict_protocol/reports/COMPLETE_27_MODEL_REPORT.docx](strict_protocol/reports/COMPLETE_27_MODEL_REPORT.docx)
- [strict_protocol/reports/COMPLETE_27_MODEL_REPORT_TOP5_CONCLUSION.docx](strict_protocol/reports/COMPLETE_27_MODEL_REPORT_TOP5_CONCLUSION.docx)

### Strict workflow scripts

- [strict_protocol/scripts/rigorous_experiment.py](strict_protocol/scripts/rigorous_experiment.py)
- [strict_protocol/scripts/neural_models_experiment.py](strict_protocol/scripts/neural_models_experiment.py)
- [strict_protocol/scripts/comprehensive_workflow.py](strict_protocol/scripts/comprehensive_workflow.py)
- [strict_protocol/scripts/build_complete_27_model_report.py](strict_protocol/scripts/build_complete_27_model_report.py)

The strict folder also contains the complete project Python script set,
including model builders, tuning scripts, benchmark scripts, robustness
scripts, report builders, and validation utilities.

## Model artifacts

Each protocol keeps its model artifacts separate:

- `80-20/models/` contains models trained on the clean.csv random split.
- `strict_protocol/models/` contains strict-protocol deployment and experiment
  artifacts.

Do not replace a strict artifact with an 80/20 artifact. They represent
different training and evaluation protocols.

## Metrics and figures

The archive includes:

- Accuracy, macro-F1, and weighted-F1
- Class-wise precision, recall, F1, and support
- Confusion matrices
- Benchmark accuracy and macro-F1 graphs
- Latency figures
- MAE figures
- Channel-dropout robustness
- Noise robustness
- Missing-value robustness
- Clock-jitter robustness
- Window-size sensitivity
- Hyperparameter tuning outputs

Figures are stored under each protocol's `figures/` folder. CSV and JSON
measurements are stored under `metrics/` and `logs/`.

## Reproducing the 80/20 experiment

From the project root, run:

```powershell
python scripts/run_clean_80_20_all_models.py
```

The script reads the configured processed clean CSV and writes results to:

```text
80/20/
```

The archived copy of the runner is:

[80-20/scripts/run_clean_80_20_all_models.py](80-20/scripts/run_clean_80_20_all_models.py)

## Reproducing the strict experiments

The strict scripts expect the project dependencies and the four root split
files. From the project root, the main workflows are:

```powershell
python scripts/rigorous_experiment.py
python scripts/neural_models_experiment.py
python scripts/comprehensive_workflow.py
```

The archived strict scripts are provided for reference and reproducibility.
They may need to be run from the project root so imports such as `utils` and
`inference_layer` resolve correctly.

## Important limitations

1. The random 80/20 protocol uses overlapping windows and may overestimate
   generalization.
2. The strict four-file test is the authoritative protocol for independent
   evaluation.
3. Software latency measurements are not Raspberry Pi 5 hardware
   measurements.
4. Model artifacts from the two protocols must not be mixed.
5. A reported `N/A` in a legacy supplementary table means that a particular
   measurement was not preserved in that source artifact; it is not a
   fabricated score.
6. The sensor feature order is column-name based and normalized to
   `Ax, Ay, Az, Gx, Gy, Gz, Speed`.

## Recommended result usage

- Use `80-20/` to reproduce the earlier paper-style comparison.
- Use `strict_protocol/` for deployment decisions and independent-test
  reporting.
- Use the top-five conclusion report for a concise comparison of the best
  measured models across both protocols and robustness checks.

