# Separated model files

These files separate the experiment by model family and input set. The six
XGBoost wrappers share `xgboost/common.py`; the GRU files define the four
network architectures. Data loading, split generation, configuration lookup,
scaling, repeated evaluation, and metric calculation remain in the experiment
runners under `models/` and are not duplicated in each wrapper.

## XGBoost

| File | Input family |
|---|---|
| `xgboost/citations_only.py` | Early citation history |
| `xgboost/patent_text.py` | Citations + patent text |
| `xgboost/centroids.py` | Citations + cluster centroids |
| `xgboost/paper_activity.py` | Citations + scientific-paper activity |
| `xgboost/finance_activity.py` | Citations + financial-report activity |
| `xgboost/paper_finance_combined.py` | Citations + paper + finance activity |

## GRU

| File | Input family |
|---|---|
| `gru/citations_only.py` | Citation sequence + citation summaries |
| `gru/patent_text.py` | Citation sequence + summaries + text embedding |
| `gru/centroids.py` | Citation sequence + summaries + cluster centroid vector |
| `gru/paper_activity.py` | Citation, paper-count and keyword-hit sequences + summaries |

The paper GRU definition is included from the supplied paper-GRU notebook as
requested. The notebook's data preparation and final-run workflow are not
represented as a standalone reproduction by this model-definition file.

## Baselines

| File | Baseline |
|---|---|
| `baselines/persistence.py` | Carry forward the age-4 annual citation count |
| `baselines/fitted_log_sum.py` | Train-only linear regression from log early-citation sum to log future citations |

The repository also retains the combined experiment runners used to execute
the repeated-split analyses. These separate files make each model/input family
easy to locate; they do not replace the runners' data and evaluation pipeline.
