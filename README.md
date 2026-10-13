# Patent Citation Forecasting

This repository contains code for a dissertation study forecasting the additional forward citations granted patents receive over the next one to five years.

The study uses a patent’s first five years of citation history and tests whether adding patent text, technology-cluster information, scientific-paper activity or financial-report activity improves forecasts. It compares XGBoost and GRU models with simple citation-history baselines.

## Code

- `src/xgboost/` contains the six XGBoost input models.
- `src/gru/` contains the four GRU input models.
- `src/baselines/` contains the persistence and fitted log-sum baselines.
- `src/models/` contains the experiment runners.
- `src/interpretation/` and `src/validation/` contain supporting analysis code.

The code needs project data and configuration files that are not included here, so cloning the repository alone is not enough to reproduce the results. Check the inputs expected by each script before running it.

## Data

The repository does not contain patent records, citation counts, paper or financial-report text, keyword clusters, embeddings, fitted models, predictions or SHAP values. The keyword clusters and embeddings are being held for separate publication. Please cite the primary source of the patent citation data and the dissertation when using this code.

## Notes on the results

The evaluation used repeated random patent splits, not a chronological test on later patent cohorts. Patents could also appear in tuning and evaluation roles across different splits, which is a limitation of the results.

Financial-report activity was assigned by reporting-period year. The analysis did not verify that every report was publicly available by the end of that year. SHAP describes how the fitted models used their inputs; it does not establish causal science-push or market-pull effects.

## Citation

Please cite the dissertation and the primary source of the patent citation data. Add the dissertation’s full reference when its publication details are available.

## Reuse

No reuse license has been added. Please contact the author before reusing the code.
