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

Data are not included in this repository.Please cite the dissertation and the original patent-data source when referring to the study or its methods.

## Reuse

No reuse license has been added. Please contact the author before reusing the code.
