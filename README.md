# Patent Citation Forecasting

Code accompanying a dissertation study that forecasts additional forward citations to granted patents over one- to five-year horizons. It compares XGBoost and GRU models and tests whether patent text, technology-cluster representations, scientific-paper activity and financial-report activity add predictive information beyond early citation history.

**Code availability:** [Add the public GitHub repository URL after publication].

## Repository contents

```text
src/
  models/
    xgboost_six_input_families.py
    full_panel_xgboost_gru.py
  interpretation/
    xgboost_shap_seed20_refits.py
  validation/
    audit_external_source_counts.py
data/
  README.md
docs/
  DATA_AND_INPUTS.md
  RELEASE_CHECKLIST.md
requirements.txt
```

The scripts are staged copies of project files. They may expect input files and settings in specific working directories; consult each script before running it. This repository is not yet a one-command reproduction package.

## Data are not included

No patent-level records, citation counts, financial reports, papers, keyword clusters, embeddings, feature arrays, model checkpoints, predictions or SHAP values are hosted here. The keyword clusters and embeddings are reserved for separate publication. Patent citation counts should be obtained and cited from the primary source documented in the dissertation. Input access and expected filenames are described in [`docs/DATA_AND_INPUTS.md`](docs/DATA_AND_INPUTS.md).

## Running the code

Use Python with the packages listed in `requirements.txt`. Install them in an isolated environment, then obtain the authorized input and configuration files described in each script. Start with a script's input-validation or dry-run option where available. Exact package versions must be recorded from the environment used for each final run before claiming exact reproducibility.

## Evaluation and interpretation notes

The reported evaluation used repeated random splits of patents, not a chronological holdout of later publication cohorts. Patent records could recur across tuning and evaluation roles in different random splits; this creates a possible model-selection optimism limitation. Corrected paired tests account for dependence among repeated split estimates, but do not remove tuning/evaluation overlap.

Financial-report activity was indexed by reporting-period year. The analysis did not verify that every report was publicly available at the end of that year, so financial-source results should be interpreted as reporting-year-aligned predictive associations, not verified real-time forecasts. SHAP values describe model attribution and do not establish causal science-push or market-pull effects.

## Citation

Please cite the dissertation and the primary source of the patent citation data. Add the final dissertation citation here when its bibliographic details are confirmed.

## License

No license is provided yet. Until a license is added, the code remains subject to its default copyright status.
