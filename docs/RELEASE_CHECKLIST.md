# Before making the repository public

- Confirm the four staged scripts are the exact versions used for the reported analyses; this folder was assembled from the available working directory and is a staging copy.
- Review each script for private paths, credentials, embedded records and unintended output files.
- Do not add raw or processed datasets, keyword-cluster tables, embedding arrays, patent text, model checkpoints, prediction files, or SHAP values.
- Record the run-specific Python and package versions from the actual environments. Do not assume one XGBoost version applies to every experiment.
- Explain that evaluation used repeated random patent splits and that tuning/evaluation patent overlap is a limitation; the runs are not a chronological future-cohort test.
- Keep the financial-report reporting-year/publication-date limitation visible in the documentation.
- Add a license only after deciding what reuse permissions you intend to grant. No license is included in this starter.
- Replace placeholders in the README with the final repository URL and the verified primary citation-data source.
- Run a clean-environment smoke test using authorized input files before describing the repository as fully reproducible.
