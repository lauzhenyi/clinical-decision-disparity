# Clinical Decision Disparity

This repository contains the analysis pipeline for estimating presentation-conditioned clinical decision disparities in MIMIC-IV and MIMIC-IV-ED. It compares structured-only survival models with structured + chief-complaint text models to ask whether patients with similar observed presentations receive radiology orders, vasopressors, or invasive ventilation at different rates across gender, race, and language groups.

## Pipeline

- `00_data_processing_survival.ipynb`: converts raw MIMIC tables and builds radiology, vasopressor, and ventilation time-to-event cohorts.
- `01_build_cached_inputs.ipynb`: creates shared imputed analytic inputs and TF-IDF/SVD chief-complaint features.
- `02_radiology.ipynb`: estimates radiology-order disparities and radiology robustness outputs.
- `03_icu_pressor_vent.ipynb`: estimates vasopressor and ventilation disparities.
- `02_delete_a_group_jackknife.py` / `03_delete_a_group_jackknife.py`: refit delete-a-group jackknife models and summarize uncertainty for disparity and keyword results.
- `04_visualization.ipynb`: generates descriptive figures, main disparity plots, and keyword-decomposition figures.
- `05_robustness_reader.ipynb`: collects robustness checks, diagnostics, and export tables.
- `06_construct_robustness.ipynb` / `07_cc_construct_analysis.ipynb`: label chief-complaint components and analyze patient-reported non-trauma complaint construct scores.

## Outputs

The pipeline writes cached inputs, model artifacts, jackknife summaries, robustness tables, and figures under `main/` and `figures/`. Data are expected to be available locally through the user's MIMIC access.
