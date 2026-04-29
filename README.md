# NYC Accident Hotspot Benchmark

This repository now contains a multi-model NYC accident hotspot pipeline with:

- leakage-aware tabular feature engineering
- benchmark training across classic ML models
- sequence and spatial dataset builders for RNN and CNN baselines
- dual all-crash and severe-crash hotspot tracks
- SHAP-based hotspot explanation exports for the best tree-based tabular model on each track
- best-model promotion into the default prediction outputs

The project is still NYC-only and keeps 2022 as the held-out test year.

For a design-level walkthrough of the architecture, see
[SYSTEM_DESIGN.md](/home/virtual/stats-421/SYSTEM_DESIGN.md:1).

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If `geopandas` or related geo packages fail under `pip`, use a Conda env and
then install the Python requirements inside it.

## Main commands

Run the whole pipeline:

```bash
python3 -m hotspots.pipeline run-all
```

Run just the benchmark for selected models:

```bash
python3 -m hotspots.pipeline benchmark --models logreg,xgboost,svm_rbf
```

Run only one prediction track:

```bash
python3 -m hotspots.pipeline benchmark --targets severe
```

Promote the saved best model outputs again:

```bash
python3 -m hotspots.pipeline evaluate-best
```

Export Kepler.gl-ready hotspot layers and HTML:

```bash
python3 -m hotspots.pipeline export-kepler
```

## Stages

- `prepare-data`
- `build-grid`
- `build-tabular-features`
- `sample`
- `build-sequence-data`
- `build-spatial-data`
- `benchmark`
- `evaluate-best`
- `export-kepler`
- `plot`

`run-all` executes the full staged flow.

## Model roster

Tabular benchmark models:

- `logreg`
- `svm_rbf`
- `bagging_tree`
- `random_forest`
- `extra_trees`
- `hist_gb`
- `xgboost`

Neural portfolio models:

- `rnn_sequence`
- `cnn_spatial`

PyTorch is required for the neural models. If it is not installed, the tabular
benchmark can still run and the neural entries will fail with a clear message.

## Outputs

Processed data:

- `hotspots/data/processed/prepared_accidents.parquet`
- `hotspots/data/processed/grid_accidents.parquet`
- `hotspots/data/processed/cell_catalog.parquet`
- `hotspots/data/processed/nypd_hourly.parquet`
- `hotspots/data/processed/features.parquet`
- `hotspots/data/processed/sample.parquet`
- `hotspots/data/processed/sequence_data.npz`
- `hotspots/data/processed/spatial_data.npz`

Best-model outputs:

- `hotspots/outputs/best_model.json`
- `hotspots/outputs/best_model.pkl` or `hotspots/outputs/best_model.pt`
- `hotspots/outputs/explain_model.pkl`
- `hotspots/outputs/feature_columns.json`
- `hotspots/outputs/metrics.json`
- `hotspots/outputs/test_predictions.csv`
- `hotspots/outputs/top_hotspots.csv`
- `hotspots/outputs/shap_summary.png`
- `hotspots/outputs/hotspot_explanations.csv`

Severe-track outputs:

- `hotspots/outputs/severe/best_model.json`
- `hotspots/outputs/severe/best_model.pkl`
- `hotspots/outputs/severe/explain_model.pkl`
- `hotspots/outputs/severe/metrics.json`
- `hotspots/outputs/severe/test_predictions.csv`
- `hotspots/outputs/severe/top_hotspots.csv`
- `hotspots/outputs/severe/shap_summary.png`
- `hotspots/outputs/severe/hotspot_explanations.csv`

Benchmark artifacts:

- `hotspots/outputs/benchmark/leaderboard.csv`
- `hotspots/outputs/benchmark/metrics.json`
- `hotspots/outputs/benchmark/best_model_test_predictions.csv`
- `hotspots/outputs/benchmark/leaderboard.png`
- `hotspots/outputs/severe/benchmark/leaderboard.csv`
- `hotspots/outputs/severe/benchmark/metrics.json`
- `hotspots/outputs/severe/benchmark/best_model_test_predictions.csv`
- `hotspots/outputs/severe/benchmark/leaderboard.png`
- `hotspots/outputs/track_comparison.json`
- `hotspots/outputs/track_comparison.png`

Kepler artifacts:

- `hotspots/outputs/kepler/prediction_cells.geojson`
- `hotspots/outputs/kepler/top_hotspot_cells.geojson`
- `hotspots/outputs/kepler/kepler_config.json`
- `hotspots/outputs/kepler/nyc_hotspots_map.html`
- `hotspots/outputs/kepler/nyc_hotspots_leaflet.html`

Plots:

- `hotspots/outputs/metrics_summary.png`
- `hotspots/outputs/prediction_score_distribution.png`
- `hotspots/outputs/top_hotspots_map.png`
- `hotspots/outputs/hourly_risk_heatmap.png`
- `hotspots/outputs/severe/metrics_summary.png`
- `hotspots/outputs/severe/prediction_score_distribution.png`
- `hotspots/outputs/severe/top_hotspots_map.png`
- `hotspots/outputs/severe/hourly_risk_heatmap.png`

## Notes

- `hotspots/kaggledataset.py` is legacy scratch work and is not part of the
  runnable pipeline.
- The severe track uses `Kaggle Severity >= 3` as its target label.
- Neural models remain all-crash only in this phase; the severe track is
  tabular-only.
- SHAP exports are generated from the promoted tree-based tabular winner, or
  the best successful tree-based fallback if a non-tree model wins promotion.
