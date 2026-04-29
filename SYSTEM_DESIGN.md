# System Design

## Overview

This pipeline predicts hourly NYC traffic accident risk at `250m x 250m` grid
resolution. It now supports two aligned prediction tracks:

- `all_crash`
- `severe`

The system is designed as a staged benchmark pipeline rather than a single
model script.

The system has four main responsibilities:

1. ingest and normalize raw crash data
2. build aligned spatial-temporal learning datasets
3. benchmark multiple model families on the same split policy
4. promote the best model for each track into stable outputs used for analysis and plots
5. generate explainability artifacts for the best tree-based tabular model on each track

The current implementation is centered in
[hotspots/pipeline.py](/home/virtual/stats-421/hotspots/pipeline.py:1) and
configured through
[hotspots/config.py](/home/virtual/stats-421/hotspots/config.py:1).

## Design Goals

- Keep the workflow rerunnable from raw data to final outputs
- Use leakage-safe historical features
- Compare model families under a consistent evaluation policy
- Support both practical tabular models and portfolio neural baselines
- Export stable artifacts so downstream analysis reads one canonical set of
  outputs

## Inputs

The system reads three local datasets:

- `US_Accidents_March23.csv`
  Primary training source for crash events, severity, timing, coordinates, and
  in-file weather labels.
- `Motor_Vehicle_Collisions_-_Crashes.csv`
  NYC enrichment source for injury, fatality, and contributing-factor history.
- `accident.csv`
  FARS validation source for post-hoc hotspot overlap checks.

No online data fetch is required during the main pipeline run.

## Core Data Model

### Spatial unit

Each event is assigned to a `cell_id` derived from projected `grid_x` and
`grid_y` coordinates in `EPSG:32618`.

### Temporal unit

Each learning row represents one real hourly timestamp:

- `bucket_start`
- one `cell_id`
- `label` for whether at least one crash occurred in that cell during that hour
- `severe_label` for whether at least one Kaggle event with `Severity >= 3`
  occurred in that cell during that hour

This is intentionally different from a recurring template bucket. The system
needs real time ordering so that:

- historical feature engineering is leakage-safe
- sequence models can use true lookback windows
- spatial patches align to an exact prediction hour

## Pipeline Stages

### 1. `prepare-data`

Reads the Kaggle accident CSV in chunks, filters to New York State, narrows to
NYC using city/county logic, normalizes timestamps, derives borough labels, and
maps raw weather descriptions into coarse categories.

Output:

- `hotspots/data/processed/prepared_accidents.parquet`

### 2. `build-grid`

Projects accident points into `EPSG:32618`, assigns them to 250m cells, floors
timestamps to the hour, and aggregates accident events into one positive row per
`cell_id x bucket_start`.

This stage also writes a unique cell catalog used later by negative sampling and
spatial-neighbor logic.

The aggregated grid table now carries both targets:

- `label`
- `severe_label`

Outputs:

- `hotspots/data/processed/grid_accidents.parquet`
- `hotspots/data/processed/cell_catalog.parquet`

### 3. `build-tabular-features`

Constructs the benchmark feature table.

This stage:

- creates sampled negative `cell_id x bucket_start` rows
- merges NYPD hourly enrichment
- computes static cell-level neighborhood statistics
- computes leakage-safe historical features using only prior rows
- writes the final tabular feature matrix

Feature groups:

- current-time temporal features
- cell centroid features
- borough indicators
- prior crash/severity/night/weather histories
- prior NYPD collision/injury/factor histories
- neighboring-cell aggregate risk features

Outputs:

- `hotspots/data/processed/nypd_hourly.parquet`
- `hotspots/data/processed/features.parquet`
- `hotspots/outputs/feature_columns.json`

### 4. `sample`

Applies the benchmark split policy:

- `2020-2021` feed train and validation
- `2022` stays untouched as test

Historic rows are stratified and class-balanced before the train/validation
split so the learning problem remains laptop-friendly.

Output:

- `hotspots/data/processed/sample.parquet`

The sampled dataset keeps both `label` and `severe_label` so the all-crash and
severe tracks are directly comparable on the same rows.

### 5. `build-sequence-data`

Builds a fixed-length lookback tensor for each sampled row, grouped by cell and
ordered by time.

Each sample contains:

- `lookback x sequence_feature_count`
- aligned row IDs
- labels by split

Output:

- `hotspots/data/processed/sequence_data.npz`

### 6. `build-spatial-data`

Builds a local spatial neighborhood tensor for each sampled row using the target
cell and nearby cells at the same prediction hour.

Each sample contains:

- `channel_count x patch_height x patch_width`
- aligned row IDs
- labels by split

Output:

- `hotspots/data/processed/spatial_data.npz`

### 7. `benchmark`

Runs all requested models against the aligned datasets and records metrics in a
shared format.

Track behavior:

- all-crash track:
  - tabular benchmark models
  - RNN baseline
  - CNN baseline
- severe track:
  - tabular benchmark models only

Tabular models train from `sample.parquet` using either `label` or
`severe_label`.
Sequence and spatial neural models train from the `.npz` datasets on the
all-crash track only.

The benchmark then:

- computes validation and test metrics
- records failures without aborting the entire run
- selects the best successful model per track
- saves promoted-model metadata
- saves the best tree-based tabular model for SHAP export
- triggers per-track output export

Outputs:

- `hotspots/outputs/benchmark/leaderboard.csv`
- `hotspots/outputs/benchmark/metrics.json`
- `hotspots/outputs/benchmark/best_model_test_predictions.csv`

### 8. `evaluate-best`

Takes the promoted model’s predictions and turns them into the canonical output
files used by the rest of the project.

This stage:

- writes `test_predictions.csv` per track
- aggregates `top_hotspots.csv` per track
- computes summary metrics per track
- computes optional FARS overlap per track
- generates SHAP and textual explanation artifacts per track when a tree-based
  explainer model is available
- generates per-track plots and an all-crash vs severe comparison summary

Outputs:

- `hotspots/outputs/test_predictions.csv`
- `hotspots/outputs/top_hotspots.csv`
- `hotspots/outputs/metrics.json`
- `hotspots/outputs/shap_summary.png`
- `hotspots/outputs/hotspot_explanations.csv`
- `hotspots/outputs/severe/*`
- plot PNGs

### 9. `export-kepler`

Builds Kepler.gl-ready geospatial artifacts from the promoted best-model
outputs without retraining any models.

This stage:

- reads the canonical prediction and hotspot outputs
- reconstructs each `250m x 250m` prediction cell as a polygon in `EPSG:4326`
- writes time-aware and aggregated hotspot GeoJSON layers
- saves a Kepler configuration and HTML map export

Outputs:

- `hotspots/outputs/kepler/prediction_cells.geojson`
- `hotspots/outputs/kepler/top_hotspot_cells.geojson`
- `hotspots/outputs/kepler/kepler_config.json`
- `hotspots/outputs/kepler/nyc_hotspots_map.html`

## Model Architecture

### Tabular track

These are the expected practical workhorses:

- `logreg`
- `svm_rbf`
- `bagging_tree`
- `random_forest`
- `extra_trees`
- `hist_gb`
- `xgboost`

All tabular models consume the same engineered feature columns from the sample
table.

### Neural track

These are included as valid portfolio baselines, not assumed winners:

- `rnn_sequence`
- `cnn_spatial`

The RNN consumes per-cell temporal sequences.
The CNN consumes local spatial patches.

PyTorch is optional at the environment level but required for those two models.
If PyTorch is missing, the benchmark logs model-level failures and continues
with the remaining models.

## Model Selection Policy

The benchmark promotes exactly one best model using validation metrics only.

Selection priority:

1. `validation_top_5pct_capture`
2. `validation_average_precision`
3. `validation_roc_auc`

Test metrics are reported after selection and are not used to choose the winner.

If the promoted model is not tree-based, explainability is delegated to the
best successful tree-based tabular model for that same track.

## Evaluation Design

The main policy metric is:

- what fraction of true crashes fall inside the top 5% of predicted cells

Tracked metrics:

- ROC AUC
- average precision
- top-5% capture
- optional FARS hotspot overlap

The pipeline also writes a track comparison summary covering:

- side-by-side test metrics
- FARS overlap
- average predicted concentration among the top hotspot cells

This makes the system useful both as a machine-learning benchmark and as a
decision-support ranking pipeline.

## Failure Handling

The pipeline is intentionally tolerant in a few places:

- missing PyTorch only disables the neural models
- invalid FARS coordinates do not crash the benchmark
- duplicate cell catalog rows are deduplicated defensively

This keeps the end-to-end run resilient enough for iterative development.

## Key Artifacts

### Processed data

- `prepared_accidents.parquet`
- `grid_accidents.parquet`
- `cell_catalog.parquet`
- `nypd_hourly.parquet`
- `features.parquet`
- `sample.parquet`
- `sequence_data.npz`
- `spatial_data.npz`

### Benchmark outputs

- `leaderboard.csv`
- `benchmark/metrics.json`
- `best_model_test_predictions.csv`

### Promoted outputs

- `best_model.json`
- `best_model.pkl` or `best_model.pt`
- `test_predictions.csv`
- `top_hotspots.csv`
- `metrics.json`

## Extension Points

The easiest next extensions are:

- richer road-network features
- better negative-sampling strategy
- stronger tree-model tuning
- a more formal experiment registry

The architecture is already staged so these can be added without rewriting the
entire pipeline.
