# Geo-Temporal Traffic Accident Hotspot Prediction in New York City Using Tabular and Neural Spatiotemporal Models

## Abstract
Traffic crashes are a major public-safety problem, but safety interventions are easier to target when cities can identify where risk is likely to be concentrated before severe outcomes occur. This project builds a geo-temporal accident hotspot prediction pipeline for New York City using a `250m x 250m` spatial grid and hourly prediction buckets. The system combines a large crash-history source derived from the US-Accidents dataset with NYC collision enrichment data and FARS fatal-crash validation data. We benchmark logistic regression, support vector machines, tree ensembles, gradient boosting, a recurrent neural network (RNN), and a spatial convolutional neural network (CNN) under a shared train/validation/test policy. In the saved all-crash experiment outputs, the promoted model is an RNN sequence model, selected by a policy-oriented metric that measures how many observed crashes are captured inside the top `5%` of predicted areas. The saved all-crash winner achieves validation ROC AUC `0.9222`, validation average precision `0.8386`, validation top-`5%` capture `0.2503`, test ROC AUC `0.8977`, test average precision `0.7184`, and test top-`5%` capture `0.3180`. These results suggest that compact spatiotemporal neural models can be useful for ranking high-risk areas even when traditional tabular models remain competitive on conventional discrimination metrics.

## Introduction
Motor vehicle crashes remain one of the most persistent transportation safety problems in the United States. For a city transportation agency, the operational question is not just whether crashes happen, but where and when risk concentrates strongly enough to justify targeted intervention. In this project, that question is framed as **hotspot prediction**: given historical accident and context features, can a model predict which small regions of New York City are most likely to experience a crash in a given hour?

This problem matters because city resources are limited. Road redesigns, signal timing changes, targeted enforcement, and safety audits cannot happen everywhere at once. A useful model should therefore do more than score every location independently. It should rank locations in a way that helps planners focus on the most dangerous subset of the city. This project emphasizes that operational use case through a policy metric: the fraction of observed crashes captured in the top `5%` of predicted high-risk areas.

Prior work on traffic safety has followed several related directions. One line of research focuses on **crash hotspot identification** using spatial statistics such as kernel density estimation and kriging, which are useful for describing historical concentration patterns but are less directly suited to forward-looking prediction (Thakali et al., 2015). A second line of work uses **machine learning** to predict crash frequency or severity from structured features such as time, weather, and road conditions (Butt & Shafique, 2025; the 2023 Heliyon severity-comparison paper). A third line of work uses **spatiotemporal deep learning** to capture temporal dependencies and local spatial structure, often through recurrent, convolutional, or hybrid neural architectures (Kashifi et al., 2022). A separate but increasingly important concern is **explainability**, especially when city planners need interpretable evidence for why a location was flagged as risky. SHAP is one widely used framework for local feature attribution in complex models (Lundberg & Lee, 2017).

This project is not a nationwide forecasting system. Instead, it is an NYC-focused benchmark pipeline built on top of a country-scale accident source and local enrichment data. The main data backbone comes from US-Accidents, a widely used public crash dataset that aggregates traffic incident records with temporal and environmental context (Moosavi et al., 2019). NYC-specific collision history is used to enrich the features, and the Fatality Analysis Reporting System (FARS) is used as a post-hoc validation source for overlap with fatal-crash areas (National Highway Traffic Safety Administration [NHTSA], n.d.).

### Contributions
This project makes four practical contributions:

1. It reformulates NYC crash prediction as an hourly **grid-cell hotspot ranking** problem using `250m x 250m` cells rather than only intersection-level summaries.
2. It builds a leakage-aware benchmark pipeline that compares classic tabular machine learning models against lightweight neural sequence and spatial baselines under the same split policy.
3. It evaluates models using both standard ranking metrics and a more operational **top-`5%` capture** metric that reflects the reality of limited safety resources.
4. It produces an end-to-end artifact pipeline from raw data through processed datasets, predictions, ranked hotspots, and map-ready exports for downstream planning analysis.

Plain-language summary: the project chops New York City into many small squares, learns from old crash patterns, and then ranks which squares look most dangerous for a future hour.

## Methods
### Problem Formulation
The prediction unit is a pair `(cell_id, bucket_start)`, where `cell_id` identifies a `250m x 250m` grid cell and `bucket_start` identifies a real hourly timestamp. A positive label means that at least one crash occurred in that cell during that hour. Formally, for cell `c` and hour `t`, the all-crash label is:

`y(c, t) = 1` if at least one crash occurs in cell `c` during hour `t`, and `0` otherwise.

Although the codebase now also supports a severe-crash track, this paper focuses on the **completed all-crash results** because those are the only fully generated outputs available in the current workspace.

### Data Sources
The pipeline uses three local datasets:

1. **US-Accidents / Kaggle derivative**: the main crash-history source, derived from the public US-Accidents collection described by Moosavi et al. (2019). In this project it provides crash time, location, severity, basic weather labels, and day/night context.
2. **NYC Open Data collision reports**: used for city-specific enrichment, including prior injuries, fatalities, and contributing-factor patterns such as speeding, distraction, aggressive driving, or alcohol involvement (NYC Open Data, n.d.).
3. **FARS**: used after prediction as a fatal-crash validation source to measure whether predicted hotspots overlap areas with known severe outcomes (NHTSA, n.d.).

The current saved run does **not** add numeric NOAA weather fields, OpenStreetMap road-network attributes, or population-exposure features. Those are part of the future-work agenda rather than the completed experiment reported here.

### Spatial and Temporal Representation
Accident points are projected into `EPSG:32618`, then assigned to fixed `250m` grid cells. Time is floored to the nearest hour, so each learning row corresponds to one real cell-hour pair. This representation has two advantages. First, it preserves true time ordering for leakage-safe history features. Second, it supports both sequence models and spatial patch models without changing the underlying task definition.

### Feature Engineering
The saved benchmark uses a tabular feature set containing temporal, historical, and neighborhood signals. The main feature groups are:

- **Current-time temporal features**: hour of day, day of week, month, weekend indicator, rush-hour indicator, and night indicator.
- **Cell location features**: centroid latitude/longitude and borough indicators.
- **Historical crash features**: prior crash counts, recent rolling crash counts, prior severe counts, same-hour recurrence, same-day-and-hour recurrence, time since the last crash, and prior average severity.
- **Neighborhood features**: historical neighbor crash totals, hotspot density, and observed neighboring-cell count.
- **NYPD enrichment features**: prior collision, injury, fatality, aggressive-driving, distracted-driving, failure-to-yield, speeding, and alcohol-related counts.
- **Coarse weather-history features**: prior rain, snow, and fog histories derived from the in-file weather labels.

The feature design is intentionally historical. For any row being predicted, only information available before that hour is used when building lagged or cumulative features.

Plain-language summary: the model looks at clues like time of day, whether the area has crashed before, whether nearby cells are risky, and whether NYPD history suggests repeated dangerous behavior in that area.

### Sampling and Split Policy
The pipeline keeps years `2020` through `2022`, with `2022` held out as test data. Historical rows from `2020` and `2021` feed train and validation, with a validation fraction of `0.20`. The processed sample also uses class balancing and negative sampling to keep the task computationally manageable:

- historical sample fraction: `0.20`
- positive-to-negative ratio: `4`
- negative-pool ratio: `8`

This design keeps the learning problem laptop-friendly while preserving a clean temporal boundary between model development and final testing.

### Model Families
The benchmark compares nine models:

- Logistic regression
- RBF-kernel support vector machine
- Bagging tree ensemble
- Random forest
- Extra trees
- Histogram gradient boosting
- XGBoost
- RNN sequence model
- CNN spatial model

The neural models are deliberately lightweight. The RNN uses an `8`-step lookback sequence, and the CNN uses a local spatial patch with radius `1` around the target cell. This makes the benchmark feasible without requiring a large GPU training setup.

### Selection Rule and Evaluation
Each model is evaluated on:

- ROC AUC
- average precision (AP)
- top-`5%` capture

The promoted winner is chosen by the rule:

`validation top_5pct_capture -> validation average_precision -> validation roc_auc`

This is an important design decision. It means the project prioritizes **policy usefulness** over a pure discrimination metric. In practice, the goal is to rank the top set of places that a planner might inspect or intervene on, not only to maximize average pairwise ranking performance across all rows.

After promotion, the pipeline exports canonical predictions, ranked hotspots, plots, and a FARS overlap statistic. The current saved `test_predictions.csv` contains `152,860` test prediction rows, and the saved `top_hotspots.csv` contains `168` ranked hotspot cells.

## Experiments
### Experimental Setup
All experiments in this paper come from the saved all-crash benchmark artifacts in `hotspots/outputs/` and `hotspots/outputs/benchmark/`. The benchmark leaderboard includes successful runs for all nine model families. The most important comparison is not just which model has the highest ROC AUC, but which model captures the greatest share of observed crashes in the top `5%` of predicted areas.

### Benchmark Results
Table 1 summarizes the saved all-crash results.

| Model | Family | Validation ROC AUC | Validation AP | Validation Top-5% Capture | Test ROC AUC | Test AP | Test Top-5% Capture |
|---|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Tabular | 0.9189 | 0.8361 | 0.0061 | 0.8843 | 0.6976 | 0.0504 |
| SVM (RBF) | Tabular | 0.9409 | 0.8366 | 0.0051 | 0.8922 | 0.6436 | 0.0489 |
| Bagging Tree | Tabular | 0.9630 | 0.8901 | 0.0061 | 0.8967 | 0.7261 | 0.0494 |
| Random Forest | Tabular | 0.9574 | 0.8741 | 0.0112 | 0.8986 | 0.7187 | 0.0461 |
| Extra Trees | Tabular | 0.9474 | 0.8692 | 0.0071 | 0.9075 | 0.7306 | 0.0488 |
| HistGradientBoosting | Tabular | 0.9673 | 0.8984 | 0.0051 | 0.9062 | 0.7520 | 0.0493 |
| XGBoost | Tabular | 0.9661 | 0.8965 | 0.0061 | 0.9134 | 0.7517 | 0.0500 |
| **RNN Sequence** | **Neural Sequence** | **0.9222** | **0.8386** | **0.2503** | **0.8977** | **0.7184** | **0.3180** |
| CNN Spatial | Neural Spatial | 0.8912 | 0.6902 | 0.2146 | 0.8476 | 0.5370 | 0.2279 |

The strongest conventional metrics on the tabular side come from boosted/tree models, especially HistGradientBoosting and XGBoost. On validation ROC AUC and validation AP, those models outperform the RNN. However, the RNN dominates on the project’s chosen policy metric, top-`5%` capture. Because the system explicitly promotes models using that metric first, the RNN becomes the saved best model.

### Interpreting the Winner
The most important takeaway is that “best” depends on what the city wants to optimize. If the target is overall ranking quality across all test rows, XGBoost and HistGradientBoosting are very strong candidates. If the target is to identify a small set of highly concentrated risk cells for intervention, the RNN performs much better on the chosen policy metric. This suggests that short-range temporal recurrence inside a cell may be especially valuable when the output is used for hotspot targeting rather than broad probability calibration.

One plausible explanation is that the RNN is better at capturing **localized temporal persistence**. The sequence representation gives the model direct access to recent lagged history over an `8`-step lookback window, which may help it recognize repeating hourly patterns that are not as directly expressed in a static tabular row. By contrast, the tabular models rely on manually engineered lag summaries. Those summaries are still informative, but they may smooth away some fine-grained recurrence patterns.

### Final Saved Outputs
The promoted all-crash model metadata records:

- saved best model: `rnn_sequence`
- feature set: `tabular_v2`
- selection rule: `validation top_5pct_capture -> average_precision -> roc_auc`
- explainability fallback model: `random_forest`

The saved best-model metrics file reports:

- validation ROC AUC: `0.9219`
- validation AP: `0.8386`
- validation top-`5%` capture: `0.2503`
- test ROC AUC: `0.8967`
- test AP: `0.7168`
- test top-`5%` capture: `0.3178`
- FARS hotspot overlap: `0.02994`

The FARS overlap value is modest, which is a useful cautionary finding. It suggests the model is learning meaningful spatial-temporal crash risk for the all-crash task, but that alignment with fatal-crash concentration remains limited. That is not necessarily a failure, because the optimization target in this run is general crash occurrence rather than fatality prediction. Still, it supports the motivation for a later severity-aware modeling track.

Plain-language summary: the RNN did not win because it was best at every metric. It won because it was best at finding a small set of places where many real crashes happened, which matches the project’s practical goal.

## Discussion
This project demonstrates that a relatively compact pipeline can move from raw crash data to map-ready NYC hotspot rankings while comparing multiple modeling families under one evaluation policy. That is already useful for a course project because it turns a broad transportation-safety question into a reproducible experimental workflow.

The strongest result is not simply “deep learning wins.” The stronger conclusion is narrower: **an RNN sequence model was the best model under the project’s chosen intervention-oriented metric**, even though tree-based models were stronger on ROC AUC and AP. That distinction matters. It shows how evaluation design changes the story. If the deployment goal is to highlight a small number of inspection targets, then top-`5%` capture may be more meaningful than a standard metric alone.

At the same time, several limitations remain:

1. The completed saved outputs in this workspace are all-crash only. The codebase now supports a severe-crash track, but those artifacts were not generated successfully in the current checkout and therefore are not treated as completed experimental evidence here.
2. The data pipeline currently relies on coarse in-file weather labels rather than numeric NOAA variables such as precipitation amount, temperature, or visibility.
3. The current run does not yet include OpenStreetMap road-network features such as intersection density, speed limits, or traffic control details.
4. Population or exposure features, such as pedestrian activity or traffic-volume proxies, are not part of the completed benchmark.
5. Although the codebase includes SHAP-related logic and records a tree-based fallback explainer model, the saved explanation artifacts are not present in the current outputs directory and therefore are best described as future work rather than finished results.

These limitations point to a clear next phase. A stronger follow-up system would combine the current grid-and-history setup with numeric NOAA weather data, road-network covariates from OSM, and a fully generated severe-hotspot pipeline. That combination would make the predictions more causal, more interpretable, and more relevant to injury-prevention planning.

## Conclusion
This paper presented an NYC accident hotspot prediction benchmark that converts crash history into hourly predictions over `250m x 250m` map cells. The system compares linear, kernel, tree-based, boosted, recurrent, and spatial neural models under one leakage-aware split policy and one intervention-oriented model-selection rule. In the completed all-crash results, the RNN sequence model is the promoted winner because it captures substantially more observed crashes inside the top `5%` of predicted high-risk areas than the tabular baselines. In plain language, the model is useful because it helps narrow a very large city into a much smaller set of places that deserve attention first. The current results are promising, but the next meaningful step is to complete severity-aware outputs and explanation artifacts so the system can better support high-stakes traffic-safety planning.

## What Is Still Missing
### Missing Generated Results
- `hotspots/outputs/severe/` artifact set for the completed severe track.
- `hotspots/outputs/track_comparison.json` and the associated cross-track plot.
- SHAP export artifacts such as `shap_summary.png` and `hotspot_explanations.csv` for the current saved run.

**Resource / source of truth:** current repo outputs in `hotspots/outputs/`, plus the SHAP paper by Lundberg and Lee (2017) for explainability framing.

### Missing Methodological Scope
- Numeric NOAA weather enrichment, especially precipitation, temperature, and visibility fields.
- OpenStreetMap road-network features such as intersection density, speed limits, and traffic-control information.
- Population or exposure features such as pedestrian intensity or traffic-volume proxies.

**Resource / source of truth:** NOAA climate data documentation, NYC Open Data collision metadata, and any future road-network extraction workflow built on OpenStreetMap.

### Missing Paper Polish
- One final benchmark table formatted for the paper rather than copied from CSV.
- One simple pipeline figure that shows raw data to grid to features to models to hotspot map.
- One hotspot map figure from the generated outputs.
- A cleaned reference list with final publication metadata where possible.
- A brief limitations paragraph tied directly to the currently absent severe-track and SHAP artifacts.

**Resource / source of truth:** the current benchmark CSV and plot outputs in the repo, plus the official dataset pages listed below.

## References
Butt, M. S., & Shafique, M. A. (2025). *A literature review: AI models for road safety for prediction of crash frequency and severity*. Discover Civil Engineering. https://link.springer.com/article/10.1007/s44290-025-00255-3

Kashifi, M. T., Alturki, M., Sharifi, A. W., et al. (2022). *Deep Hybrid Learning Framework for Spatiotemporal Crash Prediction Using Big Traffic Data*. International Journal of Transportation Science and Technology. https://www.researchgate.net/publication/362047375_Deep_Hybrid_Learning_Framework_for_Spatiotemporal_Crash_Prediction_Using_Big_Traffic_Data

Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems. https://proceedings.neurips.cc/paper/7062-a-unified-approach-to-interpreting-model-predictions

Moosavi, S., Samavatian, M. H., Parthasarathy, S., & Ramnath, R. (2019). *A Countrywide Traffic Accident Dataset*. https://smoosavi.org/datasets/us_accidents

National Highway Traffic Safety Administration. (n.d.). *Fatality Analysis Reporting System (FARS)*. https://www.nhtsa.gov/es/research-data/fatality-analysis-reporting-system-fars

NYC Open Data. (n.d.). *NYPD Motor Vehicle Collisions (Full version)*. https://data.cityofnewyork.us/Public-Safety/NYPD-Motor-Vehicle-Collisions-Full-version-/3tta-b6xn/about

NOAA National Centers for Environmental Information. (n.d.). *Daily Summaries: Climate Data Online*. https://www.ncdc.noaa.gov/cdo-web/datasets/GHCND/locations/FIPS%3AUS/detail

Thakali, L., Kwon, T. J., & Fu, L. (2015). *Identification of crash hotspots using kernel density estimation and kriging methods: a comparison*. Journal of Modern Transportation. https://link.springer.com/article/10.1007/s40534-015-0068-0

*Evaluating the effectiveness of machine learning techniques in forecasting the severity of traffic accidents*. (2023). *Heliyon, 9*(8), e18812. https://www.sciencedirect.com/science/article/pii/S2405844023060206
