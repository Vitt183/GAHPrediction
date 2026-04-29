Geo-temporal Accident Hotspot Prediction

The project now centers on a benchmark-driven NYC hotspot pipeline rather than a
single baseline model.

Current system:

- geography: NYC only
- target: hourly crash-risk prediction per 250m grid cell
- split policy: 2020-2021 for train and validation, 2022 held out for test
- feature tracks:
  - tabular engineered features from Kaggle accidents plus NYC NYPD enrichment
  - sequence tensors for an RNN baseline
  - spatial patch tensors for a CNN baseline
- benchmark roster:
  - logistic regression
  - RBF SVM
  - bagging trees
  - random forest
  - extra trees
  - histogram gradient boosting
  - XGBoost
  - RNN
  - CNN

Main artifacts:

- processed feature datasets in `hotspots/data/processed/`
- benchmark leaderboard in `hotspots/outputs/benchmark/leaderboard.csv`
- promoted best-model predictions in `hotspots/outputs/test_predictions.csv`
- promoted hotspot rankings in `hotspots/outputs/top_hotspots.csv`

Main commands:

```bash
python3 -m hotspots.pipeline run-all
python3 -m hotspots.pipeline benchmark --models logreg,xgboost,svm_rbf
python3 -m hotspots.pipeline evaluate-best
```
