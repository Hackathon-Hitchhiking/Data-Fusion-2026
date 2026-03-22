# Baseline quick notes

Downloaded competition files from README links into `data/competition/` and inspected `baseline_catboost.ipynb`.

## What baseline does
- Loads `train_main_features.parquet`, `test_main_features.parquet`, `train_target.parquet`.
- Detects categorical features by prefix `cat_feature` and casts them to `Int32`.
- Builds CatBoost `Pool` with all train rows and all 41 targets (multi-label setup).
- Trains `CatBoostClassifier` with:
  - `iterations=10`
  - `depth=4`
  - `learning_rate=0.25`
  - `loss_function='MultiLogloss'`
  - `nan_mode='Min'`
  - `random_seed=1234`
- Predicts on test with `prediction_type='RawFormulaVal'`.
- Renames target columns from `target_*` to `predict_*` and writes parquet submit.

## Notes
- Notebook writes output parquet in submit-like format; in this repo the template file now lives at `data/competition/sample_submit.parquet`.
- This is a very simple baseline (no validation split / hyperparameter tuning).
