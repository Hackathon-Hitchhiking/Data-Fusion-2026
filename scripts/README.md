# Scripts Map

## Entry points

Current runnable scripts remain flat in [scripts](/Users/rebelraider/python-projects/Hackatons/Data-Fusion-2026/scripts) to avoid breaking long-running experiments and existing commands.

Main groups:

- Training: `train_*`, `run_catboost_*`, `run_structured_meta_experiments.py`
- Meta-correction research: `run_tabm_error_corrector.py`, `run_object_level_gate.py`, `run_family_head_pilot.py`
- Feature-view research: `run_feature_view_rank_quantile.py`
- Feature selection: `select_target_extra_features.py`, `build_hybrid_target_feature_map.py`
- Kaggle dataset management: `manage_kaggle_dataset.py`
- Submission/inference: `predict_*`, `stack_oof.py`, `build_*submission.py`
- Analysis: `eda.py`, `analyze_prediction_complementarity.py`
- Kaggle ops: `kaggle_kernel_snapshot.py`
- Notebook generators: `build_gpu_*_notebook.py`

## Shared DRY library

Reusable helpers now live in [scripts/lib](/Users/rebelraider/python-projects/Hackatons/Data-Fusion-2026/scripts/lib):

- [layout.py](/Users/rebelraider/python-projects/Hackatons/Data-Fusion-2026/scripts/lib/layout.py)
  Project root, `data/competition` resolution, sample submit lookup.
- [data_loading.py](/Users/rebelraider/python-projects/Hackatons/Data-Fusion-2026/scripts/lib/data_loading.py)
  Shared parquet loaders and merged train/test frame helpers.
- [metrics.py](/Users/rebelraider/python-projects/Hackatons/Data-Fusion-2026/scripts/lib/metrics.py)
  Shared `macro_auc`.
- [submission.py](/Users/rebelraider/python-projects/Hackatons/Data-Fusion-2026/scripts/lib/submission.py)
  Prediction column normalization and sample-aligned submission writing.

## Refactor rule

- Any new experiment script should reuse `scripts/lib` instead of duplicating path loading, metric logic, or submission formatting.
