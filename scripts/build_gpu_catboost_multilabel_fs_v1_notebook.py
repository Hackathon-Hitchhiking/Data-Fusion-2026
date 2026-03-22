from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
NOTEBOOK_NAME = "data-fusion-2026-gpu-catboost-multilabel-fs-v1.ipynb"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output" / "kaggle-push" / "catboost-multilabel-fs-v1"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "catboost_multilabel_fs_v1_rev1_20260321"


def make_id(prefix: str, idx: int) -> str:
    return f"{prefix}-{idx:02d}"


def md_cell(text: str, idx: int) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "id": make_id("md", idx),
        "metadata": {},
        "source": [line for line in text.splitlines(keepends=True)],
    }


def code_cell(text: str, idx: int) -> dict[str, object]:
    return {
        "cell_type": "code",
        "id": make_id("code", idx),
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line for line in text.splitlines(keepends=True)],
    }


def validate_notebook_syntax(notebook: dict[str, object]) -> None:
    errors: list[str] = []
    for idx, cell in enumerate(notebook["cells"], start=1):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            ast.parse(source)
        except SyntaxError as exc:
            errors.append(f"cell {idx}: {exc}")
    if errors:
        raise SyntaxError("Generated notebook has invalid code cells:\n" + "\n".join(errors))


def main() -> None:
    bootstrap_lines = [
        "import importlib.util",
        "import json",
        "import subprocess",
        "import sys",
        "",
        f'NOTEBOOK_REVISION = "{NOTEBOOK_REVISION}"',
        "",
        "REQUIRED_PACKAGES = [",
        '    ("catboost", "catboost==1.2.10"),',
        '    ("pyarrow", "pyarrow"),',
        '    ("sklearn", "scikit-learn"),',
        "]",
        "",
        "def pip_install(args: list[str]) -> None:",
        '    subprocess.check_call([sys.executable, "-m", "pip", *args])',
        "",
        'missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]',
        'print({"stage": "bootstrap_start", "revision": NOTEBOOK_REVISION, "missing": missing}, flush=True)',
        "if missing:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing])',
        "",
        "from catboost.utils import get_gpu_device_count",
        'print({"stage": "bootstrap_done", "installed_now": missing, "gpu_count": int(get_gpu_device_count())}, flush=True)',
    ]

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU CatBoost Native Multilabel FS v1

                Objective:
                - Run native multilabel `CatBoost` on the strongest current compact feature dataset.
                - Select the final tree count by external holdout `macro ROC-AUC`, not by built-in `MultiLogloss`.

                Design:
                - One native `CatBoostClassifier` over all `41` labels with `MultiLogloss`.
                - Current best compact dataset: `199 main + 220 selected extra = 419 features`.
                - Explicit `Pool` objects with categorical feature names.
                - Holdout split aligned with the current `TabM` validation policy.
                - External checkpoint selection via validation `macro ROC-AUC` on `RawFormulaVal`.
                """
            ).strip()
            + "\n",
            1,
        ),
        code_cell("\n".join(bootstrap_lines) + "\n", 2),
        md_cell(
            dedent(
                """
                ## Run Mode

                - `RUN_MODE="auto"`:
                  - Kaggle with visible GPU -> `full_gpu`
                  - local machine -> `smoke`
                - Smoke mode validates the exact training code path on a much smaller sample.
                """
            ).strip()
            + "\n",
            3,
        ),
        code_cell(
            dedent(
                """
                from __future__ import annotations

                import gc
                import json
                import math
                import random
                import time
                from pathlib import Path

                import numpy as np
                import pandas as pd
                from catboost import CatBoostClassifier, Pool
                from catboost.utils import get_gpu_device_count
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit

                RUN_MODE = "auto"  # auto | smoke | full_gpu

                SMOKE_CFG = {
                    "seed": 42,
                    "sample_rows": 50000,
                    "smoke_min_positive": 12,
                    "val_fraction": 0.20,
                    "iterations": 300,
                    "learning_rate": 0.05,
                    "depth": 6,
                    "l2_leaf_reg": 8.0,
                    "bootstrap_type": "Bayesian",
                    "bagging_temperature": 1.0,
                    "one_hot_max_size": 4,
                    "max_ctr_complexity": 2,
                    "model_size_reg": 0.5,
                    "border_count": 128,
                    "od_wait": 80,
                    "coarse_step": 20,
                    "fine_step": 5,
                    "fine_window": 20,
                    "metric_period": 20,
                    "thread_count": 4,
                    "train_final_model": False,
                }

                FULL_CFG = {
                    "seed": 42,
                    "sample_rows": None,
                    "smoke_min_positive": 0,
                    "val_fraction": 0.10,
                    "iterations": 5000,
                    "learning_rate": 0.03,
                    "depth": 8,
                    "l2_leaf_reg": 8.0,
                    "bootstrap_type": "Bayesian",
                    "bagging_temperature": 1.0,
                    "one_hot_max_size": 4,
                    "max_ctr_complexity": 2,
                    "model_size_reg": 0.5,
                    "border_count": 254,
                    "od_wait": 300,
                    "coarse_step": 50,
                    "fine_step": 10,
                    "fine_window": 50,
                    "metric_period": 50,
                    "thread_count": 8,
                    "train_final_model": True,
                }


                def resolve_run_mode(mode: str) -> str:
                    if mode != "auto":
                        return mode
                    return "full_gpu" if Path("/kaggle/input").exists() and int(get_gpu_device_count()) > 0 else "smoke"


                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = SMOKE_CFG if RESOLVED_MODE == "smoke" else FULL_CFG
                ARTIFACT_DIR = Path("artifacts") / f"gpu_catboost_multilabel_fs_v1_{RESOLVED_MODE}"
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                random.seed(int(CFG["seed"]))
                np.random.seed(int(CFG["seed"]))

                print(
                    {
                        "resolved_mode": RESOLVED_MODE,
                        "gpu_count": int(get_gpu_device_count()),
                        "config": CFG,
                        "artifact_dir": str(ARTIFACT_DIR),
                    },
                    flush=True,
                )
                """
            ).strip()
            + "\n",
            4,
        ),
        md_cell("## Helpers\n", 5),
        code_cell(
            dedent(
                """
                def find_base_data_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "data" / "competition",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026"),
                        Path("/kaggle/input/data-fusion-2026-cybershelf"),
                    ]
                    required = [
                        "train_target.parquet",
                    ]
                    for candidate in direct_candidates:
                        if all((candidate / name).exists() for name in required):
                            return candidate
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for match in kaggle_input.rglob("train_target.parquet"):
                            candidate = match.parent
                            if all((candidate / name).exists() for name in required):
                                return candidate
                    raise FileNotFoundError("Could not locate the base parquet dataset directory.")


                def find_compact_feature_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "artifacts" / "feature_selection_v1",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                    ]
                    required = [
                        "train_compact_features_v1.parquet",
                        "test_compact_features_v1.parquet",
                        "final_feature_set.json",
                    ]
                    for candidate in direct_candidates:
                        if all((candidate / name).exists() for name in required):
                            return candidate
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for match in kaggle_input.rglob("train_compact_features_v1.parquet"):
                            candidate = match.parent
                            if all((candidate / name).exists() for name in required):
                                return candidate
                    raise FileNotFoundError("Could not locate the compact feature dataset directory.")


                def build_stratify_labels(target_df: pd.DataFrame) -> pd.Series:
                    return target_df.sum(axis=1).clip(upper=4).astype("int8").astype(str)


                def ensure_smoke_sample(
                    full_df: pd.DataFrame,
                    target_cols: list[str],
                    sample_rows: int,
                    min_positive: int,
                    seed: int,
                ) -> pd.DataFrame:
                    if sample_rows >= len(full_df):
                        return full_df.copy().reset_index(drop=True)

                    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_rows, random_state=seed)
                    base_idx, _ = next(splitter.split(full_df, build_stratify_labels(full_df[target_cols])))
                    selected = set(full_df.iloc[base_idx].index.tolist())
                    sample_df = full_df.loc[sorted(selected)].copy()

                    for target_name in target_cols:
                        current_pos = int(sample_df[target_name].sum())
                        if current_pos >= min_positive:
                            continue
                        need = min_positive - current_pos
                        extra_idx = full_df.index[(full_df[target_name] == 1) & (~full_df.index.isin(sample_df.index))]
                        if len(extra_idx) == 0:
                            continue
                        chosen = full_df.loc[extra_idx].sample(min(need, len(extra_idx)), random_state=seed).index.tolist()
                        selected.update(chosen)

                    return full_df.loc[sorted(selected)].reset_index(drop=True)


                def split_with_train_positive_coverage(
                    df: pd.DataFrame,
                    target_cols: list[str],
                    val_fraction: float,
                    seed: int,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
                    val_rows = max(1, int(round(len(df) * val_fraction)))
                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rows, random_state=seed)
                    train_idx, val_idx = next(splitter.split(df, build_stratify_labels(df[target_cols])))
                    train_idx = set(train_idx.tolist())
                    val_idx = set(val_idx.tolist())

                    for target_name in target_cols:
                        train_pos = int(df.iloc[list(train_idx)][target_name].sum())
                        if train_pos > 0:
                            continue
                        candidates = [idx for idx in val_idx if int(df.iloc[idx][target_name]) == 1]
                        if candidates:
                            moved = candidates[0]
                            val_idx.remove(moved)
                            train_idx.add(moved)

                    train_df = df.iloc[sorted(train_idx)].reset_index(drop=True)
                    val_df = df.iloc[sorted(val_idx)].reset_index(drop=True)
                    return train_df, val_df


                def prepare_feature_frames(
                    frames: list[pd.DataFrame],
                    num_cols: list[str],
                    cat_cols: list[str],
                ) -> list[pd.DataFrame]:
                    prepared: list[pd.DataFrame] = []
                    for frame in frames:
                        current = frame.copy()
                        for col in num_cols:
                            current[col] = pd.to_numeric(current[col], errors="coerce").astype("float32")
                        for col in cat_cols:
                            current[col] = pd.to_numeric(current[col], errors="coerce").fillna(-1).astype("int32")
                        prepared.append(current)
                    return prepared


                def macro_auc_from_raw(y_true: pd.DataFrame, raw_pred: np.ndarray, target_cols: list[str]) -> float:
                    scores = []
                    for idx, target_name in enumerate(target_cols):
                        y = y_true[target_name].to_numpy()
                        if np.unique(y).size < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y, raw_pred[:, idx])
                        scores.append(float(score))
                    return float(np.mean(scores))


                def per_target_auc_from_raw(y_true: pd.DataFrame, raw_pred: np.ndarray, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for idx, target_name in enumerate(target_cols):
                        y = y_true[target_name].to_numpy()
                        if np.unique(y).size < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y, raw_pred[:, idx])
                        rows.append({"target": target_name, "oof_auc": float(score)})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def evaluate_checkpoint_grid(
                    model: CatBoostClassifier,
                    valid_pool: Pool,
                    y_valid: pd.DataFrame,
                    target_cols: list[str],
                    coarse_step: int,
                    fine_step: int,
                    fine_window: int,
                ) -> tuple[int, float, pd.DataFrame]:
                    total_trees = int(model.tree_count_)
                    scored: dict[int, float] = {}

                    def score_checkpoint(ntree_end: int) -> float:
                        ntree_end = max(1, min(total_trees, int(ntree_end)))
                        if ntree_end not in scored:
                            raw_pred = np.asarray(
                                model.predict(valid_pool, prediction_type="RawFormulaVal", ntree_end=ntree_end),
                                dtype=np.float32,
                            )
                            scored[ntree_end] = macro_auc_from_raw(y_valid, raw_pred, target_cols)
                        return scored[ntree_end]

                    coarse_points = sorted(set(range(coarse_step, total_trees + 1, coarse_step)) | {total_trees})
                    coarse_best = max(coarse_points, key=score_checkpoint)
                    left = max(1, coarse_best - fine_window)
                    right = min(total_trees, coarse_best + fine_window)
                    fine_points = sorted(set(range(left, right + 1, fine_step)) | {coarse_best, total_trees})
                    best_trees = max(fine_points, key=score_checkpoint)
                    checkpoint_scores = pd.DataFrame(
                        [{"ntree_end": int(k), "macro_auc": float(v)} for k, v in sorted(scored.items())]
                    )
                    return int(best_trees), float(scored[best_trees]), checkpoint_scores
                """
            ).strip()
            + "\n",
            6,
        ),
        md_cell("## Train / Select / Predict\n", 7),
        code_cell(
            dedent(
                """
                base_data_dir = find_base_data_dir()
                compact_feature_dir = find_compact_feature_dir()

                train_features = pd.read_parquet(compact_feature_dir / "train_compact_features_v1.parquet")
                test_features = pd.read_parquet(compact_feature_dir / "test_compact_features_v1.parquet")
                target = pd.read_parquet(base_data_dir / "train_target.parquet")

                train_df = train_features.merge(target, on="customer_id", how="inner")
                target_cols = [c for c in train_df.columns if c.startswith("target_")]
                feature_cols = [c for c in train_features.columns if c != "customer_id"]
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
                num_cols = [c for c in feature_cols if c not in cat_cols]

                if CFG["sample_rows"] is not None:
                    train_df = ensure_smoke_sample(
                        full_df=train_df,
                        target_cols=target_cols,
                        sample_rows=int(CFG["sample_rows"]),
                        min_positive=int(CFG["smoke_min_positive"]),
                        seed=int(CFG["seed"]),
                    )

                train_part, valid_part = split_with_train_positive_coverage(
                    df=train_df,
                    target_cols=target_cols,
                    val_fraction=float(CFG["val_fraction"]),
                    seed=int(CFG["seed"]),
                )

                train_x, valid_x, test_x = prepare_feature_frames(
                    [train_part[feature_cols], valid_part[feature_cols], test_features[feature_cols]],
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                )
                train_y = train_part[target_cols].astype("int32")
                valid_y = valid_part[target_cols].astype("int32")

                train_pool = Pool(train_x, train_y, cat_features=cat_cols)
                valid_pool = Pool(valid_x, valid_y, cat_features=cat_cols)
                test_pool = Pool(test_x, cat_features=cat_cols)

                cat_cardinality = train_x[cat_cols].nunique(dropna=False).sort_values()
                print(
                    {
                        "base_data_dir": str(base_data_dir),
                        "feature_data_dir": str(compact_feature_dir),
                        "train_rows": int(len(train_part)),
                        "val_rows": int(len(valid_part)),
                        "test_rows": int(len(test_x)),
                        "feature_count": int(len(feature_cols)),
                        "num_features": int(len(num_cols)),
                        "cat_features": int(len(cat_cols)),
                        "cat_cardinality_summary": {
                            "min": int(cat_cardinality.min()),
                            "median": float(cat_cardinality.median()),
                            "p90": float(cat_cardinality.quantile(0.9)),
                            "max": int(cat_cardinality.max()),
                        },
                    },
                    flush=True,
                )

                task_type = "GPU" if RESOLVED_MODE == "full_gpu" else "CPU"
                params = {
                    "loss_function": "MultiLogloss",
                    "eval_metric": "MultiLogloss",
                    "task_type": task_type,
                    "devices": "0" if task_type == "GPU" else None,
                    "thread_count": int(CFG["thread_count"]),
                    "boosting_type": "Plain",
                    "grow_policy": "SymmetricTree",
                    "iterations": int(CFG["iterations"]),
                    "learning_rate": float(CFG["learning_rate"]),
                    "depth": int(CFG["depth"]),
                    "l2_leaf_reg": float(CFG["l2_leaf_reg"]),
                    "bootstrap_type": str(CFG["bootstrap_type"]),
                    "bagging_temperature": float(CFG["bagging_temperature"]),
                    "one_hot_max_size": int(CFG["one_hot_max_size"]),
                    "max_ctr_complexity": int(CFG["max_ctr_complexity"]),
                    "model_size_reg": float(CFG["model_size_reg"]),
                    "border_count": int(CFG["border_count"]),
                    "nan_mode": "Min",
                    "use_best_model": False,
                    "od_type": "Iter",
                    "od_wait": int(CFG["od_wait"]),
                    "metric_period": int(CFG["metric_period"]),
                    "verbose": int(CFG["metric_period"]),
                    "random_seed": int(CFG["seed"]),
                    "allow_writing_files": True,
                    "train_dir": str(ARTIFACT_DIR / "catboost_info"),
                    "save_snapshot": True,
                    "snapshot_file": str(ARTIFACT_DIR / "catboost_multilabel_fs_v1.snapshot"),
                    "snapshot_interval": 600,
                }
                if task_type != "GPU":
                    params.pop("devices")

                print({"stage": "catboost_fit_start", "params": params}, flush=True)
                fit_started = time.time()
                model = CatBoostClassifier(**params)
                model.fit(train_pool, eval_set=valid_pool)
                fit_seconds = time.time() - fit_started
                print(
                    {
                        "stage": "catboost_fit_done",
                        "tree_count": int(model.tree_count_),
                        "fit_seconds": round(float(fit_seconds), 2),
                        "best_score": model.get_best_score(),
                    },
                    flush=True,
                )

                best_trees, best_macro_auc, checkpoint_scores = evaluate_checkpoint_grid(
                    model=model,
                    valid_pool=valid_pool,
                    y_valid=valid_y,
                    target_cols=target_cols,
                    coarse_step=int(CFG["coarse_step"]),
                    fine_step=int(CFG["fine_step"]),
                    fine_window=int(CFG["fine_window"]),
                )
                checkpoint_scores.to_csv(ARTIFACT_DIR / "checkpoint_scores.csv", index=False)

                valid_raw = np.asarray(
                    model.predict(valid_pool, prediction_type="RawFormulaVal", ntree_end=best_trees),
                    dtype=np.float32,
                )
                valid_prob = np.asarray(
                    model.predict(valid_pool, prediction_type="Probability", ntree_end=best_trees),
                    dtype=np.float32,
                )

                target_scores = per_target_auc_from_raw(valid_y, valid_raw, target_cols)
                target_scores.to_csv(ARTIFACT_DIR / "target_scores.csv", index=False)

                validation_predictions = pd.DataFrame(
                    valid_prob,
                    columns=[c.replace("target_", "predict_") for c in target_cols],
                )
                validation_predictions.insert(0, "customer_id", valid_part["customer_id"].to_numpy())
                validation_predictions.to_parquet(ARTIFACT_DIR / "validation_predictions.parquet", index=False)

                metrics = {
                    "validation_macro_auc": float(best_macro_auc),
                    "resolved_mode": RESOLVED_MODE,
                    "device": task_type.lower(),
                    "rows_train": int(len(train_part)),
                    "rows_val": int(len(valid_part)),
                    "feature_count": int(len(feature_cols)),
                    "num_features": int(len(num_cols)),
                    "cat_features": int(len(cat_cols)),
                    "selected_ntree_end": int(best_trees),
                    "trained_tree_count": int(model.tree_count_),
                    "fit_seconds": float(fit_seconds),
                    "params": params,
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
                print(json.dumps(metrics, indent=2), flush=True)
                print(target_scores.head(12).to_string(index=False), flush=True)

                model.save_model(ARTIFACT_DIR / "validation_model.cbm")
                del valid_raw, valid_prob
                gc.collect()

                if CFG["train_final_model"]:
                    full_x, final_test_x = prepare_feature_frames(
                        [train_df[feature_cols], test_features[feature_cols]],
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                    )
                    full_y = train_df[target_cols].astype("int32")
                    full_pool = Pool(full_x, full_y, cat_features=cat_cols)
                    final_test_pool = Pool(final_test_x, cat_features=cat_cols)

                    final_params = dict(params)
                    final_params["iterations"] = int(best_trees)
                    final_params.pop("use_best_model", None)
                    final_params.pop("od_type", None)
                    final_params.pop("od_wait", None)
                    final_params["save_snapshot"] = False
                    final_params["snapshot_file"] = str(ARTIFACT_DIR / "catboost_multilabel_fs_v1_final.snapshot")

                    print({"stage": "catboost_full_fit_start", "iterations": int(best_trees)}, flush=True)
                    final_started = time.time()
                    final_model = CatBoostClassifier(**final_params)
                    final_model.fit(full_pool)
                    final_seconds = time.time() - final_started
                    print({"stage": "catboost_full_fit_done", "fit_seconds": round(float(final_seconds), 2)}, flush=True)

                    test_prob = np.asarray(final_model.predict(final_test_pool, prediction_type="Probability"), dtype=np.float32)
                    submission = pd.DataFrame(test_prob, columns=target_cols)
                    submission.insert(0, "customer_id", test_features["customer_id"].to_numpy())
                    submission.to_parquet(ARTIFACT_DIR / "submission.parquet", index=False)
                    final_model.save_model(ARTIFACT_DIR / "final_model.cbm")
                    (ARTIFACT_DIR / "full_train_metrics.json").write_text(
                        json.dumps({"selected_iterations": int(best_trees), "full_train_fit_seconds": float(final_seconds)}, indent=2)
                    )
                    print(
                        {
                            "stage": "submission_written",
                            "submission_path": str(ARTIFACT_DIR / "submission.parquet"),
                            "rows": int(len(submission)),
                            "cols": int(submission.shape[1]),
                        },
                        flush=True,
                    )
                else:
                    print("Smoke mode: final full-train submission step skipped.", flush=True)
                """
            ).strip()
            + "\n",
            8,
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    validate_notebook_syntax(notebook)

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=True, indent=2) + "\n")

    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK_PATH, PUSH_DIR / NOTEBOOK_NAME)
    kernel_metadata = {
        "id": "chesnikovleonid/data-fusion-2026-task-2-catboost-multilabel-fs-v1",
        "title": "Data Fusion 2026 task 2 catboost multilabel fs v1",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "keywords": [],
        "dataset_sources": [
            "chesnikovleonid/data-fusion-2026-2-task-dataset",
            "chesnikovleonid/data-fusion-2026-2-task-artifacts",
        ],
        "kernel_sources": [],
        "competition_sources": [],
        "model_sources": [],
        "machine_shape": "Gpu",
    }
    KERNEL_METADATA_PATH.write_text(json.dumps(kernel_metadata, indent=2) + "\n")
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {KERNEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
