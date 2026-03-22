from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
PROJECT_ROOT_LITERAL = str(ROOT)
NOTEBOOK_NAME = "data-fusion-2026-histgb-ovr-fs-v2-missing-lite.ipynb"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output" / "kaggle-push" / "histgb-ovr-fs-v2-missing-lite"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "histgb_ovr_fs_v2_missing_lite_rev1_20260322"


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
        '    ("pyarrow", "pyarrow"),',
        '    ("sklearn", "scikit-learn"),',
        "]",
        "",
        "def pip_install(args: list[str]) -> None:",
        '    subprocess.check_call([sys.executable, "-m", "pip", *args])',
        "",
        "missing = []",
        "for module_name, pip_name in REQUIRED_PACKAGES:",
        "    if importlib.util.find_spec(module_name) is None:",
        "        missing.append(pip_name)",
        "",
        'print({"stage": "bootstrap_start", "revision": NOTEBOOK_REVISION, "missing": missing}, flush=True)',
        "if missing:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing])',
        'print({"stage": "bootstrap_done", "installed_now": missing}, flush=True)',
    ]

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 HistGradientBoosting OvR FS v2 Missing Lite

                Objective:
                - Probe a fast tree donor with a bias different from `XGBoost`, `CatBoost`, and `LightGBM`.
                - Use `v2_missing_lite` and exact holdout evaluation.
                - Produce donor predictions for future blend analysis.

                Note:
                - `HistGradientBoostingClassifier` is CPU-only. The notebook runs on Kaggle, but training itself uses CPU.
                """
            ).strip()
            + "\n",
            1,
        ),
        code_cell("\n".join(bootstrap_lines) + "\n", 2),
        code_cell(
            dedent(
                """
                from __future__ import annotations

                import gc
                import json
                import random
                from pathlib import Path

                import numpy as np
                import pandas as pd
                from sklearn.ensemble import HistGradientBoostingClassifier
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit

                PROJECT_ROOT = Path(r"__PROJECT_ROOT__")
                RUN_MODE = "auto"  # auto | smoke | full_cpu

                SMOKE_CFG = {
                    "seed": 42,
                    "sample_rows": 120_000,
                    "smoke_min_positive": 12,
                    "learning_rate": 0.05,
                    "max_iter": 160,
                    "max_depth": 6,
                    "max_leaf_nodes": 31,
                    "min_samples_leaf": 128,
                    "l2_regularization": 1.0,
                    "max_bins": 255,
                    "n_iter_no_change": 20,
                    "train_final_model": False,
                }

                FULL_CFG = {
                    "seed": 42,
                    "sample_rows": None,
                    "smoke_min_positive": 0,
                    "learning_rate": 0.03,
                    "max_iter": 400,
                    "max_depth": 6,
                    "max_leaf_nodes": 63,
                    "min_samples_leaf": 128,
                    "l2_regularization": 1.0,
                    "max_bins": 255,
                    "n_iter_no_change": 30,
                    "train_final_model": True,
                }


                def resolve_run_mode(mode: str) -> str:
                    if mode != "auto":
                        return mode
                    return "full_cpu" if Path("/kaggle/input").exists() else "smoke"


                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = SMOKE_CFG if RESOLVED_MODE == "smoke" else FULL_CFG
                ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / f"histgb_ovr_fs_v2_missing_lite_{RESOLVED_MODE}"
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                random.seed(int(CFG["seed"]))
                np.random.seed(int(CFG["seed"]))
                print({"resolved_mode": RESOLVED_MODE, "config": CFG, "artifact_dir": str(ARTIFACT_DIR)}, flush=True)
                """
            ).strip().replace("__PROJECT_ROOT__", PROJECT_ROOT_LITERAL)
            + "\n",
            3,
        ),
        code_cell(
            dedent(
                """
                def find_base_data_dir() -> Path:
                    direct_candidates = [
                        PROJECT_ROOT / "output" / "kaggle-datasets" / "base-task-dataset",
                        PROJECT_ROOT / "data" / "competition",
                        Path.cwd() / "output" / "kaggle-datasets" / "base-task-dataset",
                        Path.cwd() / "data" / "competition",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026-2-task-dataset"),
                        Path("/kaggle/input/data-fusion-2026"),
                        Path("/kaggle/input/data-fusion-2026-cybershelf"),
                    ]
                    required = ["train_target.parquet"]
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


                def find_feature_artifact_dir() -> Path:
                    direct_candidates = [
                        PROJECT_ROOT / "output" / "kaggle-datasets" / "task-artifacts",
                        Path.cwd() / "output" / "kaggle-datasets" / "task-artifacts",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                    ]
                    required = [
                        "val_ids.parquet",
                        "current_best_validation_reference.parquet",
                        "current_best_submission_reference.parquet",
                    ]
                    for candidate in direct_candidates:
                        if all((candidate / name).exists() for name in required):
                            return candidate
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for match in kaggle_input.rglob("val_ids.parquet"):
                            candidate = match.parent
                            if all((candidate / name).exists() for name in required):
                                return candidate
                    raise FileNotFoundError("Could not locate the artifact reference directory.")


                def find_compact_feature_dir() -> Path:
                    direct_candidates = [
                        PROJECT_ROOT / "output" / "kaggle-datasets" / "task-artifacts",
                        PROJECT_ROOT / "artifacts" / "feature_selection_v2_missing_lite",
                        Path.cwd() / "output" / "kaggle-datasets" / "task-artifacts",
                        Path.cwd() / "artifacts" / "feature_selection_v2_missing_lite",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                    ]
                    required = [
                        "train_compact_features_v2_missing_lite.parquet",
                        "test_compact_features_v2_missing_lite.parquet",
                    ]
                    for candidate in direct_candidates:
                        if all((candidate / name).exists() for name in required):
                            return candidate
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for match in kaggle_input.rglob("train_compact_features_v2_missing_lite.parquet"):
                            candidate = match.parent
                            if all((candidate / name).exists() for name in required):
                                return candidate
                    raise FileNotFoundError("Could not locate the fs_v2_missing_lite directory.")


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


                def prepare_hist_frames(
                    frames: list[pd.DataFrame],
                    feature_cols: list[str],
                    cat_cols: list[str],
                ) -> list[np.ndarray]:
                    prepared = [frame[feature_cols].copy() for frame in frames]
                    for col in cat_cols:
                        combined = pd.concat([pd.to_numeric(frame[col], errors="coerce") for frame in prepared], axis=0)
                        unique = pd.Series(combined.fillna(-1).astype("int32").unique()).sort_values().tolist()
                        mapping = {int(v): idx for idx, v in enumerate(unique)}
                        for frame in prepared:
                            values = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int32")
                            frame[col] = values.map(mapping).astype("float32")
                    for frame in prepared:
                        for col in feature_cols:
                            if col not in cat_cols:
                                frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")
                    return [frame.to_numpy(dtype=np.float32, copy=False) for frame in prepared]


                def macro_auc_from_prob(y_true: pd.DataFrame, pred_prob: np.ndarray, target_cols: list[str]) -> float:
                    scores = []
                    for idx, target_name in enumerate(target_cols):
                        y = y_true[target_name].to_numpy()
                        score = 0.5 if np.unique(y).size < 2 else roc_auc_score(y, pred_prob[:, idx])
                        scores.append(float(score))
                    return float(np.mean(scores))
                """
            ).strip()
            + "\n",
            4,
        ),
        code_cell(
            dedent(
                """
                base_data_dir = find_base_data_dir()
                feature_artifact_dir = find_feature_artifact_dir()
                compact_feature_dir = find_compact_feature_dir()

                train_features = pd.read_parquet(compact_feature_dir / "train_compact_features_v2_missing_lite.parquet")
                test_features = pd.read_parquet(compact_feature_dir / "test_compact_features_v2_missing_lite.parquet")
                target = pd.read_parquet(base_data_dir / "train_target.parquet")
                baseline_val_ref = pd.read_parquet(feature_artifact_dir / "current_best_validation_reference.parquet").sort_values("customer_id").reset_index(drop=True)
                val_ids = pd.read_parquet(feature_artifact_dir / "val_ids.parquet")

                train_df = train_features.merge(target, on="customer_id", how="inner")
                target_cols = [c for c in train_df.columns if c.startswith("target_")]
                feature_cols = [c for c in train_features.columns if c != "customer_id"]
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

                if CFG["sample_rows"] is not None:
                    train_df = ensure_smoke_sample(train_df, target_cols, int(CFG["sample_rows"]), int(CFG["smoke_min_positive"]), int(CFG["seed"]))
                    train_part, valid_part = split_with_train_positive_coverage(train_df, target_cols, 0.20, int(CFG["seed"]))
                    baseline_full_macro = None
                else:
                    val_id_set = set(val_ids["customer_id"].astype("int64").tolist())
                    valid_mask = train_df["customer_id"].astype("int64").isin(val_id_set)
                    valid_part = train_df.loc[valid_mask].sort_values("customer_id").reset_index(drop=True)
                    train_part = train_df.loc[~valid_mask].sort_values("customer_id").reset_index(drop=True)
                    baseline_full_macro = macro_auc_from_prob(
                        valid_part[target_cols],
                        baseline_val_ref[[c.replace("target_", "predict_") for c in target_cols]].to_numpy(dtype=np.float64),
                        target_cols,
                    )

                train_x, valid_x, test_x = prepare_hist_frames([train_part, valid_part, test_features], feature_cols, cat_cols)
                train_y = train_part[target_cols].astype("int8")
                valid_y = valid_part[target_cols].astype("int8")
                full_train_x, full_test_x = prepare_hist_frames([train_features, test_features], feature_cols, cat_cols)
                full_train_y = target.astype("int8")

                print(
                    {
                        "base_data_dir": str(base_data_dir),
                        "feature_data_dir": str(compact_feature_dir),
                        "train_rows": int(len(train_part)),
                        "val_rows": int(len(valid_part)),
                        "test_rows": int(len(test_x)),
                        "feature_count": int(len(feature_cols)),
                        "cat_features": int(len(cat_cols)),
                        "baseline_full_macro": baseline_full_macro,
                    },
                    flush=True,
                )

                params = {
                    "loss": "log_loss",
                    "learning_rate": float(CFG["learning_rate"]),
                    "max_iter": int(CFG["max_iter"]),
                    "max_depth": int(CFG["max_depth"]),
                    "max_leaf_nodes": int(CFG["max_leaf_nodes"]),
                    "min_samples_leaf": int(CFG["min_samples_leaf"]),
                    "l2_regularization": float(CFG["l2_regularization"]),
                    "max_bins": int(CFG["max_bins"]),
                    "early_stopping": True,
                    "n_iter_no_change": int(CFG["n_iter_no_change"]),
                    "validation_fraction": None,
                    "random_state": int(CFG["seed"]),
                }
                print({"stage": "histgb_ovr_start", "params": params}, flush=True)

                val_pred = np.zeros((len(valid_x), len(target_cols)), dtype=np.float64)
                test_pred = np.zeros((len(test_x), len(target_cols)), dtype=np.float64)
                target_rows = []
                best_iterations = {}

                for idx, target_name in enumerate(target_cols, start=1):
                    y_train = train_y[target_name].to_numpy()
                    y_valid = valid_y[target_name].to_numpy()
                    print({"stage": "histgb_target_start", "target": target_name, "index": idx, "total": len(target_cols)}, flush=True)
                    clf = HistGradientBoostingClassifier(**params)
                    clf.fit(train_x, y_train)
                    best_iter = int(getattr(clf, "n_iter_", params["max_iter"]))
                    best_iterations[target_name] = best_iter
                    val_prob = clf.predict_proba(valid_x)[:, 1].astype(np.float64)
                    val_pred[:, idx - 1] = val_prob
                    auc = 0.5 if np.unique(y_valid).size < 2 else roc_auc_score(y_valid, val_prob)
                    baseline_auc = None if baseline_full_macro is None else float(
                        roc_auc_score(
                            y_valid,
                            baseline_val_ref[target_name.replace("target_", "predict_")].to_numpy(dtype=np.float64),
                        )
                    )
                    target_rows.append(
                        {
                            "target": target_name,
                            "oof_auc": float(auc),
                            "baseline_oof_auc": baseline_auc,
                            "delta_vs_current_best": None if baseline_auc is None else float(auc - baseline_auc),
                            "best_iteration": best_iter,
                        }
                    )
                    print({"stage": "histgb_target_done", "target": target_name, "oof_auc": float(auc), "best_iteration": best_iter}, flush=True)
                    if CFG["train_final_model"]:
                        final_clf = HistGradientBoostingClassifier(**{**params, "max_iter": best_iter, "early_stopping": False})
                        final_clf.fit(full_train_x, full_train_y[target_name].to_numpy())
                        test_pred[:, idx - 1] = final_clf.predict_proba(full_test_x)[:, 1].astype(np.float64)
                        del final_clf
                    gc.collect()

                full_macro = macro_auc_from_prob(valid_y, val_pred, target_cols)
                metrics = {
                    "validation_macro_auc": float(full_macro),
                    "baseline_full_macro_auc": baseline_full_macro,
                    "delta_vs_current_best": None if baseline_full_macro is None else float(full_macro - baseline_full_macro),
                    "resolved_mode": RESOLVED_MODE,
                    "feature_count": int(len(feature_cols)),
                    "cat_features": int(len(cat_cols)),
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
                print(json.dumps(metrics, indent=2), flush=True)

                target_scores = pd.DataFrame(target_rows).sort_values("oof_auc").reset_index(drop=True)
                target_scores.to_csv(ARTIFACT_DIR / "target_scores.csv", index=False)
                validation_predictions = pd.DataFrame(val_pred, columns=[c.replace("target_", "predict_") for c in target_cols])
                validation_predictions.insert(0, "customer_id", valid_part["customer_id"].to_numpy())
                validation_predictions.to_parquet(ARTIFACT_DIR / "validation_predictions.parquet", index=False)
                print(target_scores.head(12).to_string(index=False), flush=True)

                if CFG["train_final_model"]:
                    submission = pd.DataFrame(test_pred, columns=[c.replace("target_", "predict_") for c in target_cols])
                    submission.insert(0, "customer_id", test_features["customer_id"].astype("int32").to_numpy())
                    submission.to_parquet(ARTIFACT_DIR / "submission.parquet", index=False)
                    print({"stage": "submission_written", "submission_path": str(ARTIFACT_DIR / "submission.parquet"), "rows": int(len(submission))}, flush=True)
                else:
                    print("Smoke mode: final submission step skipped.", flush=True)
                """
            ).strip()
            + "\n",
            5,
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
        "id": "chesnikovleonid/data-fusion-2026-task-2-histgb-ovr-fs-v2-lite",
        "title": "Data Fusion 2026 task 2 histgb ovr fs v2 lite",
        "code_file": NOTEBOOK_NAME,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": False,
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
    }
    KERNEL_METADATA_PATH.write_text(json.dumps(kernel_metadata, indent=2) + "\n")
    print(f"Wrote {NOTEBOOK_PATH}")
    print(f"Wrote {KERNEL_METADATA_PATH}")


if __name__ == "__main__":
    main()
