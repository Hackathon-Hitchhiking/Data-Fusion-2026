from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
NOTEBOOK_NAME = "data-fusion-2026-gpu-xgboost-multioutput-fs-v2-missing-lite.ipynb"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output" / "kaggle-push" / "xgboost-multioutput-fs-v2-missing-lite"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "xgboost_multioutput_fs_v2_missing_lite_rev1_20260321"


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
        "import importlib.metadata",
        "import importlib.util",
        "import json",
        "import subprocess",
        "import sys",
        "",
        f'NOTEBOOK_REVISION = "{NOTEBOOK_REVISION}"',
        'TARGET_XGBOOST_VERSION = "3.0.5"',
        "",
        "REQUIRED_PACKAGES = [",
        '    ("xgboost", f"xgboost=={TARGET_XGBOOST_VERSION}"),',
        '    ("pyarrow", "pyarrow"),',
        '    ("sklearn", "scikit-learn"),',
        "]",
        "",
        "def pip_install(args: list[str]) -> None:",
        '    subprocess.check_call([sys.executable, "-m", "pip", *args])',
        "",
        "def package_version(name: str) -> str | None:",
        "    try:",
        "        return importlib.metadata.version(name)",
        "    except importlib.metadata.PackageNotFoundError:",
        "        return None",
        "",
        "missing = []",
        "for module_name, pip_name in REQUIRED_PACKAGES:",
        '    if module_name == "xgboost":',
        '        version = package_version("xgboost")',
        "        if version != TARGET_XGBOOST_VERSION:",
        "            missing.append(pip_name)",
        "    elif importlib.util.find_spec(module_name) is None:",
        "        missing.append(pip_name)",
        "",
        'print({"stage": "bootstrap_start", "revision": NOTEBOOK_REVISION, "missing": missing, "xgboost_version": package_version("xgboost")}, flush=True)',
        "if missing:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing])',
        "",
        "import xgboost as xgb",
        'print({"stage": "bootstrap_done", "installed_now": missing, "xgboost_version": xgb.__version__}, flush=True)',
    ]

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU XGBoost Multi-Output FS v2 Missing Lite

                Objective:
                - Add a third backbone with a different tree bias from `CatBoost` and `TabM`.
                - Re-run the proven fast `XGBoost` multi-output pipeline on the new `fs_v2_missing_lite` dataset.
                - Select the final iteration count by external holdout `macro ROC-AUC` on the exact current holdout.

                Design:
                - `XGBClassifier` with native multi-output support.
                - Stable strategy: `multi_strategy="one_output_per_tree"`, not `multi_output_tree`.
                - Native categorical handling through pandas categorical dtype and `enable_categorical=True`.
                - GPU histogram algorithm.
                - External checkpoint selection on the exact current holdout for direct comparison with the best live stack.
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
                - Smoke mode validates the full code path quickly and skips the final full-train stage.
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
                import random
                import time
                from pathlib import Path

                import numpy as np
                import pandas as pd
                import xgboost as xgb
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit

                RUN_MODE = "auto"  # auto | smoke | full_gpu

                SMOKE_CFG = {
                    "seed": 42,
                    "sample_rows": 60_000,
                    "smoke_min_positive": 12,
                    "val_fraction": 0.20,
                    "n_estimators": 250,
                    "learning_rate": 0.08,
                    "max_depth": 5,
                    "min_child_weight": 6.0,
                    "subsample": 0.80,
                    "colsample_bytree": 0.60,
                    "colsample_bylevel": 0.60,
                    "reg_lambda": 4.0,
                    "reg_alpha": 0.0,
                    "gamma": 0.0,
                    "max_bin": 128,
                    "max_cat_to_onehot": 4,
                    "max_cat_threshold": 64,
                    "sampling_method": "uniform",
                    "early_stopping_rounds": 40,
                    "coarse_step": 20,
                    "fine_step": 5,
                    "fine_window": 20,
                    "verbosity": 1,
                    "n_jobs": 4,
                    "train_final_model": False,
                }

                FULL_CFG = {
                    "seed": 42,
                    "sample_rows": None,
                    "smoke_min_positive": 0,
                    "val_fraction": 0.10,
                    "n_estimators": 3000,
                    "learning_rate": 0.03,
                    "max_depth": 6,
                    "min_child_weight": 8.0,
                    "subsample": 0.80,
                    "colsample_bytree": 0.60,
                    "colsample_bylevel": 0.60,
                    "reg_lambda": 8.0,
                    "reg_alpha": 0.0,
                    "gamma": 0.0,
                    "max_bin": 128,
                    "max_cat_to_onehot": 4,
                    "max_cat_threshold": 64,
                    "sampling_method": "uniform",
                    "early_stopping_rounds": 150,
                    "coarse_step": 50,
                    "fine_step": 10,
                    "fine_window": 60,
                    "verbosity": 1,
                    "n_jobs": 8,
                    "train_final_model": True,
                }


                def resolve_run_mode(mode: str) -> str:
                    if mode != "auto":
                        return mode
                    return "full_gpu" if Path("/kaggle/input").exists() else "smoke"


                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = SMOKE_CFG if RESOLVED_MODE == "smoke" else FULL_CFG
                ARTIFACT_DIR = Path("artifacts") / f"gpu_xgboost_multioutput_fs_v2_missing_lite_{RESOLVED_MODE}"
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                random.seed(int(CFG["seed"]))
                np.random.seed(int(CFG["seed"]))

                print(
                    {
                        "resolved_mode": RESOLVED_MODE,
                        "xgboost_version": xgb.__version__,
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


                def find_feature_artifact_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "artifacts" / "tabrs_pilot_inputs",
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
                        Path.cwd() / "artifacts" / "feature_selection_v2_missing_lite",
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                        Path.cwd() / "output" / "kaggle-datasets" / "task-artifacts",
                        Path.cwd(),
                    ]
                    required = [
                        "train_compact_features_v2_missing_lite.parquet",
                        "test_compact_features_v2_missing_lite.parquet",
                        "column_groups_v2_missing_lite.json",
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


                def prepare_feature_frames(
                    frames: list[pd.DataFrame],
                    num_cols: list[str],
                    cat_cols: list[str],
                ) -> list[pd.DataFrame]:
                    prepared = [frame.copy() for frame in frames]
                    for frame in prepared:
                        for col in num_cols:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32")

                    for col in cat_cols:
                        all_values = []
                        for frame in prepared:
                            series = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int32")
                            all_values.append(series)
                        categories = sorted(set(np.concatenate([series.to_numpy() for series in all_values]).tolist()))
                        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
                        for frame, series in zip(prepared, all_values):
                            frame[col] = pd.Categorical(series, dtype=dtype)
                    return prepared


                def macro_auc_from_prob(y_true: pd.DataFrame, pred_prob: np.ndarray, target_cols: list[str]) -> float:
                    scores = []
                    for idx, target_name in enumerate(target_cols):
                        y = y_true[target_name].to_numpy()
                        if np.unique(y).size < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y, pred_prob[:, idx])
                        scores.append(float(score))
                    return float(np.mean(scores))


                def per_target_auc_from_prob(y_true: pd.DataFrame, pred_prob: np.ndarray, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for idx, target_name in enumerate(target_cols):
                        y = y_true[target_name].to_numpy()
                        if np.unique(y).size < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y, pred_prob[:, idx])
                        rows.append({"target": target_name, "oof_auc": float(score)})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def evaluate_checkpoint_grid(
                    model: xgb.XGBClassifier,
                    valid_x: pd.DataFrame,
                    y_valid: pd.DataFrame,
                    target_cols: list[str],
                    coarse_step: int,
                    fine_step: int,
                    fine_window: int,
                ) -> tuple[int, float, pd.DataFrame]:
                    total_rounds = int(model.get_booster().num_boosted_rounds())
                    scored: dict[int, float] = {}

                    def score_checkpoint(ntree_end: int) -> float:
                        ntree_end = max(1, min(total_rounds, int(ntree_end)))
                        if ntree_end not in scored:
                            prob = np.asarray(
                                model.predict_proba(valid_x, iteration_range=(0, ntree_end)),
                                dtype=np.float64,
                            )
                            scored[ntree_end] = macro_auc_from_prob(y_valid, prob, target_cols)
                        return scored[ntree_end]

                    coarse_points = sorted(set(range(coarse_step, total_rounds + 1, coarse_step)) | {total_rounds})
                    coarse_best = max(coarse_points, key=score_checkpoint)
                    left = max(1, coarse_best - fine_window)
                    right = min(total_rounds, coarse_best + fine_window)
                    fine_points = sorted(set(range(left, right + 1, fine_step)) | {coarse_best, total_rounds})
                    best_round = max(fine_points, key=score_checkpoint)
                    checkpoint_scores = pd.DataFrame(
                        [{"iteration": int(k), "macro_auc": float(v)} for k, v in sorted(scored.items())]
                    )
                    return int(best_round), float(scored[best_round]), checkpoint_scores
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
                    baseline_full_macro = None
                else:
                    val_id_set = set(val_ids["customer_id"].astype("int64").tolist())
                    valid_mask = train_df["customer_id"].astype("int64").isin(val_id_set)
                    valid_part = train_df.loc[valid_mask].sort_values("customer_id").reset_index(drop=True)
                    train_part = train_df.loc[~valid_mask].sort_values("customer_id").reset_index(drop=True)
                    if list(valid_part["customer_id"].astype("int64")) != list(baseline_val_ref["customer_id"].astype("int64")):
                        raise RuntimeError("baseline validation reference does not match exact val_ids ordering")
                    baseline_full_macro = macro_auc_from_prob(
                        valid_part[target_cols],
                        baseline_val_ref[[c.replace("target_", "predict_") for c in target_cols]].to_numpy(dtype=np.float64),
                        target_cols,
                    )

                train_x, valid_x, test_x = prepare_feature_frames(
                    [train_part[feature_cols], valid_part[feature_cols], test_features[feature_cols]],
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                )
                train_y = train_part[target_cols].astype("int8")
                valid_y = valid_part[target_cols].astype("int8")

                train_cat_cardinality = train_x[cat_cols].nunique(dropna=False).sort_values()
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
                        "baseline_full_macro": baseline_full_macro,
                        "cat_cardinality_summary": {
                            "min": int(train_cat_cardinality.min()),
                            "median": float(train_cat_cardinality.median()),
                            "p90": float(train_cat_cardinality.quantile(0.9)),
                            "max": int(train_cat_cardinality.max()),
                        },
                    },
                    flush=True,
                )

                params = {
                    "objective": "binary:logistic",
                    "n_estimators": int(CFG["n_estimators"]),
                    "learning_rate": float(CFG["learning_rate"]),
                    "max_depth": int(CFG["max_depth"]),
                    "min_child_weight": float(CFG["min_child_weight"]),
                    "subsample": float(CFG["subsample"]),
                    "colsample_bytree": float(CFG["colsample_bytree"]),
                    "colsample_bylevel": float(CFG["colsample_bylevel"]),
                    "reg_lambda": float(CFG["reg_lambda"]),
                    "reg_alpha": float(CFG["reg_alpha"]),
                    "gamma": float(CFG["gamma"]),
                    "max_bin": int(CFG["max_bin"]),
                    "tree_method": "hist",
                    "device": "cuda" if RESOLVED_MODE == "full_gpu" else "cpu",
                    "enable_categorical": True,
                    "max_cat_to_onehot": int(CFG["max_cat_to_onehot"]),
                    "max_cat_threshold": int(CFG["max_cat_threshold"]),
                    "multi_strategy": "one_output_per_tree",
                    "sampling_method": str(CFG["sampling_method"]),
                    "eval_metric": "logloss",
                    "early_stopping_rounds": int(CFG["early_stopping_rounds"]),
                    "verbosity": int(CFG["verbosity"]),
                    "random_state": int(CFG["seed"]),
                    "n_jobs": int(CFG["n_jobs"]),
                }

                print({"stage": "xgboost_fit_start", "params": params}, flush=True)
                fit_started = time.time()
                model = xgb.XGBClassifier(**params)
                model.fit(
                    train_x,
                    train_y,
                    eval_set=[(valid_x, valid_y)],
                    verbose=50,
                )
                fit_seconds = time.time() - fit_started
                trained_rounds = int(model.get_booster().num_boosted_rounds())
                print(
                    {
                        "stage": "xgboost_fit_done",
                        "trained_rounds": trained_rounds,
                        "best_iteration_builtin": getattr(model, "best_iteration", None),
                        "fit_seconds": round(float(fit_seconds), 2),
                    },
                    flush=True,
                )

                best_round, best_macro_auc, checkpoint_scores = evaluate_checkpoint_grid(
                    model=model,
                    valid_x=valid_x,
                    y_valid=valid_y,
                    target_cols=target_cols,
                    coarse_step=int(CFG["coarse_step"]),
                    fine_step=int(CFG["fine_step"]),
                    fine_window=int(CFG["fine_window"]),
                )
                checkpoint_scores.to_csv(ARTIFACT_DIR / "checkpoint_scores.csv", index=False)

                valid_prob = np.asarray(
                    model.predict_proba(valid_x, iteration_range=(0, best_round)),
                    dtype=np.float64,
                )
                target_scores = per_target_auc_from_prob(valid_y, valid_prob, target_cols)
                if baseline_full_macro is not None:
                    baseline_target_scores = per_target_auc_from_prob(
                        valid_y,
                        baseline_val_ref[[c.replace("target_", "predict_") for c in target_cols]].to_numpy(dtype=np.float64),
                        target_cols,
                    ).rename(columns={"oof_auc": "baseline_oof_auc"})
                    target_scores = target_scores.merge(baseline_target_scores, on="target", how="left")
                    target_scores["delta_vs_current_best"] = target_scores["oof_auc"] - target_scores["baseline_oof_auc"]
                    target_scores = target_scores.sort_values("oof_auc").reset_index(drop=True)
                    target_scores[["target", "baseline_oof_auc", "oof_auc", "delta_vs_current_best"]].to_csv(
                        ARTIFACT_DIR / "target_score_deltas_vs_current_best.csv",
                        index=False,
                    )
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
                    "device": params["device"],
                    "rows_train": int(len(train_part)),
                    "rows_val": int(len(valid_part)),
                    "feature_count": int(len(feature_cols)),
                    "num_features": int(len(num_cols)),
                    "cat_features": int(len(cat_cols)),
                    "baseline_full_macro_auc": baseline_full_macro,
                    "delta_vs_current_best": None if baseline_full_macro is None else float(best_macro_auc - baseline_full_macro),
                    "selected_iteration": int(best_round),
                    "trained_rounds": int(trained_rounds),
                    "fit_seconds": float(fit_seconds),
                    "params": params,
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
                print(json.dumps(metrics, indent=2), flush=True)
                print(target_scores.head(12).to_string(index=False), flush=True)

                model.save_model(ARTIFACT_DIR / "validation_model.json")
                del valid_prob
                gc.collect()

                if CFG["train_final_model"]:
                    full_x, final_test_x = prepare_feature_frames(
                        [train_df[feature_cols], test_features[feature_cols]],
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                    )
                    full_y = train_df[target_cols].astype("int8")

                    final_params = dict(params)
                    final_params["n_estimators"] = int(best_round)
                    final_params.pop("early_stopping_rounds", None)

                    print({"stage": "xgboost_full_fit_start", "iterations": int(best_round)}, flush=True)
                    final_started = time.time()
                    final_model = xgb.XGBClassifier(**final_params)
                    final_model.fit(full_x, full_y, verbose=50)
                    final_seconds = time.time() - final_started
                    print({"stage": "xgboost_full_fit_done", "fit_seconds": round(float(final_seconds), 2)}, flush=True)

                    test_prob = np.asarray(final_model.predict_proba(final_test_x), dtype=np.float64)
                    submission = pd.DataFrame(test_prob, columns=[c.replace("target_", "predict_") for c in target_cols])
                    submission.insert(0, "customer_id", test_features["customer_id"].astype("int32").to_numpy())
                    submission.to_parquet(ARTIFACT_DIR / "submission.parquet", index=False)
                    final_model.save_model(ARTIFACT_DIR / "final_model.json")
                    (ARTIFACT_DIR / "full_train_metrics.json").write_text(
                        json.dumps({"selected_iterations": int(best_round), "full_train_fit_seconds": float(final_seconds)}, indent=2)
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
        "id": "chesnikovleonid/data-fusion-2026-task-2-xgb-multiout-fs-v2-lite",
        "title": "Data Fusion 2026 task 2 xgb multiout fs v2 lite",
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
