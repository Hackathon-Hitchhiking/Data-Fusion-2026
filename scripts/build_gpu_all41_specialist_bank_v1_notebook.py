from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
NOTEBOOK_NAME = "data-fusion-2026-gpu-all41-specialist-bank-v1.ipynb"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output" / "kaggle-push" / "all41-specialist-bank-v1"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "all41_specialist_bank_v1_rev2_20260321"


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
        '    ("catboost", "catboost"),',
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
        'print({"stage": "bootstrap_start", "revision": NOTEBOOK_REVISION, "missing": missing, "xgboost_version": package_version("xgboost"), "catboost_version": package_version("catboost")}, flush=True)',
        "if missing:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing])',
        "",
        "import xgboost as xgb",
        'print({"stage": "bootstrap_done", "installed_now": missing, "xgboost_version": xgb.__version__, "catboost_version": package_version("catboost")}, flush=True)',
    ]

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU All-41 Specialist Bank v1

                Objective:
                - Scale the already validated specialist pattern to all `41` targets.
                - Train two stable binary specialists per target:
                  - `XGBoost-rare`
                  - `CatBoost-sqrtbalanced`
                - Replace target predictions only when a specialist clearly beats the current best global stack on the exact holdout.

                Design:
                - Exact current `75k` holdout from `val_ids.parquet`.
                - Compact `fs_v1` dataset (`419` features).
                - Baseline references:
                  - `current_best_validation_reference.parquet`
                  - `current_best_submission_reference.parquet`
                - Acceptance rule:
                  - strong accept if `delta >= 0.0020`
                  - accept if `delta >= 0.0015`
                  - otherwise keep baseline
                - Full-data retrain only for accepted targets to write the final submission.
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
                  - Kaggle -> `full_gpu`
                  - local machine -> `smoke`
                - Smoke mode keeps the full code path but uses a smaller sample and fewer trees.
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
                from catboost import CatBoostClassifier, Pool
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit

                RUN_MODE = "auto"  # auto | smoke | full_gpu

                SMOKE_CFG = {
                    "seed": 42,
                    "sample_rows": 80_000,
                    "smoke_min_positive": 10,
                    "xgb_n_estimators": 400,
                    "xgb_early_stopping_rounds": 50,
                    "cat_iterations": 500,
                    "cat_od_wait": 50,
                    "retrain_enabled": False,
                }

                FULL_CFG = {
                    "seed": 42,
                    "sample_rows": None,
                    "smoke_min_positive": 0,
                    "xgb_n_estimators": 4000,
                    "xgb_early_stopping_rounds": 200,
                    "cat_iterations": 4000,
                    "cat_od_wait": 200,
                    "retrain_enabled": True,
                }


                def resolve_run_mode(mode: str) -> str:
                    if mode != "auto":
                        return mode
                    return "full_gpu" if Path("/kaggle/input").exists() else "smoke"


                def set_all_seeds(seed: int) -> None:
                    random.seed(seed)
                    np.random.seed(seed)


                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = SMOKE_CFG if RESOLVED_MODE == "smoke" else FULL_CFG
                ARTIFACT_DIR = Path("artifacts") / f"gpu_all41_specialist_bank_v1_{RESOLVED_MODE}"
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                set_all_seeds(int(CFG["seed"]))

                print({
                    "resolved_mode": RESOLVED_MODE,
                    "xgboost_version": xgb.__version__,
                    "config": CFG,
                    "artifact_dir": str(ARTIFACT_DIR),
                }, flush=True)
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


                def find_feature_data_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "artifacts" / "tabrs_pilot_inputs",
                        Path.cwd() / "artifacts" / "feature_selection_v1",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                    ]
                    required = [
                        "train_compact_features_v1.parquet",
                        "test_compact_features_v1.parquet",
                        "val_ids.parquet",
                        "current_best_validation_reference.parquet",
                        "current_best_submission_reference.parquet",
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


                def prepare_numeric_frames(
                    train_df: pd.DataFrame,
                    other_frames: list[pd.DataFrame],
                    num_cols: list[str],
                ) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, float]]:
                    train_out = train_df.copy()
                    other_out = [frame.copy() for frame in other_frames]
                    medians = {}
                    for col in num_cols:
                        train_out[col] = pd.to_numeric(train_out[col], errors="coerce").astype("float32")
                        median = float(train_out[col].median())
                        medians[col] = median
                        train_out[col] = train_out[col].fillna(median).astype("float32")
                        for frame in other_out:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median).astype("float32")
                    return train_out, other_out, medians


                def prepare_xgb_frames(
                    train_df: pd.DataFrame,
                    valid_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    num_cols: list[str],
                    cat_cols: list[str],
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    train_out, [valid_out, test_out], _ = prepare_numeric_frames(train_df, [valid_df, test_df], num_cols)
                    all_frames = [train_out, valid_out, test_out]
                    for col in cat_cols:
                        series_list = [pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int32") for frame in all_frames]
                        categories = sorted(set(np.concatenate([series.to_numpy() for series in series_list]).tolist()))
                        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
                        for frame, series in zip(all_frames, series_list):
                            frame[col] = pd.Categorical(series, dtype=dtype)
                    return train_out, valid_out, test_out


                def prepare_catboost_frames(
                    train_df: pd.DataFrame,
                    valid_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    num_cols: list[str],
                    cat_cols: list[str],
                ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
                    train_out, [valid_out, test_out], _ = prepare_numeric_frames(train_df, [valid_df, test_df], num_cols)
                    for frame in [train_out, valid_out, test_out]:
                        for col in cat_cols:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int32")
                    return train_out, valid_out, test_out


                def safe_auc(y_true: np.ndarray, pred: np.ndarray) -> float:
                    if np.unique(y_true).size < 2:
                        return 0.5
                    return float(roc_auc_score(y_true, pred))


                def macro_auc(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    scores = []
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        scores.append(safe_auc(y_true[target_name].to_numpy(dtype=np.int8), pred_df[pred_col].to_numpy(dtype=np.float64)))
                    return float(np.mean(scores))


                def scale_pos_weight_from_labels(y: np.ndarray) -> float:
                    pos = float(y.sum())
                    neg = float(len(y) - pos)
                    return float(min(50.0, neg / max(pos, 1.0)))


                def build_xgb_params(scale_pos_weight: float, seed: int, n_estimators: int, early_stopping_rounds: int) -> dict[str, object]:
                    return {
                        "objective": "binary:logistic",
                        "eval_metric": "auc",
                        "n_estimators": int(n_estimators),
                        "learning_rate": 0.03,
                        "max_depth": 5,
                        "min_child_weight": 8.0,
                        "subsample": 0.80,
                        "colsample_bytree": 0.80,
                        "colsample_bylevel": 0.80,
                        "reg_lambda": 8.0,
                        "reg_alpha": 0.0,
                        "gamma": 0.0,
                        "max_delta_step": 1,
                        "scale_pos_weight": float(scale_pos_weight),
                        "max_bin": 256,
                        "max_cat_to_onehot": 4,
                        "max_cat_threshold": 64,
                        "tree_method": "hist",
                        "device": "cuda",
                        "enable_categorical": True,
                        "early_stopping_rounds": int(early_stopping_rounds),
                        "verbosity": 0,
                        "random_state": int(seed),
                        "n_jobs": 8,
                    }


                def build_cat_params(seed: int, iterations: int, od_wait: int) -> dict[str, object]:
                    return {
                        "loss_function": "Logloss",
                        "eval_metric": "AUC",
                        "task_type": "GPU",
                        "devices": "0",
                        "iterations": int(iterations),
                        "learning_rate": 0.03,
                        "depth": 6,
                        "l2_leaf_reg": 8.0,
                        "random_strength": 1.0,
                        "bagging_temperature": 0.5,
                        "bootstrap_type": "Bayesian",
                        "auto_class_weights": "SqrtBalanced",
                        "one_hot_max_size": 4,
                        "od_type": "Iter",
                        "od_wait": int(od_wait),
                        "use_best_model": True,
                        "random_seed": int(seed),
                        "verbose": 200,
                        "allow_writing_files": False,
                    }
                """
            ).strip()
            + "\n",
            6,
        ),
        md_cell("## Load Data And Exact Holdout\n", 7),
        code_cell(
            dedent(
                """
                BASE_DATA_DIR = find_base_data_dir()
                FEATURE_DATA_DIR = find_feature_data_dir()

                train_features = pd.read_parquet(FEATURE_DATA_DIR / "train_compact_features_v1.parquet")
                test_features = pd.read_parquet(FEATURE_DATA_DIR / "test_compact_features_v1.parquet")
                target = pd.read_parquet(BASE_DATA_DIR / "train_target.parquet")
                val_ids = pd.read_parquet(FEATURE_DATA_DIR / "val_ids.parquet")
                baseline_val_ref = pd.read_parquet(FEATURE_DATA_DIR / "current_best_validation_reference.parquet").sort_values("customer_id").reset_index(drop=True)
                baseline_submit_ref = pd.read_parquet(FEATURE_DATA_DIR / "current_best_submission_reference.parquet").sort_values("customer_id").reset_index(drop=True)

                labeled_df = train_features.merge(target, on="customer_id", how="inner")
                test_df = test_features.copy().sort_values("customer_id").reset_index(drop=True)

                target_cols = [c for c in labeled_df.columns if c.startswith("target_")]
                feature_cols = [c for c in train_features.columns if c != "customer_id"]
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
                num_cols = [c for c in feature_cols if c not in cat_cols]

                if RESOLVED_MODE == "smoke":
                    labeled_df = ensure_smoke_sample(
                        labeled_df,
                        target_cols=target_cols,
                        sample_rows=int(CFG["sample_rows"]),
                        min_positive=int(CFG["smoke_min_positive"]),
                        seed=int(CFG["seed"]),
                    )
                    train_df, val_df = split_with_train_positive_coverage(
                        labeled_df,
                        target_cols=target_cols,
                        val_fraction=0.20,
                        seed=int(CFG["seed"]),
                    )
                else:
                    val_id_set = set(val_ids["customer_id"].astype("int64").tolist())
                    val_mask = labeled_df["customer_id"].astype("int64").isin(val_id_set)
                    val_df = labeled_df.loc[val_mask].sort_values("customer_id").reset_index(drop=True)
                    train_df = labeled_df.loc[~val_mask].sort_values("customer_id").reset_index(drop=True)
                    if list(val_df["customer_id"].astype("int64")) != list(baseline_val_ref["customer_id"].astype("int64")):
                        raise RuntimeError("baseline validation reference does not match exact val_ids ordering")
                    if list(test_df["customer_id"].astype("int64")) != list(baseline_submit_ref["customer_id"].astype("int64")):
                        raise RuntimeError("baseline submission reference does not match test ordering")

                print({
                    "base_data_dir": str(BASE_DATA_DIR),
                    "feature_data_dir": str(FEATURE_DATA_DIR),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "feature_count": len(feature_cols),
                    "num_features": len(num_cols),
                    "cat_features": len(cat_cols),
                    "target_count": len(target_cols),
                }, flush=True)
                """
            ).strip()
            + "\n",
            8,
        ),
        md_cell("## Train Specialists On Exact Holdout\n", 9),
        code_cell(
            dedent(
                """
                xgb_train_x, xgb_val_x, xgb_test_x = prepare_xgb_frames(
                    train_df[feature_cols].copy(),
                    val_df[feature_cols].copy(),
                    test_df[feature_cols].copy(),
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                )
                cat_train_x, cat_val_x, cat_test_x = prepare_catboost_frames(
                    train_df[feature_cols].copy(),
                    val_df[feature_cols].copy(),
                    test_df[feature_cols].copy(),
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                )
                cat_feature_indices = [cat_train_x.columns.get_loc(c) for c in cat_cols]
                full_baseline_macro = None if RESOLVED_MODE == "smoke" else macro_auc(val_df[["customer_id"] + target_cols], baseline_val_ref, target_cols)

                xgb_val_preds = pd.DataFrame({"customer_id": val_df["customer_id"].astype("int32").values})
                cat_val_preds = pd.DataFrame({"customer_id": val_df["customer_id"].astype("int32").values})
                selection_rows = []
                best_iterations: dict[str, dict[str, int | float]] = {}

                for idx, target_name in enumerate(target_cols, start=1):
                    pred_col = target_name.replace("target_", "predict_")
                    y_train = train_df[target_name].to_numpy(dtype=np.int8)
                    y_val = val_df[target_name].to_numpy(dtype=np.int8)
                    baseline_auc = float("nan") if RESOLVED_MODE == "smoke" else safe_auc(y_val, baseline_val_ref[pred_col].to_numpy(dtype=np.float64))
                    print({"stage": "all41_target_start", "target": target_name, "index": idx, "total": len(target_cols), "baseline_auc": baseline_auc}, flush=True)

                    xgb_params = build_xgb_params(
                        scale_pos_weight=scale_pos_weight_from_labels(y_train),
                        seed=int(CFG["seed"]),
                        n_estimators=int(CFG["xgb_n_estimators"]),
                        early_stopping_rounds=int(CFG["xgb_early_stopping_rounds"]),
                    )
                    xgb_model = xgb.XGBClassifier(**xgb_params)
                    xgb_model.fit(
                        xgb_train_x,
                        y_train,
                        eval_set=[(xgb_val_x, y_val)],
                        verbose=False,
                    )
                    xgb_best_round = int(xgb_model.best_iteration) + 1 if getattr(xgb_model, "best_iteration", None) is not None else int(xgb_model.get_booster().num_boosted_rounds())
                    xgb_val_prob = np.asarray(xgb_model.predict_proba(xgb_val_x, iteration_range=(0, xgb_best_round))[:, 1], dtype=np.float64)
                    xgb_auc = safe_auc(y_val, xgb_val_prob)
                    xgb_val_preds[pred_col] = xgb_val_prob

                    train_pool = Pool(cat_train_x, y_train, cat_features=cat_feature_indices)
                    valid_pool = Pool(cat_val_x, y_val, cat_features=cat_feature_indices)
                    cat_model = CatBoostClassifier(**build_cat_params(int(CFG["seed"]), int(CFG["cat_iterations"]), int(CFG["cat_od_wait"])))
                    cat_model.fit(train_pool, eval_set=valid_pool)
                    cat_best_iter = int(cat_model.get_best_iteration()) + 1 if int(cat_model.get_best_iteration()) >= 0 else int(cat_model.tree_count_)
                    cat_val_prob = np.asarray(cat_model.predict_proba(valid_pool)[:, 1], dtype=np.float64)
                    cat_auc = safe_auc(y_val, cat_val_prob)
                    cat_val_preds[pred_col] = cat_val_prob

                    best_specialist_source = "xgb_rare_v1" if xgb_auc >= cat_auc else "cat_sqrtbalanced_v1"
                    best_specialist_auc = xgb_auc if xgb_auc >= cat_auc else cat_auc
                    delta = float("nan") if RESOLVED_MODE == "smoke" else float(best_specialist_auc - baseline_auc)
                    if RESOLVED_MODE == "smoke":
                        chosen_source = best_specialist_source
                        decision = "smoke"
                    else:
                        if delta >= 0.0020:
                            chosen_source = best_specialist_source
                            decision = "strong_accept"
                        elif delta >= 0.0015:
                            chosen_source = best_specialist_source
                            decision = "accept"
                        else:
                            chosen_source = "baseline"
                            decision = "keep_baseline"

                    selection_rows.append(
                        {
                            "target": target_name,
                            "baseline_auc": baseline_auc,
                            "xgb_auc": xgb_auc,
                            "cat_auc": cat_auc,
                            "chosen_source": chosen_source,
                            "best_specialist_source": best_specialist_source,
                            "delta": delta,
                            "decision": decision,
                            "xgb_best_iteration": int(xgb_best_round),
                            "cat_best_iteration": int(cat_best_iter),
                        }
                    )
                    best_iterations[target_name] = {
                        "xgb_best_iteration": int(xgb_best_round),
                        "cat_best_iteration": int(cat_best_iter),
                    }

                    print(
                        {
                            "stage": "all41_target_done",
                            "target": target_name,
                            "xgb_auc": xgb_auc,
                            "cat_auc": cat_auc,
                            "baseline_auc": baseline_auc,
                            "chosen_source": chosen_source,
                            "decision": decision,
                            "delta": delta,
                        },
                        flush=True,
                    )

                    del xgb_model, cat_model, train_pool, valid_pool, y_train, y_val, xgb_val_prob, cat_val_prob
                    gc.collect()

                selection_df = pd.DataFrame(selection_rows).sort_values(["decision", "delta"], ascending=[True, False]).reset_index(drop=True)
                selection_df.to_csv(ARTIFACT_DIR / "specialist_bank_all41_selection.csv", index=False)
                xgb_val_preds.to_parquet(ARTIFACT_DIR / "xgb_validation_predictions.parquet", index=False)
                cat_val_preds.to_parquet(ARTIFACT_DIR / "cat_validation_predictions.parquet", index=False)
                """
            ).strip()
            + "\n",
            10,
        ),
        md_cell("## Build Final Validation Overlay\n", 11),
        code_cell(
            dedent(
                """
                if RESOLVED_MODE == "smoke":
                    specialist_bank_val = xgb_val_preds.copy()
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        if pred_col not in specialist_bank_val.columns:
                            specialist_bank_val[pred_col] = 0.0
                    specialist_bank_macro = macro_auc(val_df[["customer_id"] + target_cols], specialist_bank_val[["customer_id"] + [t.replace("target_", "predict_") for t in target_cols]], target_cols)
                else:
                    specialist_bank_val = baseline_val_ref.copy()
                    for row in selection_df.to_dict("records"):
                        target_name = row["target"]
                        pred_col = target_name.replace("target_", "predict_")
                        if row["chosen_source"] == "xgb_rare_v1":
                            specialist_bank_val[pred_col] = xgb_val_preds[pred_col].astype(np.float64).values
                        elif row["chosen_source"] == "cat_sqrtbalanced_v1":
                            specialist_bank_val[pred_col] = cat_val_preds[pred_col].astype(np.float64).values
                    specialist_bank_macro = macro_auc(val_df[["customer_id"] + target_cols], specialist_bank_val, target_cols)

                specialist_bank_val.to_parquet(ARTIFACT_DIR / "specialist_bank_all41_validation_predictions.parquet", index=False)
                specialist_bank_target_scores = pd.DataFrame([
                    {
                        "target": target_name,
                        "oof_auc": safe_auc(
                            val_df[target_name].to_numpy(dtype=np.int8),
                            specialist_bank_val[target_name.replace("target_", "predict_")].to_numpy(dtype=np.float64),
                        ),
                    }
                    for target_name in target_cols
                ]).sort_values("oof_auc").reset_index(drop=True)
                specialist_bank_target_scores.to_csv(ARTIFACT_DIR / "specialist_bank_target_scores.csv", index=False)

                summary = {
                    "baseline_full_macro_auc": full_baseline_macro,
                    "specialist_bank_full_macro_auc": specialist_bank_macro,
                    "delta_vs_baseline": None if full_baseline_macro is None else float(specialist_bank_macro - full_baseline_macro),
                    "accepted_target_count": int((selection_df["chosen_source"] != "baseline").sum()) if RESOLVED_MODE != "smoke" else int(len(selection_df)),
                    "accepted_targets": selection_df.loc[selection_df["chosen_source"] != "baseline", "target"].tolist() if RESOLVED_MODE != "smoke" else selection_df["target"].tolist(),
                }
                (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
                print(json.dumps(summary, indent=2), flush=True)
                print(specialist_bank_target_scores.head(12).to_string(index=False), flush=True)
                """
            ).strip()
            + "\n",
            12,
        ),
        md_cell("## Retrain Accepted Specialists On Full Data\n", 13),
        code_cell(
            dedent(
                """
                if bool(CFG["retrain_enabled"]):
                    accepted_rows = selection_df[selection_df["chosen_source"] != "baseline"].copy().reset_index(drop=True)
                    specialist_submit = baseline_submit_ref.copy()

                    full_xgb_train_x, _, full_xgb_test_x = prepare_xgb_frames(
                        labeled_df[feature_cols].copy(),
                        labeled_df[feature_cols].copy().iloc[:1].copy(),
                        test_df[feature_cols].copy(),
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                    )
                    full_cat_train_x, _, full_cat_test_x = prepare_catboost_frames(
                        labeled_df[feature_cols].copy(),
                        labeled_df[feature_cols].copy().iloc[:1].copy(),
                        test_df[feature_cols].copy(),
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                    )
                    full_cat_feature_indices = [full_cat_train_x.columns.get_loc(c) for c in cat_cols]
                    full_test_pool = Pool(full_cat_test_x, cat_features=full_cat_feature_indices)

                    for idx, row in enumerate(accepted_rows.to_dict("records"), start=1):
                        target_name = row["target"]
                        pred_col = target_name.replace("target_", "predict_")
                        chosen_source = row["chosen_source"]
                        y_full = labeled_df[target_name].to_numpy(dtype=np.int8)

                        print({"stage": "retrain_target_start", "target": target_name, "index": idx, "total": len(accepted_rows), "chosen_source": chosen_source}, flush=True)

                        if chosen_source == "xgb_rare_v1":
                            final_rounds = int(row["xgb_best_iteration"])
                            final_xgb_params = build_xgb_params(
                                scale_pos_weight=scale_pos_weight_from_labels(y_full),
                                seed=int(CFG["seed"]),
                                n_estimators=max(1, final_rounds),
                                early_stopping_rounds=int(CFG["xgb_early_stopping_rounds"]),
                            )
                            final_xgb_params.pop("early_stopping_rounds", None)
                            final_model = xgb.XGBClassifier(**final_xgb_params)
                            final_model.fit(full_xgb_train_x, y_full, verbose=False)
                            test_prob = np.asarray(final_model.predict_proba(full_xgb_test_x)[:, 1], dtype=np.float64)
                        elif chosen_source == "cat_sqrtbalanced_v1":
                            final_iters = int(row["cat_best_iteration"])
                            full_train_pool = Pool(full_cat_train_x, y_full, cat_features=full_cat_feature_indices)
                            final_cat_params = build_cat_params(int(CFG["seed"]), max(1, final_iters), int(CFG["cat_od_wait"]))
                            final_cat_params["use_best_model"] = False
                            final_cat_params.pop("od_type", None)
                            final_cat_params.pop("od_wait", None)
                            final_model = CatBoostClassifier(**final_cat_params)
                            final_model.fit(full_train_pool)
                            test_prob = np.asarray(final_model.predict_proba(full_test_pool)[:, 1], dtype=np.float64)
                        else:
                            raise ValueError(f"Unexpected chosen_source: {chosen_source}")

                        specialist_submit[pred_col] = test_prob.astype(np.float64)
                        print({"stage": "retrain_target_done", "target": target_name, "chosen_source": chosen_source}, flush=True)

                        del y_full, final_model, test_prob
                        gc.collect()

                    specialist_submit.to_parquet(ARTIFACT_DIR / "specialist_bank_all41_submission.parquet", index=False)
                    print({"stage": "submission_written", "path": str(ARTIFACT_DIR / "specialist_bank_all41_submission.parquet"), "rows": len(specialist_submit)}, flush=True)
                else:
                    print("Smoke mode: final retrain/submission step skipped.")
                """
            ).strip()
            + "\n",
            14,
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")

    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    push_notebook = PUSH_DIR / "data-fusion-2026-task-2-all41-specialist-bank-v1.ipynb"
    shutil.copy2(NOTEBOOK_PATH, push_notebook)
    metadata = {
        "id": "chesnikovleonid/data-fusion-2026-task-2-all41-specialist-bank-v1",
        "title": "Data Fusion 2026 task 2 all41 specialist bank v1",
        "code_file": push_notebook.name,
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
    KERNEL_METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n")
    print(
        {
            "notebook": str(NOTEBOOK_PATH),
            "push_notebook": str(push_notebook),
            "metadata": str(KERNEL_METADATA_PATH),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
