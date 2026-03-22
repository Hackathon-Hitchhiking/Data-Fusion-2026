from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root

ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook/data-fusion-2026-gpu-top1-ensemble.ipynb"


def md_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line for line in text.splitlines(keepends=True)],
    }


def code_cell(text: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line for line in text.splitlines(keepends=True)],
    }


def main() -> None:
    top_features = json.loads(TOP_FEATURES_FILE.read_text())
    top_features_literal = json.dumps(top_features, ensure_ascii=True)
    hard_targets_literal = json.dumps(
        [
            "target_3_1",
            "target_9_3",
            "target_9_6",
            "target_2_4",
            "target_6_1",
            "target_6_2",
            "target_2_6",
            "target_3_3",
            "target_10_1",
            "target_5_1",
            "target_5_2",
            "target_9_7",
        ],
        ensure_ascii=True,
    )

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU Top-1 Ensemble

                Objective:
                - Spend the full budget on the most justified current path.
                - Use a strong deep backbone, targeted specialists, and a holdout-learned final ensemble.

                Current reasoning behind this notebook:
                - `GANDALF` is the only branch that already beat the tree family on the leaderboard.
                - A simple global blend between deep and tree predictions improved holdout quality locally.
                - The remaining weakness is concentrated in a small hard-target tail, so specialists are justified.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                import importlib.util

                REQUIRED_PACKAGES = [
                    ("torch", "torch"),
                    ("lightgbm", "lightgbm"),
                    ("pytorch_tabular", "pytorch-tabular"),
                    ("autogluon", "autogluon.tabular"),
                    ("pyarrow", "pyarrow"),
                    ("sklearn", "scikit-learn"),
                ]

                missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]
                if missing:
                    %pip install -q { " ".join(missing) }
                print({"installed_now": missing})
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                f"""
                import gc
                import json
                import warnings
                from pathlib import Path

                import numpy as np
                import pandas as pd
                import torch
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit
                import lightgbm as lgb

                warnings.filterwarnings("ignore")

                RUN_MODE = "auto"  # one of: auto, smoke, full_gpu
                SEED = 42
                TOP_EXTRA_FEATURES = {top_features_literal}
                HARD_TARGETS = {hard_targets_literal}

                CFG_BY_MODE = {{
                    "smoke": {{
                        "sample_rows": 100000,
                        "smoke_min_positive": 8,
                        "extra_top_k": 40,
                        "val_fraction": 0.20,
                        "gandalf_seeds": [42],
                        "gandalf_max_epochs": 4,
                        "gandalf_final_epochs": 6,
                        "gandalf_batch_size": 1024,
                        "gandalf_learning_rate": 0.0010,
                        "gandalf_gflu_stages": 4,
                        "gandalf_gflu_dropout": 0.10,
                        "lgbm_n_estimators": 140,
                        "lgbm_learning_rate": 0.05,
                        "lgbm_num_leaves": 63,
                        "autogluon_presets": "medium_quality",
                        "autogluon_max_rows": 60000,
                        "autogluon_time_limit": 120,
                        "autogluon_final_time_limit": 180,
                        "autogluon_min_rows": 20000,
                        "autogluon_memory_fraction": 0.28,
                        "autogluon_memory_multiplier": 3.5,
                        "train_final_models": False,
                    }},
                    "full_gpu": {{
                        "sample_rows": None,
                        "smoke_min_positive": 0,
                        "extra_top_k": 100,
                        "val_fraction": 0.10,
                        "gandalf_seeds": [42, 52, 62],
                        "gandalf_max_epochs": 22,
                        "gandalf_final_epochs": 30,
                        "gandalf_batch_size": 2048,
                        "gandalf_learning_rate": 0.0008,
                        "gandalf_gflu_stages": 6,
                        "gandalf_gflu_dropout": 0.15,
                        "lgbm_n_estimators": 320,
                        "lgbm_learning_rate": 0.05,
                        "lgbm_num_leaves": 63,
                        "autogluon_presets": "good_quality",
                        "autogluon_max_rows": 180000,
                        "autogluon_time_limit": 180,
                        "autogluon_final_time_limit": 240,
                        "autogluon_min_rows": 50000,
                        "autogluon_memory_fraction": 0.28,
                        "autogluon_memory_multiplier": 3.5,
                        "train_final_models": True,
                    }},
                }}


                def resolve_run_mode(mode: str) -> str:
                    if mode == "auto":
                        return "full_gpu" if torch.cuda.is_available() else "smoke"
                    return mode


                def find_data_dir() -> Path:
                    candidates = [Path.cwd() / "data" / "competition", Path.cwd(), Path("/kaggle/input"), Path("/kaggle/working")]
                    required = {{
                        "train_main_features.parquet",
                        "train_extra_features.parquet",
                        "train_target.parquet",
                        "test_main_features.parquet",
                        "test_extra_features.parquet",
                    }}
                    for base in candidates:
                        if not base.exists():
                            continue
                        if required.issubset({{p.name for p in base.glob("*.parquet")}}):
                            return base
                        for subdir in base.rglob("*"):
                            if not subdir.is_dir():
                                continue
                            names = {{p.name for p in subdir.glob("*.parquet")}}
                            if required.issubset(names):
                                return subdir
                    raise FileNotFoundError("Could not locate the required parquet dataset files.")


                def build_stratify_labels(target_df: pd.DataFrame) -> pd.Series:
                    return target_df.sum(axis=1).clip(upper=4).astype("int8").astype(str)


                def ensure_smoke_sample(full_df: pd.DataFrame, target_cols: list[str], sample_rows: int, min_positive: int, seed: int) -> pd.DataFrame:
                    if sample_rows is None or sample_rows >= len(full_df):
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


                def split_with_train_positive_coverage(df: pd.DataFrame, target_cols: list[str], val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
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
                    return df.iloc[sorted(train_idx)].reset_index(drop=True), df.iloc[sorted(val_idx)].reset_index(drop=True)


                def compute_target_scores(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        score = 0.5 if y_true[target_name].nunique() < 2 else roc_auc_score(y_true[target_name], pred_df[pred_col])
                        rows.append({{"target": target_name, "oof_auc": float(score)}})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def macro_auc(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    return float(compute_target_scores(y_true, pred_df, target_cols)["oof_auc"].mean())


                def normalize_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
                    rename_map = {{c: c.replace("target_", "predict_") for c in df.columns if c.startswith("target_")}}
                    return df.rename(columns=rename_map)


                def find_external_files(filename: str, hint: str | None = None) -> list[Path]:
                    kaggle_input = Path("/kaggle/input")
                    if not kaggle_input.exists():
                        return []
                    matches = []
                    for path in kaggle_input.rglob(filename):
                        path_str = str(path).lower()
                        if hint and hint.lower() not in path_str:
                            continue
                        matches.append(path)
                    return sorted(matches)


                def find_external_artifact_dirs(name: str) -> list[Path]:
                    kaggle_input = Path("/kaggle/input")
                    if not kaggle_input.exists():
                        return []
                    return sorted([p for p in kaggle_input.rglob(name) if p.is_dir()])


                def load_submit_like(path: Path, expected_pred_cols: list[str]) -> pd.DataFrame:
                    df = normalize_pred_columns(pd.read_parquet(path)).sort_values("customer_id").reset_index(drop=True)
                    out = pd.DataFrame({{"customer_id": df["customer_id"].astype("int32").values}})
                    for pred_col in expected_pred_cols:
                        out[pred_col] = df[pred_col].astype("float64").values
                    return out
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Load Data And Holdout Split\n"),
        code_cell(
            dedent(
                """
                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = CFG_BY_MODE[RESOLVED_MODE]
                ARTIFACT_DIR = Path("artifacts/gpu_top1_ensemble_" + RESOLVED_MODE)
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
                EXTERNAL_TOP1_ARTIFACT_DIRS = find_external_artifact_dirs("gpu_top1_ensemble_full_gpu")
                EXTERNAL_GANDALF_VAL_FILES = find_external_files("validation_predictions.parquet", hint="gandalf")
                EXTERNAL_GANDALF_TEST_FILES = find_external_files("submission.parquet", hint="gandalf")

                DATA_DIR = find_data_dir()
                EXTRA_FEATURES = TOP_EXTRA_FEATURES[: CFG["extra_top_k"]]

                train_main = pd.read_parquet(DATA_DIR / "train_main_features.parquet")
                train_extra = pd.read_parquet(DATA_DIR / "train_extra_features.parquet", columns=["customer_id"] + EXTRA_FEATURES)
                train_target = pd.read_parquet(DATA_DIR / "train_target.parquet")
                test_main = pd.read_parquet(DATA_DIR / "test_main_features.parquet")
                test_extra = pd.read_parquet(DATA_DIR / "test_extra_features.parquet", columns=["customer_id"] + EXTRA_FEATURES)

                labeled_df = train_main.merge(train_extra, on="customer_id", how="inner").merge(train_target, on="customer_id", how="inner")
                test_df = test_main.merge(test_extra, on="customer_id", how="inner")

                target_cols = [c for c in train_target.columns if c.startswith("target_")]
                feature_cols = [c for c in labeled_df.columns if c not in {"customer_id", *target_cols}]
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
                cont_cols = [c for c in feature_cols if c not in cat_cols]

                labeled_df = ensure_smoke_sample(
                    labeled_df,
                    target_cols=target_cols,
                    sample_rows=CFG["sample_rows"],
                    min_positive=CFG["smoke_min_positive"],
                    seed=SEED,
                )
                train_df, val_df = split_with_train_positive_coverage(
                    labeled_df,
                    target_cols=target_cols,
                    val_fraction=CFG["val_fraction"],
                    seed=SEED,
                )

                print({
                    "resolved_mode": RESOLVED_MODE,
                    "labeled_rows": len(labeled_df),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "feature_count": len(feature_cols),
                    "hard_targets": HARD_TARGETS,
                    "external_top1_artifact_dirs": [str(p) for p in EXTERNAL_TOP1_ARTIFACT_DIRS],
                    "external_gandalf_val_files": [str(p) for p in EXTERNAL_GANDALF_VAL_FILES],
                    "external_gandalf_test_files": [str(p) for p in EXTERNAL_GANDALF_TEST_FILES],
                })
                """
            ).strip()
            + "\n"
        ),
        md_cell("## GANDALF Multi-Seed Backbone\n"),
        code_cell(
            dedent(
                """
                from pytorch_tabular import TabularModel
                from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
                from pytorch_tabular.models import GANDALFConfig


                def preprocess_frames(train_features: pd.DataFrame, other_frames: list[pd.DataFrame]):
                    train_out = train_features.copy()
                    others = [frame.copy() for frame in other_frames]
                    for col in cat_cols:
                        train_series = pd.to_numeric(train_out[col], errors="coerce").fillna(-1).astype("int64")
                        categories = pd.Index(train_series.unique())
                        if -1 not in categories:
                            categories = pd.Index(np.concatenate((np.array([-1], dtype=np.int64), categories.to_numpy(dtype=np.int64))))
                        train_out[col] = pd.Categorical(train_series, categories=categories)
                        for frame in others:
                            frame_series = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int64")
                            frame_series = frame_series.where(frame_series.isin(categories), -1)
                            frame[col] = pd.Categorical(frame_series, categories=categories)
                    for col in cont_cols:
                        train_out[col] = pd.to_numeric(train_out[col], errors="coerce").astype("float32")
                        median = float(train_out[col].median())
                        train_out[col] = train_out[col].fillna(median)
                        for frame in others:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32").fillna(median)
                    return train_out.copy(), [frame.copy() for frame in others]


                def clear_runtime_memory() -> None:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()


                def extract_probability_frame(raw_pred: pd.DataFrame) -> pd.DataFrame:
                    pred_map = {}
                    for target_name in target_cols:
                        positive_col = f"{target_name}_1_probability"
                        if positive_col in raw_pred.columns:
                            pred_map[target_name.replace("target_", "predict_")] = raw_pred[positive_col].astype("float64").to_numpy()
                        else:
                            fallback_cols = [c for c in raw_pred.columns if c.startswith(f"{target_name}_") and c.endswith("_probability")]
                            chosen_col = sorted(fallback_cols)[-1] if fallback_cols else None
                            pred_map[target_name.replace("target_", "predict_")] = raw_pred[chosen_col].astype("float64").to_numpy() if chosen_col else np.zeros(len(raw_pred), dtype=np.float64)
                    return pd.DataFrame(pred_map)


                def trainer_accelerator() -> str:
                    return "gpu" if torch.cuda.is_available() and RESOLVED_MODE != "smoke" else "auto"


                val_train_features, [val_features, val_test_features] = preprocess_frames(
                    train_df[feature_cols].reset_index(drop=True),
                    [val_df[feature_cols].reset_index(drop=True), test_df[feature_cols].reset_index(drop=True)],
                )
                train_ready = pd.concat([val_train_features, train_df[target_cols].reset_index(drop=True)], axis=1)
                val_ready = pd.concat([val_features, val_df[target_cols].reset_index(drop=True)], axis=1)

                full_ready = None
                full_test_features = None
                if CFG["train_final_models"]:
                    full_features, [full_test_features] = preprocess_frames(
                        labeled_df[feature_cols].reset_index(drop=True),
                        [test_df[feature_cols].reset_index(drop=True)],
                    )
                    full_ready = pd.concat([full_features, labeled_df[target_cols].reset_index(drop=True)], axis=1)

                gandalf_val_list = []
                gandalf_test_list = []
                gandalf_seed_metrics = []
                gandalf_val_cache = ARTIFACT_DIR / "gandalf_val_submit.parquet"
                gandalf_test_cache = ARTIFACT_DIR / "gandalf_test_submit.parquet"
                gandalf_metrics_cache = ARTIFACT_DIR / "gandalf_seed_metrics.csv"
                expected_pred_cols = [c.replace("target_", "predict_") for c in target_cols]

                if gandalf_val_cache.exists() and (not CFG["train_final_models"] or gandalf_test_cache.exists()):
                    print({"stage": "gandalf_resume_local"})
                    gandalf_val_submit = load_submit_like(gandalf_val_cache, expected_pred_cols)
                    if CFG["train_final_models"]:
                        gandalf_test_submit = load_submit_like(gandalf_test_cache, expected_pred_cols)
                    if gandalf_metrics_cache.exists():
                        gandalf_seed_metrics = pd.read_csv(gandalf_metrics_cache).to_dict(orient="records")
                    else:
                        gandalf_seed_metrics = [{"seed": "local_cache", "validation_macro_auc": macro_auc(val_df[target_cols], gandalf_val_submit, target_cols)}]
                elif EXTERNAL_GANDALF_VAL_FILES and (not CFG["train_final_models"] or EXTERNAL_GANDALF_TEST_FILES):
                    print({"stage": "gandalf_resume_external", "val_file": str(EXTERNAL_GANDALF_VAL_FILES[0])})
                    external_val = normalize_pred_columns(pd.read_parquet(EXTERNAL_GANDALF_VAL_FILES[0]))
                    gandalf_val_submit = external_val.merge(val_df[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
                    gandalf_val_submit = load_submit_like(EXTERNAL_GANDALF_VAL_FILES[0], expected_pred_cols).merge(
                        val_df[["customer_id"]], on="customer_id", how="inner"
                    ).sort_values("customer_id").reset_index(drop=True)
                    if CFG["train_final_models"]:
                        gandalf_test_submit = load_submit_like(EXTERNAL_GANDALF_TEST_FILES[0], expected_pred_cols)
                    gandalf_seed_metrics = [{"seed": "external_kernel", "validation_macro_auc": macro_auc(val_df[target_cols], gandalf_val_submit, target_cols)}]
                    gandalf_val_submit.to_parquet(gandalf_val_cache, index=False)
                    if CFG["train_final_models"]:
                        gandalf_test_submit.to_parquet(gandalf_test_cache, index=False)
                    pd.DataFrame(gandalf_seed_metrics).to_csv(gandalf_metrics_cache, index=False)
                else:
                    for seed in CFG["gandalf_seeds"]:
                        print({"stage": "gandalf_val_fit_start", "seed": seed, "train_rows": len(train_ready), "val_rows": len(val_ready)})

                        data_config = DataConfig(
                            target=target_cols,
                            continuous_cols=cont_cols,
                            categorical_cols=cat_cols,
                            normalize_continuous_features=True,
                            handle_missing_values=True,
                            num_workers=0,
                        )
                        optimizer_config = OptimizerConfig()
                        model_config = GANDALFConfig(
                            task="classification",
                            learning_rate=CFG["gandalf_learning_rate"],
                            metrics=["accuracy"],
                            metrics_prob_input=[False],
                            gflu_stages=CFG["gandalf_gflu_stages"],
                            gflu_dropout=CFG["gandalf_gflu_dropout"],
                        )
                        trainer_config = TrainerConfig(
                            batch_size=CFG["gandalf_batch_size"],
                            max_epochs=CFG["gandalf_max_epochs"],
                            accelerator=trainer_accelerator(),
                            devices=1,
                            progress_bar="none",
                            checkpoints="valid_loss",
                            checkpoints_path=str(ARTIFACT_DIR / f"gandalf_seed_{seed}" / "checkpoints"),
                            load_best=True,
                            seed=seed,
                        )
                        val_model = TabularModel(
                            data_config=data_config,
                            model_config=model_config,
                            optimizer_config=optimizer_config,
                            trainer_config=trainer_config,
                            verbose=False,
                            suppress_lightning_logger=True,
                        )
                        val_model.fit(train=train_ready, validation=val_ready, metrics=[], metrics_prob_inputs=[])
                        val_pred = extract_probability_frame(val_model.predict(val_ready, include_input_features=False, progress_bar="none"))
                        gandalf_val_list.append(val_pred)
                        seed_auc = macro_auc(val_df[target_cols], pd.concat([val_df[["customer_id"]].reset_index(drop=True), val_pred.reset_index(drop=True)], axis=1), target_cols)
                        gandalf_seed_metrics.append({"seed": seed, "validation_macro_auc": seed_auc})
                        print({"stage": "gandalf_val_fit_done", "seed": seed, "validation_macro_auc": seed_auc})
                        del val_model
                        clear_runtime_memory()

                        if CFG["train_final_models"]:
                            print({"stage": "gandalf_final_fit_start", "seed": seed, "train_rows": len(full_ready), "test_rows": len(full_test_features)})
                            final_trainer = TrainerConfig(
                                batch_size=CFG["gandalf_batch_size"],
                                max_epochs=CFG["gandalf_final_epochs"],
                                accelerator=trainer_accelerator(),
                                devices=1,
                                progress_bar="none",
                                checkpoints="valid_loss",
                                checkpoints_path=str(ARTIFACT_DIR / f"gandalf_seed_{seed}" / "final_checkpoints"),
                                load_best=True,
                                seed=seed,
                            )
                            final_model = TabularModel(
                                data_config=data_config,
                                model_config=model_config,
                                optimizer_config=optimizer_config,
                                trainer_config=final_trainer,
                                verbose=False,
                                suppress_lightning_logger=True,
                            )
                            final_model.fit(train=full_ready, validation=None, metrics=[], metrics_prob_inputs=[])
                            test_pred = extract_probability_frame(final_model.predict(full_test_features, include_input_features=False, progress_bar="none"))
                            gandalf_test_list.append(test_pred)
                            print({"stage": "gandalf_final_fit_done", "seed": seed})
                            del final_model
                            clear_runtime_memory()

                    gandalf_val_mean = pd.concat(gandalf_val_list).groupby(level=0).mean()
                    gandalf_val_submit = pd.concat([val_df[["customer_id"]].reset_index(drop=True), gandalf_val_mean.reset_index(drop=True)], axis=1)
                    gandalf_val_submit.to_parquet(gandalf_val_cache, index=False)
                    if CFG["train_final_models"]:
                        gandalf_test_mean = pd.concat(gandalf_test_list).groupby(level=0).mean()
                        gandalf_test_submit = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                        for pred_col in expected_pred_cols:
                            gandalf_test_submit[pred_col] = gandalf_test_mean[pred_col].astype("float64").values
                        gandalf_test_submit.to_parquet(gandalf_test_cache, index=False)
                    pd.DataFrame(gandalf_seed_metrics).to_csv(gandalf_metrics_cache, index=False)

                gandalf_macro = macro_auc(val_df[target_cols], gandalf_val_submit, target_cols)
                gandalf_seed_metrics
                """
            ).strip()
            + "\n"
        ),
        md_cell("## LightGBM Complement Branch\n"),
        code_cell(
            dedent(
                """
                def prepare_lgbm_frames(train_features: pd.DataFrame, other_frames: list[pd.DataFrame]):
                    train_out = train_features.copy()
                    others = [frame.copy() for frame in other_frames]
                    for col in cat_cols:
                        train_out[col] = train_out[col].fillna(-1).astype("int64").astype("category")
                        for frame in others:
                            frame[col] = frame[col].fillna(-1).astype("int64").astype("category")
                    for col in cont_cols:
                        train_out[col] = pd.to_numeric(train_out[col], errors="coerce").astype("float32")
                        median = float(train_out[col].median())
                        train_out[col] = train_out[col].fillna(median)
                        for frame in others:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32").fillna(median)
                    return train_out, others


                xtr_lgbm, [xva_lgbm, xtest_lgbm] = prepare_lgbm_frames(
                    train_df[feature_cols],
                    [val_df[feature_cols], test_df[feature_cols]],
                )
                full_x_lgbm = None
                full_test_lgbm = None
                if CFG["train_final_models"]:
                    full_x_lgbm, [full_test_lgbm] = prepare_lgbm_frames(
                        labeled_df[feature_cols],
                        [test_df[feature_cols]],
                    )
                lgbm_val_cache = ARTIFACT_DIR / "lgbm_val_submit.parquet"
                lgbm_test_cache = ARTIFACT_DIR / "lgbm_test_submit.parquet"
                expected_pred_cols = [c.replace("target_", "predict_") for c in target_cols]
                if lgbm_val_cache.exists() and (not CFG["train_final_models"] or lgbm_test_cache.exists()):
                    print({"stage": "lgbm_resume_local"})
                    lgbm_val_submit = load_submit_like(lgbm_val_cache, expected_pred_cols)
                    if CFG["train_final_models"]:
                        lgbm_test_submit = load_submit_like(lgbm_test_cache, expected_pred_cols)
                else:
                    lgbm_val_submit = pd.DataFrame({"customer_id": val_df["customer_id"].values})
                    lgbm_test_submit = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    print({"stage": "lgbm_branch_start", "target_count": len(target_cols), "train_final_models": bool(CFG["train_final_models"])})

                    for target_idx, target_name in enumerate(target_cols, start=1):
                        print({"stage": "lgbm_target_start", "target": target_name, "index": target_idx, "total": len(target_cols)})
                        ytr = train_df[target_name]
                        pred_col = target_name.replace("target_", "predict_")
                        if ytr.nunique() < 2:
                            const_pred = float(ytr.mean())
                            lgbm_val_submit[pred_col] = const_pred
                            lgbm_test_submit[pred_col] = const_pred
                            print({"stage": "lgbm_target_done", "target": target_name, "index": target_idx, "mode": "constant"})
                            continue
                        pos_rate = float(ytr.mean())
                        model = lgb.LGBMClassifier(
                            objective="binary",
                            metric="auc",
                            boosting_type="gbdt",
                            n_estimators=CFG["lgbm_n_estimators"],
                            learning_rate=CFG["lgbm_learning_rate"],
                            num_leaves=CFG["lgbm_num_leaves"],
                            feature_fraction=0.9,
                            bagging_fraction=0.9,
                            bagging_freq=1,
                            random_state=SEED,
                            n_jobs=-1,
                            verbose=-1,
                            scale_pos_weight=(1 - pos_rate) / (pos_rate + 1e-6),
                        )
                        model.fit(
                            xtr_lgbm,
                            ytr,
                            eval_set=[(xva_lgbm, val_df[target_name])],
                            eval_metric="auc",
                            categorical_feature=cat_cols,
                            callbacks=[lgb.early_stopping(30, verbose=False)],
                        )
                        lgbm_val_submit[pred_col] = model.predict_proba(xva_lgbm)[:, 1]
                        if CFG["train_final_models"]:
                            final_model = lgb.LGBMClassifier(
                                objective="binary",
                                metric="auc",
                                boosting_type="gbdt",
                                n_estimators=int(CFG["lgbm_n_estimators"] * 1.15),
                                learning_rate=CFG["lgbm_learning_rate"],
                                num_leaves=CFG["lgbm_num_leaves"],
                                feature_fraction=0.9,
                                bagging_fraction=0.9,
                                bagging_freq=1,
                                random_state=SEED,
                                n_jobs=-1,
                                verbose=-1,
                                scale_pos_weight=(1 - pos_rate) / (pos_rate + 1e-6),
                            )
                            final_model.fit(full_x_lgbm, labeled_df[target_name], categorical_feature=cat_cols)
                            lgbm_test_submit[pred_col] = final_model.predict_proba(full_test_lgbm)[:, 1]
                            del final_model
                        print({"stage": "lgbm_target_done", "target": target_name, "index": target_idx})

                    lgbm_val_submit.to_parquet(lgbm_val_cache, index=False)
                    if CFG["train_final_models"]:
                        lgbm_test_submit.to_parquet(lgbm_test_cache, index=False)

                lgbm_macro = macro_auc(val_df[target_cols], lgbm_val_submit, target_cols)
                {"gandalf_macro": gandalf_macro, "lgbm_macro": lgbm_macro}
                """
            ).strip()
            + "\n"
        ),
        md_cell("## AutoGluon Hard-Target Specialists\n"),
        code_cell(
            dedent(
                """
                from autogluon.tabular import TabularPredictor

                ag_val_submit = gandalf_val_submit.copy()
                ag_test_submit = None
                if CFG["train_final_models"]:
                    ag_test_submit = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    for pred in [c.replace("target_", "predict_") for c in target_cols]:
                        ag_test_submit[pred] = np.nan

                def build_ag_subset(df: pd.DataFrame, target_name: str, max_rows: int, seed: int) -> pd.DataFrame:
                    if max_rows is None or len(df) <= max_rows:
                        return df.copy()
                    pos_df = df[df[target_name] == 1]
                    neg_df = df[df[target_name] == 0]
                    if len(pos_df) == 0 or len(neg_df) == 0:
                        return df.copy().reset_index(drop=True)
                    pos_take = int(round(max_rows * (len(pos_df) / len(df))))
                    pos_take = min(len(pos_df), max(1, pos_take))
                    neg_take = max_rows - pos_take
                    if neg_take < 1:
                        neg_take = 1
                        pos_take = max_rows - 1
                    neg_take = min(len(neg_df), max(1, neg_take))
                    remaining = max_rows - pos_take - neg_take
                    if remaining > 0:
                        extra_pos = min(len(pos_df) - pos_take, remaining)
                        pos_take += max(0, extra_pos)
                        remaining -= max(0, extra_pos)
                    if remaining > 0:
                        extra_neg = min(len(neg_df) - neg_take, remaining)
                        neg_take += max(0, extra_neg)
                    pos_sample = pos_df.sample(pos_take, random_state=seed)
                    neg_sample = neg_df.sample(neg_take, random_state=seed)
                    return pd.concat([pos_sample, neg_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

                def available_memory_mb() -> float | None:
                    meminfo = Path("/proc/meminfo")
                    if meminfo.exists():
                        values = {}
                        for line in meminfo.read_text().splitlines():
                            if ":" not in line:
                                continue
                            key, value = line.split(":", 1)
                            parts = value.strip().split()
                            if parts:
                                values[key] = float(parts[0]) / 1024.0
                        if "MemAvailable" in values:
                            return values["MemAvailable"]
                    return None

                def estimate_ag_memory_mb(train_ag: pd.DataFrame, val_ag: pd.DataFrame) -> float:
                    train_mb = train_ag.memory_usage(deep=True).sum() / (1024 ** 2)
                    val_mb = val_ag.memory_usage(deep=True).sum() / (1024 ** 2)
                    return (train_mb + val_mb) * CFG["autogluon_memory_multiplier"]

                def fit_safe_ag_subset(df: pd.DataFrame, val_ag: pd.DataFrame, target_name: str, seed: int) -> pd.DataFrame | None:
                    max_rows = CFG["autogluon_max_rows"]
                    min_rows = CFG["autogluon_min_rows"]
                    candidate = build_ag_subset(df, target_name, max_rows, seed)
                    available_mb = available_memory_mb()
                    estimated_mb = estimate_ag_memory_mb(candidate, val_ag)
                    print({
                        "stage": "autogluon_memory_check",
                        "target": target_name,
                        "rows": len(candidate),
                        "estimated_required_mb": round(estimated_mb, 1),
                        "available_mb": None if available_mb is None else round(available_mb, 1),
                    })
                    while (
                        available_mb is not None
                        and estimated_mb > available_mb * CFG["autogluon_memory_fraction"]
                        and len(candidate) > min_rows
                    ):
                        next_rows = max(min_rows, int(len(candidate) * 0.75))
                        if next_rows >= len(candidate):
                            break
                        candidate = build_ag_subset(df, target_name, next_rows, seed)
                        estimated_mb = estimate_ag_memory_mb(candidate, val_ag)
                        print({
                            "stage": "autogluon_memory_reduce",
                            "target": target_name,
                            "rows": len(candidate),
                            "estimated_required_mb": round(estimated_mb, 1),
                            "available_mb": round(available_mb, 1),
                        })
                    if available_mb is not None and estimated_mb > available_mb * CFG["autogluon_memory_fraction"]:
                        print({
                            "stage": "autogluon_target_skipped",
                            "target": target_name,
                            "reason": "memory_guard",
                            "rows": len(candidate),
                            "estimated_required_mb": round(estimated_mb, 1),
                            "available_mb": round(available_mb, 1),
                        })
                        return None
                    if candidate[target_name].nunique() < 2:
                        print({
                            "stage": "autogluon_target_skipped",
                            "target": target_name,
                            "reason": "one_class_subset",
                            "unique_values": candidate[target_name].dropna().unique().tolist(),
                            "rows": len(candidate),
                        })
                        return None
                    return candidate


                def try_load_predictor(path: Path) -> TabularPredictor | None:
                    try:
                        if path.exists() and (path / "predictor.pkl").exists():
                            return TabularPredictor.load(str(path))
                    except Exception as exc:
                        print({"stage": "autogluon_predictor_load_failed", "path": str(path), "error": str(exc)})
                    return None


                def find_resume_predictor(target_name: str, final: bool = False) -> TabularPredictor | None:
                    suffix = "_full" if final else ""
                    local_path = ARTIFACT_DIR / "autogluon_hard" / f"{target_name}{suffix}"
                    predictor = try_load_predictor(local_path)
                    if predictor is not None:
                        return predictor
                    for base_dir in EXTERNAL_TOP1_ARTIFACT_DIRS:
                        predictor = try_load_predictor(base_dir / "autogluon_hard" / f"{target_name}{suffix}")
                        if predictor is not None:
                            return predictor
                    return None


                ag_val_cache = ARTIFACT_DIR / "ag_val_submit.parquet"
                ag_test_cache = ARTIFACT_DIR / "ag_test_submit.parquet"
                ag_rows_cache = ARTIFACT_DIR / "ag_target_rows_running.csv"
                if ag_val_cache.exists():
                    print({"stage": "autogluon_resume_cache_val"})
                    ag_val_submit = load_submit_like(ag_val_cache, [c.replace("target_", "predict_") for c in target_cols])
                if CFG["train_final_models"] and ag_test_cache.exists():
                    print({"stage": "autogluon_resume_cache_test"})
                    ag_test_submit = load_submit_like(ag_test_cache, [c.replace("target_", "predict_") for c in target_cols])

                ag_target_rows = []
                if ag_rows_cache.exists():
                    ag_target_rows = pd.read_csv(ag_rows_cache).to_dict(orient="records")

                def save_ag_progress() -> None:
                    ag_val_submit.to_parquet(ag_val_cache, index=False)
                    if CFG["train_final_models"] and ag_test_submit is not None:
                        ag_test_submit.to_parquet(ag_test_cache, index=False)
                    pd.DataFrame(ag_target_rows).to_csv(ag_rows_cache, index=False)

                HARD_TARGETS_RUNTIME = HARD_TARGETS
                print({"stage": "autogluon_branch_start", "hard_target_count": len(HARD_TARGETS_RUNTIME), "train_final_models": bool(CFG["train_final_models"])})
                for target_idx, target_name in enumerate(HARD_TARGETS_RUNTIME, start=1):
                    print({"stage": "autogluon_target_start", "target": target_name, "index": target_idx, "total": len(HARD_TARGETS_RUNTIME)})
                    pred_col = target_name.replace("target_", "predict_")
                    has_cached_row = any(row.get("target") == target_name for row in ag_target_rows)
                    has_cached_test = (not CFG["train_final_models"]) or (ag_test_submit is not None and pred_col in ag_test_submit.columns and ag_test_submit[pred_col].notna().any())
                    if has_cached_row and has_cached_test:
                        print({"stage": "autogluon_target_done", "target": target_name, "index": target_idx, "mode": "resume_cached_row"})
                        continue
                    ag_path = ARTIFACT_DIR / "autogluon_hard" / target_name
                    train_ag_full = train_df[feature_cols + [target_name]].copy()
                    val_ag = val_df[feature_cols + [target_name]].copy()
                    predictor = find_resume_predictor(target_name, final=False)
                    if predictor is None:
                        train_ag = fit_safe_ag_subset(train_ag_full, val_ag, target_name, seed=SEED + target_idx)
                        if train_ag is None:
                            continue
                        predictor = TabularPredictor(
                            label=target_name,
                            problem_type="binary",
                            eval_metric="roc_auc",
                            path=str(ag_path),
                        )
                        predictor.fit(
                            train_data=train_ag,
                            tuning_data=val_ag,
                            presets=CFG["autogluon_presets"],
                            num_gpus=0,
                            auto_stack=False,
                            dynamic_stacking=False,
                            num_bag_folds=0,
                            num_stack_levels=0,
                            fit_strategy="sequential",
                            time_limit=CFG["autogluon_time_limit"],
                                verbosity=2,
                            )
                    else:
                        print({"stage": "autogluon_resume_predictor", "target": target_name, "path": str(predictor.path)})
                    try:
                        ag_val_submit[pred_col] = pd.Series(predictor.predict_proba(val_df[feature_cols], as_multiclass=False)).astype("float64").values
                    except Exception as exc:
                        print({"stage": "autogluon_resume_predictor_failed", "target": target_name, "path": str(getattr(predictor, "path", "")), "error": str(exc)})
                        train_ag = fit_safe_ag_subset(train_ag_full, val_ag, target_name, seed=SEED + target_idx)
                        if train_ag is None:
                            continue
                        predictor = TabularPredictor(
                            label=target_name,
                            problem_type="binary",
                            eval_metric="roc_auc",
                            path=str(ag_path),
                        )
                        predictor.fit(
                            train_data=train_ag,
                            tuning_data=val_ag,
                            presets=CFG["autogluon_presets"],
                            num_gpus=0,
                            auto_stack=False,
                            dynamic_stacking=False,
                            num_bag_folds=0,
                            num_stack_levels=0,
                            fit_strategy="sequential",
                            time_limit=CFG["autogluon_time_limit"],
                            verbosity=2,
                        )
                        ag_val_submit[pred_col] = pd.Series(predictor.predict_proba(val_df[feature_cols], as_multiclass=False)).astype("float64").values
                    ag_score = roc_auc_score(val_df[target_name], ag_val_submit[pred_col])
                    if not any(row.get("target") == target_name for row in ag_target_rows):
                        ag_target_rows.append({"target": target_name, "oof_auc": float(ag_score)})
                    save_ag_progress()

                    if CFG["train_final_models"]:
                        full_predictor = find_resume_predictor(target_name, final=True)
                        if full_predictor is None:
                            full_ag_raw = labeled_df[feature_cols + [target_name]].copy()
                            full_ag = fit_safe_ag_subset(full_ag_raw, val_ag, target_name, seed=SEED + 100 + target_idx)
                            if full_ag is None:
                                print({"stage": "autogluon_target_done", "target": target_name, "index": target_idx, "oof_auc": float(ag_score), "final_model": "skipped_memory_guard"})
                                save_ag_progress()
                                continue
                            full_predictor = TabularPredictor(
                                label=target_name,
                                problem_type="binary",
                                eval_metric="roc_auc",
                                path=str(ag_path) + "_full",
                            )
                            full_predictor.fit(
                                train_data=full_ag,
                                presets=CFG["autogluon_presets"],
                                num_gpus=0,
                                auto_stack=False,
                                dynamic_stacking=False,
                                num_bag_folds=0,
                                num_stack_levels=0,
                                fit_strategy="sequential",
                                time_limit=CFG["autogluon_final_time_limit"],
                                verbosity=2,
                            )
                        else:
                            print({"stage": "autogluon_resume_predictor_full", "target": target_name, "path": str(full_predictor.path)})
                        try:
                            ag_test_submit[pred_col] = pd.Series(full_predictor.predict_proba(test_df[feature_cols], as_multiclass=False)).astype("float64").values
                        except Exception as exc:
                            print({"stage": "autogluon_resume_predictor_full_failed", "target": target_name, "path": str(getattr(full_predictor, "path", "")), "error": str(exc)})
                            full_ag_raw = labeled_df[feature_cols + [target_name]].copy()
                            full_ag = fit_safe_ag_subset(full_ag_raw, val_ag, target_name, seed=SEED + 100 + target_idx)
                            if full_ag is None:
                                print({"stage": "autogluon_target_done", "target": target_name, "index": target_idx, "oof_auc": float(ag_score), "final_model": "skipped_after_resume_failure"})
                                save_ag_progress()
                                continue
                            full_predictor = TabularPredictor(
                                label=target_name,
                                problem_type="binary",
                                eval_metric="roc_auc",
                                path=str(ag_path) + "_full",
                            )
                            full_predictor.fit(
                                train_data=full_ag,
                                presets=CFG["autogluon_presets"],
                                num_gpus=0,
                                auto_stack=False,
                                dynamic_stacking=False,
                                num_bag_folds=0,
                                num_stack_levels=0,
                                fit_strategy="sequential",
                                time_limit=CFG["autogluon_final_time_limit"],
                                verbosity=2,
                            )
                            ag_test_submit[pred_col] = pd.Series(full_predictor.predict_proba(test_df[feature_cols], as_multiclass=False)).astype("float64").values
                        save_ag_progress()
                    print({"stage": "autogluon_target_done", "target": target_name, "index": target_idx, "oof_auc": float(ag_score)})

                ag_target_rows = pd.DataFrame(ag_target_rows).sort_values("oof_auc").reset_index(drop=True)
                print({"stage": "autogluon_branch_done", "rows": len(ag_target_rows)})
                ag_target_rows
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Holdout-Learned Final Ensemble\n"),
        code_cell(
            dedent(
                """
                candidate_val_frames = {
                    "gandalf_mean": gandalf_val_submit,
                    "gandalf_95_lgbm_05": pd.concat(
                        [
                            val_df[["customer_id"]].reset_index(drop=True),
                            (0.95 * gandalf_val_submit.drop(columns=["customer_id"]).reset_index(drop=True) + 0.05 * lgbm_val_submit.drop(columns=["customer_id"]).reset_index(drop=True)),
                        ],
                        axis=1,
                    ),
                    "gandalf_90_lgbm_10": pd.concat(
                        [
                            val_df[["customer_id"]].reset_index(drop=True),
                            (0.9 * gandalf_val_submit.drop(columns=["customer_id"]).reset_index(drop=True) + 0.1 * lgbm_val_submit.drop(columns=["customer_id"]).reset_index(drop=True)),
                        ],
                        axis=1,
                    ),
                    "gandalf_aghard_override": ag_val_submit,
                }
                candidate_val_frames["gandalf_90_lgbm_10_aghard_override"] = candidate_val_frames["gandalf_90_lgbm_10"].copy()
                candidate_val_frames["gandalf_95_lgbm_05_aghard_override"] = candidate_val_frames["gandalf_95_lgbm_05"].copy()
                for pred_col in [c.replace("target_", "predict_") for c in HARD_TARGETS]:
                    candidate_val_frames["gandalf_90_lgbm_10_aghard_override"][pred_col] = ag_val_submit[pred_col].values
                    candidate_val_frames["gandalf_95_lgbm_05_aghard_override"][pred_col] = ag_val_submit[pred_col].values

                def build_label_graph_weights(target_df: pd.DataFrame, target_cols: list[str], top_k: int = 3) -> np.ndarray:
                    y = target_df[target_cols].to_numpy(dtype=np.float64)
                    priors = y.mean(axis=0)
                    joint = (y.T @ y) / len(y)
                    lift = joint / np.maximum(np.outer(priors, priors), 1e-6)
                    np.fill_diagonal(lift, 1.0)
                    weights = np.log(np.maximum(lift, 1.0))
                    np.fill_diagonal(weights, 0.0)
                    out = np.zeros_like(weights)
                    for i in range(weights.shape[0]):
                        idx = np.argsort(weights[i])[::-1][:top_k]
                        out[i, idx] = weights[i, idx]
                    return out

                def graph_smooth(base_pred: pd.DataFrame, target_df: pd.DataFrame, target_cols: list[str], weights: np.ndarray, alpha: float) -> pd.DataFrame:
                    pred_cols = [c.replace("target_", "predict_") for c in target_cols]
                    p = base_pred[pred_cols].to_numpy(dtype=np.float64)
                    priors = target_df[target_cols].mean(axis=0).to_numpy(dtype=np.float64)
                    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
                    neighbor_effect = (p - priors) @ weights.T
                    smoothed = 1.0 / (1.0 + np.exp(-(logits + alpha * neighbor_effect)))
                    out = pd.DataFrame({"customer_id": base_pred["customer_id"].astype("int32").values})
                    for i, pred_col in enumerate(pred_cols):
                        out[pred_col] = smoothed[:, i].astype("float64")
                    return out

                graph_weights = build_label_graph_weights(labeled_df, target_cols, top_k=3)
                candidate_val_frames["gandalf_95_lgbm_05_graphsmooth"] = graph_smooth(
                    candidate_val_frames["gandalf_95_lgbm_05"],
                    labeled_df,
                    target_cols,
                    graph_weights,
                    alpha=0.20,
                )
                candidate_val_frames["gandalf_95_lgbm_05_aghard_override_graphsmooth"] = graph_smooth(
                    candidate_val_frames["gandalf_95_lgbm_05_aghard_override"],
                    labeled_df,
                    target_cols,
                    graph_weights,
                    alpha=0.20,
                )

                candidate_rows = []
                for name, frame in candidate_val_frames.items():
                    candidate_rows.append({"candidate": name, "macro_auc": macro_auc(val_df[target_cols], frame, target_cols)})
                candidate_metrics = pd.DataFrame(candidate_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
                print({"stage": "candidate_metrics_ready", "candidate_count": len(candidate_metrics), "best_candidate": str(candidate_metrics.iloc[0]["candidate"]), "best_macro_auc": float(candidate_metrics.iloc[0]["macro_auc"])})
                candidate_metrics
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                target_choice_rows = []
                for target_name in target_cols:
                    pred_col = target_name.replace("target_", "predict_")
                    best_name = None
                    best_score = -1.0
                    for name, frame in candidate_val_frames.items():
                        score = roc_auc_score(val_df[target_name], frame[pred_col]) if val_df[target_name].nunique() > 1 else 0.5
                        if score > best_score:
                            best_score = score
                            best_name = name
                    target_choice_rows.append({"target": target_name, "choice": best_name, "best_auc": float(best_score)})
                target_choices = pd.DataFrame(target_choice_rows).sort_values("best_auc").reset_index(drop=True)
                print({"stage": "target_choices_ready", "target_count": len(target_choices), "worst_target": str(target_choices.iloc[0]["target"]), "worst_best_auc": float(target_choices.iloc[0]["best_auc"])})
                target_choices
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                if CFG["train_final_models"]:
                    print({"stage": "submission_build_start"})
                    combo_test_submit = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    for pred_col in [c.replace("target_", "predict_") for c in target_cols]:
                        combo_test_submit[pred_col] = (0.9 * gandalf_test_submit[pred_col] + 0.1 * lgbm_test_submit[pred_col]).astype("float64").values
                    combo95_test_submit = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    for pred_col in [c.replace("target_", "predict_") for c in target_cols]:
                        combo95_test_submit[pred_col] = (0.95 * gandalf_test_submit[pred_col] + 0.05 * lgbm_test_submit[pred_col]).astype("float64").values

                    aghard_test_submit = gandalf_test_submit.copy()
                    if ag_test_submit is not None:
                        for target_name in HARD_TARGETS:
                            pred_col = target_name.replace("target_", "predict_")
                            if pred_col in ag_test_submit.columns and ag_test_submit[pred_col].notna().any():
                                aghard_test_submit[pred_col] = ag_test_submit[pred_col].astype("float64").values

                    combo_aghard_test_submit = combo_test_submit.copy()
                    combo95_aghard_test_submit = combo95_test_submit.copy()
                    if ag_test_submit is not None:
                        for target_name in HARD_TARGETS:
                            pred_col = target_name.replace("target_", "predict_")
                            if pred_col in ag_test_submit.columns and ag_test_submit[pred_col].notna().any():
                                combo_aghard_test_submit[pred_col] = ag_test_submit[pred_col].astype("float64").values
                                combo95_aghard_test_submit[pred_col] = ag_test_submit[pred_col].astype("float64").values

                    candidate_test_frames = {
                        "gandalf_mean": gandalf_test_submit,
                        "gandalf_95_lgbm_05": combo95_test_submit,
                        "gandalf_90_lgbm_10": combo_test_submit,
                        "gandalf_aghard_override": aghard_test_submit,
                        "gandalf_95_lgbm_05_aghard_override": combo95_aghard_test_submit,
                        "gandalf_90_lgbm_10_aghard_override": combo_aghard_test_submit,
                    }
                    candidate_test_frames["gandalf_95_lgbm_05_graphsmooth"] = graph_smooth(
                        candidate_test_frames["gandalf_95_lgbm_05"],
                        labeled_df,
                        target_cols,
                        graph_weights,
                        alpha=0.20,
                    )
                    candidate_test_frames["gandalf_95_lgbm_05_aghard_override_graphsmooth"] = graph_smooth(
                        candidate_test_frames["gandalf_95_lgbm_05_aghard_override"],
                        labeled_df,
                        target_cols,
                        graph_weights,
                        alpha=0.20,
                    )

                    best_global_name = candidate_metrics.iloc[0]["candidate"]
                    best_global_submission = candidate_test_frames[best_global_name].copy()
                    best_targetwise_submission = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        choice = target_choices.loc[target_choices["target"] == target_name, "choice"].iloc[0]
                        best_targetwise_submission[pred_col] = candidate_test_frames[choice][pred_col].astype("float64").values

                    pd.DataFrame(gandalf_seed_metrics).to_csv(ARTIFACT_DIR / "gandalf_seed_metrics.csv", index=False)
                    compute_target_scores(val_df[target_cols], gandalf_val_submit, target_cols).to_csv(ARTIFACT_DIR / "gandalf_target_scores.csv", index=False)
                    compute_target_scores(val_df[target_cols], lgbm_val_submit, target_cols).to_csv(ARTIFACT_DIR / "lgbm_target_scores.csv", index=False)
                    ag_target_rows.to_csv(ARTIFACT_DIR / "autogluon_hard_target_scores.csv", index=False)
                    candidate_metrics.to_csv(ARTIFACT_DIR / "candidate_metrics.csv", index=False)
                    target_choices.to_csv(ARTIFACT_DIR / "target_choices.csv", index=False)

                    gandalf_test_submit.to_parquet(ARTIFACT_DIR / "submission_gandalf_mean.parquet", index=False)
                    lgbm_test_submit.to_parquet(ARTIFACT_DIR / "submission_lgbm.parquet", index=False)
                    best_global_submission.to_parquet(ARTIFACT_DIR / "submission_best_global.parquet", index=False)
                    best_targetwise_submission.to_parquet(ARTIFACT_DIR / "submission_best_targetwise.parquet", index=False)

                    summary = {
                        "mode": RESOLVED_MODE,
                        "gandalf_holdout_macro_auc": float(gandalf_macro),
                        "lgbm_holdout_macro_auc": float(lgbm_macro),
                        "best_global_candidate": str(best_global_name),
                        "best_global_holdout_macro_auc": float(candidate_metrics.iloc[0]["macro_auc"]),
                        "best_targetwise_holdout_macro_auc": float(target_choices["best_auc"].mean()),
                    }
                    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
                    print({"stage": "submission_build_done", **summary})
                    summary
                else:
                    print("Smoke mode: final submission writing skipped.")
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Expected Outputs

                The notebook writes the following to its artifact directory:
                - `gandalf_seed_metrics.csv`
                - `candidate_metrics.csv`
                - `target_choices.csv`
                - `summary.json`
                - `submission_gandalf_mean.parquet`
                - `submission_lgbm.parquet`
                - `submission_best_global.parquet`
                - `submission_best_targetwise.parquet`
                """
            ).strip()
            + "\n"
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

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
