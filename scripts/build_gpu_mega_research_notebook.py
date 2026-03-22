from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root

ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook/data-fusion-2026-gpu-mega-research.ipynb"


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
                # Experiment: Data Fusion 2026 GPU Mega Research

                Objective:
                - Run the strongest currently justified branches in one notebook.
                - Produce comparable holdout metrics, per-target diagnostics, and ready-to-upload submissions.
                - Favor ideas that already have evidence of complementarity instead of collecting more weak baselines.

                Included branches:
                - `GANDALF` multi-seed multi-target deep model.
                - `LightGBM` per-target baseline on the same holdout split.
                - `AutoGluon` hard-target specialists for the weakest tail.
                - Holdout-based blend search and final target-wise blend submission.

                Intentionally excluded:
                - `TabNet`: prior runs underperformed materially and wasted GPU budget.
                - `AutoGluon extreme`: official docs recommend it primarily for datasets below `100k`; our task is much larger.
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
                import math
                import os
                import warnings
                from pathlib import Path

                import numpy as np
                import pandas as pd
                import torch
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit
                import lightgbm as lgb

                warnings.filterwarnings("ignore")

                RUN_MODE = "full_gpu"  # one of: smoke, full_gpu
                SEED = 42
                TOP_EXTRA_FEATURES = {top_features_literal}
                HARD_TARGETS = {hard_targets_literal}

                CFG_BY_MODE = {{
                    "smoke": {{
                        "sample_rows": 120000,
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
                        "lgbm_n_estimators": 120,
                        "lgbm_learning_rate": 0.05,
                        "lgbm_num_leaves": 63,
                        "autogluon_enabled": False,
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
                        "gandalf_seeds": [42, 52],
                        "gandalf_max_epochs": 20,
                        "gandalf_final_epochs": 28,
                        "gandalf_batch_size": 2048,
                        "gandalf_learning_rate": 0.0008,
                        "gandalf_gflu_stages": 6,
                        "gandalf_gflu_dropout": 0.15,
                        "lgbm_n_estimators": 300,
                        "lgbm_learning_rate": 0.05,
                        "lgbm_num_leaves": 63,
                        "autogluon_enabled": True,
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
                    return mode


                def find_data_dir() -> Path:
                    candidates = [
                        Path.cwd() / "data" / "competition",
                        Path.cwd(),
                        Path("/kaggle/input"),
                        Path("/kaggle/working"),
                    ]
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
                    train_df = df.iloc[sorted(train_idx)].reset_index(drop=True)
                    val_df = df.iloc[sorted(val_idx)].reset_index(drop=True)
                    return train_df, val_df


                def compute_target_scores(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        if y_true[target_name].nunique() < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y_true[target_name], pred_df[pred_col])
                        rows.append({{"target": target_name, "oof_auc": float(score)}})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def macro_auc(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    return float(compute_target_scores(y_true, pred_df, target_cols)["oof_auc"].mean())


                def normalize_submission_frame(df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    pred_cols = [c.replace("target_", "predict_") for c in target_cols]
                    submit = pd.DataFrame({{"customer_id": df["customer_id"].astype("int32").values}})
                    for pred_col in pred_cols:
                        source_col = pred_col if pred_col in df.columns else pred_col.replace("predict_", "target_")
                        submit[pred_col] = df[source_col].astype("float64").values
                    return submit
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Load Data\n"),
        code_cell(
            dedent(
                """
                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = CFG_BY_MODE[RESOLVED_MODE]
                ARTIFACT_DIR = Path("artifacts/gpu_mega_research_" + RESOLVED_MODE)
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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
                    "data_dir": str(DATA_DIR),
                    "labeled_rows": len(labeled_df),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "feature_count": len(feature_cols),
                    "hard_targets": HARD_TARGETS,
                })
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Branch 1: LightGBM Baseline

                This branch is CPU-friendly and gives us a strong, interpretable tabular comparator on the same holdout split.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                def prepare_lgbm_frames(train_features: pd.DataFrame, other_frames: list[pd.DataFrame], cat_cols: list[str], cont_cols: list[str]):
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


                xtr_lgbm, [xva_lgbm, xtest_lgbm] = prepare_lgbm_frames(
                    train_df[feature_cols],
                    [val_df[feature_cols], test_df[feature_cols]],
                    cat_cols=cat_cols,
                    cont_cols=cont_cols,
                )
                full_x_lgbm = None
                full_test_lgbm = None
                if CFG["train_final_models"]:
                    full_x_lgbm, [full_test_lgbm] = prepare_lgbm_frames(
                        labeled_df[feature_cols],
                        [test_df[feature_cols]],
                        cat_cols=cat_cols,
                        cont_cols=cont_cols,
                    )

                lgbm_val_pred = pd.DataFrame({"customer_id": val_df["customer_id"].values})
                lgbm_test_pred = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                print({"stage": "lgbm_branch_start", "target_count": len(target_cols), "train_final_models": bool(CFG["train_final_models"])})

                for target_idx, target_name in enumerate(target_cols, start=1):
                    print({"stage": "lgbm_target_start", "target": target_name, "index": target_idx, "total": len(target_cols)})
                    ytr = train_df[target_name]
                    if ytr.nunique() < 2:
                        const_pred = float(ytr.mean())
                        lgbm_val_pred[target_name.replace("target_", "predict_")] = const_pred
                        lgbm_test_pred[target_name.replace("target_", "predict_")] = const_pred
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
                    lgbm_val_pred[target_name.replace("target_", "predict_")] = model.predict_proba(xva_lgbm)[:, 1]
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
                        lgbm_test_pred[target_name.replace("target_", "predict_")] = final_model.predict_proba(full_test_lgbm)[:, 1]
                        del final_model
                    del model
                    gc.collect()
                    print({"stage": "lgbm_target_done", "target": target_name, "index": target_idx})

                lgbm_target_scores = compute_target_scores(val_df[target_cols], lgbm_val_pred, target_cols)
                lgbm_macro = float(lgbm_target_scores["oof_auc"].mean())
                print({"lgbm_validation_macro_auc": lgbm_macro})
                lgbm_target_scores.head(12)
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Branch 2: GANDALF Multi-Seed

                This is the current strongest model family. We keep the proven workaround:
                compute validation metrics outside the library and disable the broken default multitask metric path in `fit(...)`.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                from pytorch_tabular import TabularModel
                from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
                from pytorch_tabular.models import GANDALFConfig


                def trainer_accelerator_for_mode(mode: str) -> str:
                    return "gpu" if torch.cuda.is_available() and mode != "smoke" else "auto"


                def preprocess_frames(train_features: pd.DataFrame, other_frames: list[pd.DataFrame], cat_cols: list[str], cont_cols: list[str]):
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


                def extract_probability_frame(raw_pred: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    pred_map = {}
                    for target_name in target_cols:
                        positive_col = f"{target_name}_1_probability"
                        if positive_col in raw_pred.columns:
                            pred_map[target_name.replace("target_", "predict_")] = raw_pred[positive_col].astype("float64").to_numpy()
                            continue
                        fallback_cols = [c for c in raw_pred.columns if c.startswith(f"{target_name}_") and c.endswith("_probability")]
                        if fallback_cols:
                            chosen_col = sorted(fallback_cols)[-1]
                            pred_map[target_name.replace("target_", "predict_")] = raw_pred[chosen_col].astype("float64").to_numpy()
                        else:
                            pred_map[target_name.replace("target_", "predict_")] = np.zeros(len(raw_pred), dtype=np.float64)
                    return pd.DataFrame(pred_map)


                train_features, [val_features, test_features_branch] = preprocess_frames(
                    train_df[feature_cols].reset_index(drop=True),
                    [val_df[feature_cols].reset_index(drop=True), test_df[feature_cols].reset_index(drop=True)],
                    cat_cols=cat_cols,
                    cont_cols=cont_cols,
                )
                train_ready = pd.concat([train_features, train_df[target_cols].reset_index(drop=True)], axis=1)
                val_ready = pd.concat([val_features, val_df[target_cols].reset_index(drop=True)], axis=1)

                full_ready = None
                test_features_final = None
                if CFG["train_final_models"]:
                    full_features, [test_features_final] = preprocess_frames(
                        labeled_df[feature_cols].reset_index(drop=True),
                        [test_df[feature_cols].reset_index(drop=True)],
                        cat_cols=cat_cols,
                        cont_cols=cont_cols,
                    )
                    full_ready = pd.concat([full_features, labeled_df[target_cols].reset_index(drop=True)], axis=1)

                gandalf_val_candidates = []
                gandalf_test_candidates = []
                gandalf_seed_scores = []

                for seed in CFG["gandalf_seeds"]:
                    print({"stage": "gandalf_val_fit_start", "seed": seed, "train_rows": len(train_ready), "val_rows": len(val_ready)})

                    trainer_config = TrainerConfig(
                        batch_size=CFG["gandalf_batch_size"],
                        max_epochs=CFG["gandalf_max_epochs"],
                        accelerator=trainer_accelerator_for_mode(RESOLVED_MODE),
                        devices=1,
                        progress_bar="none",
                        checkpoints="valid_loss",
                        checkpoints_path=str(ARTIFACT_DIR / f"gandalf_seed_{seed}" / "checkpoints"),
                        load_best=True,
                        seed=seed,
                    )
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
                    val_model = TabularModel(
                        data_config=data_config,
                        model_config=model_config,
                        optimizer_config=optimizer_config,
                        trainer_config=trainer_config,
                        verbose=False,
                        suppress_lightning_logger=True,
                    )
                    val_model.fit(train=train_ready, validation=val_ready, metrics=[], metrics_prob_inputs=[])
                    raw_val_pred = val_model.predict(val_ready, include_input_features=False, progress_bar="none")
                    val_pred_df = extract_probability_frame(raw_val_pred, target_cols)
                    gandalf_val_candidates.append(val_pred_df)
                    seed_macro = macro_auc(val_df[target_cols], val_pred_df, target_cols)
                    gandalf_seed_scores.append({"seed": seed, "validation_macro_auc": seed_macro})
                    print({"stage": "gandalf_val_fit_done", "seed": seed, "validation_macro_auc": seed_macro})
                    del val_model
                    clear_runtime_memory()

                    if CFG["train_final_models"]:
                        print({"stage": "gandalf_final_fit_start", "seed": seed, "train_rows": len(full_ready), "test_rows": len(test_features_final)})
                        final_trainer_config = TrainerConfig(
                            batch_size=CFG["gandalf_batch_size"],
                            max_epochs=CFG["gandalf_final_epochs"],
                            accelerator=trainer_accelerator_for_mode(RESOLVED_MODE),
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
                            trainer_config=final_trainer_config,
                            verbose=False,
                            suppress_lightning_logger=True,
                        )
                        final_model.fit(train=full_ready, validation=None, metrics=[], metrics_prob_inputs=[])
                        raw_test_pred = final_model.predict(test_features_final, include_input_features=False, progress_bar="none")
                        gandalf_test_candidates.append(extract_probability_frame(raw_test_pred, target_cols))
                        print({"stage": "gandalf_final_fit_done", "seed": seed})
                        del final_model
                        clear_runtime_memory()

                gandalf_val_mean = pd.concat(gandalf_val_candidates).groupby(level=0).mean()
                gandalf_test_mean = pd.concat(gandalf_test_candidates).groupby(level=0).mean() if gandalf_test_candidates else None
                gandalf_macro = macro_auc(val_df[target_cols], gandalf_val_mean, target_cols)
                print({"gandalf_seed_scores": gandalf_seed_scores, "gandalf_mean_validation_macro_auc": gandalf_macro})
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Branch 3: AutoGluon Hard-Target Specialists

                This branch is intentionally narrow. Local analysis showed the biggest remaining pain is the weakest tail, not the already-strong targets.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                autogluon_val_pred = pd.DataFrame({"customer_id": val_df["customer_id"].values})
                autogluon_test_pred = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                autogluon_scores = []
                autogluon_enabled_runtime = bool(CFG["autogluon_enabled"])
                print({"stage": "autogluon_branch_ready", "enabled": autogluon_enabled_runtime, "hard_target_count": len(HARD_TARGETS)})

                def build_ag_subset(df: pd.DataFrame, target_name: str, max_rows: int, seed: int) -> pd.DataFrame:
                    if max_rows is None or len(df) <= max_rows:
                        return df.copy()
                    pos_df = df[df[target_name] == 1]
                    neg_df = df[df[target_name] == 0]
                    if len(pos_df) >= max_rows:
                        return pos_df.sample(max_rows, random_state=seed).copy()
                    neg_take = min(len(neg_df), max_rows - len(pos_df))
                    if neg_take <= 0:
                        return pos_df.copy()
                    neg_sample = neg_df.sample(neg_take, random_state=seed)
                    return pd.concat([pos_df, neg_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)


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
                    return candidate

                if autogluon_enabled_runtime:
                    try:
                        from autogluon.tabular import TabularPredictor

                        print({"stage": "autogluon_branch_start", "hard_target_count": len(HARD_TARGETS), "train_final_models": bool(CFG["train_final_models"])})
                        for target_idx, target_name in enumerate(HARD_TARGETS, start=1):
                            print({"stage": "autogluon_target_start", "target": target_name, "index": target_idx, "total": len(HARD_TARGETS)})
                            ag_path = ARTIFACT_DIR / "autogluon" / target_name
                            train_ag_full = train_df[feature_cols + [target_name]].copy()
                            val_ag = val_df[feature_cols + [target_name]].copy()
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
                            val_prob = predictor.predict_proba(val_df[feature_cols], as_multiclass=False)
                            autogluon_val_pred[target_name.replace("target_", "predict_")] = pd.Series(val_prob).astype("float64").values
                            if CFG["train_final_models"]:
                                full_ag_raw = labeled_df[feature_cols + [target_name]].copy()
                                full_ag = fit_safe_ag_subset(full_ag_raw, val_ag, target_name, seed=SEED + 100 + target_idx)
                                if full_ag is None:
                                    score = roc_auc_score(val_df[target_name], autogluon_val_pred[target_name.replace("target_", "predict_")])
                                    autogluon_scores.append({"target": target_name, "oof_auc": float(score)})
                                    print({"stage": "autogluon_target_done", "target": target_name, "index": target_idx, "oof_auc": float(score), "final_model": "skipped_memory_guard"})
                                    continue
                                final_predictor = TabularPredictor(
                                    label=target_name,
                                    problem_type="binary",
                                    eval_metric="roc_auc",
                                    path=str(ag_path) + "_full",
                                )
                                final_predictor.fit(
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
                                test_prob = final_predictor.predict_proba(test_df[feature_cols], as_multiclass=False)
                                autogluon_test_pred[target_name.replace("target_", "predict_")] = pd.Series(test_prob).astype("float64").values
                            score = roc_auc_score(val_df[target_name], autogluon_val_pred[target_name.replace("target_", "predict_")])
                            autogluon_scores.append({"target": target_name, "oof_auc": float(score)})
                            print({"stage": "autogluon_target_done", "target": target_name, "index": target_idx, "oof_auc": float(score)})
                    except Exception as exc:
                        autogluon_enabled_runtime = False
                        print({"autogluon_status": "skipped_after_error", "error": repr(exc)})
                else:
                    print({"autogluon_status": "disabled_by_config"})

                if autogluon_scores:
                    pd.DataFrame(autogluon_scores).sort_values("oof_auc").reset_index(drop=True)
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Blend Search And Final Outputs\n"),
        code_cell(
            dedent(
                """
                candidate_val_frames = {
                    "gandalf_mean": pd.concat([val_df[["customer_id"]].reset_index(drop=True), gandalf_val_mean.reset_index(drop=True)], axis=1),
                    "lgbm": lgbm_val_pred,
                }
                if set(lgbm_val_pred.columns) == set(candidate_val_frames["gandalf_mean"].columns):
                    candidate_val_frames["gandalf_90_lgbm_10"] = pd.concat(
                        [
                            val_df[["customer_id"]].reset_index(drop=True),
                            (0.9 * gandalf_val_mean.reset_index(drop=True) + 0.1 * lgbm_val_pred.drop(columns=["customer_id"]).reset_index(drop=True)),
                        ],
                        axis=1,
                    )

                candidate_test_frames = {}
                if gandalf_test_mean is not None:
                    candidate_test_frames["gandalf_mean"] = normalize_submission_frame(
                        pd.concat([test_df[["customer_id"]].reset_index(drop=True), gandalf_test_mean.reset_index(drop=True)], axis=1),
                        target_cols,
                    )
                if CFG["train_final_models"]:
                    candidate_test_frames["lgbm"] = normalize_submission_frame(lgbm_test_pred, target_cols)
                    if "gandalf_mean" in candidate_test_frames and "lgbm" in candidate_test_frames:
                        blend_frame = candidate_test_frames["gandalf_mean"].copy()
                        pred_cols = [c for c in blend_frame.columns if c != "customer_id"]
                        blend_frame[pred_cols] = 0.9 * candidate_test_frames["gandalf_mean"][pred_cols] + 0.1 * candidate_test_frames["lgbm"][pred_cols]
                        candidate_test_frames["gandalf_90_lgbm_10"] = blend_frame

                if autogluon_enabled_runtime and autogluon_scores and "gandalf_mean" in candidate_val_frames:
                    aghard_val = candidate_val_frames["gandalf_mean"].copy()
                    for pred_col in [c for c in autogluon_val_pred.columns if c != "customer_id"]:
                        aghard_val[pred_col] = autogluon_val_pred[pred_col].values
                    candidate_val_frames["gandalf_aghard_override"] = aghard_val

                    if "gandalf_90_lgbm_10" in candidate_val_frames:
                        gblend_ag_val = candidate_val_frames["gandalf_90_lgbm_10"].copy()
                        for pred_col in [c for c in autogluon_val_pred.columns if c != "customer_id"]:
                            gblend_ag_val[pred_col] = autogluon_val_pred[pred_col].values
                        candidate_val_frames["gandalf_90_lgbm_10_aghard_override"] = gblend_ag_val

                    if CFG["train_final_models"] and "gandalf_mean" in candidate_test_frames:
                        aghard_test = candidate_test_frames["gandalf_mean"].copy()
                        for pred_col in [c for c in autogluon_test_pred.columns if c != "customer_id"]:
                            aghard_test[pred_col] = autogluon_test_pred[pred_col].astype("float64").values
                        candidate_test_frames["gandalf_aghard_override"] = aghard_test

                        if "gandalf_90_lgbm_10" in candidate_test_frames:
                            gblend_ag_test = candidate_test_frames["gandalf_90_lgbm_10"].copy()
                            for pred_col in [c for c in autogluon_test_pred.columns if c != "customer_id"]:
                                gblend_ag_test[pred_col] = autogluon_test_pred[pred_col].astype("float64").values
                            candidate_test_frames["gandalf_90_lgbm_10_aghard_override"] = gblend_ag_test

                candidate_metric_rows = []
                for name, pred_df in candidate_val_frames.items():
                    candidate_metric_rows.append({"candidate": name, "macro_auc": macro_auc(val_df[target_cols], pred_df, target_cols)})
                candidate_metrics = pd.DataFrame(candidate_metric_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
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
                pred_cols = [c.replace("target_", "predict_") for c in target_cols]
                for target_name in target_cols:
                    pred_col = target_name.replace("target_", "predict_")
                    best_name = None
                    best_score = -1.0
                    for name, pred_df in candidate_val_frames.items():
                        score = roc_auc_score(val_df[target_name], pred_df[pred_col]) if val_df[target_name].nunique() > 1 else 0.5
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
                candidate_metrics.to_csv(ARTIFACT_DIR / "candidate_metrics.csv", index=False)
                target_choices.to_csv(ARTIFACT_DIR / "target_choices.csv", index=False)
                lgbm_target_scores.to_csv(ARTIFACT_DIR / "lgbm_target_scores.csv", index=False)
                pd.DataFrame(gandalf_seed_scores).to_csv(ARTIFACT_DIR / "gandalf_seed_scores.csv", index=False)
                compute_target_scores(val_df[target_cols], candidate_val_frames["gandalf_mean"], target_cols).to_csv(
                    ARTIFACT_DIR / "gandalf_target_scores.csv", index=False
                )
                if autogluon_scores:
                    pd.DataFrame(autogluon_scores).sort_values("oof_auc").to_csv(
                        ARTIFACT_DIR / "autogluon_hard_target_scores.csv", index=False
                    )

                summary = {
                    "mode": RESOLVED_MODE,
                    "best_global_candidate": str(candidate_metrics.iloc[0]["candidate"]),
                    "best_global_holdout_macro_auc": float(candidate_metrics.iloc[0]["macro_auc"]),
                    "best_targetwise_holdout_macro_auc": float(target_choices["best_auc"].mean()),
                    "gandalf_holdout_macro_auc": float(gandalf_macro),
                    "lgbm_holdout_macro_auc": float(lgbm_macro),
                    "autogluon_enabled_runtime": bool(autogluon_enabled_runtime),
                    "train_final_models": bool(CFG["train_final_models"]),
                }
                (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
                print({"stage": "summary_written", **summary})
                summary
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                if CFG["train_final_models"]:
                    print({"stage": "submission_build_start", "candidate_count": len(candidate_test_frames)})
                    best_global_name = candidate_metrics.iloc[0]["candidate"]
                    best_global_submission = candidate_test_frames[best_global_name].copy()
                    best_targetwise_submission = pd.DataFrame({"customer_id": candidate_test_frames[best_global_name]["customer_id"].astype("int32").values})
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        choice = target_choices.loc[target_choices["target"] == target_name, "choice"].iloc[0]
                        best_targetwise_submission[pred_col] = candidate_test_frames[choice][pred_col].astype("float64").values

                    normalize_submission_frame(lgbm_test_pred, target_cols).to_parquet(ARTIFACT_DIR / "submission_lgbm.parquet", index=False)
                    candidate_test_frames["gandalf_mean"].to_parquet(ARTIFACT_DIR / "submission_gandalf_mean.parquet", index=False)
                    best_global_submission.to_parquet(ARTIFACT_DIR / "submission_best_global.parquet", index=False)
                    best_targetwise_submission.to_parquet(ARTIFACT_DIR / "submission_best_targetwise.parquet", index=False)
                    print({"stage": "submission_build_done", "submission_best_global": str(ARTIFACT_DIR / "submission_best_global.parquet")})
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

                The notebook writes:
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
