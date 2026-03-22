from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root

ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook/data-fusion-2026-gpu-gandalf-multitask.ipynb"


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
    top_features_literal = json.dumps(top_features, ensure_ascii=True, indent=4)

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU GANDALF MultiTask

                Objective:
                - Train a GPU-ready GANDALF multi-target classifier for all 41 binary targets.
                - Avoid the broken default multitask metric path in `pytorch-tabular`.
                - Keep the notebook reproducible and locally smoke-tested before Kaggle GPU.

                Design choices:
                - Use `pytorch-tabular` only for training and prediction.
                - Disable built-in multitask metrics in `fit(...)`, because they crash on one-class rare-target slices.
                - Compute validation macro ROC-AUC externally with `sklearn`.
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
                    ("pytorch_tabular", "pytorch-tabular"),
                    ("requests", "requests"),
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
        md_cell(
            dedent(
                """
                ## Run Mode

                - `RUN_MODE="auto"` keeps local execution in smoke mode and Kaggle GPU execution in full mode.
                - Smoke mode proves correctness end-to-end; it is not the final quality benchmark.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            (
                dedent(
                    """
                    from __future__ import annotations

                    import json
                    import random
                    from pathlib import Path

                    import numpy as np
                    import pandas as pd
                    import torch
                    from sklearn.metrics import roc_auc_score
                    from sklearn.model_selection import StratifiedShuffleSplit
                    from pytorch_tabular import TabularModel
                    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
                    from pytorch_tabular.models import GANDALFConfig

                    RUN_MODE = "auto"  # one of: auto, smoke, full_gpu
                    SEED = 42
                    random.seed(SEED)
                    np.random.seed(SEED)

                    TOP_EXTRA_FEATURES = """
                ).strip()
                + " "
                + top_features_literal
                + "\n\n"
                + dedent(
                    """
                    def resolve_run_mode(run_mode: str) -> str:
                        if run_mode != "auto":
                            return run_mode
                        on_kaggle = Path("/kaggle/input").exists()
                        return "full_gpu" if on_kaggle and torch.cuda.is_available() else "smoke"

                    RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                    CFG = {
                        "smoke": {
                            "sample_rows": 20_000,
                            "smoke_min_positive": 20,
                            "extra_top_k": 30,
                            "val_fraction": 0.20,
                            "max_epochs": 2,
                            "final_max_epochs": 2,
                            "batch_size": 512,
                            "learning_rate": 1e-3,
                            "gflu_stages": 4,
                            "gflu_dropout": 0.1,
                            "train_final_model": False,
                        },
                        "full_gpu": {
                            "sample_rows": None,
                            "smoke_min_positive": 0,
                            "extra_top_k": 100,
                            "val_fraction": 0.10,
                            "max_epochs": 20,
                            "final_max_epochs": 28,
                            "batch_size": 2048,
                            "learning_rate": 8e-4,
                            "gflu_stages": 6,
                            "gflu_dropout": 0.15,
                            "train_final_model": True,
                        },
                    }[RESOLVED_MODE]

                    ARTIFACT_DIR = Path("artifacts") / f"gpu_gandalf_{RESOLVED_MODE}"
                    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                    print({
                        "resolved_mode": RESOLVED_MODE,
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "config": CFG,
                        "artifact_dir": str(ARTIFACT_DIR),
                    })
                    """
                ).strip()
                + "\n"
            )
        ),
        md_cell(
            dedent(
                """
                ## Helpers

                These helpers keep the notebook robust:
                - parquet auto-discovery
                - rare-target-preserving smoke sampling
                - split repair so each target has positives in train
                - external macro ROC-AUC and target-level scores
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                def find_data_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "data" / "competition",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026"),
                        Path("/kaggle/input/data-fusion-2026-cybershelf"),
                    ]
                    required = [
                        "train_main_features.parquet",
                        "train_target.parquet",
                        "train_extra_features.parquet",
                        "test_main_features.parquet",
                        "test_extra_features.parquet",
                    ]
                    for candidate in direct_candidates:
                        if all((candidate / name).exists() for name in required):
                            return candidate
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for match in kaggle_input.rglob("train_main_features.parquet"):
                            candidate = match.parent
                            if all((candidate / name).exists() for name in required):
                                return candidate
                    raise FileNotFoundError("Could not locate the parquet dataset directory.")


                def trainer_accelerator_for_mode(mode: str) -> str:
                    if mode == "smoke":
                        return "cpu"
                    return "gpu" if torch.cuda.is_available() else "auto"


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


                def preprocess_frames(
                    train_df: pd.DataFrame,
                    other_frames: list[pd.DataFrame],
                    cat_cols: list[str],
                    cont_cols: list[str],
                ) -> tuple[pd.DataFrame, list[pd.DataFrame]]:
                    train_out = train_df.copy()
                    others = [frame.copy() for frame in other_frames]

                    for col in cat_cols:
                        train_out[col] = train_out[col].fillna(-1).astype("int64").astype(str)
                        for frame in others:
                            frame[col] = frame[col].fillna(-1).astype("int64").astype(str)

                    for col in cont_cols:
                        train_out[col] = pd.to_numeric(train_out[col], errors="coerce").astype("float32")
                        median = float(train_out[col].median())
                        train_out[col] = train_out[col].fillna(median)
                        for frame in others:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32").fillna(median)

                    return train_out, others


                def target_scores(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for target_name in target_cols:
                        if y_true[target_name].nunique() < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y_true[target_name], pred_df[target_name])
                        rows.append({"target": target_name, "oof_auc": float(score)})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def extract_probability_frame(raw_pred: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    pred_map: dict[str, np.ndarray] = {}
                    for target_name in target_cols:
                        positive_col = f"{target_name}_1_probability"
                        if positive_col in raw_pred.columns:
                            pred_map[target_name] = raw_pred[positive_col].astype("float32").to_numpy()
                            continue

                        fallback_cols = [c for c in raw_pred.columns if c.startswith(f"{target_name}_") and c.endswith("_probability")]
                        if fallback_cols:
                            def class_key(col_name: str) -> tuple[int, float, str]:
                                class_label = col_name[len(target_name) + 1 : -len("_probability")]
                                try:
                                    numeric = float(class_label)
                                    return (1, numeric, class_label)
                                except ValueError:
                                    return (0, 0.0, class_label)

                            chosen_col = sorted(fallback_cols, key=class_key)[-1]
                            pred_map[target_name] = raw_pred[chosen_col].astype("float32").to_numpy()
                        else:
                            pred_map[target_name] = np.zeros(len(raw_pred), dtype=np.float32)

                    return pd.DataFrame(pred_map)
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Load Data

                Uses the locally discovered parquet files or Kaggle input dataset.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                DATA_DIR = find_data_dir()
                EXTRA_FEATURES = TOP_EXTRA_FEATURES[: CFG["extra_top_k"]]

                train_main = pd.read_parquet(DATA_DIR / "train_main_features.parquet")
                train_target = pd.read_parquet(DATA_DIR / "train_target.parquet")
                train_extra = pd.read_parquet(DATA_DIR / "train_extra_features.parquet", columns=["customer_id"] + EXTRA_FEATURES)
                test_main = pd.read_parquet(DATA_DIR / "test_main_features.parquet")
                test_extra = pd.read_parquet(DATA_DIR / "test_extra_features.parquet", columns=["customer_id"] + EXTRA_FEATURES)

                labeled_df = train_main.merge(train_extra, on="customer_id", how="inner").merge(train_target, on="customer_id", how="inner")
                test_df = test_main.merge(test_extra, on="customer_id", how="inner")

                target_cols = [c for c in labeled_df.columns if c.startswith("target_")]
                feature_cols = [c for c in labeled_df.columns if c not in ["customer_id"] + target_cols]
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
                cont_cols = [c for c in feature_cols if c not in cat_cols]

                if RESOLVED_MODE == "smoke":
                    labeled_df = ensure_smoke_sample(
                        full_df=labeled_df,
                        target_cols=target_cols,
                        sample_rows=CFG["sample_rows"],
                        min_positive=CFG["smoke_min_positive"],
                        seed=SEED,
                    )

                print({
                    "data_dir": str(DATA_DIR),
                    "labeled_shape": labeled_df.shape,
                    "test_shape": test_df.shape,
                    "feature_count": len(feature_cols),
                    "target_count": len(target_cols),
                    "extra_top_k": len(EXTRA_FEATURES),
                })
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                train_df, val_df = split_with_train_positive_coverage(
                    df=labeled_df,
                    target_cols=target_cols,
                    val_fraction=CFG["val_fraction"],
                    seed=SEED,
                )

                train_features, [val_features, test_features] = preprocess_frames(
                    train_df[feature_cols].reset_index(drop=True),
                    [val_df[feature_cols].reset_index(drop=True), test_df[feature_cols].reset_index(drop=True)],
                    cat_cols=cat_cols,
                    cont_cols=cont_cols,
                )

                train_ready = pd.concat([train_features.reset_index(drop=True), train_df[target_cols].reset_index(drop=True)], axis=1)
                val_ready = pd.concat([val_features.reset_index(drop=True), val_df[target_cols].reset_index(drop=True)], axis=1)

                train_pos = train_df[target_cols].sum(axis=0)
                print({
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "min_train_positive": int(train_pos.min()),
                    "targets_with_zero_train_positive": int((train_pos == 0).sum()),
                })
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Validation Training

                Critical implementation detail:
                - `fit(..., metrics=[], metrics_prob_inputs=[])` disables the broken default multitask metric path.
                - Validation macro ROC-AUC is computed outside the library from `*_1_probability` columns.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                trainer_config = TrainerConfig(
                    batch_size=CFG["batch_size"],
                    max_epochs=CFG["max_epochs"],
                    accelerator=trainer_accelerator_for_mode(RESOLVED_MODE),
                    devices=1,
                    progress_bar="none",
                    checkpoints="valid_loss",
                    checkpoints_path=str(ARTIFACT_DIR / "checkpoints"),
                    load_best=True,
                    seed=SEED,
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
                    learning_rate=CFG["learning_rate"],
                    metrics=["accuracy"],
                    metrics_prob_input=[False],
                    gflu_stages=CFG["gflu_stages"],
                    gflu_dropout=CFG["gflu_dropout"],
                )

                val_model = TabularModel(
                    data_config=data_config,
                    model_config=model_config,
                    optimizer_config=optimizer_config,
                    trainer_config=trainer_config,
                    verbose=False,
                    suppress_lightning_logger=True,
                )
                val_model.fit(
                    train=train_ready,
                    validation=val_ready,
                    metrics=[],
                    metrics_prob_inputs=[],
                )

                raw_val_pred = val_model.predict(val_ready, include_input_features=False, progress_bar="none")
                val_pred_df = extract_probability_frame(raw_val_pred, target_cols)
                target_scores_df = target_scores(val_df[target_cols], val_pred_df, target_cols)
                val_macro_auc = float(target_scores_df["oof_auc"].mean())

                print({"validation_macro_auc": val_macro_auc})
                target_scores_df.head(12)
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                validation_predictions = val_pred_df.copy()
                validation_predictions.insert(0, "customer_id", val_df["customer_id"].values)
                validation_predictions.to_parquet(ARTIFACT_DIR / "validation_predictions.parquet", index=False)
                target_scores_df.to_csv(ARTIFACT_DIR / "target_scores.csv", index=False)

                metrics_payload = {
                    "mode": RESOLVED_MODE,
                    "validation_macro_auc": float(val_macro_auc),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "feature_count": int(len(feature_cols)),
                    "extra_top_k": int(len(EXTRA_FEATURES)),
                    "config": CFG,
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
                metrics_payload
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Optional Final Full-Train Submission

                In `full_gpu` mode the notebook retrains on all labeled data and writes `submission.parquet`.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                submission_path = None
                if CFG["train_final_model"]:
                    full_features, [test_features_final] = preprocess_frames(
                        labeled_df[feature_cols].reset_index(drop=True),
                        [test_df[feature_cols].reset_index(drop=True)],
                        cat_cols=cat_cols,
                        cont_cols=cont_cols,
                    )
                    full_ready = pd.concat([full_features.reset_index(drop=True), labeled_df[target_cols].reset_index(drop=True)], axis=1)

                    final_trainer_config = TrainerConfig(
                        batch_size=CFG["batch_size"],
                        max_epochs=CFG["final_max_epochs"],
                        accelerator=trainer_accelerator_for_mode(RESOLVED_MODE),
                        devices=1,
                        progress_bar="none",
                        checkpoints="valid_loss",
                        checkpoints_path=str(ARTIFACT_DIR / "final_checkpoints"),
                        load_best=True,
                        seed=SEED,
                    )
                    final_model = TabularModel(
                        data_config=data_config,
                        model_config=model_config,
                        optimizer_config=optimizer_config,
                        trainer_config=final_trainer_config,
                        verbose=False,
                        suppress_lightning_logger=True,
                    )
                    final_model.fit(
                        train=full_ready,
                        validation=None,
                        metrics=[],
                        metrics_prob_inputs=[],
                    )

                    raw_test_pred = final_model.predict(test_features_final, include_input_features=False, progress_bar="none")
                    test_pred_df = extract_probability_frame(raw_test_pred, target_cols)
                    submission = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    for target_name in target_cols:
                        submission[target_name.replace("target_", "predict_")] = test_pred_df[target_name].astype("float64").values

                    submission_path = ARTIFACT_DIR / "submission.parquet"
                    submission.to_parquet(submission_path, index=False)
                    print({"submission_path": str(submission_path), "rows": len(submission)})
                else:
                    print("Smoke mode: final full-train submission step skipped.")
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Next Steps

                - Compare this GANDALF branch to TabNet and the best tree stack on hardest targets first.
                - If GANDALF beats TabNet but not the stack, use it as a diverse base model for future ensembling.
                """
            ).strip()
            + "\n"
        ),
    ]

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=True, indent=2) + "\n")
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
