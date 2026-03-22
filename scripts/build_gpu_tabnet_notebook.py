from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root

ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook/data-fusion-2026-gpu-tabnet-multitask.ipynb"


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
                # Experiment: Data Fusion 2026 GPU TabNet MultiTask

                Objective:
                - Train a GPU-ready multi-task deep tabular model for the full 41-target task.
                - Keep the notebook self-sufficient and locally smoke-tested before moving to Kaggle.

                Why this notebook exists:
                - Group-wise classifier chains underperformed badly on CPU and are not worth scaling.
                - `pytorch-tabular` multi-target classification looked attractive on paper, but failed in this environment on real smoke runs.
                - `TabNetMultiTaskClassifier` passed a real local smoke run once the smoke sampling logic guaranteed positive coverage for rare targets.
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
                    ("pytorch_tabnet", "pytorch-tabnet"),
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
                - Smoke mode is intentionally conservative: it proves the pipeline works end-to-end, it is not a quality benchmark.
                - Full mode trains on all labeled rows and writes a submission parquet.
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
                    from pytorch_tabnet.multitask import TabNetMultiTaskClassifier

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
                            "batch_size": 1024,
                            "virtual_batch_size": 128,
                            "n_d": 16,
                            "n_a": 16,
                            "n_steps": 4,
                            "gamma": 1.4,
                            "lambda_sparse": 1e-5,
                            "cat_emb_dim": 4,
                            "lr": 2e-2,
                            "train_final_model": False,
                        },
                        "full_gpu": {
                            "sample_rows": None,
                            "smoke_min_positive": 0,
                            "extra_top_k": 100,
                            "val_fraction": 0.10,
                            "max_epochs": 25,
                            "final_max_epochs": 35,
                            "batch_size": 4096,
                            "virtual_batch_size": 512,
                            "n_d": 32,
                            "n_a": 32,
                            "n_steps": 5,
                            "gamma": 1.5,
                            "lambda_sparse": 1e-5,
                            "cat_emb_dim": 8,
                            "lr": 1e-2,
                            "train_final_model": True,
                        },
                    }[RESOLVED_MODE]

                    ARTIFACT_DIR = Path("artifacts") / f"gpu_tabnet_{RESOLVED_MODE}"
                    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                    print({
                        "resolved_mode": RESOLVED_MODE,
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
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

                These helpers do the minimum needed for a stable end-to-end run:
                - robust data discovery
                - smoke sampling that preserves positives for rare targets
                - train/validation split repair so every target keeps positives in train
                - external macro ROC-AUC computation that tolerates one-class validation slices
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


                def build_stratify_labels(target_df: pd.DataFrame) -> pd.Series:
                    label_count = target_df.sum(axis=1).clip(upper=4).astype("int8")
                    return label_count.astype(str)


                def ensure_smoke_sample(
                    full_df: pd.DataFrame,
                    target_cols: list[str],
                    sample_rows: int,
                    min_positive: int,
                    seed: int,
                ) -> pd.DataFrame:
                    if sample_rows >= len(full_df):
                        return full_df.copy().reset_index(drop=True)

                    strat_labels = build_stratify_labels(full_df[target_cols])
                    splitter = StratifiedShuffleSplit(n_splits=1, train_size=sample_rows, random_state=seed)
                    base_idx, _ = next(splitter.split(full_df, strat_labels))
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
                        take = min(need, len(extra_idx))
                        chosen = full_df.loc[extra_idx].sample(take, random_state=seed).index.tolist()
                        selected.update(chosen)

                    return full_df.loc[sorted(selected)].reset_index(drop=True)


                def split_with_train_positive_coverage(
                    df: pd.DataFrame,
                    target_cols: list[str],
                    val_fraction: float,
                    seed: int,
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
                    strat_labels = build_stratify_labels(df[target_cols])
                    val_rows = max(1, int(round(len(df) * val_fraction)))
                    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_rows, random_state=seed)
                    train_idx, val_idx = next(splitter.split(df, strat_labels))
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


                def fit_apply_preprocessor(
                    train_features: pd.DataFrame,
                    other_frames: list[pd.DataFrame],
                    feature_cols: list[str],
                    cat_cols: list[str],
                    cont_cols: list[str],
                ) -> tuple[pd.DataFrame, list[pd.DataFrame], dict[str, object]]:
                    train_encoded = train_features.copy()
                    transformed = [frame.copy() for frame in other_frames]
                    medians: dict[str, float] = {}
                    cat_idxs: list[int] = []
                    cat_dims: list[int] = []

                    for col in cont_cols:
                        train_encoded[col] = pd.to_numeric(train_encoded[col], errors="coerce").astype("float32")
                        median = float(train_encoded[col].median())
                        medians[col] = median
                        train_encoded[col] = train_encoded[col].fillna(median)
                        for frame in transformed:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").astype("float32").fillna(median)

                    for col in cat_cols:
                        train_vals = train_encoded[col].fillna("__NA__").astype(str)
                        categories = pd.Index(pd.unique(train_vals))
                        mapping = {value: idx for idx, value in enumerate(categories)}
                        unknown_idx = len(mapping)
                        train_encoded[col] = train_vals.map(mapping).astype("int64")
                        for frame in transformed:
                            vals = frame[col].fillna("__NA__").astype(str)
                            frame[col] = vals.map(mapping).fillna(unknown_idx).astype("int64")
                        cat_idxs.append(feature_cols.index(col))
                        cat_dims.append(unknown_idx + 1)

                    meta = {
                        "cat_idxs": cat_idxs,
                        "cat_dims": cat_dims,
                        "medians": medians,
                    }
                    return train_encoded, transformed, meta


                def arrays_from_frames(
                    feature_df: pd.DataFrame,
                    target_df: pd.DataFrame,
                    feature_cols: list[str],
                    target_cols: list[str],
                ) -> tuple[np.ndarray, np.ndarray]:
                    X = feature_df[feature_cols].to_numpy(dtype=np.float32)
                    y = target_df[target_cols].to_numpy(dtype=np.int64)
                    return X, y


                def prob_list_to_frame(prob_list: list[np.ndarray], target_cols: list[str]) -> pd.DataFrame:
                    pred_map: dict[str, np.ndarray] = {}
                    for idx, target_name in enumerate(target_cols):
                        arr = prob_list[idx]
                        if arr.ndim != 2:
                            raise ValueError(f"Unexpected predict_proba ndim for {target_name}: {arr.ndim}")
                        if arr.shape[1] == 1:
                            pred_map[target_name] = np.zeros(arr.shape[0], dtype=np.float32)
                        else:
                            pred_map[target_name] = arr[:, 1].astype(np.float32)
                    return pd.DataFrame(pred_map)


                def compute_target_scores(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for target_name in target_cols:
                        if y_true[target_name].nunique() < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y_true[target_name], pred_df[target_name])
                        rows.append({"target": target_name, "oof_auc": float(score)})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def compute_macro_auc(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    return float(compute_target_scores(y_true, pred_df, target_cols)["oof_auc"].mean())
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Load Data

                The notebook embeds the best known global `top100 extra` feature list from local CPU research.
                Smoke mode only uses the first `top_k` slice of that list to stay fast.
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
                train_extra = pd.read_parquet(
                    DATA_DIR / "train_extra_features.parquet",
                    columns=["customer_id"] + EXTRA_FEATURES,
                )
                test_main = pd.read_parquet(DATA_DIR / "test_main_features.parquet")
                test_extra = pd.read_parquet(
                    DATA_DIR / "test_extra_features.parquet",
                    columns=["customer_id"] + EXTRA_FEATURES,
                )

                labeled_df = train_main.merge(train_extra, on="customer_id", how="inner").merge(
                    train_target, on="customer_id", how="inner"
                )
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
                    "categorical_count": len(cat_cols),
                    "continuous_count": len(cont_cols),
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

                train_features_raw = train_df[feature_cols].copy()
                val_features_raw = val_df[feature_cols].copy()
                test_features_raw = test_df[feature_cols].copy()

                train_features_enc, transformed_frames, preprocess_meta = fit_apply_preprocessor(
                    train_features=train_features_raw,
                    other_frames=[val_features_raw, test_features_raw],
                    feature_cols=feature_cols,
                    cat_cols=cat_cols,
                    cont_cols=cont_cols,
                )
                val_features_enc, test_features_enc = transformed_frames

                X_train, y_train = arrays_from_frames(train_features_enc, train_df, feature_cols, target_cols)
                X_val, y_val = arrays_from_frames(val_features_enc, val_df, feature_cols, target_cols)
                X_test = test_features_enc[feature_cols].to_numpy(dtype=np.float32)

                train_pos = train_df[target_cols].sum(axis=0)
                print({
                    "train_shape": X_train.shape,
                    "val_shape": X_val.shape,
                    "test_shape": X_test.shape,
                    "min_train_positive": int(train_pos.min()),
                    "targets_with_zero_train_positive": int((train_pos == 0).sum()),
                    "cat_dims_count": len(preprocess_meta["cat_dims"]),
                })
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Validation Training

                Internal TabNet validation metrics are intentionally disabled here.
                We already observed that built-in multitask metrics can fail on one-class slices for rare targets,
                while the external `sklearn` macro ROC-AUC remains stable and explicit.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                tabnet_params = dict(
                    n_d=CFG["n_d"],
                    n_a=CFG["n_a"],
                    n_steps=CFG["n_steps"],
                    gamma=CFG["gamma"],
                    lambda_sparse=CFG["lambda_sparse"],
                    cat_idxs=preprocess_meta["cat_idxs"],
                    cat_dims=preprocess_meta["cat_dims"],
                    cat_emb_dim=CFG["cat_emb_dim"],
                    optimizer_params={"lr": CFG["lr"]},
                    seed=SEED,
                    verbose=1,
                    device_name="auto",
                )

                val_model = TabNetMultiTaskClassifier(**tabnet_params)
                val_model.fit(
                    X_train=X_train,
                    y_train=y_train,
                    max_epochs=CFG["max_epochs"],
                    patience=CFG["max_epochs"],
                    batch_size=CFG["batch_size"],
                    virtual_batch_size=CFG["virtual_batch_size"],
                    num_workers=0,
                    drop_last=False,
                    pin_memory=False,
                )

                val_pred_df = prob_list_to_frame(val_model.predict_proba(X_val), target_cols)
                target_scores_df = compute_target_scores(val_df[target_cols], val_pred_df, target_cols)
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
                    "tabnet_params": tabnet_params,
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

                In full GPU mode the notebook retrains the same TabNet configuration on all labeled rows,
                applies the train-fitted preprocessing to test features, and writes a submission parquet.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                submission_path = None
                if CFG["train_final_model"]:
                    full_features_raw = labeled_df[feature_cols].copy()
                    full_features_enc, [test_features_final], preprocess_meta_full = fit_apply_preprocessor(
                        train_features=full_features_raw,
                        other_frames=[test_df[feature_cols].copy()],
                        feature_cols=feature_cols,
                        cat_cols=cat_cols,
                        cont_cols=cont_cols,
                    )
                    X_full = full_features_enc[feature_cols].to_numpy(dtype=np.float32)
                    y_full = labeled_df[target_cols].to_numpy(dtype=np.int64)
                    X_test_full = test_features_final[feature_cols].to_numpy(dtype=np.float32)

                    final_params = {
                        **tabnet_params,
                        "cat_idxs": preprocess_meta_full["cat_idxs"],
                        "cat_dims": preprocess_meta_full["cat_dims"],
                    }
                    final_model = TabNetMultiTaskClassifier(**final_params)
                    final_model.fit(
                        X_train=X_full,
                        y_train=y_full,
                        max_epochs=CFG["final_max_epochs"],
                        patience=CFG["final_max_epochs"],
                        batch_size=CFG["batch_size"],
                        virtual_batch_size=CFG["virtual_batch_size"],
                        num_workers=0,
                        drop_last=False,
                        pin_memory=False,
                    )

                    test_pred_df = prob_list_to_frame(final_model.predict_proba(X_test_full), target_cols)
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

                - If smoke mode passed, switch `RUN_MODE` to `full_gpu` on Kaggle and run the notebook unchanged.
                - First compare this TabNet branch to the current tree-based stack on the hardest targets, not only on macro AUC.
                - If TabNet still underperforms, the next GPU branch should be a custom structured model rather than another plain tabular baseline.
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
