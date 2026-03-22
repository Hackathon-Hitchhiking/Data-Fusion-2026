from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root

ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook/data-fusion-2026-gpu-tabm-multitask.ipynb"


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
                # Experiment: Data Fusion 2026 GPU TabM MultiTask

                Objective:
                - Evaluate `TabM` as the next backbone candidate after `GANDALF`.
                - Keep the notebook Kaggle-ready, stage-logged, and smoke-testable locally.

                Design:
                - Multi-label training over all `41` targets with `BCEWithLogitsLoss`.
                - Correct `TabM` training: optimize the mean loss over all ensemble members, not the loss of the averaged prediction.
                - Average probabilities, not logits, at inference time.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            dedent(
                """
                import importlib.util
                import subprocess
                import sys

                REQUIRED_PACKAGES = [
                    ("torch", "torch"),
                    ("tabm", "tabm"),
                    ("rtdl_num_embeddings", "rtdl-num-embeddings"),
                    ("pyarrow", "pyarrow"),
                    ("sklearn", "scikit-learn"),
                ]

                missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]
                print({"stage": "bootstrap_start", "missing": missing}, flush=True)
                if missing:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
                print({"stage": "bootstrap_done", "installed_now": missing}, flush=True)
                """
            ).strip()
            + "\n"
        ),
        md_cell(
            dedent(
                """
                ## Run Mode

                - `RUN_MODE="auto"` means:
                  - Kaggle GPU -> `full_gpu`
                  - local machine -> `smoke`
                - Smoke mode validates the full code path quickly and does not aim for final quality.
                """
            ).strip()
            + "\n"
        ),
        code_cell(
            (
                dedent(
                    """
                    from __future__ import annotations

                    import gc
                    import json
                    import random
                    from pathlib import Path

                    import numpy as np
                    import pandas as pd
                    import torch
                    import torch.nn.functional as F
                    from rtdl_num_embeddings import LinearReLUEmbeddings
                    from sklearn.metrics import roc_auc_score
                    from sklearn.model_selection import StratifiedShuffleSplit
                    from tabm import TabM
                    from torch.utils.data import DataLoader, TensorDataset

                    RUN_MODE = "auto"  # one of: auto, smoke, full_gpu
                    SEED = 42
                    random.seed(SEED)
                    np.random.seed(SEED)
                    torch.manual_seed(SEED)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(SEED)

                    TOP_EXTRA_FEATURES = """
                ).strip()
                + " "
                + top_features_literal
                + "\n\n"
                + dedent(
                    """
                    def resolve_run_mode(mode: str) -> str:
                        if mode != "auto":
                            return mode
                        return "full_gpu" if torch.cuda.is_available() and Path("/kaggle/input").exists() else "smoke"


                    RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                    CFG = {
                        "smoke": {
                            "sample_rows": 120_000,
                            "smoke_min_positive": 12,
                            "extra_top_k": 40,
                            "val_fraction": 0.20,
                            "epochs": 3,
                            "batch_size": 512,
                            "eval_batch_size": 2048,
                            "lr": 2e-3,
                            "weight_decay": 1e-4,
                            "use_num_embeddings": True,
                            "min_epochs": 2,
                            "early_stopping_patience": 2,
                            "train_final_model": False,
                        },
                        "full_gpu": {
                            "sample_rows": None,
                            "smoke_min_positive": 0,
                            "extra_top_k": 100,
                            "val_fraction": 0.10,
                            "epochs": 18,
                            "batch_size": 1024,
                            "eval_batch_size": 4096,
                            "lr": 2e-3,
                            "weight_decay": 1e-4,
                            "use_num_embeddings": True,
                            "min_epochs": 8,
                            "early_stopping_patience": 4,
                            "train_final_model": True,
                        },
                    }[RESOLVED_MODE]

                    ARTIFACT_DIR = Path("artifacts") / f"gpu_tabm_{RESOLVED_MODE}"
                    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")

                    print({
                        "resolved_mode": RESOLVED_MODE,
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "mps_available": torch.backends.mps.is_available(),
                        "config": CFG,
                        "artifact_dir": str(ARTIFACT_DIR),
                    })
                    """
                ).strip()
                + "\n"
            )
        ),
        md_cell("## Helpers\n"),
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


                def pick_device() -> str:
                    if torch.cuda.is_available():
                        return "cuda"
                    if torch.backends.mps.is_available():
                        return "mps"
                    return "cpu"


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


                def prepare_arrays(
                    train_df: pd.DataFrame,
                    other_frames: list[pd.DataFrame],
                    num_cols: list[str],
                    cat_cols: list[str],
                    target_cols: list[str] | None,
                ) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], list[int]]:
                    train_num = train_df[num_cols].copy()
                    other_num = [frame[num_cols].copy() for frame in other_frames]
                    train_cat = train_df[cat_cols].copy()
                    other_cat = [frame[cat_cols].copy() for frame in other_frames]

                    for col in num_cols:
                        train_num[col] = pd.to_numeric(train_num[col], errors="coerce")
                        median = float(train_num[col].median())
                        train_num[col] = train_num[col].fillna(median)
                        mean = float(train_num[col].mean())
                        std = float(train_num[col].std())
                        std = std if std > 1e-6 else 1.0
                        train_num[col] = ((train_num[col] - mean) / std).astype("float32")
                        for frame in other_num:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median)
                            frame[col] = ((frame[col] - mean) / std).astype("float32")

                    cat_cardinalities = []
                    for col in cat_cols:
                        train_series = pd.to_numeric(train_cat[col], errors="coerce").fillna(-1).astype("int64")
                        other_series = [pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int64") for frame in other_cat]
                        known_vals = sorted(v for v in train_series.unique().tolist() if v >= 0)
                        mapping = {v: i for i, v in enumerate(known_vals)}
                        unk = len(mapping)
                        train_cat[col] = train_series.map(mapping).fillna(unk).astype("int16")
                        for idx, series in enumerate(other_series):
                            other_cat[idx][col] = series.map(mapping).fillna(unk).astype("int16")
                        cat_cardinalities.append(int(unk + 1))

                    train_arrays = {
                        "x_num": train_num.to_numpy(dtype=np.float32, copy=True),
                        "x_cat": train_cat.to_numpy(dtype=np.int16, copy=True),
                    }
                    if target_cols is not None:
                        train_arrays["y"] = train_df[target_cols].to_numpy(dtype=np.float32, copy=True)

                    other_arrays = []
                    for idx in range(len(other_frames)):
                        payload = {
                            "x_num": other_num[idx].to_numpy(dtype=np.float32, copy=True),
                            "x_cat": other_cat[idx].to_numpy(dtype=np.int16, copy=True),
                        }
                        if target_cols is not None and all(col in other_frames[idx].columns for col in target_cols):
                            payload["y"] = other_frames[idx][target_cols].to_numpy(dtype=np.float32, copy=True)
                        other_arrays.append(payload)

                    return train_arrays, other_arrays, cat_cardinalities


                def make_loader(arrays: dict[str, np.ndarray], batch_size: int, shuffle: bool) -> DataLoader:
                    y = arrays.get("y")
                    dataset = TensorDataset(
                        torch.from_numpy(arrays["x_num"]),
                        torch.from_numpy(arrays["x_cat"]),
                        torch.from_numpy(y if y is not None else np.zeros((len(arrays["x_num"]), 1), dtype=np.float32)),
                    )
                    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


                @torch.no_grad()
                def predict_proba(model: TabM, loader: DataLoader, device: str) -> np.ndarray:
                    model.eval()
                    outputs = []
                    for x_num, x_cat, _ in loader:
                        x_num = x_num.to(device, non_blocking=device == "cuda")
                        x_cat = x_cat.to(device, non_blocking=device == "cuda").long()
                        logits = model(x_num, x_cat)
                        probs = torch.sigmoid(logits).mean(1)
                        outputs.append(probs.cpu().numpy())
                    return np.concatenate(outputs, axis=0)


                def compute_target_scores(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> pd.DataFrame:
                    rows = []
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        if y_true[target_name].nunique() < 2:
                            score = 0.5
                        else:
                            score = roc_auc_score(y_true[target_name], pred_df[pred_col])
                        rows.append({"target": target_name, "oof_auc": float(score)})
                    return pd.DataFrame(rows).sort_values("oof_auc").reset_index(drop=True)


                def macro_auc(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    scores = []
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        target_values = y_true[target_name]
                        if target_values.nunique() < 2:
                            scores.append(0.5)
                            continue
                        scores.append(float(roc_auc_score(target_values, pred_df[pred_col])))
                    return float(np.mean(scores))


                def train_tabm(
                    train_arrays: dict[str, np.ndarray],
                    val_arrays: dict[str, np.ndarray],
                    target_cols: list[str],
                    cat_cardinalities: list[int],
                    n_num_features: int,
                    cfg: dict[str, object],
                    device: str,
                    val_customer_ids: np.ndarray,
                    val_target_df: pd.DataFrame,
                    artifact_dir: Path,
                ) -> tuple[TabM, pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
                    train_loader = make_loader(train_arrays, int(cfg["batch_size"]), shuffle=True)
                    val_loader = make_loader(val_arrays, int(cfg["eval_batch_size"]), shuffle=False)

                    num_embeddings = LinearReLUEmbeddings(n_num_features) if bool(cfg["use_num_embeddings"]) else None
                    model = TabM.make(
                        n_num_features=n_num_features,
                        cat_cardinalities=cat_cardinalities,
                        d_out=len(target_cols),
                        num_embeddings=num_embeddings,
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(cfg["lr"]),
                        weight_decay=float(cfg["weight_decay"]),
                    )

                    history_rows = []
                    best_val = -np.inf
                    best_epoch = 1
                    best_state = None
                    best_pred_df = None
                    epochs_without_improvement = 0

                    for epoch in range(1, int(cfg["epochs"]) + 1):
                        model.train()
                        losses = []
                        for x_num, x_cat, y in train_loader:
                            x_num = x_num.to(device, non_blocking=device == "cuda")
                            x_cat = x_cat.to(device, non_blocking=device == "cuda").long()
                            y = y.to(device, non_blocking=device == "cuda")

                            optimizer.zero_grad(set_to_none=True)
                            logits = model(x_num, x_cat)
                            y_expanded = y.unsqueeze(1).expand(-1, logits.shape[1], -1)
                            loss = F.binary_cross_entropy_with_logits(logits, y_expanded)
                            loss.backward()
                            optimizer.step()
                            losses.append(float(loss.detach().cpu()))

                        val_probs = predict_proba(model, val_loader, device)
                        val_pred_df = pd.DataFrame(val_probs, columns=[c.replace("target_", "predict_") for c in target_cols])
                        val_pred_df.insert(0, "customer_id", val_customer_ids)
                        val_macro = float(macro_auc(val_target_df, val_pred_df, target_cols))
                        history_rows.append({
                            "epoch": epoch,
                            "train_loss": float(np.mean(losses)),
                            "val_macro_auc": val_macro,
                        })
                        history_df = pd.DataFrame(history_rows)
                        history_df.to_csv(artifact_dir / "history_running.csv", index=False)
                        print({"stage": "tabm_epoch_done", "epoch": epoch, "train_loss": float(np.mean(losses)), "val_macro_auc": val_macro}, flush=True)

                        if val_macro > best_val:
                            best_val = val_macro
                            best_epoch = epoch
                            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                            best_pred_df = val_pred_df.copy()
                            best_pred_df.to_parquet(artifact_dir / "best_validation_predictions_running.parquet", index=False)
                            compute_target_scores(val_target_df, best_pred_df, target_cols).to_csv(
                                artifact_dir / "best_target_scores_running.csv",
                                index=False,
                            )
                            (artifact_dir / "metrics_running.json").write_text(
                                json.dumps(
                                    {
                                        "best_val_macro_auc": float(best_val),
                                        "best_epoch": int(best_epoch),
                                        "resolved_mode": RESOLVED_MODE,
                                        "device": device,
                                        "rows_train": int(len(train_arrays["x_num"])),
                                        "rows_val": int(len(val_arrays["x_num"])),
                                        "feature_count": int(n_num_features + len(cat_cardinalities)),
                                        "num_features": int(n_num_features),
                                        "cat_features": int(len(cat_cardinalities)),
                                        "use_num_embeddings": bool(cfg["use_num_embeddings"]),
                                    },
                                    indent=2,
                                )
                            )
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1

                        if epoch >= int(cfg["min_epochs"]) and epochs_without_improvement >= int(cfg["early_stopping_patience"]):
                            print(
                                {
                                    "stage": "tabm_early_stop",
                                    "epoch": epoch,
                                    "best_epoch": int(best_epoch),
                                    "best_val_macro_auc": float(best_val),
                                    "patience": int(cfg["early_stopping_patience"]),
                                },
                                flush=True,
                            )
                            break

                    if best_state is not None:
                        model.load_state_dict(best_state)

                    if best_pred_df is None:
                        best_probs = predict_proba(model, val_loader, device)
                        best_pred_df = pd.DataFrame(best_probs, columns=[c.replace("target_", "predict_") for c in target_cols])
                        best_pred_df.insert(0, "customer_id", val_customer_ids)
                    history_df = pd.DataFrame(history_rows)

                    return model, best_pred_df, history_df, {"best_val_macro_auc": float(best_val), "best_epoch": int(best_epoch)}
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Load Data\n"),
        code_cell(
            dedent(
                """
                DATA_DIR = find_data_dir()
                EXTRA_FEATURES = TOP_EXTRA_FEATURES[: CFG["extra_top_k"]]
                DEVICE = pick_device()

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
                num_cols = [c for c in feature_cols if c not in cat_cols]

                if RESOLVED_MODE == "smoke":
                    labeled_df = ensure_smoke_sample(
                        full_df=labeled_df,
                        target_cols=target_cols,
                        sample_rows=int(CFG["sample_rows"]),
                        min_positive=int(CFG["smoke_min_positive"]),
                        seed=SEED,
                    )

                print({
                    "data_dir": str(DATA_DIR),
                    "labeled_rows": len(labeled_df),
                    "test_rows": len(test_df),
                    "feature_count": len(feature_cols),
                    "num_features": len(num_cols),
                    "cat_features": len(cat_cols),
                    "device": DEVICE,
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
                    val_fraction=float(CFG["val_fraction"]),
                    seed=SEED,
                )

                train_arrays, [val_arrays], cat_cardinalities = prepare_arrays(
                    train_df=train_df[feature_cols + target_cols],
                    other_frames=[val_df[feature_cols + target_cols]],
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    target_cols=target_cols,
                )

                print({
                    "stage": "tabm_val_prepare_done",
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "min_train_positive": int(train_df[target_cols].sum().min()),
                    "max_cat_cardinality": int(max(cat_cardinalities)),
                })
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Validation Training\n"),
        code_cell(
            dedent(
                """
                model, val_pred_df, history_df, fit_stats = train_tabm(
                    train_arrays=train_arrays,
                    val_arrays=val_arrays,
                    target_cols=target_cols,
                    cat_cardinalities=cat_cardinalities,
                    n_num_features=len(num_cols),
                    cfg=CFG,
                    device=DEVICE,
                    val_customer_ids=val_df["customer_id"].values,
                    val_target_df=val_df[["customer_id"] + target_cols].copy(),
                    artifact_dir=ARTIFACT_DIR,
                )

                history_df.to_csv(ARTIFACT_DIR / "history.csv", index=False)
                val_pred_df.to_parquet(ARTIFACT_DIR / "validation_predictions.parquet", index=False)
                target_scores_df = compute_target_scores(val_df[["customer_id"] + target_cols], val_pred_df, target_cols)
                target_scores_df.to_csv(ARTIFACT_DIR / "target_scores.csv", index=False)

                metrics_payload = {
                    "validation_macro_auc": float(fit_stats["best_val_macro_auc"]),
                    "best_epoch": int(fit_stats["best_epoch"]),
                    "resolved_mode": RESOLVED_MODE,
                    "device": DEVICE,
                    "rows_train": int(len(train_df)),
                    "rows_val": int(len(val_df)),
                    "feature_count": int(len(feature_cols)),
                    "num_features": int(len(num_cols)),
                    "cat_features": int(len(cat_cols)),
                    "use_num_embeddings": bool(CFG["use_num_embeddings"]),
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
                print(json.dumps(metrics_payload, indent=2), flush=True)
                print(target_scores_df.head(12).to_string(index=False), flush=True)
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Final Train And Submission\n"),
        code_cell(
            dedent(
                """
                if bool(CFG["train_final_model"]):
                    full_arrays, [test_arrays], full_cat_cardinalities = prepare_arrays(
                        train_df=labeled_df[feature_cols + target_cols],
                        other_frames=[test_df[feature_cols]],
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                        target_cols=target_cols,
                    )

                    full_loader = make_loader(full_arrays, int(CFG["batch_size"]), shuffle=True)
                    test_loader = make_loader(test_arrays, int(CFG["eval_batch_size"]), shuffle=False)

                    final_num_embeddings = LinearReLUEmbeddings(len(num_cols)) if bool(CFG["use_num_embeddings"]) else None
                    final_model = TabM.make(
                        n_num_features=len(num_cols),
                        cat_cardinalities=full_cat_cardinalities,
                        d_out=len(target_cols),
                        num_embeddings=final_num_embeddings,
                    ).to(DEVICE)
                    final_optimizer = torch.optim.AdamW(
                        final_model.parameters(),
                        lr=float(CFG["lr"]),
                        weight_decay=float(CFG["weight_decay"]),
                    )

                    print({"stage": "tabm_final_train_start", "epochs": int(fit_stats["best_epoch"]), "rows": len(labeled_df), "test_rows": len(test_df)}, flush=True)
                    for epoch in range(1, int(fit_stats["best_epoch"]) + 1):
                        final_model.train()
                        losses = []
                        for x_num, x_cat, y in full_loader:
                            x_num = x_num.to(DEVICE, non_blocking=DEVICE == "cuda")
                            x_cat = x_cat.to(DEVICE, non_blocking=DEVICE == "cuda").long()
                            y = y.to(DEVICE, non_blocking=DEVICE == "cuda")
                            final_optimizer.zero_grad(set_to_none=True)
                            logits = final_model(x_num, x_cat)
                            y_expanded = y.unsqueeze(1).expand(-1, logits.shape[1], -1)
                            loss = F.binary_cross_entropy_with_logits(logits, y_expanded)
                            loss.backward()
                            final_optimizer.step()
                            losses.append(float(loss.detach().cpu()))
                        print({"stage": "tabm_final_epoch_done", "epoch": epoch, "train_loss": float(np.mean(losses))}, flush=True)

                    test_probs = predict_proba(final_model, test_loader, DEVICE)
                    submission = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    for idx, target_name in enumerate(target_cols):
                        submission[target_name.replace("target_", "predict_")] = test_probs[:, idx].astype("float64")

                    submission_path = ARTIFACT_DIR / "submission.parquet"
                    submission.to_parquet(submission_path, index=False)
                    print({"stage": "submission_written", "submission_path": str(submission_path), "rows": len(submission)}, flush=True)
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

                - If `TabM` is still far below `GANDALF`, keep it only as a diversity candidate.
                - If it reaches or beats `GANDALF`, it becomes the next main backbone branch.
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
