from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
NOTEBOOK_NAME = "data-fusion-2026-gpu-realmlp-tdlike-ovr.ipynb"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output/kaggle-push/realmlp-tdlike-ovr"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "realmlp_tdlike_ovr_v3_rev4_20260320"


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
    top_features = json.loads(TOP_FEATURES_FILE.read_text())
    top_features_literal = json.dumps(top_features, ensure_ascii=True, indent=4)
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
        '    ("lightning", "lightning==2.6.1"),',
        '    ("torchmetrics", "torchmetrics==1.8.2"),',
        '    ("pytabkit", "pytabkit==1.7.3"),',
        "]",
        "",
        "def torch_runtime_info() -> dict[str, object] | None:",
        '    if importlib.util.find_spec("torch") is None:',
        "        return None",
        "    import torch",
        "    info = {}",
        '    info["version"] = torch.__version__',
        '    info["cuda_available"] = bool(torch.cuda.is_available())',
        '    info["capability"] = None',
        '    if info["cuda_available"]:',
        "        try:",
        "            major, minor = torch.cuda.get_device_capability(0)",
        '            info["capability"] = [int(major), int(minor)]',
        "        except Exception:",
        '            info["capability"] = None',
        "    return info",
        "",
        "def pip_install(args: list[str]) -> None:",
        '    subprocess.check_call([sys.executable, "-m", "pip", *args])',
        "",
        'missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]',
        "torch_info = torch_runtime_info()",
        'print({"stage": "bootstrap_start", "revision": NOTEBOOK_REVISION, "missing": missing, "torch_info": torch_info}, flush=True)',
        "",
        'if torch_info and torch_info.get("cuda_available") and torch_info.get("capability") and int(torch_info["capability"][0]) < 7:',
        '    raise RuntimeError("Unsupported Kaggle GPU for RealMLP CUDA path: P100/sm_60. Restart until T4+ or run locally on MPS.")',
        "",
        "if missing:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing])',
        "",
        'import_check = "from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier; print(RealMLP_TD_Classifier.__name__)"',
        'subprocess.check_call([sys.executable, "-c", import_check])',
        'print({"stage": "bootstrap_done", "installed_now": missing}, flush=True)',
    ]
    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU RealMLP TD-like OVR

                Objective:
                - Run a strong first `RealMLP` challenger with a different inductive bias than `TabM`.
                - Evaluate it on the shared holdout split and then produce a full-train submission.

                Design:
                - Official `RealMLP_TD_Classifier` interface from `pytabkit`.
                - `one-vs-rest` over all `41` binary targets, because official `RealMLP` does not support multilabel directly.
                - Match the strongest current feature recipe: `main_features + top100 extra features`.
                - Strong `TD`-like defaults, but with runtime-bounded `batch_size` and `n_epochs` so the full notebook can actually finish on Kaggle GPU.
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
                  - Kaggle GPU -> `full_gpu`
                  - local machine -> `smoke`
                - Smoke mode runs the same code path on a small sample and only a few targets.
                """
            ).strip()
            + "\n",
            3,
        ),
        code_cell(
            (
                dedent(
                    """
                    from __future__ import annotations

                    import gc
                    import json
                    import os
                    import random
                    import time
                    from pathlib import Path

                    import numpy as np
                    import pandas as pd
                    import torch
                    from pytabkit.models.sklearn.sklearn_interfaces import RealMLP_TD_Classifier
                    from sklearn.metrics import roc_auc_score
                    from sklearn.model_selection import StratifiedShuffleSplit

                    RUN_MODE = "auto"  # auto | smoke | full_gpu

                    TOP_EXTRA_FEATURES = """
                ).strip()
                + " "
                + top_features_literal
                + "\n\n"
                + dedent(
                    """

                    SMOKE_CFG = {
                        "sample_rows": 60000,
                        "target_limit": 4,
                        "val_fraction": 0.20,
                        "seed": 42,
                        "n_epochs": 6,
                        "batch_size": 1024,
                        "predict_batch_size": 4096,
                        "n_threads": 4,
                        "train_final_model": False,
                    }

                    FULL_CFG = {
                        "sample_rows": None,
                        "target_limit": None,
                        "val_fraction": 0.10,
                        "seed": 42,
                        "n_epochs": 64,
                        "batch_size": 4096,
                        "predict_batch_size": 16384,
                        "n_threads": 8,
                        "train_final_model": True,
                    }

                    def find_data_dir() -> Path:
                        direct_candidates = [
                            Path.cwd() / "data" / "competition",
                            Path.cwd(),
                            Path("/kaggle/input/data-fusion-2026"),
                            Path("/kaggle/input/data-fusion-2026-cybershelf"),
                        ]
                        required = [
                            "train_main_features.parquet",
                            "train_extra_features.parquet",
                            "train_target.parquet",
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


                    def pick_device_info() -> dict[str, object]:
                        if torch.cuda.is_available():
                            capability = None
                            try:
                                major, minor = torch.cuda.get_device_capability(0)
                                capability = [int(major), int(minor)]
                            except Exception:
                                capability = None
                            raw_name = torch.cuda.get_device_name(0)
                            if capability and int(capability[0]) < 7:
                            return {
                                "device": "cuda",
                                "raw_device": "cuda",
                                "capability": capability,
                                "raw_name": raw_name,
                                "fallback_reason": "cuda_sm_lt_70_realmlp_unsupported",
                            }
                            return {
                                "device": "cuda",
                                "raw_device": "cuda",
                                "capability": capability,
                                "raw_name": raw_name,
                                "fallback_reason": None,
                            }
                        if torch.backends.mps.is_available():
                            return {
                                "device": "mps",
                                "raw_device": "mps",
                                "capability": None,
                                "raw_name": "Apple MPS",
                                "fallback_reason": None,
                            }
                        return {
                            "device": "cpu",
                            "raw_device": "cpu",
                            "capability": None,
                            "raw_name": "CPU",
                            "fallback_reason": None,
                        }


                    def resolve_mode(run_mode: str, device_info: dict[str, object]) -> str:
                        device = str(device_info["device"])
                        if run_mode == "smoke":
                            return "smoke"
                        if run_mode == "full_gpu":
                            return "full_gpu"
                        return "full_gpu" if device == "cuda" else "smoke"


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
                        train_df: pd.DataFrame,
                        other_frames: list[pd.DataFrame],
                        feature_cols: list[str],
                        cat_cols: list[str],
                        num_cols: list[str],
                    ) -> tuple[pd.DataFrame, list[pd.DataFrame], np.ndarray]:
                        train_x = train_df[feature_cols].copy()
                        other_x = [frame[feature_cols].copy() for frame in other_frames]

                        for col in num_cols:
                            train_x[col] = pd.to_numeric(train_x[col], errors="coerce")
                            median = float(train_x[col].median()) if train_x[col].notna().any() else 0.0
                            train_x[col] = train_x[col].fillna(median).astype("float32")
                            for frame in other_x:
                                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median).astype("float32")

                        for col in cat_cols:
                            train_x[col] = pd.to_numeric(train_x[col], errors="coerce").fillna(-1).astype("int32")
                            for frame in other_x:
                                frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int32")

                        cat_indicator = np.array([col in cat_cols for col in feature_cols], dtype=bool)
                        return train_x, other_x, cat_indicator


                    def target_to_pred(target_name: str) -> str:
                        return target_name.replace("target_", "predict_")


                    def single_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
                        if len(np.unique(y_true)) < 2:
                            return 0.5
                        return float(roc_auc_score(y_true, y_score))


                    def macro_auc(target_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                        scores = []
                        for target_name in target_cols:
                            pred_col = target_to_pred(target_name)
                            scores.append(single_auc(target_df[target_name].to_numpy(), pred_df[pred_col].to_numpy()))
                        return float(np.mean(scores))


                    def make_realmlp_model(cfg: dict[str, object], device: str, seed: int) -> RealMLP_TD_Classifier:
                        return RealMLP_TD_Classifier(
                        device=device,
                        random_state=seed,
                        n_cv=1,
                        n_refit=0,
                        val_fraction=float(cfg["val_fraction"]),
                        n_threads=int(cfg["n_threads"]),
                        verbosity=0,
                        hidden_sizes=[256, 256, 256],
                        max_one_hot_cat_size=9,
                        embedding_size=8,
                        weight_param="ntk",
                        bias_lr_factor=0.1,
                        act="selu",
                        use_parametric_act=True,
                        act_lr_factor=0.1,
                        block_str="w-b-a-d",
                        p_drop=0.15,
                        p_drop_sched="flat_cos",
                        add_front_scale=True,
                        scale_lr_factor=6.0,
                        bias_init_mode="he+5",
                        weight_init_mode="std",
                        wd=2e-2,
                        wd_sched="flat_cos",
                        bias_wd_factor=0.0,
                        use_ls=True,
                        ls_eps=0.1,
                        num_emb_type="pbld",
                        plr_sigma=0.1,
                        plr_hidden_1=16,
                        plr_hidden_2=4,
                        plr_lr_factor=0.1,
                        lr=4e-2,
                        tfms=["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding"],
                        n_epochs=int(cfg["n_epochs"]),
                        batch_size=int(cfg["batch_size"]),
                        predict_batch_size=int(cfg["predict_batch_size"]),
                        lr_sched="coslog4",
                        opt="adam",
                        sq_mom=0.95,
                        use_early_stopping=True,
                        early_stopping_additive_patience=20,
                    )


                    def fit_predict_target(
                        target_name: str,
                    train_x: pd.DataFrame,
                    train_y: np.ndarray,
                    val_x: pd.DataFrame,
                    val_y: np.ndarray,
                    test_x: pd.DataFrame,
                    cat_indicator: np.ndarray,
                    cfg: dict[str, object],
                    device: str,
                    seed: int,
                    ) -> tuple[np.ndarray, np.ndarray, float]:
                        model = make_realmlp_model(cfg=cfg, device=device, seed=seed)
                        start = time.time()
                        model.fit(train_x, train_y, X_val=val_x, y_val=val_y, cat_indicator=cat_indicator)
                        val_proba = model.predict_proba(val_x)[:, 1].astype("float32")
                        test_proba = model.predict_proba(test_x)[:, 1].astype("float32")
                        auc = single_auc(val_y, val_proba)
                        elapsed = float(time.time() - start)
                        print(
                            {
                                "stage": "target_done",
                                "target": target_name,
                                "val_auc": auc,
                                "elapsed_seconds": round(elapsed, 2),
                                "positives_train": int(train_y.sum()),
                                "positives_val": int(val_y.sum()),
                            },
                            flush=True,
                        )
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        return val_proba, test_proba, elapsed


                    DATA_DIR = find_data_dir()
                    DEVICE_INFO = pick_device_info()
                    if DEVICE_INFO.get("fallback_reason"):
                        raise RuntimeError("Unsupported Kaggle GPU for RealMLP CUDA path: P100/sm_60. Restart until T4+ or run locally on MPS.")
                    DEVICE = str(DEVICE_INFO["device"])
                    RESOLVED_MODE = resolve_mode(RUN_MODE, DEVICE_INFO)
                    CFG = FULL_CFG if RESOLVED_MODE == "full_gpu" else SMOKE_CFG
                    ARTIFACT_DIR = Path("artifacts") / f"gpu_realmlp_tdlike_{RESOLVED_MODE}"
                    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                    print(
                        {
                            "resolved_mode": RESOLVED_MODE,
                            "device": DEVICE,
                            "device_info": DEVICE_INFO,
                            "extra_top_k": int(len(TOP_EXTRA_FEATURES)),
                            "config": CFG,
                        },
                        flush=True,
                    )
                    """
                ).strip()
            )
            + "\n",
            4,
        ),
        md_cell("## Load Data\n", 5),
        code_cell(
            dedent(
                """
                train_main = pd.read_parquet(DATA_DIR / "train_main_features.parquet")
                train_extra = pd.read_parquet(DATA_DIR / "train_extra_features.parquet", columns=["customer_id"] + TOP_EXTRA_FEATURES)
                train_target = pd.read_parquet(DATA_DIR / "train_target.parquet")
                test_main = pd.read_parquet(DATA_DIR / "test_main_features.parquet")
                test_extra = pd.read_parquet(DATA_DIR / "test_extra_features.parquet", columns=["customer_id"] + TOP_EXTRA_FEATURES)

                labeled_df = (
                    train_main
                    .merge(train_extra, on="customer_id", how="left")
                    .merge(train_target, on="customer_id", how="inner")
                    .sort_values("customer_id")
                    .reset_index(drop=True)
                )
                test_df = (
                    test_main
                    .merge(test_extra, on="customer_id", how="left")
                    .sort_values("customer_id")
                    .reset_index(drop=True)
                )

                feature_cols = [c for c in labeled_df.columns if c != "customer_id" and not c.startswith("target_")]
                target_cols = [c for c in labeled_df.columns if c.startswith("target_")]
                target_cols = sorted(target_cols, key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])))
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
                num_cols = [c for c in feature_cols if c not in cat_cols]

                if CFG["sample_rows"] is not None:
                    labeled_df = ensure_smoke_sample(
                        full_df=labeled_df,
                        target_cols=target_cols,
                        sample_rows=int(CFG["sample_rows"]),
                        min_positive=20,
                        seed=int(CFG["seed"]),
                    )
                    if CFG["target_limit"] is not None:
                        target_cols = target_cols[: int(CFG["target_limit"])]

                train_df, val_df = split_with_train_positive_coverage(
                    df=labeled_df,
                    target_cols=target_cols,
                    val_fraction=float(CFG["val_fraction"]),
                    seed=int(CFG["seed"]),
                )

                print(
                    {
                        "data_dir": str(DATA_DIR),
                        "labeled_rows": len(labeled_df),
                        "train_rows": len(train_df),
                        "val_rows": len(val_df),
                        "test_rows": len(test_df),
                        "feature_count": len(feature_cols),
                        "num_features": len(num_cols),
                        "cat_features": len(cat_cols),
                        "target_count": len(target_cols),
                        "device": DEVICE,
                    },
                    flush=True,
                )
                """
            ).strip()
            + "\n",
            6,
        ),
        md_cell("## Validation Run\n", 7),
        code_cell(
            dedent(
                """
                train_x, [val_x, test_x], cat_indicator = prepare_feature_frames(
                    train_df=train_df,
                    other_frames=[val_df, test_df],
                    feature_cols=feature_cols,
                    cat_cols=cat_cols,
                    num_cols=num_cols,
                )

                val_pred_df = pd.DataFrame({"customer_id": val_df["customer_id"].astype("int32").values})
                test_pred_from_split_df = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                target_rows = []

                for target_name in target_cols:
                    train_y = train_df[target_name].astype("int8").to_numpy()
                    val_y = val_df[target_name].astype("int8").to_numpy()
                    val_proba, test_proba, elapsed = fit_predict_target(
                        target_name=target_name,
                        train_x=train_x,
                        train_y=train_y,
                        val_x=val_x,
                        val_y=val_y,
                        test_x=test_x,
                        cat_indicator=cat_indicator,
                        cfg=CFG,
                        device=DEVICE,
                        seed=int(CFG["seed"]),
                    )
                    pred_col = target_to_pred(target_name)
                    val_pred_df[pred_col] = val_proba.astype("float64")
                    test_pred_from_split_df[pred_col] = test_proba.astype("float64")
                    target_rows.append(
                        {
                            "target": target_name,
                            "val_auc": float(single_auc(val_y, val_proba)),
                            "train_positive_rate": float(train_y.mean()),
                            "val_positive_rate": float(val_y.mean()),
                            "elapsed_seconds": float(elapsed),
                        }
                    )

                target_score_df = pd.DataFrame(target_rows).sort_values("val_auc").reset_index(drop=True)
                validation_macro_auc = float(macro_auc(val_df[["customer_id"] + target_cols], val_pred_df, target_cols))

                val_pred_df.to_parquet(ARTIFACT_DIR / "validation_predictions.parquet", index=False)
                test_pred_from_split_df.to_parquet(ARTIFACT_DIR / "split_model_test_predictions.parquet", index=False)
                target_score_df.to_csv(ARTIFACT_DIR / "per_target_auc.csv", index=False)

                metrics_payload = {
                    "validation_macro_auc": validation_macro_auc,
                    "resolved_mode": RESOLVED_MODE,
                    "device": DEVICE,
                    "rows_train": int(len(train_df)),
                    "rows_val": int(len(val_df)),
                    "rows_test": int(len(test_df)),
                    "feature_count": int(len(feature_cols)),
                    "num_features": int(len(num_cols)),
                    "cat_features": int(len(cat_cols)),
                    "target_count": int(len(target_cols)),
                    "seed": int(CFG["seed"]),
                    "model": "RealMLP_TD_Classifier_one_vs_rest",
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
                print(json.dumps(metrics_payload, indent=2), flush=True)
                print(target_score_df.head(12).to_string(index=False), flush=True)
                """
            ).strip()
            + "\n",
            8,
        ),
        md_cell("## Full Train And Submission\n", 9),
        code_cell(
            dedent(
                """
                if bool(CFG["train_final_model"]):
                    full_x, [test_x_full], full_cat_indicator = prepare_feature_frames(
                        train_df=labeled_df,
                        other_frames=[test_df],
                        feature_cols=feature_cols,
                        cat_cols=cat_cols,
                        num_cols=num_cols,
                    )

                    submission = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                    final_rows = []
                    for target_name in target_cols:
                        full_y = labeled_df[target_name].astype("int8").to_numpy()
                        model = make_realmlp_model(cfg=CFG, device=DEVICE, seed=int(CFG["seed"]))
                        start = time.time()
                        model.fit(full_x, full_y, cat_indicator=full_cat_indicator)
                        test_proba = model.predict_proba(test_x_full)[:, 1].astype("float64")
                        submission[target_to_pred(target_name)] = test_proba
                        final_rows.append(
                            {
                                "target": target_name,
                                "elapsed_seconds": float(time.time() - start),
                                "positive_rate_full": float(full_y.mean()),
                            }
                        )
                        print(
                            {"stage": "full_target_done", "target": target_name, "elapsed_seconds": round(final_rows[-1]["elapsed_seconds"], 2)},
                            flush=True,
                        )
                        del model
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    submission_path = ARTIFACT_DIR / "submission.parquet"
                    submission.to_parquet(submission_path, index=False)
                    pd.DataFrame(final_rows).to_csv(ARTIFACT_DIR / "full_train_runtime.csv", index=False)
                    print({"stage": "submission_written", "submission_path": str(submission_path), "rows": len(submission)}, flush=True)
                else:
                    print("Smoke mode: final full-train submission step skipped.", flush=True)
                """
            ).strip()
            + "\n",
            10,
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
        "id": "chesnikovleonid/data-fusion-2026-task-2-realmlp-tdlike-ovr-v3",
        "title": "Data Fusion 2026 task 2 RealMLP TDlike OVR v3",
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
