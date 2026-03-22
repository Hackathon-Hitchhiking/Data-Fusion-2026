from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root

ROOT = project_root()
TOP_FEATURES_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
NOTEBOOK_PATH = ROOT / "output/jupyter-notebook/data-fusion-2026-gpu-tabm-final-only.ipynb"
LONGRUN_V3_METRICS = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/metrics.json"
)


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


def load_fixed_final_epochs() -> dict[int, int]:
    default = {42: 19, 52: 18}
    if not LONGRUN_V3_METRICS.exists():
        return default
    try:
        payload = json.loads(LONGRUN_V3_METRICS.read_text())
        seed_metrics = payload.get("seed_metrics", [])
        parsed = {
            int(item["seed"]): int(item["final_train_epochs"])
            for item in seed_metrics
            if "seed" in item and "final_train_epochs" in item
        }
        return parsed or default
    except Exception:
        return default


def main() -> None:
    top_features = json.loads(TOP_FEATURES_FILE.read_text())
    top_features_literal = json.dumps(top_features, ensure_ascii=True, indent=4)
    final_epochs_literal = json.dumps(load_fixed_final_epochs(), ensure_ascii=True, indent=4, sort_keys=True)

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU TabM Final Only

                Objective:
                - Skip validation entirely and train only the already-selected `TabM v3` recipe on full labeled data.
                - Use the proven per-seed epoch counts from the completed `longrun v3` validation artifacts.

                Design:
                - Multi-label training over all `41` targets with `BCEWithLogitsLoss`.
                - `PiecewiseLinearEmbeddings(version="B")` with the same robust preprocessing as `longrun v3`.
                - Full-train only, multi-seed averaging on test predictions, and production-safe logging.
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
                - Smoke mode runs the exact same code path on a small sample.
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
                    from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
                    from sklearn.model_selection import StratifiedShuffleSplit
                    from tabm import TabM
                    from torch.utils.data import DataLoader, TensorDataset

                    RUN_MODE = "auto"  # one of: auto, smoke, full_gpu

                    TOP_EXTRA_FEATURES = """
                ).strip()
                + " "
                + top_features_literal
                + "\n\nFIXED_FINAL_EPOCHS = "
                + final_epochs_literal
                + "\n\n"
                + dedent(
                    """
                    def resolve_run_mode(mode: str) -> str:
                        if mode != "auto":
                            return mode
                        return "full_gpu" if torch.cuda.is_available() and Path("/kaggle/input").exists() else "smoke"


                    def set_all_seeds(seed: int) -> None:
                        random.seed(seed)
                        np.random.seed(seed)
                        torch.manual_seed(seed)
                        if torch.cuda.is_available():
                            torch.cuda.manual_seed_all(seed)


                    RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                    CFG = {
                        "smoke": {
                            "seeds": [42],
                            "sample_rows": 80_000,
                            "smoke_min_positive": 12,
                            "extra_top_k": 40,
                            "seed_epochs": {"42": 2},
                            "batch_size": 256,
                            "eval_batch_size": 1024,
                            "lr": 2e-3,
                            "weight_decay": 1e-4,
                            "num_embedding_type": "piecewise",
                            "piecewise_n_bins": 16,
                            "piecewise_bin_sample_rows": 40_000,
                            "piecewise_d_embedding": 16,
                            "piecewise_activation": False,
                            "clip_value": 8.0,
                            "tabm_arch_type": "tabm",
                            "tabm_k": 32,
                            "tabm_d_block": 512,
                            "tabm_n_blocks": 2,
                            "tabm_dropout": 0.10,
                        },
                        "full_gpu": {
                            "seeds": [42, 52],
                            "sample_rows": None,
                            "smoke_min_positive": 0,
                            "extra_top_k": 100,
                            "seed_epochs": FIXED_FINAL_EPOCHS,
                            "batch_size": 896,
                            "eval_batch_size": 4096,
                            "lr": 2e-3,
                            "weight_decay": 1e-4,
                            "num_embedding_type": "piecewise",
                            "piecewise_n_bins": 32,
                            "piecewise_bin_sample_rows": 200_000,
                            "piecewise_d_embedding": 16,
                            "piecewise_activation": False,
                            "clip_value": 8.0,
                            "tabm_arch_type": "tabm",
                            "tabm_k": 32,
                            "tabm_d_block": 512,
                            "tabm_n_blocks": 2,
                            "tabm_dropout": 0.10,
                        },
                    }[RESOLVED_MODE]

                    ARTIFACT_DIR = Path("artifacts") / f"gpu_tabm_final_only_{RESOLVED_MODE}"
                    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                    if hasattr(torch, "set_float32_matmul_precision"):
                        torch.set_float32_matmul_precision("high")

                    set_all_seeds(int(CFG["seeds"][0]))

                    print({
                        "resolved_mode": RESOLVED_MODE,
                        "torch_version": torch.__version__,
                        "cuda_available": torch.cuda.is_available(),
                        "mps_available": torch.backends.mps.is_available(),
                        "config": CFG,
                        "artifact_dir": str(ARTIFACT_DIR),
                    }, flush=True)
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


                def smooth_clip(values: pd.Series, clip_value: float) -> pd.Series:
                    clipped = clip_value * np.tanh(values.to_numpy(dtype=np.float32, copy=False) / clip_value)
                    return pd.Series(clipped, index=values.index, dtype="float32")


                def prepare_arrays(
                    train_df: pd.DataFrame,
                    other_frames: list[pd.DataFrame],
                    num_cols: list[str],
                    cat_cols: list[str],
                    target_cols: list[str] | None,
                    clip_value: float,
                ) -> tuple[dict[str, np.ndarray], list[dict[str, np.ndarray]], list[int], list[str]]:
                    train_num = train_df[num_cols].copy()
                    other_num = [frame[num_cols].copy() for frame in other_frames]
                    train_cat = train_df[cat_cols].copy()
                    other_cat = [frame[cat_cols].copy() for frame in other_frames]

                    for col in num_cols:
                        train_num[col] = pd.to_numeric(train_num[col], errors="coerce")
                        median = float(train_num[col].median())
                        q1 = float(train_num[col].quantile(0.25))
                        q3 = float(train_num[col].quantile(0.75))
                        iqr = q3 - q1
                        if iqr <= 1e-6:
                            std = float(train_num[col].std())
                            scale = std if std > 1e-6 else 1.0
                        else:
                            scale = iqr
                        train_num[col] = train_num[col].fillna(median)
                        train_num[col] = ((train_num[col] - median) / scale).astype("float32")
                        train_num[col] = smooth_clip(train_num[col], clip_value)
                        for frame in other_num:
                            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median)
                            frame[col] = ((frame[col] - median) / scale).astype("float32")
                            frame[col] = smooth_clip(frame[col], clip_value)

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

                    train_num_matrix = train_num.to_numpy(dtype=np.float32, copy=True)
                    if train_num_matrix.shape[1] == 0:
                        varying_num_cols = []
                    else:
                        num_ptp = np.nanmax(train_num_matrix, axis=0) - np.nanmin(train_num_matrix, axis=0)
                        varying_num_cols = [
                            col
                            for col, keep in zip(num_cols, np.isfinite(num_ptp) & (num_ptp > 0))
                            if keep
                        ]
                    train_num_final = train_num[varying_num_cols] if varying_num_cols else train_num.iloc[:, :0]
                    other_num_final = [frame[varying_num_cols] if varying_num_cols else frame.iloc[:, :0] for frame in other_num]

                    train_arrays = {
                        "x_num": train_num_final.to_numpy(dtype=np.float32, copy=True),
                        "x_cat": train_cat.to_numpy(dtype=np.int16, copy=True),
                    }
                    if target_cols is not None:
                        train_arrays["y"] = train_df[target_cols].to_numpy(dtype=np.float32, copy=True)

                    other_arrays = []
                    for idx in range(len(other_frames)):
                        payload = {
                            "x_num": other_num_final[idx].to_numpy(dtype=np.float32, copy=True),
                            "x_cat": other_cat[idx].to_numpy(dtype=np.int16, copy=True),
                        }
                        if target_cols is not None and all(col in other_frames[idx].columns for col in target_cols):
                            payload["y"] = other_frames[idx][target_cols].to_numpy(dtype=np.float32, copy=True)
                        other_arrays.append(payload)

                    return train_arrays, other_arrays, cat_cardinalities, varying_num_cols


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


                def make_num_embeddings(train_x_num: np.ndarray, cfg: dict[str, object], seed: int):
                    num_embedding_type = str(cfg["num_embedding_type"])
                    if train_x_num.shape[1] == 0:
                        return None, {}
                    if num_embedding_type == "piecewise":
                        sample_rows = int(cfg["piecewise_bin_sample_rows"])
                        rng = np.random.default_rng(seed)
                        if len(train_x_num) > sample_rows:
                            indices = rng.choice(len(train_x_num), size=sample_rows, replace=False)
                            bin_source = train_x_num[indices]
                        else:
                            bin_source = train_x_num
                        used_full_fallback = False
                        bin_ptp = np.nanmax(bin_source, axis=0) - np.nanmin(bin_source, axis=0)
                        if not np.all(np.isfinite(bin_ptp) & (bin_ptp > 0)):
                            bin_source = train_x_num
                            used_full_fallback = True
                            bin_ptp = np.nanmax(bin_source, axis=0) - np.nanmin(bin_source, axis=0)
                            if not np.all(np.isfinite(bin_ptp) & (bin_ptp > 0)):
                                bad_count = int((~(np.isfinite(bin_ptp) & (bin_ptp > 0))).sum())
                                raise ValueError(
                                    f"Piecewise bin source still contains {bad_count} constant/invalid numeric columns "
                                    "after prepare_arrays filtering."
                                )
                        bin_tensor = torch.from_numpy(bin_source).float().cpu()
                        bins = compute_bins(bin_tensor, n_bins=int(cfg["piecewise_n_bins"]))
                        embedding = PiecewiseLinearEmbeddings(
                            bins,
                            d_embedding=int(cfg["piecewise_d_embedding"]),
                            activation=bool(cfg["piecewise_activation"]),
                            version="B",
                        )
                        info = {
                            "type": "piecewise",
                            "n_bins": int(cfg["piecewise_n_bins"]),
                            "bin_rows": int(len(bin_source)),
                            "used_full_fallback": used_full_fallback,
                            "d_embedding": int(cfg["piecewise_d_embedding"]),
                            "activation": bool(cfg["piecewise_activation"]),
                            "version": "B",
                        }
                        return embedding, info
                    raise ValueError(f"Unknown num_embedding_type: {num_embedding_type}")


                def train_final_single_seed(
                    full_arrays: dict[str, np.ndarray],
                    test_arrays: dict[str, np.ndarray],
                    target_cols: list[str],
                    cat_cardinalities: list[int],
                    cfg: dict[str, object],
                    device: str,
                    artifact_dir: Path,
                    seed: int,
                    final_epochs: int,
                ) -> np.ndarray:
                    set_all_seeds(seed)
                    full_loader = make_loader(full_arrays, int(cfg["batch_size"]), shuffle=True)
                    test_loader = make_loader(test_arrays, int(cfg["eval_batch_size"]), shuffle=False)

                    num_embeddings, embedding_info = make_num_embeddings(full_arrays["x_num"], cfg, seed)
                    print({"stage": "tabm_final_num_embeddings_ready", "seed": seed, **embedding_info}, flush=True)

                    model = TabM.make(
                        n_num_features=full_arrays["x_num"].shape[1],
                        cat_cardinalities=cat_cardinalities,
                        d_out=len(target_cols),
                        num_embeddings=num_embeddings,
                        arch_type=str(cfg["tabm_arch_type"]),
                        k=int(cfg["tabm_k"]),
                        d_block=int(cfg["tabm_d_block"]),
                        n_blocks=int(cfg["tabm_n_blocks"]),
                        dropout=float(cfg["tabm_dropout"]),
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(cfg["lr"]),
                        weight_decay=float(cfg["weight_decay"]),
                    )

                    history_rows = []
                    print(
                        {
                            "stage": "tabm_final_train_start",
                            "seed": seed,
                            "epochs": int(final_epochs),
                            "rows": len(full_arrays["x_num"]),
                            "test_rows": len(test_arrays["x_num"]),
                        },
                        flush=True,
                    )
                    for epoch in range(1, int(final_epochs) + 1):
                        model.train()
                        losses = []
                        for x_num, x_cat, y in full_loader:
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
                        history_rows.append(
                            {"seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses))}
                        )
                        pd.DataFrame(history_rows).to_csv(artifact_dir / f"seed_{seed}_history.csv", index=False)
                        print(
                            {"stage": "tabm_final_epoch_done", "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses))},
                            flush=True,
                        )

                    return predict_proba(model, test_loader, device)
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
                        seed=int(CFG["seeds"][0]),
                    )

                print({
                    "data_dir": str(DATA_DIR),
                    "labeled_rows": len(labeled_df),
                    "test_rows": len(test_df),
                    "feature_count": len(feature_cols),
                    "num_features": len(num_cols),
                    "cat_features": len(cat_cols),
                    "device": DEVICE,
                }, flush=True)
                """
            ).strip()
            + "\n"
        ),
        md_cell("## Full Train And Submission\n"),
        code_cell(
            dedent(
                """
                full_arrays, [test_arrays], full_cat_cardinalities, full_varying_num_cols = prepare_arrays(
                    train_df=labeled_df[feature_cols + target_cols],
                    other_frames=[test_df[feature_cols]],
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    target_cols=target_cols,
                    clip_value=float(CFG["clip_value"]),
                )

                print({
                    "stage": "tabm_final_prepare_done",
                    "train_rows": len(labeled_df),
                    "test_rows": len(test_df),
                    "max_cat_cardinality": int(max(full_cat_cardinalities)),
                    "num_embedding_type": str(CFG["num_embedding_type"]),
                    "varying_num_features": int(len(full_varying_num_cols)),
                    "seed_epochs": {int(k): int(v) for k, v in CFG["seed_epochs"].items()},
                }, flush=True)

                test_prob_blocks = []
                seed_rows = []
                for seed in CFG["seeds"]:
                    final_epochs = int(CFG["seed_epochs"][str(seed)] if str(seed) in CFG["seed_epochs"] else CFG["seed_epochs"][seed])
                    probs = train_final_single_seed(
                        full_arrays=full_arrays,
                        test_arrays=test_arrays,
                        target_cols=target_cols,
                        cat_cardinalities=full_cat_cardinalities,
                        cfg=CFG,
                        device=DEVICE,
                        artifact_dir=ARTIFACT_DIR,
                        seed=int(seed),
                        final_epochs=final_epochs,
                    )
                    test_prob_blocks.append(probs)
                    seed_rows.append({"seed": int(seed), "final_epochs": int(final_epochs)})

                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                pd.DataFrame(seed_rows).to_csv(ARTIFACT_DIR / "seed_metrics.csv", index=False)

                test_probs = np.mean(test_prob_blocks, axis=0)
                submission = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                for idx, target_name in enumerate(target_cols):
                    submission[target_name.replace("target_", "predict_")] = test_probs[:, idx].astype("float64")

                submission_path = ARTIFACT_DIR / "submission.parquet"
                submission.to_parquet(submission_path, index=False)
                (ARTIFACT_DIR / "config.json").write_text(
                    json.dumps(
                        {
                            "resolved_mode": RESOLVED_MODE,
                            "device": DEVICE,
                            "seed_epochs": {int(k): int(v) for k, v in CFG["seed_epochs"].items()},
                            "labeled_rows": int(len(labeled_df)),
                            "test_rows": int(len(test_df)),
                            "feature_count": int(len(feature_cols)),
                            "num_features": int(len(num_cols)),
                            "cat_features": int(len(cat_cols)),
                            "varying_num_features": int(len(full_varying_num_cols)),
                            "num_embedding_type": str(CFG["num_embedding_type"]),
                        },
                        indent=2,
                    )
                )
                print({"stage": "submission_written", "submission_path": str(submission_path), "rows": len(submission)}, flush=True)
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, indent=2))
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
