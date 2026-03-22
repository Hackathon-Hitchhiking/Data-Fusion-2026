from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
NOTEBOOK_NAME = "data-fusion-2026-gpu-tabm-family-head-v2-fs-v1.ipynb"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output" / "kaggle-push" / "tabm-family-head-v2-fs-v1"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "tabm_family_head_v2_fs_v1_rev1_20260321"


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
        '    ("tabm", "tabm"),',
        '    ("rtdl_num_embeddings", "rtdl-num-embeddings"),',
        '    ("pyarrow", "pyarrow"),',
        '    ("sklearn", "scikit-learn"),',
        "]",
        "",
        "def pip_install(args: list[str]) -> None:",
        '    subprocess.check_call([sys.executable, "-m", "pip", *args])',
        "",
        "def torch_runtime_info() -> dict[str, object] | None:",
        '    if importlib.util.find_spec("torch") is None:',
        "        return None",
        '    code = """',
        "import json",
        "import torch",
        "info = {",
        '    "version": torch.__version__,',
        '    "cuda_available": bool(torch.cuda.is_available()),',
        '    "capability": None,',
        "}",
        'if info["cuda_available"]:',
        "    try:",
        "        major, minor = torch.cuda.get_device_capability(0)",
        '        info["capability"] = [int(major), int(minor)]',
        "    except Exception:",
        '        info["capability"] = None',
        "print(json.dumps(info))",
        '""".strip()',
        '    raw = subprocess.check_output([sys.executable, "-c", code], text=True).strip()',
        "    return json.loads(raw)",
        "",
        "torch_info = torch_runtime_info()",
        'missing = [pip_name for module_name, pip_name in REQUIRED_PACKAGES if importlib.util.find_spec(module_name) is None]',
        'print({"stage": "bootstrap_start", "revision": NOTEBOOK_REVISION, "missing": missing, "torch_info": torch_info}, flush=True)',
        "",
        'if torch_info and torch_info.get("cuda_available") and torch_info.get("capability"):',
        '    capability = torch_info["capability"]',
        '    if int(capability[0]) < 7 and str(torch_info["version"]).startswith("2.10."):',
        "        pip_install([",
        '            "install",',
        '            "--upgrade",',
        '            "--no-cache-dir",',
        '            "--force-reinstall",',
        '            "--index-url",',
        '            "https://download.pytorch.org/whl/cu126",',
        '            "torch==2.9.0",',
        "        ])",
        "        torch_info = torch_runtime_info()",
        '        print({"stage": "torch_runtime_patched", "torch_info": torch_info}, flush=True)',
        "",
        "if missing:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing])',
        'print({"stage": "bootstrap_done", "installed_now": missing, "torch_info": torch_info}, flush=True)',
    ]

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU TabM Family Head v2 FS v1

                Objective:
                - Keep the strongest confirmed `TabM` trunk and only change the output architecture.
                - Train on the exact current `75k` holdout (`val_ids.parquet`) and the compact `fs_v1` feature set.
                - Test whether family-conditioned heads add a new stable signal without introducing a new backbone.

                Design:
                - Exact same preprocessing and numerical embedding recipe as `TabM longrun v3 fs_v1`.
                - Shared `TabM` trunk with `d_out=None`.
                - `10` family heads derived from the target namespace (`target_<family>_<label>`).
                - `41` label heads consuming `[trunk_embedding, family_embedding]`.
                - No auxiliary losses, no graph loss, no extra feature views.
                - Soft capped `pos_weight = min(cap, sqrt(neg/pos))` to help the rarest labels without destabilizing training.
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
                - Smoke mode only validates the full code path.
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
                import math
                import random
                import re
                from pathlib import Path

                import numpy as np
                import pandas as pd
                import torch
                import torch.nn as nn
                import torch.nn.functional as F
                from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
                from sklearn.metrics import roc_auc_score
                from sklearn.model_selection import StratifiedShuffleSplit
                from tabm import TabM
                from torch.utils.data import DataLoader, TensorDataset

                RUN_MODE = "auto"  # auto | smoke | full_gpu

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
                        "sample_rows": 60_000,
                        "smoke_min_positive": 12,
                        "epochs": 4,
                        "batch_size": 256,
                        "eval_batch_size": 1024,
                        "lr": 2e-3,
                        "weight_decay": 1e-4,
                        "min_epochs": 2,
                        "early_stopping_patience": 2,
                        "train_final_model": False,
                        "num_embedding_type": "piecewise",
                        "piecewise_n_bins": 16,
                        "piecewise_bin_sample_rows": 40_000,
                        "piecewise_d_embedding": 16,
                        "piecewise_activation": False,
                        "clip_value": 8.0,
                        "checkpoint_average_top_k": 2,
                        "tabm_arch_type": "tabm",
                        "tabm_k": 32,
                        "tabm_d_block": 512,
                        "tabm_n_blocks": 2,
                        "tabm_dropout": 0.10,
                        "family_hidden_dim": 32,
                        "family_dropout": 0.10,
                        "label_head_hidden_dim": 32,
                        "label_head_dropout": 0.10,
                        "pos_weight_cap": 12.0,
                    },
                    "full_gpu": {
                        "seeds": [42, 52, 62],
                        "sample_rows": None,
                        "smoke_min_positive": 0,
                        "epochs": 32,
                        "batch_size": 768,
                        "eval_batch_size": 4096,
                        "lr": 2e-3,
                        "weight_decay": 1e-4,
                        "min_epochs": 14,
                        "early_stopping_patience": 8,
                        "train_final_model": True,
                        "num_embedding_type": "piecewise",
                        "piecewise_n_bins": 32,
                        "piecewise_bin_sample_rows": 200_000,
                        "piecewise_d_embedding": 16,
                        "piecewise_activation": False,
                        "clip_value": 8.0,
                        "checkpoint_average_top_k": 4,
                        "tabm_arch_type": "tabm",
                        "tabm_k": 32,
                        "tabm_d_block": 512,
                        "tabm_n_blocks": 2,
                        "tabm_dropout": 0.10,
                        "family_hidden_dim": 32,
                        "family_dropout": 0.10,
                        "label_head_hidden_dim": 32,
                        "label_head_dropout": 0.10,
                        "pos_weight_cap": 12.0,
                    },
                }[RESOLVED_MODE]

                ARTIFACT_DIR = Path("artifacts") / f"gpu_tabm_family_head_v2_fs_v1_{RESOLVED_MODE}"
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
                        Path.cwd() / "artifacts" / "feature_selection_v1",
                        Path.cwd() / "artifacts" / "tabrs_pilot_inputs",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                    ]
                    required = [
                        "train_compact_features_v1.parquet",
                        "test_compact_features_v1.parquet",
                        "val_ids.parquet",
                        "current_best_validation_reference.parquet",
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


                def macro_auc(y_true: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    scores = []
                    for target_name in target_cols:
                        pred_col = target_name.replace("target_", "predict_")
                        target_values = y_true[target_name]
                        if target_values.nunique() < 2:
                            scores.append(0.5)
                        else:
                            scores.append(float(roc_auc_score(target_values, pred_df[pred_col])))
                    return float(np.mean(scores))


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


                def average_state_dicts(state_dicts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
                    assert state_dicts, "state_dicts must be non-empty"
                    avg_state = {}
                    keys = state_dicts[0].keys()
                    for key in keys:
                        first = state_dicts[0][key]
                        if torch.is_floating_point(first):
                            stacked = torch.stack([state[key].float() for state in state_dicts], dim=0)
                            avg_state[key] = stacked.mean(0).to(first.dtype)
                        else:
                            avg_state[key] = first.clone()
                    return avg_state


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


                def build_family_spec(target_cols: list[str]) -> tuple[list[str], dict[str, str], dict[str, list[str]]]:
                    family_to_targets: dict[str, list[str]] = {}
                    target_to_family: dict[str, str] = {}
                    for target_name in target_cols:
                        match = re.fullmatch(r"target_(\\d+)_\\d+", target_name)
                        if match is None:
                            raise ValueError(f"Unexpected target name format: {target_name}")
                        family_id = match.group(1)
                        family_to_targets.setdefault(family_id, []).append(target_name)
                        target_to_family[target_name] = family_id
                    family_ids = sorted(family_to_targets, key=lambda x: int(x))
                    family_to_targets = {k: sorted(v, key=lambda x: tuple(map(int, x.replace("target_", "").split("_")))) for k, v in family_to_targets.items()}
                    return family_ids, target_to_family, family_to_targets


                def compute_pos_weight(y: np.ndarray, cap: float) -> np.ndarray:
                    pos = y.sum(axis=0).astype(np.float64)
                    neg = float(len(y)) - pos
                    raw = np.sqrt(np.divide(neg, np.maximum(pos, 1.0)))
                    return np.clip(raw, 1.0, cap).astype(np.float32)


                class FamilyConditionedTabM(nn.Module):
                    def __init__(
                        self,
                        *,
                        n_num_features: int,
                        cat_cardinalities: list[int],
                        num_embeddings,
                        target_cols: list[str],
                        family_ids: list[str],
                        target_to_family: dict[str, str],
                        cfg: dict[str, object],
                    ) -> None:
                        super().__init__()
                        self.target_cols = list(target_cols)
                        self.family_ids = list(family_ids)
                        self.target_to_family = dict(target_to_family)
                        self.trunk = TabM.make(
                            n_num_features=n_num_features,
                            cat_cardinalities=cat_cardinalities,
                            d_out=None,
                            num_embeddings=num_embeddings,
                            arch_type=str(cfg["tabm_arch_type"]),
                            k=int(cfg["tabm_k"]),
                            d_block=int(cfg["tabm_d_block"]),
                            n_blocks=int(cfg["tabm_n_blocks"]),
                            dropout=float(cfg["tabm_dropout"]),
                        )
                        d_trunk = int(self.trunk.backbone.get_original_output_shape()[0])
                        family_hidden = int(cfg["family_hidden_dim"])
                        label_hidden = int(cfg["label_head_hidden_dim"])
                        family_dropout = float(cfg["family_dropout"])
                        label_dropout = float(cfg["label_head_dropout"])
                        self.family_heads = nn.ModuleDict({
                            family_id: nn.Sequential(
                                nn.Linear(d_trunk, family_hidden),
                                nn.ReLU(),
                                nn.Dropout(family_dropout),
                                nn.Linear(family_hidden, family_hidden),
                            )
                            for family_id in self.family_ids
                        })
                        self.label_heads = nn.ModuleDict({
                            target_name: nn.Sequential(
                                nn.Linear(d_trunk + family_hidden, label_hidden),
                                nn.ReLU(),
                                nn.Dropout(label_dropout),
                                nn.Linear(label_hidden, 1),
                            )
                            for target_name in self.target_cols
                        })

                    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
                        trunk_h = self.trunk(x_num, x_cat)
                        family_cache = {
                            family_id: head(trunk_h)
                            for family_id, head in self.family_heads.items()
                        }
                        logits = []
                        for target_name in self.target_cols:
                            family_id = self.target_to_family[target_name]
                            head_input = torch.cat([trunk_h, family_cache[family_id]], dim=-1)
                            logits.append(self.label_heads[target_name](head_input))
                        return torch.cat(logits, dim=-1)


                @torch.no_grad()
                def predict_proba(model: nn.Module, loader: DataLoader, device: str) -> np.ndarray:
                    model.eval()
                    outputs = []
                    for x_num, x_cat, _ in loader:
                        x_num = x_num.to(device, non_blocking=device == "cuda")
                        x_cat = x_cat.to(device, non_blocking=device == "cuda").long()
                        logits = model(x_num, x_cat)
                        probs = torch.sigmoid(logits).mean(1)
                        outputs.append(probs.cpu().numpy())
                    return np.concatenate(outputs, axis=0)


                def train_single_seed(
                    train_arrays: dict[str, np.ndarray],
                    val_arrays: dict[str, np.ndarray],
                    target_cols: list[str],
                    cat_cardinalities: list[int],
                    family_ids: list[str],
                    target_to_family: dict[str, str],
                    cfg: dict[str, object],
                    device: str,
                    val_customer_ids: np.ndarray,
                    val_target_df: pd.DataFrame,
                    artifact_dir: Path,
                    seed: int,
                ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
                    set_all_seeds(seed)
                    train_loader = make_loader(train_arrays, int(cfg["batch_size"]), shuffle=True)
                    val_loader = make_loader(val_arrays, int(cfg["eval_batch_size"]), shuffle=False)

                    num_embeddings, embedding_info = make_num_embeddings(train_arrays["x_num"], cfg, seed)
                    print({"stage": "tabm_family_num_embeddings_ready", "seed": seed, **embedding_info}, flush=True)

                    model = FamilyConditionedTabM(
                        n_num_features=train_arrays["x_num"].shape[1],
                        cat_cardinalities=cat_cardinalities,
                        num_embeddings=num_embeddings,
                        target_cols=target_cols,
                        family_ids=family_ids,
                        target_to_family=target_to_family,
                        cfg=cfg,
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(cfg["lr"]),
                        weight_decay=float(cfg["weight_decay"]),
                    )
                    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer,
                        mode="max",
                        factor=0.5,
                        patience=2,
                        min_lr=2e-5,
                    )
                    pos_weight = torch.from_numpy(compute_pos_weight(train_arrays["y"], float(cfg["pos_weight_cap"]))).to(device)

                    history_rows = []
                    best_val = -np.inf
                    best_epoch = 1
                    epochs_without_improvement = 0
                    top_states: list[dict[str, torch.Tensor]] = []
                    top_epochs: list[int] = []
                    top_scores: list[float] = []

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
                            loss = F.binary_cross_entropy_with_logits(
                                logits,
                                y_expanded,
                                pos_weight=pos_weight.view(1, 1, -1),
                            )
                            loss.backward()
                            optimizer.step()
                            losses.append(float(loss.detach().cpu()))

                        val_probs = predict_proba(model, val_loader, device)
                        val_pred_df = pd.DataFrame(val_probs, columns=[c.replace("target_", "predict_") for c in target_cols])
                        val_pred_df.insert(0, "customer_id", val_customer_ids)
                        val_macro = float(macro_auc(val_target_df, val_pred_df, target_cols))

                        history_rows.append({
                            "seed": seed,
                            "epoch": epoch,
                            "train_loss": float(np.mean(losses)),
                            "val_macro_auc": val_macro,
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        })
                        pd.DataFrame(history_rows).to_csv(artifact_dir / "history_running.csv", index=False)

                        print(
                            {
                                "stage": "tabm_family_epoch_done",
                                "seed": seed,
                                "epoch": epoch,
                                "train_loss": float(np.mean(losses)),
                                "val_macro_auc": val_macro,
                                "lr": float(optimizer.param_groups[0]["lr"]),
                            },
                            flush=True,
                        )

                        state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        packed = list(zip(top_scores, top_epochs, top_states))
                        packed.append((val_macro, epoch, state_copy))
                        packed = sorted(packed, key=lambda x: (x[0], x[1]), reverse=True)[: int(cfg["checkpoint_average_top_k"])]
                        top_scores = [float(x[0]) for x in packed]
                        top_epochs = [int(x[1]) for x in packed]
                        top_states = [x[2] for x in packed]

                        if val_macro > best_val:
                            best_val = val_macro
                            best_epoch = epoch
                            val_pred_df.to_parquet(artifact_dir / "best_validation_predictions_running.parquet", index=False)
                            compute_target_scores(val_target_df, val_pred_df, target_cols).to_csv(
                                artifact_dir / "best_target_scores_running.csv",
                                index=False,
                            )
                            epochs_without_improvement = 0
                        else:
                            epochs_without_improvement += 1

                        (artifact_dir / "metrics_running.json").write_text(
                            json.dumps(
                                {
                                    "seed": seed,
                                    "best_single_val_macro_auc": float(best_val),
                                    "best_single_epoch": int(best_epoch),
                                    "top_checkpoint_epochs": top_epochs,
                                    "top_checkpoint_scores": top_scores,
                                    "resolved_mode": RESOLVED_MODE,
                                    "device": device,
                                    "rows_train": int(len(train_arrays["x_num"])),
                                    "rows_val": int(len(val_arrays["x_num"])),
                                    "feature_count": int(train_arrays["x_num"].shape[1] + len(cat_cardinalities)),
                                    "num_features": int(train_arrays["x_num"].shape[1]),
                                    "cat_features": int(len(cat_cardinalities)),
                                    "embedding_info": embedding_info,
                                },
                                indent=2,
                            )
                        )

                        scheduler.step(val_macro)

                        if epoch >= int(cfg["min_epochs"]) and epochs_without_improvement >= int(cfg["early_stopping_patience"]):
                            print(
                                {
                                    "stage": "tabm_family_early_stop",
                                    "seed": seed,
                                    "epoch": epoch,
                                    "best_single_epoch": int(best_epoch),
                                    "best_single_val_macro_auc": float(best_val),
                                    "top_checkpoint_epochs": top_epochs,
                                },
                                flush=True,
                            )
                            break

                    averaged_state = average_state_dicts(top_states if top_states else [state_copy])
                    model.load_state_dict(averaged_state)
                    avg_probs = predict_proba(model, val_loader, device)
                    avg_pred_df = pd.DataFrame(avg_probs, columns=[c.replace("target_", "predict_") for c in target_cols])
                    avg_pred_df.insert(0, "customer_id", val_customer_ids)
                    avg_val_macro = float(macro_auc(val_target_df, avg_pred_df, target_cols))

                    pd.DataFrame(history_rows).to_csv(artifact_dir / "history.csv", index=False)
                    avg_pred_df.to_parquet(artifact_dir / "validation_predictions.parquet", index=False)
                    compute_target_scores(val_target_df, avg_pred_df, target_cols).to_csv(
                        artifact_dir / "target_scores.csv",
                        index=False,
                    )

                    fit_stats = {
                        "seed": seed,
                        "best_single_val_macro_auc": float(best_val),
                        "best_single_epoch": int(best_epoch),
                        "avg_val_macro_auc": float(avg_val_macro),
                        "top_checkpoint_epochs": top_epochs,
                        "top_checkpoint_scores": top_scores,
                        "final_train_epochs": int(max(1, round(float(np.mean(top_epochs if top_epochs else [best_epoch]))))),
                        "embedding_info": embedding_info,
                    }

                    print({"stage": "tabm_family_seed_done", **fit_stats}, flush=True)
                    return avg_pred_df, pd.DataFrame(history_rows), fit_stats


                def train_final_single_seed(
                    full_arrays: dict[str, np.ndarray],
                    test_arrays: dict[str, np.ndarray],
                    target_cols: list[str],
                    cat_cardinalities: list[int],
                    family_ids: list[str],
                    target_to_family: dict[str, str],
                    cfg: dict[str, object],
                    device: str,
                    seed: int,
                    final_epochs: int,
                ) -> np.ndarray:
                    set_all_seeds(seed)
                    full_loader = make_loader(full_arrays, int(cfg["batch_size"]), shuffle=True)
                    test_loader = make_loader(test_arrays, int(cfg["eval_batch_size"]), shuffle=False)

                    num_embeddings, embedding_info = make_num_embeddings(full_arrays["x_num"], cfg, seed)
                    print({"stage": "tabm_family_final_num_embeddings_ready", "seed": seed, **embedding_info}, flush=True)

                    model = FamilyConditionedTabM(
                        n_num_features=full_arrays["x_num"].shape[1],
                        cat_cardinalities=cat_cardinalities,
                        num_embeddings=num_embeddings,
                        target_cols=target_cols,
                        family_ids=family_ids,
                        target_to_family=target_to_family,
                        cfg=cfg,
                    ).to(device)
                    optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=float(cfg["lr"]),
                        weight_decay=float(cfg["weight_decay"]),
                    )
                    pos_weight = torch.from_numpy(compute_pos_weight(full_arrays["y"], float(cfg["pos_weight_cap"]))).to(device)

                    print({"stage": "tabm_family_final_train_start", "seed": seed, "epochs": int(final_epochs), "rows": len(full_arrays["x_num"]), "test_rows": len(test_arrays["x_num"])}, flush=True)
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
                            loss = F.binary_cross_entropy_with_logits(
                                logits,
                                y_expanded,
                                pos_weight=pos_weight.view(1, 1, -1),
                            )
                            loss.backward()
                            optimizer.step()
                            losses.append(float(loss.detach().cpu()))
                        print({"stage": "tabm_family_final_epoch_done", "seed": seed, "epoch": epoch, "train_loss": float(np.mean(losses))}, flush=True)

                    return predict_proba(model, test_loader, device)
                """
            ).strip()
            + "\n",
            6,
        ),
        md_cell("## Load Data\n", 7),
        code_cell(
            dedent(
                """
                BASE_DATA_DIR = find_base_data_dir()
                FEATURE_DATA_DIR = find_feature_data_dir()
                DEVICE = pick_device()

                train_compact = pd.read_parquet(FEATURE_DATA_DIR / "train_compact_features_v1.parquet")
                train_target = pd.read_parquet(BASE_DATA_DIR / "train_target.parquet")
                test_compact = pd.read_parquet(FEATURE_DATA_DIR / "test_compact_features_v1.parquet")
                val_ids = pd.read_parquet(FEATURE_DATA_DIR / "val_ids.parquet")
                current_best_validation_reference = pd.read_parquet(FEATURE_DATA_DIR / "current_best_validation_reference.parquet")

                labeled_df = train_compact.merge(train_target, on="customer_id", how="inner")
                test_df = test_compact.copy()

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
                    train_df, val_df = split_with_train_positive_coverage(
                        df=labeled_df,
                        target_cols=target_cols,
                        val_fraction=0.20,
                        seed=int(CFG["seeds"][0]),
                    )
                    baseline_val_ref = None
                else:
                    val_id_set = set(val_ids["customer_id"].astype("int64").tolist())
                    val_mask = labeled_df["customer_id"].astype("int64").isin(val_id_set)
                    val_df = labeled_df.loc[val_mask].reset_index(drop=True)
                    train_df = labeled_df.loc[~val_mask].reset_index(drop=True)
                    baseline_val_ref = current_best_validation_reference.copy().sort_values("customer_id").reset_index(drop=True)
                    expected_val = sorted(val_df["customer_id"].astype("int64").tolist())
                    got_val = sorted(baseline_val_ref["customer_id"].astype("int64").tolist())
                    if expected_val != got_val:
                        raise RuntimeError("current_best_validation_reference customer_id set does not match exact holdout val_ids")

                family_ids, target_to_family, family_to_targets = build_family_spec(target_cols)

                print({
                    "base_data_dir": str(BASE_DATA_DIR),
                    "feature_data_dir": str(FEATURE_DATA_DIR),
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "test_rows": len(test_df),
                    "feature_count": len(feature_cols),
                    "num_features": len(num_cols),
                    "cat_features": len(cat_cols),
                    "family_count": len(family_ids),
                    "device": DEVICE,
                }, flush=True)
                print({"family_to_targets": family_to_targets}, flush=True)
                """
            ).strip()
            + "\n",
            8,
        ),
        code_cell(
            dedent(
                """
                train_arrays, [val_arrays], cat_cardinalities, varying_num_cols = prepare_arrays(
                    train_df=train_df[feature_cols + target_cols],
                    other_frames=[val_df[feature_cols + target_cols]],
                    num_cols=num_cols,
                    cat_cols=cat_cols,
                    target_cols=target_cols,
                    clip_value=float(CFG["clip_value"]),
                )

                baseline_full_macro = None
                if baseline_val_ref is not None:
                    baseline_full_macro = float(macro_auc(val_df[["customer_id"] + target_cols], baseline_val_ref, target_cols))
                    baseline_target_scores = compute_target_scores(
                        val_df[["customer_id"] + target_cols],
                        baseline_val_ref,
                        target_cols,
                    )
                    baseline_target_scores.to_csv(ARTIFACT_DIR / "baseline_target_scores.csv", index=False)

                print({
                    "stage": "tabm_family_val_prepare_done",
                    "train_rows": len(train_df),
                    "val_rows": len(val_df),
                    "min_train_positive": int(train_df[target_cols].sum().min()),
                    "max_cat_cardinality": int(max(cat_cardinalities)),
                    "num_embedding_type": str(CFG["num_embedding_type"]),
                    "varying_num_features": int(len(varying_num_cols)),
                    "baseline_full_macro": baseline_full_macro,
                }, flush=True)
                """
            ).strip()
            + "\n",
            9,
        ),
        md_cell("## Validation Training\n", 10),
        code_cell(
            dedent(
                """
                seed_histories = []
                seed_metrics = []
                seed_val_frames = []

                for seed in CFG["seeds"]:
                    seed_dir = ARTIFACT_DIR / f"seed_{seed}"
                    seed_dir.mkdir(parents=True, exist_ok=True)
                    val_pred_df, history_df, fit_stats = train_single_seed(
                        train_arrays=train_arrays,
                        val_arrays=val_arrays,
                        target_cols=target_cols,
                        cat_cardinalities=cat_cardinalities,
                        family_ids=family_ids,
                        target_to_family=target_to_family,
                        cfg=CFG,
                        device=DEVICE,
                        val_customer_ids=val_df["customer_id"].values,
                        val_target_df=val_df[["customer_id"] + target_cols].copy(),
                        artifact_dir=seed_dir,
                        seed=int(seed),
                    )
                    seed_histories.append(history_df)
                    seed_metrics.append(fit_stats)
                    seed_val_frames.append(val_pred_df.sort_values("customer_id").reset_index(drop=True))

                    del val_pred_df, history_df
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                avg_val_df = seed_val_frames[0].copy()
                pred_cols = [c for c in avg_val_df.columns if c != "customer_id"]
                for col in pred_cols:
                    avg_val_df[col] = np.mean([frame[col].to_numpy(dtype=np.float32) for frame in seed_val_frames], axis=0).astype(np.float32)

                validation_macro_auc = float(macro_auc(val_df[["customer_id"] + target_cols], avg_val_df, target_cols))
                pd.concat(seed_histories, ignore_index=True).to_csv(ARTIFACT_DIR / "history.csv", index=False)
                avg_val_df.to_parquet(ARTIFACT_DIR / "validation_predictions.parquet", index=False)
                target_scores_df = compute_target_scores(val_df[["customer_id"] + target_cols], avg_val_df, target_cols)
                target_scores_df.to_csv(ARTIFACT_DIR / "target_scores.csv", index=False)
                pd.DataFrame(seed_metrics).to_csv(ARTIFACT_DIR / "seed_metrics.csv", index=False)

                delta_vs_current_best = None if baseline_full_macro is None else float(validation_macro_auc - baseline_full_macro)
                if baseline_val_ref is not None:
                    baseline_scores = compute_target_scores(val_df[["customer_id"] + target_cols], baseline_val_ref, target_cols).rename(columns={"oof_auc": "baseline_auc"})
                    compare_df = target_scores_df.rename(columns={"oof_auc": "family_head_auc"}).merge(baseline_scores, on="target", how="left")
                    compare_df["delta_vs_current_best"] = compare_df["family_head_auc"] - compare_df["baseline_auc"]
                    compare_df = compare_df.sort_values("delta_vs_current_best").reset_index(drop=True)
                    compare_df.to_csv(ARTIFACT_DIR / "target_score_deltas_vs_current_best.csv", index=False)

                metrics_payload = {
                    "validation_macro_auc": validation_macro_auc,
                    "baseline_full_macro_auc": baseline_full_macro,
                    "delta_vs_current_best_validation_reference": delta_vs_current_best,
                    "resolved_mode": RESOLVED_MODE,
                    "device": DEVICE,
                    "rows_train": int(len(train_df)),
                    "rows_val": int(len(val_df)),
                    "feature_count": int(len(feature_cols)),
                    "num_features": int(len(num_cols)),
                    "cat_features": int(len(cat_cols)),
                    "family_count": int(len(family_ids)),
                    "num_embedding_type": str(CFG["num_embedding_type"]),
                    "seed_metrics": seed_metrics,
                }
                (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics_payload, indent=2))
                print(json.dumps(metrics_payload, indent=2), flush=True)
                print(target_scores_df.head(12).to_string(index=False), flush=True)
                """
            ).strip()
            + "\n",
            11,
        ),
        md_cell("## Final Train And Submission\n", 12),
        code_cell(
            dedent(
                """
                if bool(CFG["train_final_model"]):
                    full_arrays, [test_arrays], full_cat_cardinalities, full_varying_num_cols = prepare_arrays(
                        train_df=labeled_df[feature_cols + target_cols],
                        other_frames=[test_df[feature_cols]],
                        num_cols=num_cols,
                        cat_cols=cat_cols,
                        target_cols=target_cols,
                        clip_value=float(CFG["clip_value"]),
                    )

                    test_prob_blocks = []
                    for fit_stats in seed_metrics:
                        seed = int(fit_stats["seed"])
                        final_epochs = int(fit_stats["final_train_epochs"])
                        probs = train_final_single_seed(
                            full_arrays=full_arrays,
                            test_arrays=test_arrays,
                            target_cols=target_cols,
                            cat_cardinalities=full_cat_cardinalities,
                            family_ids=family_ids,
                            target_to_family=target_to_family,
                            cfg=CFG,
                            device=DEVICE,
                            seed=seed,
                            final_epochs=final_epochs,
                        )
                        test_prob_blocks.append(probs)

                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                    test_probs = np.mean(test_prob_blocks, axis=0)
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
            + "\n",
            13,
        ),
        md_cell(
            dedent(
                """
                ## Interpretation

                This run is successful only if the family-conditioned head improves the current exact-holdout reference materially,
                not merely a few isolated targets.
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
    push_notebook = PUSH_DIR / "data-fusion-2026-task-2-tabm-family-head-v2-fs-v1.ipynb"
    shutil.copy2(NOTEBOOK_PATH, push_notebook)
    metadata = {
        "id": "chesnikovleonid/data-fusion-2026-task-2-tabm-family-head-v2-fs-v1",
        "title": "Data Fusion 2026 task 2 Tabm family head v2 fs v1",
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
