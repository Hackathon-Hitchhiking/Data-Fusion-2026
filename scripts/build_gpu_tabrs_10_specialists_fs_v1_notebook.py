from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from textwrap import dedent

from lib.layout import project_root


ROOT = project_root()
NOTEBOOK_NAME = "data-fusion-2026-gpu-tabrs-10-specialists-fs-v1.ipynb"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / NOTEBOOK_NAME
PUSH_DIR = ROOT / "output" / "kaggle-push" / "tabrs-10-specialists-fs-v1"
KERNEL_METADATA_PATH = PUSH_DIR / "kernel-metadata.json"
NOTEBOOK_REVISION = "tabrs_10_specialists_fs_v1_rev3_20260321"
TABR_COMMIT = "17baa9082506f8e7a0f8d11bb1e08212926a1507"
SPECIALIST_TARGETS = [
    "target_2_8",
    "target_2_3",
    "target_8_3",
    "target_10_1",
    "target_9_7",
    "target_3_1",
    "target_3_3",
    "target_5_1",
    "target_6_2",
    "target_9_3",
]


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
    target_literal = json.dumps(SPECIALIST_TARGETS, ensure_ascii=True)

    bootstrap_lines = [
        "import importlib.metadata",
        "import importlib.util",
        "import json",
        "import os",
        "from pathlib import Path",
        "import shutil",
        "import subprocess",
        "import sys",
        "",
        f'NOTEBOOK_REVISION = "{NOTEBOOK_REVISION}"',
        f'TABR_COMMIT = "{TABR_COMMIT}"',
        'TABR_REPO_URL = "https://github.com/yandex-research/tabular-dl-tabr.git"',
        "",
        "REQUIRED_PACKAGES = [",
        '    ("delu", "delu==0.0.15", "0.0.15"),',
        '    ("faiss", "faiss-cpu", None),',
        '    ("loguru", "loguru", None),',
        '    ("pyarrow", "pyarrow", None),',
        '    ("sklearn", "scikit-learn", None),',
        '    ("tensorboard", "tensorboard", None),',
        '    ("tomli", "tomli", None),',
        '    ("tomli_w", "tomli-w", None),',
        '    ("tqdm", "tqdm", None),',
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
        "def patch_tabr_repo(repo_dir: Path) -> None:",
        '    tabr_path = repo_dir / "bin" / "tabr_scaling.py"',
        "    text = tabr_path.read_text()",
        '    if "TABR_CPU_FAISS_FALLBACK" not in text:',
        '        text = text.replace(',
        '            "import faiss.contrib.torch_utils  # noqa  << this line makes faiss work with PyTorch",',
        '            "try:\\n    import faiss.contrib.torch_utils  # noqa\\nexcept Exception:\\n    pass\\n# TABR_CPU_FAISS_FALLBACK_IMPORT",',
        "        )",
        '        old_block = """        if context_idx is None:',
        "            with torch.no_grad():",
        "                if self.search_index is None:",
        "                    self.search_index = (",
        "                        faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)",
        "                        if device.type == 'cuda'",
        "                        else faiss.IndexFlatL2(d_main)",
        "                    )",
        "                self.search_index.reset()",
        "                self.search_index.add(candidate_k)  # type: ignore[code]",
        "                distances: Tensor",
        "                distances, context_idx = self.search_index.search(  # type: ignore[code]",
        "                    k, context_size + (1 if is_train else 0)",
        "                )",
        "                assert isinstance(context_idx, Tensor)",
        "                if is_train:",
        "                    distances[",
        "                        context_idx == torch.arange(batch_size, device=device)[:, None]",
        "                    ] = torch.inf",
        "                    context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])\"\"\"",
        '        new_block = """        if context_idx is None:',
        "            with torch.no_grad():",
        "                if self.search_index is None:",
        "                    self._search_use_cpu_numpy = False",
        "                    if device.type == 'cuda':",
        "                        try:",
        "                            self.search_index = faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), d_main)",
        "                        except Exception:",
        "                            self.search_index = faiss.IndexFlatL2(d_main)",
        "                            self._search_use_cpu_numpy = True",
        "                    else:",
        "                        self.search_index = faiss.IndexFlatL2(d_main)",
        "                        self._search_use_cpu_numpy = True",
        "                self.search_index.reset()",
        "                distances: Tensor",
        "                if getattr(self, '_search_use_cpu_numpy', False):",
        "                    candidate_np = candidate_k.detach().float().cpu().numpy()",
        "                    query_np = k.detach().float().cpu().numpy()",
        "                    self.search_index.add(candidate_np)",
        "                    distances_np, context_idx_np = self.search_index.search(",
        "                        query_np, context_size + (1 if is_train else 0)",
        "                    )",
        "                    distances = torch.from_numpy(distances_np).to(device=device)",
        "                    context_idx = torch.from_numpy(context_idx_np.astype('int64')).to(device=device)",
        "                else:",
        "                    self.search_index.add(candidate_k)  # type: ignore[code]",
        "                    distances, context_idx = self.search_index.search(  # type: ignore[code]",
        "                        k, context_size + (1 if is_train else 0)",
        "                    )",
        "                    assert isinstance(context_idx, Tensor)",
        "                if is_train:",
        "                    distances[",
        "                        context_idx == torch.arange(batch_size, device=device)[:, None]",
        "                    ] = torch.inf",
        "                    context_idx = context_idx.gather(-1, distances.argsort()[:, :-1])",
        "        # TABR_CPU_FAISS_FALLBACK\"\"\"",
        "        if old_block not in text:",
        '            raise RuntimeError("Could not find the expected FAISS search block in tabr_scaling.py")',
        "        text = text.replace(old_block, new_block)",
        "        tabr_path.write_text(text)",
        "",
        '    metrics_path = repo_dir / "lib" / "metrics.py"',
        "    metrics_text = metrics_path.read_text()",
        '    if "TABR_SAFE_ROC_AUC" not in metrics_text:',
        '        old_metrics = """        if task_type == TaskType.BINCLASS and probs is not None:',
        "            result['roc-auc'] = sklearn.metrics.roc_auc_score(y_true, probs)\"\"\"",
        '        new_metrics = """        if task_type == TaskType.BINCLASS and probs is not None:',
        "            try:",
        "                result['roc-auc'] = sklearn.metrics.roc_auc_score(y_true, probs)",
        "            except ValueError:",
        "                result['roc-auc'] = 0.5",
        "            # TABR_SAFE_ROC_AUC\"\"\"",
        "        if old_metrics not in metrics_text:",
        '            raise RuntimeError("Could not find the expected roc-auc block in metrics.py")',
        "        metrics_text = metrics_text.replace(old_metrics, new_metrics)",
        "        metrics_path.write_text(metrics_text)",
        "",
        "torch_info = torch_runtime_info()",
        "missing = []",
        "missing_no_deps = []",
        "missing_regular = []",
        "for module_name, pip_name, wanted_version in REQUIRED_PACKAGES:",
        "    if importlib.util.find_spec(module_name) is None:",
        "        missing.append(pip_name)",
        '        if module_name == "delu":',
        "            missing_no_deps.append(pip_name)",
        "        else:",
        "            missing_regular.append(pip_name)",
        "        continue",
        "    if wanted_version is not None and package_version(module_name) != wanted_version:",
        "        missing.append(pip_name)",
        '        if module_name == "delu":',
        "            missing_no_deps.append(pip_name)",
        "        else:",
        "            missing_regular.append(pip_name)",
        "",
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
        "if missing_no_deps:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", "--no-deps", *missing_no_deps])',
        "if missing_regular:",
        '    pip_install(["install", "--upgrade", "--no-cache-dir", *missing_regular])',
        "",
        'repo_dir = Path("external") / "tabr_repo"',
        "if repo_dir.exists() and not (repo_dir / '.git').exists():",
        "    shutil.rmtree(repo_dir)",
        "if not repo_dir.exists():",
        "    repo_dir.parent.mkdir(parents=True, exist_ok=True)",
        '    subprocess.check_call(["git", "clone", TABR_REPO_URL, str(repo_dir)])',
        'subprocess.check_call(["git", "-C", str(repo_dir), "fetch", "--depth", "1", "origin", TABR_COMMIT])',
        'subprocess.check_call(["git", "-C", str(repo_dir), "checkout", TABR_COMMIT])',
        "patch_tabr_repo(repo_dir)",
        'print({"stage": "bootstrap_done", "installed_now": missing, "installed_no_deps": missing_no_deps, "installed_regular": missing_regular, "torch_info": torch_info, "repo_dir": str(repo_dir), "commit": TABR_COMMIT}, flush=True)',
    ]

    cells = [
        md_cell(
            dedent(
                """
                # Experiment: Data Fusion 2026 GPU TabR-S Specialists FS v1

                Objective:
                - Run `10` binary one-vs-rest `TabR-S` specialists on the strongest current compact dataset.
                - Compare them against the current best target-wise ensemble reference on the exact same `75k` holdout.
                - Build a restrained hybrid submission by replacing only accepted targets.

                Design:
                - Official `tabr_scaling.py` implementation from the TabR repository.
                - Fixed validation split from the current production holdout (`val_ids.parquet`).
                - Shared prepared feature tensors across all targets to avoid duplicating the `419`-feature dataset on disk.
                - Numerical median imputation + quantile transform, categorical ordinal remapping, and explicit missingness binary block.
                - Acceptance rule:
                  - accept if `delta >= +0.0020` vs current best reference on validation,
                  - mark as near-accept if `delta >= +0.0015`.
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
                - Smoke mode uses only two targets and a much smaller sampled train subset to validate the full code path.
                """
            ).strip()
            + "\n",
            3,
        ),
        code_cell(
            dedent(
                f"""
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
                from sklearn.metrics import roc_auc_score

                RUN_MODE = "auto"  # auto | smoke | full_gpu
                SPECIALIST_TARGETS = {target_literal}

                SMOKE_CFG = {{
                    "seed": 42,
                    "target_names": SPECIALIST_TARGETS[:2],
                    "sample_rows": 50_000,
                    "batch_size": 128,
                    "patience": 4,
                    "max_epochs": 8,
                    "context_size": 32,
                    "freeze_contexts_after_n_epochs": 1,
                    "lr": 3.0e-4,
                    "weight_decay": 1.0e-6,
                    "d_main": 160,
                    "d_multiplier": 2.0,
                    "encoder_n_blocks": 0,
                    "predictor_n_blocks": 1,
                    "mixer_normalization": "auto",
                    "context_dropout": 0.25,
                    "dropout0": 0.25,
                    "dropout1": 0.0,
                    "normalization": "LayerNorm",
                    "activation": "ReLU",
                    "acceptance_delta": 0.0020,
                    "near_acceptance_delta": 0.0015,
                    "add_missing_bin": True,
                }}

                FULL_CFG = {{
                    "seed": 42,
                    "target_names": SPECIALIST_TARGETS,
                    "sample_rows": None,
                    "batch_size": 256,
                    "patience": 16,
                    "max_epochs": 64,
                    "context_size": 96,
                    "freeze_contexts_after_n_epochs": 2,
                    "lr": 3.121273641315169e-4,
                    "weight_decay": 1.2260352006404615e-6,
                    "d_main": 265,
                    "d_multiplier": 2.0,
                    "encoder_n_blocks": 0,
                    "predictor_n_blocks": 1,
                    "mixer_normalization": "auto",
                    "context_dropout": 0.38920071545944357,
                    "dropout0": 0.38852797479169876,
                    "dropout1": 0.0,
                    "normalization": "LayerNorm",
                    "activation": "ReLU",
                    "acceptance_delta": 0.0020,
                    "near_acceptance_delta": 0.0015,
                    "add_missing_bin": True,
                }}


                def resolve_run_mode(mode: str) -> str:
                    if mode != "auto":
                        return mode
                    return "full_gpu" if Path("/kaggle/input").exists() and torch.cuda.is_available() else "smoke"


                RESOLVED_MODE = resolve_run_mode(RUN_MODE)
                CFG = SMOKE_CFG if RESOLVED_MODE == "smoke" else FULL_CFG
                ARTIFACT_DIR = Path("artifacts") / f"gpu_tabrs_10_specialists_fs_v1_{{RESOLVED_MODE}}"
                ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
                random.seed(int(CFG["seed"]))
                np.random.seed(int(CFG["seed"]))
                torch.manual_seed(int(CFG["seed"]))
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(int(CFG["seed"]))

                print(
                    {{
                        "resolved_mode": RESOLVED_MODE,
                        "torch_version": torch.__version__,
                        "cuda_available": bool(torch.cuda.is_available()),
                        "config": CFG,
                        "artifact_dir": str(ARTIFACT_DIR),
                    }},
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
                from sklearn.preprocessing import QuantileTransformer


                def find_base_data_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "data" / "competition",
                        Path.cwd(),
                        Path("/kaggle/input/data-fusion-2026"),
                        Path("/kaggle/input/datasets/chesnikovleonid/data-fusion-2026-2-task-dataset"),
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


                def find_feature_dir() -> Path:
                    direct_candidates = [
                        Path.cwd() / "artifacts" / "feature_selection_v1",
                        Path.cwd() / "artifacts" / "tabrs_pilot_inputs",
                        Path("/kaggle/input/data-fusion-2026-2-task-artifacts"),
                        Path("/kaggle/input/datasets/chesnikovleonid/data-fusion-2026-2-task-artifacts"),
                    ]
                    required = [
                        "train_compact_features_v1.parquet",
                        "test_compact_features_v1.parquet",
                        "final_feature_set.json",
                        "val_ids.parquet",
                        "current_best_validation_reference.parquet",
                        "current_best_submission_reference.parquet",
                    ]
                    for candidate in direct_candidates:
                        if all((candidate / name).exists() for name in required):
                            return candidate
                    kaggle_input = Path("/kaggle/input")
                    if kaggle_input.exists():
                        for match in kaggle_input.rglob("current_best_validation_reference.parquet"):
                            candidate = match.parent
                            if all((candidate / name).exists() for name in required):
                                return candidate
                    raise FileNotFoundError("Could not locate the feature-selection v1 directory with TabR-S pilot inputs.")


                def target_to_pred(target_name: str) -> str:
                    return "predict_" + target_name.split("target_", 1)[1]


                def sigmoid(z: np.ndarray) -> np.ndarray:
                    z = np.asarray(z, dtype=np.float64)
                    return 1.0 / (1.0 + np.exp(-z))


                def safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
                    y_true = np.asarray(y_true, dtype=np.int8)
                    y_pred = np.asarray(y_pred, dtype=np.float64)
                    return float(roc_auc_score(y_true, y_pred)) if np.unique(y_true).size > 1 else 0.5


                def macro_auc(target_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
                    merged = target_df[["customer_id"] + target_cols].merge(pred_df, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
                    values = []
                    for target_name in target_cols:
                        pred_col = target_to_pred(target_name)
                        values.append(safe_auc(merged[target_name].to_numpy(dtype=np.int8), merged[pred_col].to_numpy(dtype=np.float64)))
                    return float(np.mean(values))


                def hardlink_or_copy(src: Path, dst: Path) -> None:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if dst.exists():
                        dst.unlink()
                    try:
                        os.link(src, dst)
                    except OSError:
                        shutil.copy2(src, dst)


                def write_info_json(path: Path) -> None:
                    path.write_text(json.dumps({"task_type": "binclass"}, ensure_ascii=True, indent=2) + "\\n")


                def sample_train_rows(train_df: pd.DataFrame, target_cols: list[str], sample_rows: int, seed: int) -> pd.DataFrame:
                    if sample_rows is None or sample_rows >= len(train_df):
                        return train_df.sort_values("customer_id").reset_index(drop=True)
                    rng = np.random.default_rng(seed)
                    positive_mask = train_df[target_cols].sum(axis=1).to_numpy(dtype=np.int16) > 0
                    positive_idx = np.flatnonzero(positive_mask)
                    negative_idx = np.flatnonzero(~positive_mask)
                    n_positive = min(len(positive_idx), max(len(target_cols) * 256, int(sample_rows * 0.25)))
                    n_negative = max(0, sample_rows - n_positive)
                    selected = []
                    if n_positive > 0:
                        selected.append(rng.choice(positive_idx, size=n_positive, replace=False))
                    if n_negative > 0:
                        selected.append(rng.choice(negative_idx, size=n_negative, replace=False))
                    sampled_idx = np.concatenate(selected) if selected else rng.choice(np.arange(len(train_df)), size=sample_rows, replace=False)
                    sampled = train_df.iloc[np.unique(sampled_idx)].copy()
                    return sampled.sort_values("customer_id").reset_index(drop=True)


                def encode_categorical_frames(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, cat_cols: list[str]) -> dict[str, np.ndarray]:
                    if not cat_cols:
                        return {}
                    out = {
                        "train": np.empty((len(train_df), len(cat_cols)), dtype=np.int64),
                        "val": np.empty((len(val_df), len(cat_cols)), dtype=np.int64),
                        "test": np.empty((len(test_df), len(cat_cols)), dtype=np.int64),
                    }
                    for idx, col in enumerate(cat_cols):
                        train_raw = train_df[col].fillna(-1).round().astype("int64")
                        values = sorted(int(x) for x in pd.Index(train_raw).unique().tolist())
                        mapping = {value: mapped for mapped, value in enumerate(values)}
                        unseen_value = len(mapping)

                        for part_name, frame in [("train", train_df), ("val", val_df), ("test", test_df)]:
                            raw = frame[col].fillna(-1).round().astype("int64")
                            mapped = raw.map(mapping).fillna(unseen_value).astype("int64").to_numpy()
                            out[part_name][:, idx] = mapped
                    return out


                def prepare_numeric_and_missing_bin(
                    train_df: pd.DataFrame,
                    val_df: pd.DataFrame,
                    test_df: pd.DataFrame,
                    num_cols: list[str],
                    seed: int,
                    add_missing_bin: bool,
                ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
                    train_num = train_df[num_cols].to_numpy(dtype=np.float64)
                    val_num = val_df[num_cols].to_numpy(dtype=np.float64)
                    test_num = test_df[num_cols].to_numpy(dtype=np.float64)

                    train_missing = np.isnan(train_num)
                    val_missing = np.isnan(val_num)
                    test_missing = np.isnan(test_num)

                    medians = np.nanmedian(train_num, axis=0)
                    medians = np.where(np.isfinite(medians), medians, 0.0)

                    train_filled = np.where(train_missing, medians, train_num)
                    val_filled = np.where(val_missing, medians, val_num)
                    test_filled = np.where(test_missing, medians, test_num)

                    n_quantiles = max(min(train_filled.shape[0] // 30, 1000), 10)
                    qt = QuantileTransformer(
                        output_distribution="normal",
                        n_quantiles=n_quantiles,
                        subsample=1_000_000_000,
                        random_state=int(seed),
                    )
                    stds = np.std(train_filled, axis=0, keepdims=True)
                    noise_std = 1e-3 / np.maximum(stds, 1e-3)
                    rng = np.random.default_rng(int(seed))
                    train_for_fit = train_filled + noise_std * rng.standard_normal(train_filled.shape)
                    qt.fit(train_for_fit)

                    num_arrays = {
                        "train": qt.transform(train_filled).astype(np.float32),
                        "val": qt.transform(val_filled).astype(np.float32),
                        "test": qt.transform(test_filled).astype(np.float32),
                    }
                    bin_arrays = (
                        {
                            "train": train_missing.astype(np.float32),
                            "val": val_missing.astype(np.float32),
                            "test": test_missing.astype(np.float32),
                        }
                        if add_missing_bin
                        else {}
                    )
                    return num_arrays, bin_arrays
                """
            ).strip()
            + "\n",
            6,
        ),
        md_cell("## Data Preparation\n", 7),
        code_cell(
            dedent(
                """
                BASE_DATA_DIR = find_base_data_dir()
                FEATURE_DIR = find_feature_dir()

                train_target = pd.read_parquet(BASE_DATA_DIR / "train_target.parquet").sort_values("customer_id").reset_index(drop=True)
                train_feat = pd.read_parquet(FEATURE_DIR / "train_compact_features_v1.parquet").sort_values("customer_id").reset_index(drop=True)
                test_feat = pd.read_parquet(FEATURE_DIR / "test_compact_features_v1.parquet").sort_values("customer_id").reset_index(drop=True)
                val_ids = pd.read_parquet(FEATURE_DIR / "val_ids.parquet").sort_values("customer_id").reset_index(drop=True)
                baseline_val_ref = pd.read_parquet(FEATURE_DIR / "current_best_validation_reference.parquet").sort_values("customer_id").reset_index(drop=True)
                baseline_submit_ref = pd.read_parquet(FEATURE_DIR / "current_best_submission_reference.parquet").sort_values("customer_id").reset_index(drop=True)

                target_cols = [c for c in train_target.columns if c != "customer_id"]
                feature_cols = [c for c in train_feat.columns if c != "customer_id"]
                cat_cols = [c for c in feature_cols if c.startswith("cat_feature_")]
                num_cols = [c for c in feature_cols if c.startswith("num_feature_")]
                selected_targets = list(CFG["target_names"])

                labeled = train_feat.merge(train_target, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
                val_id_set = set(int(x) for x in val_ids["customer_id"].tolist())
                val_df = labeled[labeled["customer_id"].isin(val_id_set)].copy().sort_values("customer_id").reset_index(drop=True)
                train_df = labeled[~labeled["customer_id"].isin(val_id_set)].copy().sort_values("customer_id").reset_index(drop=True)

                if CFG["sample_rows"] is not None:
                    train_df = sample_train_rows(train_df, selected_targets, int(CFG["sample_rows"]), int(CFG["seed"]))

                baseline_val_ref = baseline_val_ref.merge(val_df[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
                baseline_submit_ref = baseline_submit_ref.sort_values("customer_id").reset_index(drop=True)
                test_df = test_feat.copy().sort_values("customer_id").reset_index(drop=True)

                if not np.array_equal(val_df["customer_id"].to_numpy(), baseline_val_ref["customer_id"].to_numpy()):
                    raise RuntimeError("Baseline validation reference does not align with the fixed holdout ids.")
                if not np.array_equal(test_df["customer_id"].to_numpy(), baseline_submit_ref["customer_id"].to_numpy()):
                    raise RuntimeError("Baseline submission reference does not align with test customer_id ordering.")

                processed_dir = ARTIFACT_DIR / "prepared_data"
                shared_dir = processed_dir / "shared"
                shared_dir.mkdir(parents=True, exist_ok=True)

                cat_arrays = encode_categorical_frames(train_df, val_df, test_df, cat_cols)
                num_arrays, bin_arrays = prepare_numeric_and_missing_bin(
                    train_df,
                    val_df,
                    test_df,
                    num_cols,
                    seed=int(CFG["seed"]),
                    add_missing_bin=bool(CFG["add_missing_bin"]),
                )

                np.save(shared_dir / "X_num_train.npy", num_arrays["train"])
                np.save(shared_dir / "X_num_val.npy", num_arrays["val"])
                np.save(shared_dir / "X_num_test.npy", num_arrays["test"])
                if cat_arrays:
                    np.save(shared_dir / "X_cat_train.npy", cat_arrays["train"])
                    np.save(shared_dir / "X_cat_val.npy", cat_arrays["val"])
                    np.save(shared_dir / "X_cat_test.npy", cat_arrays["test"])
                if bin_arrays:
                    np.save(shared_dir / "X_bin_train.npy", bin_arrays["train"])
                    np.save(shared_dir / "X_bin_val.npy", bin_arrays["val"])
                    np.save(shared_dir / "X_bin_test.npy", bin_arrays["test"])

                dataset_root = processed_dir / "datasets"
                dataset_root.mkdir(parents=True, exist_ok=True)
                shared_files = [p for p in shared_dir.glob("*.npy")]
                zero_test = np.zeros(len(test_df), dtype=np.int64)

                dataset_dirs = {}
                for target_name in selected_targets:
                    target_dir = dataset_root / target_name
                    target_dir.mkdir(parents=True, exist_ok=True)
                    for src in shared_files:
                        hardlink_or_copy(src, target_dir / src.name)
                    np.save(target_dir / "Y_train.npy", train_df[target_name].to_numpy(dtype=np.int64))
                    np.save(target_dir / "Y_val.npy", val_df[target_name].to_numpy(dtype=np.int64))
                    np.save(target_dir / "Y_test.npy", zero_test)
                    write_info_json(target_dir / "info.json")
                    dataset_dirs[target_name] = target_dir

                prep_summary = {
                    "base_data_dir": str(BASE_DATA_DIR),
                    "feature_dir": str(FEATURE_DIR),
                    "train_rows": int(len(train_df)),
                    "val_rows": int(len(val_df)),
                    "test_rows": int(len(test_df)),
                    "feature_count": int(len(feature_cols)),
                    "num_features": int(len(num_cols)),
                    "cat_features": int(len(cat_cols)),
                    "bin_features": int(bin_arrays["train"].shape[1]) if bin_arrays else 0,
                    "target_count": int(len(selected_targets)),
                    "targets": selected_targets,
                }
                (ARTIFACT_DIR / "prepared_data_summary.json").write_text(json.dumps(prep_summary, ensure_ascii=True, indent=2) + "\\n")
                print(prep_summary, flush=True)
                """
            ).strip()
            + "\n",
            8,
        ),
        md_cell("## Train Specialists\n", 9),
        code_cell(
            dedent(
                """
                import sys

                TABR_REPO_DIR = Path("external") / "tabr_repo"
                os.environ["PROJECT_DIR"] = str(TABR_REPO_DIR.resolve())
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
                if str(TABR_REPO_DIR.resolve()) not in sys.path:
                    sys.path.insert(0, str(TABR_REPO_DIR.resolve()))

                import lib
                from bin import tabr_scaling

                lib.configure_libraries()

                pred_cols = [target_to_pred(x) for x in target_cols]
                baseline_full_macro = macro_auc(val_df[["customer_id"] + target_cols], baseline_val_ref[["customer_id"] + pred_cols], target_cols)

                specialists_val = pd.DataFrame({"customer_id": val_df["customer_id"].astype("int32").values})
                specialists_test = pd.DataFrame({"customer_id": test_df["customer_id"].astype("int32").values})
                results = []
                runs_root = ARTIFACT_DIR / "runs"
                runs_root.mkdir(parents=True, exist_ok=True)

                for target_name in selected_targets:
                    pred_col = target_to_pred(target_name)
                    run_dir = runs_root / target_name
                    start_time = time.time()
                    status = "ok"
                    error_text = ""
                    try:
                        config = {
                            "seed": int(CFG["seed"]),
                            "data": {
                                "path": str(dataset_dirs[target_name]),
                                "num_policy": None,
                                "cat_policy": None,
                                "y_policy": None,
                                "score": "roc-auc",
                                "seed": int(CFG["seed"]),
                                "cache": False,
                            },
                            "model": {
                                "num_embeddings": None,
                                "d_main": int(CFG["d_main"]),
                                "d_multiplier": float(CFG["d_multiplier"]),
                                "encoder_n_blocks": int(CFG["encoder_n_blocks"]),
                                "predictor_n_blocks": int(CFG["predictor_n_blocks"]),
                                "mixer_normalization": CFG["mixer_normalization"],
                                "context_dropout": float(CFG["context_dropout"]),
                                "dropout0": float(CFG["dropout0"]),
                                "dropout1": float(CFG["dropout1"]),
                                "normalization": CFG["normalization"],
                                "activation": CFG["activation"],
                            },
                            "context_size": int(CFG["context_size"]),
                            "optimizer": {
                                "type": "AdamW",
                                "lr": float(CFG["lr"]),
                                "weight_decay": float(CFG["weight_decay"]),
                            },
                            "batch_size": int(CFG["batch_size"]),
                            "patience": int(CFG["patience"]),
                            "n_epochs": int(CFG["max_epochs"]),
                            "freeze_contexts_after_n_epochs": int(CFG["freeze_contexts_after_n_epochs"]),
                        }
                        report = tabr_scaling.main(config, run_dir, force=True)
                        if report is None:
                            raise RuntimeError("tabr_scaling.main returned None")
                        preds = lib.load_predictions(run_dir)
                        val_pred = sigmoid(preds["val"].reshape(-1))
                        test_pred = sigmoid(preds["test"].reshape(-1))

                        specialists_val[pred_col] = val_pred.astype(np.float64)
                        specialists_test[pred_col] = test_pred.astype(np.float64)

                        baseline_auc = safe_auc(
                            val_df[target_name].to_numpy(dtype=np.int8),
                            baseline_val_ref[pred_col].to_numpy(dtype=np.float64),
                        )
                        tabr_auc = safe_auc(
                            val_df[target_name].to_numpy(dtype=np.int8),
                            val_pred,
                        )
                        delta = tabr_auc - baseline_auc
                        accepted = bool(delta >= float(CFG["acceptance_delta"]))
                        near_accept = bool((not accepted) and delta >= float(CFG["near_acceptance_delta"]))

                        result_row = {
                            "target": target_name,
                            "status": status,
                            "baseline_auc": baseline_auc,
                            "tabrs_auc": tabr_auc,
                            "delta": delta,
                            "accepted": accepted,
                            "near_accept": near_accept,
                            "best_epoch": int(report.get("best_epoch", -1)),
                            "val_score_reported": float(report["metrics"]["val"]["score"]),
                            "fit_seconds": float(time.time() - start_time),
                            "run_dir": str(run_dir),
                        }
                        results.append(result_row)
                        print({"stage": "target_done", **result_row}, flush=True)
                    except Exception as err:
                        status = "error"
                        error_text = str(err)
                        results.append(
                            {
                                "target": target_name,
                                "status": status,
                                "baseline_auc": float("nan"),
                                "tabrs_auc": float("nan"),
                                "delta": float("nan"),
                                "accepted": False,
                                "near_accept": False,
                                "best_epoch": -1,
                                "val_score_reported": float("nan"),
                                "fit_seconds": float(time.time() - start_time),
                                "run_dir": str(run_dir),
                                "error": error_text[:500],
                            }
                        )
                        print({"stage": "target_failed", "target": target_name, "error": error_text[:500]}, flush=True)
                    finally:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                results_df = pd.DataFrame(results).sort_values(["accepted", "delta"], ascending=[False, False]).reset_index(drop=True)
                results_df.to_csv(ARTIFACT_DIR / "target_results.csv", index=False)

                accepted_targets = results_df.loc[results_df["accepted"], "target"].tolist()
                near_accept_targets = results_df.loc[results_df["near_accept"], "target"].tolist()
                accepted_pred_cols = [target_to_pred(x) for x in accepted_targets]
                all_specialist_cols = [c for c in specialists_val.columns if c != "customer_id"]

                accepted_hybrid_val = baseline_val_ref.copy()
                accepted_hybrid_test = baseline_submit_ref.copy()
                for pred_col in accepted_pred_cols:
                    accepted_hybrid_val[pred_col] = specialists_val[pred_col].astype(np.float64)
                    accepted_hybrid_test[pred_col] = specialists_test[pred_col].astype(np.float64)

                all_specialists_val = baseline_val_ref.copy()
                all_specialists_test = baseline_submit_ref.copy()
                for pred_col in all_specialist_cols:
                    all_specialists_val[pred_col] = specialists_val[pred_col].astype(np.float64)
                    all_specialists_test[pred_col] = specialists_test[pred_col].astype(np.float64)

                accepted_hybrid_macro = macro_auc(val_df[["customer_id"] + target_cols], accepted_hybrid_val[["customer_id"] + pred_cols], target_cols)
                all_specialists_macro = macro_auc(val_df[["customer_id"] + target_cols], all_specialists_val[["customer_id"] + pred_cols], target_cols)
                shortlist_baseline_macro = macro_auc(
                    val_df[["customer_id"] + selected_targets],
                    baseline_val_ref[["customer_id"] + [target_to_pred(x) for x in selected_targets]],
                    selected_targets,
                )
                shortlist_all_specialists_macro = macro_auc(
                    val_df[["customer_id"] + selected_targets],
                    all_specialists_val[["customer_id"] + [target_to_pred(x) for x in selected_targets]],
                    selected_targets,
                )

                specialists_val.to_parquet(ARTIFACT_DIR / "tabrs_specialists_validation_predictions.parquet", index=False)
                specialists_test.to_parquet(ARTIFACT_DIR / "tabrs_specialists_test_predictions.parquet", index=False)
                accepted_hybrid_val.to_parquet(ARTIFACT_DIR / "accepted_hybrid_validation_predictions.parquet", index=False)
                accepted_hybrid_test.to_parquet(ARTIFACT_DIR / "accepted_hybrid_submission.parquet", index=False)
                all_specialists_val.to_parquet(ARTIFACT_DIR / "all_specialists_validation_predictions.parquet", index=False)
                all_specialists_test.to_parquet(ARTIFACT_DIR / "all_specialists_submission.parquet", index=False)

                summary = {
                    "validation_macro_auc_baseline_full": baseline_full_macro,
                    "validation_macro_auc_accepted_hybrid": accepted_hybrid_macro,
                    "validation_macro_auc_all_specialists_overlay": all_specialists_macro,
                    "shortlist_macro_auc_baseline": shortlist_baseline_macro,
                    "shortlist_macro_auc_all_specialists_overlay": shortlist_all_specialists_macro,
                    "target_count": len(selected_targets),
                    "successful_targets": int((results_df["status"] == "ok").sum()),
                    "accepted_targets": accepted_targets,
                    "near_accept_targets": near_accept_targets,
                    "failed_targets": results_df.loc[results_df["status"] != "ok", "target"].tolist(),
                }
                (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\\n")
                (ARTIFACT_DIR / "accepted_targets.json").write_text(json.dumps({"accepted_targets": accepted_targets, "near_accept_targets": near_accept_targets}, ensure_ascii=True, indent=2) + "\\n")
                print(summary, flush=True)
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
    NOTEBOOK_PATH.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n")

    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    push_notebook = PUSH_DIR / "data-fusion-2026-task-2-tabrs-10-specialists-fs-v1.ipynb"
    shutil.copy2(NOTEBOOK_PATH, push_notebook)
    metadata = {
        "id": "chesnikovleonid/data-fusion-2026-task-2-tabrs-10-specialists-fs-v1",
        "title": "Data Fusion 2026 task 2 Tabrs 10 specialists fs v1",
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
