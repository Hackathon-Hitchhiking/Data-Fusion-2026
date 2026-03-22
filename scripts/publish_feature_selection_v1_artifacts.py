from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from lib.layout import project_root


ROOT = project_root()
PYTHON = ROOT / ".venv" / "bin" / "python"
MANAGER = ROOT / "scripts" / "manage_kaggle_dataset.py"
OUT_DIR = ROOT / "artifacts" / "feature_selection_v1"
REQUIRED_FILES = [
    OUT_DIR / "train_compact_features_v1.parquet",
    OUT_DIR / "test_compact_features_v1.parquet",
    OUT_DIR / "final_feature_set.json",
    OUT_DIR / "final_feature_table.csv",
    OUT_DIR / "selection_summary.json",
]
OPTIONAL_FILES = [
    OUT_DIR / "wrapper_eval_summary.csv",
    OUT_DIR / "wrapper_eval_validation_predictions.parquet",
    OUT_DIR / "composite_extra_scores.csv",
    OUT_DIR / "redundancy_pruned_extra.csv",
    OUT_DIR / "global_extra_scores.csv",
    OUT_DIR / "weak_target_rescue_scores.csv",
    OUT_DIR / "mi_per_target.csv",
    OUT_DIR / "missingness_signal_scores.csv",
    OUT_DIR / "missingness_signal_per_target.csv",
    OUT_DIR / "lgbm_gain_per_target.csv",
    OUT_DIR / "feature_inventory.csv",
    OUT_DIR / "materialized_dataset_summary.json",
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    missing = [str(path) for path in REQUIRED_FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required feature-selection artifacts are missing: {missing}")
    files_to_stage = REQUIRED_FILES + [path for path in OPTIONAL_FILES if path.exists()]

    stage_dir = ROOT / "output" / "kaggle-datasets" / "task-artifacts"
    if stage_dir.exists():
        for path in stage_dir.iterdir():
            if path.name == "dataset-metadata.json":
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

    run([str(PYTHON), str(MANAGER), "init", "--profile", "artifacts_dataset"])
    run([str(PYTHON), str(MANAGER), "add", "--profile", "artifacts_dataset", *map(str, files_to_stage)])
    run(
        [
            str(PYTHON),
            str(MANAGER),
            "push",
            "--profile",
            "artifacts_dataset",
            "--create-if-missing",
            "-m",
            "feature_selection_v1 compact dataset and selection artifacts",
        ]
    )


if __name__ == "__main__":
    main()
