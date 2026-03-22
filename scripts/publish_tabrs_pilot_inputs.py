from __future__ import annotations

import subprocess
from pathlib import Path

from lib.layout import project_root


ROOT = project_root()
PYTHON = ROOT / ".venv" / "bin" / "python"
MANAGER = ROOT / "scripts" / "manage_kaggle_dataset.py"
BUILD_REFERENCES = ROOT / "scripts" / "build_tabrs_pilot_reference_artifacts.py"
ARTIFACT_DIR = ROOT / "artifacts" / "tabrs_pilot_inputs"
FILES = [
    ARTIFACT_DIR / "current_best_validation_reference.parquet",
    ARTIFACT_DIR / "current_best_submission_reference.parquet",
    ARTIFACT_DIR / "val_ids.parquet",
    ARTIFACT_DIR / "summary.json",
]


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    run([str(PYTHON), str(BUILD_REFERENCES)])
    missing = [str(path) for path in FILES if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing TabR-S pilot input files: {missing}")

    run([str(PYTHON), str(MANAGER), "init", "--profile", "artifacts_dataset"])
    run([str(PYTHON), str(MANAGER), "add", "--profile", "artifacts_dataset", *map(str, FILES)])
    run(
        [
            str(PYTHON),
            str(MANAGER),
            "push",
            "--profile",
            "artifacts_dataset",
            "-m",
            "add TabR-S pilot references and fixed validation ids",
        ]
    )


if __name__ == "__main__":
    main()
