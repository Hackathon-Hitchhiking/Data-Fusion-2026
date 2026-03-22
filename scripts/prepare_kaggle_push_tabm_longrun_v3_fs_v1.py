from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from lib.layout import project_root


ROOT = project_root()
BUILDER = ROOT / "scripts" / "build_gpu_tabm_longrun_v3_fs_v1_notebook.py"
NOTEBOOK_PATH = ROOT / "output" / "jupyter-notebook" / "data-fusion-2026-gpu-tabm-longrun-v3-fs-v1.ipynb"
PUSH_DIR = ROOT / "output" / "kaggle-push" / "tabm-longrun-v3-fs-v1"
PUSH_NOTEBOOK = PUSH_DIR / "data-fusion-2026-task-2-tabm-longrun-v3-fs-v1.ipynb"
METADATA_PATH = PUSH_DIR / "kernel-metadata.json"


def main() -> None:
    subprocess.run(["./.venv/bin/python", str(BUILDER)], check=True, cwd=ROOT)
    PUSH_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(NOTEBOOK_PATH, PUSH_NOTEBOOK)

    metadata = {
        "id": "chesnikovleonid/data-fusion-2026-task-2-tabm-longrun-v3-fs-v1",
        "title": "Data Fusion 2026 task 2 TabM longrun v3 fs v1",
        "code_file": PUSH_NOTEBOOK.name,
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
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=True, indent=2) + "\n")
    print(
        {
            "push_dir": str(PUSH_DIR),
            "notebook": str(PUSH_NOTEBOOK),
            "metadata": str(METADATA_PATH),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
