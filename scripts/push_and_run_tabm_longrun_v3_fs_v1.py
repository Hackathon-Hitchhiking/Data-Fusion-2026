from __future__ import annotations

import subprocess
from pathlib import Path

from lib.layout import project_root


ROOT = project_root()
PYTHON = ROOT / ".venv" / "bin" / "python"
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
PREPARE = ROOT / "scripts" / "prepare_kaggle_push_tabm_longrun_v3_fs_v1.py"
PUSH_DIR = ROOT / "output" / "kaggle-push" / "tabm-longrun-v3-fs-v1"
KERNEL_ID = "chesnikovleonid/data-fusion-2026-task-2-tabm-longrun-v3-fs-v1"


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, cwd=ROOT)


def main() -> None:
    run([str(PYTHON), str(PREPARE)])
    run([str(KAGGLE), "kernels", "push", "-p", str(PUSH_DIR)])
    run([str(KAGGLE), "kernels", "status", KERNEL_ID])


if __name__ == "__main__":
    main()
