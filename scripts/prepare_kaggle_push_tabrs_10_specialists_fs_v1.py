from __future__ import annotations

import subprocess
from pathlib import Path

from lib.layout import project_root


ROOT = project_root()
PYTHON = ROOT / ".venv" / "bin" / "python"
PUBLISH_INPUTS = ROOT / "scripts" / "publish_tabrs_pilot_inputs.py"
BUILDER = ROOT / "scripts" / "build_gpu_tabrs_10_specialists_fs_v1_notebook.py"


def main() -> None:
    subprocess.run([str(PYTHON), str(PUBLISH_INPUTS)], check=True, cwd=ROOT)
    subprocess.run([str(PYTHON), str(BUILDER)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
