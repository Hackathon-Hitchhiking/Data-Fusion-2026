from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from lib.layout import project_root

ROOT = project_root()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, capture_output=True, text=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Kaggle kernel status/log/output into a local snapshot folder.")
    p.add_argument("kernel", help="Kernel slug in form owner/kernel-name")
    p.add_argument("--out-dir", type=Path, default=ROOT / "output" / "kaggle-snapshots")
    p.add_argument("--download-output", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    kernel_name = args.kernel.split("/", 1)[1]
    snapshot_dir = args.out_dir / kernel_name
    log_dir = snapshot_dir / "log"
    output_dir = snapshot_dir / "output"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    status_proc = run(["kaggle", "kernels", "status", args.kernel])
    files_proc = run(["kaggle", "kernels", "files", args.kernel])
    log_proc = run(
        [
            "kaggle",
            "kernels",
            "output",
            args.kernel,
            "--file-pattern",
            r".*\.log$",
            "-p",
            str(log_dir),
            "-o",
        ]
    )

    summary = {
        "kernel": args.kernel,
        "status_stdout": status_proc.stdout.strip(),
        "status_stderr": status_proc.stderr.strip(),
        "files_stdout": files_proc.stdout.strip(),
        "files_stderr": files_proc.stderr.strip(),
        "log_stdout": log_proc.stdout.strip(),
        "log_stderr": log_proc.stderr.strip(),
    }

    if args.download_output:
        out_proc = run(["kaggle", "kernels", "output", args.kernel, "-p", str(output_dir), "-o"])
        summary["output_stdout"] = out_proc.stdout.strip()
        summary["output_stderr"] = out_proc.stderr.strip()

    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
