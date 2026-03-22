from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from lib.layout import project_root


ROOT = project_root()
DEFAULT_WEIGHT_FILE = ROOT / "artifacts" / "adversarial_validation_v1" / "weights_train.parquet"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "catboost_weighted_specialists_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Wrapper for weighted CatBoost specialists.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--weight-file", type=Path, default=DEFAULT_WEIGHT_FILE)
    p.add_argument("--weight-column", type=str, default="weight_norm")
    p.add_argument("--hardest-count", type=int, default=12)
    p.add_argument("--baseline-mode", choices=["gandalf", "g95s05", "g90s10"], default="gandalf")
    p.add_argument("--boosting-type", choices=["Plain", "Ordered"], default="Plain")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--l2-leaf-reg", type=float, default=10.0)
    p.add_argument("--od-wait", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thread-count", type=int, default=4)
    p.add_argument("--used-ram-limit", type=str, default="8gb")
    p.add_argument("--max-ctr-complexity", type=int, default=1)
    p.add_argument("--one-hot-max-size", type=int, default=32)
    p.add_argument("--border-count", type=int, default=64)
    p.add_argument("--alpha-grid", type=str, default="0.05,0.10,0.20,0.30")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "run_catboost_extended_specialists.py"),
        "--out-dir",
        str(args.out_dir),
        "--weight-file",
        str(args.weight_file),
        "--weight-column",
        args.weight_column,
        "--hardest-count",
        str(args.hardest_count),
        "--baseline-mode",
        args.baseline_mode,
        "--boosting-type",
        args.boosting_type,
        "--iterations",
        str(args.iterations),
        "--depth",
        str(args.depth),
        "--learning-rate",
        str(args.learning_rate),
        "--l2-leaf-reg",
        str(args.l2_leaf_reg),
        "--od-wait",
        str(args.od_wait),
        "--seed",
        str(args.seed),
        "--thread-count",
        str(args.thread_count),
        "--used-ram-limit",
        args.used_ram_limit,
        "--max-ctr-complexity",
        str(args.max_ctr_complexity),
        "--one-hot-max-size",
        str(args.one_hot_max_size),
        "--border-count",
        str(args.border_count),
        "--alpha-grid",
        args.alpha_grid,
    ]
    raise SystemExit(subprocess.call(cmd))


if __name__ == "__main__":
    main()
