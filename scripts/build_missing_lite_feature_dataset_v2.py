from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.layout import project_root


ROOT = project_root()
DEFAULT_IN_DIR = ROOT / "artifacts" / "feature_selection_v2_missing_core"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "feature_selection_v2_missing_lite"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fs_v2_missing_lite from fs_v2_missing_core.")
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = in_dir / "train_compact_features_v2_missing_core.parquet"
    test_path = in_dir / "test_compact_features_v2_missing_core.parquet"
    column_groups_path = in_dir / "column_groups.json"

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    column_groups = json.loads(column_groups_path.read_text())

    base_feature_cols = list(column_groups["base_feature_cols"])
    cat_cols = list(column_groups["cat_cols"])
    num_cols = list(column_groups["num_cols"])
    missing_indicator_cols = list(column_groups["missing_indicator_cols"])
    aggregate_missing_cols = list(column_groups["aggregate_missing_cols"])

    lite_feature_cols = base_feature_cols + missing_indicator_cols + aggregate_missing_cols
    ordered_cols = ["customer_id"] + lite_feature_cols

    train_lite = train_df[ordered_cols].copy()
    test_lite = test_df[ordered_cols].copy()

    train_lite_path = out_dir / "train_compact_features_v2_missing_lite.parquet"
    test_lite_path = out_dir / "test_compact_features_v2_missing_lite.parquet"
    column_groups_lite_path = out_dir / "column_groups_v2_missing_lite.json"
    summary_path = out_dir / "summary_v2_missing_lite.json"

    train_lite.to_parquet(train_lite_path, index=False, compression=args.compression)
    test_lite.to_parquet(test_lite_path, index=False, compression=args.compression)

    lite_groups = {
        "base_feature_cols": base_feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "missing_indicator_cols": missing_indicator_cols,
        "aggregate_missing_cols": aggregate_missing_cols,
        "lite_feature_cols": lite_feature_cols,
        "cluster_names": list(column_groups.get("cluster_names", [])),
    }
    column_groups_lite_path.write_text(json.dumps(lite_groups, ensure_ascii=True, indent=2) + "\n")

    summary = {
        "source_dataset": str(in_dir),
        "out_dir": str(out_dir),
        "train_rows": int(len(train_lite)),
        "test_rows": int(len(test_lite)),
        "base_feature_count": int(len(base_feature_cols)),
        "numeric_feature_count": int(len(num_cols)),
        "categorical_feature_count": int(len(cat_cols)),
        "missing_indicator_count": int(len(missing_indicator_cols)),
        "aggregate_missing_count": int(len(aggregate_missing_cols)),
        "lite_feature_count_excluding_customer_id": int(len(lite_feature_cols)),
        "files": {
            "train_lite": str(train_lite_path),
            "test_lite": str(test_lite_path),
            "column_groups": str(column_groups_lite_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
