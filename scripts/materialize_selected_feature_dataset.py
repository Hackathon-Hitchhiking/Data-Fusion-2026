from __future__ import annotations

import argparse
import json
from pathlib import Path

from lib.feature_selection_v1 import ensure_out_dir, materialize_selected_feature_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize compact train/test datasets from the final feature set.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--feature-set-file", default=None)
    parser.add_argument("--train-file-name", default="train_compact_features_v1.parquet")
    parser.add_argument("--test-file-name", default="test_compact_features_v1.parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    feature_set_file = Path(args.feature_set_file or (out_dir / "final_feature_set.json"))
    payload = json.loads(feature_set_file.read_text())
    summary = materialize_selected_feature_dataset(
        final_feature_set=payload,
        out_dir=out_dir,
        data_dir=args.data_dir,
        train_file_name=args.train_file_name,
        test_file_name=args.test_file_name,
    )
    (out_dir / "materialized_dataset_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
