from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from lib.feature_selection_v1 import (
    build_final_feature_set_payload,
    ensure_out_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the final feature set payload/table/summary.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--feature-inventory-file", default=None)
    parser.add_argument("--composite-file", default=None)
    parser.add_argument("--pruned-file", default=None)
    parser.add_argument("--wrapper-file", default=None)
    parser.add_argument("--default-n-final-extra", type=int, default=180)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    feature_inventory = pd.read_csv(args.feature_inventory_file or (out_dir / "feature_inventory.csv"))
    composite_df = pd.read_csv(args.composite_file or (out_dir / "composite_extra_scores.csv"))
    pruned_df = pd.read_csv(args.pruned_file or (out_dir / "redundancy_pruned_extra.csv"))
    wrapper_path = Path(args.wrapper_file) if args.wrapper_file else (out_dir / "wrapper_eval_summary.csv")
    wrapper_df = pd.read_csv(wrapper_path) if wrapper_path.exists() else None
    payload, final_table, summary = build_final_feature_set_payload(
        feature_inventory_df=feature_inventory,
        composite_df=composite_df,
        pruned_df=pruned_df,
        wrapper_eval_df=wrapper_df,
        default_n_final_extra=args.default_n_final_extra,
    )
    (out_dir / "final_feature_set.json").write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    final_table.to_csv(out_dir / "final_feature_table.csv", index=False)
    (out_dir / "selection_summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print({"feature_set_file": str(out_dir / "final_feature_set.json"), "selected_extra": int(len(payload["extra_features"]))}, flush=True)


if __name__ == "__main__":
    main()
