from __future__ import annotations

import argparse
import pandas as pd

from lib.feature_selection_v1 import (
    build_composite_feature_scores_df,
    build_global_extra_scores_df,
    ensure_out_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build global and composite extra feature scores.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--feature-inventory-file", default=None)
    parser.add_argument("--lgbm-file", default=None)
    parser.add_argument("--mi-file", default=None)
    parser.add_argument("--weak-file", default=None)
    parser.add_argument("--missingness-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    feature_inventory = pd.read_csv(args.feature_inventory_file or (out_dir / "feature_inventory.csv"))
    lgbm_df = pd.read_csv(args.lgbm_file or (out_dir / "lgbm_gain_per_target.csv"))
    mi_df = pd.read_csv(args.mi_file or (out_dir / "mi_per_target.csv"))
    weak_df = pd.read_csv(args.weak_file or (out_dir / "weak_target_rescue_scores.csv"))
    missingness_df = pd.read_csv(args.missingness_file or (out_dir / "missingness_signal_scores.csv"))

    global_df = build_global_extra_scores_df(lgbm_df, mi_df)
    composite_df = build_composite_feature_scores_df(
        feature_inventory_df=feature_inventory,
        global_scores_df=global_df,
        weak_scores_df=weak_df,
        missingness_df=missingness_df,
    )

    global_path = out_dir / "global_extra_scores.csv"
    composite_path = out_dir / "composite_extra_scores.csv"
    global_df.to_csv(global_path, index=False)
    composite_df.to_csv(composite_path, index=False)
    print({"global_file": str(global_path), "composite_file": str(composite_path), "rows": int(len(composite_df))}, flush=True)


if __name__ == "__main__":
    main()
