from __future__ import annotations

import argparse
import pandas as pd

from lib.feature_selection_v1 import ensure_out_dir, run_redundancy_pruning_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prune redundant extra features from the composite shortlist.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--composite-file", default=None)
    parser.add_argument("--preselect-n", type=int, default=300)
    parser.add_argument("--corr-threshold", type=float, default=0.98)
    parser.add_argument("--sample-rows", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    composite_df = pd.read_csv(args.composite_file or (out_dir / "composite_extra_scores.csv"))
    pruned_df = run_redundancy_pruning_df(
        composite_df=composite_df,
        data_dir=args.data_dir,
        preselect_n=args.preselect_n,
        corr_threshold=args.corr_threshold,
        sample_rows=args.sample_rows,
        seed=args.seed,
    )
    out_path = out_dir / "redundancy_pruned_extra.csv"
    pruned_df.to_csv(out_path, index=False)
    print({"out_file": str(out_path), "rows": int(len(pruned_df)), "kept": int(pruned_df["kept_flag"].sum())}, flush=True)


if __name__ == "__main__":
    main()
