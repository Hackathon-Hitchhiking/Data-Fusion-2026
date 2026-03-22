from __future__ import annotations

import argparse

from lib.feature_selection_v1 import ScanConfig, ensure_out_dir, run_mi_feature_scan_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-target mutual-information scan for extra features.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=4.0)
    parser.add_argument("--max-rows-per-target", type=int, default=120000)
    parser.add_argument("--min-positive-count", type=int, default=50)
    parser.add_argument("--n-neighbors", type=int, default=3)
    parser.add_argument("--feature-block-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    df = run_mi_feature_scan_df(
        data_dir=args.data_dir,
        config=ScanConfig(
            seed=args.seed,
            negative_ratio=args.negative_ratio,
            max_rows_per_target=args.max_rows_per_target,
            min_positive_count=args.min_positive_count,
        ),
        n_neighbors=args.n_neighbors,
        feature_block_size=args.feature_block_size,
    )
    out_path = out_dir / "mi_per_target.csv"
    df.to_csv(out_path, index=False)
    print({"out_file": str(out_path), "rows": int(len(df))}, flush=True)


if __name__ == "__main__":
    main()
