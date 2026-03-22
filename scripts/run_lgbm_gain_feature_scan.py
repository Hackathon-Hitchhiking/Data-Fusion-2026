from __future__ import annotations

import argparse

from lib.feature_selection_v1 import ScanConfig, ensure_out_dir, run_lgbm_gain_scan_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-target LightGBM gain scan for extra features.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=4.0)
    parser.add_argument("--max-rows-per-target", type=int, default=120000)
    parser.add_argument("--min-positive-count", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--feature-fraction", type=float, default=0.8)
    parser.add_argument("--bagging-fraction", type=float, default=0.85)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    df = run_lgbm_gain_scan_df(
        data_dir=args.data_dir,
        config=ScanConfig(
            seed=args.seed,
            negative_ratio=args.negative_ratio,
            max_rows_per_target=args.max_rows_per_target,
            min_positive_count=args.min_positive_count,
        ),
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        feature_fraction=args.feature_fraction,
        bagging_fraction=args.bagging_fraction,
        bagging_freq=args.bagging_freq,
        n_jobs=args.n_jobs,
    )
    out_path = out_dir / "lgbm_gain_per_target.csv"
    df.to_csv(out_path, index=False)
    print({"out_file": str(out_path), "rows": int(len(df))}, flush=True)


if __name__ == "__main__":
    main()
