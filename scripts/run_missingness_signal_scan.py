from __future__ import annotations

import argparse

from lib.feature_selection_v1 import WEAKEST_TARGETS, ensure_out_dir, run_missingness_signal_scan_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan missingness signal for weakest targets.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    per_target_df, agg_df = run_missingness_signal_scan_df(data_dir=args.data_dir, weak_targets=WEAKEST_TARGETS)
    per_target_path = out_dir / "missingness_signal_per_target.csv"
    agg_path = out_dir / "missingness_signal_scores.csv"
    per_target_df.to_csv(per_target_path, index=False)
    agg_df.to_csv(agg_path, index=False)
    print({"per_target_file": str(per_target_path), "agg_file": str(agg_path), "rows": int(len(agg_df))}, flush=True)


if __name__ == "__main__":
    main()
