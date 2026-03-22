from __future__ import annotations

import argparse
import pandas as pd

from lib.feature_selection_v1 import WEAKEST_TARGETS, build_weak_target_rescue_scores_df, ensure_out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build weak-target rescue scores for extra features.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--lgbm-file", default=None)
    parser.add_argument("--mi-file", default=None)
    parser.add_argument("--k-gain", type=int, default=40)
    parser.add_argument("--k-mi", type=int, default=40)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    lgbm_file = args.lgbm_file or (out_dir / "lgbm_gain_per_target.csv")
    mi_file = args.mi_file or (out_dir / "mi_per_target.csv")
    lgbm_df = pd.read_csv(lgbm_file)
    mi_df = pd.read_csv(mi_file)
    rescue_df = build_weak_target_rescue_scores_df(
        lgbm_df,
        mi_df,
        weak_targets=WEAKEST_TARGETS,
        k_gain=args.k_gain,
        k_mi=args.k_mi,
    )
    out_path = out_dir / "weak_target_rescue_scores.csv"
    rescue_df.to_csv(out_path, index=False)
    print({"out_file": str(out_path), "rows": int(len(rescue_df))}, flush=True)


if __name__ == "__main__":
    main()
