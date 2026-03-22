from __future__ import annotations

import argparse
import pandas as pd

from lib.feature_selection_v1 import ScanConfig, ensure_out_dir, run_wrapper_eval_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cheap wrapper evaluation for candidate final feature set sizes.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--composite-file", default=None)
    parser.add_argument("--pruned-file", default=None)
    parser.add_argument("--candidate-sizes", default="100,140,180,220,260")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--negative-ratio", type=float, default=4.0)
    parser.add_argument("--max-rows-per-target", type=int, default=120000)
    parser.add_argument("--min-positive-count", type=int, default=50)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=64)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    composite_df = pd.read_csv(args.composite_file or (out_dir / "composite_extra_scores.csv"))
    pruned_df = pd.read_csv(args.pruned_file or (out_dir / "redundancy_pruned_extra.csv"))
    kept = pruned_df.loc[pruned_df["kept_flag"] == 1, "feature_name"].tolist()
    ranked = composite_df[composite_df["feature_name"].isin(kept)].sort_values("composite_score", ascending=False)
    extra_features = ranked["feature_name"].tolist()
    candidate_sizes = [int(x.strip()) for x in args.candidate_sizes.split(",") if x.strip()]
    config = ScanConfig(
        seed=args.seed,
        negative_ratio=args.negative_ratio,
        max_rows_per_target=args.max_rows_per_target,
        min_positive_count=args.min_positive_count,
    )
    summary_df, pred_df = run_wrapper_eval_df(
        selected_extra_features=extra_features,
        candidate_sizes=candidate_sizes,
        data_dir=args.data_dir,
        config=config,
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        n_jobs=args.n_jobs,
    )
    summary_path = out_dir / "wrapper_eval_summary.csv"
    pred_path = out_dir / "wrapper_eval_validation_predictions.parquet"
    summary_df.to_csv(summary_path, index=False)
    pred_df.to_parquet(pred_path, index=False)
    print({"summary_file": str(summary_path), "pred_file": str(pred_path), "best": summary_df.iloc[0].to_dict() if not summary_df.empty else {}}, flush=True)


if __name__ == "__main__":
    main()
