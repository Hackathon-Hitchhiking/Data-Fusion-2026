from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from lib.layout import resolve_data_dir


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="EDA for Data Fusion 2026 CyberShelf")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/eda"))
    parser.add_argument("--sample-size", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_main = pl.read_parquet(args.data_dir / "train_main_features.parquet")
    target = pl.read_parquet(args.data_dir / "train_target.parquet")

    n_rows = train_main.height
    sample_n = min(args.sample_size, n_rows)
    rng = np.random.default_rng(args.seed)
    sample_idx = np.sort(rng.choice(n_rows, size=sample_n, replace=False))

    train_main_s = train_main[sample_idx]
    target_s = target[sample_idx]

    feature_cols = [c for c in train_main.columns if c != "customer_id"]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c.startswith("num_feature")]
    target_cols = [c for c in target.columns if c.startswith("target_")]

    missing_stats = (
        train_main_s.select(pl.all().null_count()).transpose(include_header=True, column_names=["missing"])
        .rename({"column": "feature"})
        .with_columns((pl.col("missing") / sample_n).alias("missing_ratio"))
        .sort("missing_ratio", descending=True)
    )

    target_rate = (
        target_s.select(pl.exclude("customer_id").mean())
        .transpose(include_header=True, column_names=["positive_rate"])
        .rename({"column": "target"})
        .sort("positive_rate")
    )

    cat_unique = []
    for c in cat_cols:
        unique_n = train_main_s.select(pl.col(c).n_unique()).item()
        cat_unique.append({"feature": c, "n_unique_sample": int(unique_n)})
    cat_unique_df = pd.DataFrame(cat_unique).sort_values("n_unique_sample", ascending=False)

    # pairwise correlation between targets
    tpdf = target_s.select(target_cols).to_pandas()
    corr = tpdf.corr()

    summary = {
        "rows_train": int(train_main.height),
        "rows_target": int(target.height),
        "n_features_main": int(len(feature_cols)),
        "n_cat_features": int(len(cat_cols)),
        "n_num_features": int(len(num_cols)),
        "n_targets": int(len(target_cols)),
        "sample_size": int(sample_n),
        "top_missing_features": missing_stats.head(20).to_dicts(),
        "target_positive_rate": target_rate.to_dicts(),
        "cat_unique_top20": cat_unique_df.head(20).to_dict(orient="records"),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    missing_stats.write_csv(args.out_dir / "missing_stats.csv")
    target_rate.write_csv(args.out_dir / "target_rate.csv")
    corr.to_csv(args.out_dir / "target_corr.csv")

    report = [
        "# EDA report",
        "",
        f"- Train rows: {summary['rows_train']}",
        f"- Main features: {summary['n_features_main']} (cat={summary['n_cat_features']}, num={summary['n_num_features']})",
        f"- Targets: {summary['n_targets']}",
        f"- Sample size used for EDA: {summary['sample_size']}",
        "",
        "## Main takeaways",
        "- In data there are many missing values; keep models robust to NaNs (LGBM/CatBoost handle this natively).",
        "- Multi-label targets are imbalanced, so per-target thresholds and class weights are likely useful.",
        "- Correlations between targets suggest ensembling heterogeneous models (tree boosters + calibrated stacker).",
    ]
    (args.out_dir / "EDA_REPORT.md").write_text("\n".join(report))
    print(f"EDA artifacts saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
