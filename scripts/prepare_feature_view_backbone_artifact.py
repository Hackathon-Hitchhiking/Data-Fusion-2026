from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.layout import project_root


ROOT = project_root()
DEFAULT_VIEW_SCORES = ROOT / "artifacts/feature_view_augmentation_v2/per_target_view_scores.csv"
DEFAULT_VIEW_SUMMARY = ROOT / "artifacts/feature_view_augmentation_v2/feature_view_ablation.csv"
DEFAULT_FEATURE_SUMMARY = ROOT / "artifacts/dataset_feature_diagnostics_v1/feature_summary.csv"
DEFAULT_TARGET_SUMMARY = ROOT / "artifacts/dataset_feature_diagnostics_v1/target_summary.csv"
DEFAULT_NUMERIC_PARTITION = ROOT / "artifacts/dataset_feature_diagnostics_v1/numeric_feature_partition.csv"
DEFAULT_WEAKEST = ROOT / "artifacts/weakest_targets_by_best_auc_v2_longrun.csv"
DEFAULT_TOP_FEATURES = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
DEFAULT_OUT = ROOT / "artifacts/feature_view_backbone_prep_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare reusable feature-view artifacts for a future backbone.")
    p.add_argument("--view-scores", type=Path, default=DEFAULT_VIEW_SCORES)
    p.add_argument("--view-summary", type=Path, default=DEFAULT_VIEW_SUMMARY)
    p.add_argument("--feature-summary", type=Path, default=DEFAULT_FEATURE_SUMMARY)
    p.add_argument("--target-summary", type=Path, default=DEFAULT_TARGET_SUMMARY)
    p.add_argument("--numeric-partition", type=Path, default=DEFAULT_NUMERIC_PARTITION)
    p.add_argument("--weakest", type=Path, default=DEFAULT_WEAKEST)
    p.add_argument("--top-features", type=Path, default=DEFAULT_TOP_FEATURES)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--top-k-features", type=int, default=24)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    view_scores = pd.read_csv(args.view_scores)
    view_summary = pd.read_csv(args.view_summary)
    feature_summary = pd.read_csv(args.feature_summary)
    target_summary = pd.read_csv(args.target_summary)
    numeric_partition = pd.read_csv(args.numeric_partition)
    weakest = pd.read_csv(args.weakest).reset_index().rename(columns={"index": "weak_rank"})
    weakest["weak_rank"] = weakest["weak_rank"] + 1
    top_features = json.loads(args.top_features.read_text())[: args.top_k_features]

    candidate_views = ["raw_plus_quantile", "raw_plus_clipped", "quantile_only", "clipped_only"]
    filtered = view_scores[view_scores["view_type"].isin(candidate_views)].copy()
    best_view = filtered.sort_values(["target", "delta_vs_raw"], ascending=[True, False]).groupby("target", as_index=False).first()
    best_view = best_view.rename(
        columns={
            "view_type": "best_view",
            "auc": "best_view_auc",
            "delta_vs_raw": "best_view_delta_vs_raw",
        }
    )

    quant = filtered[filtered["view_type"] == "raw_plus_quantile"][["target", "auc", "delta_vs_raw"]].rename(
        columns={"auc": "raw_plus_quantile_auc", "delta_vs_raw": "raw_plus_quantile_delta"}
    )
    clip = filtered[filtered["view_type"] == "raw_plus_clipped"][["target", "auc", "delta_vs_raw"]].rename(
        columns={"auc": "raw_plus_clipped_auc", "delta_vs_raw": "raw_plus_clipped_delta"}
    )
    clipped_only = filtered[filtered["view_type"] == "clipped_only"][["target", "auc", "delta_vs_raw"]].rename(
        columns={"auc": "clipped_only_auc", "delta_vs_raw": "clipped_only_delta"}
    )

    target_table = (
        best_view.merge(quant, on="target", how="left")
        .merge(clip, on="target", how="left")
        .merge(clipped_only, on="target", how="left")
        .merge(target_summary[["target", "positive_rate", "best_auc"]], on="target", how="left")
        .merge(weakest[["target", "weak_rank", "best_auc"]].rename(columns={"best_auc": "current_best_auc"}), on="target", how="left")
        .sort_values(["weak_rank", "best_view_delta_vs_raw"], ascending=[True, False])
        .reset_index(drop=True)
    )
    target_table.to_csv(args.artifact_dir / "per_target_best_view_table.csv", index=False)

    top_feature_table = feature_summary[feature_summary["feature"].isin(top_features)].copy()
    top_feature_table = top_feature_table.merge(
        numeric_partition[["feature", "group", "one_bin_warning"]].rename(
            columns={"group": "piecewise_group", "one_bin_warning": "piecewise_one_bin_warning"}
        ),
        on="feature",
        how="left",
        suffixes=("", "_partition"),
    )
    if "piecewise_group" not in top_feature_table.columns and "piecewise_group_partition" in top_feature_table.columns:
        top_feature_table["piecewise_group"] = top_feature_table["piecewise_group_partition"]
    if (
        "piecewise_one_bin_warning" not in top_feature_table.columns
        and "piecewise_one_bin_warning_partition" in top_feature_table.columns
    ):
        top_feature_table["piecewise_one_bin_warning"] = top_feature_table["piecewise_one_bin_warning_partition"]
    top_feature_table.to_csv(args.artifact_dir / "top24_feature_audit.csv", index=False)

    compact_block = {
        "primary_view": "raw_plus_quantile",
        "secondary_view": "raw_plus_clipped",
        "top_k_features": args.top_k_features,
        "top_features": top_features,
        "primary_block_columns": [f"{f}__quant" for f in top_features] + ["quant_row_mean", "quant_row_std"],
        "secondary_block_columns": [f"{f}__clipped" for f in top_features] + ["clipped_row_mean", "clipped_row_std"],
    }
    (args.artifact_dir / "compact_auxiliary_feature_block.json").write_text(json.dumps(compact_block, indent=2))

    weak_block = target_table[target_table["weak_rank"].notna() & (target_table["weak_rank"] <= 12)].copy()
    weak_block_summary = {
        "weak_targets_count": int(len(weak_block)),
        "best_view_counts": weak_block["best_view"].value_counts(dropna=False).to_dict(),
        "mean_quantile_delta_weak12": float(weak_block["raw_plus_quantile_delta"].mean()),
        "mean_clipped_delta_weak12": float(weak_block["raw_plus_clipped_delta"].mean()),
    }

    top_feature_summary = {
        "top24_piecewise_bad_count": int((top_feature_table["piecewise_group"] == "piecewise_bad").sum()),
        "top24_piecewise_ok_count": int((top_feature_table["piecewise_group"] == "piecewise_ok").sum()),
        "top24_missing_ratio_mean": float(top_feature_table["missing_ratio"].fillna(0.0).mean()),
    }

    overall_view_summary = {
        row["view_type"]: {
            "subset_macro_auc": float(row["macro_auc_subset"]),
            "positive_target_deltas": int(row["positive_target_deltas"]),
        }
        for _, row in view_summary.iterrows()
    }

    summary = {
        "primary_view": "raw_plus_quantile",
        "secondary_view": "raw_plus_clipped",
        "weak_block_summary": weak_block_summary,
        "top_feature_summary": top_feature_summary,
        "overall_view_summary": overall_view_summary,
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    md = [
        "# Feature-View Backbone Preparation",
        "",
        "## Chosen Views",
        "",
        f"- Primary candidate: `raw_plus_quantile`",
        f"- Secondary service view: `raw_plus_clipped`",
        f"- Source experiment: `{args.view_scores}`",
        "",
        "## Weak-Block Signal",
        "",
        f"- Weak targets covered: `{weak_block_summary['weak_targets_count']}`",
        f"- Mean `raw_plus_quantile` delta on weak-12: `{weak_block_summary['mean_quantile_delta_weak12']:.6f}`",
        f"- Mean `raw_plus_clipped` delta on weak-12: `{weak_block_summary['mean_clipped_delta_weak12']:.6f}`",
        f"- Best view counts on weak-12: `{weak_block_summary['best_view_counts']}`",
        "",
        "## Top-24 Feature Audit",
        "",
        f"- `piecewise_bad` among top-24: `{top_feature_summary['top24_piecewise_bad_count']}`",
        f"- `piecewise_ok` among top-24: `{top_feature_summary['top24_piecewise_ok_count']}`",
        f"- Mean missing ratio in top-24: `{top_feature_summary['top24_missing_ratio_mean']:.6f}`",
        "",
        "## Produced Artifacts",
        "",
        "- `per_target_best_view_table.csv`",
        "- `top24_feature_audit.csv`",
        "- `compact_auxiliary_feature_block.json`",
        "- `summary.json`",
    ]
    (args.artifact_dir / "feature_view_backbone_prep.md").write_text("\n".join(md) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
