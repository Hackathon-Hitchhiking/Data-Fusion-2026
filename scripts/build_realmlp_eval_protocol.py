from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.layout import project_root


ROOT = project_root()
TABM_METRICS = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/metrics.json"
)
TABM_TARGET_AUC = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/target_scores.csv"
)
TARGET_SUMMARY = ROOT / "artifacts/dataset_feature_diagnostics_v1/target_summary.csv"
WEAK_RANK = ROOT / "artifacts/weakest_targets_by_best_auc_v2_longrun.csv"
OUT_DIR = ROOT / "artifacts/realmlp_eval_protocol_v1"


def decision_bucket(row: pd.Series) -> str:
    if row["weak_rank"] <= 12:
        return "priority_weak"
    if row["weak_rank"] <= 24:
        return "mid"
    return "strong"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    tabm_metrics = json.loads(TABM_METRICS.read_text())
    tabm_auc = pd.read_csv(TABM_TARGET_AUC).rename(columns={"oof_auc": "auc_tabm_v3"})
    target_summary = pd.read_csv(TARGET_SUMMARY)
    weak_rank = pd.read_csv(WEAK_RANK)[["target"]].copy()
    weak_rank["weak_rank"] = range(1, len(weak_rank) + 1)

    comparison = (
        target_summary[["target", "positive_rate", "best_auc", "best_model"]]
        .merge(tabm_auc[["target", "auc_tabm_v3"]], on="target", how="left")
        .merge(weak_rank, on="target", how="left")
        .sort_values(["weak_rank", "target"])
        .reset_index(drop=True)
    )
    comparison["family"] = comparison["target"].map(lambda x: int(x.split("_")[1]))
    comparison["member"] = comparison["target"].map(lambda x: int(x.split("_")[2]))
    comparison["priority_bucket"] = comparison.apply(decision_bucket, axis=1)
    comparison["auc_realmlp"] = pd.NA
    comparison["delta_realmlp_vs_tabm_v3"] = pd.NA
    comparison["prediction_correlation"] = pd.NA
    comparison["weighted_avg_best_weight_realmlp"] = pd.NA
    comparison["weighted_avg_best_auc"] = pd.NA
    comparison["rank_avg_auc"] = pd.NA
    comparison["keep_revise_kill"] = pd.NA
    comparison["notes"] = pd.NA
    comparison.to_csv(OUT_DIR / "per_target_comparison_template.csv", index=False)

    ensemble_grid = pd.DataFrame(
        [
            {"method": "weighted_avg", "realmlp_weight": 0.2, "tabm_weight": 0.8, "status": "planned"},
            {"method": "weighted_avg", "realmlp_weight": 0.3, "tabm_weight": 0.7, "status": "planned"},
            {"method": "weighted_avg", "realmlp_weight": 0.5, "tabm_weight": 0.5, "status": "planned"},
            {"method": "rank_avg", "realmlp_weight": 0.5, "tabm_weight": 0.5, "status": "planned"},
        ]
    )
    ensemble_grid.to_csv(OUT_DIR / "ensemble_candidates_template.csv", index=False)

    protocol_md = "\n".join(
        [
            "# RealMLP Evaluation Protocol",
            "",
            f"- TabM v3 validation macro AUC baseline: `{tabm_metrics['validation_macro_auc']:.10f}`",
            "- First check RealMLP as a standalone backbone on full 41-target macro.",
            "- Then compare per-target AUC against TabM v3.",
            "- Then test pairwise ensembles on the shared validation split.",
            "",
            "## Keep / Revise / Kill",
            "",
            "- `keep_backbone`: RealMLP standalone macro is within 0.002 of TabM v3, or any ensemble beats TabM v3 on full macro.",
            "- `revise`: standalone is weaker, but 6+ targets show positive delta with low prediction correlation and ensemble candidates are near-flat.",
            "- `kill`: standalone is clearly weak and no weighted/rank ensemble beats TabM v3 full macro.",
            "",
            "## Per-target fields to fill after Kaggle validation",
            "",
            "- `auc_realmlp`",
            "- `delta_realmlp_vs_tabm_v3`",
            "- `prediction_correlation`",
            "- `weighted_avg_best_weight_realmlp`",
            "- `weighted_avg_best_auc`",
            "- `rank_avg_auc`",
            "- `keep_revise_kill`",
            "",
            "## Priority reading order",
            "",
            "- Weakest 12 targets first.",
            "- Then targets where RealMLP beats TabM v3 by any positive margin.",
            "- Then high-correlation targets can be ignored unless standalone macro is already strong.",
        ]
    )
    (OUT_DIR / "decision_rules.md").write_text(protocol_md + "\n")

    summary = {
        "tabm_v3_validation_macro_auc": float(tabm_metrics["validation_macro_auc"]),
        "targets": int(len(comparison)),
        "priority_weak_targets": int((comparison["priority_bucket"] == "priority_weak").sum()),
        "mid_targets": int((comparison["priority_bucket"] == "mid").sum()),
        "strong_targets": int((comparison["priority_bucket"] == "strong").sum()),
        "artifacts": {
            "comparison_template": "per_target_comparison_template.csv",
            "ensemble_grid": "ensemble_candidates_template.csv",
            "rules": "decision_rules.md",
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
