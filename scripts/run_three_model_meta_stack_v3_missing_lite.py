from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.submission import write_submission_like_sample
from scripts.run_three_model_meta_stack import (
    N_SPLITS,
    SEED,
    build_family_map,
    build_related_targets,
    fit_full_and_predict,
    load_inputs as load_inputs_legacy,
    logit,
    macro_auc,
    make_model,
    make_split_mats,
    predict_scores,
    rank_pct,
    sigmoid,
    top2_mean,
)
from scripts.run_three_model_meta_stack_v2 import (
    STAGE2_LOGISTIC_C_VALUES,
    STAGE2_RIDGE_ALPHA_VALUES,
    TARGET_GAIN_MARGINS,
    TOP_RESULTS_TO_BUILD,
    STAGE2_TARGETS,
    build_stage2_features,
    evaluate_two_stage_candidate,
    fit_two_stage_test,
)


ARTIFACT_DIR = ROOT / "artifacts" / "three_model_meta_stack_v3_missing_lite"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

UPDATED_TRI_WEIGHTS = {"tabm": 0.55, "cat": 0.10, "xgb": 0.35}
UPDATED_RESTRAINED_TARGET_OVERRIDES = [
    {"target": "target_2_8", "mode": "xgb_full"},
    {"target": "target_8_3", "mode": "tri_xgb30"},
    {"target": "target_2_3", "mode": "tri_xgb30"},
]


def load_inputs() -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    target = pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet")
    target_cols = [c for c in target.columns if c != "customer_id"]
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]

    tabm_valid = pd.read_parquet(
        ROOT
        / "output"
        / "kaggle-output"
        / "tabm-longrun-v3-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_tabm_longrun_v3_fs_v1_full_gpu"
        / "validation_predictions.parquet"
    )
    cat_valid = pd.read_parquet(
        ROOT
        / "output"
        / "kaggle-output"
        / "catboost-multilabel-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_catboost_multilabel_fs_v1_full_gpu"
        / "validation_predictions.parquet"
    )
    xgb_valid = pd.read_parquet(
        ROOT
        / "output"
        / "kaggle-output"
        / "xgboost-multioutput-fs-v2-missing-lite"
        / "pull_complete"
        / "artifacts"
        / "gpu_xgboost_multioutput_fs_v2_missing_lite_full_gpu"
        / "validation_predictions.parquet"
    )
    merged = target.merge(tabm_valid, on="customer_id", how="inner")
    merged = merged.merge(cat_valid, on="customer_id", how="inner", suffixes=("_tabm", "_cat"))
    merged = merged.merge(xgb_valid, on="customer_id", how="inner")

    valid_mats = {
        "tabm": merged[[f"{c}_tabm" for c in pred_cols]].to_numpy(dtype=np.float64),
        "cat": merged[[f"{c}_cat" for c in pred_cols]].to_numpy(dtype=np.float64),
        "xgb": merged[pred_cols].to_numpy(dtype=np.float64),
    }
    test_mats = {
        "tabm": pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
        "cat": pd.read_parquet(SUBMISSIONS_DIR / "catboost_multilabel_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
        "xgb": pd.read_parquet(SUBMISSIONS_DIR / "xgboost_multioutput_fs_v2_missing_lite_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
    }
    test_customer_ids = pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")["customer_id"].to_numpy(dtype=np.int32)
    return merged[["customer_id"] + target_cols], target_cols, valid_mats, test_mats, test_customer_ids


def current_best_valid(valid_mats: dict[str, np.ndarray], target_cols: list[str]) -> np.ndarray:
    tri_logit = (
        UPDATED_TRI_WEIGHTS["tabm"] * logit(valid_mats["tabm"])
        + UPDATED_TRI_WEIGHTS["cat"] * logit(valid_mats["cat"])
        + UPDATED_TRI_WEIGHTS["xgb"] * logit(valid_mats["xgb"])
    )
    pred = sigmoid(tri_logit)
    for override in UPDATED_RESTRAINED_TARGET_OVERRIDES:
        idx = target_cols.index(override["target"])
        if override["mode"] == "xgb_full":
            pred[:, idx] = valid_mats["xgb"][:, idx]
        elif override["mode"] == "tri_xgb30":
            pred[:, idx] = sigmoid(0.7 * logit(pred[:, idx]) + 0.3 * logit(valid_mats["xgb"][:, idx]))
        else:
            raise ValueError(f"Unsupported override mode: {override['mode']}")
    return pred


def current_best_test(target_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    df = pd.read_parquet(SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_tri_logit_055_010_035_restrained_submission.parquet")
    return df["customer_id"].to_numpy(dtype=np.int32), df[pred_cols].to_numpy(dtype=np.float64)


def build_submission_name(rank: int, candidate_name: str) -> str:
    safe = candidate_name.replace(".", "p")
    return f"three_model_stack_v3_{rank:02d}_{safe}_submission.parquet"


def main() -> None:
    target_df, target_cols, valid_mats, test_mats, _ = load_inputs()
    split_mats = make_split_mats(valid_mats, test_mats, target_cols)
    family_map = build_family_map(target_cols)
    related_map = build_related_targets(pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet"), target_cols)

    current_best_valid_prob = current_best_valid(valid_mats, target_cols)
    test_customer_ids, current_best_test_prob = current_best_test(target_cols)
    current_best_auc = macro_auc(target_df, target_cols, current_best_valid_prob)

    from scripts.run_three_model_meta_stack import evaluate_candidate

    stage1_name = "core_ridge_10.0"
    stage1_oof, stage1_auc, stage1_target_scores = evaluate_candidate(
        variant="core",
        model_kind="ridge",
        strength=10.0,
        target_df=target_df,
        target_cols=target_cols,
        family_map=family_map,
        related_map=related_map,
        split_mats=split_mats,
    )
    stage1_test = fit_full_and_predict(
        variant="core",
        model_kind="ridge",
        strength=10.0,
        target_df=target_df,
        target_cols=target_cols,
        family_map=family_map,
        related_map=related_map,
        split_mats=split_mats,
    )
    stage1_target_scores.to_csv(ARTIFACT_DIR / "stage1_core_ridge_10_target_scores.csv", index=False)

    candidate_rows = []
    candidate_test_preds: dict[str, np.ndarray] = {}
    candidate_target_tables: dict[str, pd.DataFrame] = {}
    candidate_accepts: dict[str, list[dict[str, float | str]]] = {}

    for feature_variant in ["core_plus", "family_plus"]:
        for gain_margin in TARGET_GAIN_MARGINS:
            for strength in STAGE2_LOGISTIC_C_VALUES:
                name = f"two_stage_{feature_variant}_logistic_{strength}_m{gain_margin}"
                _, score, target_table, accepted = evaluate_two_stage_candidate(
                    name=name,
                    feature_variant=feature_variant,
                    model_kind="logistic",
                    strength=strength,
                    gain_margin=gain_margin,
                    target_df=target_df,
                    target_cols=target_cols,
                    family_map=family_map,
                    related_map=related_map,
                    split_mats=split_mats,
                    current_best_prob_valid=current_best_valid_prob,
                    stage1_oof_prob=stage1_oof,
                )
                accepted_targets = [row["target"] for row in accepted]
                pred_test = fit_two_stage_test(
                    accepted_targets=accepted_targets,
                    feature_variant=feature_variant,
                    model_kind="logistic",
                    strength=strength,
                    target_df=target_df,
                    target_cols=target_cols,
                    family_map=family_map,
                    related_map=related_map,
                    split_mats=split_mats,
                    current_best_prob_valid=current_best_valid_prob,
                    current_best_prob_test=current_best_test_prob,
                    stage1_oof_prob=stage1_oof,
                    stage1_test_prob=stage1_test,
                )
                candidate_rows.append(
                    {
                        "name": name,
                        "feature_variant": feature_variant,
                        "model_kind": "logistic",
                        "strength": strength,
                        "gain_margin": gain_margin,
                        "oof_macro_auc": score,
                        "delta_vs_current_best": score - current_best_auc,
                        "accepted_targets": "|".join(accepted_targets),
                        "accepted_count": len(accepted_targets),
                    }
                )
                candidate_test_preds[name] = pred_test
                candidate_target_tables[name] = target_table
                candidate_accepts[name] = accepted

            for strength in STAGE2_RIDGE_ALPHA_VALUES:
                name = f"two_stage_{feature_variant}_ridge_{strength}_m{gain_margin}"
                _, score, target_table, accepted = evaluate_two_stage_candidate(
                    name=name,
                    feature_variant=feature_variant,
                    model_kind="ridge",
                    strength=strength,
                    gain_margin=gain_margin,
                    target_df=target_df,
                    target_cols=target_cols,
                    family_map=family_map,
                    related_map=related_map,
                    split_mats=split_mats,
                    current_best_prob_valid=current_best_valid_prob,
                    stage1_oof_prob=stage1_oof,
                )
                accepted_targets = [row["target"] for row in accepted]
                pred_test = fit_two_stage_test(
                    accepted_targets=accepted_targets,
                    feature_variant=feature_variant,
                    model_kind="ridge",
                    strength=strength,
                    target_df=target_df,
                    target_cols=target_cols,
                    family_map=family_map,
                    related_map=related_map,
                    split_mats=split_mats,
                    current_best_prob_valid=current_best_valid_prob,
                    current_best_prob_test=current_best_test_prob,
                    stage1_oof_prob=stage1_oof,
                    stage1_test_prob=stage1_test,
                )
                candidate_rows.append(
                    {
                        "name": name,
                        "feature_variant": feature_variant,
                        "model_kind": "ridge",
                        "strength": strength,
                        "gain_margin": gain_margin,
                        "oof_macro_auc": score,
                        "delta_vs_current_best": score - current_best_auc,
                        "accepted_targets": "|".join(accepted_targets),
                        "accepted_count": len(accepted_targets),
                    }
                )
                candidate_test_preds[name] = pred_test
                candidate_target_tables[name] = target_table
                candidate_accepts[name] = accepted

    candidate_df = pd.DataFrame(candidate_rows).sort_values("oof_macro_auc", ascending=False).reset_index(drop=True)
    candidate_df.to_csv(ARTIFACT_DIR / "candidate_scores.csv", index=False)

    top_outputs = []
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    for rank, row in enumerate(candidate_df.head(TOP_RESULTS_TO_BUILD).to_dict("records"), start=1):
        output_path = SUBMISSIONS_DIR / build_submission_name(rank, row["name"])
        write_submission_like_sample(
            customer_ids=test_customer_ids,
            prediction_frame=pd.DataFrame(candidate_test_preds[row["name"]], columns=pred_cols, dtype=np.float64),
            output_path=output_path,
        )
        target_table_path = ARTIFACT_DIR / f"{row['name']}_target_scores.csv"
        candidate_target_tables[row["name"]].to_csv(target_table_path, index=False)
        accepted_path = ARTIFACT_DIR / f"{row['name']}_accepted.json"
        accepted_path.write_text(json.dumps(candidate_accepts[row["name"]], indent=2) + "\n")
        top_outputs.append(
            {
                "rank": rank,
                "candidate": row["name"],
                "feature_variant": row["feature_variant"],
                "model_kind": row["model_kind"],
                "strength": float(row["strength"]),
                "gain_margin": float(row["gain_margin"]),
                "oof_macro_auc": float(row["oof_macro_auc"]),
                "delta_vs_current_best": float(row["delta_vs_current_best"]),
                "accepted_targets": row["accepted_targets"],
                "accepted_count": int(row["accepted_count"]),
                "submission_path": str(output_path),
                "target_scores_path": str(target_table_path),
                "accepted_path": str(accepted_path),
            }
        )

    summary = {
        "meta_train_protocol": "shared_75k_out_of_sample_validation_predictions_from_tabm_catboost_xgboost_missing_lite",
        "note": "Two-stage restrained stack over the updated v2lite restrained tri-blend reference. Still not full-750k K-fold OOF over all three backbones.",
        "current_best_reference_auc": float(current_best_auc),
        "stage1_candidate": stage1_name,
        "stage1_oof_macro_auc": float(stage1_auc),
        "stage2_targets": STAGE2_TARGETS,
        "top_candidates": top_outputs,
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
