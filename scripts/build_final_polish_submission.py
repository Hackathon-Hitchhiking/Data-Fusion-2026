from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_xgboost_multioutput_missing_lite_ensemble import (
    apply_overrides_by_rows,
    load_inputs as load_rule_inputs,
    logit,
    macro_auc,
    sigmoid,
    to_prediction_frame,
)
from scripts.lib.submission import write_submission_like_sample
from scripts.run_three_model_meta_stack import (
    build_family_map,
    build_related_targets,
    evaluate_candidate,
    make_split_mats,
)
from scripts.run_three_model_meta_stack_v2 import evaluate_two_stage_candidate
from scripts.run_three_model_meta_stack_v2 import current_best_valid as current_best_valid_v2
from scripts.run_three_model_meta_stack_v3_missing_lite import current_best_valid as current_best_valid_v3
from scripts.run_three_model_meta_stack_v3_missing_lite import load_inputs as load_stack_inputs_v3
from scripts.run_three_model_meta_stack import load_inputs as load_stack_inputs_v2


ARTIFACT_DIR = ROOT / "artifacts" / "final_polish_submission_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

STACK_V2_NAME = "two_stage_core_plus_ridge_4.0_m0.0001"
STACK_V3_NAME = "two_stage_core_plus_logistic_0.05_m0.0"
STACK_TARGETS = [
    "target_2_8",
    "target_2_3",
    "target_8_3",
    "target_10_1",
    "target_9_7",
    "target_3_1",
]
FEATURE_VIEW_TARGETS = [
    "target_2_6",
    "target_2_4",
    "target_9_3",
    "target_3_1",
    "target_9_7",
    "target_10_1",
]
STACK_ALPHA_GRID = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
FEATURE_VIEW_ALPHA_GRID = [0.01, 0.02, 0.05, 0.10]


def align_prediction_frame(
    base_customer_ids: np.ndarray,
    df: pd.DataFrame,
    pred_cols: list[str],
) -> np.ndarray:
    aligned = pd.DataFrame({"customer_id": base_customer_ids}).merge(
        df[["customer_id"] + pred_cols],
        on="customer_id",
        how="left",
        sort=False,
    )
    return aligned[pred_cols].to_numpy(dtype=np.float64)


def build_base_predictions() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray]:
    target_df, target_cols, val_mats, test_mats, test_customer_ids = load_rule_inputs()
    summary = json.loads((ROOT / "artifacts" / "post_xgboost_multioutput_missing_lite_blend_v1" / "summary.json").read_text())
    greedy_selected = summary["greedy_selected_targets"]
    base_valid = apply_overrides_by_rows(target_cols, blend_logit({"tabm": 0.50, "cat": 0.10, "xgb": 0.40}, val_mats), val_mats["xgb"], greedy_selected)
    base_test = apply_overrides_by_rows(target_cols, blend_logit({"tabm": 0.50, "cat": 0.10, "xgb": 0.40}, test_mats), test_mats["xgb"], greedy_selected)
    return target_df, target_cols, test_customer_ids, base_valid, base_test


def blend_logit(weights: dict[str, float], mats: dict[str, np.ndarray]) -> np.ndarray:
    total = None
    for key, weight in weights.items():
        term = weight * logit(mats[key])
        total = term if total is None else total + term
    assert total is not None
    return sigmoid(total)


def reconstruct_stack_v2(target_cols: list[str], pred_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    target_df, _, valid_mats, _, _ = load_stack_inputs_v2()
    split_mats = make_split_mats(valid_mats, valid_mats, target_cols)
    family_map = build_family_map(target_cols)
    related_map = build_related_targets(pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet"), target_cols)
    stage1_oof, _, _ = evaluate_candidate(
        variant="core",
        model_kind="ridge",
        strength=10.0,
        target_df=target_df,
        target_cols=target_cols,
        family_map=family_map,
        related_map=related_map,
        split_mats=split_mats,
    )
    pred_valid, _, _, _ = evaluate_two_stage_candidate(
        name=STACK_V2_NAME,
        feature_variant="core_plus",
        model_kind="ridge",
        strength=4.0,
        gain_margin=0.0001,
        target_df=target_df,
        target_cols=target_cols,
        family_map=family_map,
        related_map=related_map,
        split_mats=split_mats,
        current_best_prob_valid=current_best_valid_v2(valid_mats, target_cols),
        stage1_oof_prob=stage1_oof,
    )
    valid_df = pd.DataFrame(pred_valid, columns=pred_cols, dtype=np.float64)
    valid_df.insert(0, "customer_id", target_df["customer_id"].to_numpy(dtype=np.int32))
    test_df = pd.read_parquet(SUBMISSIONS_DIR / "three_model_stack_v2_01_two_stage_core_plus_ridge_4p0_m0p0001_submission.parquet")
    return valid_df, test_df[pred_cols].to_numpy(dtype=np.float64)


def reconstruct_stack_v3(target_cols: list[str], pred_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    target_df, _, valid_mats, _, _ = load_stack_inputs_v3()
    split_mats = make_split_mats(valid_mats, valid_mats, target_cols)
    family_map = build_family_map(target_cols)
    related_map = build_related_targets(pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet"), target_cols)
    stage1_oof, _, _ = evaluate_candidate(
        variant="core",
        model_kind="ridge",
        strength=10.0,
        target_df=target_df,
        target_cols=target_cols,
        family_map=family_map,
        related_map=related_map,
        split_mats=split_mats,
    )
    pred_valid, _, _, _ = evaluate_two_stage_candidate(
        name=STACK_V3_NAME,
        feature_variant="core_plus",
        model_kind="logistic",
        strength=0.05,
        gain_margin=0.0,
        target_df=target_df,
        target_cols=target_cols,
        family_map=family_map,
        related_map=related_map,
        split_mats=split_mats,
        current_best_prob_valid=current_best_valid_v3(valid_mats, target_cols),
        stage1_oof_prob=stage1_oof,
    )
    valid_df = pd.DataFrame(pred_valid, columns=pred_cols, dtype=np.float64)
    valid_df.insert(0, "customer_id", target_df["customer_id"].to_numpy(dtype=np.int32))
    test_df = pd.read_parquet(SUBMISSIONS_DIR / "three_model_stack_v3_01_two_stage_core_plus_logistic_0p05_m0p0_submission.parquet")
    return valid_df, test_df[pred_cols].to_numpy(dtype=np.float64)


def load_feature_view_probe(pred_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    valid_df = pd.read_parquet(ROOT / "artifacts" / "feature_view_submission_probe_v1" / "probe_validation_predictions.parquet")
    test_df = pd.read_parquet(ROOT / "artifacts" / "feature_view_submission_probe_v1" / "probe_test_predictions.parquet")
    return valid_df, test_df[pred_cols].to_numpy(dtype=np.float64)


def main() -> None:
    target_df, target_cols, test_customer_ids, base_valid, base_test = build_base_predictions()
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    base_auc = macro_auc(target_df, target_cols, base_valid)

    stack_v2_valid_df, stack_v2_test = reconstruct_stack_v2(target_cols, pred_cols)
    stack_v3_valid_df, stack_v3_test = reconstruct_stack_v3(target_cols, pred_cols)
    feature_view_valid_df, feature_view_test = load_feature_view_probe(pred_cols)

    stack_v2_valid = align_prediction_frame(target_df["customer_id"].to_numpy(dtype=np.int32), stack_v2_valid_df, pred_cols)
    stack_v3_valid = align_prediction_frame(target_df["customer_id"].to_numpy(dtype=np.int32), stack_v3_valid_df, pred_cols)
    feature_view_valid = align_prediction_frame(target_df["customer_id"].to_numpy(dtype=np.int32), feature_view_valid_df, pred_cols)

    stack_mean_valid = sigmoid(0.5 * logit(stack_v2_valid) + 0.5 * logit(stack_v3_valid))
    stack_mean_test = sigmoid(0.5 * logit(stack_v2_test) + 0.5 * logit(stack_v3_test))

    donors = {
        "stack_v2": {
            "valid": stack_v2_valid,
            "test": stack_v2_test,
            "targets": STACK_TARGETS,
            "alpha_grid": STACK_ALPHA_GRID,
        },
        "stack_v3": {
            "valid": stack_v3_valid,
            "test": stack_v3_test,
            "targets": STACK_TARGETS,
            "alpha_grid": STACK_ALPHA_GRID,
        },
        "stack_mean": {
            "valid": stack_mean_valid,
            "test": stack_mean_test,
            "targets": STACK_TARGETS,
            "alpha_grid": STACK_ALPHA_GRID,
        },
        "feature_view_clip": {
            "valid": feature_view_valid,
            "test": feature_view_test,
            "targets": FEATURE_VIEW_TARGETS,
            "alpha_grid": FEATURE_VIEW_ALPHA_GRID,
        },
    }

    current_valid = base_valid.copy()
    current_test = base_test.copy()
    current_auc = base_auc
    selected = []

    while True:
        best_row = None
        for donor_name, donor in donors.items():
            for target_name in donor["targets"]:
                idx = target_cols.index(target_name)
                for alpha in donor["alpha_grid"]:
                    cand_valid_col = sigmoid((1.0 - alpha) * logit(current_valid[:, idx]) + alpha * logit(donor["valid"][:, idx]))
                    cand_valid = current_valid.copy()
                    cand_valid[:, idx] = cand_valid_col
                    auc = macro_auc(target_df, target_cols, cand_valid)
                    delta = auc - current_auc
                    if best_row is None or delta > best_row["delta"]:
                        best_row = {
                            "donor": donor_name,
                            "target": target_name,
                            "alpha": float(alpha),
                            "delta": float(delta),
                            "macro_auc_after_apply": float(auc),
                        }
        if best_row is None or best_row["delta"] <= 0:
            break

        donor = donors[best_row["donor"]]
        idx = target_cols.index(best_row["target"])
        alpha = float(best_row["alpha"])
        current_valid[:, idx] = sigmoid((1.0 - alpha) * logit(current_valid[:, idx]) + alpha * logit(donor["valid"][:, idx]))
        current_test[:, idx] = sigmoid((1.0 - alpha) * logit(current_test[:, idx]) + alpha * logit(donor["test"][:, idx]))
        current_auc = float(best_row["macro_auc_after_apply"])
        selected.append(best_row)

    output_path = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_final_polish_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, current_test),
        output_path=output_path,
    )

    pd.DataFrame(selected).to_csv(ARTIFACT_DIR / "selected_polish_steps.csv", index=False)
    summary = {
        "base_submission": str(SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_besttri_greedy_submission.parquet"),
        "base_macro_auc": float(base_auc),
        "final_macro_auc": float(current_auc),
        "delta_vs_base": float(current_auc - base_auc),
        "selected_steps": selected,
        "output_path": str(output_path),
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
