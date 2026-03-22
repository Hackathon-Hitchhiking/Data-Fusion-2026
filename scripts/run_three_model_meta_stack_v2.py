from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.submission import write_submission_like_sample
from scripts.run_three_model_meta_stack import (
    N_SPLITS,
    SEED,
    build_family_map,
    build_related_targets,
    build_variant_features,
    fit_full_and_predict,
    load_inputs,
    logit,
    macro_auc,
    make_model,
    make_split_mats,
    predict_scores,
    rank_pct,
    sigmoid,
    top2_mean,
)


ARTIFACT_DIR = ROOT / "artifacts" / "three_model_meta_stack_v2"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

STAGE2_LOGISTIC_C_VALUES = [0.05, 0.1, 0.25, 0.5]
STAGE2_RIDGE_ALPHA_VALUES = [1.0, 4.0, 10.0]
TARGET_GAIN_MARGINS = [0.0, 0.0001, 0.00025]
TOP_RESULTS_TO_BUILD = 2
STAGE2_TARGETS = [
    "target_2_8",
    "target_2_3",
    "target_8_3",
    "target_10_1",
    "target_9_7",
    "target_3_1",
]


def current_best_valid(valid_mats: dict[str, np.ndarray], target_cols: list[str]) -> np.ndarray:
    tri_logit = 0.55 * logit(valid_mats["tabm"]) + 0.10 * logit(valid_mats["cat"]) + 0.35 * logit(valid_mats["xgb"])
    pred = sigmoid(tri_logit)
    pred[:, target_cols.index("target_2_8")] = valid_mats["xgb"][:, target_cols.index("target_2_8")]
    pred[:, target_cols.index("target_8_3")] = sigmoid(
        0.7 * logit(pred[:, target_cols.index("target_8_3")]) + 0.3 * logit(valid_mats["xgb"][:, target_cols.index("target_8_3")])
    )
    pred[:, target_cols.index("target_2_3")] = sigmoid(
        0.7 * logit(pred[:, target_cols.index("target_2_3")]) + 0.3 * logit(valid_mats["xgb"][:, target_cols.index("target_2_3")])
    )
    return pred


def current_best_test(target_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    df = pd.read_parquet(SUBMISSIONS_DIR / "tabm_cat_xgb_tri_logit_055_010_035_restrained_targetwise_submission.parquet")
    return df["customer_id"].to_numpy(dtype=np.int32), df[pred_cols].to_numpy(dtype=np.float64)


def family_stats(prob_mat: np.ndarray, target_name: str, target_cols: list[str], family_map: dict[str, list[str]]) -> dict[str, np.ndarray]:
    family_id = target_name.split("_")[1]
    family_targets = [t for t in family_map[family_id] if t != target_name]
    family_indices = [target_cols.index(t) for t in family_targets]
    if not family_indices:
        zeros = np.zeros(prob_mat.shape[0], dtype=np.float64)
        return {
            "mean": zeros,
            "max": zeros,
            "top2": zeros,
            "spread": zeros,
        }
    family_logit = logit(prob_mat[:, family_indices])
    return {
        "mean": family_logit.mean(axis=1),
        "max": family_logit.max(axis=1),
        "top2": top2_mean(family_logit),
        "spread": family_logit.max(axis=1) - family_logit.min(axis=1),
    }


def build_stage2_features(
    *,
    target_name: str,
    target_cols: list[str],
    family_map: dict[str, list[str]],
    related_map: dict[str, list[str]],
    split_mats: dict[str, dict[str, np.ndarray]],
    current_best_prob: np.ndarray,
    stage1_prob: np.ndarray,
    feature_variant: str,
) -> pd.DataFrame:
    idx = target_cols.index(target_name)
    base_df = build_variant_features(
        "family",
        target_name,
        target_cols,
        family_map,
        related_map,
        split_mats,
    )
    current_best_rank = rank_pct(current_best_prob[:, idx])
    stage1_rank = rank_pct(stage1_prob[:, idx])
    out = pd.DataFrame(
        {
            "current_best_prob": current_best_prob[:, idx],
            "current_best_logit": logit(current_best_prob[:, idx]),
            "current_best_rank": current_best_rank,
            "stage1_prob": stage1_prob[:, idx],
            "stage1_logit": logit(stage1_prob[:, idx]),
            "stage1_rank": stage1_rank,
            "current_minus_stage1": logit(current_best_prob[:, idx]) - logit(stage1_prob[:, idx]),
            "current_minus_tri": logit(current_best_prob[:, idx]) - split_mats["tri_logit"][:, idx],
            "current_minus_tabm": logit(current_best_prob[:, idx]) - split_mats["tabm_logit"][:, idx],
            "current_minus_cat": logit(current_best_prob[:, idx]) - split_mats["cat_logit"][:, idx],
            "current_minus_xgb": logit(current_best_prob[:, idx]) - split_mats["xgb_logit"][:, idx],
        }
    )

    core_cols = [
        "logit_tabm",
        "logit_cat",
        "logit_xgb",
        "rank_tabm",
        "rank_cat",
        "rank_xgb",
        "mean_logit",
        "max_logit",
        "min_logit",
        "spread_logit",
        "tabm_minus_cat",
        "tabm_minus_xgb",
        "cat_minus_xgb",
    ]
    out = pd.concat([out, base_df[core_cols].copy()], axis=1)
    if feature_variant == "core_plus":
        return out

    current_family = family_stats(current_best_prob, target_name, target_cols, family_map)
    stage1_family = family_stats(stage1_prob, target_name, target_cols, family_map)
    out["current_family_mean"] = current_family["mean"]
    out["current_family_max"] = current_family["max"]
    out["current_family_top2"] = current_family["top2"]
    out["current_family_spread"] = current_family["spread"]
    out["stage1_family_mean"] = stage1_family["mean"]
    out["stage1_family_max"] = stage1_family["max"]
    out["stage1_family_top2"] = stage1_family["top2"]
    out["stage1_family_spread"] = stage1_family["spread"]
    out["current_stage1_family_gap"] = current_family["mean"] - stage1_family["mean"]

    family_cols = [
        "tabm_family_mean",
        "tabm_family_max",
        "tabm_family_top2",
        "tabm_family_spread",
        "cat_family_mean",
        "cat_family_max",
        "cat_family_top2",
        "cat_family_spread",
        "xgb_family_mean",
        "xgb_family_max",
        "xgb_family_top2",
        "xgb_family_spread",
        "tri_family_mean",
        "tri_family_max",
        "tri_family_top2",
        "tri_family_spread",
        "family_disagreement",
    ]
    rel_cols = [c for c in base_df.columns if c.startswith("rel")]
    out = pd.concat([out, base_df[family_cols + rel_cols].copy()], axis=1)

    for rel_idx, related_target in enumerate(related_map[target_name], start=1):
        related_pos = target_cols.index(related_target)
        out[f"rel{rel_idx}_current_logit"] = logit(current_best_prob[:, related_pos])
        out[f"rel{rel_idx}_stage1_logit"] = logit(stage1_prob[:, related_pos])

    return out


def evaluate_two_stage_candidate(
    *,
    name: str,
    feature_variant: str,
    model_kind: str,
    strength: float,
    gain_margin: float,
    target_df: pd.DataFrame,
    target_cols: list[str],
    family_map: dict[str, list[str]],
    related_map: dict[str, list[str]],
    split_mats: dict[str, dict[str, np.ndarray]],
    current_best_prob_valid: np.ndarray,
    stage1_oof_prob: np.ndarray,
) -> tuple[np.ndarray, float, pd.DataFrame, list[dict[str, float | str]]]:
    candidate = current_best_prob_valid.copy()
    target_rows = []
    accepted = []
    for target_name in STAGE2_TARGETS:
        idx = target_cols.index(target_name)
        y = target_df[target_name].to_numpy(dtype=np.int8)
        X = build_stage2_features(
            target_name=target_name,
            target_cols=target_cols,
            family_map=family_map,
            related_map=related_map,
            split_mats=split_mats["valid"],
            current_best_prob=current_best_prob_valid,
            stage1_prob=stage1_oof_prob,
            feature_variant=feature_variant,
        )
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        target_oof = np.zeros(len(X), dtype=np.float64)
        for train_idx, valid_idx in skf.split(X, y):
            model = make_model(model_kind, strength)
            model.fit(X.iloc[train_idx], y[train_idx])
            target_oof[valid_idx] = predict_scores(model_kind, model, X.iloc[valid_idx])

        baseline_auc = float(roc_auc_score(y, current_best_prob_valid[:, idx]))
        corrected_auc = float(roc_auc_score(y, target_oof))
        gain = corrected_auc - baseline_auc
        kept = gain > gain_margin
        if kept:
            candidate[:, idx] = target_oof
            accepted.append(
                {
                    "target": target_name,
                    "baseline_auc": baseline_auc,
                    "corrected_auc": corrected_auc,
                    "gain": gain,
                }
            )
        target_rows.append(
            {
                "target": target_name,
                "baseline_auc": baseline_auc,
                "corrected_auc": corrected_auc,
                "gain": gain,
                "kept_flag": int(kept),
            }
        )

    score = macro_auc(target_df, target_cols, candidate)
    return candidate, score, pd.DataFrame(target_rows).sort_values("gain", ascending=False).reset_index(drop=True), accepted


def fit_two_stage_test(
    *,
    accepted_targets: list[str],
    feature_variant: str,
    model_kind: str,
    strength: float,
    target_df: pd.DataFrame,
    target_cols: list[str],
    family_map: dict[str, list[str]],
    related_map: dict[str, list[str]],
    split_mats: dict[str, dict[str, np.ndarray]],
    current_best_prob_valid: np.ndarray,
    current_best_prob_test: np.ndarray,
    stage1_oof_prob: np.ndarray,
    stage1_test_prob: np.ndarray,
) -> np.ndarray:
    pred = current_best_prob_test.copy()
    for target_name in accepted_targets:
        idx = target_cols.index(target_name)
        y = target_df[target_name].to_numpy(dtype=np.int8)
        X_train = build_stage2_features(
            target_name=target_name,
            target_cols=target_cols,
            family_map=family_map,
            related_map=related_map,
            split_mats=split_mats["valid"],
            current_best_prob=current_best_prob_valid,
            stage1_prob=stage1_oof_prob,
            feature_variant=feature_variant,
        )
        X_test = build_stage2_features(
            target_name=target_name,
            target_cols=target_cols,
            family_map=family_map,
            related_map=related_map,
            split_mats=split_mats["test"],
            current_best_prob=current_best_prob_test,
            stage1_prob=stage1_test_prob,
            feature_variant=feature_variant,
        )
        model = make_model(model_kind, strength)
        model.fit(X_train, y)
        pred[:, idx] = predict_scores(model_kind, model, X_test)
    return pred


def build_submission_name(rank: int, candidate_name: str) -> str:
    safe = candidate_name.replace(".", "p")
    return f"three_model_stack_v2_{rank:02d}_{safe}_submission.parquet"


def main() -> None:
    target_df, target_cols, valid_mats, test_mats, _ = load_inputs()
    split_mats = make_split_mats(valid_mats, test_mats, target_cols)
    family_map = build_family_map(target_cols)
    related_map = build_related_targets(pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet"), target_cols)

    current_best_valid_prob = current_best_valid(valid_mats, target_cols)
    test_customer_ids, current_best_test_prob = current_best_test(target_cols)
    current_best_auc = macro_auc(target_df, target_cols, current_best_valid_prob)

    stage1_name = "core_ridge_10.0"
    from scripts.run_three_model_meta_stack import evaluate_candidate

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
    candidate_valid_preds: dict[str, np.ndarray] = {}
    candidate_test_preds: dict[str, np.ndarray] = {}
    candidate_target_tables: dict[str, pd.DataFrame] = {}
    candidate_accepts: dict[str, list[dict[str, float | str]]] = {}

    for feature_variant in ["core_plus", "family_plus"]:
        for gain_margin in TARGET_GAIN_MARGINS:
            for strength in STAGE2_LOGISTIC_C_VALUES:
                name = f"two_stage_{feature_variant}_logistic_{strength}_m{gain_margin}"
                pred_valid, score, target_table, accepted = evaluate_two_stage_candidate(
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
                candidate_valid_preds[name] = pred_valid
                candidate_test_preds[name] = pred_test
                candidate_target_tables[name] = target_table
                candidate_accepts[name] = accepted

            for strength in STAGE2_RIDGE_ALPHA_VALUES:
                name = f"two_stage_{feature_variant}_ridge_{strength}_m{gain_margin}"
                pred_valid, score, target_table, accepted = evaluate_two_stage_candidate(
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
                candidate_valid_preds[name] = pred_valid
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
        "meta_train_protocol": "shared_75k_out_of_sample_validation_predictions_from_tabm_catboost_xgboost",
        "note": "Two-stage restrained stack over the current best tri-blend reference. Still not full-750k K-fold OOF over all three backbones.",
        "stage1_candidate": stage1_name,
        "stage1_oof_macro_auc": float(stage1_auc),
        "current_best_reference_auc": float(current_best_auc),
        "stage2_targets": STAGE2_TARGETS,
        "top_candidates": top_outputs,
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
