from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.submission import write_submission_like_sample


ARTIFACT_DIR = ROOT / "artifacts" / "post_xgboost_multioutput_missing_lite_blend_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

TABM_CAT_BASE_CAT_WEIGHT = 0.30
LEGACY_TRI_WEIGHTS = {"tabm": 0.55, "cat": 0.10, "xgb": 0.35}
LEGACY_RESTRAINED_TARGET_OVERRIDES = [
    {"target": "target_2_8", "mode": "xgb_full"},
    {"target": "target_8_3", "mode": "tri_xgb30"},
    {"target": "target_2_3", "mode": "tri_xgb30"},
]


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def macro_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> float:
    values = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        values.append(float(roc_auc_score(y, pred[:, idx])) if np.unique(y).size > 1 else 0.5)
    return float(np.mean(values))


def per_target_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        score = float(roc_auc_score(y, pred[:, idx])) if np.unique(y).size > 1 else 0.5
        rows.append({"target": target_name, "oof_auc": score})
    return pd.DataFrame(rows)


def blend_logit(weights: dict[str, float], mats: dict[str, np.ndarray]) -> np.ndarray:
    total = None
    for key, weight in weights.items():
        term = weight * logit(mats[key])
        total = term if total is None else total + term
    assert total is not None
    return sigmoid(total)


def to_prediction_frame(target_cols: list[str], pred_mat: np.ndarray) -> pd.DataFrame:
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    return pd.DataFrame(pred_mat, columns=pred_cols, dtype=np.float64)


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

    val_mats = {
        "tabm": merged[[f"{c}_tabm" for c in pred_cols]].to_numpy(dtype=np.float64),
        "cat": merged[[f"{c}_cat" for c in pred_cols]].to_numpy(dtype=np.float64),
        "xgb": merged[pred_cols].to_numpy(dtype=np.float64),
    }
    test_mats = {
        "tabm": pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
        "cat": pd.read_parquet(SUBMISSIONS_DIR / "catboost_multilabel_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
        "xgb": pd.read_parquet(
            ROOT
            / "output"
            / "kaggle-output"
            / "xgboost-multioutput-fs-v2-missing-lite"
            / "pull_complete"
            / "artifacts"
            / "gpu_xgboost_multioutput_fs_v2_missing_lite_full_gpu"
            / "submission.parquet"
        )[pred_cols].to_numpy(dtype=np.float64),
    }
    test_customer_ids = pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")["customer_id"].to_numpy(dtype=np.int32)
    return merged[["customer_id"] + target_cols], target_cols, val_mats, test_mats, test_customer_ids


def build_coarse_grid(target_df: pd.DataFrame, target_cols: list[str], val_mats: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    rows.append({"kind": "solo", "name": "tabm", "auc": macro_auc(target_df, target_cols, val_mats["tabm"])})
    rows.append({"kind": "solo", "name": "cat", "auc": macro_auc(target_df, target_cols, val_mats["cat"])})
    rows.append({"kind": "solo", "name": "xgb_missing_lite", "auc": macro_auc(target_df, target_cols, val_mats["xgb"])})

    tabm_cat = blend_logit({"tabm": 0.70, "cat": 0.30}, val_mats)
    rows.append({"kind": "baseline", "name": "tabm_cat_logit30", "auc": macro_auc(target_df, target_cols, tabm_cat)})

    weights = np.round(np.arange(0.05, 0.55, 0.05), 2)
    for w in weights:
        rows.append(
            {
                "kind": "pair_logit",
                "name": f"tabm_xgb_logit_{w:.2f}",
                "auc": macro_auc(target_df, target_cols, blend_logit({"tabm": 1.0 - w, "xgb": w}, val_mats)),
                "wt": round(1.0 - w, 2),
                "wc": 0.0,
                "wx": float(w),
            }
        )

    for wc in weights:
        for wx in weights:
            wt = round(1.0 - wc - wx, 2)
            if wt < 0.40 or wt <= 0:
                continue
            rows.append(
                {
                    "kind": "tri_logit",
                    "name": f"tri_logit_t{wt:.2f}_c{wc:.2f}_x{wx:.2f}",
                    "auc": macro_auc(target_df, target_cols, blend_logit({"tabm": wt, "cat": wc, "xgb": wx}, val_mats)),
                    "wt": float(wt),
                    "wc": float(wc),
                    "wx": float(wx),
                }
            )

    grid_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    grid_df.to_csv(ARTIFACT_DIR / "coarse_blend_grid.csv", index=False)
    return grid_df


def best_override_for_target(
    *,
    current_valid: np.ndarray,
    xgb_valid: np.ndarray,
    y: np.ndarray,
    idx: int,
) -> dict[str, float | str]:
    current_auc = float(roc_auc_score(y, current_valid[:, idx])) if np.unique(y).size > 1 else 0.5

    candidates = []
    for mode, pred in [
        ("xgb_full", xgb_valid[:, idx]),
        ("tri_xgb20", sigmoid(0.8 * logit(current_valid[:, idx]) + 0.2 * logit(xgb_valid[:, idx]))),
        ("tri_xgb30", sigmoid(0.7 * logit(current_valid[:, idx]) + 0.3 * logit(xgb_valid[:, idx]))),
    ]:
        auc = float(roc_auc_score(y, pred)) if np.unique(y).size > 1 else 0.5
        candidates.append({"mode": mode, "auc": auc, "delta": auc - current_auc})
    best = max(candidates, key=lambda row: row["delta"])
    return {"tri_auc": current_auc, **best}


def apply_override(mode: str, current: np.ndarray, xgb: np.ndarray, idx: int) -> np.ndarray:
    out = current.copy()
    if mode == "xgb_full":
        out[:, idx] = xgb[:, idx]
    elif mode == "tri_xgb20":
        out[:, idx] = sigmoid(0.8 * logit(out[:, idx]) + 0.2 * logit(xgb[:, idx]))
    elif mode == "tri_xgb30":
        out[:, idx] = sigmoid(0.7 * logit(out[:, idx]) + 0.3 * logit(xgb[:, idx]))
    else:
        raise ValueError(f"Unsupported override mode: {mode}")
    return out


def analyze_targetwise_xgb(
    target_df: pd.DataFrame,
    target_cols: list[str],
    tri_valid: np.ndarray,
    xgb_valid: np.ndarray,
) -> tuple[pd.DataFrame, list[dict[str, object]], np.ndarray]:
    rows = []
    tri_plus = tri_valid.copy()
    tri_macro = macro_auc(target_df, target_cols, tri_valid)
    selected = []

    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        best = best_override_for_target(current_valid=tri_valid, xgb_valid=xgb_valid, y=y, idx=idx)
        rows.append({"target": target_name, **best})

    targetwise_df = pd.DataFrame(rows).sort_values("delta", ascending=False).reset_index(drop=True)
    targetwise_df.to_csv(ARTIFACT_DIR / "targetwise_xgb_opportunities.csv", index=False)

    for row in targetwise_df.itertuples(index=False):
        if float(row.delta) <= 0:
            continue
        idx = target_cols.index(row.target)
        cand = apply_override(str(row.mode), tri_plus, xgb_valid, idx)
        auc = macro_auc(target_df, target_cols, cand)
        delta = auc - tri_macro
        if delta <= 0:
            continue
        tri_plus = cand
        tri_macro = auc
        selected.append(
            {
                "target": row.target,
                "mode": row.mode,
                "macro_auc_after_apply": auc,
                "delta_vs_previous": delta,
            }
        )

    pd.DataFrame(selected).to_csv(ARTIFACT_DIR / "greedy_targetwise_overrides.csv", index=False)
    return targetwise_df, selected, tri_plus


def apply_overrides_by_rows(
    target_cols: list[str],
    base_pred: np.ndarray,
    xgb_pred: np.ndarray,
    rows: list[dict[str, object]],
) -> np.ndarray:
    out = base_pred.copy()
    for row in rows:
        idx = target_cols.index(str(row["target"]))
        out = apply_override(str(row["mode"]), out, xgb_pred, idx)
    return out


def apply_legacy_overrides(target_cols: list[str], tri_pred: np.ndarray, xgb_pred: np.ndarray) -> np.ndarray:
    out = tri_pred.copy()
    for override in LEGACY_RESTRAINED_TARGET_OVERRIDES:
        idx = target_cols.index(override["target"])
        out = apply_override(override["mode"], out, xgb_pred, idx)
    return out


def main() -> None:
    target_df, target_cols, val_mats, test_mats, test_customer_ids = load_inputs()

    coarse_grid = build_coarse_grid(target_df, target_cols, val_mats)

    current_best = blend_logit({"tabm": 0.70, "cat": 0.30}, val_mats)
    current_best_auc = macro_auc(target_df, target_cols, current_best)

    legacy_tri_valid = blend_logit(LEGACY_TRI_WEIGHTS, val_mats)
    legacy_tri_test = blend_logit(LEGACY_TRI_WEIGHTS, test_mats)
    legacy_tri_auc = macro_auc(target_df, target_cols, legacy_tri_valid)
    legacy_restrained_valid = apply_legacy_overrides(target_cols, legacy_tri_valid, val_mats["xgb"])
    legacy_restrained_test = apply_legacy_overrides(target_cols, legacy_tri_test, test_mats["xgb"])
    legacy_restrained_auc = macro_auc(target_df, target_cols, legacy_restrained_valid)

    best_grid_row = coarse_grid.iloc[0].to_dict()
    best_weights = {"tabm": float(best_grid_row["wt"]), "cat": float(best_grid_row["wc"]), "xgb": float(best_grid_row["wx"])}
    best_tri_valid = blend_logit(best_weights, val_mats)
    best_tri_test = blend_logit(best_weights, test_mats)
    best_tri_auc = macro_auc(target_df, target_cols, best_tri_valid)

    targetwise_df, greedy_selected, greedy_valid = analyze_targetwise_xgb(
        target_df=target_df,
        target_cols=target_cols,
        tri_valid=best_tri_valid,
        xgb_valid=val_mats["xgb"],
    )
    greedy_auc = macro_auc(target_df, target_cols, greedy_valid)
    greedy_test = apply_overrides_by_rows(target_cols, best_tri_test, test_mats["xgb"], greedy_selected)

    per_target_xgb = per_target_auc(target_df, target_cols, val_mats["xgb"]).rename(columns={"oof_auc": "xgb_missing_lite_auc"})
    per_target_old_xgb = pd.read_csv(
        ROOT
        / "output"
        / "kaggle-output"
        / "xgboost-multioutput-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_xgboost_multioutput_fs_v1_full_gpu"
        / "target_scores.csv"
    ).rename(columns={"oof_auc": "xgb_fs_v1_auc"})
    xgb_delta = per_target_xgb.merge(per_target_old_xgb, on="target", how="left")
    xgb_delta["delta_vs_xgb_fs_v1"] = xgb_delta["xgb_missing_lite_auc"] - xgb_delta["xgb_fs_v1_auc"]
    xgb_delta = xgb_delta.sort_values("delta_vs_xgb_fs_v1", ascending=False).reset_index(drop=True)
    xgb_delta.to_csv(ARTIFACT_DIR / "xgb_missing_lite_vs_fs_v1_target_deltas.csv", index=False)

    legacy_tri_submit_path = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_tri_logit_055_010_035_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, legacy_tri_test),
        output_path=legacy_tri_submit_path,
    )

    legacy_restrained_submit_path = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_tri_logit_055_010_035_restrained_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, legacy_restrained_test),
        output_path=legacy_restrained_submit_path,
    )

    best_tri_submit_path = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_besttri_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, best_tri_test),
        output_path=best_tri_submit_path,
    )

    greedy_submit_path = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_besttri_greedy_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, greedy_test),
        output_path=greedy_submit_path,
    )

    xgb_submit_path = SUBMISSIONS_DIR / "xgboost_multioutput_fs_v2_missing_lite_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, test_mats["xgb"]),
        output_path=xgb_submit_path,
    )

    summary = {
        "tabm_cat_logit30_auc": float(current_best_auc),
        "xgb_missing_lite_standalone_auc": float(macro_auc(target_df, target_cols, val_mats["xgb"])),
        "best_coarse_grid_row": best_grid_row,
        "legacy_tri_auc": float(legacy_tri_auc),
        "legacy_restrained_auc": float(legacy_restrained_auc),
        "legacy_restrained_delta_vs_current_best": float(legacy_restrained_auc - current_best_auc),
        "best_tri_auc": float(best_tri_auc),
        "best_tri_delta_vs_current_best": float(best_tri_auc - current_best_auc),
        "greedy_targetwise_auc": float(greedy_auc),
        "greedy_targetwise_delta_vs_current_best": float(greedy_auc - current_best_auc),
        "greedy_targetwise_delta_vs_best_tri": float(greedy_auc - best_tri_auc),
        "greedy_selected_targets": greedy_selected,
        "legacy_tri_submission_path": str(legacy_tri_submit_path),
        "legacy_restrained_submission_path": str(legacy_restrained_submit_path),
        "best_tri_submission_path": str(best_tri_submit_path),
        "greedy_submission_path": str(greedy_submit_path),
        "xgb_submission_path": str(xgb_submit_path),
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(targetwise_df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
