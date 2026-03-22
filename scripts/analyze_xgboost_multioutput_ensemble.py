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


ARTIFACT_DIR = ROOT / "artifacts" / "post_xgboost_multioutput_blend_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

TABM_CAT_BASE_CAT_WEIGHT = 0.30
BEST_TRI_WEIGHTS = {"tabm": 0.55, "cat": 0.10, "xgb": 0.35}
RESTRAINED_TARGET_OVERRIDES = [
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
        / "xgboost-multioutput-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_xgboost_multioutput_fs_v1_full_gpu"
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
            / "xgboost-multioutput-fs-v1"
            / "pull_complete"
            / "artifacts"
            / "gpu_xgboost_multioutput_fs_v1_full_gpu"
            / "submission.parquet"
        )[pred_cols].to_numpy(dtype=np.float64),
    }
    test_customer_ids = pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")["customer_id"].to_numpy(dtype=np.int32)
    return merged[["customer_id"] + target_cols], target_cols, val_mats, test_mats, test_customer_ids


def build_coarse_grid(target_df: pd.DataFrame, target_cols: list[str], val_mats: dict[str, np.ndarray]) -> pd.DataFrame:
    rows = []
    rows.append({"kind": "solo", "name": "tabm", "auc": macro_auc(target_df, target_cols, val_mats["tabm"])})
    rows.append({"kind": "solo", "name": "cat", "auc": macro_auc(target_df, target_cols, val_mats["cat"])})
    rows.append({"kind": "solo", "name": "xgb", "auc": macro_auc(target_df, target_cols, val_mats["xgb"])})

    tabm_cat = blend_logit({"tabm": 0.70, "cat": 0.30}, val_mats)
    rows.append({"kind": "baseline", "name": "tabm_cat_logit30", "auc": macro_auc(target_df, target_cols, tabm_cat)})

    weights = np.round(np.arange(0.05, 0.55, 0.05), 2)
    for w in weights:
        rows.append(
            {
                "kind": "pair_logit",
                "name": f"tabm_xgb_logit_{w:.2f}",
                "auc": macro_auc(target_df, target_cols, blend_logit({"tabm": 1.0 - w, "xgb": w}, val_mats)),
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
                    "wt": wt,
                    "wc": wc,
                    "wx": wx,
                }
            )

    grid_df = pd.DataFrame(rows).sort_values("auc", ascending=False).reset_index(drop=True)
    grid_df.to_csv(ARTIFACT_DIR / "coarse_blend_grid.csv", index=False)
    return grid_df


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
        tri_auc = float(roc_auc_score(y, tri_valid[:, idx])) if np.unique(y).size > 1 else 0.5
        xgb_auc = float(roc_auc_score(y, xgb_valid[:, idx])) if np.unique(y).size > 1 else 0.5
        tri_xgb20 = sigmoid(0.8 * logit(tri_valid[:, idx]) + 0.2 * logit(xgb_valid[:, idx]))
        tri_xgb30 = sigmoid(0.7 * logit(tri_valid[:, idx]) + 0.3 * logit(xgb_valid[:, idx]))
        tri_xgb20_auc = float(roc_auc_score(y, tri_xgb20)) if np.unique(y).size > 1 else 0.5
        tri_xgb30_auc = float(roc_auc_score(y, tri_xgb30)) if np.unique(y).size > 1 else 0.5
        rows.append(
            {
                "target": target_name,
                "tri_auc": tri_auc,
                "xgb_auc": xgb_auc,
                "xgb_delta": xgb_auc - tri_auc,
                "tri_xgb20_auc": tri_xgb20_auc,
                "tri_xgb20_delta": tri_xgb20_auc - tri_auc,
                "tri_xgb30_auc": tri_xgb30_auc,
                "tri_xgb30_delta": tri_xgb30_auc - tri_auc,
                "best_delta": max(xgb_auc - tri_auc, tri_xgb20_auc - tri_auc, tri_xgb30_auc - tri_auc),
            }
        )

    targetwise_df = pd.DataFrame(rows).sort_values("best_delta", ascending=False).reset_index(drop=True)
    targetwise_df.to_csv(ARTIFACT_DIR / "targetwise_xgb_opportunities.csv", index=False)

    for override in RESTRAINED_TARGET_OVERRIDES:
        idx = target_cols.index(override["target"])
        cand = tri_plus.copy()
        if override["mode"] == "xgb_full":
            cand[:, idx] = xgb_valid[:, idx]
        elif override["mode"] == "tri_xgb30":
            cand[:, idx] = sigmoid(0.7 * logit(tri_plus[:, idx]) + 0.3 * logit(xgb_valid[:, idx]))
        elif override["mode"] == "tri_xgb20":
            cand[:, idx] = sigmoid(0.8 * logit(tri_plus[:, idx]) + 0.2 * logit(xgb_valid[:, idx]))
        else:
            raise ValueError(f"Unsupported override mode: {override['mode']}")

        auc = macro_auc(target_df, target_cols, cand)
        delta = auc - tri_macro
        tri_plus = cand
        tri_macro = auc
        selected.append(
            {
                "target": override["target"],
                "mode": override["mode"],
                "macro_auc_after_apply": auc,
                "delta_vs_previous": delta,
            }
        )

    pd.DataFrame(selected).to_csv(ARTIFACT_DIR / "restrained_targetwise_overrides.csv", index=False)
    return targetwise_df, selected, tri_plus


def apply_restrained_overrides(target_cols: list[str], tri_test: np.ndarray, xgb_test: np.ndarray) -> np.ndarray:
    out = tri_test.copy()
    for override in RESTRAINED_TARGET_OVERRIDES:
        idx = target_cols.index(override["target"])
        if override["mode"] == "xgb_full":
            out[:, idx] = xgb_test[:, idx]
        elif override["mode"] == "tri_xgb30":
            out[:, idx] = sigmoid(0.7 * logit(out[:, idx]) + 0.3 * logit(xgb_test[:, idx]))
        elif override["mode"] == "tri_xgb20":
            out[:, idx] = sigmoid(0.8 * logit(out[:, idx]) + 0.2 * logit(xgb_test[:, idx]))
        else:
            raise ValueError(f"Unsupported override mode: {override['mode']}")
    return out


def main() -> None:
    target_df, target_cols, val_mats, test_mats, test_customer_ids = load_inputs()

    coarse_grid = build_coarse_grid(target_df, target_cols, val_mats)

    current_best = blend_logit({"tabm": 0.70, "cat": 0.30}, val_mats)
    current_best_auc = macro_auc(target_df, target_cols, current_best)

    tri_valid = blend_logit(BEST_TRI_WEIGHTS, val_mats)
    tri_test = blend_logit(BEST_TRI_WEIGHTS, test_mats)
    tri_auc = macro_auc(target_df, target_cols, tri_valid)

    targetwise_df, restrained_selected, restrained_valid = analyze_targetwise_xgb(
        target_df=target_df,
        target_cols=target_cols,
        tri_valid=tri_valid,
        xgb_valid=val_mats["xgb"],
    )
    restrained_auc = macro_auc(target_df, target_cols, restrained_valid)
    restrained_test = apply_restrained_overrides(target_cols, tri_test, test_mats["xgb"])

    tri_submit_path = SUBMISSIONS_DIR / "tabm_cat_xgb_tri_logit_055_010_035_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, tri_test),
        output_path=tri_submit_path,
    )

    restrained_submit_path = (
        SUBMISSIONS_DIR / "tabm_cat_xgb_tri_logit_055_010_035_restrained_targetwise_submission.parquet"
    )
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, restrained_test),
        output_path=restrained_submit_path,
    )

    xgb_submit_path = SUBMISSIONS_DIR / "xgboost_multioutput_fs_v1_submission.parquet"
    write_submission_like_sample(
        customer_ids=test_customer_ids,
        prediction_frame=to_prediction_frame(target_cols, test_mats["xgb"]),
        output_path=xgb_submit_path,
    )

    summary = {
        "current_best_tabm_cat_logit30_auc": float(current_best_auc),
        "xgb_standalone_auc": float(macro_auc(target_df, target_cols, val_mats["xgb"])),
        "best_coarse_grid_row": coarse_grid.iloc[0].to_dict(),
        "tri_logit_055_010_035_auc": float(tri_auc),
        "tri_logit_055_010_035_delta_vs_current_best": float(tri_auc - current_best_auc),
        "restrained_targetwise_auc": float(restrained_auc),
        "restrained_targetwise_delta_vs_current_best": float(restrained_auc - current_best_auc),
        "restrained_targetwise_delta_vs_tri": float(restrained_auc - tri_auc),
        "restrained_selected_targets": restrained_selected,
        "tri_submission_path": str(tri_submit_path),
        "restrained_submission_path": str(restrained_submit_path),
        "xgb_submission_path": str(xgb_submit_path),
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
