from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auc_from_matrix(target_df: pd.DataFrame, target_cols: list[str], mat: np.ndarray) -> float:
    scores: list[float] = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy()
        s = mat[:, idx]
        scores.append(float(roc_auc_score(y, s)) if np.unique(y).size > 1 else 0.5)
    return float(np.mean(scores))


def main() -> None:
    base = "output/kaggle-output"
    artifact_dir = Path("artifacts/post_catboost_blend_analysis_v1")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cat_pred = pd.read_parquet(
        f"{base}/catboost-multilabel-fs-v1/pull_complete/artifacts/gpu_catboost_multilabel_fs_v1_full_gpu/validation_predictions.parquet"
    )
    tabm_pred = pd.read_parquet(
        f"{base}/tabm-longrun-v3-fs-v1/pull_complete/artifacts/gpu_tabm_longrun_v3_fs_v1_full_gpu/validation_predictions.parquet"
    )
    target = pd.read_parquet("data/competition/train_target.parquet")

    merged = target.merge(tabm_pred, on="customer_id", how="inner")
    merged = merged.merge(cat_pred, on="customer_id", how="inner", suffixes=("_tabm", "_cat"))

    target_cols = [c for c in target.columns if c != "customer_id"]
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]

    tabm_mat = merged[[f"{p}_tabm" for p in pred_cols]].to_numpy(dtype=np.float64)
    cat_mat = merged[[f"{p}_cat" for p in pred_cols]].to_numpy(dtype=np.float64)

    tabm_macro_auc = macro_auc_from_matrix(merged, target_cols, tabm_mat)
    cat_macro_auc = macro_auc_from_matrix(merged, target_cols, cat_mat)
    print({"rows": len(merged), "same_rows": len(merged) == len(tabm_pred) == len(cat_pred)})
    print({"tabm_macro_auc": tabm_macro_auc})
    print({"cat_macro_auc": cat_macro_auc})

    best_name = "tabm"
    best_weight = None
    best_score = tabm_macro_auc
    blend_rows: list[dict[str, object]] = [
        {"blend": "tabm", "cat_weight": 0.0, "macro_auc": tabm_macro_auc},
        {"blend": "catboost", "cat_weight": 1.0, "macro_auc": cat_macro_auc},
    ]

    for weight in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        prob_blend = (1.0 - weight) * tabm_mat + weight * cat_mat
        score = macro_auc_from_matrix(merged, target_cols, prob_blend)
        blend_rows.append({"blend": "prob", "cat_weight": weight, "macro_auc": score})
        print({"blend": "prob", "cat_weight": weight, "macro_auc": score})
        if score > best_score:
            best_name, best_weight, best_score = "prob", weight, score

    def logit(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 1e-8, 1.0 - 1e-8)
        return np.log(x / (1.0 - x))

    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    tabm_logit = logit(tabm_mat)
    cat_logit = logit(cat_mat)
    for weight in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
        logit_blend = sigmoid((1.0 - weight) * tabm_logit + weight * cat_logit)
        score = macro_auc_from_matrix(merged, target_cols, logit_blend)
        blend_rows.append({"blend": "logit", "cat_weight": weight, "macro_auc": score})
        print({"blend": "logit", "cat_weight": weight, "macro_auc": score})
        if score > best_score:
            best_name, best_weight, best_score = "logit", weight, score

    fallback = tabm_mat.copy()
    fallback_targets: list[tuple[str, float]] = []
    for idx, target_name in enumerate(target_cols):
        y = merged[target_name].to_numpy()
        tabm_auc = float(roc_auc_score(y, tabm_mat[:, idx]))
        cat_auc = float(roc_auc_score(y, cat_mat[:, idx]))
        if cat_auc > tabm_auc:
            fallback[:, idx] = cat_mat[:, idx]
            fallback_targets.append((target_name, cat_auc - tabm_auc))
    fallback_score = macro_auc_from_matrix(merged, target_cols, fallback)
    blend_rows.append({"blend": "target_fallback", "cat_weight": None, "macro_auc": fallback_score})
    print({"blend": "target_fallback", "macro_auc": fallback_score, "targets": fallback_targets})
    if fallback_score > best_score:
        best_name, best_weight, best_score = "target_fallback", None, fallback_score

    pd.DataFrame(blend_rows).sort_values(["macro_auc", "blend"], ascending=[False, True]).to_csv(
        artifact_dir / "blend_grid.csv", index=False
    )
    summary = {
        "rows": int(len(merged)),
        "tabm_macro_auc": float(tabm_macro_auc),
        "catboost_macro_auc": float(cat_macro_auc),
        "best_blend": best_name,
        "best_cat_weight": best_weight,
        "best_macro_auc": float(best_score),
        "fallback_targets": [{"target": t, "delta_vs_tabm": float(d)} for t, d in fallback_targets],
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print({"best_blend": best_name, "cat_weight": best_weight, "best_macro_auc": best_score})


if __name__ == "__main__":
    main()
