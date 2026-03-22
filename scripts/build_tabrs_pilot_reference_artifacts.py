from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.layout import project_root


ROOT = project_root()
ARTIFACT_DIR = ROOT / "artifacts" / "tabrs_pilot_inputs"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TABM_VALID = (
    ROOT
    / "output"
    / "kaggle-output"
    / "tabm-longrun-v3-fs-v1"
    / "pull_complete"
    / "artifacts"
    / "gpu_tabm_longrun_v3_fs_v1_full_gpu"
    / "validation_predictions.parquet"
)
CAT_VALID = (
    ROOT
    / "output"
    / "kaggle-output"
    / "catboost-multilabel-fs-v1"
    / "pull_complete"
    / "artifacts"
    / "gpu_catboost_multilabel_fs_v1_full_gpu"
    / "validation_predictions.parquet"
)
XGB_VALID = (
    ROOT
    / "output"
    / "kaggle-output"
    / "xgboost-multioutput-fs-v1"
    / "pull_complete"
    / "artifacts"
    / "gpu_xgboost_multioutput_fs_v1_full_gpu"
    / "validation_predictions.parquet"
)
BEST_SUBMISSION = (
    ROOT
    / "output"
    / "submissions"
    / "tabm_cat_xgb_tri_logit_055_010_035_restrained_targetwise_submission.parquet"
)


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def build_current_best_validation_reference() -> pd.DataFrame:
    tabm = pd.read_parquet(TABM_VALID).sort_values("customer_id").reset_index(drop=True)
    cat = pd.read_parquet(CAT_VALID).sort_values("customer_id").reset_index(drop=True)
    xgb = pd.read_parquet(XGB_VALID).sort_values("customer_id").reset_index(drop=True)

    pred_cols = [c for c in tabm.columns if c != "customer_id"]
    merged = (
        tabm.merge(cat, on="customer_id", how="inner", suffixes=("_tabm", "_cat"))
        .merge(xgb, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )

    out = pd.DataFrame({"customer_id": merged["customer_id"].astype("int32").values})
    for col in pred_cols:
        z = (
            0.55 * logit(merged[f"{col}_tabm"].to_numpy(dtype=np.float64))
            + 0.10 * logit(merged[f"{col}_cat"].to_numpy(dtype=np.float64))
            + 0.35 * logit(merged[col].to_numpy(dtype=np.float64))
        )
        out[col] = sigmoid(z)

    out["predict_2_8"] = merged["predict_2_8"].to_numpy(dtype=np.float64)
    out["predict_8_3"] = sigmoid(
        0.7 * logit(out["predict_8_3"].to_numpy(dtype=np.float64))
        + 0.3 * logit(merged["predict_8_3"].to_numpy(dtype=np.float64))
    )
    out["predict_2_3"] = sigmoid(
        0.7 * logit(out["predict_2_3"].to_numpy(dtype=np.float64))
        + 0.3 * logit(merged["predict_2_3"].to_numpy(dtype=np.float64))
    )
    return out


def main() -> None:
    current_best_val = build_current_best_validation_reference()
    current_best_submit = pd.read_parquet(BEST_SUBMISSION).sort_values("customer_id").reset_index(drop=True)
    val_ids = current_best_val[["customer_id"]].copy()

    current_best_val_path = ARTIFACT_DIR / "current_best_validation_reference.parquet"
    current_best_submit_path = ARTIFACT_DIR / "current_best_submission_reference.parquet"
    val_ids_path = ARTIFACT_DIR / "val_ids.parquet"
    summary_path = ARTIFACT_DIR / "summary.json"

    current_best_val.to_parquet(current_best_val_path, index=False)
    current_best_submit.to_parquet(current_best_submit_path, index=False)
    val_ids.to_parquet(val_ids_path, index=False)

    summary = {
        "current_best_validation_reference": str(current_best_val_path),
        "current_best_submission_reference": str(current_best_submit_path),
        "val_ids": str(val_ids_path),
        "validation_rows": int(len(current_best_val)),
        "test_rows": int(len(current_best_submit)),
        "target_count": int(len([c for c in current_best_val.columns if c != "customer_id"])),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
