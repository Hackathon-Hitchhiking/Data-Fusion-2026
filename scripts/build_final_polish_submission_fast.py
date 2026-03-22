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


ARTIFACT_DIR = ROOT / "artifacts" / "final_polish_submission_fast_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

BASE_SUBMISSION = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_besttri_greedy_submission.parquet"
STACK_V2_SUBMISSION = SUBMISSIONS_DIR / "three_model_stack_v2_01_two_stage_core_plus_ridge_4p0_m0p0001_submission.parquet"
STACK_V3_SUBMISSION = SUBMISSIONS_DIR / "three_model_stack_v3_01_two_stage_core_plus_logistic_0p05_m0p0_submission.parquet"
OUTPUT_SUBMISSION = SUBMISSIONS_DIR / "tabm_cat_xgb_v2lite_final_polish_submission.parquet"

TARGET_ALPHA_PLAN = [
    {"target": "target_2_3", "donor": "stack_mean", "alpha": 0.20},
    {"target": "target_2_8", "donor": "stack_v2", "alpha": 0.12},
    {"target": "target_8_3", "donor": "stack_mean", "alpha": 0.12},
    {"target": "target_10_1", "donor": "stack_mean", "alpha": 0.08},
    {"target": "target_9_7", "donor": "stack_v3", "alpha": 0.05},
    {"target": "target_3_1", "donor": "stack_v3", "alpha": 0.05},
]


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def main() -> None:
    base_df = pd.read_parquet(BASE_SUBMISSION)
    stack_v2_df = pd.read_parquet(STACK_V2_SUBMISSION)
    stack_v3_df = pd.read_parquet(STACK_V3_SUBMISSION)

    pred_cols = [c for c in base_df.columns if c.startswith("predict_")]
    customer_ids = base_df["customer_id"].to_numpy(dtype=np.int32)

    base_mat = base_df[pred_cols].to_numpy(dtype=np.float64)
    stack_v2_mat = stack_v2_df[pred_cols].to_numpy(dtype=np.float64)
    stack_v3_mat = stack_v3_df[pred_cols].to_numpy(dtype=np.float64)
    stack_mean_mat = sigmoid(0.5 * logit(stack_v2_mat) + 0.5 * logit(stack_v3_mat))

    donor_mats = {
        "stack_v2": stack_v2_mat,
        "stack_v3": stack_v3_mat,
        "stack_mean": stack_mean_mat,
    }

    final_mat = base_mat.copy()
    for row in TARGET_ALPHA_PLAN:
        pred_name = f"predict_{row['target'].split('target_', 1)[1]}"
        idx = pred_cols.index(pred_name)
        donor = donor_mats[row["donor"]]
        alpha = float(row["alpha"])
        final_mat[:, idx] = sigmoid((1.0 - alpha) * logit(final_mat[:, idx]) + alpha * logit(donor[:, idx]))

    write_submission_like_sample(
        customer_ids=customer_ids,
        prediction_frame=pd.DataFrame(final_mat, columns=pred_cols, dtype=np.float64),
        output_path=OUTPUT_SUBMISSION,
    )

    summary = {
        "base_submission": str(BASE_SUBMISSION),
        "stack_v2_submission": str(STACK_V2_SUBMISSION),
        "stack_v3_submission": str(STACK_V3_SUBMISSION),
        "output_submission": str(OUTPUT_SUBMISSION),
        "target_alpha_plan": TARGET_ALPHA_PLAN,
        "note": "Fast final polish: online-positive stack donors only, conservative target-wise logit blending over the current best v2lite greedy base.",
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
