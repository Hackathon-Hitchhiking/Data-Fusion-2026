from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def main() -> None:
    root = Path.cwd()
    submissions_dir = root / "output" / "submissions"
    submissions_dir.mkdir(parents=True, exist_ok=True)

    tabm_path = submissions_dir / "tabm_longrun_v3_fs_v1_submission.parquet"
    catboost_path = (
        root
        / "output"
        / "kaggle-output"
        / "catboost-multilabel-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_catboost_multilabel_fs_v1_full_gpu"
        / "submission.parquet"
    )
    standalone_out = submissions_dir / "catboost_multilabel_fs_v1_submission.parquet"
    blend_out = submissions_dir / "tabm_fs_v1_catboost_multilabel_logit30_submission.parquet"
    meta_out = submissions_dir / "tabm_fs_v1_catboost_multilabel_logit30_submission.json"

    tabm = pd.read_parquet(tabm_path)
    catboost = pd.read_parquet(catboost_path)

    if not tabm["customer_id"].equals(catboost["customer_id"]):
        raise ValueError("customer_id order mismatch between TabM and CatBoost submissions")

    target_cols = [c for c in tabm.columns if c != "customer_id"]
    catboost_target_cols = [c for c in catboost.columns if c != "customer_id"]
    renamed_catboost_cols = {c: c.replace("target_", "predict_", 1) for c in catboost_target_cols}
    catboost = catboost.rename(columns=renamed_catboost_cols)
    if target_cols != [c for c in catboost.columns if c != "customer_id"]:
        raise ValueError("Target columns mismatch between TabM and CatBoost submissions after normalization")

    catboost[target_cols] = catboost[target_cols].astype("float64")
    catboost.to_parquet(standalone_out, index=False)

    w_cat = 0.3
    w_tabm = 1.0 - w_cat

    tabm_mat = tabm[target_cols].to_numpy(dtype=np.float64)
    cat_mat = catboost[target_cols].to_numpy(dtype=np.float64)
    blend_mat = sigmoid(w_tabm * logit(tabm_mat) + w_cat * logit(cat_mat)).astype(np.float64)

    blend = pd.DataFrame(blend_mat, columns=target_cols)
    blend.insert(0, "customer_id", tabm["customer_id"].to_numpy())
    blend.to_parquet(blend_out, index=False)

    meta = {
        "type": "logit_blend",
        "tabm_submission": str(tabm_path),
        "catboost_submission": str(catboost_path),
        "catboost_weight": w_cat,
        "tabm_weight": w_tabm,
        "output_path": str(blend_out),
    }
    meta_out.write_text(json.dumps(meta, indent=2) + "\n")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
