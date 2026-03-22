from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.data_loading import load_labeled_frame
from lib.layout import project_root
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DEFAULT_V1 = ROOT / "output/kaggle-output/tabm-backbone/watch_v4/artifacts/gpu_tabm_full_gpu/validation_predictions.parquet"
DEFAULT_V3 = ROOT / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
DEFAULT_OUT = ROOT / "artifacts/tabm_family_ensemble_local_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local pairwise TabM family ensemble check.")
    p.add_argument("--left", type=Path, default=DEFAULT_V1)
    p.add_argument("--right", type=Path, default=DEFAULT_V3)
    p.add_argument("--left-name", type=str, default="tabm_v1")
    p.add_argument("--right-name", type=str, default="tabm_v3")
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def load_pred(path: Path) -> pd.DataFrame:
    return normalize_prediction_columns(pd.read_parquet(path)).sort_values("customer_id").reset_index(drop=True)


def rank_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"customer_id": df["customer_id"].astype("int32").values})
    pred_cols = [c for c in df.columns if c != "customer_id"]
    n = len(df)
    for col in pred_cols:
        ranks = df[col].rank(method="average", pct=True).astype("float32")
        out[col] = ranks.values
    return out


def blend_frame(left: pd.DataFrame, right: pd.DataFrame, right_weight: float) -> pd.DataFrame:
    pred_cols = [c for c in left.columns if c != "customer_id"]
    out = pd.DataFrame({"customer_id": left["customer_id"].astype("int32").values})
    left_w = 1.0 - right_weight
    for col in pred_cols:
        out[col] = (left_w * left[col].to_numpy(dtype=np.float32) + right_weight * right[col].to_numpy(dtype=np.float32)).astype(np.float32)
    return out


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    left = load_pred(args.left)
    right = load_pred(args.right)
    if not left["customer_id"].equals(right["customer_id"]):
        raise ValueError("Validation prediction frames are not aligned by customer_id")

    labeled = load_labeled_frame()
    target_cols = sorted([c for c in labeled.columns if c.startswith("target_")], key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])))
    y_df = labeled[["customer_id"] + target_cols].copy()
    y_df = y_df[y_df["customer_id"].isin(set(left["customer_id"]))].sort_values("customer_id").reset_index(drop=True)

    left_macro = float(macro_auc(y_df, left, target_cols))
    right_macro = float(macro_auc(y_df, right, target_cols))
    left_rank = rank_frame(left)
    right_rank = rank_frame(right)

    weights = [0.8, 0.7, 0.5]
    rows: list[dict[str, object]] = []
    best_name = None
    best_macro = -np.inf
    best_frame = None

    for right_weight in weights:
        left_weight = 1.0 - right_weight
        for mode, ldf, rdf in [
            ("weighted_avg", left, right),
            ("rank_avg", left_rank, right_rank),
        ]:
            candidate = blend_frame(ldf, rdf, right_weight=right_weight)
            full_macro = float(macro_auc(y_df, candidate, target_cols))
            rows.append(
                {
                    "candidate": f"{mode}_{args.left_name}_{left_weight:.1f}_{args.right_name}_{right_weight:.1f}",
                    "mode": mode,
                    "left_weight": left_weight,
                    "right_weight": right_weight,
                    "full_macro_auc": full_macro,
                    "delta_vs_left": full_macro - left_macro,
                    "delta_vs_right": full_macro - right_macro,
                }
            )
            if full_macro > best_macro:
                best_macro = full_macro
                best_name = rows[-1]["candidate"]
                best_frame = candidate.copy()

    result_df = pd.DataFrame(rows).sort_values("full_macro_auc", ascending=False).reset_index(drop=True)
    result_df.to_csv(args.artifact_dir / "candidate_scores.csv", index=False)
    if best_frame is not None:
        best_frame.to_parquet(args.artifact_dir / "best_candidate_validation_predictions.parquet", index=False)

        per_target_rows = []
        pred_cols = [c for c in best_frame.columns if c != "customer_id"]
        for target_name in target_cols:
            col = target_name.replace("target_", "predict_")
            base_auc = float(macro_auc(y_df[["customer_id", target_name]], right[["customer_id", col]], [target_name]))
            ens_auc = float(macro_auc(y_df[["customer_id", target_name]], best_frame[["customer_id", col]], [target_name]))
            per_target_rows.append(
                {
                    "target": target_name,
                    "base_right_auc": base_auc,
                    "ensemble_auc": ens_auc,
                    "delta_vs_right": ens_auc - base_auc,
                }
            )
        pd.DataFrame(per_target_rows).sort_values("delta_vs_right", ascending=False).to_csv(
            args.artifact_dir / "best_candidate_per_target_delta.csv", index=False
        )

    summary = {
        "left_name": args.left_name,
        "right_name": args.right_name,
        "left_macro_auc": left_macro,
        "right_macro_auc": right_macro,
        "best_candidate": best_name,
        "best_macro_auc": best_macro,
        "best_delta_vs_right": best_macro - right_macro,
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
