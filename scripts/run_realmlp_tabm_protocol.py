from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.data_loading import read_competition_parquet
from lib.layout import project_root


ROOT = project_root()
DEFAULT_TABM_PRED = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_REALMLP_ARTIFACT_DIR = ROOT / "output/kaggle-output/realmlp-tdlike-ovr-v3/current_pull/artifacts/gpu_realmlp_tdlike_full_gpu"
DEFAULT_OUT = ROOT / "artifacts/realmlp_vs_tabm_protocol_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare RealMLP vs TabM v3 on the shared validation split.")
    p.add_argument("--tabm-pred", type=Path, default=DEFAULT_TABM_PRED)
    p.add_argument("--realmlp-artifact-dir", type=Path, default=DEFAULT_REALMLP_ARTIFACT_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return p.parse_args()


def target_to_pred(target: str) -> str:
    return target.replace("target_", "predict_")


def pred_to_target(pred: str) -> str:
    return pred.replace("predict_", "target_")


def single_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_score))


def macro_auc(target_df: pd.DataFrame, pred_df: pd.DataFrame, target_cols: list[str]) -> float:
    return float(np.mean([single_auc(target_df[t].to_numpy(), pred_df[target_to_pred(t)].to_numpy()) for t in target_cols]))


def rank_average(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    sa = pd.Series(a).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    sb = pd.Series(b).rank(method="average", pct=True).to_numpy(dtype=np.float64)
    return ((sa + sb) / 2.0).astype(np.float64)


def main() -> None:
    args = parse_args()
    realmlp_pred_path = args.realmlp_artifact_dir / "validation_predictions.parquet"
    if not realmlp_pred_path.exists():
        raise FileNotFoundError(f"RealMLP validation predictions not found: {realmlp_pred_path}")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    tabm_pred = pd.read_parquet(args.tabm_pred).sort_values("customer_id").reset_index(drop=True)
    realmlp_pred = pd.read_parquet(realmlp_pred_path).sort_values("customer_id").reset_index(drop=True)
    if not tabm_pred["customer_id"].equals(realmlp_pred["customer_id"]):
        realmlp_pred = tabm_pred[["customer_id"]].merge(realmlp_pred, on="customer_id", how="inner")
        if len(realmlp_pred) != len(tabm_pred):
            raise ValueError("RealMLP validation predictions do not align with TabM validation ids.")

    target_df = read_competition_parquet("train_target.parquet")
    target_df = tabm_pred[["customer_id"]].merge(target_df, on="customer_id", how="left")
    target_cols = sorted([c for c in target_df.columns if c.startswith("target_")], key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])))

    standalone_rows = []
    for target in target_cols:
        pred_col = target_to_pred(target)
        y = target_df[target].to_numpy()
        tabm_score = tabm_pred[pred_col].to_numpy()
        realmlp_score = realmlp_pred[pred_col].to_numpy()
        corr = float(np.corrcoef(tabm_score, realmlp_score)[0, 1]) if np.std(tabm_score) > 0 and np.std(realmlp_score) > 0 else 0.0
        standalone_rows.append(
            {
                "target": target,
                "family": int(target.split("_")[1]),
                "member": int(target.split("_")[2]),
                "auc_tabm_v3": single_auc(y, tabm_score),
                "auc_realmlp": single_auc(y, realmlp_score),
                "delta_realmlp_vs_tabm_v3": single_auc(y, realmlp_score) - single_auc(y, tabm_score),
                "prediction_correlation": corr,
            }
        )
    per_target = pd.DataFrame(standalone_rows).sort_values("delta_realmlp_vs_tabm_v3").reset_index(drop=True)
    per_target.to_csv(args.out_dir / "per_target_comparison.csv", index=False)

    candidate_rows = []
    candidate_pred_frames: dict[str, pd.DataFrame] = {}
    for weight in [0.2, 0.3, 0.5]:
        frame = pd.DataFrame({"customer_id": tabm_pred["customer_id"].values})
        for target in target_cols:
            pred_col = target_to_pred(target)
            frame[pred_col] = ((1.0 - weight) * tabm_pred[pred_col].to_numpy() + weight * realmlp_pred[pred_col].to_numpy()).astype(np.float64)
        candidate_name = f"weighted_realmlp_{weight:.1f}"
        candidate_pred_frames[candidate_name] = frame
        candidate_rows.append({"candidate": candidate_name, "macro_auc": macro_auc(target_df, frame, target_cols)})

    rank_frame = pd.DataFrame({"customer_id": tabm_pred["customer_id"].values})
    for target in target_cols:
        pred_col = target_to_pred(target)
        rank_frame[pred_col] = rank_average(tabm_pred[pred_col].to_numpy(), realmlp_pred[pred_col].to_numpy())
    candidate_pred_frames["rank_average"] = rank_frame
    candidate_rows.append({"candidate": "rank_average", "macro_auc": macro_auc(target_df, rank_frame, target_cols)})

    tabm_macro = macro_auc(target_df, tabm_pred, target_cols)
    realmlp_macro = macro_auc(target_df, realmlp_pred, target_cols)
    candidates = pd.DataFrame(candidate_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    candidates["delta_vs_tabm_v3"] = candidates["macro_auc"] - tabm_macro
    candidates.to_csv(args.out_dir / "ensemble_candidate_scores.csv", index=False)

    best_candidate = candidates.iloc[0].to_dict()
    summary = {
        "tabm_v3_macro_auc": tabm_macro,
        "realmlp_macro_auc": realmlp_macro,
        "realmlp_delta_vs_tabm_v3": realmlp_macro - tabm_macro,
        "best_ensemble_candidate": best_candidate,
        "positive_realmlp_targets": int((per_target["delta_realmlp_vs_tabm_v3"] > 0).sum()),
        "low_corr_positive_targets": int(((per_target["delta_realmlp_vs_tabm_v3"] > 0) & (per_target["prediction_correlation"] < 0.98)).sum()),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
