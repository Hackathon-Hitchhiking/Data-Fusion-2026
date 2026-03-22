from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.layout import project_root
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DEFAULT_OUT_DIR = ROOT / "artifacts" / "targetwise_router_v1"
DEFAULT_TARGETS = ROOT / "data" / "competition" / "train_target.parquet"
DEFAULT_GANDALF = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
DEFAULT_CATBOOST = ROOT / "artifacts" / "catboost_extended_specialists_gandalf_h12_router" / "specialist_val_predictions.parquet"
DEFAULT_LGBM = ROOT / "artifacts" / "lgbm_extended_specialists_gandalf_h12_v1" / "specialist_val_predictions.parquet"
DEFAULT_AG = ROOT / "output" / "kaggle-output" / "top1-recovery" / "artifacts" / "gpu_top1_ensemble_full_gpu" / "ag_val_submit.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a conservative per-target router over specialist families.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--target-file", type=Path, default=DEFAULT_TARGETS)
    p.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF)
    p.add_argument("--catboost-val", type=Path, default=DEFAULT_CATBOOST)
    p.add_argument("--lgbm-val", type=Path, default=DEFAULT_LGBM)
    p.add_argument("--autogluon-val", type=Path, default=DEFAULT_AG)
    return p.parse_args()


def load_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return normalize_prediction_columns(pd.read_parquet(path)).sort_values("customer_id").reset_index(drop=True)


def family_caps() -> dict[str, float]:
    return {
        "gandalf": 0.0,
        "catboost": 0.25,
        "lgbm": 0.20,
        "autogluon": 0.10,
    }


def delta_to_alpha(delta: float) -> float:
    if delta > 0.020:
        return 0.25
    if delta > 0.010:
        return 0.15
    if delta > 0.003:
        return 0.10
    return 0.0


def build_candidates(
    base: pd.DataFrame,
    family_frames: dict[str, pd.DataFrame],
    target_df: pd.DataFrame,
    pred_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    rows: list[dict[str, float | str]] = []
    routed_conservative = base[["customer_id"] + pred_cols].copy()
    routed_balanced = base[["customer_id"] + pred_cols].copy()
    routed_family_pick = base[["customer_id"] + pred_cols].copy()
    caps = family_caps()

    for pred_col in pred_cols:
        target_name = pred_col.replace("predict_", "target_")
        y_true = target_df[target_name].to_numpy()
        base_auc = roc_auc_score(y_true, base[pred_col])
        family_scores = {"gandalf": float(base_auc)}
        for family_name, frame in family_frames.items():
            family_scores[family_name] = float(roc_auc_score(y_true, frame[pred_col]))

        best_family = max((k for k in family_scores if k != "gandalf"), key=lambda k: family_scores[k], default="gandalf")
        best_auc = family_scores[best_family]
        delta = best_auc - base_auc
        alpha_conservative = min(delta_to_alpha(delta), caps.get(best_family, 0.0))
        alpha_balanced = min(alpha_conservative + 0.05 if alpha_conservative > 0 else 0.0, caps.get(best_family, 0.0))
        alpha_family_pick = caps.get(best_family, 0.0) if delta > 0 else 0.0

        rows.append(
            {
                "target": target_name,
                "gandalf_auc": float(base_auc),
                **{f"{family}_auc": float(auc) for family, auc in family_scores.items() if family != "gandalf"},
                "best_family": best_family,
                "best_auc": float(best_auc),
                "delta_auc": float(delta),
                "alpha_conservative": float(alpha_conservative),
                "alpha_balanced": float(alpha_balanced),
                "alpha_family_pick": float(alpha_family_pick),
            }
        )

        if alpha_conservative > 0:
            routed_conservative[pred_col] = (
                (1.0 - alpha_conservative) * base[pred_col] + alpha_conservative * family_frames[best_family][pred_col]
            ).astype("float32")
        if alpha_balanced > 0:
            routed_balanced[pred_col] = (
                (1.0 - alpha_balanced) * base[pred_col] + alpha_balanced * family_frames[best_family][pred_col]
            ).astype("float32")
        if alpha_family_pick > 0:
            routed_family_pick[pred_col] = (
                (1.0 - alpha_family_pick) * base[pred_col] + alpha_family_pick * family_frames[best_family][pred_col]
            ).astype("float32")

    per_target = pd.DataFrame(rows).sort_values(["delta_auc", "best_auc"], ascending=[False, False]).reset_index(drop=True)
    candidates = {
        "gandalf_base": base[["customer_id"] + pred_cols].copy(),
        "router_conservative": routed_conservative,
        "router_balanced": routed_balanced,
        "router_family_pick": routed_family_pick,
    }
    candidate_rows = []
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]
    for name, frame in candidates.items():
        candidate_rows.append({"candidate": name, "macro_auc": float(macro_auc(target_df, frame, target_cols))})
    candidate_df = pd.DataFrame(candidate_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    return per_target, candidate_df, candidates


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = load_frame(args.gandalf_val)
    if base is None:
        raise FileNotFoundError(args.gandalf_val)

    family_frames = {
        name: frame
        for name, frame in {
            "catboost": load_frame(args.catboost_val),
            "lgbm": load_frame(args.lgbm_val),
            "autogluon": load_frame(args.autogluon_val),
        }.items()
        if frame is not None
    }
    if not family_frames:
        raise RuntimeError("No specialist validation frames found")

    pred_cols = [c for c in base.columns if c != "customer_id"]
    target_df = pd.read_parquet(args.target_file)
    target_df = target_df.merge(base[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)

    for frame in family_frames.values():
        if not base["customer_id"].equals(frame["customer_id"]):
            raise ValueError("Validation frames must share the same customer_id order")

    per_target_df, candidate_df, candidates = build_candidates(base, family_frames, target_df, pred_cols)
    per_target_df.to_csv(args.out_dir / "per_target_router.csv", index=False)
    candidate_df.to_csv(args.out_dir / "candidate_macro_auc.csv", index=False)
    best_candidate = str(candidate_df.iloc[0]["candidate"])
    candidates[best_candidate].to_parquet(args.out_dir / "best_candidate_val_predictions.parquet", index=False)
    candidates["router_conservative"].to_parquet(args.out_dir / "router_conservative_val_predictions.parquet", index=False)
    candidates["router_balanced"].to_parquet(args.out_dir / "router_balanced_val_predictions.parquet", index=False)
    candidates["router_family_pick"].to_parquet(args.out_dir / "router_family_pick_val_predictions.parquet", index=False)

    summary = {
        "families": sorted(family_frames.keys()),
        "best_candidate": best_candidate,
        "best_macro_auc": float(candidate_df.iloc[0]["macro_auc"]),
        "base_macro_auc": float(candidate_df.loc[candidate_df["candidate"] == "gandalf_base", "macro_auc"].iloc[0]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(per_target_df.to_string(index=False), flush=True)
    print(candidate_df.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
