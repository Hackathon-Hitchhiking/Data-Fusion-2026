from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from lib.meta_views import (
    ROOT,
    align_prediction_frames,
    build_family_map,
    load_holdout_targets,
    load_prediction_frame,
    safe_logit,
    target_to_pred,
)
from lib.metrics import macro_auc


DEFAULT_TABM_VAL = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_GANDALF_VAL = ROOT / "artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/family_head_pilot_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run family-structured meta-head pilot on holdout predictions.")
    parser.add_argument("--tabm-val", type=Path, default=DEFAULT_TABM_VAL)
    parser.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.10, 0.20, 0.30, 0.40])
    return parser.parse_args()


def build_family_features(
    tabm_df: pd.DataFrame,
    gandalf_df: pd.DataFrame,
    target_name: str,
    families: dict[str, list[str]],
) -> pd.DataFrame:
    pred_col = target_to_pred(target_name)
    _, family_id, _ = target_name.split("_")
    family_targets = [t for t in families[family_id] if t != target_name]
    out = pd.DataFrame({"customer_id": tabm_df["customer_id"].values})
    out["tabm_score"] = tabm_df[pred_col].astype("float32").values
    out["gandalf_score"] = gandalf_df[pred_col].astype("float32").values
    out["tabm_logit"] = safe_logit(out["tabm_score"]).astype("float32")
    out["gandalf_logit"] = safe_logit(out["gandalf_score"]).astype("float32")
    out["gandalf_minus_tabm"] = (out["gandalf_score"] - out["tabm_score"]).astype("float32")

    family_pred_cols = [target_to_pred(t) for t in family_targets]
    if family_pred_cols:
        family_block = tabm_df[family_pred_cols].astype("float32")
        out["family_mean"] = family_block.mean(axis=1).astype("float32")
        out["family_std"] = family_block.std(axis=1).fillna(0.0).astype("float32")
        out["family_max"] = family_block.max(axis=1).astype("float32")
        for other_target in family_targets:
            other_col = target_to_pred(other_target)
            out[f"fam__{other_col}"] = tabm_df[other_col].astype("float32").values
            out[f"famlogit__{other_col}"] = safe_logit(tabm_df[other_col]).astype("float32")
    return out


def crossfit_meta(features: pd.DataFrame, y: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    x = features.drop(columns=["customer_id"]).to_numpy(dtype=np.float32)
    oof = np.zeros(len(features), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x, y):
        train_y = y[train_idx]
        if np.unique(train_y).size < 2:
            oof[val_idx] = float(train_y.mean())
            continue
        model = LogisticRegression(
            C=0.25,
            max_iter=700,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(x[train_idx], train_y)
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1].astype(np.float32)
    return oof


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    tabm_df = pd.read_parquet(args.tabm_val)
    pred_cols = [c for c in tabm_df.columns if c.startswith("predict_")]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]
    named_frames = {
        "tabm": load_prediction_frame(args.tabm_val, target_cols),
        "gandalf": load_prediction_frame(args.gandalf_val, target_cols),
    }
    named_frames = align_prediction_frames(named_frames)
    tabm_df = named_frames["tabm"]
    gandalf_df = named_frames["gandalf"]
    y_df = load_holdout_targets(tabm_df["customer_id"], target_cols)
    families = build_family_map(target_cols)

    meta_map: dict[str, np.ndarray] = {}
    target_rows: list[dict[str, object]] = []
    base_pred_df = tabm_df.copy()

    for target_name in target_cols:
        feature_df = build_family_features(tabm_df, gandalf_df, target_name, families)
        y = y_df[target_name].to_numpy(dtype=np.int8)
        meta_proba = crossfit_meta(feature_df, y, seed=args.seed, folds=args.folds)
        meta_map[target_name] = meta_proba
        target_rows.append(
            {
                "target": target_name,
                "family_id": target_name.split("_")[1],
                "base_auc": float(macro_auc(y_df[["customer_id", target_name]], base_pred_df[["customer_id", target_to_pred(target_name)]], [target_name])),
                "meta_auc": float(
                    macro_auc(
                        y_df[["customer_id", target_name]],
                        pd.DataFrame({"customer_id": y_df["customer_id"].values, target_to_pred(target_name): meta_proba}),
                        [target_name],
                    )
                ),
            }
        )

    pd.DataFrame(target_rows).to_csv(args.artifact_dir / "per_target_family_auc.csv", index=False)

    candidate_rows: list[dict[str, object]] = []
    best_alpha = None
    best_score = -np.inf
    best_pred_df = None
    for alpha in args.alphas:
        pred_df = base_pred_df.copy()
        for target_name in target_cols:
            pred_col = target_to_pred(target_name)
            base_scores = pred_df[pred_col].to_numpy(dtype=np.float32)
            meta_scores = meta_map[target_name]
            pred_df[pred_col] = np.clip(
                base_scores + float(alpha) * (meta_scores - base_scores),
                1e-6,
                1.0 - 1e-6,
            ).astype(np.float32)
        score = macro_auc(y_df, pred_df, target_cols)
        candidate_rows.append({"alpha": float(alpha), "macro_auc_all": float(score)})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred_df = pred_df.copy()

    pd.DataFrame(candidate_rows).to_csv(args.artifact_dir / "candidate_macro_auc.csv", index=False)
    if best_pred_df is not None:
        best_pred_df.to_parquet(args.artifact_dir / "best_candidate_val_predictions.parquet", index=False)

    summary = {
        "best_alpha": best_alpha,
        "best_macro_auc": float(best_score),
        "base_macro_auc": float(macro_auc(y_df, base_pred_df, target_cols)),
        "family_count": len(families),
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
