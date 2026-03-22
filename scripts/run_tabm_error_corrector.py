from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from lib.data_loading import load_labeled_frame
from lib.layout import project_root
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DEFAULT_TABM_VAL = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_GANDALF_VAL = ROOT / "artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet"
DEFAULT_SPECIALIST_VAL = (
    ROOT / "artifacts/catboost_extended_specialists_gandalf_h12_router/specialist_val_predictions.parquet"
)
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/tabm_error_corrector_v1"
DEFAULT_TARGETS = [
    "target_9_6",
    "target_9_3",
    "target_3_1",
    "target_2_4",
    "target_6_1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual error-corrector experiments on top of TabM.")
    parser.add_argument("--tabm-val", type=Path, default=DEFAULT_TABM_VAL)
    parser.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    parser.add_argument("--specialist-val", type=Path, default=DEFAULT_SPECIALIST_VAL)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--raw-top-k", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.30])
    return parser.parse_args()


def load_prediction_frame(path: Path, required_targets: list[str]) -> pd.DataFrame:
    df = normalize_prediction_columns(pd.read_parquet(path))
    keep = ["customer_id"] + [target.replace("target_", "predict_") for target in required_targets]
    return df[keep].copy()


def load_raw_features(customer_ids: pd.Series, raw_top_k: int) -> pd.DataFrame:
    if raw_top_k <= 0:
        return pd.DataFrame({"customer_id": customer_ids.astype("int32")})
    top_features = json.loads(TOP_FEATURES_FILE.read_text())[:raw_top_k]
    labeled = load_labeled_frame(extra_feature_cols=top_features)
    raw = labeled[["customer_id"] + [c for c in top_features if c in labeled.columns]].copy()
    raw = raw[raw["customer_id"].isin(customer_ids)].copy()
    raw = raw.sort_values("customer_id").reset_index(drop=True)
    for col in raw.columns:
        if col == "customer_id":
            continue
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
        median = float(raw[col].median()) if raw[col].notna().any() else 0.0
        raw[col] = raw[col].fillna(median).astype("float32")
    return raw


def build_meta_features(
    base_df: pd.DataFrame,
    gandalf_df: pd.DataFrame,
    specialist_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    pred_col = target_name.replace("target_", "predict_")
    base = base_df[["customer_id", pred_col]].rename(columns={pred_col: "tabm_score"})
    gandalf = gandalf_df[["customer_id", pred_col]].rename(columns={pred_col: "gandalf_score"})
    specialist = specialist_df[["customer_id", pred_col]].rename(columns={pred_col: "specialist_score"})
    merged = base.merge(gandalf, on="customer_id", how="inner").merge(specialist, on="customer_id", how="inner")
    if len(raw_df.columns) > 1:
        merged = merged.merge(raw_df, on="customer_id", how="left")
    eps = 1e-6
    merged["tabm_logit"] = np.log(np.clip(merged["tabm_score"], eps, 1 - eps) / np.clip(1 - merged["tabm_score"], eps, 1 - eps))
    merged["specialist_logit"] = np.log(np.clip(merged["specialist_score"], eps, 1 - eps) / np.clip(1 - merged["specialist_score"], eps, 1 - eps))
    merged["gandalf_logit"] = np.log(np.clip(merged["gandalf_score"], eps, 1 - eps) / np.clip(1 - merged["gandalf_score"], eps, 1 - eps))
    merged["specialist_minus_tabm"] = merged["specialist_score"] - merged["tabm_score"]
    merged["gandalf_minus_tabm"] = merged["gandalf_score"] - merged["tabm_score"]
    return merged


def crossfit_meta_proba(
    feature_df: pd.DataFrame,
    y: np.ndarray,
    *,
    seed: int,
    folds: int,
) -> np.ndarray:
    x = feature_df.drop(columns=["customer_id"]).to_numpy(dtype=np.float32)
    oof = np.zeros(len(feature_df), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x, y):
        model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed)
        model.fit(x[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1].astype(np.float32)
    return oof


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    tabm_df = load_prediction_frame(args.tabm_val, args.targets).sort_values("customer_id").reset_index(drop=True)
    gandalf_df = load_prediction_frame(args.gandalf_val, args.targets).sort_values("customer_id").reset_index(drop=True)
    specialist_df = load_prediction_frame(args.specialist_val, args.targets).sort_values("customer_id").reset_index(drop=True)
    customer_ids = tabm_df["customer_id"].astype("int32")
    raw_df = load_raw_features(customer_ids, args.raw_top_k)

    labeled = load_labeled_frame()
    y_df = labeled[["customer_id"] + args.targets].copy()
    y_df = y_df[y_df["customer_id"].isin(customer_ids)].sort_values("customer_id").reset_index(drop=True)

    base_pred_df = tabm_df.copy()
    target_rows: list[dict[str, object]] = []
    corrected_prob_map: dict[str, np.ndarray] = {}

    for target_name in args.targets:
        feature_df = build_meta_features(tabm_df, gandalf_df, specialist_df, raw_df, target_name)
        feature_df = feature_df.sort_values("customer_id").reset_index(drop=True)
        y = y_df[target_name].to_numpy(dtype=np.int8)
        meta_proba = crossfit_meta_proba(feature_df, y, seed=args.seed, folds=args.folds)
        pred_col = target_name.replace("target_", "predict_")
        base_scores = tabm_df[pred_col].to_numpy(dtype=np.float32)
        corrected_prob_map[target_name] = meta_proba
        target_rows.append(
            {
                "target": target_name,
                "base_mean": float(base_scores.mean()),
                "meta_mean": float(meta_proba.mean()),
                "corr_mean_delta": float((meta_proba - base_scores).mean()),
            }
        )

    pd.DataFrame(target_rows).to_csv(args.artifact_dir / "per_target_features.csv", index=False)

    candidate_rows: list[dict[str, object]] = []
    best_alpha = None
    best_score = -np.inf
    best_pred_df = None
    for alpha in args.alphas:
        pred_df = base_pred_df.copy()
        for target_name in args.targets:
            pred_col = target_name.replace("target_", "predict_")
            base_scores = pred_df[pred_col].to_numpy(dtype=np.float32)
            meta_proba = corrected_prob_map[target_name]
            corrected = np.clip(base_scores + float(alpha) * (meta_proba - base_scores), 1e-6, 1 - 1e-6)
            pred_df[pred_col] = corrected.astype(np.float32)
        score = macro_auc(y_df, pred_df, args.targets)
        candidate_rows.append({"alpha": float(alpha), "macro_auc_subset": float(score)})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred_df = pred_df.copy()

    pd.DataFrame(candidate_rows).to_csv(args.artifact_dir / "candidate_subset_auc.csv", index=False)
    if best_pred_df is not None:
        best_pred_df.to_parquet(args.artifact_dir / "best_candidate_val_predictions.parquet", index=False)

    summary = {
        "targets": args.targets,
        "best_alpha": best_alpha,
        "best_subset_macro_auc": float(best_score),
        "base_subset_macro_auc": float(macro_auc(y_df, base_pred_df, args.targets)),
        "tabm_val": str(args.tabm_val),
        "gandalf_val": str(args.gandalf_val),
        "specialist_val": str(args.specialist_val),
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
