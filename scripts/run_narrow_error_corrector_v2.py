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
from lib.meta_views import safe_logit
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
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/narrow_error_corrector_v2"
DEFAULT_TARGETS = [
    "target_9_3",
    "target_3_1",
    "target_2_4",
    "target_10_1",
    "target_9_7",
    "target_2_6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run narrow error-corrector ablations on the best shortlist.")
    parser.add_argument("--tabm-val", type=Path, default=DEFAULT_TABM_VAL)
    parser.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    parser.add_argument("--specialist-val", type=Path, default=DEFAULT_SPECIALIST_VAL)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--raw-top-k", type=int, default=12)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.03, 0.05, 0.07, 0.10, 0.15, 0.20])
    return parser.parse_args()


def load_prediction_frame(path: Path) -> pd.DataFrame:
    return normalize_prediction_columns(pd.read_parquet(path)).sort_values("customer_id").reset_index(drop=True)


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


def add_rank_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    out = raw_df[["customer_id"]].copy()
    for col in raw_df.columns:
        if col == "customer_id":
            continue
        values = raw_df[col].to_numpy(dtype=np.float32)
        order = np.argsort(values, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(values), dtype=np.float32)
        out[f"{col}__rank"] = (ranks / max(len(values), 1)).astype(np.float32)
    return out


def build_meta_features(
    base_df: pd.DataFrame,
    gandalf_df: pd.DataFrame,
    specialist_df: pd.DataFrame,
    raw_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    target_name: str,
    *,
    use_rank_features: bool,
) -> pd.DataFrame:
    pred_col = target_name.replace("target_", "predict_")
    merged = (
        base_df[["customer_id", pred_col]]
        .rename(columns={pred_col: "tabm_score"})
        .merge(gandalf_df[["customer_id", pred_col]].rename(columns={pred_col: "gandalf_score"}), on="customer_id", how="inner")
        .merge(specialist_df[["customer_id", pred_col]].rename(columns={pred_col: "specialist_score"}), on="customer_id", how="inner")
        .merge(raw_df, on="customer_id", how="left")
    )
    if use_rank_features:
        merged = merged.merge(rank_df, on="customer_id", how="left")
    merged["tabm_logit"] = safe_logit(merged["tabm_score"]).astype("float32")
    merged["gandalf_logit"] = safe_logit(merged["gandalf_score"]).astype("float32")
    merged["specialist_logit"] = safe_logit(merged["specialist_score"]).astype("float32")
    merged["specialist_minus_tabm"] = (merged["specialist_score"] - merged["tabm_score"]).astype("float32")
    merged["gandalf_minus_tabm"] = (merged["gandalf_score"] - merged["tabm_score"]).astype("float32")
    return merged.sort_values("customer_id").reset_index(drop=True)


def crossfit_meta_proba(feature_df: pd.DataFrame, y: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    x = feature_df.drop(columns=["customer_id"]).to_numpy(dtype=np.float32)
    oof = np.zeros(len(feature_df), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    for train_idx, val_idx in splitter.split(x, y):
        model = LogisticRegression(max_iter=500, class_weight="balanced", solver="liblinear", random_state=seed)
        model.fit(x[train_idx], y[train_idx])
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1].astype(np.float32)
    return oof


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    tabm_df = load_prediction_frame(args.tabm_val)
    gandalf_df = load_prediction_frame(args.gandalf_val)
    specialist_df = load_prediction_frame(args.specialist_val)
    customer_ids = tabm_df["customer_id"].astype("int32")

    raw_df = load_raw_features(customer_ids, args.raw_top_k)
    rank_df = add_rank_features(raw_df)

    labeled = load_labeled_frame()
    all_targets = [c for c in labeled.columns if c.startswith("target_")]
    y_df = labeled[["customer_id"] + all_targets].copy()
    y_df = y_df[y_df["customer_id"].isin(customer_ids)].sort_values("customer_id").reset_index(drop=True)

    full_base_pred_df = tabm_df[["customer_id"] + [c for c in tabm_df.columns if c.startswith("predict_")]].copy()
    full_base_macro = float(macro_auc(y_df, full_base_pred_df, all_targets))
    base_subset_macro = float(macro_auc(y_df[["customer_id"] + args.targets], full_base_pred_df[["customer_id"] + [t.replace("target_", "predict_") for t in args.targets]], args.targets))

    meta_maps: dict[str, dict[str, np.ndarray]] = {"raw": {}, "rank": {}}
    scan_rows: list[dict[str, object]] = []

    for target_name in args.targets:
        y = y_df[target_name].to_numpy(dtype=np.int8)
        pred_col = target_name.replace("target_", "predict_")
        base_target_score = float(
            macro_auc(
                y_df[["customer_id", target_name]],
                full_base_pred_df[["customer_id", pred_col]],
                [target_name],
            )
        )
        for feature_mode, use_rank in [("raw", False), ("rank", True)]:
            feature_df = build_meta_features(
                tabm_df,
                gandalf_df,
                specialist_df,
                raw_df,
                rank_df,
                target_name,
                use_rank_features=use_rank,
            )
            meta_proba = crossfit_meta_proba(feature_df, y, seed=args.seed, folds=args.folds)
            meta_maps[feature_mode][target_name] = meta_proba
            best_auc = -np.inf
            best_alpha = None
            for alpha in args.alphas:
                corrected = np.clip(
                    full_base_pred_df[pred_col].to_numpy(dtype=np.float32) + float(alpha) * (meta_proba - full_base_pred_df[pred_col].to_numpy(dtype=np.float32)),
                    1e-6,
                    1 - 1e-6,
                )
                auc = float(
                    macro_auc(
                        y_df[["customer_id", target_name]],
                        pd.DataFrame({"customer_id": y_df["customer_id"].values, pred_col: corrected}),
                        [target_name],
                    )
                )
                if auc > best_auc:
                    best_auc = auc
                    best_alpha = float(alpha)
            scan_rows.append(
                {
                    "target": target_name,
                    "feature_mode": feature_mode,
                    "base_auc": base_target_score,
                    "best_auc": best_auc,
                    "delta": best_auc - base_target_score,
                    "best_alpha": best_alpha,
                }
            )

    scan_df = pd.DataFrame(scan_rows).sort_values(["delta", "target"], ascending=[False, True]).reset_index(drop=True)
    scan_df.to_csv(args.artifact_dir / "per_target_scan.csv", index=False)

    candidate_rows: list[dict[str, object]] = []
    best_full_macro = -np.inf
    best_name = None
    best_pred_df = None

    for feature_mode in ["raw", "rank"]:
        for alpha in args.alphas:
            pred_df = full_base_pred_df.copy()
            for target_name in args.targets:
                pred_col = target_name.replace("target_", "predict_")
                base_scores = pred_df[pred_col].to_numpy(dtype=np.float32)
                meta_proba = meta_maps[feature_mode][target_name]
                corrected = np.clip(base_scores + float(alpha) * (meta_proba - base_scores), 1e-6, 1 - 1e-6)
                pred_df[pred_col] = corrected.astype(np.float32)
            full_macro = float(macro_auc(y_df, pred_df, all_targets))
            subset_macro = float(
                macro_auc(
                    y_df[["customer_id"] + args.targets],
                    pred_df[["customer_id"] + [t.replace("target_", "predict_") for t in args.targets]],
                    args.targets,
                )
            )
            row = {
                "candidate": f"{feature_mode}_meta_alpha_{str(alpha).replace('.', 'p')}",
                "feature_mode": feature_mode,
                "alpha": float(alpha),
                "full_macro_auc": full_macro,
                "subset_macro_auc": subset_macro,
                "delta_vs_base_full": full_macro - full_base_macro,
                "delta_vs_base_subset": subset_macro - base_subset_macro,
            }
            candidate_rows.append(row)
            if full_macro > best_full_macro:
                best_full_macro = full_macro
                best_name = row["candidate"]
                best_pred_df = pred_df.copy()

    candidate_df = pd.DataFrame(candidate_rows).sort_values("full_macro_auc", ascending=False).reset_index(drop=True)
    candidate_df.to_csv(args.artifact_dir / "candidate_scores.csv", index=False)
    if best_pred_df is not None:
        best_pred_df.to_parquet(args.artifact_dir / "best_candidate_val_predictions.parquet", index=False)

    summary = {
        "targets": args.targets,
        "base_full_macro_auc": full_base_macro,
        "base_subset_macro_auc": base_subset_macro,
        "best_candidate": best_name,
        "best_full_macro_auc": best_full_macro,
        "best_delta_vs_base_full": best_full_macro - full_base_macro,
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
