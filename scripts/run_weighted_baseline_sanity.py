from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.data_loading import load_labeled_frame
from lib.layout import project_root, resolve_data_dir
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DEFAULT_OUT_DIR = ROOT / "artifacts" / "weighted_baseline_sanity_v1"
TOP_EXTRA_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
WEIGHTS_FILE = ROOT / "artifacts" / "adversarial_validation_v1" / "weights_train.parquet"
GANDALF_VAL_FILE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
GANDALF_TARGET_SCORE_FILE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "target_scores.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sanity-check if adversarial weights help a robust local LGBM baseline")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--weights-file", type=Path, default=WEIGHTS_FILE)
    p.add_argument("--gandalf-val-file", type=Path, default=GANDALF_VAL_FILE)
    p.add_argument("--gandalf-target-score-file", type=Path, default=GANDALF_TARGET_SCORE_FILE)
    p.add_argument("--top-extra-file", type=Path, default=TOP_EXTRA_FILE)
    p.add_argument("--hardest-count", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--feature-fraction", type=float, default=0.7)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def frame_mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def prepare_frame(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].fillna(-1).astype("float32").astype("Int32")
    for col in feature_cols:
        if col not in cat_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    extra_features = json.loads(args.top_extra_file.read_text())
    labeled = load_labeled_frame(data_dir=args.data_dir, extra_feature_cols=extra_features)
    gandalf_val = normalize_prediction_columns(pd.read_parquet(args.gandalf_val_file)).sort_values("customer_id").reset_index(drop=True)
    gandalf_target_scores = pd.read_csv(args.gandalf_target_score_file).sort_values("oof_auc")
    weight_df = pd.read_parquet(args.weights_file)

    hardest_targets = gandalf_target_scores.head(args.hardest_count)["target"].tolist()
    pred_cols = [target.replace("target_", "predict_") for target in hardest_targets]
    val_ids = gandalf_val[["customer_id"]].copy()
    val_id_set = set(val_ids["customer_id"].tolist())

    train_df = labeled[~labeled["customer_id"].isin(val_id_set)].sort_values("customer_id").reset_index(drop=True)
    val_df = labeled[labeled["customer_id"].isin(val_id_set)].sort_values("customer_id").reset_index(drop=True)
    val_ref = gandalf_val[["customer_id"] + pred_cols].copy()

    train_df = train_df.merge(weight_df[["customer_id", "weight_norm"]], on="customer_id", how="left")
    if train_df["weight_norm"].isna().any():
        raise ValueError("Missing adversarial weights for some training rows")

    feature_cols = [c for c in labeled.columns if c not in ["customer_id", "weight_norm"] and not c.startswith("target_")]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

    train_X = prepare_frame(train_df[feature_cols], feature_cols, cat_cols)
    val_X = prepare_frame(val_df[feature_cols], feature_cols, cat_cols)
    print(
        {
            "stage": "data_ready",
            "train_rows": int(len(train_df)),
            "val_rows": int(len(val_df)),
            "feature_count": len(feature_cols),
            "hardest_count": len(hardest_targets),
            "train_x_mb": round(frame_mem_mb(train_X), 2),
            "val_x_mb": round(frame_mem_mb(val_X), 2),
        },
        flush=True,
    )

    weighted_pred = pd.DataFrame({"customer_id": val_df["customer_id"].astype("int32").values})
    unweighted_pred = pd.DataFrame({"customer_id": val_df["customer_id"].astype("int32").values})
    per_target_rows: list[dict[str, float | str | int]] = []

    base_params = {
        "objective": "binary",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "random_state": args.seed,
        "n_jobs": args.n_jobs,
        "verbose": -1,
    }

    for idx, target_name in enumerate(hardest_targets, start=1):
        pred_col = target_name.replace("target_", "predict_")
        y_train = train_df[target_name]
        y_val = val_df[target_name]
        print({"stage": "target_start", "target": target_name, "index": idx, "total": len(hardest_targets)}, flush=True)

        if y_train.nunique() < 2:
            const_value = float(y_train.mean())
            weighted_pred[pred_col] = const_value
            unweighted_pred[pred_col] = const_value
            per_target_rows.append(
                {
                    "target": target_name,
                    "gandalf_auc": float(roc_auc_score(y_val, val_ref[pred_col])) if y_val.nunique() > 1 else 0.5,
                    "unweighted_auc": 0.5,
                    "weighted_auc": 0.5,
                    "delta_weighted_vs_unweighted": 0.0,
                    "delta_weighted_vs_gandalf": 0.0,
                    "best_iteration_unweighted": 0,
                    "best_iteration_weighted": 0,
                }
            )
            continue

        model_unweighted = lgb.LGBMClassifier(**base_params)
        model_unweighted.fit(
            train_X,
            y_train,
            eval_set=[(val_X, y_val)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred_unweighted = model_unweighted.predict_proba(val_X)[:, 1].astype("float64")
        unweighted_pred[pred_col] = pred_unweighted

        model_weighted = lgb.LGBMClassifier(**base_params)
        model_weighted.fit(
            train_X,
            y_train,
            sample_weight=train_df["weight_norm"].to_numpy(dtype="float64"),
            eval_set=[(val_X, y_val)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(40, verbose=False)],
        )
        pred_weighted = model_weighted.predict_proba(val_X)[:, 1].astype("float64")
        weighted_pred[pred_col] = pred_weighted

        gandalf_auc = float(roc_auc_score(y_val, val_ref[pred_col])) if y_val.nunique() > 1 else 0.5
        unweighted_auc = float(roc_auc_score(y_val, pred_unweighted)) if y_val.nunique() > 1 else 0.5
        weighted_auc = float(roc_auc_score(y_val, pred_weighted)) if y_val.nunique() > 1 else 0.5
        per_target_rows.append(
            {
                "target": target_name,
                "gandalf_auc": gandalf_auc,
                "unweighted_auc": unweighted_auc,
                "weighted_auc": weighted_auc,
                "delta_weighted_vs_unweighted": weighted_auc - unweighted_auc,
                "delta_weighted_vs_gandalf": weighted_auc - gandalf_auc,
                "best_iteration_unweighted": int(getattr(model_unweighted, "best_iteration_", args.n_estimators) or args.n_estimators),
                "best_iteration_weighted": int(getattr(model_weighted, "best_iteration_", args.n_estimators) or args.n_estimators),
            }
        )
        print(
            {
                "stage": "target_done",
                "target": target_name,
                "gandalf_auc": gandalf_auc,
                "unweighted_auc": unweighted_auc,
                "weighted_auc": weighted_auc,
            },
            flush=True,
        )
        del model_unweighted, model_weighted
        gc.collect()

    y_val_all = val_df[hardest_targets].reset_index(drop=True)
    gandalf_macro = macro_auc(y_val_all, val_ref[["customer_id"] + pred_cols], hardest_targets)
    unweighted_macro = macro_auc(y_val_all, unweighted_pred, hardest_targets)
    weighted_macro = macro_auc(y_val_all, weighted_pred, hardest_targets)

    per_target_df = pd.DataFrame(per_target_rows).sort_values("weighted_auc").reset_index(drop=True)
    per_target_df.to_csv(args.out_dir / "target_scores.csv", index=False)
    unweighted_pred.to_parquet(args.out_dir / "validation_predictions_unweighted.parquet", index=False)
    weighted_pred.to_parquet(args.out_dir / "validation_predictions_weighted.parquet", index=False)

    metrics = {
        "targets": hardest_targets,
        "gandalf_macro_auc": gandalf_macro,
        "unweighted_macro_auc": unweighted_macro,
        "weighted_macro_auc": weighted_macro,
        "delta_weighted_vs_unweighted": weighted_macro - unweighted_macro,
        "delta_weighted_vs_gandalf": weighted_macro - gandalf_macro,
        "weight_mean": float(train_df["weight_norm"].mean()),
        "weight_std": float(train_df["weight_norm"].std()),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
