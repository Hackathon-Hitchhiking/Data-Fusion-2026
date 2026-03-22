from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lib.data_loading import read_competition_parquet
from lib.layout import project_root, resolve_data_dir


ROOT = project_root()
TOP_EXTRA_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "adversarial_validation_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Measure train/test covariate shift and derive importance weights")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--top-extra-file", type=Path, default=TOP_EXTRA_FILE)
    p.add_argument("--extra-top-k", type=int, default=100)
    p.add_argument("--folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--clip-min", type=float, default=0.25)
    p.add_argument("--clip-max", type=float, default=4.0)
    p.add_argument("--n-jobs", type=int, default=4)
    return p.parse_args()


def frame_mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def prepare_frame(df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cat_cols:
        out[col] = out[col].fillna(-1).astype("int32")
    for col in feature_cols:
        if col not in cat_cols:
            out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    extra_features = json.loads(args.top_extra_file.read_text())[: args.extra_top_k]
    extra_cols = ["customer_id"] + extra_features

    print({"stage": "load_data_start", "data_dir": str(args.data_dir), "extra_top_k": len(extra_features)}, flush=True)
    train_main = read_competition_parquet("train_main_features.parquet", data_dir=args.data_dir)
    test_main = read_competition_parquet("test_main_features.parquet", data_dir=args.data_dir)
    train_extra = read_competition_parquet("train_extra_features.parquet", data_dir=args.data_dir, columns=extra_cols)
    test_extra = read_competition_parquet("test_extra_features.parquet", data_dir=args.data_dir, columns=extra_cols)

    train_df = train_main.merge(train_extra, on="customer_id", how="left")
    test_df = test_main.merge(test_extra, on="customer_id", how="left")
    feature_cols = [c for c in train_df.columns if c != "customer_id"]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

    train_df = prepare_frame(train_df, feature_cols, cat_cols)
    test_df = prepare_frame(test_df, feature_cols, cat_cols)

    full_df = pd.concat(
        [
            train_df.assign(is_test=np.int8(0)),
            test_df.assign(is_test=np.int8(1)),
        ],
        axis=0,
        ignore_index=True,
    )
    print(
        {
            "stage": "data_ready",
            "train_rows": int(len(train_df)),
            "test_rows": int(len(test_df)),
            "feature_count": len(feature_cols),
            "cat_count": len(cat_cols),
            "full_mem_mb": round(frame_mem_mb(full_df[["customer_id"] + feature_cols]), 2),
        },
        flush=True,
    )

    X = full_df[feature_cols]
    y = full_df["is_test"].astype("int8")

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(full_df), dtype=np.float64)
    feature_gain = np.zeros(len(feature_cols), dtype=np.float64)
    feature_split = np.zeros(len(feature_cols), dtype=np.float64)
    fold_rows: list[dict[str, float | int]] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        xtr = X.iloc[train_idx]
        xva = X.iloc[val_idx]
        ytr = y.iloc[train_idx]
        yva = y.iloc[val_idx]

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            random_state=args.seed + fold,
            n_jobs=args.n_jobs,
            verbose=-1,
        )
        print(
            {"stage": "fold_start", "fold": fold, "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))},
            flush=True,
        )
        model.fit(
            xtr,
            ytr,
            eval_set=[(xva, yva)],
            eval_metric="auc",
            categorical_feature=cat_cols,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        pred = model.predict_proba(xva)[:, 1].astype(np.float64)
        oof[val_idx] = pred
        fold_auc = float(roc_auc_score(yva, pred))
        fold_rows.append(
            {
                "fold": fold,
                "auc": fold_auc,
                "best_iteration": int(getattr(model, "best_iteration_", args.n_estimators) or args.n_estimators),
            }
        )
        booster = model.booster_
        feature_gain += booster.feature_importance(importance_type="gain")
        feature_split += booster.feature_importance(importance_type="split")
        print({"stage": "fold_done", "fold": fold, "auc": fold_auc}, flush=True)
        del xtr, xva, ytr, yva, model, booster
        gc.collect()

    overall_auc = float(roc_auc_score(y, oof))
    train_prior = float((y == 0).mean())
    test_prior = float((y == 1).mean())
    prior_ratio = train_prior / max(test_prior, 1e-12)

    density_ratio_raw = (oof / np.clip(1.0 - oof, 1e-6, None)) * prior_ratio
    density_ratio_clipped = np.clip(density_ratio_raw, args.clip_min, args.clip_max)
    train_mask = full_df["is_test"].to_numpy() == 0
    train_norm_factor = float(np.mean(density_ratio_clipped[train_mask]))
    density_ratio_norm = density_ratio_clipped / max(train_norm_factor, 1e-12)

    full_pred = pd.DataFrame(
        {
            "customer_id": full_df["customer_id"].astype("int32").values,
            "is_test": full_df["is_test"].astype("int8").values,
            "p_test": oof.astype("float64"),
            "density_ratio_raw": density_ratio_raw.astype("float64"),
            "weight_clipped": density_ratio_clipped.astype("float64"),
            "weight_norm": density_ratio_norm.astype("float64"),
        }
    )
    train_weights = full_pred[full_pred["is_test"] == 0].drop(columns="is_test").reset_index(drop=True)
    test_weights = full_pred[full_pred["is_test"] == 1].drop(columns="is_test").reset_index(drop=True)

    train_weights.to_parquet(args.out_dir / "weights_train.parquet", index=False)
    test_weights.to_parquet(args.out_dir / "weights_test.parquet", index=False)

    importance_df = pd.DataFrame(
        {
            "feature": feature_cols,
            "gain": feature_gain / args.folds,
            "split": feature_split / args.folds,
        }
    ).sort_values("gain", ascending=False)
    importance_df.to_csv(args.out_dir / "feature_importance.csv", index=False)

    metrics = {
        "overall_auc": overall_auc,
        "folds": fold_rows,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_count": len(feature_cols),
        "cat_count": len(cat_cols),
        "extra_top_k": len(extra_features),
        "train_prior": train_prior,
        "test_prior": test_prior,
        "prior_ratio": prior_ratio,
        "clip_min": args.clip_min,
        "clip_max": args.clip_max,
        "train_weight_clipped_mean": float(train_weights["weight_clipped"].mean()),
        "train_weight_clipped_std": float(train_weights["weight_clipped"].std()),
        "train_weight_norm_mean": float(train_weights["weight_norm"].mean()),
        "train_weight_norm_std": float(train_weights["weight_norm"].std()),
        "test_weight_clipped_mean": float(test_weights["weight_clipped"].mean()),
        "test_weight_clipped_std": float(test_weights["weight_clipped"].std()),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
