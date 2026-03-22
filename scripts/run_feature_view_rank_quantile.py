from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lib.data_loading import load_labeled_frame
from lib.meta_views import ROOT, load_prediction_frame, load_top_extra_features, target_to_pred
from lib.metrics import macro_auc


DEFAULT_VAL_IDS = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/feature_view_rank_quantile_v1"
DEFAULT_TARGETS = [
    "target_9_6",
    "target_9_3",
    "target_3_1",
    "target_2_4",
    "target_6_1",
    "target_6_2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test rank/quantile feature view on weak targets.")
    parser.add_argument("--val-predictions", type=Path, default=DEFAULT_VAL_IDS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--top-k-features", type=int, default=24)
    parser.add_argument("--c-value", type=float, default=0.2)
    return parser.parse_args()


def empirical_percentile(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_train = np.sort(train_values.astype(np.float64))
    n = max(len(sorted_train), 1)
    return np.searchsorted(sorted_train, values.astype(np.float64), side="right") / float(n)


def build_transformed_views(train_df: pd.DataFrame, val_df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train_df.index)
    val_out = pd.DataFrame(index=val_df.index)
    train_array_parts: list[np.ndarray] = []
    val_array_parts: list[np.ndarray] = []

    for col in feature_cols:
        train_col = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        val_col = pd.to_numeric(val_df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        q1 = float(np.quantile(train_col, 0.25))
        q3 = float(np.quantile(train_col, 0.75))
        iqr = max(q3 - q1, 1e-6)
        median = float(np.median(train_col))
        train_scaled = np.tanh((train_col - median) / (4.0 * iqr))
        val_scaled = np.tanh((val_col - median) / (4.0 * iqr))
        train_rank = empirical_percentile(train_col, train_col).astype(np.float32)
        val_rank = empirical_percentile(train_col, val_col).astype(np.float32)
        train_bin = np.floor(train_rank * 10.0).astype(np.float32) / 10.0
        val_bin = np.floor(val_rank * 10.0).astype(np.float32) / 10.0
        train_out[f"{col}__scaled"] = train_scaled
        train_out[f"{col}__rank"] = train_rank
        train_out[f"{col}__qbin"] = train_bin
        val_out[f"{col}__scaled"] = val_scaled
        val_out[f"{col}__rank"] = val_rank
        val_out[f"{col}__qbin"] = val_bin
        train_array_parts.append(train_scaled[:, None])
        train_array_parts.append(train_rank[:, None])
        val_array_parts.append(val_scaled[:, None])
        val_array_parts.append(val_rank[:, None])

    train_stack = np.hstack(train_array_parts) if train_array_parts else np.zeros((len(train_df), 1), dtype=np.float32)
    val_stack = np.hstack(val_array_parts) if val_array_parts else np.zeros((len(val_df), 1), dtype=np.float32)
    for prefix, arr_train, arr_val in [
        ("row_mean", train_stack.mean(axis=1), val_stack.mean(axis=1)),
        ("row_std", train_stack.std(axis=1), val_stack.std(axis=1)),
        ("row_min", train_stack.min(axis=1), val_stack.min(axis=1)),
        ("row_max", train_stack.max(axis=1), val_stack.max(axis=1)),
    ]:
        train_out[prefix] = arr_train.astype(np.float32)
        val_out[prefix] = arr_val.astype(np.float32)
    train_out["row_high_share"] = (train_stack > 0.8).mean(axis=1).astype(np.float32)
    val_out["row_high_share"] = (val_stack > 0.8).mean(axis=1).astype(np.float32)
    return train_out, val_out


def fit_predict(train_x: pd.DataFrame, train_y: np.ndarray, val_x: pd.DataFrame, *, c_value: float) -> np.ndarray:
    model = LogisticRegression(
        C=c_value,
        max_iter=500,
        solver="liblinear",
        class_weight="balanced",
        random_state=42,
    )
    model.fit(train_x.to_numpy(dtype=np.float32), train_y)
    return model.predict_proba(val_x.to_numpy(dtype=np.float32))[:, 1].astype(np.float32)


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    val_ids = load_prediction_frame(args.val_predictions, args.targets)[["customer_id"]]
    top_features = load_top_extra_features(args.top_k_features)
    labeled = load_labeled_frame(extra_feature_cols=top_features)
    feature_cols = [c for c in top_features if c in labeled.columns]
    keep_cols = ["customer_id"] + feature_cols + args.targets
    labeled = labeled[keep_cols].copy()

    val_df = labeled.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    train_df = labeled[~labeled["customer_id"].isin(set(val_ids["customer_id"]))].sort_values("customer_id").reset_index(drop=True)

    raw_train = train_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    raw_val = val_df[feature_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).astype("float32")
    transform_train, transform_val = build_transformed_views(train_df, val_df, feature_cols)
    combo_train = pd.concat([raw_train.reset_index(drop=True), transform_train.reset_index(drop=True)], axis=1)
    combo_val = pd.concat([raw_val.reset_index(drop=True), transform_val.reset_index(drop=True)], axis=1)

    target_rows: list[dict[str, object]] = []
    candidate_frames = {
        "raw_only": pd.DataFrame({"customer_id": val_df["customer_id"].values}),
        "rank_quantile_only": pd.DataFrame({"customer_id": val_df["customer_id"].values}),
        "combined": pd.DataFrame({"customer_id": val_df["customer_id"].values}),
    }

    for target_name in args.targets:
        train_y = train_df[target_name].to_numpy(dtype=np.int8)
        for view_name, train_x, val_x in [
            ("raw_only", raw_train, raw_val),
            ("rank_quantile_only", transform_train, transform_val),
            ("combined", combo_train, combo_val),
        ]:
            pred = fit_predict(train_x, train_y, val_x, c_value=args.c_value)
            candidate_frames[view_name][target_to_pred(target_name)] = pred
            target_rows.append(
                {
                    "target": target_name,
                    "view": view_name,
                    "val_auc": float(
                        macro_auc(
                            val_df[["customer_id", target_name]],
                            pd.DataFrame({"customer_id": val_df["customer_id"].values, target_to_pred(target_name): pred}),
                            [target_name],
                        )
                    ),
                }
            )

    for view_name, pred_df in candidate_frames.items():
        pred_df.to_parquet(args.artifact_dir / f"{view_name}_validation_predictions.parquet", index=False)

    target_score_df = pd.DataFrame(target_rows)
    target_score_df.to_csv(args.artifact_dir / "per_target_view_auc.csv", index=False)

    summary_rows = []
    y_holdout = val_df[["customer_id"] + args.targets].copy()
    for view_name, pred_df in candidate_frames.items():
        summary_rows.append(
            {
                "view": view_name,
                "macro_auc_subset": float(macro_auc(y_holdout, pred_df, args.targets)),
                "targets": len(args.targets),
                "top_k_features": int(args.top_k_features),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("macro_auc_subset", ascending=False).reset_index(drop=True)
    summary.to_csv(args.artifact_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
