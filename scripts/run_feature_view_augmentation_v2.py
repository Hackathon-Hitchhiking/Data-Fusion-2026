from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import QuantileTransformer

from lib.data_loading import load_labeled_frame
from lib.meta_views import ROOT, load_prediction_frame, load_top_extra_features, target_to_pred
from lib.metrics import macro_auc


DEFAULT_VAL_IDS = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/feature_view_augmentation_v2"
DEFAULT_TARGETS = [
    "target_9_3",
    "target_9_6",
    "target_3_1",
    "target_6_1",
    "target_6_2",
    "target_2_4",
    "target_10_1",
    "target_9_7",
    "target_2_6",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ablate transformed feature views on the weak block.")
    parser.add_argument("--val-predictions", type=Path, default=DEFAULT_VAL_IDS)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--top-k-features", type=int, default=24)
    parser.add_argument("--c-value", type=float, default=0.2)
    parser.add_argument("--quantiles", type=int, default=256)
    return parser.parse_args()


def empirical_percentile(train_values: np.ndarray, values: np.ndarray) -> np.ndarray:
    sorted_train = np.sort(train_values.astype(np.float64))
    n = max(len(sorted_train), 1)
    return np.searchsorted(sorted_train, values.astype(np.float64), side="right") / float(n)


def _numeric_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = float(out[col].median()) if out[col].notna().any() else 0.0
        out[col] = out[col].fillna(median).astype("float32")
    return out


def build_rank_view(train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train_raw.index)
    val_out = pd.DataFrame(index=val_raw.index)
    for col in train_raw.columns:
        train_col = train_raw[col].to_numpy(dtype=np.float32)
        val_col = val_raw[col].to_numpy(dtype=np.float32)
        train_rank = empirical_percentile(train_col, train_col).astype(np.float32)
        val_rank = empirical_percentile(train_col, val_col).astype(np.float32)
        train_out[f"{col}__rank"] = train_rank
        val_out[f"{col}__rank"] = val_rank
    train_out["rank_row_mean"] = train_out.mean(axis=1).astype(np.float32)
    val_out["rank_row_mean"] = val_out.mean(axis=1).astype(np.float32)
    train_out["rank_row_std"] = train_out.std(axis=1).astype(np.float32)
    val_out["rank_row_std"] = val_out.std(axis=1).astype(np.float32)
    return train_out, val_out


def build_quantile_view(
    train_raw: pd.DataFrame,
    val_raw: pd.DataFrame,
    *,
    n_quantiles: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, max(10, len(train_raw))),
        output_distribution="normal",
        random_state=42,
        subsample=int(1e9),
    )
    train_np = qt.fit_transform(train_raw.to_numpy(dtype=np.float32)).astype(np.float32)
    val_np = qt.transform(val_raw.to_numpy(dtype=np.float32)).astype(np.float32)
    train_out = pd.DataFrame(train_np, columns=[f"{c}__quant" for c in train_raw.columns], index=train_raw.index)
    val_out = pd.DataFrame(val_np, columns=[f"{c}__quant" for c in val_raw.columns], index=val_raw.index)
    train_out["quant_row_mean"] = train_out.mean(axis=1).astype(np.float32)
    val_out["quant_row_mean"] = val_out.mean(axis=1).astype(np.float32)
    train_out["quant_row_std"] = train_out.std(axis=1).astype(np.float32)
    val_out["quant_row_std"] = val_out.std(axis=1).astype(np.float32)
    return train_out, val_out


def build_clipped_view(train_raw: pd.DataFrame, val_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_out = pd.DataFrame(index=train_raw.index)
    val_out = pd.DataFrame(index=val_raw.index)
    for col in train_raw.columns:
        train_col = train_raw[col].to_numpy(dtype=np.float32)
        val_col = val_raw[col].to_numpy(dtype=np.float32)
        median = float(np.median(train_col))
        q1 = float(np.quantile(train_col, 0.25))
        q3 = float(np.quantile(train_col, 0.75))
        iqr = max(q3 - q1, 1e-6)
        train_scaled = np.tanh((train_col - median) / (4.0 * iqr)).astype(np.float32)
        val_scaled = np.tanh((val_col - median) / (4.0 * iqr)).astype(np.float32)
        train_out[f"{col}__clipped"] = train_scaled
        val_out[f"{col}__clipped"] = val_scaled
    train_out["clipped_row_mean"] = train_out.mean(axis=1).astype(np.float32)
    val_out["clipped_row_mean"] = val_out.mean(axis=1).astype(np.float32)
    train_out["clipped_row_std"] = train_out.std(axis=1).astype(np.float32)
    val_out["clipped_row_std"] = val_out.std(axis=1).astype(np.float32)
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

    raw_train = _numeric_frame(train_df, feature_cols)
    raw_val = _numeric_frame(val_df, feature_cols)
    rank_train, rank_val = build_rank_view(raw_train, raw_val)
    quant_train, quant_val = build_quantile_view(raw_train, raw_val, n_quantiles=args.quantiles)
    clipped_train, clipped_val = build_clipped_view(raw_train, raw_val)

    view_blocks = {
        "raw_only": (raw_train, raw_val),
        "rank_only": (rank_train, rank_val),
        "quantile_only": (quant_train, quant_val),
        "clipped_only": (clipped_train, clipped_val),
        "raw_plus_rank": (
            pd.concat([raw_train.reset_index(drop=True), rank_train.reset_index(drop=True)], axis=1),
            pd.concat([raw_val.reset_index(drop=True), rank_val.reset_index(drop=True)], axis=1),
        ),
        "raw_plus_quantile": (
            pd.concat([raw_train.reset_index(drop=True), quant_train.reset_index(drop=True)], axis=1),
            pd.concat([raw_val.reset_index(drop=True), quant_val.reset_index(drop=True)], axis=1),
        ),
        "raw_plus_clipped": (
            pd.concat([raw_train.reset_index(drop=True), clipped_train.reset_index(drop=True)], axis=1),
            pd.concat([raw_val.reset_index(drop=True), clipped_val.reset_index(drop=True)], axis=1),
        ),
        "raw_plus_rank_plus_quantile": (
            pd.concat([raw_train.reset_index(drop=True), rank_train.reset_index(drop=True), quant_train.reset_index(drop=True)], axis=1),
            pd.concat([raw_val.reset_index(drop=True), rank_val.reset_index(drop=True), quant_val.reset_index(drop=True)], axis=1),
        ),
    }

    target_rows: list[dict[str, object]] = []
    candidate_frames = {name: pd.DataFrame({"customer_id": val_df["customer_id"].values}) for name in view_blocks}

    for target_name in args.targets:
        train_y = train_df[target_name].to_numpy(dtype=np.int8)
        for view_name, (train_x, val_x) in view_blocks.items():
            pred = fit_predict(train_x, train_y, val_x, c_value=args.c_value)
            pred_col = target_to_pred(target_name)
            candidate_frames[view_name][pred_col] = pred
            auc = float(
                macro_auc(
                    val_df[["customer_id", target_name]],
                    pd.DataFrame({"customer_id": val_df["customer_id"].values, pred_col: pred}),
                    [target_name],
                )
            )
            target_rows.append(
                {
                    "target": target_name,
                    "view_type": view_name,
                    "model_type": "logreg",
                    "auc": auc,
                }
            )

    target_score_df = pd.DataFrame(target_rows)
    raw_map = (
        target_score_df[target_score_df["view_type"] == "raw_only"][["target", "auc"]]
        .rename(columns={"auc": "raw_auc"})
        .reset_index(drop=True)
    )
    target_score_df = target_score_df.merge(raw_map, on="target", how="left")
    target_score_df["delta_vs_raw"] = target_score_df["auc"] - target_score_df["raw_auc"]
    target_score_df.to_csv(args.artifact_dir / "per_target_view_scores.csv", index=False)

    summary_rows = []
    y_holdout = val_df[["customer_id"] + args.targets].copy()
    for view_name, pred_df in candidate_frames.items():
        subset_macro = float(macro_auc(y_holdout, pred_df, args.targets))
        summary_rows.append(
            {
                "view_type": view_name,
                "macro_auc_subset": subset_macro,
                "targets": len(args.targets),
                "top_k_features": int(args.top_k_features),
                "positive_target_deltas": int(
                    (
                        target_score_df[(target_score_df["view_type"] == view_name)]["delta_vs_raw"] > 0
                    ).sum()
                ),
            }
        )
        pred_df.to_parquet(args.artifact_dir / f"{view_name}_validation_predictions.parquet", index=False)

    summary = pd.DataFrame(summary_rows).sort_values("macro_auc_subset", ascending=False).reset_index(drop=True)
    summary.to_csv(args.artifact_dir / "feature_view_ablation.csv", index=False)
    payload = {
        "targets": args.targets,
        "top_k_features": int(args.top_k_features),
        "best_view": str(summary.iloc[0]["view_type"]),
        "best_macro_auc_subset": float(summary.iloc[0]["macro_auc_subset"]),
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
