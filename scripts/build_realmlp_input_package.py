from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

from lib.data_loading import load_labeled_frame, load_test_frame
from lib.layout import project_root


ROOT = project_root()
DEFAULT_VAL_PRED = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
TOP_FEATURES_FILE = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
DEFAULT_AUX_SPEC = ROOT / "artifacts/feature_view_backbone_prep_v1/compact_auxiliary_feature_block.json"
DEFAULT_TARGET_SUMMARY = ROOT / "artifacts/dataset_feature_diagnostics_v1/target_summary.csv"
DEFAULT_OUT = ROOT / "artifacts/realmlp_input_package_v2"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a reusable RealMLP input package with shared split and aux-view block.")
    p.add_argument("--val-pred", type=Path, default=DEFAULT_VAL_PRED)
    p.add_argument("--aux-spec", type=Path, default=DEFAULT_AUX_SPEC)
    p.add_argument("--target-summary", type=Path, default=DEFAULT_TARGET_SUMMARY)
    p.add_argument("--artifact-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--quantiles", type=int, default=256)
    return p.parse_args()


def smooth_clip(values: np.ndarray, clip_value: float = 8.0) -> np.ndarray:
    return (clip_value * np.tanh(values / clip_value)).astype(np.float32)


def prepare_main_blocks(
    train_df: pd.DataFrame,
    apply_frames: dict[str, pd.DataFrame],
    num_cols: list[str],
    cat_cols: list[str],
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    train_num = train_df[num_cols].copy()
    apply_num = {name: frame[num_cols].copy() for name, frame in apply_frames.items()}
    train_cat = train_df[cat_cols].copy()
    apply_cat = {name: frame[cat_cols].copy() for name, frame in apply_frames.items()}

    num_stats_rows = []
    for col in num_cols:
        train_num[col] = pd.to_numeric(train_num[col], errors="coerce")
        median = float(train_num[col].median()) if train_num[col].notna().any() else 0.0
        q1 = float(train_num[col].quantile(0.25)) if train_num[col].notna().any() else 0.0
        q3 = float(train_num[col].quantile(0.75)) if train_num[col].notna().any() else 0.0
        iqr = q3 - q1
        scale = iqr if iqr > 1e-6 else 1.0
        std = float(train_num[col].std()) if train_num[col].notna().any() else 0.0

        train_num[col] = train_num[col].fillna(median).astype(np.float32)
        robust_train = ((train_num[col].to_numpy(dtype=np.float32) - median) / scale).astype(np.float32)
        train_num[col] = smooth_clip(robust_train)

        for name, frame in apply_num.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median).astype(np.float32)
            robust_apply = ((frame[col].to_numpy(dtype=np.float32) - median) / scale).astype(np.float32)
            frame[col] = smooth_clip(robust_apply)

        num_stats_rows.append(
            {
                "feature": col,
                "median": median,
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "scale_used": scale,
                "std": std,
            }
        )

    cat_stats_rows = []
    for col in cat_cols:
        train_cat[col] = pd.to_numeric(train_cat[col], errors="coerce").fillna(-1).astype("int32")
        for name, frame in apply_cat.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(-1).astype("int32")
        cat_stats_rows.append(
            {
                "feature": col,
                "min_value": int(train_cat[col].min()),
                "max_value": int(train_cat[col].max()),
                "n_unique": int(train_cat[col].nunique()),
                "already_integer_encoded": True,
            }
        )

    train_baseline = pd.concat(
        [train_df[["customer_id"]].reset_index(drop=True), train_num.reset_index(drop=True), train_cat.reset_index(drop=True)],
        axis=1,
    )
    apply_baseline = {
        name: pd.concat(
            [apply_frames[name][["customer_id"]].reset_index(drop=True), apply_num[name].reset_index(drop=True), apply_cat[name].reset_index(drop=True)],
            axis=1,
        )
        for name in apply_frames
    }
    stats_df = pd.concat(
        [
            pd.DataFrame(num_stats_rows).assign(block="robust_scaled_numeric"),
            pd.DataFrame(cat_stats_rows).assign(block="encoded_categorical"),
        ],
        axis=0,
        ignore_index=True,
    )
    return train_baseline, apply_baseline, stats_df


def build_aux_views(
    train_extra: pd.DataFrame,
    apply_frames: dict[str, pd.DataFrame],
    feature_cols: list[str],
    *,
    n_quantiles: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    train_raw = train_extra[["customer_id"] + feature_cols].copy()
    apply_raw = {name: frame[["customer_id"] + feature_cols].copy() for name, frame in apply_frames.items()}

    aux_stats_rows = []
    for col in feature_cols:
        train_raw[col] = pd.to_numeric(train_raw[col], errors="coerce")
        missing_ratio_train = float(train_raw[col].isna().mean())
        median = float(train_raw[col].median()) if train_raw[col].notna().any() else 0.0
        train_raw[col] = train_raw[col].fillna(median).astype(np.float32)
        for name, frame in apply_raw.items():
            frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(median).astype(np.float32)
        aux_stats_rows.append(
            {
                "feature": col,
                "median_fill": median,
                "missing_ratio_train": missing_ratio_train,
            }
        )

    qt = QuantileTransformer(
        n_quantiles=min(n_quantiles, max(16, len(train_raw))),
        output_distribution="normal",
        random_state=42,
        subsample=int(1e9),
    )
    train_np = qt.fit_transform(train_raw[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    apply_np = {
        name: qt.transform(frame[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
        for name, frame in apply_raw.items()
    }

    train_quant = pd.DataFrame(train_np, columns=[f"{c}__quant" for c in feature_cols], index=train_raw.index)
    apply_quant = {
        name: pd.DataFrame(block, columns=[f"{c}__quant" for c in feature_cols], index=frame.index)
        for name, (frame, block) in zip(apply_raw.keys(), zip(apply_raw.values(), apply_np.values()))
    }

    def clipped_from_train_stats(train_frame: pd.DataFrame, apply_frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=apply_frame.index)
        for col in feature_cols:
            train_col = train_frame[col].to_numpy(dtype=np.float32)
            apply_col = apply_frame[col].to_numpy(dtype=np.float32)
            median = float(np.median(train_col))
            q1 = float(np.quantile(train_col, 0.25))
            q3 = float(np.quantile(train_col, 0.75))
            iqr = max(q3 - q1, 1e-6)
            out[f"{col}__clipped"] = np.tanh((apply_col - median) / (4.0 * iqr)).astype(np.float32)
        return out

    train_clip = clipped_from_train_stats(train_raw, train_raw)
    apply_clip = {name: clipped_from_train_stats(train_raw, frame) for name, frame in apply_raw.items()}

    train_aux = pd.concat(
        [
            train_raw[["customer_id"]].reset_index(drop=True),
            train_quant.reset_index(drop=True),
            train_clip.reset_index(drop=True),
        ],
        axis=1,
    )
    train_aux["quant_row_mean"] = train_quant.mean(axis=1).astype(np.float32)
    train_aux["quant_row_std"] = train_quant.std(axis=1).astype(np.float32)
    train_aux["clipped_row_mean"] = train_clip.mean(axis=1).astype(np.float32)
    train_aux["clipped_row_std"] = train_clip.std(axis=1).astype(np.float32)

    apply_aux = {}
    for name in apply_raw:
        block = pd.concat(
            [
                apply_raw[name][["customer_id"]].reset_index(drop=True),
                apply_quant[name].reset_index(drop=True),
                apply_clip[name].reset_index(drop=True),
            ],
            axis=1,
        )
        block["quant_row_mean"] = apply_quant[name].mean(axis=1).astype(np.float32)
        block["quant_row_std"] = apply_quant[name].std(axis=1).astype(np.float32)
        block["clipped_row_mean"] = apply_clip[name].mean(axis=1).astype(np.float32)
        block["clipped_row_std"] = apply_clip[name].std(axis=1).astype(np.float32)
        apply_aux[name] = block

    stats_df = pd.DataFrame(aux_stats_rows)
    return train_aux, apply_aux, stats_df


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    aux_spec = json.loads(args.aux_spec.read_text())
    aux_features = aux_spec["top_features"]
    top_features = json.loads(TOP_FEATURES_FILE.read_text())

    val_pred = pd.read_parquet(args.val_pred)[["customer_id"]].sort_values("customer_id").reset_index(drop=True)
    val_ids = set(val_pred["customer_id"].tolist())

    labeled_main = load_labeled_frame(extra_feature_cols=top_features).sort_values("customer_id").reset_index(drop=True)
    labeled_aux = load_labeled_frame(extra_feature_cols=aux_features).sort_values("customer_id").reset_index(drop=True)
    test_main = load_test_frame(extra_feature_cols=top_features).sort_values("customer_id").reset_index(drop=True)
    test_aux = load_test_frame(extra_feature_cols=aux_features).sort_values("customer_id").reset_index(drop=True)

    feature_cols = [c for c in labeled_main.columns if c != "customer_id" and not c.startswith("target_")]
    target_cols = [c for c in labeled_main.columns if c.startswith("target_")]
    target_cols = sorted(target_cols, key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])))
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    train_main = labeled_main[~labeled_main["customer_id"].isin(val_ids)].copy().reset_index(drop=True)
    val_main = labeled_main[labeled_main["customer_id"].isin(val_ids)].copy().reset_index(drop=True)
    train_aux_src = labeled_aux[~labeled_aux["customer_id"].isin(val_ids)].copy().reset_index(drop=True)
    val_aux_src = labeled_aux[labeled_aux["customer_id"].isin(val_ids)].copy().reset_index(drop=True)

    train_baseline, apply_baseline, baseline_stats = prepare_main_blocks(
        train_df=train_main,
        apply_frames={"val": val_main, "test": test_main},
        num_cols=num_cols,
        cat_cols=cat_cols,
    )

    train_aux_block, apply_aux_block, aux_stats = build_aux_views(
        train_extra=train_aux_src,
        apply_frames={"val": val_aux_src, "test": test_aux},
        feature_cols=aux_features,
        n_quantiles=args.quantiles,
    )

    baseline_plus_aux_train = train_baseline.merge(train_aux_block, on="customer_id", how="inner")
    baseline_plus_aux_val = apply_baseline["val"].merge(apply_aux_block["val"], on="customer_id", how="inner")
    baseline_plus_aux_test = apply_baseline["test"].merge(apply_aux_block["test"], on="customer_id", how="inner")

    train_targets = train_main[["customer_id"] + target_cols].copy()
    val_targets = val_main[["customer_id"] + target_cols].copy()
    target_summary = pd.read_csv(args.target_summary)
    target_defs = pd.DataFrame(
        {
            "target": target_cols,
            "family": [int(t.split("_")[1]) for t in target_cols],
            "member": [int(t.split("_")[2]) for t in target_cols],
        }
    ).merge(target_summary[["target", "positive_rate", "best_auc", "best_model"]], on="target", how="left")

    feature_schema = pd.DataFrame(
        {
            "feature": feature_cols,
            "role": ["categorical" if c in cat_cols else "numeric_raw" for c in feature_cols],
            "included_in_baseline": True,
            "included_in_aux_package": False,
        }
    )
    aux_schema = pd.DataFrame(
        {
            "feature": [c for c in baseline_plus_aux_train.columns if c not in baseline_plus_aux_train.columns[: len(feature_cols) + 1]],
            "role": "aux_numeric",
            "included_in_baseline": False,
            "included_in_aux_package": True,
        }
    )
    feature_schema = pd.concat([feature_schema, aux_schema], ignore_index=True)

    out = args.artifact_dir
    train_baseline.to_parquet(out / "baseline_input_train.parquet", index=False)
    apply_baseline["val"].to_parquet(out / "baseline_input_val.parquet", index=False)
    apply_baseline["test"].to_parquet(out / "baseline_input_test.parquet", index=False)
    train_aux_block.to_parquet(out / "auxiliary_feature_block_train.parquet", index=False)
    apply_aux_block["val"].to_parquet(out / "auxiliary_feature_block_val.parquet", index=False)
    apply_aux_block["test"].to_parquet(out / "auxiliary_feature_block_test.parquet", index=False)
    baseline_plus_aux_train.to_parquet(out / "baseline_plus_aux_train.parquet", index=False)
    baseline_plus_aux_val.to_parquet(out / "baseline_plus_aux_val.parquet", index=False)
    baseline_plus_aux_test.to_parquet(out / "baseline_plus_aux_test.parquet", index=False)
    train_targets.to_parquet(out / "targets_train.parquet", index=False)
    val_targets.to_parquet(out / "targets_val.parquet", index=False)
    pd.DataFrame({"customer_id": train_main["customer_id"].astype("int32")}).to_parquet(out / "train_ids.parquet", index=False)
    pd.DataFrame({"customer_id": val_main["customer_id"].astype("int32")}).to_parquet(out / "val_ids.parquet", index=False)
    baseline_stats.to_csv(out / "baseline_transform_stats.csv", index=False)
    aux_stats.to_csv(out / "auxiliary_transform_stats.csv", index=False)
    feature_schema.to_csv(out / "feature_schema.csv", index=False)
    target_defs.to_csv(out / "target_definitions.csv", index=False)

    manifest = {
        "split_source": str(args.val_pred),
        "baseline_input": {
            "train": "baseline_input_train.parquet",
            "val": "baseline_input_val.parquet",
            "test": "baseline_input_test.parquet",
        },
        "auxiliary_block": {
            "train": "auxiliary_feature_block_train.parquet",
            "val": "auxiliary_feature_block_val.parquet",
            "test": "auxiliary_feature_block_test.parquet",
        },
        "baseline_plus_aux": {
            "train": "baseline_plus_aux_train.parquet",
            "val": "baseline_plus_aux_val.parquet",
            "test": "baseline_plus_aux_test.parquet",
        },
        "targets": {
            "train": "targets_train.parquet",
            "val": "targets_val.parquet",
        },
        "feature_schema": "feature_schema.csv",
        "target_definitions": "target_definitions.csv",
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "baseline_top_extra_features": top_features,
        "aux_top_features": aux_features,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    summary = {
        "rows_train": int(len(train_main)),
        "rows_val": int(len(val_main)),
        "rows_test": int(len(test_main)),
        "baseline_features": int(len(feature_cols)),
        "numeric_features": int(len(num_cols)),
        "categorical_features": int(len(cat_cols)),
        "aux_top_features": int(len(aux_features)),
        "aux_block_features": int(train_aux_block.shape[1] - 1),
        "baseline_plus_aux_features": int(baseline_plus_aux_train.shape[1] - 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
