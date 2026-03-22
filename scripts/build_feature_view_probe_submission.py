from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from lib.data_loading import load_labeled_frame, load_test_frame
from lib.layout import project_root
from lib.metrics import macro_auc
from lib.meta_views import safe_logit
from lib.submission import normalize_prediction_columns, write_submission_like_sample


ROOT = project_root()
DEFAULT_TABM_VAL = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_TABM_TEST = ROOT / "output/submissions/tabm_longrun_v3_finalonly_submission.parquet"
DEFAULT_GANDALF_VAL = ROOT / "artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet"
DEFAULT_GANDALF_TEST = ROOT / "artifacts/gpu_gandalf_full_gpu/submission.parquet"
DEFAULT_SPECIALIST_VAL = (
    ROOT / "artifacts/catboost_extended_specialists_gandalf_h12_router/specialist_val_predictions.parquet"
)
DEFAULT_SPECIALIST_TEST = ROOT / "artifacts/catboost_specialist_submissions_v1/specialist_test_predictions.parquet"
DEFAULT_TOP_FEATURES = ROOT / "artifacts/feature_selection/top100_extra_gain.json"
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/feature_view_submission_probe_v1"
DEFAULT_OUTPUT = ROOT / "output/submissions/tabm_v3_feature_view_clip_probe_a0p01_submission.parquet"
DEFAULT_TARGETS = [
    "target_2_6",
    "target_2_4",
    "target_9_3",
    "target_3_1",
    "target_9_7",
    "target_10_1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build one conservative feature-view probe submission.")
    parser.add_argument("--tabm-val", type=Path, default=DEFAULT_TABM_VAL)
    parser.add_argument("--tabm-test", type=Path, default=DEFAULT_TABM_TEST)
    parser.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    parser.add_argument("--gandalf-test", type=Path, default=DEFAULT_GANDALF_TEST)
    parser.add_argument("--specialist-val", type=Path, default=DEFAULT_SPECIALIST_VAL)
    parser.add_argument("--specialist-test", type=Path, default=DEFAULT_SPECIALIST_TEST)
    parser.add_argument("--top-features-file", type=Path, default=DEFAULT_TOP_FEATURES)
    parser.add_argument("--top-k-features", type=int, default=24)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--c-value", type=float, default=1.0)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def pred_col(target_name: str) -> str:
    return target_name.replace("target_", "predict_")


def load_prediction_frame(path: Path) -> pd.DataFrame:
    return normalize_prediction_columns(pd.read_parquet(path)).sort_values("customer_id").reset_index(drop=True)


def load_feature_names(path: Path, top_k: int) -> list[str]:
    names = json.loads(path.read_text())
    return list(dict.fromkeys(names[:top_k]))


def numeric_frame(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    out = df[["customer_id"] + feature_cols].copy()
    for col in feature_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        median = float(out[col].median()) if out[col].notna().any() else 0.0
        out[col] = out[col].fillna(median).astype("float32")
    return out.sort_values("customer_id").reset_index(drop=True)


def build_clipped_view_from_stats(
    train_raw: pd.DataFrame,
    apply_raw: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    out = pd.DataFrame({"customer_id": apply_raw["customer_id"].to_numpy(dtype=np.int32)})
    clipped_cols: list[str] = []
    for col in feature_cols:
        train_col = train_raw[col].to_numpy(dtype=np.float32)
        apply_col = apply_raw[col].to_numpy(dtype=np.float32)
        median = float(np.median(train_col))
        q1 = float(np.quantile(train_col, 0.25))
        q3 = float(np.quantile(train_col, 0.75))
        iqr = max(q3 - q1, 1e-6)
        transformed = np.tanh((apply_col - median) / (4.0 * iqr)).astype(np.float32)
        out[f"{col}__clipped"] = transformed
        clipped_cols.append(f"{col}__clipped")
    clipped_np = out[clipped_cols].to_numpy(dtype=np.float32)
    out["clipped_row_mean"] = clipped_np.mean(axis=1).astype(np.float32)
    out["clipped_row_std"] = clipped_np.std(axis=1).astype(np.float32)
    return out


def build_meta_feature_frame(
    tabm_df: pd.DataFrame,
    gandalf_df: pd.DataFrame,
    specialist_df: pd.DataFrame,
    clipped_df: pd.DataFrame,
    target_name: str,
) -> pd.DataFrame:
    col = pred_col(target_name)
    merged = (
        tabm_df[["customer_id", col]]
        .rename(columns={col: "tabm_score"})
        .merge(gandalf_df[["customer_id", col]].rename(columns={col: "gandalf_score"}), on="customer_id", how="inner")
        .merge(specialist_df[["customer_id", col]].rename(columns={col: "specialist_score"}), on="customer_id", how="inner")
        .merge(clipped_df, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    merged["tabm_logit"] = safe_logit(merged["tabm_score"]).astype("float32")
    merged["gandalf_logit"] = safe_logit(merged["gandalf_score"]).astype("float32")
    merged["specialist_logit"] = safe_logit(merged["specialist_score"]).astype("float32")
    merged["gandalf_minus_tabm"] = (merged["gandalf_score"] - merged["tabm_score"]).astype("float32")
    merged["specialist_minus_tabm"] = (merged["specialist_score"] - merged["tabm_score"]).astype("float32")
    return merged


def fit_oof_and_test(
    train_meta: pd.DataFrame,
    train_raw: pd.DataFrame,
    test_meta_base: pd.DataFrame,
    test_raw: pd.DataFrame,
    train_y: np.ndarray,
    *,
    folds: int,
    seed: int,
    c_value: float,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    oof = np.zeros(len(train_meta), dtype=np.float32)

    for fold_train_idx, fold_val_idx in splitter.split(np.zeros(len(train_y)), train_y):
        fold_train_raw = train_raw.iloc[fold_train_idx].reset_index(drop=True)
        fold_val_raw = train_raw.iloc[fold_val_idx].reset_index(drop=True)
        fold_train_clip = build_clipped_view_from_stats(fold_train_raw, fold_train_raw, feature_cols)
        fold_val_clip = build_clipped_view_from_stats(fold_train_raw, fold_val_raw, feature_cols)

        train_block = train_meta.iloc[fold_train_idx][["customer_id", "tabm_score", "gandalf_score", "specialist_score", "tabm_logit", "gandalf_logit", "specialist_logit", "gandalf_minus_tabm", "specialist_minus_tabm"]]
        val_block = train_meta.iloc[fold_val_idx][["customer_id", "tabm_score", "gandalf_score", "specialist_score", "tabm_logit", "gandalf_logit", "specialist_logit", "gandalf_minus_tabm", "specialist_minus_tabm"]]
        fold_train_x = train_block.merge(fold_train_clip, on="customer_id", how="inner").drop(columns=["customer_id"]).to_numpy(dtype=np.float32)
        fold_val_x = val_block.merge(fold_val_clip, on="customer_id", how="inner").drop(columns=["customer_id"]).to_numpy(dtype=np.float32)

        model = LogisticRegression(
            C=c_value,
            max_iter=500,
            solver="liblinear",
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(fold_train_x, train_y[fold_train_idx])
        oof[fold_val_idx] = model.predict_proba(fold_val_x)[:, 1].astype(np.float32)

    full_clip_train = build_clipped_view_from_stats(train_raw, train_raw, feature_cols)
    full_clip_test = build_clipped_view_from_stats(train_raw, test_raw, feature_cols)
    full_train_x = (
        train_meta[["customer_id", "tabm_score", "gandalf_score", "specialist_score", "tabm_logit", "gandalf_logit", "specialist_logit", "gandalf_minus_tabm", "specialist_minus_tabm"]]
        .merge(full_clip_train, on="customer_id", how="inner")
        .drop(columns=["customer_id"])
        .to_numpy(dtype=np.float32)
    )
    full_test_x = (
        test_meta_base[["customer_id", "tabm_score", "gandalf_score", "specialist_score", "tabm_logit", "gandalf_logit", "specialist_logit", "gandalf_minus_tabm", "specialist_minus_tabm"]]
        .merge(full_clip_test, on="customer_id", how="inner")
        .drop(columns=["customer_id"])
        .to_numpy(dtype=np.float32)
    )
    final_model = LogisticRegression(
        C=c_value,
        max_iter=500,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )
    final_model.fit(full_train_x, train_y)
    test_proba = final_model.predict_proba(full_test_x)[:, 1].astype(np.float32)
    return oof, test_proba


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = load_feature_names(args.top_features_file, args.top_k_features)

    tabm_val = load_prediction_frame(args.tabm_val)
    gandalf_val = load_prediction_frame(args.gandalf_val)
    specialist_val = load_prediction_frame(args.specialist_val)
    tabm_test = load_prediction_frame(args.tabm_test)
    gandalf_test = load_prediction_frame(args.gandalf_test)
    specialist_test = load_prediction_frame(args.specialist_test)

    val_customer_ids = tabm_val["customer_id"].astype("int32")
    labeled = load_labeled_frame(extra_feature_cols=feature_cols)
    all_targets = sorted([c for c in labeled.columns if c.startswith("target_")], key=lambda x: (int(x.split("_")[1]), int(x.split("_")[2])))

    val_df = labeled[labeled["customer_id"].isin(set(val_customer_ids))][["customer_id"] + args.targets + feature_cols].copy()
    val_df = val_df.sort_values("customer_id").reset_index(drop=True)
    val_raw = numeric_frame(val_df, feature_cols)

    test_df = load_test_frame(extra_feature_cols=feature_cols)
    test_df = test_df[["customer_id"] + feature_cols].copy().sort_values("customer_id").reset_index(drop=True)
    test_raw = numeric_frame(test_df, feature_cols)

    y_holdout = labeled[labeled["customer_id"].isin(set(val_customer_ids))][["customer_id"] + all_targets].copy()
    y_holdout = y_holdout.sort_values("customer_id").reset_index(drop=True)

    base_val = tabm_val[["customer_id"] + [pred_col(t) for t in all_targets]].copy()
    base_test = tabm_test[["customer_id"] + [pred_col(t) for t in all_targets]].copy()
    probe_val = base_val.copy()
    probe_test = base_test.copy()

    per_target_rows: list[dict[str, object]] = []

    for target_name in args.targets:
        train_meta = build_meta_feature_frame(tabm_val, gandalf_val, specialist_val, val_raw, target_name)
        test_meta = build_meta_feature_frame(tabm_test, gandalf_test, specialist_test, test_raw, target_name)
        y = y_holdout[target_name].to_numpy(dtype=np.int8)

        oof_proba, test_proba = fit_oof_and_test(
            train_meta=train_meta,
            train_raw=val_raw,
            test_meta_base=test_meta,
            test_raw=test_raw,
            train_y=y,
            folds=args.folds,
            seed=args.seed,
            c_value=args.c_value,
            feature_cols=feature_cols,
        )

        col = pred_col(target_name)
        base_val_scores = base_val[col].to_numpy(dtype=np.float32)
        base_test_scores = base_test[col].to_numpy(dtype=np.float32)
        blended_val = np.clip(base_val_scores + args.alpha * (oof_proba - base_val_scores), 1e-6, 1 - 1e-6)
        blended_test = np.clip(base_test_scores + args.alpha * (test_proba - base_test_scores), 1e-6, 1 - 1e-6)
        probe_val[col] = blended_val.astype(np.float32)
        probe_test[col] = blended_test.astype(np.float32)

        base_auc = float(
            macro_auc(
                y_holdout[["customer_id", target_name]],
                base_val[["customer_id", col]],
                [target_name],
            )
        )
        probe_auc = float(
            macro_auc(
                y_holdout[["customer_id", target_name]],
                probe_val[["customer_id", col]],
                [target_name],
            )
        )
        per_target_rows.append(
            {
                "target": target_name,
                "base_auc": base_auc,
                "probe_auc": probe_auc,
                "delta": probe_auc - base_auc,
                "alpha": args.alpha,
                "feature_view": "clip_only",
            }
        )

    base_macro = float(macro_auc(y_holdout, base_val, all_targets))
    probe_macro = float(macro_auc(y_holdout, probe_val, all_targets))
    subset_base = float(
        macro_auc(
            y_holdout[["customer_id"] + args.targets],
            base_val[["customer_id"] + [pred_col(t) for t in args.targets]],
            args.targets,
        )
    )
    subset_probe = float(
        macro_auc(
            y_holdout[["customer_id"] + args.targets],
            probe_val[["customer_id"] + [pred_col(t) for t in args.targets]],
            args.targets,
        )
    )

    write_submission_like_sample(
        customer_ids=probe_test["customer_id"].to_numpy(dtype=np.int32),
        prediction_frame=probe_test,
        output_path=args.output_path,
    )

    per_target_df = pd.DataFrame(per_target_rows).sort_values("delta", ascending=False).reset_index(drop=True)
    per_target_df.to_csv(args.artifact_dir / "per_target_probe_delta.csv", index=False)
    probe_val.to_parquet(args.artifact_dir / "probe_validation_predictions.parquet", index=False)
    probe_test.to_parquet(args.artifact_dir / "probe_test_predictions.parquet", index=False)

    diff_rows = []
    for target_name in args.targets:
        col = pred_col(target_name)
        diff = np.abs(probe_test[col].to_numpy(dtype=np.float32) - base_test[col].to_numpy(dtype=np.float32))
        diff_rows.append(
            {
                "target": target_name,
                "mean_abs_shift": float(diff.mean()),
                "p95_abs_shift": float(np.quantile(diff, 0.95)),
                "max_abs_shift": float(diff.max()),
            }
        )
    diff_df = pd.DataFrame(diff_rows).sort_values("mean_abs_shift", ascending=False).reset_index(drop=True)
    diff_df.to_csv(args.artifact_dir / "test_shift_summary.csv", index=False)

    summary = {
        "candidate_name": "tabm_v3_feature_view_clip_probe_a0p01",
        "rationale": "Conservative probe: clip-only transformed feature view as a tiny correction layer on six weak targets.",
        "targets": args.targets,
        "alpha": args.alpha,
        "top_k_features": args.top_k_features,
        "feature_view": "clip_only",
        "base_holdout_macro_auc": base_macro,
        "probe_holdout_macro_auc": probe_macro,
        "delta_vs_base_holdout_macro": probe_macro - base_macro,
        "base_holdout_subset_auc": subset_base,
        "probe_holdout_subset_auc": subset_probe,
        "delta_vs_base_holdout_subset": subset_probe - subset_base,
        "submission_path": str(args.output_path),
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
