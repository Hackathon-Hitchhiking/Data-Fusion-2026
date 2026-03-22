from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.layout import project_root


ROOT = project_root()
OUT_DIR = ROOT / "artifacts" / "post_fs_v1_local_analysis_v1"

TARGET_COLS = [f"target_{family}_{idx}" for family, count in {
    1: 5,
    2: 8,
    3: 5,
    4: 1,
    5: 2,
    6: 5,
    7: 3,
    8: 3,
    9: 8,
    10: 1,
}.items() for idx in range(1, count + 1)]

FOCAL_TARGETS = ["target_2_8", "target_2_3"]
ALPHAS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]


def ensure_out_dir() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def normalize_pred_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in df.columns:
        if col.startswith("predict_"):
            rename_map[col] = "target_" + col[len("predict_") :]
    return df.rename(columns=rename_map)


def load_target_scores(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_col = [c for c in df.columns if c != "target"][0]
    return df.rename(columns={value_col: "oof_auc"}).assign(model=model_name)


def load_validation_predictions(path: Path, model_name: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = normalize_pred_columns(df)
    keep_cols = ["customer_id", *TARGET_COLS]
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise KeyError(f"{model_name}: missing columns {missing}")
    return df[keep_cols].copy()


def compute_macro_auc(y_true: pd.DataFrame, pred: pd.DataFrame) -> float:
    scores = []
    for target in TARGET_COLS:
        scores.append(roc_auc_score(y_true[target], pred[target]))
    return float(np.mean(scores))


def compute_target_auc(y_true: pd.DataFrame, pred: pd.DataFrame, target: str) -> float:
    return float(roc_auc_score(y_true[target], pred[target]))


def main() -> None:
    out_dir = ensure_out_dir()

    scores_paths = {
        "tabm_v1": ROOT / "output/kaggle-output/tabm-backbone/watch_v4/artifacts/gpu_tabm_full_gpu/target_scores.csv",
        "tabm_v3": ROOT / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/target_scores.csv",
        "tabm_fs_v1": ROOT / "output/kaggle-output/tabm-longrun-v3-fs-v1/pull_complete/artifacts/gpu_tabm_longrun_v3_fs_v1_full_gpu/target_scores.csv",
        "gandalf": ROOT / "artifacts/gpu_gandalf_full_gpu/target_scores.csv",
        "tabnet": ROOT / "artifacts/gpu_tabnet_full_gpu/target_scores.csv",
        "lgbm_750k": ROOT / "artifacts/full_lgbm_top100_750k_2fold/target_scores.csv",
    }
    score_tables = [load_target_scores(path, name) for name, path in scores_paths.items()]
    all_scores = pd.concat(score_tables, ignore_index=True)

    fs_v1_scores = all_scores[all_scores["model"] == "tabm_fs_v1"][["target", "oof_auc"]].rename(
        columns={"oof_auc": "oof_auc_fs_v1"}
    )
    v3_scores = all_scores[all_scores["model"] == "tabm_v3"][["target", "oof_auc"]].rename(
        columns={"oof_auc": "oof_auc_tabm_v3"}
    )
    weakest = (
        fs_v1_scores.merge(v3_scores, on="target", how="left")
        .assign(delta_vs_tabm_v3=lambda df: df["oof_auc_fs_v1"] - df["oof_auc_tabm_v3"])
        .sort_values("oof_auc_fs_v1", ascending=True)
        .reset_index(drop=True)
    )
    weakest["rank_fs_v1"] = np.arange(1, len(weakest) + 1)
    weakest.to_csv(out_dir / "weakest_targets_fs_v1.csv", index=False)

    focal_scores = (
        all_scores[all_scores["target"].isin(FOCAL_TARGETS)]
        .pivot(index="target", columns="model", values="oof_auc")
        .reset_index()
    )
    focal_scores.to_csv(out_dir / "focal_targets_model_scores.csv", index=False)

    pred_paths = {
        "tabm_fs_v1": ROOT / "output/kaggle-output/tabm-longrun-v3-fs-v1/pull_complete/artifacts/gpu_tabm_longrun_v3_fs_v1_full_gpu/validation_predictions.parquet",
        "tabm_v3": ROOT / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet",
        "tabm_v1": ROOT / "output/kaggle-output/tabm-backbone/watch_v4/artifacts/gpu_tabm_full_gpu/validation_predictions.parquet",
        "gandalf": ROOT / "artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet",
    }
    pred_frames = {name: load_validation_predictions(path, name) for name, path in pred_paths.items()}

    baseline = pred_frames["tabm_fs_v1"].copy()
    val_ids = baseline[["customer_id"]].copy()

    train_target = pd.read_parquet(ROOT / "data/competition/train_target.parquet")
    y_val = val_ids.merge(train_target, on="customer_id", how="left")
    if y_val[TARGET_COLS].isna().any().any():
        raise ValueError("Validation targets contain NaN after merge")
    y_true = y_val[TARGET_COLS].copy()
    baseline_pred = baseline[TARGET_COLS].copy()

    baseline_macro = compute_macro_auc(y_true, baseline_pred)

    aligned_sources: dict[str, pd.DataFrame] = {}
    for name, df in pred_frames.items():
        aligned = val_ids.merge(df, on="customer_id", how="left")
        if aligned[TARGET_COLS].isna().any().any():
            raise ValueError(f"{name}: NaN after aligning validation predictions")
        aligned_sources[name] = aligned[TARGET_COLS].copy()

    rows: list[dict[str, float | str]] = []
    for target in FOCAL_TARGETS:
        for source_name in ["tabm_v3", "tabm_v1", "gandalf"]:
            source_pred = aligned_sources[source_name]
            for alpha in ALPHAS:
                candidate = baseline_pred.copy()
                candidate[target] = (1.0 - alpha) * baseline_pred[target] + alpha * source_pred[target]
                rows.append(
                    {
                        "target": target,
                        "source_model": source_name,
                        "alpha": alpha,
                        "target_auc": compute_target_auc(y_true, candidate, target),
                        "full_macro_auc": compute_macro_auc(y_true, candidate),
                        "delta_vs_baseline": compute_macro_auc(y_true, candidate) - baseline_macro,
                    }
                )
            full_replace = baseline_pred.copy()
            full_replace[target] = source_pred[target]
            rows.append(
                {
                    "target": target,
                    "source_model": source_name,
                    "alpha": 1.0,
                    "target_auc": compute_target_auc(y_true, full_replace, target),
                    "full_macro_auc": compute_macro_auc(y_true, full_replace),
                    "delta_vs_baseline": compute_macro_auc(y_true, full_replace) - baseline_macro,
                }
            )

    fallback_grid = pd.DataFrame(rows).sort_values(["target", "delta_vs_baseline"], ascending=[True, False])
    fallback_grid.to_csv(out_dir / "focal_target_fallback_grid.csv", index=False)

    best_single = fallback_grid.groupby("target", as_index=False).first()

    combo_rows: list[dict[str, float | str]] = []
    best_candidates = {}
    for _, row in best_single.iterrows():
        best_candidates[row["target"]] = {
            "source_model": row["source_model"],
            "alpha": float(row["alpha"]),
        }

    # Evaluate combined best-single configuration on both focal targets.
    combo = baseline_pred.copy()
    combo_desc = {}
    for target, cfg in best_candidates.items():
        source = aligned_sources[str(cfg["source_model"])]
        alpha = float(cfg["alpha"])
        combo[target] = (1.0 - alpha) * baseline_pred[target] + alpha * source[target]
        combo_desc[target] = cfg

    combo_macro = compute_macro_auc(y_true, combo)
    combo_rows.append(
        {
            "candidate": "combined_best_single_fallbacks",
            "full_macro_auc": combo_macro,
            "delta_vs_baseline": combo_macro - baseline_macro,
            "details": json.dumps(combo_desc, ensure_ascii=True),
        }
    )

    # Also evaluate hard replacement only for target_2_8 from GANDALF if available.
    if "gandalf" in aligned_sources:
        t28 = baseline_pred.copy()
        t28["target_2_8"] = aligned_sources["gandalf"]["target_2_8"]
        macro = compute_macro_auc(y_true, t28)
        combo_rows.append(
            {
                "candidate": "replace_target_2_8_with_gandalf",
                "full_macro_auc": macro,
                "delta_vs_baseline": macro - baseline_macro,
                "details": json.dumps({"target_2_8": {"source_model": "gandalf", "alpha": 1.0}}, ensure_ascii=True),
            }
        )

    combo_df = pd.DataFrame(combo_rows).sort_values("delta_vs_baseline", ascending=False)
    combo_df.to_csv(out_dir / "focal_target_combo_candidates.csv", index=False)

    summary = {
        "baseline_macro_auc": baseline_macro,
        "weakest_12_fs_v1": weakest.head(12)[["target", "oof_auc_fs_v1", "delta_vs_tabm_v3"]].to_dict(orient="records"),
        "focal_model_scores": focal_scores.to_dict(orient="records"),
        "best_single_fallbacks": best_single.to_dict(orient="records"),
        "best_combo_candidate": combo_df.iloc[0].to_dict() if not combo_df.empty else {},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
