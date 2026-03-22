from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from lib.layout import project_root, resolve_data_dir
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DATA_DIR = resolve_data_dir()
DEFAULT_OUT_DIR = ROOT / "artifacts" / "lgbm_extended_specialists"
TOP_EXTRA_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
GANDALF_VAL_FILE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
GANDALF_TARGET_SCORE_FILE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "target_scores.csv"
STACK_OOF_FILE = ROOT / "artifacts" / "stack_top100_self_c005_full_750k" / "oof_predictions.parquet"
BASE_OOF_FILE = ROOT / "artifacts" / "full_lgbm_top100_750k_2fold" / "oof_predictions.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run extended LGBM specialists on hardest targets")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hardest-count", type=int, default=12)
    p.add_argument("--baseline-mode", choices=["gandalf", "g95s05", "g90s10"], default="gandalf")
    p.add_argument("--alpha-grid", type=str, default="0.05,0.10,0.15,0.20,0.30")
    p.add_argument("--n-estimators", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=128)
    p.add_argument("--feature-fraction", type=float, default=0.7)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-child-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def add_group_score_features(frame: pd.DataFrame, prefix: str, pred_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    groups: dict[str, list[str]] = {}
    for col in pred_cols:
        target_name = col.replace("predict_", "target_")
        group = target_name.split("_")[1]
        groups.setdefault(group, []).append(col)
    for group, cols in groups.items():
        out[f"{prefix}group_{group}_sum"] = out[cols].sum(axis=1)
        out[f"{prefix}group_{group}_max"] = out[cols].max(axis=1)
    return out


def load_feature_frame() -> tuple[pd.DataFrame, list[str], list[str]]:
    train_main = pd.read_parquet(DATA_DIR / "train_main_features.parquet")
    train_extra = pd.read_parquet(DATA_DIR / "train_extra_features.parquet")
    train_target = pd.read_parquet(DATA_DIR / "train_target.parquet")
    top_extra = json.loads(TOP_EXTRA_FILE.read_text())

    labeled = train_main.merge(train_target, on="customer_id", how="inner")
    labeled = labeled.merge(train_extra[["customer_id"] + top_extra], on="customer_id", how="left")

    target_cols = [c for c in labeled.columns if c.startswith("target_")]
    feature_cols = [c for c in labeled.columns if c not in ["customer_id"] + target_cols]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    return labeled, feature_cols, cat_cols


def load_prediction_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gandalf = normalize_prediction_columns(pd.read_parquet(GANDALF_VAL_FILE))
    stack = pd.read_parquet(STACK_OOF_FILE)
    base = pd.read_parquet(BASE_OOF_FILE)
    return gandalf, stack, base


def get_hardest_targets(count: int) -> list[str]:
    score_df = pd.read_csv(GANDALF_TARGET_SCORE_FILE).sort_values("oof_auc")
    return score_df.head(count)["target"].tolist()


def build_baseline(gandalf_pred: pd.DataFrame, stack_pred: pd.DataFrame, mode: str) -> pd.DataFrame:
    pred_cols = [c for c in gandalf_pred.columns if c.startswith("predict_")]
    out = gandalf_pred[["customer_id"] + pred_cols].copy()
    for col in pred_cols:
        out[col] = out[col].astype("float32")

    if mode == "gandalf":
        return out

    if mode == "g95s05":
        alpha = 0.05
    elif mode == "g90s10":
        alpha = 0.10
    else:
        raise ValueError(f"Unsupported baseline mode: {mode}")

    for col in pred_cols:
        out[col] = (
            (1.0 - alpha) * gandalf_pred[col].to_numpy(dtype="float32")
            + alpha * stack_pred[col].to_numpy(dtype="float32")
        ).astype("float32")
    return out


def frame_mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]

    hardest_targets = get_hardest_targets(args.hardest_count)
    print(
        {"stage": "load_data_start", "hardest_targets": hardest_targets, "baseline_mode": args.baseline_mode},
        flush=True,
    )
    labeled, raw_feature_cols, cat_cols = load_feature_frame()
    gandalf_val, stack_oof, base_oof = load_prediction_frames()

    pred_cols = [c for c in gandalf_val.columns if c.startswith("predict_")]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]

    val_ids = gandalf_val[["customer_id"]].copy()
    val_id_set = set(val_ids["customer_id"])
    val_df = labeled.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    train_df = labeled[~labeled["customer_id"].isin(val_id_set)].sort_values("customer_id").reset_index(drop=True)

    stack_val = stack_oof.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    base_val = base_oof.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    stack_train = stack_oof[~stack_oof["customer_id"].isin(val_id_set)].sort_values("customer_id").reset_index(drop=True)
    base_train = base_oof[~base_oof["customer_id"].isin(val_id_set)].sort_values("customer_id").reset_index(drop=True)

    for frame, prefix in [(stack_train, "stack_"), (stack_val, "stack_"), (base_train, "base_"), (base_val, "base_")]:
        rename_map = {c: f"{prefix}{c}" for c in frame.columns if c.startswith("predict_")}
        frame.rename(columns=rename_map, inplace=True)
        score_cols = [c for c in frame.columns if c.startswith(prefix)]
        for col in score_cols:
            if col != "customer_id":
                frame[col] = frame[col].astype("float32")

    stack_pred_cols = [c for c in stack_train.columns if c.startswith("stack_predict_")]
    base_pred_cols = [c for c in base_train.columns if c.startswith("base_predict_")]
    stack_train = add_group_score_features(stack_train, "stack_", stack_pred_cols)
    stack_val = add_group_score_features(stack_val, "stack_", stack_pred_cols)
    base_train = add_group_score_features(base_train, "base_", base_pred_cols)
    base_val = add_group_score_features(base_val, "base_", base_pred_cols)

    train_feat = (
        train_df[["customer_id"] + raw_feature_cols]
        .merge(stack_train, on="customer_id", how="left")
        .merge(base_train, on="customer_id", how="left")
    )
    val_feat = (
        val_df[["customer_id"] + raw_feature_cols]
        .merge(stack_val, on="customer_id", how="left")
        .merge(base_val, on="customer_id", how="left")
    )

    model_feature_cols = [c for c in train_feat.columns if c != "customer_id"]
    train_X = train_feat[model_feature_cols].copy()
    val_X = val_feat[model_feature_cols].copy()

    for col in cat_cols:
        train_X[col] = train_X[col].fillna(-1).astype("float32").astype("Int32")
        val_X[col] = val_X[col].fillna(-1).astype("float32").astype("Int32")

    num_cols = [c for c in model_feature_cols if c not in cat_cols]
    for col in num_cols:
        train_X[col] = pd.to_numeric(train_X[col], errors="coerce").astype("float32")
        val_X[col] = pd.to_numeric(val_X[col], errors="coerce").astype("float32")

    print(
        {
            "stage": "memory_ready",
            "train_x_mb": round(frame_mem_mb(train_X), 2),
            "val_x_mb": round(frame_mem_mb(val_X), 2),
            "feature_count": len(model_feature_cols),
        },
        flush=True,
    )

    gandalf_sorted = gandalf_val.sort_values("customer_id").reset_index(drop=True)
    stack_val_for_blend = stack_oof.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    baseline_candidate = build_baseline(gandalf_sorted, stack_val_for_blend, args.baseline_mode)

    blended_candidates: dict[str, pd.DataFrame] = {f"{args.baseline_mode}_base": baseline_candidate.copy()}
    for alpha in alpha_grid:
        blended_candidates[f"specialist_blend_{alpha:.2f}"] = baseline_candidate.copy()

    specialist_candidate = baseline_candidate.copy()
    per_target_rows: list[dict[str, float | int | str]] = []
    y_val_full = val_df[["customer_id"] + target_cols].copy()
    print(
        {"stage": "lgbm_specialists_start", "target_count": len(hardest_targets), "feature_count": len(model_feature_cols)},
        flush=True,
    )

    del labeled, stack_train, stack_val, base_train, base_val, train_feat, val_feat, stack_oof, base_oof
    gc.collect()

    for idx, target_name in enumerate(hardest_targets, start=1):
        pred_col = target_name.replace("target_", "predict_")
        y_train = train_df[target_name].to_numpy()
        y_val = val_df[target_name].to_numpy()
        pos_rate = float(y_train.mean())
        scale_pos_weight = (1.0 - pos_rate) / (pos_rate + 1e-6)
        print(
            {"stage": "lgbm_target_start", "target": target_name, "index": idx, "positive_rate_train": pos_rate},
            flush=True,
        )

        model = lgb.LGBMClassifier(
            objective="binary",
            metric="auc",
            boosting_type="gbdt",
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
            feature_fraction=args.feature_fraction,
            bagging_fraction=args.bagging_fraction,
            bagging_freq=args.bagging_freq,
            min_child_samples=args.min_child_samples,
            random_state=args.seed,
            n_jobs=4,
            verbose=-1,
            scale_pos_weight=scale_pos_weight,
        )
        model.fit(
            train_X,
            y_train,
            eval_set=[(val_X, y_val)],
            eval_metric="auc",
            categorical_feature=[c for c in cat_cols if c in train_X.columns],
            callbacks=[lgb.early_stopping(80, verbose=False)],
        )
        val_pred = model.predict_proba(val_X)[:, 1].astype("float32")

        baseline_auc = roc_auc_score(y_val, baseline_candidate[pred_col])
        lgbm_auc = roc_auc_score(y_val, val_pred)
        per_target_rows.append(
            {
                "target": target_name,
                "baseline_auc": float(baseline_auc),
                "lgbm_auc": float(lgbm_auc),
                "delta_auc": float(lgbm_auc - baseline_auc),
                "best_iteration": int(getattr(model, "best_iteration_", args.n_estimators) or args.n_estimators),
            }
        )
        print(
            {"stage": "lgbm_target_done", "target": target_name, "lgbm_auc": float(lgbm_auc), "delta_auc": float(lgbm_auc - baseline_auc)},
            flush=True,
        )

        specialist_candidate[pred_col] = val_pred
        for alpha in alpha_grid:
            cand = blended_candidates[f"specialist_blend_{alpha:.2f}"]
            cand[pred_col] = ((1.0 - alpha) * cand[pred_col].to_numpy(dtype="float32") + alpha * val_pred).astype("float32")

        del model, y_train, y_val, val_pred
        gc.collect()

    per_target_df = pd.DataFrame(per_target_rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)
    per_target_df.to_csv(args.out_dir / "per_target_auc.csv", index=False)

    candidate_rows: list[dict[str, float | str]] = []
    for name, pred_df in blended_candidates.items():
        auc = macro_auc(y_val_full, pred_df, target_cols)
        candidate_rows.append({"candidate": name, "macro_auc": auc})
        print({"stage": "candidate_done", "candidate": name, "macro_auc": float(auc)}, flush=True)

    cand_df = pd.DataFrame(candidate_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    cand_df.to_csv(args.out_dir / "candidate_macro_auc.csv", index=False)

    baseline_candidate.to_parquet(args.out_dir / "baseline_val_predictions.parquet", index=False)
    specialist_candidate.to_parquet(args.out_dir / "specialist_val_predictions.parquet", index=False)
    best_candidate_name = str(cand_df.iloc[0]["candidate"])
    blended_candidates[best_candidate_name].to_parquet(args.out_dir / "best_candidate_val_predictions.parquet", index=False)

    summary = {
        "hardest_targets": hardest_targets,
        "baseline_mode": args.baseline_mode,
        "best_candidate": best_candidate_name,
        "best_macro_auc": float(cand_df.iloc[0]["macro_auc"]),
        "base_macro_auc": float(cand_df.loc[cand_df["candidate"] == f"{args.baseline_mode}_base", "macro_auc"].iloc[0]),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(per_target_df.to_string(index=False), flush=True)
    print(cand_df.to_string(index=False), flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
