from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score

from lib.layout import project_root, resolve_data_dir
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DATA_DIR = resolve_data_dir()
ARTIFACT_DIR = ROOT / "artifacts" / "catboost_hard_specialists"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

TOP_EXTRA_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
GANDALF_VAL_FILE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
STACK_OOF_FILE = ROOT / "artifacts" / "stack_top100_self_c005_full_750k" / "oof_predictions.parquet"
BASE_OOF_FILE = ROOT / "artifacts" / "full_lgbm_top100_750k_2fold" / "oof_predictions.parquet"

HARDEST_TARGETS = [
    "target_3_1",
    "target_9_3",
    "target_9_6",
    "target_2_4",
]


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


def main() -> None:
    labeled, raw_feature_cols, cat_cols = load_feature_frame()
    gandalf_val, stack_oof, base_oof = load_prediction_frames()

    pred_cols = [c for c in gandalf_val.columns if c.startswith("predict_")]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]

    val_ids = gandalf_val[["customer_id"]].copy()
    val_df = labeled.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    train_df = labeled[~labeled["customer_id"].isin(set(val_ids["customer_id"]))].sort_values("customer_id").reset_index(drop=True)

    stack_val = stack_oof.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    base_val = base_oof.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    stack_train = stack_oof[~stack_oof["customer_id"].isin(set(val_ids["customer_id"]))].sort_values("customer_id").reset_index(drop=True)
    base_train = base_oof[~base_oof["customer_id"].isin(set(val_ids["customer_id"]))].sort_values("customer_id").reset_index(drop=True)

    for frame in [stack_train, stack_val, base_train, base_val]:
        rename_map = {c: f"stack_{c}" for c in frame.columns if c.startswith("predict_")} if frame is stack_train or frame is stack_val else {c: f"base_{c}" for c in frame.columns if c.startswith("predict_")}
        frame.rename(columns=rename_map, inplace=True)

    stack_pred_cols = [c for c in stack_train.columns if c.startswith("stack_predict_")]
    base_pred_cols = [c for c in base_train.columns if c.startswith("base_predict_")]
    stack_train = add_group_score_features(stack_train, "stack_", stack_pred_cols)
    stack_val = add_group_score_features(stack_val, "stack_", stack_pred_cols)
    base_train = add_group_score_features(base_train, "base_", base_pred_cols)
    base_val = add_group_score_features(base_val, "base_", base_pred_cols)

    train_feat = train_df[["customer_id"] + raw_feature_cols].merge(stack_train, on="customer_id", how="left").merge(base_train, on="customer_id", how="left")
    val_feat = val_df[["customer_id"] + raw_feature_cols].merge(stack_val, on="customer_id", how="left").merge(base_val, on="customer_id", how="left")

    model_feature_cols = [c for c in train_feat.columns if c != "customer_id"]
    train_X = train_feat[model_feature_cols].copy()
    val_X = val_feat[model_feature_cols].copy()

    for col in cat_cols:
        train_X[col] = train_X[col].fillna(-1).astype("int64").astype("category")
        val_X[col] = val_X[col].fillna(-1).astype("int64").astype("category")

    num_cols = [c for c in model_feature_cols if c not in cat_cols]
    for col in num_cols:
        train_X[col] = train_X[col].astype("float32")
        val_X[col] = val_X[col].astype("float32")

    gandalf_pred = gandalf_val.sort_values("customer_id").reset_index(drop=True)
    base_candidate = gandalf_pred[["customer_id"] + pred_cols].copy()

    per_target = []
    blended_candidates: dict[str, pd.DataFrame] = {
        "gandalf_base": base_candidate.copy(),
    }
    for alpha in [0.10, 0.20, 0.30]:
        blended_candidates[f"replace_blend_{alpha:.2f}"] = base_candidate.copy()

    cat_feature_indices = [train_X.columns.get_loc(c) for c in cat_cols]

    for target_name in HARDEST_TARGETS:
        pred_col = target_name.replace("target_", "predict_")
        y_train = train_df[target_name].to_numpy()
        y_val = val_df[target_name].to_numpy()

        train_pool = Pool(train_X, y_train, cat_features=cat_feature_indices)
        val_pool = Pool(val_X, y_val, cat_features=cat_feature_indices)

        model = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=600,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=10.0,
            random_seed=42,
            auto_class_weights="Balanced",
            verbose=False,
        )
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)
        val_pred = model.predict_proba(val_pool)[:, 1].astype("float64")
        gandalf_auc = roc_auc_score(y_val, base_candidate[pred_col])
        cat_auc = roc_auc_score(y_val, val_pred)

        per_target.append(
            {
                "target": target_name,
                "gandalf_auc": float(gandalf_auc),
                "catboost_auc": float(cat_auc),
                "delta_auc": float(cat_auc - gandalf_auc),
                "best_iteration": int(model.get_best_iteration()),
            }
        )

        for alpha in [0.10, 0.20, 0.30]:
            cand = blended_candidates[f"replace_blend_{alpha:.2f}"]
            cand[pred_col] = (1.0 - alpha) * cand[pred_col].to_numpy(dtype="float64") + alpha * val_pred

    per_target_df = pd.DataFrame(per_target).sort_values("delta_auc", ascending=False).reset_index(drop=True)
    per_target_df.to_csv(ARTIFACT_DIR / "per_target_auc.csv", index=False)

    candidate_rows = []
    y_val_full = val_df[["customer_id"] + target_cols].copy()
    for name, pred_df in blended_candidates.items():
        auc = macro_auc(y_val_full, pred_df, target_cols)
        candidate_rows.append({"candidate": name, "macro_auc": auc})

    cand_df = pd.DataFrame(candidate_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    cand_df.to_csv(ARTIFACT_DIR / "candidate_macro_auc.csv", index=False)

    summary = {
        "hardest_targets": HARDEST_TARGETS,
        "best_candidate": str(cand_df.iloc[0]["candidate"]),
        "best_macro_auc": float(cand_df.iloc[0]["macro_auc"]),
        "base_macro_auc": float(cand_df.loc[cand_df["candidate"] == "gandalf_base", "macro_auc"].iloc[0]),
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(per_target_df.to_string(index=False))
    print(cand_df.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
