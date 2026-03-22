from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "artifacts" / "post_pu_noisy_label_local_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

BASE_GLOBAL_CAT_WEIGHT = 0.3
WORST_BLEND_TARGET_COUNT = 8
TOP_CAT_ADVANTAGE_COUNT = 3
TOP_RAW_FEATURES_PER_TARGET = 8
TOP_RELATED_TARGETS = 3
CV_SPLITS = 5
SOFT_LABEL_ALPHAS = [0.10, 0.20, 0.30]
PU_CANDIDATE_RATIOS = [0.05, 0.10, 0.20, 0.30]
LOGREG_C = 0.25
MIN_CANDIDATES = 20


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def macro_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> float:
    scores: list[float] = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        s = pred[:, idx]
        scores.append(float(roc_auc_score(y, s)) if np.unique(y).size > 1 else 0.5)
    return float(np.mean(scores))


def per_target_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        s = pred[:, idx]
        score = float(roc_auc_score(y, s)) if np.unique(y).size > 1 else 0.5
        rows.append({"target": target_name, "oof_auc": score})
    return pd.DataFrame(rows)


def build_global_logit_blend(tabm: np.ndarray, cat: np.ndarray, cat_weight: float) -> np.ndarray:
    return sigmoid((1.0 - cat_weight) * logit(tabm) + cat_weight * logit(cat))


def compute_related_targets(target_matrix: pd.DataFrame, focus_targets: list[str], top_k: int = 3) -> dict[str, list[str]]:
    cols = [c for c in target_matrix.columns if c != "customer_id"]
    corr = target_matrix[cols].corr()
    related: dict[str, list[str]] = {}
    for target_name in focus_targets:
        related[target_name] = corr[target_name].drop(labels=[target_name]).abs().sort_values(ascending=False).head(top_k).index.tolist()
    return related


def load_prediction_inputs() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    target = pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet")
    tabm_valid = pd.read_parquet(
        ROOT
        / "output"
        / "kaggle-output"
        / "tabm-longrun-v3-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_tabm_longrun_v3_fs_v1_full_gpu"
        / "validation_predictions.parquet"
    )
    cat_valid = pd.read_parquet(
        ROOT
        / "output"
        / "kaggle-output"
        / "catboost-multilabel-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_catboost_multilabel_fs_v1_full_gpu"
        / "validation_predictions.parquet"
    )
    tabm_test = pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")
    cat_test = pd.read_parquet(SUBMISSIONS_DIR / "catboost_multilabel_fs_v1_submission.parquet")

    merged = target.merge(tabm_valid, on="customer_id", how="inner")
    merged = merged.merge(cat_valid, on="customer_id", how="inner", suffixes=("_tabm", "_cat"))

    target_cols = [c for c in target.columns if c != "customer_id"]
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]

    tabm_valid_mat = merged[[f"{c}_tabm" for c in pred_cols]].to_numpy(dtype=np.float64)
    cat_valid_mat = merged[[f"{c}_cat" for c in pred_cols]].to_numpy(dtype=np.float64)
    tabm_test_mat = tabm_test[pred_cols].to_numpy(dtype=np.float64)
    cat_test_mat = cat_test[pred_cols].to_numpy(dtype=np.float64)
    return merged[["customer_id"] + target_cols], target_cols, tabm_valid_mat, cat_valid_mat, tabm_test_mat, cat_test_mat


def select_focus_targets(
    target_df: pd.DataFrame,
    target_cols: list[str],
    tabm_valid: np.ndarray,
    cat_valid: np.ndarray,
) -> pd.DataFrame:
    blend_valid = build_global_logit_blend(tabm_valid, cat_valid, BASE_GLOBAL_CAT_WEIGHT)
    tabm_auc = per_target_auc(target_df, target_cols, tabm_valid).rename(columns={"oof_auc": "tabm_auc"})
    cat_auc = per_target_auc(target_df, target_cols, cat_valid).rename(columns={"oof_auc": "cat_auc"})
    blend_auc = per_target_auc(target_df, target_cols, blend_valid).rename(columns={"oof_auc": "blend_auc"})
    focus_df = tabm_auc.merge(cat_auc, on="target").merge(blend_auc, on="target")
    focus_df["cat_minus_tabm"] = focus_df["cat_auc"] - focus_df["tabm_auc"]
    worst_targets = focus_df.sort_values("blend_auc").head(WORST_BLEND_TARGET_COUNT)["target"].tolist()
    cat_advantaged = (
        focus_df[focus_df["cat_minus_tabm"] > 0]
        .sort_values("cat_minus_tabm", ascending=False)
        .head(TOP_CAT_ADVANTAGE_COUNT)["target"]
        .tolist()
    )

    ordered_focus: list[str] = []
    for target_name in worst_targets + cat_advantaged:
        if target_name not in ordered_focus:
            ordered_focus.append(target_name)

    focus_df["focus_flag"] = focus_df["target"].isin(ordered_focus).astype(int)
    focus_df["focus_rank"] = focus_df["target"].map({name: idx + 1 for idx, name in enumerate(ordered_focus)})
    focus_df.sort_values(["focus_flag", "focus_rank", "blend_auc"], ascending=[False, True, True]).to_csv(
        ARTIFACT_DIR / "focus_targets.csv",
        index=False,
    )
    return focus_df


def select_target_raw_features(focus_targets: list[str]) -> dict[str, list[str]]:
    gain_df = pd.read_csv(ROOT / "artifacts" / "feature_selection_v1" / "lgbm_gain_per_target.csv")
    final_df = pd.read_csv(ROOT / "artifacts" / "feature_selection_v1" / "final_feature_table.csv")
    selected_lookup = set(final_df.loc[final_df["selected_flag"] == 1, "feature_name"].tolist())
    target_to_features: dict[str, list[str]] = {}
    rows = []
    for target_name in focus_targets:
        target_rows = gain_df[(gain_df["target"] == target_name) & (gain_df["feature_name"].isin(selected_lookup))]
        features = target_rows.sort_values("gain_rank_within_target").head(TOP_RAW_FEATURES_PER_TARGET)["feature_name"].tolist()
        target_to_features[target_name] = features
        rows.append({"target": target_name, "selected_raw_features": json.dumps(features)})
    pd.DataFrame(rows).to_csv(ARTIFACT_DIR / "target_raw_feature_map.csv", index=False)
    return target_to_features


def load_compact_feature_frames(feature_cols: list[str], holdout_ids: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_path = ROOT / "artifacts" / "feature_selection_v1" / "train_compact_features_v1.parquet"
    test_path = ROOT / "artifacts" / "feature_selection_v1" / "test_compact_features_v1.parquet"
    columns = ["customer_id"] + list(dict.fromkeys(feature_cols))

    train_full = pd.read_parquet(train_path, columns=columns)
    train_holdout = train_full[train_full["customer_id"].isin(set(int(x) for x in holdout_ids))].copy()
    train_holdout = train_holdout.sort_values("customer_id").reset_index(drop=True)
    test_full = pd.read_parquet(test_path, columns=columns).sort_values("customer_id").reset_index(drop=True)
    return train_holdout, test_full


def build_target_model_features(
    target_name: str,
    target_cols: list[str],
    related_targets: list[str],
    raw_features: list[str],
    raw_df: pd.DataFrame,
    tabm_mat: np.ndarray,
    cat_mat: np.ndarray,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    target_to_idx = {t: i for i, t in enumerate(target_cols)}
    idx = target_to_idx[target_name]
    blend_prob = build_global_logit_blend(tabm_mat[:, [idx]], cat_mat[:, [idx]], BASE_GLOBAL_CAT_WEIGHT).ravel()
    data: dict[str, object] = {
        "tabm_logit": logit(tabm_mat[:, idx]),
        "cat_logit": logit(cat_mat[:, idx]),
        "blend_logit": logit(blend_prob),
        "tabm_prob": tabm_mat[:, idx],
        "cat_prob": cat_mat[:, idx],
        "blend_prob": blend_prob,
        "prob_gap": cat_mat[:, idx] - tabm_mat[:, idx],
    }
    for related in related_targets:
        ridx = target_to_idx[related]
        rel_blend = build_global_logit_blend(tabm_mat[:, [ridx]], cat_mat[:, [ridx]], BASE_GLOBAL_CAT_WEIGHT).ravel()
        data[f"rel_{related}_tabm_logit"] = logit(tabm_mat[:, ridx])
        data[f"rel_{related}_cat_logit"] = logit(cat_mat[:, ridx])
        data[f"rel_{related}_blend_prob"] = rel_blend

    feature_df = pd.DataFrame(data, index=raw_df.index)
    for feature_name in raw_features:
        feature_df[feature_name] = raw_df[feature_name].values

    numeric_cols = [c for c in feature_df.columns if not c.startswith("cat_feature_")]
    categorical_cols = [c for c in raw_features if c.startswith("cat_feature_")]
    return feature_df, numeric_cols, categorical_cols


def build_pipeline(numeric_cols: list[str], categorical_cols: list[str]) -> Pipeline:
    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_cols,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "clf",
                LogisticRegression(
                    C=LOGREG_C,
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight="balanced",
                ),
            ),
        ]
    )


def candidate_indices(
    y_train: np.ndarray,
    score_train: np.ndarray,
    ratio: float,
) -> np.ndarray:
    neg_idx = np.flatnonzero(y_train == 0)
    pos_count = int((y_train == 1).sum())
    if pos_count == 0 or neg_idx.size == 0:
        return np.array([], dtype=np.int64)
    count = min(neg_idx.size, max(MIN_CANDIDATES, int(round(pos_count * ratio))))
    order = np.argsort(score_train[neg_idx])[::-1]
    return neg_idx[order[:count]]


def build_soft_training_frame(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    candidate_idx: np.ndarray,
    alpha: float,
    mode: str,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    candidate_set = set(int(i) for i in candidate_idx.tolist())
    clean_rows = [i for i in range(len(X_train)) if i not in candidate_set]

    X_clean = X_train.iloc[clean_rows].reset_index(drop=True)
    y_clean = y_train[clean_rows].astype(np.int8)

    if mode == "drop" or candidate_idx.size == 0:
        weights = np.ones(len(X_clean), dtype=np.float64)
        return X_clean, y_clean, weights

    X_candidate = X_train.iloc[candidate_idx].reset_index(drop=True)
    y_candidate_neg = np.zeros(len(X_candidate), dtype=np.int8)
    y_candidate_pos = np.ones(len(X_candidate), dtype=np.int8)

    X_final = pd.concat([X_clean, X_candidate, X_candidate], axis=0, ignore_index=True)
    y_final = np.concatenate([y_clean, y_candidate_neg, y_candidate_pos])
    w_final = np.concatenate(
        [
            np.ones(len(X_clean), dtype=np.float64),
            np.full(len(X_candidate), 1.0 - alpha, dtype=np.float64),
            np.full(len(X_candidate), alpha, dtype=np.float64),
        ]
    )
    return X_final, y_final, w_final


def compute_group_profile(
    feature_df: pd.DataFrame,
    target_name: str,
    related_targets: list[str],
    target_df: pd.DataFrame,
    base_prob: np.ndarray,
    ratio: float,
) -> dict[str, object]:
    y = target_df[target_name].to_numpy(dtype=np.int8)
    neg_mask = y == 0
    neg_idx = np.flatnonzero(neg_mask)
    candidate_idx = candidate_indices(y, base_prob, ratio)
    candidate_mask = np.zeros(len(y), dtype=bool)
    candidate_mask[candidate_idx] = True
    clean_neg_mask = neg_mask & ~candidate_mask
    pos_mask = y == 1

    raw_cols = [c for c in feature_df.columns if c.startswith("num_feature_") or c.startswith("cat_feature_")]
    numeric_raw_cols = [c for c in raw_cols if c.startswith("num_feature_")]

    def centroid_distance(mask_a: np.ndarray, mask_b: np.ndarray) -> float | None:
        if not numeric_raw_cols or mask_a.sum() == 0 or mask_b.sum() == 0:
            return None
        a = feature_df.loc[mask_a, numeric_raw_cols]
        b = feature_df.loc[mask_b, numeric_raw_cols]
        pooled_std = feature_df[numeric_raw_cols].std(ddof=0).replace(0.0, np.nan)
        diff = (a.mean() - b.mean()).abs() / pooled_std
        return float(diff.fillna(0.0).mean())

    related_profile = {}
    for related in related_targets:
        related_y = target_df[related].to_numpy(dtype=np.int8)
        related_profile[related] = {
            "positive_rate_among_positives": float(related_y[pos_mask].mean()) if pos_mask.sum() else None,
            "positive_rate_among_candidates": float(related_y[candidate_mask].mean()) if candidate_mask.sum() else None,
            "positive_rate_among_clean_negatives": float(related_y[clean_neg_mask].mean()) if clean_neg_mask.sum() else None,
        }

    return {
        "target": target_name,
        "positive_count": int(pos_mask.sum()),
        "candidate_negative_count": int(candidate_mask.sum()),
        "clean_negative_count": int(clean_neg_mask.sum()),
        "candidate_ratio_used": ratio,
        "positive_blend_prob_mean": float(base_prob[pos_mask].mean()) if pos_mask.sum() else None,
        "candidate_blend_prob_mean": float(base_prob[candidate_mask].mean()) if candidate_mask.sum() else None,
        "clean_negative_blend_prob_mean": float(base_prob[clean_neg_mask].mean()) if clean_neg_mask.sum() else None,
        "candidate_to_positive_raw_distance": centroid_distance(candidate_mask, pos_mask),
        "clean_negative_to_positive_raw_distance": centroid_distance(clean_neg_mask, pos_mask),
        "related_target_profile": json.dumps(related_profile),
    }


def evaluate_target_pu(
    *,
    target_name: str,
    target_df: pd.DataFrame,
    target_cols: list[str],
    feature_valid: pd.DataFrame,
    feature_test: pd.DataFrame,
    tabm_valid: np.ndarray,
    cat_valid: np.ndarray,
    tabm_test: np.ndarray,
    cat_test: np.ndarray,
    related_targets: list[str],
) -> dict[str, object]:
    idx = {t: i for i, t in enumerate(target_cols)}[target_name]
    y = target_df[target_name].to_numpy(dtype=np.int8)
    base_valid = build_global_logit_blend(tabm_valid[:, [idx]], cat_valid[:, [idx]], BASE_GLOBAL_CAT_WEIGHT).ravel()
    base_test = build_global_logit_blend(tabm_test[:, [idx]], cat_test[:, [idx]], BASE_GLOBAL_CAT_WEIGHT).ravel()
    base_auc = float(roc_auc_score(y, base_valid))

    raw_cols = [c for c in feature_valid.columns if c.startswith("num_feature_") or c.startswith("cat_feature_")]
    numeric_cols = [c for c in feature_valid.columns if c not in [c for c in raw_cols if c.startswith("cat_feature_")]]
    categorical_cols = [c for c in raw_cols if c.startswith("cat_feature_")]

    grid_rows = []
    best = None
    best_oof = None
    cv = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    for mode in ["drop", "soft"]:
        for ratio in PU_CANDIDATE_RATIOS:
            alpha_values = [0.0] if mode == "drop" else SOFT_LABEL_ALPHAS
            for alpha in alpha_values:
                oof = np.zeros(len(feature_valid), dtype=np.float64)
                fold_candidate_counts: list[int] = []
                for tr_idx, va_idx in cv.split(feature_valid, y):
                    X_tr = feature_valid.iloc[tr_idx].reset_index(drop=True)
                    X_va = feature_valid.iloc[va_idx].reset_index(drop=True)
                    y_tr = y[tr_idx]
                    y_va = y[va_idx]
                    base_tr = base_valid[tr_idx]

                    cand_idx = candidate_indices(y_tr, base_tr, ratio)
                    fold_candidate_counts.append(int(cand_idx.size))
                    fit_X, fit_y, fit_w = build_soft_training_frame(X_tr, y_tr, cand_idx, alpha=alpha, mode=mode)
                    model = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
                    model.fit(fit_X, fit_y, clf__sample_weight=fit_w)
                    oof[va_idx] = model.predict_proba(X_va)[:, 1]

                cv_auc = float(roc_auc_score(y, oof))
                row = {
                    "target": target_name,
                    "mode": mode,
                    "candidate_ratio": ratio,
                    "soft_alpha": alpha,
                    "cv_auc": cv_auc,
                    "delta_vs_base": cv_auc - base_auc,
                    "mean_candidate_count": float(np.mean(fold_candidate_counts)),
                }
                grid_rows.append(row)
                if best is None or cv_auc > best["cv_auc"]:
                    best = row
                    best_oof = oof.copy()

    assert best is not None and best_oof is not None
    pd.DataFrame(grid_rows).to_csv(ARTIFACT_DIR / f"grid_{target_name}.csv", index=False)

    full_candidate_idx = candidate_indices(y, base_valid, float(best["candidate_ratio"]))
    fit_X, fit_y, fit_w = build_soft_training_frame(
        feature_valid.reset_index(drop=True),
        y,
        full_candidate_idx,
        alpha=float(best["soft_alpha"]),
        mode=str(best["mode"]),
    )
    final_model = build_pipeline(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
    final_model.fit(fit_X, fit_y, clf__sample_weight=fit_w)
    test_pred = final_model.predict_proba(feature_test)[:, 1]

    profile = compute_group_profile(
        feature_df=feature_valid,
        target_name=target_name,
        related_targets=related_targets,
        target_df=target_df,
        base_prob=base_valid,
        ratio=float(best["candidate_ratio"]),
    )
    return {
        "target": target_name,
        "base_auc": base_auc,
        "best_cv_auc": float(best["cv_auc"]),
        "delta_vs_base": float(best["cv_auc"] - base_auc),
        "best_mode": str(best["mode"]),
        "best_candidate_ratio": float(best["candidate_ratio"]),
        "best_soft_alpha": float(best["soft_alpha"]),
        "mean_candidate_count": float(best["mean_candidate_count"]),
        "candidate_holdout_pred": best_oof,
        "test_pred": test_pred,
        "profile": profile,
        "grid_rows": grid_rows,
    }


def main() -> None:
    target_df, target_cols, tabm_valid, cat_valid, tabm_test, cat_test = load_prediction_inputs()
    blend_valid = build_global_logit_blend(tabm_valid, cat_valid, BASE_GLOBAL_CAT_WEIGHT)
    blend_test = build_global_logit_blend(tabm_test, cat_test, BASE_GLOBAL_CAT_WEIGHT)
    base_macro = macro_auc(target_df, target_cols, blend_valid)

    focus_overview = select_focus_targets(target_df, target_cols, tabm_valid, cat_valid)
    focus_targets = focus_overview.loc[focus_overview["focus_flag"] == 1].sort_values("focus_rank")["target"].tolist()
    raw_feature_map = select_target_raw_features(focus_targets)
    related_map = compute_related_targets(pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet"), focus_targets, top_k=TOP_RELATED_TARGETS)

    union_raw_features = []
    for target_name in focus_targets:
        union_raw_features.extend(raw_feature_map[target_name])
    union_raw_features = list(dict.fromkeys(union_raw_features))
    holdout_raw, test_raw = load_compact_feature_frames(union_raw_features, target_df["customer_id"].to_numpy())

    if not np.array_equal(holdout_raw["customer_id"].to_numpy(), target_df["customer_id"].to_numpy()):
        raise RuntimeError("Holdout feature rows are not aligned with validation predictions.")

    target_results = []
    profile_rows = []
    all_grid_rows: list[dict[str, object]] = []
    target_pred_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for target_name in focus_targets:
        raw_features = raw_feature_map[target_name]
        feature_valid, _, _ = build_target_model_features(
            target_name,
            target_cols,
            related_map[target_name],
            raw_features,
            holdout_raw,
            tabm_valid,
            cat_valid,
        )
        feature_test, _, _ = build_target_model_features(
            target_name,
            target_cols,
            related_map[target_name],
            raw_features,
            test_raw,
            tabm_test,
            cat_test,
        )
        result = evaluate_target_pu(
            target_name=target_name,
            target_df=target_df,
            target_cols=target_cols,
            feature_valid=feature_valid,
            feature_test=feature_test,
            tabm_valid=tabm_valid,
            cat_valid=cat_valid,
            tabm_test=tabm_test,
            cat_test=cat_test,
            related_targets=related_map[target_name],
        )
        target_results.append(
            {
                "target": result["target"],
                "base_auc": result["base_auc"],
                "best_cv_auc": result["best_cv_auc"],
                "delta_vs_base": result["delta_vs_base"],
                "best_mode": result["best_mode"],
                "best_candidate_ratio": result["best_candidate_ratio"],
                "best_soft_alpha": result["best_soft_alpha"],
                "mean_candidate_count": result["mean_candidate_count"],
                "raw_features": json.dumps(raw_features),
                "related_targets": json.dumps(related_map[target_name]),
            }
        )
        profile_rows.append(result["profile"])
        all_grid_rows.extend(result["grid_rows"])
        target_pred_map[target_name] = (result["candidate_holdout_pred"], result["test_pred"])

    pd.DataFrame(all_grid_rows).to_csv(ARTIFACT_DIR / "pu_config_grid.csv", index=False)

    candidate_valid = blend_valid.copy()
    candidate_test = blend_test.copy()
    current_macro = base_macro
    selected_targets = []
    for row in sorted(target_results, key=lambda x: x["delta_vs_base"], reverse=True):
        target_name = str(row["target"])
        target_idx = {t: i for i, t in enumerate(target_cols)}[target_name]
        holdout_pred, test_pred = target_pred_map[target_name]
        base_pred = candidate_valid[:, target_idx].copy()
        candidate_valid[:, target_idx] = holdout_pred
        candidate_test[:, target_idx] = test_pred
        macro_candidate = macro_auc(target_df, target_cols, candidate_valid)
        macro_delta = macro_candidate - current_macro
        row["macro_auc_candidate"] = macro_candidate
        row["macro_delta_vs_current"] = macro_delta
        if macro_delta > 0:
            current_macro = macro_candidate
            selected_targets.append(
                {
                    "target": target_name,
                    "best_mode": row["best_mode"],
                    "best_candidate_ratio": row["best_candidate_ratio"],
                    "best_soft_alpha": row["best_soft_alpha"],
                    "delta_vs_base_target": row["delta_vs_base"],
                    "macro_auc_after_apply": macro_candidate,
                    "macro_delta_vs_previous": macro_delta,
                }
            )
        else:
            candidate_valid[:, target_idx] = base_pred
            candidate_test[:, target_idx] = blend_test[:, target_idx]

    target_result_df = pd.DataFrame(target_results).sort_values("delta_vs_base", ascending=False).reset_index(drop=True)
    target_result_df.to_csv(ARTIFACT_DIR / "pu_target_results.csv", index=False)
    pd.DataFrame(profile_rows).to_csv(ARTIFACT_DIR / "pu_candidate_profiles.csv", index=False)
    pd.DataFrame(selected_targets).to_csv(ARTIFACT_DIR / "pu_selected_targets.csv", index=False)

    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    submit_df = pd.DataFrame(candidate_test, columns=pred_cols, dtype=np.float64)
    submit_df.insert(0, "customer_id", pd.read_parquet(SUBMISSIONS_DIR / "tabm_fs_v1_catboost_multilabel_logit30_submission.parquet")["customer_id"].to_numpy())
    submit_path = SUBMISSIONS_DIR / "tabm_catboost_puweak_v1_submission.parquet"
    submit_df.to_parquet(submit_path, index=False)

    summary = {
        "base_global_blend_macro_auc": float(base_macro),
        "best_pu_macro_auc": float(current_macro),
        "delta_vs_base_global_blend": float(current_macro - base_macro),
        "focus_targets": focus_targets,
        "selected_targets": selected_targets,
        "submission_path": str(submit_path),
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
