from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path.cwd()
ARTIFACT_DIR = ROOT / "artifacts" / "post_tabm_catboost_weak_local_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

BASE_GLOBAL_CAT_WEIGHT = 0.3
FIRST_WAVE_TARGETS = ["target_2_8", "target_2_3", "target_8_3"]
EXTENDED_TARGETS = ["target_10_1", "target_9_7", "target_3_1"]
ALL_CANDIDATE_TARGETS = FIRST_WAVE_TARGETS + EXTENDED_TARGETS


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def macro_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> float:
    scores: list[float] = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy()
        s = pred[:, idx]
        scores.append(float(roc_auc_score(y, s)) if np.unique(y).size > 1 else 0.5)
    return float(np.mean(scores))


def per_target_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy()
        s = pred[:, idx]
        score = float(roc_auc_score(y, s)) if np.unique(y).size > 1 else 0.5
        rows.append({"target": target_name, "oof_auc": score})
    return pd.DataFrame(rows)


def to_rank_matrix(mat: np.ndarray) -> np.ndarray:
    ranks = np.empty_like(mat, dtype=np.float64)
    for j in range(mat.shape[1]):
        ranks[:, j] = pd.Series(mat[:, j]).rank(method="average").to_numpy(dtype=np.float64)
    return ranks


def load_inputs() -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def build_global_logit_blend(tabm: np.ndarray, cat: np.ndarray, cat_weight: float) -> np.ndarray:
    return sigmoid((1.0 - cat_weight) * logit(tabm) + cat_weight * logit(cat))


def shrink_weight(weight: float) -> float:
    return round(max(0.0, weight - 0.10), 2)


def run_soft_blend(
    target_df: pd.DataFrame,
    target_cols: list[str],
    tabm_valid: np.ndarray,
    cat_valid: np.ndarray,
    tabm_test: np.ndarray,
    cat_test: np.ndarray,
) -> dict[str, object]:
    base_valid = build_global_logit_blend(tabm_valid, cat_valid, BASE_GLOBAL_CAT_WEIGHT)
    base_test = build_global_logit_blend(tabm_test, cat_test, BASE_GLOBAL_CAT_WEIGHT)
    base_macro = macro_auc(target_df, target_cols, base_valid)

    target_to_idx = {t: i for i, t in enumerate(target_cols)}
    search_rows: list[dict[str, object]] = []
    for target_name in ALL_CANDIDATE_TARGETS:
        idx = target_to_idx[target_name]
        y = target_df[target_name].to_numpy()
        base_auc = float(roc_auc_score(y, base_valid[:, idx]))
        best_row = None
        for blend_kind in ["prob", "logit"]:
            for w in np.round(np.linspace(0.0, 0.40, 41), 2):
                if blend_kind == "prob":
                    pred = (1.0 - w) * tabm_valid[:, idx] + w * cat_valid[:, idx]
                else:
                    pred = sigmoid((1.0 - w) * logit(tabm_valid[:, idx]) + w * logit(cat_valid[:, idx]))
                auc = float(roc_auc_score(y, pred))
                row = {
                    "target": target_name,
                    "blend": blend_kind,
                    "cat_weight": float(w),
                    "target_auc": auc,
                    "delta_vs_base_global": auc - base_auc,
                }
                search_rows.append(row)
                if best_row is None or auc > best_row["target_auc"]:
                    best_row = row
        assert best_row is not None

    search_df = pd.DataFrame(search_rows)
    search_df.to_csv(ARTIFACT_DIR / "soft_blend_weight_grid.csv", index=False)

    best_per_target = (
        search_df.sort_values(["target", "target_auc"], ascending=[True, False])
        .groupby("target", as_index=False)
        .first()
        .copy()
    )
    best_per_target["shrunken_cat_weight"] = best_per_target["cat_weight"].map(shrink_weight)
    best_per_target.to_csv(ARTIFACT_DIR / "soft_blend_best_per_target.csv", index=False)

    current_valid = base_valid.copy()
    current_test = base_test.copy()
    current_macro = base_macro
    greedy_rows: list[dict[str, object]] = []
    selected_targets: list[dict[str, object]] = []

    candidate_rows = best_per_target.sort_values("delta_vs_base_global", ascending=False)
    for row in candidate_rows.to_dict("records"):
        if row["delta_vs_base_global"] <= 0:
            continue
        target_name = str(row["target"])
        idx = target_to_idx[target_name]
        blend_kind = str(row["blend"])
        shrunken_w = float(row["shrunken_cat_weight"])
        if shrunken_w <= 0:
            continue

        cand_valid = current_valid.copy()
        cand_test = current_test.copy()
        if blend_kind == "prob":
            cand_valid[:, idx] = (1.0 - shrunken_w) * tabm_valid[:, idx] + shrunken_w * cat_valid[:, idx]
            cand_test[:, idx] = (1.0 - shrunken_w) * tabm_test[:, idx] + shrunken_w * cat_test[:, idx]
        else:
            cand_valid[:, idx] = sigmoid((1.0 - shrunken_w) * logit(tabm_valid[:, idx]) + shrunken_w * logit(cat_valid[:, idx]))
            cand_test[:, idx] = sigmoid((1.0 - shrunken_w) * logit(tabm_test[:, idx]) + shrunken_w * logit(cat_test[:, idx]))

        cand_macro = macro_auc(target_df, target_cols, cand_valid)
        greedy_rows.append(
            {
                "target": target_name,
                "blend": blend_kind,
                "best_cat_weight_local": float(row["cat_weight"]),
                "cat_weight_applied": shrunken_w,
                "macro_auc_candidate": cand_macro,
                "delta_vs_current": cand_macro - current_macro,
            }
        )
        if cand_macro > current_macro:
            current_valid = cand_valid
            current_test = cand_test
            current_macro = cand_macro
            selected_targets.append(
                {
                    "target": target_name,
                    "blend": blend_kind,
                    "best_cat_weight_local": float(row["cat_weight"]),
                    "cat_weight_applied": shrunken_w,
                }
            )

    greedy_df = pd.DataFrame(greedy_rows)
    greedy_df.to_csv(ARTIFACT_DIR / "soft_blend_greedy.csv", index=False)

    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    candidate_submission = pd.DataFrame(current_test, columns=pred_cols, dtype=np.float64)
    candidate_submission.insert(0, "customer_id", pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")["customer_id"].to_numpy())
    candidate_path = SUBMISSIONS_DIR / "tabm_catboost_targetwise_softblend_v1_submission.parquet"
    candidate_submission.to_parquet(candidate_path, index=False)

    summary = {
        "base_global_logit30_macro_auc": float(base_macro),
        "best_softblend_macro_auc": float(current_macro),
        "delta_vs_base_global": float(current_macro - base_macro),
        "selected_targets": selected_targets,
        "submission_path": str(candidate_path),
    }
    (ARTIFACT_DIR / "soft_blend_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return {
        "base_valid": base_valid,
        "base_test": base_test,
        "soft_valid": current_valid,
        "soft_test": current_test,
        "summary": summary,
    }


def compute_related_targets(target_matrix: pd.DataFrame, focus_targets: list[str], top_k: int = 4) -> dict[str, list[str]]:
    corr = target_matrix[focus_targets + [c for c in target_matrix.columns if c not in ["customer_id"] + focus_targets]].drop(columns=["customer_id"], errors="ignore").corr()
    related: dict[str, list[str]] = {}
    for target_name in focus_targets:
        series = corr[target_name].drop(labels=[target_name]).abs().sort_values(ascending=False)
        related[target_name] = series.head(top_k).index.tolist()
    return related


def build_meta_features(
    target_cols: list[str],
    target_name: str,
    related_targets: list[str],
    tabm_mat: np.ndarray,
    cat_mat: np.ndarray,
) -> pd.DataFrame:
    target_to_idx = {t: i for i, t in enumerate(target_cols)}
    idx = target_to_idx[target_name]
    tabm_rank = pd.Series(tabm_mat[:, idx]).rank(method="average").to_numpy(dtype=np.float64)
    cat_rank = pd.Series(cat_mat[:, idx]).rank(method="average").to_numpy(dtype=np.float64)
    data = {
        "tabm_logit": logit(tabm_mat[:, idx]),
        "cat_logit": logit(cat_mat[:, idx]),
        "tabm_prob": tabm_mat[:, idx],
        "cat_prob": cat_mat[:, idx],
        "prob_avg": 0.5 * (tabm_mat[:, idx] + cat_mat[:, idx]),
        "prob_diff": cat_mat[:, idx] - tabm_mat[:, idx],
        "tabm_rank": tabm_rank,
        "cat_rank": cat_rank,
    }
    for related in related_targets:
        ridx = target_to_idx[related]
        data[f"rel_{related}_tabm_logit"] = logit(tabm_mat[:, ridx])
        data[f"rel_{related}_cat_logit"] = logit(cat_mat[:, ridx])
    return pd.DataFrame(data)


def run_meta_layer(
    target_df: pd.DataFrame,
    target_cols: list[str],
    base_valid: np.ndarray,
    base_test: np.ndarray,
    tabm_valid: np.ndarray,
    cat_valid: np.ndarray,
    tabm_test: np.ndarray,
    cat_test: np.ndarray,
) -> dict[str, object]:
    focus_targets = ALL_CANDIDATE_TARGETS
    target_raw = pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet")
    related_map = compute_related_targets(target_raw, focus_targets, top_k=4)
    target_to_idx = {t: i for i, t in enumerate(target_cols)}

    current_valid = base_valid.copy()
    current_test = base_test.copy()
    current_macro = macro_auc(target_df, target_cols, current_valid)

    model_rows: list[dict[str, object]] = []
    selected_targets: list[dict[str, object]] = []

    for target_name in focus_targets:
        idx = target_to_idx[target_name]
        features = build_meta_features(target_cols, target_name, related_map[target_name], tabm_valid, cat_valid)
        test_features = build_meta_features(target_cols, target_name, related_map[target_name], tabm_test, cat_test)
        y = target_df[target_name].to_numpy()

        best_c = None
        best_auc = -np.inf
        best_oof = None

        for c_value in [0.25, 0.5, 1.0, 2.0]:
            oof = np.zeros(len(features), dtype=np.float64)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            for tr_idx, va_idx in cv.split(features, y):
                pipe = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "clf",
                            LogisticRegression(
                                C=float(c_value),
                                max_iter=2000,
                                class_weight="balanced",
                                solver="lbfgs",
                            ),
                        ),
                    ]
                )
                pipe.fit(features.iloc[tr_idx], y[tr_idx])
                oof[va_idx] = pipe.predict_proba(features.iloc[va_idx])[:, 1]
            auc = float(roc_auc_score(y, oof))
            if auc > best_auc:
                best_auc = auc
                best_c = float(c_value)
                best_oof = oof.copy()

        assert best_c is not None and best_oof is not None
        base_auc = float(roc_auc_score(y, current_valid[:, idx]))
        model_rows.append(
            {
                "target": target_name,
                "best_c": best_c,
                "meta_oof_auc": best_auc,
                "base_auc": base_auc,
                "delta_vs_base": best_auc - base_auc,
                "related_targets": json.dumps(related_map[target_name]),
            }
        )

    model_df = pd.DataFrame(model_rows).sort_values("delta_vs_base", ascending=False).reset_index(drop=True)
    model_df.to_csv(ARTIFACT_DIR / "weak_meta_grid.csv", index=False)

    for row in model_df.to_dict("records"):
        if row["delta_vs_base"] <= 0:
            continue
        target_name = str(row["target"])
        idx = target_to_idx[target_name]
        related_targets = json.loads(row["related_targets"])
        features = build_meta_features(target_cols, target_name, related_targets, tabm_valid, cat_valid)
        test_features = build_meta_features(target_cols, target_name, related_targets, tabm_test, cat_test)
        y = target_df[target_name].to_numpy()

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        oof = np.zeros(len(features), dtype=np.float64)
        for tr_idx, va_idx in cv.split(features, y):
            pipe = Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            C=float(row["best_c"]),
                            max_iter=2000,
                            class_weight="balanced",
                            solver="lbfgs",
                        ),
                    ),
                ]
            )
            pipe.fit(features.iloc[tr_idx], y[tr_idx])
            oof[va_idx] = pipe.predict_proba(features.iloc[va_idx])[:, 1]

        final_pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=float(row["best_c"]),
                        max_iter=2000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        final_pipe.fit(features, y)
        test_pred = final_pipe.predict_proba(test_features)[:, 1]

        cand_valid = current_valid.copy()
        cand_test = current_test.copy()
        cand_valid[:, idx] = oof
        cand_test[:, idx] = test_pred
        cand_macro = macro_auc(target_df, target_cols, cand_valid)
        row["macro_auc_candidate"] = cand_macro
        row["delta_macro_vs_current"] = cand_macro - current_macro

        if cand_macro > current_macro:
            current_valid = cand_valid
            current_test = cand_test
            current_macro = cand_macro
            selected_targets.append(
                {
                    "target": target_name,
                    "best_c": float(row["best_c"]),
                    "meta_oof_auc": float(row["meta_oof_auc"]),
                    "delta_vs_base": float(row["delta_vs_base"]),
                }
            )

    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
    candidate_submission = pd.DataFrame(current_test, columns=pred_cols, dtype=np.float64)
    candidate_submission.insert(0, "customer_id", pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")["customer_id"].to_numpy())
    candidate_path = SUBMISSIONS_DIR / "tabm_catboost_weakmeta_v1_submission.parquet"
    candidate_submission.to_parquet(candidate_path, index=False)

    summary = {
        "base_macro_auc": float(macro_auc(target_df, target_cols, base_valid)),
        "best_meta_macro_auc": float(current_macro),
        "delta_vs_base": float(current_macro - macro_auc(target_df, target_cols, base_valid)),
        "selected_targets": selected_targets,
        "submission_path": str(candidate_path),
    }
    (ARTIFACT_DIR / "weak_meta_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    target_df, target_cols, tabm_valid, cat_valid, tabm_test, cat_test = load_inputs()
    soft_result = run_soft_blend(target_df, target_cols, tabm_valid, cat_valid, tabm_test, cat_test)
    meta_summary = run_meta_layer(
        target_df,
        target_cols,
        soft_result["base_valid"],
        soft_result["base_test"],
        tabm_valid,
        cat_valid,
        tabm_test,
        cat_test,
    )
    summary = {
        "soft_blend": soft_result["summary"],
        "meta_layer": meta_summary,
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
