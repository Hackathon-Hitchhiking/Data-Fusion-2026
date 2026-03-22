from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.lib.submission import write_submission_like_sample


ARTIFACT_DIR = ROOT / "artifacts" / "three_model_meta_stack_v1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
SUBMISSIONS_DIR = ROOT / "output" / "submissions"

N_SPLITS = 5
SEED = 42
LOGISTIC_C_VALUES = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
RIDGE_ALPHA_VALUES = [0.1, 0.5, 1.0, 4.0, 10.0]
TOP_EXTERNAL_CORR = 2
TOP_RESULTS_TO_BUILD = 2

MODEL_ORDER = ["tabm", "cat", "xgb"]


def logit(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(x / (1.0 - x))


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-z))


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(dtype=np.float64)


def macro_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> float:
    scores = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        scores.append(float(roc_auc_score(y, pred[:, idx])) if np.unique(y).size > 1 else 0.5)
    return float(np.mean(scores))


def per_target_auc(target_df: pd.DataFrame, target_cols: list[str], pred: np.ndarray) -> pd.DataFrame:
    rows = []
    for idx, target_name in enumerate(target_cols):
        y = target_df[target_name].to_numpy(dtype=np.int8)
        score = float(roc_auc_score(y, pred[:, idx])) if np.unique(y).size > 1 else 0.5
        rows.append({"target": target_name, "oof_auc": score})
    return pd.DataFrame(rows)


def load_inputs() -> tuple[pd.DataFrame, list[str], dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    target = pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet")
    target_cols = [c for c in target.columns if c != "customer_id"]
    pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]

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
    xgb_valid = pd.read_parquet(
        ROOT
        / "output"
        / "kaggle-output"
        / "xgboost-multioutput-fs-v1"
        / "pull_complete"
        / "artifacts"
        / "gpu_xgboost_multioutput_fs_v1_full_gpu"
        / "validation_predictions.parquet"
    )
    merged = target.merge(tabm_valid, on="customer_id", how="inner")
    merged = merged.merge(cat_valid, on="customer_id", how="inner", suffixes=("_tabm", "_cat"))
    merged = merged.merge(xgb_valid, on="customer_id", how="inner")

    valid_mats = {
        "tabm": merged[[f"{c}_tabm" for c in pred_cols]].to_numpy(dtype=np.float64),
        "cat": merged[[f"{c}_cat" for c in pred_cols]].to_numpy(dtype=np.float64),
        "xgb": merged[pred_cols].to_numpy(dtype=np.float64),
    }
    test_mats = {
        "tabm": pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
        "cat": pd.read_parquet(SUBMISSIONS_DIR / "catboost_multilabel_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
        "xgb": pd.read_parquet(SUBMISSIONS_DIR / "xgboost_multioutput_fs_v1_submission.parquet")[pred_cols].to_numpy(dtype=np.float64),
    }
    test_customer_ids = pd.read_parquet(SUBMISSIONS_DIR / "tabm_longrun_v3_fs_v1_submission.parquet")["customer_id"].to_numpy(dtype=np.int32)
    return merged[["customer_id"] + target_cols], target_cols, valid_mats, test_mats, test_customer_ids


def parse_family(target_name: str) -> str:
    return target_name.split("_")[1]


def build_family_map(target_cols: list[str]) -> dict[str, list[str]]:
    families: dict[str, list[str]] = {}
    for target_name in target_cols:
        families.setdefault(parse_family(target_name), []).append(target_name)
    return families


def build_related_targets(target_frame: pd.DataFrame, target_cols: list[str]) -> dict[str, list[str]]:
    corr = target_frame[target_cols].corr().abs()
    family_map = build_family_map(target_cols)
    related: dict[str, list[str]] = {}
    for target_name in target_cols:
        family_id = parse_family(target_name)
        blocked = set(family_map[family_id]) | {target_name}
        series = corr[target_name].drop(labels=list(blocked), errors="ignore").sort_values(ascending=False)
        related[target_name] = series.head(TOP_EXTERNAL_CORR).index.tolist()
    return related


def precompute_feature_primitives(target_cols: list[str], mats: dict[str, np.ndarray]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for split_name, model_mats in [("valid", mats)]:
        raise RuntimeError("unused")


def make_split_mats(valid_mats: dict[str, np.ndarray], test_mats: dict[str, np.ndarray], target_cols: list[str]) -> dict[str, dict[str, np.ndarray]]:
    by_split = {
        "valid": valid_mats,
        "test": test_mats,
    }
    computed: dict[str, dict[str, np.ndarray]] = {}
    for split_name, mats in by_split.items():
        split_dict: dict[str, np.ndarray] = {}
        for model_name, mat in mats.items():
            split_dict[f"{model_name}_prob"] = mat
            split_dict[f"{model_name}_logit"] = logit(mat)
            rank_mat = np.empty_like(mat, dtype=np.float64)
            for col_idx in range(mat.shape[1]):
                rank_mat[:, col_idx] = rank_pct(mat[:, col_idx])
            split_dict[f"{model_name}_rank"] = rank_mat

        tri_logit = 0.55 * split_dict["tabm_logit"] + 0.10 * split_dict["cat_logit"] + 0.35 * split_dict["xgb_logit"]
        split_dict["tri_logit"] = tri_logit
        split_dict["tri_prob"] = sigmoid(tri_logit)
        computed[split_name] = split_dict
    return computed


def top2_mean(arr: np.ndarray, axis: int = 1) -> np.ndarray:
    if arr.shape[axis] == 0:
        return np.zeros(arr.shape[0], dtype=np.float64)
    if arr.shape[axis] == 1:
        return np.take(arr, 0, axis=axis).astype(np.float64)
    part = np.partition(arr, kth=-2, axis=axis)
    return part[:, -2:].mean(axis=1).astype(np.float64)


def build_variant_features(
    variant: str,
    target_name: str,
    target_cols: list[str],
    family_map: dict[str, list[str]],
    related_map: dict[str, list[str]],
    split_features: dict[str, np.ndarray],
) -> pd.DataFrame:
    idx = target_cols.index(target_name)
    own = {
        "logit_tabm": split_features["tabm_logit"][:, idx],
        "logit_cat": split_features["cat_logit"][:, idx],
        "logit_xgb": split_features["xgb_logit"][:, idx],
        "rank_tabm": split_features["tabm_rank"][:, idx],
        "rank_cat": split_features["cat_rank"][:, idx],
        "rank_xgb": split_features["xgb_rank"][:, idx],
    }

    logit_stack = np.column_stack([own["logit_tabm"], own["logit_cat"], own["logit_xgb"]])
    own["mean_logit"] = logit_stack.mean(axis=1)
    own["max_logit"] = logit_stack.max(axis=1)
    own["min_logit"] = logit_stack.min(axis=1)
    own["spread_logit"] = own["max_logit"] - own["min_logit"]
    own["tabm_minus_cat"] = own["logit_tabm"] - own["logit_cat"]
    own["tabm_minus_xgb"] = own["logit_tabm"] - own["logit_xgb"]
    own["cat_minus_xgb"] = own["logit_cat"] - own["logit_xgb"]

    feature_df = pd.DataFrame(own)
    if variant == "core":
        return feature_df

    family_id = parse_family(target_name)
    family_targets = [t for t in family_map[family_id] if t != target_name]
    family_indices = [target_cols.index(t) for t in family_targets]

    if family_indices:
        for model_name in MODEL_ORDER:
            family_logits = split_features[f"{model_name}_logit"][:, family_indices]
            feature_df[f"{model_name}_family_mean"] = family_logits.mean(axis=1)
            feature_df[f"{model_name}_family_max"] = family_logits.max(axis=1)
            feature_df[f"{model_name}_family_top2"] = top2_mean(family_logits)
            feature_df[f"{model_name}_family_spread"] = family_logits.max(axis=1) - family_logits.min(axis=1)

        tri_family = split_features["tri_logit"][:, family_indices]
        feature_df["tri_family_mean"] = tri_family.mean(axis=1)
        feature_df["tri_family_max"] = tri_family.max(axis=1)
        feature_df["tri_family_top2"] = top2_mean(tri_family)
        feature_df["tri_family_spread"] = tri_family.max(axis=1) - tri_family.min(axis=1)
        family_means = np.column_stack(
            [
                feature_df["tabm_family_mean"].to_numpy(dtype=np.float64),
                feature_df["cat_family_mean"].to_numpy(dtype=np.float64),
                feature_df["xgb_family_mean"].to_numpy(dtype=np.float64),
            ]
        )
        feature_df["family_disagreement"] = family_means.std(axis=1)
    else:
        for col_name in [
            "tabm_family_mean",
            "tabm_family_max",
            "tabm_family_top2",
            "tabm_family_spread",
            "cat_family_mean",
            "cat_family_max",
            "cat_family_top2",
            "cat_family_spread",
            "xgb_family_mean",
            "xgb_family_max",
            "xgb_family_top2",
            "xgb_family_spread",
            "tri_family_mean",
            "tri_family_max",
            "tri_family_top2",
            "tri_family_spread",
            "family_disagreement",
        ]:
            feature_df[col_name] = 0.0

    for related_idx, related_target in enumerate(related_map[target_name], start=1):
        rel_target_idx = target_cols.index(related_target)
        feature_df[f"rel{related_idx}_tri_logit"] = split_features["tri_logit"][:, rel_target_idx]
        feature_df[f"rel{related_idx}_tri_prob"] = split_features["tri_prob"][:, rel_target_idx]

    return feature_df


def make_model(model_kind: str, strength: float):
    if model_kind == "logistic":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(strength),
                        class_weight="balanced",
                        max_iter=2000,
                        solver="lbfgs",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    if model_kind == "ridge":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    RidgeClassifier(
                        alpha=float(strength),
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported model kind: {model_kind}")


def predict_scores(model_kind: str, fitted_model: Pipeline, X: pd.DataFrame) -> np.ndarray:
    if model_kind == "logistic":
        return fitted_model.predict_proba(X)[:, 1].astype(np.float64)
    scores = fitted_model.decision_function(X).astype(np.float64)
    return sigmoid(scores)


def evaluate_candidate(
    *,
    variant: str,
    model_kind: str,
    strength: float,
    target_df: pd.DataFrame,
    target_cols: list[str],
    family_map: dict[str, list[str]],
    related_map: dict[str, list[str]],
    split_mats: dict[str, dict[str, np.ndarray]],
) -> tuple[np.ndarray, float, pd.DataFrame]:
    oof = np.zeros((len(target_df), len(target_cols)), dtype=np.float64)
    target_rows = []
    for idx, target_name in enumerate(target_cols):
        X = build_variant_features(variant, target_name, target_cols, family_map, related_map, split_mats["valid"])
        y = target_df[target_name].to_numpy(dtype=np.int8)
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        target_oof = np.zeros(len(X), dtype=np.float64)
        for train_idx, valid_idx in skf.split(X, y):
            model = make_model(model_kind, strength)
            model.fit(X.iloc[train_idx], y[train_idx])
            target_oof[valid_idx] = predict_scores(model_kind, model, X.iloc[valid_idx])

        oof[:, idx] = target_oof
        target_rows.append(
            {
                "target": target_name,
                "oof_auc": float(roc_auc_score(y, target_oof)) if np.unique(y).size > 1 else 0.5,
            }
        )

    score = macro_auc(target_df, target_cols, oof)
    return oof, score, pd.DataFrame(target_rows).sort_values("oof_auc").reset_index(drop=True)


def fit_full_and_predict(
    *,
    variant: str,
    model_kind: str,
    strength: float,
    target_df: pd.DataFrame,
    target_cols: list[str],
    family_map: dict[str, list[str]],
    related_map: dict[str, list[str]],
    split_mats: dict[str, dict[str, np.ndarray]],
) -> np.ndarray:
    pred = np.zeros((split_mats["test"]["tri_prob"].shape[0], len(target_cols)), dtype=np.float64)
    for idx, target_name in enumerate(target_cols):
        X_train = build_variant_features(variant, target_name, target_cols, family_map, related_map, split_mats["valid"])
        X_test = build_variant_features(variant, target_name, target_cols, family_map, related_map, split_mats["test"])
        y = target_df[target_name].to_numpy(dtype=np.int8)
        model = make_model(model_kind, strength)
        model.fit(X_train, y)
        pred[:, idx] = predict_scores(model_kind, model, X_test)
    return pred


def build_submission_name(rank: int, variant: str, model_kind: str, strength: float) -> str:
    strength_str = str(strength).replace(".", "p")
    return f"three_model_stack_{rank:02d}_{variant}_{model_kind}_{strength_str}_submission.parquet"


def main() -> None:
    target_df, target_cols, valid_mats, test_mats, test_customer_ids = load_inputs()
    split_mats = make_split_mats(valid_mats, test_mats, target_cols)
    family_map = build_family_map(target_cols)
    related_map = build_related_targets(pd.read_parquet(ROOT / "data" / "competition" / "train_target.parquet"), target_cols)

    current_best = split_mats["valid"]["tri_prob"].copy()
    current_best[:, target_cols.index("target_2_8")] = valid_mats["xgb"][:, target_cols.index("target_2_8")]
    current_best[:, target_cols.index("target_8_3")] = sigmoid(
        0.7 * logit(current_best[:, target_cols.index("target_8_3")]) + 0.3 * logit(valid_mats["xgb"][:, target_cols.index("target_8_3")])
    )
    current_best[:, target_cols.index("target_2_3")] = sigmoid(
        0.7 * logit(current_best[:, target_cols.index("target_2_3")]) + 0.3 * logit(valid_mats["xgb"][:, target_cols.index("target_2_3")])
    )
    current_best_auc = macro_auc(target_df, target_cols, current_best)

    candidate_rows = []
    oof_cache: dict[str, np.ndarray] = {}
    target_score_cache: dict[str, pd.DataFrame] = {}
    for variant in ["core", "family"]:
        for strength in LOGISTIC_C_VALUES:
            name = f"{variant}_logistic_{strength}"
            oof, score, target_scores = evaluate_candidate(
                variant=variant,
                model_kind="logistic",
                strength=strength,
                target_df=target_df,
                target_cols=target_cols,
                family_map=family_map,
                related_map=related_map,
                split_mats=split_mats,
            )
            oof_cache[name] = oof
            target_score_cache[name] = target_scores
            candidate_rows.append(
                {
                    "name": name,
                    "variant": variant,
                    "model_kind": "logistic",
                    "strength": strength,
                    "oof_macro_auc": score,
                    "delta_vs_current_best": score - current_best_auc,
                }
            )

        for strength in RIDGE_ALPHA_VALUES:
            name = f"{variant}_ridge_{strength}"
            oof, score, target_scores = evaluate_candidate(
                variant=variant,
                model_kind="ridge",
                strength=strength,
                target_df=target_df,
                target_cols=target_cols,
                family_map=family_map,
                related_map=related_map,
                split_mats=split_mats,
            )
            oof_cache[name] = oof
            target_score_cache[name] = target_scores
            candidate_rows.append(
                {
                    "name": name,
                    "variant": variant,
                    "model_kind": "ridge",
                    "strength": strength,
                    "oof_macro_auc": score,
                    "delta_vs_current_best": score - current_best_auc,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows).sort_values("oof_macro_auc", ascending=False).reset_index(drop=True)
    candidate_df.to_csv(ARTIFACT_DIR / "candidate_scores.csv", index=False)

    outputs = []
    for rank, row in enumerate(candidate_df.head(TOP_RESULTS_TO_BUILD).to_dict("records"), start=1):
        pred_test = fit_full_and_predict(
            variant=row["variant"],
            model_kind=row["model_kind"],
            strength=float(row["strength"]),
            target_df=target_df,
            target_cols=target_cols,
            family_map=family_map,
            related_map=related_map,
            split_mats=split_mats,
        )
        output_path = SUBMISSIONS_DIR / build_submission_name(rank, row["variant"], row["model_kind"], float(row["strength"]))
        pred_cols = [f"predict_{c.split('target_', 1)[1]}" for c in target_cols]
        write_submission_like_sample(
            customer_ids=test_customer_ids,
            prediction_frame=pd.DataFrame(pred_test, columns=pred_cols, dtype=np.float64),
            output_path=output_path,
        )
        target_score_path = ARTIFACT_DIR / f"{row['name']}_target_scores.csv"
        target_score_cache[row["name"]].to_csv(target_score_path, index=False)
        outputs.append(
            {
                "rank": rank,
                "candidate": row["name"],
                "variant": row["variant"],
                "model_kind": row["model_kind"],
                "strength": float(row["strength"]),
                "oof_macro_auc": float(row["oof_macro_auc"]),
                "delta_vs_current_best": float(row["delta_vs_current_best"]),
                "submission_path": str(output_path),
                "target_scores_path": str(target_score_path),
            }
        )

    summary = {
        "meta_train_protocol": "shared_75k_out_of_sample_validation_predictions_from_tabm_catboost_xgboost",
        "note": "This is a strict non-leaky holdout meta-stack, not full-750k K-fold OOF over all three backbones.",
        "current_best_reference_auc": float(current_best_auc),
        "top_candidates": outputs,
    }
    (ARTIFACT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
