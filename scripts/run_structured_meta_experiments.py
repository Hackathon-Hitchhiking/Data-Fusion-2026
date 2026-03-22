from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lib.layout import resolve_competition_file
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run cheap structured meta experiments on aligned holdout predictions.")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument("--gandalf-val", type=Path, default=Path("artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet"))
    p.add_argument("--stack-oof", type=Path, default=Path("artifacts/stack_top100_self_c005_full_750k/oof_predictions.parquet"))
    p.add_argument("--base-oof", type=Path, default=Path("artifacts/full_lgbm_top100_750k_2fold/oof_predictions.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/structured_meta_experiments"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-splits", type=int, default=5)
    return p.parse_args()


def build_group_map(target_cols: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for target_name in target_cols:
        _, group_id, _ = target_name.split("_")
        groups.setdefault(group_id, []).append(target_name)
    return groups


def build_base_frame(g: pd.DataFrame, s: pd.DataFrame, b: pd.DataFrame, pred_cols: list[str]) -> pd.DataFrame:
    df = pd.DataFrame({"customer_id": g["customer_id"].values})
    for pred_col in pred_cols:
        df[f"g__{pred_col}"] = g[pred_col].astype("float32").values
        df[f"s__{pred_col}"] = s[pred_col].astype("float32").values
        df[f"b__{pred_col}"] = b[pred_col].astype("float32").values
    return df


def stratify_labels(target_df: pd.DataFrame) -> np.ndarray:
    return target_df.sum(axis=1).clip(upper=4).astype("int8").to_numpy()


def build_feature_set(frame: pd.DataFrame, target_name: str, groups: dict[str, list[str]], kind: str) -> list[str]:
    pred_col = target_name.replace("target_", "predict_")
    _, group_id, _ = target_name.split("_")
    same_group = [t.replace("target_", "predict_") for t in groups[group_id] if t != target_name]

    local = [f"g__{pred_col}", f"s__{pred_col}", f"b__{pred_col}"]
    if kind == "local3":
        return local

    if kind == "group":
        cols = local.copy()
        for other_pred in same_group:
            cols.extend([f"g__{other_pred}", f"s__{other_pred}", f"b__{other_pred}"])
        return cols

    if kind == "global_gs":
        cols = [c for c in frame.columns if c.startswith("g__") or c.startswith("s__")]
        cols.append(f"b__{pred_col}")
        return cols

    raise ValueError(f"Unknown feature kind: {kind}")


def run_logistic_meta(
    X_all: pd.DataFrame,
    y_df: pd.DataFrame,
    target_cols: list[str],
    groups: dict[str, list[str]],
    *,
    feature_kind: str,
    n_splits: int,
    seed: int,
    c_value: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = pd.DataFrame({"customer_id": X_all["customer_id"].values})
    target_rows: list[dict[str, float | str]] = []
    stratify = stratify_labels(y_df[target_cols])

    for target_name in target_cols:
        pred_col = target_name.replace("target_", "predict_")
        feature_cols = build_feature_set(X_all, target_name, groups, feature_kind)
        x_target = X_all[feature_cols].to_numpy(dtype=np.float32)
        y_target = y_df[target_name].to_numpy(dtype=np.int8)
        pred = np.zeros(len(X_all), dtype=np.float64)

        for train_idx, val_idx in skf.split(x_target, stratify):
            x_train = x_target[train_idx]
            y_train = y_target[train_idx]
            x_val = x_target[val_idx]
            if np.unique(y_train).size < 2:
                pred[val_idx] = float(y_train.mean())
                continue
            model = LogisticRegression(
                C=c_value,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(x_train, y_train)
            pred[val_idx] = model.predict_proba(x_val)[:, 1]

        oof[pred_col] = pred
        score = roc_auc_score(y_target, pred) if np.unique(y_target).size > 1 else 0.5
        target_rows.append({"target": target_name, "oof_auc": float(score), "feature_kind": feature_kind})

    return oof, pd.DataFrame(target_rows).sort_values("oof_auc").reset_index(drop=True)


def build_label_graph_weights(target_df: pd.DataFrame, target_cols: list[str], top_k: int = 3) -> np.ndarray:
    y = target_df[target_cols].to_numpy(dtype=np.float64)
    base_rate = y.mean(axis=0)
    eps = 1e-6
    joint = (y.T @ y) / len(y)
    lift = joint / np.maximum(np.outer(base_rate, base_rate), eps)
    np.fill_diagonal(lift, 1.0)
    weights = np.log(np.maximum(lift, 1.0))
    np.fill_diagonal(weights, 0.0)
    out = np.zeros_like(weights)
    for i in range(weights.shape[0]):
        idx = np.argsort(weights[i])[::-1][:top_k]
        out[i, idx] = weights[i, idx]
    return out


def graph_smooth(
    base_pred: pd.DataFrame,
    target_df: pd.DataFrame,
    target_cols: list[str],
    weights: np.ndarray,
    *,
    alpha: float,
) -> pd.DataFrame:
    pred_cols = [t.replace("target_", "predict_") for t in target_cols]
    p = base_pred[pred_cols].to_numpy(dtype=np.float64)
    priors = target_df[target_cols].mean(axis=0).to_numpy(dtype=np.float64)
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    neighbor_effect = (p - priors) @ weights.T
    smoothed = 1.0 / (1.0 + np.exp(-(logits + alpha * neighbor_effect)))
    out = pd.DataFrame({"customer_id": base_pred["customer_id"].values})
    for i, pred_col in enumerate(pred_cols):
        out[pred_col] = smoothed[:, i]
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_parquet(args.target_file)
    gandalf = normalize_prediction_columns(pd.read_parquet(args.gandalf_val))
    stack = pd.read_parquet(args.stack_oof)
    base = pd.read_parquet(args.base_oof)

    pred_cols = [c for c in gandalf.columns if c != "customer_id"]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]
    groups = build_group_map(target_cols)

    y_holdout = (
        target_df.merge(gandalf[["customer_id"]], on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    gandalf = gandalf.sort_values("customer_id").reset_index(drop=True)
    stack = stack.merge(gandalf[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    base = base.merge(gandalf[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)

    base_frame = build_base_frame(gandalf, stack, base, pred_cols)
    baseline_frames = {
        "gandalf": gandalf,
        "g_90_s_10": pd.concat(
            [
                gandalf[["customer_id"]],
                0.9 * gandalf[pred_cols].reset_index(drop=True) + 0.1 * stack[pred_cols].reset_index(drop=True),
            ],
            axis=1,
        ),
    }

    summary_rows: list[dict[str, float | str]] = []
    target_tables: dict[str, pd.DataFrame] = {}

    for name, frame in baseline_frames.items():
        summary_rows.append({"experiment": name, "macro_auc": macro_auc(y_holdout, frame, target_cols), "kind": "baseline"})

    for feature_kind, c_value in [("local3", 0.5), ("group", 0.25), ("global_gs", 0.1)]:
        oof, target_scores = run_logistic_meta(
            base_frame,
            y_holdout,
            target_cols,
            groups,
            feature_kind=feature_kind,
            n_splits=args.n_splits,
            seed=args.seed,
            c_value=c_value,
        )
        score = macro_auc(y_holdout, oof, target_cols)
        exp_name = f"logit_meta_{feature_kind}"
        summary_rows.append({"experiment": exp_name, "macro_auc": score, "kind": "meta"})
        target_tables[exp_name] = target_scores
        oof.to_parquet(args.out_dir / f"{exp_name}_oof.parquet", index=False)
        target_scores.to_csv(args.out_dir / f"{exp_name}_target_scores.csv", index=False)

    weights = build_label_graph_weights(target_df, target_cols, top_k=3)
    blend_base = baseline_frames["g_90_s_10"]
    for alpha in [0.10, 0.20, 0.35, 0.50]:
        smoothed = graph_smooth(blend_base, target_df, target_cols, weights, alpha=alpha)
        summary_rows.append(
            {
                "experiment": f"graph_smooth_alpha_{alpha:.2f}",
                "macro_auc": macro_auc(y_holdout, smoothed, target_cols),
                "kind": "graph",
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    (args.out_dir / "summary.json").write_text(json.dumps(summary.to_dict(orient="records"), indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
