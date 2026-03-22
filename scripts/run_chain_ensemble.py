from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lib.layout import project_root, resolve_competition_file
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DEFAULT_OUT_DIR = ROOT / "artifacts" / "chain_ensemble_hardblock_v1"
DEFAULT_GANDALF_VAL = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
DEFAULT_STACK_OOF = ROOT / "artifacts" / "stack_top100_self_c005_full_750k" / "oof_predictions.parquet"
DEFAULT_BASE_OOF = ROOT / "artifacts" / "full_lgbm_top100_750k_2fold" / "oof_predictions.parquet"
DEFAULT_GANDALF_TARGET_SCORE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "target_scores.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run hard-block classifier chain ensemble over prediction space.")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    p.add_argument("--stack-oof", type=Path, default=DEFAULT_STACK_OOF)
    p.add_argument("--base-oof", type=Path, default=DEFAULT_BASE_OOF)
    p.add_argument("--gandalf-target-scores", type=Path, default=DEFAULT_GANDALF_TARGET_SCORE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hardest-count", type=int, default=12)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--n-orders", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--c-value", type=float, default=0.2)
    return p.parse_args()


def hardest_targets(score_file: Path, count: int) -> list[str]:
    score_df = pd.read_csv(score_file).sort_values("oof_auc")
    return score_df.head(count)["target"].tolist()


def row_stratify(target_frame: pd.DataFrame, target_cols: list[str]) -> np.ndarray:
    return target_frame[target_cols].sum(axis=1).clip(upper=4).astype("int8").to_numpy()


def build_feature_frame(
    gandalf: pd.DataFrame,
    stack: pd.DataFrame,
    base: pd.DataFrame,
    pred_cols: list[str],
) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {"customer_id": gandalf["customer_id"].values}
    for pred_col in pred_cols:
        data[f"g__{pred_col}"] = gandalf[pred_col].astype("float32").values
        data[f"s__{pred_col}"] = stack[pred_col].astype("float32").values
        data[f"b__{pred_col}"] = base[pred_col].astype("float32").values
    return pd.DataFrame(data)


def fit_predict_chain_fold(
    x_train_base: np.ndarray,
    x_val_base: np.ndarray,
    y_train_df: pd.DataFrame,
    chain_targets: list[str],
    c_value: float,
    seed: int,
) -> dict[str, np.ndarray]:
    train_aug = x_train_base
    val_aug = x_val_base
    val_predictions: dict[str, np.ndarray] = {}

    for target_name in chain_targets:
        y_train = y_train_df[target_name].to_numpy(dtype=np.int8)
        if np.unique(y_train).size < 2:
            pred_val = np.full(len(val_aug), float(y_train.mean()), dtype=np.float64)
        else:
            model = LogisticRegression(
                C=c_value,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed,
            )
            model.fit(train_aug, y_train)
            pred_val = model.predict_proba(val_aug)[:, 1]

        val_predictions[target_name] = pred_val.astype("float64")
        train_aug = np.column_stack([train_aug, y_train.astype("float32")])
        val_aug = np.column_stack([val_aug, pred_val.astype("float32")])

    return val_predictions


def run_chain_ensemble(
    feature_frame: pd.DataFrame,
    y_frame: pd.DataFrame,
    chain_targets: list[str],
    *,
    n_splits: int,
    n_orders: int,
    seed: int,
    c_value: float,
) -> dict[str, np.ndarray]:
    base_feature_cols = [c for c in feature_frame.columns if c != "customer_id"]
    x_all = feature_frame[base_feature_cols].to_numpy(dtype=np.float32)
    stratify = row_stratify(y_frame, chain_targets)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    oof_predictions = {target: np.zeros(len(feature_frame), dtype=np.float64) for target in chain_targets}
    order_rows: list[dict[str, object]] = []

    for order_idx in range(n_orders):
        if order_idx == 0:
            order = chain_targets.copy()
        else:
            order = chain_targets.copy()
            rng.shuffle(order)
        order_rows.append({"order_index": order_idx, "order": order})
        print({"stage": "chain_order_start", "order_index": order_idx, "order": order}, flush=True)

        fold_pred = {target: np.zeros(len(feature_frame), dtype=np.float64) for target in chain_targets}
        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(x_all, stratify), start=1):
            print(
                {"stage": "chain_fold_start", "order_index": order_idx, "fold_index": fold_idx, "train_rows": len(train_idx), "val_rows": len(val_idx)},
                flush=True,
            )
            y_train_df = y_frame.iloc[train_idx][chain_targets].reset_index(drop=True)
            pred_dict = fit_predict_chain_fold(
                x_all[train_idx],
                x_all[val_idx],
                y_train_df,
                order,
                c_value=c_value,
                seed=seed + order_idx * 100 + fold_idx,
            )
            for target_name, pred_val in pred_dict.items():
                fold_pred[target_name][val_idx] = pred_val
            print({"stage": "chain_fold_done", "order_index": order_idx, "fold_index": fold_idx}, flush=True)

        for target_name in chain_targets:
            oof_predictions[target_name] += fold_pred[target_name] / n_orders
        print({"stage": "chain_order_done", "order_index": order_idx}, flush=True)

    return {"predictions": oof_predictions, "orders": order_rows}


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gandalf = normalize_prediction_columns(pd.read_parquet(args.gandalf_val)).sort_values("customer_id").reset_index(drop=True)
    pred_cols = [c for c in gandalf.columns if c != "customer_id"]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]
    hard_targets = hardest_targets(args.gandalf_target_scores, args.hardest_count)

    holdout_ids = gandalf[["customer_id"]]
    target_df = (
        pd.read_parquet(args.target_file)
        .merge(holdout_ids, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    stack = (
        pd.read_parquet(args.stack_oof)
        .merge(holdout_ids, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    base = (
        pd.read_parquet(args.base_oof)
        .merge(holdout_ids, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )

    feature_frame = build_feature_frame(gandalf, stack, base, pred_cols)
    print(
        {
            "stage": "chain_ensemble_start",
            "hard_targets": hard_targets,
            "rows": len(feature_frame),
            "feature_count": len(feature_frame.columns) - 1,
            "n_splits": args.n_splits,
            "n_orders": args.n_orders,
            "c_value": args.c_value,
        },
        flush=True,
    )

    result = run_chain_ensemble(
        feature_frame,
        target_df,
        hard_targets,
        n_splits=args.n_splits,
        n_orders=args.n_orders,
        seed=args.seed,
        c_value=args.c_value,
    )

    pred_frame = gandalf[["customer_id"] + pred_cols].copy()
    for target_name in hard_targets:
        pred_col = target_name.replace("target_", "predict_")
        pred_frame[pred_col] = result["predictions"][target_name]

    hard_pred_cols = [t.replace("target_", "predict_") for t in hard_targets]
    baseline_hard = gandalf[["customer_id"] + hard_pred_cols].copy()
    chain_hard = pred_frame[["customer_id"] + hard_pred_cols].copy()

    per_target_rows: list[dict[str, float | str]] = []
    for target_name in hard_targets:
        pred_col = target_name.replace("target_", "predict_")
        y_true = target_df[target_name].to_numpy(dtype=np.int8)
        baseline_auc = roc_auc_score(y_true, baseline_hard[pred_col])
        chain_auc = roc_auc_score(y_true, chain_hard[pred_col])
        per_target_rows.append(
            {
                "target": target_name,
                "baseline_auc": float(baseline_auc),
                "chain_auc": float(chain_auc),
                "delta_auc": float(chain_auc - baseline_auc),
            }
        )

    candidate_rows: list[dict[str, float | str]] = []
    for alpha in [0.30, 0.50, 0.70, 1.00]:
        candidate = gandalf[["customer_id"] + pred_cols].copy()
        for target_name in hard_targets:
            pred_col = target_name.replace("target_", "predict_")
            candidate[pred_col] = (
                (1.0 - alpha) * gandalf[pred_col].to_numpy(dtype=np.float64)
                + alpha * pred_frame[pred_col].to_numpy(dtype=np.float64)
            )
        candidate_rows.append(
            {
                "candidate": f"chain_blend_{alpha:.2f}",
                "macro_auc_all": float(macro_auc(target_df, candidate, target_cols)),
                "macro_auc_hardblock": float(macro_auc(target_df, candidate[["customer_id"] + hard_pred_cols], hard_targets)),
            }
        )

    candidate_rows.append(
        {
            "candidate": "gandalf_base",
            "macro_auc_all": float(macro_auc(target_df, gandalf, target_cols)),
            "macro_auc_hardblock": float(macro_auc(target_df, baseline_hard, hard_targets)),
        }
    )
    candidate_rows.append(
        {
            "candidate": "chain_replace",
            "macro_auc_all": float(macro_auc(target_df, pred_frame, target_cols)),
            "macro_auc_hardblock": float(macro_auc(target_df, chain_hard, hard_targets)),
        }
    )

    candidate_df = pd.DataFrame(candidate_rows).sort_values("macro_auc_all", ascending=False).reset_index(drop=True)
    per_target_df = pd.DataFrame(per_target_rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)

    pred_frame.to_parquet(args.out_dir / "oof_predictions.parquet", index=False)
    per_target_df.to_csv(args.out_dir / "per_target_auc.csv", index=False)
    candidate_df.to_csv(args.out_dir / "candidate_macro_auc.csv", index=False)
    (args.out_dir / "orders.json").write_text(json.dumps(result["orders"], indent=2))

    metrics = {
        "hard_targets": hard_targets,
        "best_candidate": candidate_df.iloc[0]["candidate"],
        "best_macro_auc_all": float(candidate_df.iloc[0]["macro_auc_all"]),
        "best_macro_auc_hardblock": float(candidate_df.iloc[0]["macro_auc_hardblock"]),
        "gandalf_macro_auc_all": float(candidate_df[candidate_df["candidate"] == "gandalf_base"].iloc[0]["macro_auc_all"]),
        "gandalf_macro_auc_hardblock": float(candidate_df[candidate_df["candidate"] == "gandalf_base"].iloc[0]["macro_auc_hardblock"]),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(candidate_df.to_string(index=False), flush=True)
    print(per_target_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
