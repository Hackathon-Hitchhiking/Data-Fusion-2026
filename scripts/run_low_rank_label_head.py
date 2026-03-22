from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from lib.layout import project_root, resolve_competition_file
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DEFAULT_OUT_DIR = ROOT / "artifacts" / "low_rank_label_head_v1"
DEFAULT_GANDALF_VAL = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
DEFAULT_STACK_OOF = ROOT / "artifacts" / "stack_top100_self_c005_full_750k" / "oof_predictions.parquet"
DEFAULT_BASE_OOF = ROOT / "artifacts" / "full_lgbm_top100_750k_2fold" / "oof_predictions.parquet"
DEFAULT_GANDALF_TARGET_SCORE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "target_scores.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run low-rank label interaction head over aligned prediction space.")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    p.add_argument("--stack-oof", type=Path, default=DEFAULT_STACK_OOF)
    p.add_argument("--base-oof", type=Path, default=DEFAULT_BASE_OOF)
    p.add_argument("--gandalf-target-scores", type=Path, default=DEFAULT_GANDALF_TARGET_SCORE)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--hardest-count", type=int, default=12)
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--rank", type=int, default=12)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--c-value", type=float, default=0.2)
    return p.parse_args()


def hardest_targets(score_file: Path, count: int) -> list[str]:
    return pd.read_csv(score_file).sort_values("oof_auc").head(count)["target"].tolist()


def row_stratify(target_frame: pd.DataFrame, target_cols: list[str]) -> np.ndarray:
    return target_frame[target_cols].sum(axis=1).clip(upper=4).astype("int8").to_numpy()


def load_aligned_frames(
    target_file: Path,
    gandalf_val: Path,
    stack_oof: Path,
    base_oof: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    gandalf = normalize_prediction_columns(pd.read_parquet(gandalf_val)).sort_values("customer_id").reset_index(drop=True)
    holdout_ids = gandalf[["customer_id"]]
    target_df = (
        pd.read_parquet(target_file)
        .merge(holdout_ids, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    stack = (
        pd.read_parquet(stack_oof)
        .merge(holdout_ids, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    base = (
        pd.read_parquet(base_oof)
        .merge(holdout_ids, on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    pred_cols = [c for c in gandalf.columns if c != "customer_id"]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]
    return target_df, gandalf, stack, base, pred_cols, target_cols


def build_family_matrix(gandalf: pd.DataFrame, stack: pd.DataFrame, base: pd.DataFrame, pred_cols: list[str]) -> np.ndarray:
    mats = [
        gandalf[pred_cols].to_numpy(dtype=np.float32),
        stack[pred_cols].to_numpy(dtype=np.float32),
        base[pred_cols].to_numpy(dtype=np.float32),
    ]
    return np.concatenate(mats, axis=1)


def run_low_rank_cv(
    family_matrix: np.ndarray,
    target_df: pd.DataFrame,
    pred_cols: list[str],
    hard_targets: list[str],
    *,
    n_splits: int,
    rank: int,
    seed: int,
    c_value: float,
    gandalf: pd.DataFrame,
    stack: pd.DataFrame,
    base: pd.DataFrame,
) -> pd.DataFrame:
    stratify = row_stratify(target_df, hard_targets)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof = gandalf[["customer_id"] + pred_cols].copy()
    for target_name in hard_targets:
        pred_col = target_name.replace("target_", "predict_")
        oof[pred_col] = gandalf[pred_col].to_numpy(dtype=np.float64)

    for target_name in hard_targets:
        pred_col = target_name.replace("target_", "predict_")
        y_target = target_df[target_name].to_numpy(dtype=np.int8)
        pred = np.zeros(len(target_df), dtype=np.float64)

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(family_matrix, stratify), start=1):
            x_train_base = family_matrix[train_idx]
            x_val_base = family_matrix[val_idx]
            svd = TruncatedSVD(n_components=rank, random_state=seed + fold_idx)
            z_train = svd.fit_transform(x_train_base)
            z_val = svd.transform(x_val_base)

            j = pred_cols.index(pred_col)
            core_train = np.column_stack(
                [
                    gandalf.iloc[train_idx][pred_col].to_numpy(dtype=np.float32),
                    stack.iloc[train_idx][pred_col].to_numpy(dtype=np.float32),
                    base.iloc[train_idx][pred_col].to_numpy(dtype=np.float32),
                    z_train.astype(np.float32),
                ]
            )
            core_val = np.column_stack(
                [
                    gandalf.iloc[val_idx][pred_col].to_numpy(dtype=np.float32),
                    stack.iloc[val_idx][pred_col].to_numpy(dtype=np.float32),
                    base.iloc[val_idx][pred_col].to_numpy(dtype=np.float32),
                    z_val.astype(np.float32),
                ]
            )
            y_train = y_target[train_idx]
            if np.unique(y_train).size < 2:
                pred[val_idx] = float(y_train.mean())
                continue
            model = LogisticRegression(
                C=c_value,
                max_iter=1000,
                solver="liblinear",
                class_weight="balanced",
                random_state=seed + fold_idx,
            )
            model.fit(core_train, y_train)
            pred[val_idx] = model.predict_proba(core_val)[:, 1]

        oof[pred_col] = pred
        print({"stage": "low_rank_target_done", "target": target_name}, flush=True)

    return oof


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_df, gandalf, stack, base, pred_cols, target_cols = load_aligned_frames(
        args.target_file, args.gandalf_val, args.stack_oof, args.base_oof
    )
    hard_targets = hardest_targets(args.gandalf_target_scores, args.hardest_count)
    family_matrix = build_family_matrix(gandalf, stack, base, pred_cols)
    print(
        {
            "stage": "low_rank_start",
            "hard_targets": hard_targets,
            "rows": len(target_df),
            "family_feature_count": int(family_matrix.shape[1]),
            "rank": args.rank,
            "n_splits": args.n_splits,
            "c_value": args.c_value,
        },
        flush=True,
    )

    oof = run_low_rank_cv(
        family_matrix,
        target_df,
        pred_cols,
        hard_targets,
        n_splits=args.n_splits,
        rank=args.rank,
        seed=args.seed,
        c_value=args.c_value,
        gandalf=gandalf,
        stack=stack,
        base=base,
    )

    hard_pred_cols = [t.replace("target_", "predict_") for t in hard_targets]
    per_target_rows: list[dict[str, float | str]] = []
    for target_name in hard_targets:
        pred_col = target_name.replace("target_", "predict_")
        y_true = target_df[target_name].to_numpy(dtype=np.int8)
        base_auc = roc_auc_score(y_true, gandalf[pred_col])
        head_auc = roc_auc_score(y_true, oof[pred_col])
        per_target_rows.append(
            {
                "target": target_name,
                "baseline_auc": float(base_auc),
                "low_rank_auc": float(head_auc),
                "delta_auc": float(head_auc - base_auc),
            }
        )

    candidate_rows: list[dict[str, float | str]] = []
    for alpha in [0.30, 0.50, 0.70, 1.00]:
        candidate = gandalf[["customer_id"] + pred_cols].copy()
        for target_name in hard_targets:
            pred_col = target_name.replace("target_", "predict_")
            candidate[pred_col] = (
                (1.0 - alpha) * gandalf[pred_col].to_numpy(dtype=np.float64)
                + alpha * oof[pred_col].to_numpy(dtype=np.float64)
            )
        candidate_rows.append(
            {
                "candidate": f"low_rank_blend_{alpha:.2f}",
                "macro_auc_all": float(macro_auc(target_df, candidate, target_cols)),
                "macro_auc_hardblock": float(macro_auc(target_df, candidate[["customer_id"] + hard_pred_cols], hard_targets)),
            }
        )

    candidate_rows.append(
        {
            "candidate": "gandalf_base",
            "macro_auc_all": float(macro_auc(target_df, gandalf, target_cols)),
            "macro_auc_hardblock": float(macro_auc(target_df, gandalf[["customer_id"] + hard_pred_cols], hard_targets)),
        }
    )
    candidate_rows.append(
        {
            "candidate": "low_rank_replace",
            "macro_auc_all": float(macro_auc(target_df, oof, target_cols)),
            "macro_auc_hardblock": float(macro_auc(target_df, oof[["customer_id"] + hard_pred_cols], hard_targets)),
        }
    )

    per_target_df = pd.DataFrame(per_target_rows).sort_values("delta_auc", ascending=False).reset_index(drop=True)
    candidate_df = pd.DataFrame(candidate_rows).sort_values("macro_auc_all", ascending=False).reset_index(drop=True)

    oof.to_parquet(args.out_dir / "oof_predictions.parquet", index=False)
    per_target_df.to_csv(args.out_dir / "per_target_auc.csv", index=False)
    candidate_df.to_csv(args.out_dir / "candidate_macro_auc.csv", index=False)

    metrics = {
        "hard_targets": hard_targets,
        "best_candidate": str(candidate_df.iloc[0]["candidate"]),
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
