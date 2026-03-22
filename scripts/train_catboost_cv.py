from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from lib.layout import resolve_data_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CatBoost multi-label model with CV")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/catboost_cv"))
    p.add_argument("--feature-set", choices=["main", "all"], default="main")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-rows", type=int, default=0, help="0 means full data")
    p.add_argument("--subsample-stratified", action="store_true")
    p.add_argument("--iterations", type=int, default=800)
    p.add_argument("--depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--l2-leaf-reg", type=float, default=5.0)
    p.add_argument("--random-strength", type=float, default=1.0)
    p.add_argument("--thread-count", type=int, default=4)
    p.add_argument("--od-wait", type=int, default=100)
    p.add_argument("--cv-strategy", choices=["kfold", "stratified-groups"], default="stratified-groups")
    p.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    p.add_argument("--devices", type=str, default=None)
    p.add_argument("--max-extra-missing-ratio", type=float, default=1.0)
    p.add_argument("--drop-duplicate-extra", action="store_true")
    p.add_argument("--duplicate-check-sample-size", type=int, default=50000)
    p.add_argument(
        "--extra-features-file",
        type=Path,
        default=None,
        help="Optional JSON file with a list of extra feature names to keep before pruning.",
    )
    p.add_argument("--used-ram-limit", type=str, default="8gb")
    p.add_argument("--max-ctr-complexity", type=int, default=1)
    p.add_argument("--one-hot-max-size", type=int, default=32)
    p.add_argument("--border-count", type=int, default=64)
    return p.parse_args()


def group_target_columns(columns: list[str]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for col in columns:
        if not col.startswith("target_"):
            continue
        _, group_id, _ = col.split("_")
        groups[group_id].append(col)
    return {k: sorted(v) for k, v in sorted(groups.items(), key=lambda item: int(item[0]))}


def build_stratify_labels(target_df: pd.DataFrame, n_splits: int) -> pd.Series:
    target_cols = [c for c in target_df.columns if c.startswith("target_")]
    target_groups = group_target_columns(target_cols)
    group_presence = pd.DataFrame(
        {
            f"group_{group_id}": (target_df[group_cols].sum(axis=1) > 0).astype("int8")
            for group_id, group_cols in target_groups.items()
        }
    )
    label_count = target_df[target_cols].sum(axis=1).clip(upper=4).astype("int8")
    dominant_group = group_presence.idxmax(axis=1).fillna("group_0")
    strat_labels = dominant_group + "__c" + label_count.astype(str)
    current_counts = strat_labels.value_counts()
    rare_mask = strat_labels.map(current_counts) < n_splits
    strat_labels.loc[rare_mask] = "count_" + label_count.astype(str).loc[rare_mask]
    return strat_labels


def find_duplicate_columns(df: pd.DataFrame, sample_size: int, seed: int) -> list[list[str]]:
    if df.empty:
        return []

    sample_n = min(sample_size, len(df))
    sample_df = df.sample(sample_n, random_state=seed) if sample_n < len(df) else df
    hash_groups: dict[int, list[str]] = defaultdict(list)

    for col in sample_df.columns:
        col_hash = int(pd.util.hash_pandas_object(sample_df[col], index=False).sum())
        hash_groups[col_hash].append(col)

    duplicate_groups: list[list[str]] = []
    for candidate_cols in hash_groups.values():
        if len(candidate_cols) < 2:
            continue
        remaining = candidate_cols[:]
        while remaining:
            base = remaining.pop(0)
            group = [base]
            still_remaining: list[str] = []
            for col in remaining:
                if df[base].equals(df[col]):
                    group.append(col)
                else:
                    still_remaining.append(col)
            remaining = still_remaining
            if len(group) > 1:
                duplicate_groups.append(sorted(group))

    return sorted(duplicate_groups, key=len, reverse=True)


def load_data(
    data_dir: Path,
    feature_set: str,
    max_extra_missing_ratio: float,
    drop_duplicate_extra: bool,
    duplicate_check_sample_size: int,
    extra_features_file: Path | None,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    train_main = pd.read_parquet(data_dir / "train_main_features.parquet")
    target = pd.read_parquet(data_dir / "train_target.parquet")
    feature_manifest: dict[str, object] = {
        "feature_set": feature_set,
        "main_feature_count": int(train_main.shape[1] - 1),
        "extra_feature_count_raw": 0,
        "extra_feature_count_selected": 0,
        "dropped_for_missing_ratio": [],
        "duplicate_groups": [],
        "dropped_duplicate_columns": [],
    }

    if feature_set == "all":
        train_extra = pd.read_parquet(data_dir / "train_extra_features.parquet")
        extra_feature_cols = [c for c in train_extra.columns if c != "customer_id"]
        feature_manifest["extra_feature_count_raw"] = int(len(extra_feature_cols))

        selected_extra_cols = extra_feature_cols[:]
        if extra_features_file is not None:
            selected_from_file = set(json.loads(extra_features_file.read_text()))
            selected_extra_cols = [c for c in selected_extra_cols if c in selected_from_file]
            feature_manifest["extra_features_file"] = str(extra_features_file)
            feature_manifest["extra_features_file_count"] = int(len(selected_extra_cols))
        if max_extra_missing_ratio < 1.0:
            missing_ratio = train_extra[selected_extra_cols].isna().mean()
            dropped_for_missing = missing_ratio[missing_ratio > max_extra_missing_ratio].index.tolist()
            selected_extra_cols = [c for c in selected_extra_cols if c not in set(dropped_for_missing)]
            feature_manifest["dropped_for_missing_ratio"] = dropped_for_missing

        duplicate_groups: list[list[str]] = []
        dropped_duplicate_columns: list[str] = []
        if drop_duplicate_extra and selected_extra_cols:
            duplicate_groups = find_duplicate_columns(
                train_extra[selected_extra_cols], sample_size=duplicate_check_sample_size, seed=seed
            )
            dropped_duplicate_columns = [col for group in duplicate_groups for col in group[1:]]
            selected_extra_cols = [c for c in selected_extra_cols if c not in set(dropped_duplicate_columns)]

        feature_manifest["duplicate_groups"] = duplicate_groups
        feature_manifest["dropped_duplicate_columns"] = dropped_duplicate_columns
        feature_manifest["extra_feature_count_selected"] = int(len(selected_extra_cols))
        train = train_main.merge(train_extra[["customer_id"] + selected_extra_cols], on="customer_id", how="inner")
    else:
        train = train_main

    train = train.merge(target, on="customer_id", how="inner")
    return train, feature_manifest


def frame_mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train, feature_manifest = load_data(
        data_dir=args.data_dir,
        feature_set=args.feature_set,
        max_extra_missing_ratio=args.max_extra_missing_ratio,
        drop_duplicate_extra=args.drop_duplicate_extra,
        duplicate_check_sample_size=args.duplicate_check_sample_size,
        extra_features_file=args.extra_features_file,
        seed=args.seed,
    )
    target_cols = [c for c in train.columns if c.startswith("target_")]
    feature_cols = [c for c in train.columns if c not in target_cols + ["customer_id"]]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    for col in cat_cols:
        train[col] = train[col].astype("int32")
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").astype("float32")

    if args.subsample_rows and args.subsample_rows < len(train):
        if args.subsample_stratified:
            strat_labels = build_stratify_labels(train[target_cols].astype("float32"), n_splits=max(args.folds, 2))
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.subsample_rows, random_state=args.seed)
            sample_idx, _ = next(splitter.split(train, strat_labels))
            train = train.iloc[sample_idx].sort_values("customer_id").reset_index(drop=True)
        else:
            train = train.sample(args.subsample_rows, random_state=args.seed).sort_values("customer_id").reset_index(drop=True)

    X = train[feature_cols]
    Y = train[target_cols].astype("float32")
    print(
        {
            "stage": "data_ready",
            "rows": int(len(train)),
            "feature_count": len(feature_cols),
            "target_count": len(target_cols),
            "cat_count": len(cat_cols),
            "x_mem_mb": round(frame_mem_mb(X), 2),
            "y_mem_mb": round(frame_mem_mb(Y), 2),
            "used_ram_limit": args.used_ram_limit,
            "max_ctr_complexity": args.max_ctr_complexity,
            "one_hot_max_size": args.one_hot_max_size,
            "border_count": args.border_count,
        },
        flush=True,
    )

    if args.cv_strategy == "stratified-groups":
        strat_labels = build_stratify_labels(Y, n_splits=args.folds)
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X, strat_labels)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X)

    oof = np.zeros((len(train), len(target_cols)), dtype=np.float32)
    fold_scores: list[dict[str, float]] = []
    model_paths: list[str] = []

    model_params = {
        "loss_function": "MultiLogloss",
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "l2_leaf_reg": args.l2_leaf_reg,
        "random_strength": args.random_strength,
        "eval_metric": "MultiLogloss",
        "thread_count": args.thread_count,
        "random_seed": args.seed,
        "allow_writing_files": False,
        "verbose": False,
        "task_type": args.task_type,
        "used_ram_limit": args.used_ram_limit,
        "max_ctr_complexity": args.max_ctr_complexity,
        "one_hot_max_size": args.one_hot_max_size,
        "border_count": args.border_count,
    }
    if args.devices is not None:
        model_params["devices"] = args.devices

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        print({"stage": "fold_start", "fold": fold, "train_rows": int(len(tr_idx)), "valid_rows": int(len(va_idx))}, flush=True)
        xtr = X.iloc[tr_idx]
        xva = X.iloc[va_idx]
        ytr = Y.iloc[tr_idx]
        yva = Y.iloc[va_idx]

        train_pool = Pool(xtr, ytr, cat_features=cat_cols)
        valid_pool = Pool(xva, yva, cat_features=cat_cols)

        model = CatBoostClassifier(**model_params)
        model.fit(
            train_pool,
            eval_set=valid_pool,
            use_best_model=True,
            early_stopping_rounds=args.od_wait,
        )

        pred = model.predict(valid_pool, prediction_type="RawFormulaVal")
        pred = np.asarray(pred, dtype=np.float32)
        if pred.ndim == 1:
            pred = pred.reshape(-1, 1)
        oof[va_idx] = pred

        fold_stat: dict[str, float] = {}
        for idx, target_name in enumerate(tqdm(target_cols, desc=f"Fold {fold}")):
            y_true = yva.iloc[:, idx]
            score = roc_auc_score(y_true, pred[:, idx]) if y_true.nunique() > 1 else 0.5
            fold_stat[target_name] = float(score)
        fold_stat["macro_auc"] = float(np.mean(list(fold_stat.values())))
        fold_scores.append(fold_stat)
        print(f"Fold {fold} macro AUC: {fold_stat['macro_auc']:.6f}", flush=True)

        del xtr, xva, ytr, yva, train_pool, valid_pool, model, pred
        gc.collect()

        model_path = args.out_dir / f"model_fold{fold}.cbm"
        model.save_model(model_path)
        model_paths.append(str(model_path))

    target_scores: list[dict[str, float]] = []
    for idx, target_name in enumerate(target_cols):
        y_true = Y.iloc[:, idx]
        score = roc_auc_score(y_true, oof[:, idx]) if y_true.nunique() > 1 else 0.5
        target_scores.append({"target": target_name, "oof_auc": float(score)})
    macro_auc = float(np.mean([row["oof_auc"] for row in target_scores]))
    print(f"OOF macro AUC: {macro_auc:.6f}")

    oof_df = pd.DataFrame(oof, columns=[c.replace("target_", "predict_") for c in target_cols])
    oof_df.insert(0, "customer_id", train["customer_id"].values)
    oof_df.to_parquet(args.out_dir / "oof_predictions.parquet", index=False)

    pd.DataFrame(fold_scores).to_csv(args.out_dir / "fold_scores.csv", index=False)
    pd.DataFrame(target_scores).sort_values("oof_auc").to_csv(args.out_dir / "target_scores.csv", index=False)
    (args.out_dir / "models_index.json").write_text(json.dumps({"fold_models": model_paths}, indent=2))
    (args.out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "oof_macro_auc": float(macro_auc),
                "fold_macro_auc": [float(f["macro_auc"]) for f in fold_scores],
                "n_rows": int(len(train)),
                "n_features": int(len(feature_cols)),
                "cv_strategy": args.cv_strategy,
                "feature_set": args.feature_set,
                "feature_manifest": feature_manifest,
                "task_type": args.task_type,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
