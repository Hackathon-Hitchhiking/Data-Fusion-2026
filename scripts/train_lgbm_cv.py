from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from tqdm import tqdm

from lib.layout import resolve_data_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train multi-label LGBM with CV")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/lgbm_cv"))
    p.add_argument("--feature-set", choices=["main", "all"], default="all")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--subsample-rows", type=int, default=0, help="0 means full data")
    p.add_argument("--subsample-stratified", action="store_true")
    p.add_argument("--n-estimators", type=int, default=1200)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--num-leaves", type=int, default=128)
    p.add_argument("--feature-fraction", type=float, default=0.7)
    p.add_argument("--bagging-fraction", type=float, default=0.8)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--cv-strategy", choices=["kfold", "stratified-groups"], default="stratified-groups")
    p.add_argument("--max-extra-missing-ratio", type=float, default=1.0)
    p.add_argument("--drop-duplicate-extra", action="store_true")
    p.add_argument("--duplicate-check-sample-size", type=int, default=50000)
    p.add_argument(
        "--extra-features-file",
        type=Path,
        default=None,
        help="Optional JSON file with a list of extra feature names to keep before pruning.",
    )
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


def load_extra_feature_spec(extra_features_file: Path | None) -> tuple[list[str] | None, dict[str, list[str]]]:
    if extra_features_file is None:
        return None, {}

    raw = json.loads(extra_features_file.read_text())
    if isinstance(raw, list):
        return raw, {}
    if isinstance(raw, dict):
        target_extra_map = {
            str(target_name): list(dict.fromkeys(feature_names))
            for target_name, feature_names in raw.items()
            if isinstance(feature_names, list)
        }
        union_features: list[str] = []
        seen: set[str] = set()
        for feature_names in target_extra_map.values():
            for feature_name in feature_names:
                if feature_name not in seen:
                    union_features.append(feature_name)
                    seen.add(feature_name)
        return union_features, target_extra_map
    raise ValueError("extra_features_file must contain either a list of feature names or a dict target->feature_list")


def load_data(
    data_dir: Path,
    feature_set: str,
    max_extra_missing_ratio: float,
    drop_duplicate_extra: bool,
    duplicate_check_sample_size: int,
    extra_features_file: Path | None,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], list[str], dict[str, list[str]]]:
    train_main = pd.read_parquet(data_dir / "train_main_features.parquet")
    target = pd.read_parquet(data_dir / "train_target.parquet")
    main_feature_cols = [c for c in train_main.columns if c != "customer_id"]
    feature_manifest: dict[str, object] = {
        "feature_set": feature_set,
        "main_feature_count": int(train_main.shape[1] - 1),
        "extra_feature_count_raw": 0,
        "extra_feature_count_selected": 0,
        "dropped_for_missing_ratio": [],
        "duplicate_groups": [],
        "dropped_duplicate_columns": [],
        "selected_extra_columns": [],
        "target_extra_features": {},
    }
    target_extra_map: dict[str, list[str]] = {}

    if feature_set == "all":
        train_extra = pd.read_parquet(data_dir / "train_extra_features.parquet")
        extra_feature_cols = [c for c in train_extra.columns if c != "customer_id"]
        feature_manifest["extra_feature_count_raw"] = int(len(extra_feature_cols))

        selected_extra_cols = extra_feature_cols[:]
        if extra_features_file is not None:
            selected_from_file, target_extra_map = load_extra_feature_spec(extra_features_file)
            if selected_from_file is not None:
                selected_lookup = set(selected_from_file)
                selected_extra_cols = [c for c in selected_extra_cols if c in selected_lookup]
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
        feature_manifest["selected_extra_columns"] = selected_extra_cols
        if target_extra_map:
            selected_lookup = set(selected_extra_cols)
            feature_manifest["target_extra_features"] = {
                target_name: [col for col in feature_names if col in selected_lookup]
                for target_name, feature_names in target_extra_map.items()
            }
            target_extra_map = feature_manifest["target_extra_features"]
        train = train_main.merge(train_extra[["customer_id"] + selected_extra_cols], on="customer_id", how="inner")
    else:
        train = train_main

    train = train.merge(target, on="customer_id", how="inner")
    return train, target, feature_manifest, main_feature_cols, target_extra_map


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train, target, feature_manifest, main_feature_cols, target_extra_map = load_data(
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
    global_extra_cols = [c for c in feature_cols if c not in set(main_feature_cols)]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

    for c in cat_cols:
        train[c] = train[c].astype("float32").astype("Int32")

    for c in feature_cols:
        if c not in cat_cols:
            train[c] = pd.to_numeric(train[c], errors="coerce").astype("float32")

    if args.subsample_rows and args.subsample_rows < len(train):
        if args.subsample_stratified:
            strat_labels = build_stratify_labels(train[target_cols], n_splits=max(args.folds, 2))
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=args.subsample_rows, random_state=args.seed)
            sample_idx, _ = next(splitter.split(train, strat_labels))
            train = train.iloc[sample_idx].sort_values("customer_id").reset_index(drop=True)
        else:
            train = train.sample(args.subsample_rows, random_state=args.seed).sort_values("customer_id").reset_index(drop=True)

    X = train[feature_cols]
    Y = train[target_cols]

    if args.cv_strategy == "stratified-groups":
        strat_labels = build_stratify_labels(Y, n_splits=args.folds)
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X, strat_labels)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X)

    oof = np.zeros((len(X), len(target_cols)), dtype=np.float32)
    fold_scores: list[dict[str, float]] = []
    model_paths: dict[str, list[str]] = {t: [] for t in target_cols}

    base_params = {
        "objective": "binary",
        "metric": "auc",
        "boosting_type": "gbdt",
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        fold_stat: dict[str, float] = {}

        for t_idx, target_name in enumerate(tqdm(target_cols, desc=f"Fold {fold}")):
            current_extra_cols = target_extra_map.get(target_name, global_extra_cols)
            current_feature_cols = main_feature_cols + current_extra_cols
            xtr = X.iloc[tr_idx][current_feature_cols]
            xva = X.iloc[va_idx][current_feature_cols]
            ytr = Y.iloc[tr_idx][target_name]
            yva = Y.iloc[va_idx][target_name]

            pos_rate = float(ytr.mean())
            scale_pos_weight = (1 - pos_rate) / (pos_rate + 1e-6)

            if ytr.nunique() < 2:
                pred = np.full(len(yva), pos_rate, dtype=np.float32)
                oof[va_idx, t_idx] = pred
                auc = 0.5
                fold_stat[target_name] = float(auc)
                model_file = args.out_dir / f"model_{target_name}_fold{fold}.pkl"
                with open(model_file, "wb") as f:
                    pickle.dump({"constant_pred": pos_rate}, f)
                model_paths[target_name].append(str(model_file))
                continue

            params = {**base_params, "scale_pos_weight": scale_pos_weight}
            model = lgb.LGBMClassifier(**params)
            model.fit(
                xtr,
                ytr,
                eval_set=[(xva, yva)],
                eval_metric="auc",
                categorical_feature=[c for c in cat_cols if c in xtr.columns],
                callbacks=[lgb.early_stopping(120, verbose=False)],
            )
            pred = model.predict_proba(xva)[:, 1]
            oof[va_idx, t_idx] = pred
            auc = roc_auc_score(yva, pred) if yva.nunique() > 1 else 0.5
            fold_stat[target_name] = float(auc)

            model_file = args.out_dir / f"model_{target_name}_fold{fold}.pkl"
            with open(model_file, "wb") as f:
                pickle.dump(model, f)
            model_paths[target_name].append(str(model_file))

        fold_stat["macro_auc"] = float(np.mean(list(fold_stat.values())))
        fold_scores.append(fold_stat)
        print(f"Fold {fold} macro AUC: {fold_stat['macro_auc']:.6f}")

    target_scores: list[dict[str, float]] = []
    for t_idx, target_name in enumerate(target_cols):
        score = roc_auc_score(Y.iloc[:, t_idx], oof[:, t_idx]) if Y.iloc[:, t_idx].nunique() > 1 else 0.5
        target_scores.append({"target": target_name, "oof_auc": float(score)})
    macro_auc = float(np.mean([row["oof_auc"] for row in target_scores]))
    print(f"OOF macro AUC: {macro_auc:.6f}")

    oof_df = pd.DataFrame(oof, columns=[c.replace("target_", "predict_") for c in target_cols])
    oof_df.insert(0, "customer_id", train["customer_id"].values)
    oof_df.to_parquet(args.out_dir / "oof_predictions.parquet", index=False)

    pd.DataFrame(fold_scores).to_csv(args.out_dir / "fold_scores.csv", index=False)
    pd.DataFrame(target_scores).sort_values("oof_auc").to_csv(args.out_dir / "target_scores.csv", index=False)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "oof_macro_auc": float(macro_auc),
                "fold_macro_auc": [float(f["macro_auc"]) for f in fold_scores],
                "feature_set": args.feature_set,
                "n_rows": int(len(train)),
                "n_features": int(len(feature_cols)),
                "cv_strategy": args.cv_strategy,
                "feature_manifest": feature_manifest,
            },
            indent=2,
        )
    )
    (args.out_dir / "models_index.json").write_text(json.dumps(model_paths, indent=2))
    (args.out_dir / "feature_manifest.json").write_text(json.dumps(feature_manifest, indent=2))


if __name__ == "__main__":
    main()
