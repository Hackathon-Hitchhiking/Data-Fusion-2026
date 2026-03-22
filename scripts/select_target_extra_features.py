from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from tqdm import tqdm

from lib.layout import resolve_data_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select top extra features per target with quick LGBM models")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--out-file", type=Path, default=Path("artifacts/feature_selection/target_top100_extra.json"))
    p.add_argument("--top-n", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-extra-missing-ratio", type=float, default=1.0)
    p.add_argument("--drop-duplicate-extra", action="store_true")
    p.add_argument("--duplicate-check-sample-size", type=int, default=50000)
    p.add_argument("--negative-ratio", type=float, default=4.0)
    p.add_argument("--max-rows-per-target", type=int, default=120000)
    p.add_argument("--n-estimators", type=int, default=120)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=64)
    p.add_argument("--feature-fraction", type=float, default=0.8)
    p.add_argument("--bagging-fraction", type=float, default=0.85)
    p.add_argument("--bagging-freq", type=int, default=1)
    p.add_argument("--min-positive-count", type=int, default=50)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--target", action="append", default=[], help="Optional target name to process. Pass multiple times.")
    p.add_argument("--targets-file", type=Path, default=None, help="Optional JSON list or newline-delimited file of targets.")
    return p.parse_args()


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


def sample_target_rows(
    y: pd.Series,
    max_rows: int,
    negative_ratio: float,
    min_positive_count: int,
    seed: int,
) -> np.ndarray:
    pos_idx = np.flatnonzero(y.to_numpy() == 1)
    neg_idx = np.flatnonzero(y.to_numpy() == 0)
    if len(pos_idx) < min_positive_count:
        return np.array([], dtype=np.int64)

    rng = np.random.default_rng(seed)
    pos_target = len(pos_idx)
    neg_target = int(min(len(neg_idx), max(1, int(pos_target * negative_ratio))))
    if pos_target + neg_target > max_rows:
        pos_target = min(len(pos_idx), max(1, int(max_rows / (1 + negative_ratio))))
        neg_target = min(len(neg_idx), max(1, max_rows - pos_target))
        if pos_target + neg_target > max_rows:
            overflow = pos_target + neg_target - max_rows
            if pos_target >= neg_target:
                pos_target = max(1, pos_target - overflow)
            else:
                neg_target = max(1, neg_target - overflow)

    pos_keep = rng.choice(pos_idx, size=pos_target, replace=False) if pos_target < len(pos_idx) else pos_idx
    neg_keep = rng.choice(neg_idx, size=neg_target, replace=False) if neg_target < len(neg_idx) else neg_idx
    keep = np.concatenate([pos_keep, neg_keep]).astype(np.int64)
    return np.sort(keep)


def load_target_filter(args: argparse.Namespace, available_targets: list[str]) -> list[str]:
    requested = set(args.target)
    if args.targets_file is not None:
        raw = args.targets_file.read_text().strip()
        if raw:
            if raw.lstrip().startswith("["):
                requested.update(json.loads(raw))
            else:
                requested.update(line.strip() for line in raw.splitlines() if line.strip())
    if not requested:
        return available_targets
    return [target_name for target_name in available_targets if target_name in requested]


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    train_main = pd.read_parquet(args.data_dir / "train_main_features.parquet")
    train_extra = pd.read_parquet(args.data_dir / "train_extra_features.parquet")
    target = pd.read_parquet(args.data_dir / "train_target.parquet")

    main_feature_cols = [c for c in train_main.columns if c != "customer_id"]
    extra_feature_cols = [c for c in train_extra.columns if c != "customer_id"]

    selected_extra_cols = extra_feature_cols[:]
    if args.max_extra_missing_ratio < 1.0:
        missing_ratio = train_extra[selected_extra_cols].isna().mean()
        selected_extra_cols = [c for c in selected_extra_cols if missing_ratio[c] <= args.max_extra_missing_ratio]

    if args.drop_duplicate_extra and selected_extra_cols:
        duplicate_groups = find_duplicate_columns(
            train_extra[selected_extra_cols], sample_size=args.duplicate_check_sample_size, seed=args.seed
        )
        dropped_duplicate_columns = {col for group in duplicate_groups for col in group[1:]}
        selected_extra_cols = [c for c in selected_extra_cols if c not in dropped_duplicate_columns]

    train = train_main.merge(train_extra[["customer_id"] + selected_extra_cols], on="customer_id", how="inner")
    train = train.merge(target, on="customer_id", how="inner")

    feature_cols = main_feature_cols + selected_extra_cols
    cat_cols = [c for c in main_feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    target_cols = [c for c in train.columns if c.startswith("target_")]
    target_cols = load_target_filter(args, target_cols)

    for col in cat_cols:
        train[col] = train[col].astype("float32").astype("Int32")
    for col in num_cols:
        train[col] = pd.to_numeric(train[col], errors="coerce").astype("float32")

    X = train[feature_cols]
    selection_map: dict[str, list[str]] = {}
    score_rows: list[dict[str, object]] = []

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
        "n_jobs": args.n_jobs,
        "verbose": -1,
    }

    for target_idx, target_name in enumerate(tqdm(target_cols, desc="Targets")):
        y = train[target_name]
        keep_idx = sample_target_rows(
            y=y,
            max_rows=args.max_rows_per_target,
            negative_ratio=args.negative_ratio,
            min_positive_count=args.min_positive_count,
            seed=args.seed + target_idx,
        )
        if len(keep_idx) == 0:
            selection_map[target_name] = []
            continue

        x_sample = X.iloc[keep_idx]
        y_sample = y.iloc[keep_idx]
        if y_sample.nunique() < 2:
            selection_map[target_name] = []
            continue
        pos_rate = float(y_sample.mean())
        params = {**base_params, "scale_pos_weight": (1 - pos_rate) / (pos_rate + 1e-6)}
        model = lgb.LGBMClassifier(**params)
        model.fit(x_sample, y_sample, categorical_feature=cat_cols)

        importance = pd.Series(model.booster_.feature_importance(importance_type="gain"), index=x_sample.columns)
        extra_importance = importance.loc[selected_extra_cols].sort_values(ascending=False)
        selected = extra_importance.head(args.top_n).index.tolist()
        selection_map[target_name] = selected

        for feature_name, gain in extra_importance.head(max(args.top_n, 200)).items():
            score_rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "gain": float(gain),
                    "selected": int(feature_name in set(selected)),
                    "positive_rate": pos_rate,
                    "sample_rows": int(len(keep_idx)),
                }
            )
        args.out_file.write_text(json.dumps(selection_map, indent=2))
        pd.DataFrame(score_rows).sort_values(["target", "gain"], ascending=[True, False]).to_csv(
            args.out_file.with_suffix(".csv"),
            index=False,
        )

    args.out_file.write_text(json.dumps(selection_map, indent=2))
    pd.DataFrame(score_rows).sort_values(["target", "gain"], ascending=[True, False]).to_csv(
        args.out_file.with_suffix(".csv"),
        index=False,
    )
    print(f"Saved target feature map to {args.out_file}")


if __name__ == "__main__":
    main()
