from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from tqdm import tqdm

from lib.layout import resolve_competition_file


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stack multiple OOF prediction files with logistic regression")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/stack_oof"))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cv-strategy", choices=["kfold", "stratified-groups"], default="stratified-groups")
    p.add_argument("--model", choices=["logreg", "lgbm"], default="logreg")
    p.add_argument("--c", type=float, default=0.5)
    p.add_argument("--n-estimators", type=int, default=150)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--pair-features-top-k", type=int, default=0)
    p.add_argument(
        "--oof",
        action="append",
        default=[],
        help="Path to an OOF parquet. Pass multiple times. File stem is used as the model prefix unless --name is used.",
    )
    p.add_argument(
        "--name",
        action="append",
        default=[],
        help="Optional model name for the corresponding --oof path. If omitted, file stem is used.",
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
    counts = strat_labels.value_counts()
    rare_mask = strat_labels.map(counts) < n_splits
    strat_labels.loc[rare_mask] = "count_" + label_count.astype(str).loc[rare_mask]
    return strat_labels


def add_group_features(df: pd.DataFrame) -> pd.DataFrame:
    pred_cols = [c for c in df.columns if c != "customer_id"]
    by_model_group: dict[tuple[str, str], list[str]] = defaultdict(list)
    for col in pred_cols:
        if "__" not in col:
            continue
        model_name, pred_name = col.split("__", 1)
        _, group_id, _ = pred_name.split("_")
        by_model_group[(model_name, group_id)].append(col)

    extra_features: dict[str, pd.Series] = {}
    for (model_name, group_id), cols in by_model_group.items():
        extra_features[f"{model_name}__group_{group_id}_sum"] = df[cols].sum(axis=1)
        extra_features[f"{model_name}__group_{group_id}_max"] = df[cols].max(axis=1)

    if extra_features:
        return pd.concat([df, pd.DataFrame(extra_features)], axis=1)
    return df


def top_conditional_lift_pairs(target_df: pd.DataFrame, top_k: int) -> list[tuple[str, str]]:
    if top_k <= 0:
        return []

    target_cols = [c for c in target_df.columns if c.startswith("target_")]
    target_means = target_df[target_cols].mean()
    lifts: list[tuple[float, str, str]] = []
    for idx, target_a in enumerate(target_cols):
        mask_a = target_df[target_a] == 1
        if not mask_a.any():
            continue
        for target_b in target_cols[idx + 1 :]:
            p_b = float(target_means[target_b])
            if p_b <= 0:
                continue
            p_b_given_a = float(target_df.loc[mask_a, target_b].mean())
            lift = p_b_given_a / (p_b + 1e-12)
            lifts.append((lift, target_a, target_b))
    lifts.sort(reverse=True)
    return [(a, b) for _, a, b in lifts[:top_k]]


def add_pair_features(df: pd.DataFrame, pair_targets: list[tuple[str, str]]) -> pd.DataFrame:
    if not pair_targets:
        return df

    pred_cols = [c for c in df.columns if c != "customer_id"]
    model_names = sorted({c.split("__", 1)[0] for c in pred_cols if "__" in c})
    extra_features: dict[str, pd.Series] = {}
    for model_name in model_names:
        for target_a, target_b in pair_targets:
            pred_a = f"{model_name}__{target_a.replace('target_', 'predict_')}"
            pred_b = f"{model_name}__{target_b.replace('target_', 'predict_')}"
            if pred_a in df.columns and pred_b in df.columns:
                key = f"{model_name}__pair_{target_a}_{target_b}_prod"
                extra_features[key] = df[pred_a] * df[pred_b]
    if extra_features:
        return pd.concat([df, pd.DataFrame(extra_features)], axis=1)
    return df


def main() -> None:
    args = parse_args()
    if not args.oof:
        raise ValueError("At least one --oof file is required")
    if args.name and len(args.name) != len(args.oof):
        raise ValueError("When provided, the number of --name values must match the number of --oof paths")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_parquet(args.target_file)
    target_cols = [c for c in target_df.columns if c.startswith("target_")]
    pred_cols = [c.replace("target_", "predict_") for c in target_cols]

    meta = target_df[["customer_id"]].copy()
    model_names = args.name if args.name else [Path(path).stem for path in args.oof]
    for model_name, path_str in zip(model_names, args.oof):
        oof_df = pd.read_parquet(path_str)
        rename_map = {col: f"{model_name}__{col}" for col in oof_df.columns if col != "customer_id"}
        meta = meta.merge(oof_df.rename(columns=rename_map), on="customer_id", how="inner")

    target_df = target_df.merge(meta[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    meta = meta.sort_values("customer_id").reset_index(drop=True)
    pair_targets = top_conditional_lift_pairs(target_df, top_k=args.pair_features_top_k)
    meta = add_group_features(meta)
    meta = add_pair_features(meta, pair_targets)
    feature_cols = [c for c in meta.columns if c != "customer_id"]
    X = meta[feature_cols].astype("float32")
    Y = target_df[target_cols]

    if args.cv_strategy == "stratified-groups":
        strat_labels = build_stratify_labels(Y, n_splits=args.folds)
        splitter = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X, strat_labels)
    else:
        splitter = KFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        split_iter = splitter.split(X)

    oof = np.zeros((len(X), len(target_cols)), dtype=np.float32)
    fold_scores: list[dict[str, float]] = []

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        xtr = X.iloc[tr_idx]
        xva = X.iloc[va_idx]
        fold_stat: dict[str, float] = {}

        for target_idx, target_name in enumerate(tqdm(target_cols, desc=f"Fold {fold}")):
            ytr = Y.iloc[tr_idx][target_name]
            yva = Y.iloc[va_idx][target_name]

            if ytr.nunique() < 2:
                pred = np.full(len(yva), float(ytr.mean()), dtype=np.float32)
            else:
                if args.model == "logreg":
                    model = LogisticRegression(
                        C=args.c,
                        class_weight="balanced",
                        max_iter=1000,
                        solver="lbfgs",
                        random_state=args.seed,
                    )
                    model.fit(xtr, ytr)
                    pred = model.predict_proba(xva)[:, 1].astype(np.float32)
                else:
                    pos_rate = float(ytr.mean())
                    model = lgb.LGBMClassifier(
                        objective="binary",
                        metric="auc",
                        boosting_type="gbdt",
                        n_estimators=args.n_estimators,
                        learning_rate=args.learning_rate,
                        num_leaves=args.num_leaves,
                        feature_fraction=0.9,
                        bagging_fraction=0.9,
                        bagging_freq=1,
                        random_state=args.seed,
                        n_jobs=-1,
                        verbose=-1,
                        scale_pos_weight=(1 - pos_rate) / (pos_rate + 1e-6),
                    )
                    model.fit(
                        xtr,
                        ytr,
                        eval_set=[(xva, yva)],
                        eval_metric="auc",
                        callbacks=[lgb.early_stopping(30, verbose=False)],
                    )
                    pred = model.predict_proba(xva)[:, 1].astype(np.float32)

            oof[va_idx, target_idx] = pred
            score = roc_auc_score(yva, pred) if yva.nunique() > 1 else 0.5
            fold_stat[target_name] = float(score)

        fold_stat["macro_auc"] = float(np.mean(list(fold_stat.values())))
        fold_scores.append(fold_stat)
        print(f"Fold {fold} macro AUC: {fold_stat['macro_auc']:.6f}")

    target_scores: list[dict[str, float]] = []
    for idx, target_name in enumerate(target_cols):
        score = roc_auc_score(Y.iloc[:, idx], oof[:, idx]) if Y.iloc[:, idx].nunique() > 1 else 0.5
        target_scores.append({"target": target_name, "oof_auc": float(score)})
    macro_auc = float(np.mean([row["oof_auc"] for row in target_scores]))
    print(f"OOF macro AUC: {macro_auc:.6f}")

    out_df = pd.DataFrame(oof, columns=pred_cols)
    out_df.insert(0, "customer_id", target_df["customer_id"].values)
    out_df.to_parquet(args.out_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(fold_scores).to_csv(args.out_dir / "fold_scores.csv", index=False)
    pd.DataFrame(target_scores).sort_values("oof_auc").to_csv(args.out_dir / "target_scores.csv", index=False)
    (args.out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "oof_macro_auc": float(macro_auc),
                "fold_macro_auc": [float(f["macro_auc"]) for f in fold_scores],
                "n_features": int(len(feature_cols)),
                "base_models": list(model_names),
                "cv_strategy": args.cv_strategy,
                "meta_model": args.model,
                "pair_features_top_k": int(args.pair_features_top_k),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
