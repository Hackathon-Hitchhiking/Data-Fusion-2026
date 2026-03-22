from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from lib.layout import resolve_competition_file
from lib.submission import write_submission_like_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train stacker on full OOF predictions and apply it to test predictions")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument(
        "--train-preds",
        action="append",
        default=[],
        help="Path to train OOF/base predictions parquet. Pass multiple times.",
    )
    p.add_argument(
        "--test-preds",
        action="append",
        default=[],
        help="Path to test predictions parquet corresponding to --train-preds. Pass multiple times.",
    )
    p.add_argument(
        "--name",
        action="append",
        default=[],
        help="Optional model name for the corresponding prediction pair. If omitted, the train parquet stem is used.",
    )
    p.add_argument("--c", type=float, default=0.5)
    p.add_argument("--output", type=Path, default=Path("artifacts/stack_submission.parquet"))
    return p.parse_args()


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


def build_meta_frame(customer_ids: pd.Series, pred_paths: list[str], model_names: list[str]) -> pd.DataFrame:
    meta = pd.DataFrame({"customer_id": customer_ids})
    for model_name, path_str in zip(model_names, pred_paths):
        pred_df = pd.read_parquet(path_str)
        rename_map = {col: f"{model_name}__{col}" for col in pred_df.columns if col != "customer_id"}
        meta = meta.merge(pred_df.rename(columns=rename_map), on="customer_id", how="inner")
    meta = meta.sort_values("customer_id").reset_index(drop=True)
    return add_group_features(meta)


def main() -> None:
    args = parse_args()
    if not args.train_preds or not args.test_preds:
        raise ValueError("Both --train-preds and --test-preds are required")
    if len(args.train_preds) != len(args.test_preds):
        raise ValueError("The number of --train-preds and --test-preds paths must match")
    if args.name and len(args.name) != len(args.train_preds):
        raise ValueError("When provided, the number of --name values must match the number of prediction pairs")

    model_names = args.name if args.name else [Path(path).stem for path in args.train_preds]

    target_df = pd.read_parquet(args.target_file)
    target_cols = [c for c in target_df.columns if c.startswith("target_")]
    pred_cols = [c.replace("target_", "predict_") for c in target_cols]
    train_target = target_df[["customer_id"] + target_cols].sort_values("customer_id").reset_index(drop=True)

    train_meta = build_meta_frame(train_target["customer_id"], args.train_preds, model_names)
    train_target = train_target.merge(train_meta[["customer_id"]], on="customer_id", how="inner")
    train_target = train_target.sort_values("customer_id").reset_index(drop=True)
    train_meta = train_meta.sort_values("customer_id").reset_index(drop=True)

    test_customer_ids = pd.read_parquet(args.test_preds[0], columns=["customer_id"])["customer_id"]
    test_meta = build_meta_frame(test_customer_ids, args.test_preds, model_names)

    feature_cols = [c for c in train_meta.columns if c != "customer_id"]
    x_train = train_meta[feature_cols].astype("float32")
    x_test = test_meta[feature_cols].astype("float32")
    y_train = train_target[target_cols]

    preds = np.zeros((len(x_test), len(target_cols)), dtype=np.float64)
    for idx, target_name in enumerate(target_cols):
        y = y_train[target_name]
        if y.nunique() < 2:
            preds[:, idx] = float(y.mean())
            continue

        model = LogisticRegression(
            C=args.c,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )
        model.fit(x_train, y)
        preds[:, idx] = model.predict_proba(x_test)[:, 1].astype(np.float64)

    submit = pd.DataFrame(preds, columns=pred_cols)
    write_submission_like_sample(
        customer_ids=test_meta["customer_id"].values,
        prediction_frame=submit,
        output_path=args.output,
    )
    print(f"Saved stack submission to {args.output}")


if __name__ == "__main__":
    main()
