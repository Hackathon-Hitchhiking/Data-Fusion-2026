from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from lib.data_loading import read_competition_parquet
from lib.layout import resolve_data_dir
from lib.submission import write_submission_like_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference with trained per-target LGBM CV ensemble")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--models-dir", type=Path, default=Path("artifacts/lgbm_cv"))
    p.add_argument("--feature-set", choices=["main", "all"], default="all")
    p.add_argument("--output", type=Path, default=Path("artifacts/submission_lgbm.parquet"))
    return p.parse_args()


def load_extra_feature_spec(extra_features_path: str | None) -> tuple[list[str] | None, dict[str, list[str]]]:
    if not extra_features_path:
        return None, {}

    raw = json.loads(Path(extra_features_path).read_text())
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


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_data_dir(args.data_dir)
    feature_manifest_path = args.models_dir / "feature_manifest.json"
    feature_manifest = json.loads(feature_manifest_path.read_text()) if feature_manifest_path.exists() else {}

    test_main = read_competition_parquet("test_main_features.parquet", data_dir=args.data_dir)
    main_feature_cols = [c for c in test_main.columns if c != "customer_id"]
    selected_extra_cols: list[str] = feature_manifest.get("selected_extra_columns", [])
    target_extra_map: dict[str, list[str]] = feature_manifest.get("target_extra_features", {})
    if args.feature_set == "all" and not selected_extra_cols:
        selected_extra_cols, target_extra_map_from_file = load_extra_feature_spec(feature_manifest.get("extra_features_file"))
        if selected_extra_cols is not None:
            dropped = set(feature_manifest.get("dropped_for_missing_ratio", [])) | set(
                feature_manifest.get("dropped_duplicate_columns", [])
            )
            selected_extra_cols = [c for c in selected_extra_cols if c not in dropped]
        if not target_extra_map:
            selected_lookup = set(selected_extra_cols or [])
            target_extra_map = {
                target_name: [col for col in feature_names if col in selected_lookup]
                for target_name, feature_names in target_extra_map_from_file.items()
            }

    if args.feature_set == "all":
        extra_columns = ["customer_id"] + selected_extra_cols if selected_extra_cols else None
        test_extra = read_competition_parquet("test_extra_features.parquet", data_dir=args.data_dir, columns=extra_columns)
        test = test_main.merge(test_extra, on="customer_id", how="inner")
    else:
        test = test_main

    target = read_competition_parquet("train_target.parquet", data_dir=args.data_dir)
    target_cols = [c for c in target.columns if c.startswith("target_")]
    pred_cols = [c.replace("target_", "predict_") for c in target_cols]

    feature_cols = [c for c in test.columns if c != "customer_id"]
    global_extra_cols = [c for c in feature_cols if c not in set(main_feature_cols)]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]

    for c in cat_cols:
        test[c] = test[c].astype("float32").astype("Int32")
    for c in feature_cols:
        if c not in cat_cols:
            test[c] = pd.to_numeric(test[c], errors="coerce").astype("float32")

    models_index = json.loads((args.models_dir / "models_index.json").read_text())

    preds = np.zeros((len(test), len(target_cols)), dtype=np.float64)
    for i, target_name in enumerate(target_cols):
        current_extra_cols = target_extra_map.get(target_name, global_extra_cols)
        current_feature_cols = main_feature_cols + current_extra_cols
        X_test = test[current_feature_cols]
        fold_models = models_index[target_name]
        fold_pred = np.zeros(len(test), dtype=np.float32)
        for mp in fold_models:
            with open(mp, "rb") as f:
                model = pickle.load(f)
            if isinstance(model, dict) and "constant_pred" in model:
                pred = np.full(len(test), model["constant_pred"], dtype=np.float64)
            else:
                pred = model.predict_proba(X_test)[:, 1].astype(np.float64)
            fold_pred += pred / len(fold_models)
        preds[:, i] = fold_pred

    submit = pd.DataFrame(preds, columns=pred_cols)
    write_submission_like_sample(
        customer_ids=test["customer_id"].values,
        prediction_frame=submit,
        output_path=args.output,
    )
    print(f"Saved submission to {args.output}")


if __name__ == "__main__":
    main()
