from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from lib.meta_views import (
    ROOT,
    align_prediction_frames,
    build_target_score_frame,
    load_holdout_targets,
    load_prediction_frame,
    load_raw_feature_block,
    load_top_extra_features,
    target_to_pred,
)
from lib.metrics import macro_auc


DEFAULT_TABM_VAL = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
DEFAULT_GANDALF_VAL = ROOT / "artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet"
DEFAULT_SPECIALIST_VAL = (
    ROOT / "artifacts/catboost_extended_specialists_gandalf_h12_router/specialist_val_predictions.parquet"
)
DEFAULT_ARTIFACT_DIR = ROOT / "artifacts/object_level_gate_v1"
DEFAULT_TARGETS = ["target_3_1", "target_10_1", "target_9_7", "target_2_4", "target_9_6"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run instance-wise gating between TabM and specialists.")
    parser.add_argument("--tabm-val", type=Path, default=DEFAULT_TABM_VAL)
    parser.add_argument("--gandalf-val", type=Path, default=DEFAULT_GANDALF_VAL)
    parser.add_argument("--specialist-val", type=Path, default=DEFAULT_SPECIALIST_VAL)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--targets", nargs="*", default=DEFAULT_TARGETS)
    parser.add_argument("--raw-top-k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gate-margin", type=float, default=0.01)
    parser.add_argument("--alphas", nargs="*", type=float, default=[0.25, 0.50, 0.75, 1.0])
    parser.add_argument("--source-model", choices=["spec", "gandalf"], default="spec")
    return parser.parse_args()


def build_gate_features(score_df: pd.DataFrame, raw_df: pd.DataFrame, *, source_model: str) -> pd.DataFrame:
    merged = score_df.copy()
    source_score_col = f"{source_model}_score"
    merged["source_minus_tabm"] = merged[source_score_col] - merged["tabm_score"]
    merged["gandalf_minus_tabm"] = merged["gandalf_score"] - merged["tabm_score"]
    merged["tabm_minus_0p5"] = np.abs(merged["tabm_score"] - 0.5)
    merged["source_minus_0p5"] = np.abs(merged[source_score_col] - 0.5)
    merged["tabm_source_gap"] = np.abs(merged[source_score_col] - merged["tabm_score"])
    if len(raw_df.columns) > 1:
        merged = merged.merge(raw_df, on="customer_id", how="left")
    return merged


def crossfit_gate(features: pd.DataFrame, y_gate: np.ndarray, *, seed: int, folds: int) -> np.ndarray:
    x = features.drop(columns=["customer_id"]).to_numpy(dtype=np.float32)
    oof = np.zeros(len(features), dtype=np.float32)
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    if np.unique(y_gate).size < 2:
        oof[:] = float(y_gate.mean())
        return oof
    for train_idx, val_idx in splitter.split(x, y_gate):
        train_y = y_gate[train_idx]
        if np.unique(train_y).size < 2:
            oof[val_idx] = float(train_y.mean())
            continue
        model = LogisticRegression(
            max_iter=600,
            class_weight="balanced",
            random_state=seed,
        )
        model.fit(x[train_idx], train_y)
        oof[val_idx] = model.predict_proba(x[val_idx])[:, 1].astype(np.float32)
    return oof


def main() -> None:
    args = parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    named_frames = {
        "tabm": load_prediction_frame(args.tabm_val, args.targets),
        "gandalf": load_prediction_frame(args.gandalf_val, args.targets),
        "spec": load_prediction_frame(args.specialist_val, args.targets),
    }
    named_frames = align_prediction_frames(named_frames)
    customer_ids = named_frames["tabm"]["customer_id"]
    y_df = load_holdout_targets(customer_ids, args.targets)
    raw_cols = load_top_extra_features(args.raw_top_k)
    raw_df = load_raw_feature_block(customer_ids, raw_cols)

    base_pred_df = named_frames["tabm"].copy()
    gate_summary_rows: list[dict[str, object]] = []
    gate_prob_map: dict[str, np.ndarray] = {}

    for target_name in args.targets:
        score_df = build_target_score_frame(named_frames, target_name)
        feature_df = build_gate_features(score_df, raw_df, source_model=args.source_model).sort_values("customer_id").reset_index(drop=True)
        y = y_df[target_name].to_numpy(dtype=np.int8)
        tabm_score = feature_df["tabm_score"].to_numpy(dtype=np.float32)
        source_score = feature_df[f"{args.source_model}_score"].to_numpy(dtype=np.float32)
        source_err = np.abs(y - source_score)
        tabm_err = np.abs(y - tabm_score)
        y_gate = (source_err + args.gate_margin < tabm_err).astype(np.int8)
        gate_prob = crossfit_gate(feature_df, y_gate, seed=args.seed, folds=args.folds)
        gate_prob_map[target_name] = gate_prob
        gate_summary_rows.append(
            {
                "target": target_name,
                "gate_positive_rate": float(y_gate.mean()),
                "gate_prob_mean": float(gate_prob.mean()),
                "tabm_mean": float(tabm_score.mean()),
                "source_model": args.source_model,
                "source_mean": float(source_score.mean()),
            }
        )

    pd.DataFrame(gate_summary_rows).to_csv(args.artifact_dir / "per_target_gate_stats.csv", index=False)

    candidate_rows: list[dict[str, object]] = []
    best_alpha = None
    best_score = -np.inf
    best_pred_df = None
    for alpha in args.alphas:
        pred_df = base_pred_df.copy()
        for target_name in args.targets:
            pred_col = target_to_pred(target_name)
            tabm_score = pred_df[pred_col].to_numpy(dtype=np.float32)
            source_score = named_frames[args.source_model][pred_col].to_numpy(dtype=np.float32)
            gate_prob = gate_prob_map[target_name]
            blended = np.clip(tabm_score + float(alpha) * gate_prob * (source_score - tabm_score), 1e-6, 1.0 - 1e-6)
            pred_df[pred_col] = blended.astype(np.float32)
        score = macro_auc(y_df, pred_df, args.targets)
        candidate_rows.append({"alpha": float(alpha), "macro_auc_subset": float(score)})
        if score > best_score:
            best_score = score
            best_alpha = float(alpha)
            best_pred_df = pred_df.copy()

    pd.DataFrame(candidate_rows).to_csv(args.artifact_dir / "candidate_subset_auc.csv", index=False)
    if best_pred_df is not None:
        best_pred_df.to_parquet(args.artifact_dir / "best_candidate_val_predictions.parquet", index=False)

    summary = {
        "targets": args.targets,
        "best_alpha": best_alpha,
        "best_subset_macro_auc": float(best_score),
        "base_subset_macro_auc": float(macro_auc(y_df, base_pred_df, args.targets)),
        "raw_top_k": int(args.raw_top_k),
        "gate_margin": float(args.gate_margin),
        "source_model": args.source_model,
    }
    (args.artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
