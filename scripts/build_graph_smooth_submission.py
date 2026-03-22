from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.layout import resolve_competition_file, resolve_sample_submit
from lib.submission import normalize_prediction_columns, write_submission_like_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build graph-smoothed submission from existing GANDALF and stack submissions.")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument("--gandalf-submission", type=Path, default=Path("output/submissions/gpu_gandalf_multitask_750k_oof_0.8100369735_submission.parquet"))
    p.add_argument("--stack-submission", type=Path, default=Path("output/submissions/stack_top100_self_c005_full_750k_oof_0.8015127605_submission.parquet"))
    p.add_argument("--sample-submit", type=Path, default=resolve_sample_submit())
    p.add_argument("--out-dir", type=Path, default=Path("output/submissions"))
    p.add_argument("--alpha", type=float, default=0.20)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--weight-gandalf", type=float, default=0.90)
    p.add_argument("--weight-stack", type=float, default=0.10)
    return p.parse_args()


def build_label_graph_weights(target_df: pd.DataFrame, target_cols: list[str], top_k: int) -> np.ndarray:
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


def graph_smooth(base_pred: pd.DataFrame, target_df: pd.DataFrame, target_cols: list[str], weights: np.ndarray, alpha: float) -> pd.DataFrame:
    pred_cols = [t.replace("target_", "predict_") for t in target_cols]
    p = base_pred[pred_cols].to_numpy(dtype=np.float64)
    priors = target_df[target_cols].mean(axis=0).to_numpy(dtype=np.float64)
    logits = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
    neighbor_effect = (p - priors) @ weights.T
    smoothed = 1.0 / (1.0 + np.exp(-(logits + alpha * neighbor_effect)))
    out = pd.DataFrame({"customer_id": base_pred["customer_id"].astype("int32").values})
    for i, pred_col in enumerate(pred_cols):
        out[pred_col] = smoothed[:, i].astype("float64")
    return out


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_parquet(args.target_file)
    gandalf = normalize_prediction_columns(pd.read_parquet(args.gandalf_submission)).sort_values("customer_id").reset_index(drop=True)
    stack = normalize_prediction_columns(pd.read_parquet(args.stack_submission)).sort_values("customer_id").reset_index(drop=True)
    sample = pd.read_parquet(args.sample_submit)

    pred_cols = [c for c in sample.columns if c != "customer_id"]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]
    weights = build_label_graph_weights(target_df, target_cols, top_k=args.top_k)
    if abs(args.weight_gandalf + args.weight_stack - 1.0) > 1e-9:
        raise ValueError("weight_gandalf + weight_stack must equal 1.0")

    blend = pd.concat(
        [
            gandalf[["customer_id"]].copy(),
            args.weight_gandalf * gandalf[pred_cols].reset_index(drop=True)
            + args.weight_stack * stack[pred_cols].reset_index(drop=True),
        ],
        axis=1,
    )
    smoothed = graph_smooth(blend, target_df, target_cols, weights, alpha=args.alpha)

    g_part = int(round(args.weight_gandalf * 100))
    s_part = int(round(args.weight_stack * 100))
    file_name = f"gandalf_{g_part:02d}_stack_{s_part:02d}_graphsmooth_a{args.alpha:.2f}_k{args.top_k}_submission.parquet"
    out_path = args.out_dir / file_name
    submission = write_submission_like_sample(
        customer_ids=sample["customer_id"].values,
        prediction_frame=smoothed[pred_cols],
        output_path=out_path,
        sample_submit_path=args.sample_submit,
    )

    meta = {
        "alpha": args.alpha,
        "top_k": args.top_k,
        "weight_gandalf": args.weight_gandalf,
        "weight_stack": args.weight_stack,
        "source_gandalf_submission": str(args.gandalf_submission),
        "source_stack_submission": str(args.stack_submission),
        "output_submission": str(out_path),
    }
    (args.out_dir / file_name.replace("_submission.parquet", "_meta.json")).write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
