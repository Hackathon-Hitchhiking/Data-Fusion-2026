from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from lib.submission import write_submission_like_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Blend existing submission parquet files")
    p.add_argument("--gandalf-sub", type=Path, required=True)
    p.add_argument("--stack-sub", type=Path, required=True)
    p.add_argument("--base-sub", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--choice-file", type=Path, default=None)
    p.add_argument("--mode", choices=["global", "targetwise"], default="global")
    p.add_argument("--global-spec", type=str, default="g_90_s_10")
    return p.parse_args()


def candidate_frames(g: pd.DataFrame, s: pd.DataFrame, b: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pred_cols = [c for c in g.columns if c != "customer_id"]
    g_only = g[pred_cols]
    s_only = s[pred_cols]
    b_only = b[pred_cols]
    return {
        "gandalf": g_only,
        "stack": s_only,
        "base": b_only,
        "g_90_s_10": 0.9 * g_only + 0.1 * s_only,
        "g_85_s_15": 0.85 * g_only + 0.15 * s_only,
        "g_80_s_20": 0.8 * g_only + 0.2 * s_only,
        "g_90_b_10": 0.9 * g_only + 0.1 * b_only,
        "g_80_s_10_b_10": 0.8 * g_only + 0.1 * s_only + 0.1 * b_only,
    }


def main() -> None:
    args = parse_args()

    g = pd.read_parquet(args.gandalf_sub)
    s = pd.read_parquet(args.stack_sub)
    b = pd.read_parquet(args.base_sub)

    if list(g.columns) != list(s.columns) or list(g.columns) != list(b.columns):
        raise ValueError("All submission files must have the same columns in the same order")
    if not g["customer_id"].equals(s["customer_id"]) or not g["customer_id"].equals(b["customer_id"]):
        raise ValueError("All submission files must have identical customer_id order")

    candidates = candidate_frames(g, s, b)
    pred_cols = [c for c in g.columns if c != "customer_id"]
    submit = pd.DataFrame({"customer_id": g["customer_id"].astype("int32").values})

    if args.mode == "global":
        if args.global_spec not in candidates:
            raise ValueError(f"Unknown global blend spec: {args.global_spec}")
        chosen = candidates[args.global_spec]
        for col in pred_cols:
            submit[col] = chosen[col].astype("float64").values
    else:
        if args.choice_file is None:
            raise ValueError("--choice-file is required in targetwise mode")
        choice_df = pd.read_csv(args.choice_file)
        choice_map = dict(zip(choice_df["target"], choice_df["choice"]))
        for pred_col in pred_cols:
            target_name = pred_col.replace("predict_", "target_")
            blend_name = choice_map[target_name]
            submit[pred_col] = candidates[blend_name][pred_col].astype("float64").values

    write_submission_like_sample(
        customer_ids=submit["customer_id"].values,
        prediction_frame=submit[pred_cols],
        output_path=args.output,
    )
    print(f"Saved blend submission to {args.output}")


if __name__ == "__main__":
    main()
