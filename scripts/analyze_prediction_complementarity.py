from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.layout import resolve_competition_file
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze complementarity of prediction artifacts on a common holdout")
    p.add_argument("--target-file", type=Path, default=resolve_competition_file("train_target.parquet"))
    p.add_argument("--gandalf-val", type=Path, required=True)
    p.add_argument("--stack-oof", type=Path, required=True)
    p.add_argument("--base-oof", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts/prediction_complementarity"))
    return p.parse_args()


def build_candidates(g: pd.DataFrame, s: pd.DataFrame, b: pd.DataFrame, pred_cols: list[str]) -> dict[str, pd.DataFrame]:
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
    args.out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_parquet(args.target_file)
    gandalf = normalize_prediction_columns(pd.read_parquet(args.gandalf_val))
    stack = pd.read_parquet(args.stack_oof)
    base = pd.read_parquet(args.base_oof)

    pred_cols = [c for c in gandalf.columns if c != "customer_id"]
    target_cols = [c.replace("predict_", "target_") for c in pred_cols]

    y_holdout = (
        target_df.merge(gandalf[["customer_id"]], on="customer_id", how="inner")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    gandalf = gandalf.sort_values("customer_id").reset_index(drop=True)
    stack = stack.merge(gandalf[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    base = base.merge(gandalf[["customer_id"]], on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)

    candidates = build_candidates(gandalf, stack, base, pred_cols)
    candidate_rows: list[dict[str, float | str]] = []
    for name, pred_df in candidates.items():
        candidate_rows.append({"candidate": name, "macro_auc": macro_auc(y_holdout, pred_df, target_cols)})
    candidate_df = pd.DataFrame(candidate_rows).sort_values("macro_auc", ascending=False).reset_index(drop=True)
    candidate_df.to_csv(args.out_dir / "candidate_macro_auc.csv", index=False)

    per_target_rows: list[dict[str, float | str]] = []
    oracle_rows: list[dict[str, float | str]] = []
    for target_name in target_cols:
        pred_col = target_name.replace("target_", "predict_")
        target_scores = {}
        for name, pred_df in candidates.items():
            score = macro_auc(y_holdout[[target_name]], pred_df[[pred_col]], [target_name])
            target_scores[name] = float(score)
        winner = max(target_scores, key=target_scores.get)
        row = {"target": target_name, **target_scores, "winner": winner}
        per_target_rows.append(row)
        oracle_rows.append({"target": target_name, "choice": winner, "best_auc": target_scores[winner]})

    per_target_df = pd.DataFrame(per_target_rows).sort_values("gandalf").reset_index(drop=True)
    per_target_df.to_csv(args.out_dir / "per_target_candidate_scores.csv", index=False)
    oracle_df = pd.DataFrame(oracle_rows).sort_values("best_auc").reset_index(drop=True)
    oracle_df.to_csv(args.out_dir / "oracle_targetwise_choices.csv", index=False)

    summary = {
        "holdout_rows": int(len(gandalf)),
        "best_global_candidate": candidate_df.iloc[0]["candidate"],
        "best_global_macro_auc": float(candidate_df.iloc[0]["macro_auc"]),
        "oracle_targetwise_macro_auc": float(oracle_df["best_auc"].mean()),
        "winner_counts": oracle_df["choice"].value_counts().to_dict(),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
