from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a hybrid per-target extra feature map")
    p.add_argument("--global-features-file", type=Path, required=True)
    p.add_argument("--target-features-file", type=Path, required=True)
    p.add_argument("--target-scores-file", type=Path, required=True)
    p.add_argument("--out-file", type=Path, required=True)
    p.add_argument("--hard-threshold", type=float, default=0.75)
    p.add_argument("--global-top-k", type=int, default=30)
    p.add_argument("--target-top-k", type=int, default=80)
    return p.parse_args()


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def main() -> None:
    args = parse_args()
    args.out_file.parent.mkdir(parents=True, exist_ok=True)

    global_features = json.loads(args.global_features_file.read_text())
    target_features = json.loads(args.target_features_file.read_text())
    target_scores = pd.read_csv(args.target_scores_file)

    if not isinstance(global_features, list):
        raise ValueError("global-features-file must contain a JSON list")
    if not isinstance(target_features, dict):
        raise ValueError("target-features-file must contain a JSON object target->feature_list")

    global_top = global_features[: args.global_top_k]
    hard_targets = set(target_scores.loc[target_scores["oof_auc"] < args.hard_threshold, "target"].tolist())
    all_targets = sorted(set(target_scores["target"].tolist()) | set(target_features.keys()))

    hybrid_map: dict[str, list[str]] = {}
    for target_name in all_targets:
        if target_name in hard_targets:
            target_top = target_features.get(target_name, [])[: args.target_top_k]
            hybrid_map[target_name] = dedupe_keep_order(target_top + global_top)
        else:
            hybrid_map[target_name] = global_features

    args.out_file.write_text(json.dumps(hybrid_map, indent=2))
    print(f"Saved hybrid map to {args.out_file}")


if __name__ == "__main__":
    main()
