from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.layout import project_root
from lib.submission import normalize_prediction_columns, write_submission_like_sample


ROOT = project_root()
DEFAULT_SUBMISSION_DIR = ROOT / "output" / "submissions"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build test submissions from per-target router decisions.")
    p.add_argument("--base-submission", type=Path, required=True)
    p.add_argument("--router-file", type=Path, required=True)
    p.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--name-prefix", type=str, required=True)
    p.add_argument("--catboost-test", type=Path, default=None)
    p.add_argument("--lgbm-test", type=Path, default=None)
    p.add_argument("--autogluon-test", type=Path, default=None)
    p.add_argument("--modes", type=str, default="alpha_conservative,alpha_balanced,alpha_family_pick")
    return p.parse_args()


def load_frame(path: Path | None) -> pd.DataFrame | None:
    if path is None or not path.exists():
        return None
    return normalize_prediction_columns(pd.read_parquet(path)).sort_values("customer_id").reset_index(drop=True)


def main() -> None:
    args = parse_args()
    args.submission_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    base = normalize_prediction_columns(pd.read_parquet(args.base_submission)).sort_values("customer_id").reset_index(drop=True)
    router = pd.read_csv(args.router_file)
    family_frames = {
        name: frame
        for name, frame in {
            "catboost": load_frame(args.catboost_test),
            "lgbm": load_frame(args.lgbm_test),
            "autogluon": load_frame(args.autogluon_test),
        }.items()
        if frame is not None
    }

    for name, frame in family_frames.items():
        if not base["customer_id"].equals(frame["customer_id"]):
            raise ValueError(f"{name} test frame customer_id order mismatch")

    summary_rows: list[dict[str, str]] = []
    pred_cols = [c for c in base.columns if c != "customer_id"]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    for mode in modes:
        candidate = base[["customer_id"] + pred_cols].copy()
        applied_rows = []
        for row in router.to_dict("records"):
            family = row["best_family"]
            alpha = float(row[mode])
            if family == "gandalf" or alpha <= 0:
                continue
            frame = family_frames.get(family)
            if frame is None:
                continue
            pred_col = str(row["target"]).replace("target_", "predict_")
            if pred_col not in frame.columns or frame[pred_col].isna().all():
                continue
            candidate[pred_col] = ((1.0 - alpha) * base[pred_col] + alpha * frame[pred_col]).astype("float32")
            applied_rows.append({"target": row["target"], "family": family, "alpha": alpha, "mode": mode})

        out_name = f"{args.name_prefix}_{mode.replace('alpha_', '')}_submission.parquet"
        out_path = args.submission_dir / out_name
        write_submission_like_sample(
            customer_ids=candidate["customer_id"],
            prediction_frame=candidate[["customer_id"] + pred_cols],
            output_path=out_path,
        )
        pd.DataFrame(applied_rows).to_csv(args.out_dir / f"{mode}_applied_targets.csv", index=False)
        summary_rows.append({"mode": mode, "submission_path": str(out_path), "applied_target_count": str(len(applied_rows))})

    (args.out_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2))
    print(json.dumps(summary_rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
