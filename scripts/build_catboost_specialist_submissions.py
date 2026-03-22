from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from lib.layout import project_root, resolve_data_dir
from lib.submission import normalize_prediction_columns, write_submission_like_sample


ROOT = project_root()
DATA_DIR = resolve_data_dir()
TOP_EXTRA_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
GANDALF_SUBMISSION = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "submission.parquet"
STACK_SUBMISSION = ROOT / "artifacts" / "stack_top100_self_c005_full_750k" / "submission.parquet"
BASE_SUBMISSION = ROOT / "artifacts" / "full_lgbm_top100_750k_2fold" / "submission.parquet"
BASE_OOF_FILE = ROOT / "artifacts" / "full_lgbm_top100_750k_2fold" / "oof_predictions.parquet"
STACK_OOF_FILE = ROOT / "artifacts" / "stack_top100_self_c005_full_750k" / "oof_predictions.parquet"
DEFAULT_PILOT_PER_TARGET = ROOT / "artifacts" / "catboost_extended_specialists_gandalf_h12_fast" / "per_target_auc.csv"
DEFAULT_GANDALF_TARGET_SCORE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "target_scores.csv"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "catboost_specialist_submissions_v1"
DEFAULT_SUBMISSION_DIR = ROOT / "output" / "submissions"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train full-data CatBoost specialists and build blended submissions.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--submission-dir", type=Path, default=DEFAULT_SUBMISSION_DIR)
    p.add_argument("--gandalf-submission", type=Path, default=GANDALF_SUBMISSION)
    p.add_argument("--stack-submission", type=Path, default=STACK_SUBMISSION)
    p.add_argument("--base-submission", type=Path, default=BASE_SUBMISSION)
    p.add_argument("--stack-oof", type=Path, default=STACK_OOF_FILE)
    p.add_argument("--base-oof", type=Path, default=BASE_OOF_FILE)
    p.add_argument("--pilot-per-target", type=Path, default=DEFAULT_PILOT_PER_TARGET)
    p.add_argument("--gandalf-target-scores", type=Path, default=DEFAULT_GANDALF_TARGET_SCORE)
    p.add_argument("--hardest-count", type=int, default=12)
    p.add_argument("--boosting-type", choices=["Plain", "Ordered"], default="Plain")
    p.add_argument("--iterations", type=int, default=150)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--l2-leaf-reg", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--thread-count", type=int, default=4)
    p.add_argument("--used-ram-limit", type=str, default="8gb")
    p.add_argument("--max-ctr-complexity", type=int, default=1)
    p.add_argument("--one-hot-max-size", type=int, default=32)
    p.add_argument("--border-count", type=int, default=64)
    p.add_argument("--alpha-grid", type=str, default="0.05,0.10,0.20,0.30")
    return p.parse_args()


def frame_mem_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def load_feature_frame() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    train_main = pd.read_parquet(DATA_DIR / "train_main_features.parquet")
    train_extra = pd.read_parquet(DATA_DIR / "train_extra_features.parquet")
    train_target = pd.read_parquet(DATA_DIR / "train_target.parquet")
    test_main = pd.read_parquet(DATA_DIR / "test_main_features.parquet")
    test_extra = pd.read_parquet(DATA_DIR / "test_extra_features.parquet")
    top_extra = json.loads(TOP_EXTRA_FILE.read_text())

    labeled = train_main.merge(train_target, on="customer_id", how="inner")
    labeled = labeled.merge(train_extra[["customer_id"] + top_extra], on="customer_id", how="left")
    test = test_main.merge(test_extra[["customer_id"] + top_extra], on="customer_id", how="left")

    target_cols = [c for c in labeled.columns if c.startswith("target_")]
    feature_cols = [c for c in labeled.columns if c not in ["customer_id"] + target_cols]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    return labeled, test, feature_cols, cat_cols, target_cols


def add_group_score_features(frame: pd.DataFrame, prefix: str, pred_cols: list[str]) -> pd.DataFrame:
    out = frame.copy()
    groups: dict[str, list[str]] = {}
    for col in pred_cols:
        target_name = col.replace("predict_", "target_")
        group = target_name.split("_")[1]
        groups.setdefault(group, []).append(col)
    for group, cols in groups.items():
        out[f"{prefix}group_{group}_sum"] = out[cols].sum(axis=1)
        out[f"{prefix}group_{group}_max"] = out[cols].max(axis=1)
    return out


def hardest_targets(score_file: Path, count: int) -> list[str]:
    return pd.read_csv(score_file).sort_values("oof_auc").head(count)["target"].tolist()


def tuned_iterations_map(pilot_file: Path) -> dict[str, int]:
    if not pilot_file.exists():
        return {}
    df = pd.read_csv(pilot_file)
    return {row["target"]: int(row["best_iteration"]) + 1 for _, row in df.iterrows()}


def prepare_meta_frames(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    stack_train = pd.read_parquet(args.stack_oof).sort_values("customer_id").reset_index(drop=True)
    base_train = pd.read_parquet(args.base_oof).sort_values("customer_id").reset_index(drop=True)
    stack_test = normalize_prediction_columns(pd.read_parquet(args.stack_submission)).sort_values("customer_id").reset_index(drop=True)
    base_test = normalize_prediction_columns(pd.read_parquet(args.base_submission)).sort_values("customer_id").reset_index(drop=True)
    return stack_train, base_train, stack_test, base_test


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.submission_dir.mkdir(parents=True, exist_ok=True)

    alpha_grid = [float(x) for x in args.alpha_grid.split(",") if x.strip()]
    hard_targets = hardest_targets(args.gandalf_target_scores, args.hardest_count)
    tuned_iterations = tuned_iterations_map(args.pilot_per_target)

    labeled, test, raw_feature_cols, cat_cols, target_cols = load_feature_frame()
    stack_train, base_train, stack_test, base_test = prepare_meta_frames(args)

    for frame, prefix in [(stack_train, "stack_"), (base_train, "base_"), (stack_test, "stack_"), (base_test, "base_")]:
        rename_map = {c: f"{prefix}{c}" for c in frame.columns if c.startswith("predict_")}
        frame.rename(columns=rename_map, inplace=True)
        score_cols = [c for c in frame.columns if c.startswith(prefix)]
        for col in score_cols:
            if col != "customer_id":
                frame[col] = frame[col].astype("float32")

    stack_pred_cols = [c for c in stack_train.columns if c.startswith("stack_predict_")]
    base_pred_cols = [c for c in base_train.columns if c.startswith("base_predict_")]
    stack_train = add_group_score_features(stack_train, "stack_", stack_pred_cols)
    stack_test = add_group_score_features(stack_test, "stack_", stack_pred_cols)
    base_train = add_group_score_features(base_train, "base_", base_pred_cols)
    base_test = add_group_score_features(base_test, "base_", base_pred_cols)

    train_feat = (
        labeled[["customer_id"] + raw_feature_cols]
        .merge(stack_train, on="customer_id", how="left")
        .merge(base_train, on="customer_id", how="left")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )
    test_feat = (
        test[["customer_id"] + raw_feature_cols]
        .merge(stack_test, on="customer_id", how="left")
        .merge(base_test, on="customer_id", how="left")
        .sort_values("customer_id")
        .reset_index(drop=True)
    )

    model_feature_cols = [c for c in train_feat.columns if c != "customer_id"]
    train_X = train_feat[model_feature_cols].copy()
    test_X = test_feat[model_feature_cols].copy()

    for col in cat_cols:
        train_X[col] = train_X[col].fillna(-1).astype("int32")
        test_X[col] = test_X[col].fillna(-1).astype("int32")

    num_cols = [c for c in model_feature_cols if c not in cat_cols]
    for col in num_cols:
        train_X[col] = pd.to_numeric(train_X[col], errors="coerce").astype("float32")
        test_X[col] = pd.to_numeric(test_X[col], errors="coerce").astype("float32")

    print(
        {
            "stage": "full_specialist_memory_ready",
            "hard_targets": hard_targets,
            "train_x_mb": round(frame_mem_mb(train_X), 2),
            "test_x_mb": round(frame_mem_mb(test_X), 2),
            "feature_count": len(model_feature_cols),
            "used_ram_limit": args.used_ram_limit,
        },
        flush=True,
    )

    cat_feature_indices = [train_X.columns.get_loc(c) for c in cat_cols]
    test_pool = Pool(test_X, cat_features=cat_feature_indices)

    gandalf_base = normalize_prediction_columns(pd.read_parquet(args.gandalf_submission)).sort_values("customer_id").reset_index(drop=True)
    pred_cols = [c for c in gandalf_base.columns if c != "customer_id"]
    specialist_pred = gandalf_base[["customer_id"] + pred_cols].copy()
    per_target_rows: list[dict[str, int | str]] = []

    for idx, target_name in enumerate(hard_targets, start=1):
        pred_col = target_name.replace("target_", "predict_")
        y_train = labeled.sort_values("customer_id").reset_index(drop=True)[target_name].to_numpy()
        iterations = tuned_iterations.get(target_name, args.iterations)
        print(
            {"stage": "full_specialist_target_start", "target": target_name, "index": idx, "iterations": iterations},
            flush=True,
        )

        train_pool = Pool(train_X, y_train, cat_features=cat_feature_indices)
        model = CatBoostClassifier(
            loss_function="Logloss",
            iterations=iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.seed,
            auto_class_weights="Balanced",
            verbose=False,
            boosting_type=args.boosting_type,
            allow_writing_files=False,
            thread_count=args.thread_count,
            used_ram_limit=args.used_ram_limit,
            max_ctr_complexity=args.max_ctr_complexity,
            one_hot_max_size=args.one_hot_max_size,
            border_count=args.border_count,
        )
        model.fit(train_pool)
        test_pred = model.predict_proba(test_pool)[:, 1].astype("float32")
        specialist_pred[pred_col] = test_pred
        per_target_rows.append({"target": target_name, "iterations": int(iterations), "boosting_type": args.boosting_type})
        print({"stage": "full_specialist_target_done", "target": target_name}, flush=True)

        del model, train_pool, y_train, test_pred
        gc.collect()

    specialist_pred.to_parquet(args.out_dir / "specialist_test_predictions.parquet", index=False)
    pd.DataFrame(per_target_rows).to_csv(args.out_dir / "trained_targets.csv", index=False)

    summary_rows: list[dict[str, float | str]] = []
    for alpha in alpha_grid:
        candidate = gandalf_base[["customer_id"] + pred_cols].copy()
        for target_name in hard_targets:
            pred_col = target_name.replace("target_", "predict_")
            candidate[pred_col] = (
                (1.0 - alpha) * gandalf_base[pred_col].to_numpy(dtype=np.float32)
                + alpha * specialist_pred[pred_col].to_numpy(dtype=np.float32)
            ).astype("float32")
        name = f"gandalf_catboost12_plain_a{alpha:.2f}".replace(".", "p")
        submission_path = args.submission_dir / f"{name}_submission.parquet"
        write_submission_like_sample(
            customer_ids=candidate["customer_id"],
            prediction_frame=candidate[["customer_id"] + pred_cols],
            output_path=submission_path,
        )
        summary_rows.append({"candidate": name, "alpha": alpha, "submission_path": str(submission_path)})
        print({"stage": "submission_written", "candidate": name, "alpha": alpha}, flush=True)

    (args.out_dir / "summary.json").write_text(json.dumps({"hard_targets": hard_targets, "candidates": summary_rows}, indent=2))
    print(json.dumps({"hard_targets": hard_targets, "candidates": summary_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
