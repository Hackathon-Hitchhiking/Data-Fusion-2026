from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lib.feature_selection_v1 import (
    WEAKEST_TARGETS,
    build_composite_feature_scores_df,
    build_feature_inventory_df,
    build_final_feature_set_payload,
    build_global_extra_scores_df,
    build_mi_candidate_pool_from_lgbm,
    build_weak_target_rescue_scores_df,
    ensure_out_dir,
    materialize_selected_feature_dataset,
    run_lgbm_gain_scan_df,
    run_mi_feature_scan_df,
    run_missingness_signal_scan_df,
    run_redundancy_pruning_df,
    run_wrapper_eval_df,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full feature_selection_v1 pipeline.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--default-n-final-extra", type=int, default=180)
    parser.add_argument("--candidate-sizes", default="100,140,180,220,260")
    parser.add_argument("--corr-threshold", type=float, default=0.98)
    parser.add_argument("--preselect-n", type=int, default=300)
    parser.add_argument("--materialize-dataset", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Reuse completed step artifacts when available.")
    parser.add_argument("--mi-max-rows-per-target", type=int, default=None)
    parser.add_argument("--mi-n-neighbors", type=int, default=3)
    parser.add_argument("--mi-feature-block-size", type=int, default=256)
    parser.add_argument("--mi-global-top-n", type=int, default=600)
    parser.add_argument("--mi-weak-top-k", type=int, default=80)
    parser.add_argument("--mi-full-feature-scan", action="store_true")
    parser.add_argument("--mi-weak-targets-only", action="store_true")
    parser.add_argument("--wrapper-weak-targets-only", action="store_true")
    return parser.parse_args()


def csv_step(
    path: Path,
    *,
    resume: bool,
    build_fn,
    step_name: str,
) -> pd.DataFrame:
    if resume and path.exists():
        df = pd.read_csv(path)
        print({"stage": f"{step_name}_resume", "rows": int(len(df)), "path": str(path)}, flush=True)
        return df
    df = build_fn()
    df.to_csv(path, index=False)
    return df


def json_step(path: Path, *, resume: bool, build_fn, step_name: str) -> dict:
    if resume and path.exists():
        payload = json.loads(path.read_text())
        print({"stage": f"{step_name}_resume", "path": str(path)}, flush=True)
        return payload
    payload = build_fn()
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
    return payload


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    mi_config = None
    if args.mi_max_rows_per_target is not None:
        from lib.feature_selection_v1 import ScanConfig

        mi_config = ScanConfig(
            seed=42,
            negative_ratio=4.0,
            max_rows_per_target=args.mi_max_rows_per_target,
            min_positive_count=50,
        )

    feature_inventory = csv_step(
        out_dir / "feature_inventory.csv",
        resume=args.resume,
        build_fn=lambda: build_feature_inventory_df(data_dir=args.data_dir),
        step_name="feature_inventory",
    )
    print({"stage": "feature_inventory_done", "rows": int(len(feature_inventory))}, flush=True)

    lgbm_df = csv_step(
        out_dir / "lgbm_gain_per_target.csv",
        resume=args.resume,
        build_fn=lambda: run_lgbm_gain_scan_df(data_dir=args.data_dir),
        step_name="lgbm_gain",
    )
    print({"stage": "lgbm_gain_done", "rows": int(len(lgbm_df))}, flush=True)

    mi_feature_subset = None
    if not args.mi_full_feature_scan:
        mi_feature_subset = build_mi_candidate_pool_from_lgbm(
            lgbm_df,
            weak_targets=WEAKEST_TARGETS,
            global_top_n=args.mi_global_top_n,
            weak_top_k=args.mi_weak_top_k,
        )
        print(
            {
                "stage": "mi_candidate_pool_ready",
                "candidate_features": int(len(mi_feature_subset)),
                "global_top_n": int(args.mi_global_top_n),
                "weak_top_k": int(args.mi_weak_top_k),
            },
            flush=True,
        )
    mi_target_subset = WEAKEST_TARGETS if args.mi_weak_targets_only else None

    mi_df = csv_step(
        out_dir / "mi_per_target.csv",
        resume=args.resume,
        build_fn=lambda: run_mi_feature_scan_df(
            data_dir=args.data_dir,
            config=mi_config,
            target_subset=mi_target_subset,
            extra_feature_subset=mi_feature_subset,
            n_neighbors=args.mi_n_neighbors,
            feature_block_size=args.mi_feature_block_size,
        ),
        step_name="mi",
    )
    print({"stage": "mi_done", "rows": int(len(mi_df))}, flush=True)

    weak_df = csv_step(
        out_dir / "weak_target_rescue_scores.csv",
        resume=args.resume,
        build_fn=lambda: build_weak_target_rescue_scores_df(lgbm_df, mi_df, weak_targets=WEAKEST_TARGETS),
        step_name="weak_rescue",
    )
    print({"stage": "weak_rescue_done", "rows": int(len(weak_df))}, flush=True)

    missingness_per_target_path = out_dir / "missingness_signal_per_target.csv"
    missingness_df_path = out_dir / "missingness_signal_scores.csv"
    if args.resume and missingness_per_target_path.exists() and missingness_df_path.exists():
        missingness_per_target = pd.read_csv(missingness_per_target_path)
        missingness_df = pd.read_csv(missingness_df_path)
        print(
            {
                "stage": "missingness_resume",
                "rows": int(len(missingness_df)),
                "paths": [str(missingness_per_target_path), str(missingness_df_path)],
            },
            flush=True,
        )
    else:
        missingness_per_target, missingness_df = run_missingness_signal_scan_df(
            data_dir=args.data_dir,
            weak_targets=WEAKEST_TARGETS,
        )
        missingness_per_target.to_csv(missingness_per_target_path, index=False)
        missingness_df.to_csv(missingness_df_path, index=False)
    print({"stage": "missingness_done", "rows": int(len(missingness_df))}, flush=True)

    global_df = csv_step(
        out_dir / "global_extra_scores.csv",
        resume=args.resume,
        build_fn=lambda: build_global_extra_scores_df(lgbm_df, mi_df),
        step_name="global_scores",
    )
    composite_df = csv_step(
        out_dir / "composite_extra_scores.csv",
        resume=args.resume,
        build_fn=lambda: build_composite_feature_scores_df(
            feature_inventory_df=feature_inventory,
            global_scores_df=global_df,
            weak_scores_df=weak_df,
            missingness_df=missingness_df,
        ),
        step_name="composite",
    )
    print({"stage": "composite_done", "rows": int(len(composite_df))}, flush=True)

    pruned_df = csv_step(
        out_dir / "redundancy_pruned_extra.csv",
        resume=args.resume,
        build_fn=lambda: run_redundancy_pruning_df(
            composite_df=composite_df,
            data_dir=args.data_dir,
            preselect_n=args.preselect_n,
            corr_threshold=args.corr_threshold,
        ),
        step_name="pruning",
    )
    print({"stage": "pruning_done", "kept": int(pruned_df["kept_flag"].sum())}, flush=True)

    kept = pruned_df.loc[pruned_df["kept_flag"] == 1, "feature_name"].tolist()
    ranked = composite_df[composite_df["feature_name"].isin(kept)].sort_values("composite_score", ascending=False)
    candidate_sizes = [int(x.strip()) for x in args.candidate_sizes.split(",") if x.strip()]
    wrapper_summary_path = out_dir / "wrapper_eval_summary.csv"
    wrapper_preds_path = out_dir / "wrapper_eval_validation_predictions.parquet"
    if args.resume and wrapper_summary_path.exists() and wrapper_preds_path.exists():
        wrapper_summary = pd.read_csv(wrapper_summary_path)
        wrapper_preds = pd.read_parquet(wrapper_preds_path)
        print(
            {
                "stage": "wrapper_resume",
                "rows": int(len(wrapper_summary)),
                "paths": [str(wrapper_summary_path), str(wrapper_preds_path)],
            },
            flush=True,
        )
    else:
        wrapper_summary, wrapper_preds = run_wrapper_eval_df(
            selected_extra_features=ranked["feature_name"].tolist(),
            candidate_sizes=candidate_sizes,
            data_dir=args.data_dir,
            target_subset=WEAKEST_TARGETS if args.wrapper_weak_targets_only else None,
        )
        wrapper_summary.to_csv(wrapper_summary_path, index=False)
        wrapper_preds.to_parquet(wrapper_preds_path, index=False)
    print({"stage": "wrapper_done", "best": wrapper_summary.iloc[0].to_dict() if not wrapper_summary.empty else {}}, flush=True)

    final_feature_set_path = out_dir / "final_feature_set.json"
    final_feature_table_path = out_dir / "final_feature_table.csv"
    selection_summary_path = out_dir / "selection_summary.json"
    if args.resume and final_feature_set_path.exists() and final_feature_table_path.exists() and selection_summary_path.exists():
        payload = json.loads(final_feature_set_path.read_text())
        final_table = pd.read_csv(final_feature_table_path)
        summary = json.loads(selection_summary_path.read_text())
        print(
            {
                "stage": "final_feature_set_resume",
                "selected_extra": int(len(payload.get("extra_features", []))),
                "paths": [str(final_feature_set_path), str(final_feature_table_path), str(selection_summary_path)],
            },
            flush=True,
        )
    else:
        payload, final_table, summary = build_final_feature_set_payload(
            feature_inventory_df=feature_inventory,
            composite_df=composite_df,
            pruned_df=pruned_df,
            wrapper_eval_df=wrapper_summary,
            default_n_final_extra=args.default_n_final_extra,
        )
        final_feature_set_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")
        final_table.to_csv(final_feature_table_path, index=False)
        selection_summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print({"stage": "final_feature_set_done", "selected_extra": int(len(payload["extra_features"]))}, flush=True)

    if args.materialize_dataset:
        materialized_summary_path = out_dir / "materialized_dataset_summary.json"
        if args.resume and materialized_summary_path.exists():
            ds_summary = json.loads(materialized_summary_path.read_text())
            print({"stage": "materialized_dataset_resume", **ds_summary}, flush=True)
        else:
            ds_summary = materialize_selected_feature_dataset(final_feature_set=payload, out_dir=out_dir, data_dir=args.data_dir)
            materialized_summary_path.write_text(json.dumps(ds_summary, ensure_ascii=True, indent=2) + "\n")
        print({"stage": "materialized_dataset_done", **ds_summary}, flush=True)


if __name__ == "__main__":
    main()
