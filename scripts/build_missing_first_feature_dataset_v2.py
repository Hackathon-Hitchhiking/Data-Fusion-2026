from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from lib.layout import project_root


ROOT = project_root()
DEFAULT_IN_DIR = ROOT / "artifacts" / "feature_selection_v1"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "feature_selection_v2_missing_core"
DEFAULT_DIAGNOSTIC_DIR = ROOT / "artifacts" / "dataset_feature_diagnostics_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fs_v2_missing_core dataset on top of fs_v1.")
    parser.add_argument("--in-dir", default=str(DEFAULT_IN_DIR))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--diagnostic-dir", default=str(DEFAULT_DIAGNOSTIC_DIR))
    parser.add_argument("--sample-rows-for-partition", type=int, default=200_000)
    parser.add_argument("--n-bins", type=int, default=32)
    parser.add_argument("--effective-bins-threshold", type=int, default=2)
    parser.add_argument("--unique-threshold", type=int, default=4)
    parser.add_argument("--iqr-threshold", type=float, default=1e-6)
    parser.add_argument("--compression", default="zstd")
    return parser.parse_args()


def quantile_effective_bins(values: np.ndarray, n_bins: int) -> tuple[int, bool]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0, True
    if finite.size == 1:
        return 1, True
    q = np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float64)
    edges = np.quantile(finite, q, method="linear")
    unique_edges = np.unique(edges)
    effective = max(0, int(len(unique_edges) - 1))
    return effective, bool(effective <= 1)


def build_numeric_partition(
    train_df: pd.DataFrame,
    num_cols: list[str],
    *,
    sample_rows: int,
    n_bins: int,
    effective_bins_threshold: int,
    unique_threshold: int,
    iqr_threshold: float,
) -> pd.DataFrame:
    if not num_cols:
        return pd.DataFrame(
            columns=[
                "feature",
                "group",
                "n_unique_train",
                "n_unique_sampled_for_bins",
                "effective_bin_edges_count_sampled",
                "effective_bin_edges_count_full",
                "iqr",
                "std",
                "missing_ratio",
                "one_bin_warning",
            ]
        )

    rng = np.random.default_rng(42)
    sampled_idx = None
    if len(train_df) > sample_rows:
        sampled_idx = np.sort(rng.choice(len(train_df), size=sample_rows, replace=False))

    rows: list[dict[str, object]] = []
    for feature_name in num_cols:
        series = pd.to_numeric(train_df[feature_name], errors="coerce").astype("float32")
        values_full = series.to_numpy(dtype=np.float32, copy=False)
        values_sampled = values_full if sampled_idx is None else values_full[sampled_idx]

        finite_full = values_full[np.isfinite(values_full)]
        finite_sampled = values_sampled[np.isfinite(values_sampled)]

        n_unique_train = int(pd.Series(finite_full).nunique(dropna=True)) if finite_full.size else 0
        n_unique_sampled = int(pd.Series(finite_sampled).nunique(dropna=True)) if finite_sampled.size else 0
        iqr = float(np.subtract(*np.quantile(finite_full, [0.75, 0.25]))) if finite_full.size else 0.0
        std = float(np.std(finite_full)) if finite_full.size else 0.0
        missing_ratio = float(np.isnan(values_full).mean())

        effective_sampled, sampled_warning = quantile_effective_bins(values_sampled, n_bins=n_bins)
        effective_full, full_warning = quantile_effective_bins(values_full, n_bins=n_bins)
        one_bin_warning = bool(sampled_warning or full_warning)
        is_bad = (
            effective_sampled <= int(effective_bins_threshold)
            or n_unique_sampled <= int(unique_threshold)
            or iqr <= float(iqr_threshold)
            or one_bin_warning
        )

        rows.append(
            {
                "feature": feature_name,
                "group": "piecewise_bad" if is_bad else "piecewise_ok",
                "n_unique_train": n_unique_train,
                "n_unique_sampled_for_bins": n_unique_sampled,
                "effective_bin_edges_count_sampled": int(effective_sampled),
                "effective_bin_edges_count_full": int(effective_full),
                "iqr": iqr,
                "std": std,
                "missing_ratio": missing_ratio,
                "one_bin_warning": one_bin_warning,
            }
        )

    return pd.DataFrame(rows).sort_values(["group", "feature"]).reset_index(drop=True)


def build_cluster_manifest(
    diagnostic_dir: Path,
    current_num_cols: list[str],
) -> pd.DataFrame:
    path = diagnostic_dir / "numeric_redundancy_clusters.csv"
    if not path.exists():
        return pd.DataFrame(columns=["cluster_name", "cluster_root", "cluster_size", "piecewise_bad_count", "members"])

    cluster_df = pd.read_csv(path)
    current_num_set = set(current_num_cols)
    rows: list[dict[str, object]] = []
    cluster_idx = 1
    for row in cluster_df.to_dict("records"):
        members_raw = [part.strip() for part in str(row["members"]).split(",") if part.strip()]
        members = [member for member in members_raw if member in current_num_set]
        if len(members) < 2:
            continue
        rows.append(
            {
                "cluster_name": f"cluster_{cluster_idx:02d}",
                "cluster_root": row["cluster_root"],
                "cluster_size": int(len(members)),
                "piecewise_bad_count": int(row.get("piecewise_bad_count", 0)),
                "members": ",".join(members),
            }
        )
        cluster_idx += 1
    return pd.DataFrame(rows)


def build_missing_indicator_block(df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
    block = {}
    for feature_name in num_cols:
        block[f"miss__{feature_name}"] = df[feature_name].isna().to_numpy(dtype=np.uint8)
    return pd.DataFrame(block, index=df.index)


def build_aggregate_block(
    miss_block: pd.DataFrame,
    num_cols: list[str],
    partition_df: pd.DataFrame,
    cluster_manifest: pd.DataFrame,
) -> pd.DataFrame:
    out: dict[str, np.ndarray] = {}
    miss_cols = [f"miss__{name}" for name in num_cols]
    miss_values = miss_block[miss_cols].to_numpy(dtype=np.uint16, copy=False)

    all_count = miss_values.sum(axis=1, dtype=np.uint16)
    out["miss_count__all_num"] = all_count.astype(np.uint16, copy=False)
    out["miss_ratio__all_num"] = (all_count / float(max(1, len(num_cols)))).astype(np.float32, copy=False)

    group_map = partition_df.set_index("feature")["group"].to_dict()
    for group_name in ("piecewise_bad", "piecewise_ok"):
        group_features = [feature for feature in num_cols if group_map.get(feature) == group_name]
        if not group_features:
            continue
        group_vals = miss_block[[f"miss__{feature}" for feature in group_features]].to_numpy(dtype=np.uint16, copy=False)
        group_count = group_vals.sum(axis=1, dtype=np.uint16)
        out[f"miss_count__{group_name}"] = group_count.astype(np.uint16, copy=False)
        out[f"miss_ratio__{group_name}"] = (group_count / float(len(group_features))).astype(np.float32, copy=False)

    for row in cluster_manifest.to_dict("records"):
        members = [member for member in str(row["members"]).split(",") if member]
        cluster_vals = miss_block[[f"miss__{feature}" for feature in members]].to_numpy(dtype=np.uint16, copy=False)
        cluster_count = cluster_vals.sum(axis=1, dtype=np.uint16)
        cluster_name = str(row["cluster_name"])
        out[f"miss_count__{cluster_name}"] = cluster_count.astype(np.uint16, copy=False)
        out[f"miss_ratio__{cluster_name}"] = (cluster_count / float(len(members))).astype(np.float32, copy=False)

    return pd.DataFrame(out, index=miss_block.index)


def build_imputation_blocks(
    train_df: pd.DataFrame,
    other_df: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    med_train: dict[str, np.ndarray] = {}
    med_other: dict[str, np.ndarray] = {}
    sent_train: dict[str, np.ndarray] = {}
    sent_other: dict[str, np.ndarray] = {}
    stat_rows: list[dict[str, object]] = []

    for feature_name in num_cols:
        train_series = pd.to_numeric(train_df[feature_name], errors="coerce").astype("float32")
        other_series = pd.to_numeric(other_df[feature_name], errors="coerce").astype("float32")

        finite = train_series.to_numpy(dtype=np.float32, copy=False)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            median = 0.0
            min_train = 0.0
            iqr = 0.0
            std = 0.0
            sentinel = -1.0
        else:
            median = float(np.median(finite))
            min_train = float(np.min(finite))
            q75, q25 = np.quantile(finite, [0.75, 0.25], method="linear")
            iqr = float(q75 - q25)
            std = float(np.std(finite))
            if iqr > 0:
                sentinel = float(min_train - iqr)
            else:
                fallback_gap = std if std > 0 else max(1.0, abs(min_train) * 0.05 + 1.0)
                sentinel = float(min_train - fallback_gap)

        med_train[f"med__{feature_name}"] = train_series.fillna(median).to_numpy(dtype=np.float32)
        med_other[f"med__{feature_name}"] = other_series.fillna(median).to_numpy(dtype=np.float32)
        sent_train[f"sent__{feature_name}"] = train_series.fillna(sentinel).to_numpy(dtype=np.float32)
        sent_other[f"sent__{feature_name}"] = other_series.fillna(sentinel).to_numpy(dtype=np.float32)

        stat_rows.append(
            {
                "feature": feature_name,
                "median_train": median,
                "min_train": min_train,
                "iqr_train": iqr,
                "std_train": std,
                "sentinel_train": sentinel,
                "missing_ratio_train": float(train_series.isna().mean()),
            }
        )

    stats_df = pd.DataFrame(stat_rows).sort_values("feature").reset_index(drop=True)
    return (
        pd.DataFrame(med_train, index=train_df.index),
        pd.DataFrame(sent_train, index=train_df.index),
        pd.DataFrame(stat_rows),
    )


def build_med_sent_blocks(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    num_cols: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    med_train: dict[str, np.ndarray] = {}
    med_test: dict[str, np.ndarray] = {}
    sent_train: dict[str, np.ndarray] = {}
    sent_test: dict[str, np.ndarray] = {}
    stats: list[dict[str, object]] = []

    for feature_name in num_cols:
        train_series = pd.to_numeric(train_df[feature_name], errors="coerce").astype("float32")
        test_series = pd.to_numeric(test_df[feature_name], errors="coerce").astype("float32")

        finite = train_series.to_numpy(dtype=np.float32, copy=False)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            median = 0.0
            min_train = 0.0
            iqr = 0.0
            std = 0.0
            sentinel = -1.0
        else:
            median = float(np.median(finite))
            min_train = float(np.min(finite))
            q75, q25 = np.quantile(finite, [0.75, 0.25], method="linear")
            iqr = float(q75 - q25)
            std = float(np.std(finite))
            if iqr > 0:
                sentinel = float(min_train - iqr)
            else:
                fallback_gap = std if std > 0 else max(1.0, abs(min_train) * 0.05 + 1.0)
                sentinel = float(min_train - fallback_gap)

        med_train[f"med__{feature_name}"] = train_series.fillna(median).to_numpy(dtype=np.float32)
        med_test[f"med__{feature_name}"] = test_series.fillna(median).to_numpy(dtype=np.float32)
        sent_train[f"sent__{feature_name}"] = train_series.fillna(sentinel).to_numpy(dtype=np.float32)
        sent_test[f"sent__{feature_name}"] = test_series.fillna(sentinel).to_numpy(dtype=np.float32)
        stats.append(
            {
                "feature": feature_name,
                "median_train": median,
                "min_train": min_train,
                "iqr_train": iqr,
                "std_train": std,
                "sentinel_train": sentinel,
                "missing_ratio_train": float(train_series.isna().mean()),
            }
        )

    return (
        pd.DataFrame(med_train, index=train_df.index),
        pd.DataFrame(med_test, index=test_df.index),
        pd.DataFrame(sent_train, index=train_df.index),
        pd.DataFrame(sent_test, index=test_df.index),
        pd.DataFrame(stats).sort_values("feature").reset_index(drop=True),
    )


def main() -> None:
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    diagnostic_dir = Path(args.diagnostic_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = in_dir / "train_compact_features_v1.parquet"
    test_path = in_dir / "test_compact_features_v1.parquet"
    train_df = pd.read_parquet(train_path).sort_values("customer_id").reset_index(drop=True)
    test_df = pd.read_parquet(test_path).sort_values("customer_id").reset_index(drop=True)

    base_feature_cols = [col for col in train_df.columns if col != "customer_id"]
    num_cols = [col for col in base_feature_cols if col.startswith("num_feature")]
    cat_cols = [col for col in base_feature_cols if col.startswith("cat_feature")]

    partition_df = build_numeric_partition(
        train_df,
        num_cols,
        sample_rows=int(args.sample_rows_for_partition),
        n_bins=int(args.n_bins),
        effective_bins_threshold=int(args.effective_bins_threshold),
        unique_threshold=int(args.unique_threshold),
        iqr_threshold=float(args.iqr_threshold),
    )
    cluster_manifest = build_cluster_manifest(diagnostic_dir=diagnostic_dir, current_num_cols=num_cols)

    miss_train = build_missing_indicator_block(train_df, num_cols)
    miss_test = build_missing_indicator_block(test_df, num_cols)
    agg_train = build_aggregate_block(miss_train, num_cols, partition_df, cluster_manifest)
    agg_test = build_aggregate_block(miss_test, num_cols, partition_df, cluster_manifest)
    med_train, med_test, sent_train, sent_test, stats_df = build_med_sent_blocks(train_df, test_df, num_cols)

    train_core = pd.concat(
        [
            train_df[["customer_id"] + base_feature_cols],
            miss_train,
            agg_train,
            med_train,
            sent_train,
        ],
        axis=1,
    )
    test_core = pd.concat(
        [
            test_df[["customer_id"] + base_feature_cols],
            miss_test,
            agg_test,
            med_test,
            sent_test,
        ],
        axis=1,
    )

    train_missing_only = pd.concat([train_df[["customer_id"]], miss_train, agg_train], axis=1)
    test_missing_only = pd.concat([test_df[["customer_id"]], miss_test, agg_test], axis=1)

    train_core_path = out_dir / "train_compact_features_v2_missing_core.parquet"
    test_core_path = out_dir / "test_compact_features_v2_missing_core.parquet"
    train_missing_only_path = out_dir / "train_missing_only_features_v2.parquet"
    test_missing_only_path = out_dir / "test_missing_only_features_v2.parquet"
    partition_path = out_dir / "numeric_feature_partition_v2.csv"
    cluster_path = out_dir / "redundancy_cluster_manifest_v2.csv"
    stats_path = out_dir / "numeric_imputation_stats_v2.csv"
    column_groups_path = out_dir / "column_groups.json"
    summary_path = out_dir / "summary.json"

    train_core.to_parquet(train_core_path, index=False, compression=args.compression)
    test_core.to_parquet(test_core_path, index=False, compression=args.compression)
    train_missing_only.to_parquet(train_missing_only_path, index=False, compression=args.compression)
    test_missing_only.to_parquet(test_missing_only_path, index=False, compression=args.compression)
    partition_df.to_csv(partition_path, index=False)
    cluster_manifest.to_csv(cluster_path, index=False)
    stats_df.to_csv(stats_path, index=False)

    aggregate_cols = list(agg_train.columns)
    column_groups = {
        "base_feature_cols": base_feature_cols,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "missing_indicator_cols": list(miss_train.columns),
        "aggregate_missing_cols": aggregate_cols,
        "median_imputed_cols": list(med_train.columns),
        "sentinel_imputed_cols": list(sent_train.columns),
        "cluster_names": cluster_manifest["cluster_name"].tolist() if not cluster_manifest.empty else [],
    }
    column_groups_path.write_text(json.dumps(column_groups, ensure_ascii=True, indent=2) + "\n")

    summary = {
        "source_dataset": str(in_dir),
        "out_dir": str(out_dir),
        "train_rows": int(len(train_core)),
        "test_rows": int(len(test_core)),
        "base_feature_count": int(len(base_feature_cols)),
        "numeric_feature_count": int(len(num_cols)),
        "categorical_feature_count": int(len(cat_cols)),
        "missing_indicator_count": int(len(miss_train.columns)),
        "aggregate_missing_count": int(len(aggregate_cols)),
        "median_imputed_count": int(len(med_train.columns)),
        "sentinel_imputed_count": int(len(sent_train.columns)),
        "core_feature_count_excluding_customer_id": int(len(train_core.columns) - 1),
        "missing_only_feature_count_excluding_customer_id": int(len(train_missing_only.columns) - 1),
        "piecewise_bad_count": int((partition_df["group"] == "piecewise_bad").sum()),
        "piecewise_ok_count": int((partition_df["group"] == "piecewise_ok").sum()),
        "cluster_count": int(len(cluster_manifest)),
        "top_missing_train": (
            stats_df.sort_values("missing_ratio_train", ascending=False)[["feature", "missing_ratio_train"]]
            .head(12)
            .to_dict(orient="records")
        ),
        "files": {
            "train_core": str(train_core_path),
            "test_core": str(test_core_path),
            "train_missing_only": str(train_missing_only_path),
            "test_missing_only": str(test_missing_only_path),
            "numeric_partition": str(partition_path),
            "cluster_manifest": str(cluster_path),
            "imputation_stats": str(stats_path),
            "column_groups": str(column_groups_path),
        },
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
