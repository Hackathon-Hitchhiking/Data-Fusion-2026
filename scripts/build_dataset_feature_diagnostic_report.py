from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rtdl_num_embeddings import PiecewiseLinearEmbeddings, compute_bins
from scipy.stats import ks_2samp
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score

from lib.data_loading import load_labeled_frame
from lib.layout import project_root
from lib.submission import normalize_prediction_columns


ROOT = project_root()
ARTIFACT_DIR = ROOT / "artifacts" / "dataset_feature_diagnostics_v1"
TOP_FEATURES_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
VAL_PRED_PATH = (
    ROOT
    / "output/kaggle-output/tabm-longrun-v3/current_pull/artifacts/gpu_tabm_longrun_v3_full_gpu/validation_predictions.parquet"
)
GANDALF_VAL_PATH = ROOT / "artifacts/gpu_gandalf_full_gpu/validation_predictions.parquet"
SPECIALIST_VAL_PATH = (
    ROOT / "artifacts/catboost_extended_specialists_gandalf_h12_router/specialist_val_predictions.parquet"
)
WEAKEST_PATH = ROOT / "artifacts/weakest_targets_by_best_auc_v2_longrun.csv"

SIGNAL_SAMPLE_ROWS = 150_000
MI_SAMPLE_ROWS = 100_000
CORR_SAMPLE_ROWS = 120_000
PIECEWISE_CFG = {
    "piecewise_n_bins": 32,
    "piecewise_bin_sample_rows": 200_000,
    "piecewise_d_embedding": 16,
    "piecewise_activation": False,
    "hybrid_iqr_threshold": 1e-6,
    "hybrid_unique_threshold": 4,
    "hybrid_effective_bins_threshold": 2,
    "clip_value": 8.0,
}


def markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_empty_"
    view = df.head(max_rows).copy()
    headers = [str(col) for col in view.columns.tolist()]
    rows = [headers] + view.astype(str).values.tolist()
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]

    def fmt(row: list[str]) -> str:
        return "| " + " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) + " |"

    out = [fmt(headers), "| " + " | ".join("-" * widths[i] for i in range(len(headers))) + " |"]
    out.extend(fmt(row) for row in view.astype(str).values.tolist())
    return "\n".join(out)


def smooth_clip(values: pd.Series, clip_value: float) -> pd.Series:
    clipped = clip_value * np.tanh(values.to_numpy(dtype=np.float32, copy=False) / clip_value)
    return pd.Series(clipped, index=values.index, dtype="float32")


def infer_feature_block(name: str) -> str:
    if name.startswith("cat_feature"):
        return "cat_feature"
    if name.startswith("num_feature"):
        return "num_feature"
    if "_" in name:
        return "_".join(name.split("_")[:2])
    return name


def compute_auc_safe(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return 0.5
    if np.all(scores == scores[0]):
        return 0.5
    return float(roc_auc_score(y, scores))


def psi_score(train_values: np.ndarray, val_values: np.ndarray, bins: int = 10) -> float:
    train_values = train_values[np.isfinite(train_values)]
    val_values = val_values[np.isfinite(val_values)]
    if len(train_values) == 0 or len(val_values) == 0:
        return 0.0
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(train_values, quantiles))
    if len(edges) <= 2:
        return 0.0
    train_hist, _ = np.histogram(train_values, bins=edges)
    val_hist, _ = np.histogram(val_values, bins=edges)
    train_pct = np.clip(train_hist / max(train_hist.sum(), 1), 1e-6, None)
    val_pct = np.clip(val_hist / max(val_hist.sum(), 1), 1e-6, None)
    return float(np.sum((train_pct - val_pct) * np.log(train_pct / val_pct)))


def encode_categorical_feature(series: pd.Series) -> pd.Series:
    clean = pd.to_numeric(series, errors="coerce").fillna(-1).astype("int64")
    known_vals = sorted(v for v in clean.unique().tolist() if v >= 0)
    mapping = {v: i for i, v in enumerate(known_vals)}
    unk = len(mapping)
    return clean.map(mapping).fillna(unk).astype("int32")


def prepare_numeric_train(train_df: pd.DataFrame, num_cols: list[str], clip_value: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_num = train_df[num_cols].copy()
    stat_rows = []
    for col in num_cols:
        raw = pd.to_numeric(train_num[col], errors="coerce")
        missing_ratio = float(raw.isna().mean())
        n_unique = int(raw.nunique(dropna=True))
        median = float(raw.median())
        q1 = float(raw.quantile(0.25))
        q3 = float(raw.quantile(0.75))
        iqr_raw = q3 - q1
        if iqr_raw <= 1e-6:
            std_raw = float(raw.std())
            scale = std_raw if std_raw > 1e-6 else 1.0
        else:
            scale = iqr_raw
        scaled = raw.fillna(median)
        scaled = ((scaled - median) / scale).astype("float32")
        scaled = smooth_clip(scaled, clip_value)
        train_num[col] = scaled
        scaled_np = scaled.to_numpy(dtype=np.float32, copy=False)
        stat_rows.append(
            {
                "feature": col,
                "n_unique_train": n_unique,
                "iqr": float(np.quantile(scaled_np, 0.75) - np.quantile(scaled_np, 0.25)) if len(scaled_np) else 0.0,
                "std": float(np.std(scaled_np)) if len(scaled_np) else 0.0,
                "missing_ratio": missing_ratio,
                "is_constant": bool((np.nanmax(scaled_np) - np.nanmin(scaled_np)) <= 0) if len(scaled_np) else True,
            }
        )
    train_num_matrix = train_num.to_numpy(dtype=np.float32, copy=True)
    num_ptp = np.nanmax(train_num_matrix, axis=0) - np.nanmin(train_num_matrix, axis=0)
    varying_num_cols = [col for col, keep in zip(num_cols, np.isfinite(num_ptp) & (num_ptp > 0)) if keep]
    return train_num[varying_num_cols].copy(), pd.DataFrame(stat_rows).query("feature in @varying_num_cols").reset_index(drop=True)


def compute_effective_bin_profile(x_num: np.ndarray, n_bins: int, d_embedding: int, activation: bool) -> tuple[list[int], set[int]]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bins = compute_bins(
            torch_from_numpy_cpu(x_num),
            n_bins=n_bins,
        )
        PiecewiseLinearEmbeddings(
            bins,
            d_embedding=d_embedding,
            activation=activation,
            version="B",
        )
    warning_idx = set()
    for item in caught:
        message = str(item.message)
        if "feature has just two bin edges" in message:
            prefix = message.split("-th feature", 1)[0]
            idx_str = prefix.split("The ")[-1]
            if idx_str.isdigit():
                warning_idx.add(int(idx_str))
    return [int(len(edges)) for edges in bins], warning_idx


def torch_from_numpy_cpu(x_num: np.ndarray):
    import torch

    return torch.from_numpy(x_num).float().cpu()


def build_piecewise_partition(train_num_scaled: pd.DataFrame, num_stats_df: pd.DataFrame) -> pd.DataFrame:
    train_x_num = train_num_scaled.to_numpy(dtype=np.float32, copy=True)
    sample_rows = PIECEWISE_CFG["piecewise_bin_sample_rows"]
    rng = np.random.default_rng(42)
    if len(train_x_num) > sample_rows:
        sample_idx = np.sort(rng.choice(len(train_x_num), size=sample_rows, replace=False))
        sampled_x_num = train_x_num[sample_idx]
    else:
        sampled_x_num = train_x_num

    sampled_counts, sampled_warning_idx = compute_effective_bin_profile(
        sampled_x_num,
        n_bins=int(PIECEWISE_CFG["piecewise_n_bins"]),
        d_embedding=int(PIECEWISE_CFG["piecewise_d_embedding"]),
        activation=bool(PIECEWISE_CFG["piecewise_activation"]),
    )
    full_counts, full_warning_idx = compute_effective_bin_profile(
        train_x_num,
        n_bins=int(PIECEWISE_CFG["piecewise_n_bins"]),
        d_embedding=int(PIECEWISE_CFG["piecewise_d_embedding"]),
        activation=bool(PIECEWISE_CFG["piecewise_activation"]),
    )

    sampled_unique = [int(pd.Series(sampled_x_num[:, i]).nunique(dropna=True)) for i in range(sampled_x_num.shape[1])]
    stat_map = num_stats_df.set_index("feature").to_dict(orient="index")
    rows = []
    for i, feature_name in enumerate(train_num_scaled.columns.tolist()):
        stats = stat_map[feature_name]
        one_bin_warning = bool((i in sampled_warning_idx) or (i in full_warning_idx))
        is_bad = (
            sampled_counts[i] <= int(PIECEWISE_CFG["hybrid_effective_bins_threshold"])
            or sampled_unique[i] <= int(PIECEWISE_CFG["hybrid_unique_threshold"])
            or float(stats["iqr"]) <= float(PIECEWISE_CFG["hybrid_iqr_threshold"])
            or one_bin_warning
        )
        rows.append(
            {
                "feature": feature_name,
                "group": "piecewise_bad" if is_bad else "piecewise_ok",
                "n_unique_train": int(stats["n_unique_train"]),
                "n_unique_sampled_for_bins": int(sampled_unique[i]),
                "effective_bin_edges_count": int(sampled_counts[i]),
                "effective_bin_edges_count_full": int(full_counts[i]),
                "iqr": float(stats["iqr"]),
                "std": float(stats["std"]),
                "missing_ratio": float(stats["missing_ratio"]),
                "one_bin_warning": bool(one_bin_warning),
            }
        )
    return pd.DataFrame(rows)


def maybe_sample(df: pd.DataFrame, n_rows: int, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n_rows:
        return df.copy()
    return df.sample(n_rows, random_state=seed).reset_index(drop=True)


def compute_feature_summary(labeled: pd.DataFrame, feature_cols: list[str], cat_cols: list[str], piecewise_partition_df: pd.DataFrame) -> pd.DataFrame:
    piecewise_map = piecewise_partition_df.set_index("feature").to_dict(orient="index")
    rows = []
    for col in feature_cols:
        is_cat = col in cat_cols
        series = labeled[col]
        numeric = pd.to_numeric(series, errors="coerce")
        row = {
            "feature": col,
            "feature_type": "cat" if is_cat else "num",
            "block": infer_feature_block(col),
            "n_unique": int(series.nunique(dropna=True)),
            "missing_ratio": float(series.isna().mean()),
            "mean": float(numeric.mean()) if numeric.notna().any() else math.nan,
            "std": float(numeric.std()) if numeric.notna().any() else math.nan,
            "iqr": float(numeric.quantile(0.75) - numeric.quantile(0.25)) if numeric.notna().any() else math.nan,
            "piecewise_group": piecewise_map.get(col, {}).get("group"),
            "piecewise_one_bin_warning": piecewise_map.get(col, {}).get("one_bin_warning"),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_matrix(sample_df: pd.DataFrame, feature_cols: list[str], cat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    columns = {}
    discrete_mask = []
    for col in feature_cols:
        if col in cat_cols:
            columns[col] = encode_categorical_feature(sample_df[col]).astype("float32")
            discrete_mask.append(True)
        else:
            numeric = pd.to_numeric(sample_df[col], errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            columns[col] = numeric.fillna(median).astype("float32")
            discrete_mask.append(False)
    X = pd.DataFrame(columns, index=sample_df.index)
    return X, np.array(discrete_mask, dtype=bool)


def compute_feature_target_signal(
    labeled: pd.DataFrame,
    feature_cols: list[str],
    cat_cols: list[str],
    weak_targets: list[str],
    feature_summary_df: pd.DataFrame,
) -> pd.DataFrame:
    auc_sample = maybe_sample(labeled[["customer_id"] + feature_cols + weak_targets], SIGNAL_SAMPLE_ROWS, seed=42)
    mi_sample = maybe_sample(labeled[["customer_id"] + feature_cols + weak_targets], MI_SAMPLE_ROWS, seed=52)
    X_mi, discrete_mask = build_feature_matrix(mi_sample, feature_cols, cat_cols)

    mi_by_target: dict[str, np.ndarray] = {}
    for target_name in weak_targets:
        print({"stage": "mi_target_start", "target": target_name})
        mi_by_target[target_name] = mutual_info_classif(
            X_mi.to_numpy(dtype=np.float32),
            mi_sample[target_name].to_numpy(dtype=np.int8),
            discrete_features=discrete_mask,
            random_state=42,
            n_jobs=-1,
        )
        print({"stage": "mi_target_done", "target": target_name})

    summary_map = feature_summary_df.set_index("feature").to_dict(orient="index")
    rows = []
    for target_name in weak_targets:
        print({"stage": "univariate_target_start", "target": target_name})
        y = auc_sample[target_name].to_numpy(dtype=np.int8)
        for idx, feature_name in enumerate(feature_cols):
            series = auc_sample[feature_name]
            numeric = pd.to_numeric(series, errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            score_values = numeric.fillna(median).to_numpy(dtype=np.float32)
            missing_indicator = series.isna().astype("int8").to_numpy()
            auc_value = compute_auc_safe(y, score_values)
            missing_auc = compute_auc_safe(y, missing_indicator) if missing_indicator.sum() not in (0, len(missing_indicator)) else 0.5
            rows.append(
                {
                    "feature": feature_name,
                    "target": target_name,
                    "feature_type": summary_map[feature_name]["feature_type"],
                    "block": summary_map[feature_name]["block"],
                    "missing_ratio": summary_map[feature_name]["missing_ratio"],
                    "n_unique": summary_map[feature_name]["n_unique"],
                    "is_piecewise_bad": summary_map[feature_name]["piecewise_group"] == "piecewise_bad",
                    "univariate_auc": auc_value,
                    "auc_gain": abs(auc_value - 0.5),
                    "missingness_auc": missing_auc,
                    "missingness_gain": abs(missing_auc - 0.5),
                    "mi_score": float(mi_by_target[target_name][idx]),
                }
            )
        print({"stage": "univariate_target_done", "target": target_name})
    signal_df = pd.DataFrame(rows)
    signal_df["combined_rank_score"] = signal_df["mi_score"].rank(pct=True) + signal_df["auc_gain"].rank(pct=True)
    return signal_df.sort_values(["target", "combined_rank_score"], ascending=[True, False]).reset_index(drop=True)


def compute_top_feature_profiles(labeled: pd.DataFrame, signal_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    rows = []
    for target_name, group in signal_df.groupby("target"):
        top_features = group.sort_values(["mi_score", "auc_gain"], ascending=False).head(top_k)["feature"].tolist()
        target_sample = maybe_sample(labeled[["customer_id", target_name] + top_features], SIGNAL_SAMPLE_ROWS, seed=123)
        y = target_sample[target_name].to_numpy(dtype=np.int8)
        for feature_name in top_features:
            numeric = pd.to_numeric(target_sample[feature_name], errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            values = numeric.fillna(median)
            try:
                bins = pd.qcut(values, q=10, duplicates="drop")
                bin_rates = target_sample.groupby(bins, observed=False)[target_name].mean()
                qbin_rate_range = float(bin_rates.max() - bin_rates.min()) if len(bin_rates) else 0.0
            except ValueError:
                qbin_rate_range = 0.0
            rows.append(
                {
                    "target": target_name,
                    "feature": feature_name,
                    "qbin_rate_range": qbin_rate_range,
                }
            )
    return signal_df.merge(pd.DataFrame(rows), on=["target", "feature"], how="left").sort_values(
        ["target", "mi_score", "auc_gain"], ascending=[True, False, False]
    )


def compute_piecewise_bad_audit(signal_df: pd.DataFrame) -> pd.DataFrame:
    bad_df = signal_df[signal_df["is_piecewise_bad"]].copy()
    if bad_df.empty:
        return bad_df
    audit = (
        bad_df.groupby("feature")
        .agg(
            piecewise_bad=("is_piecewise_bad", "max"),
            max_mi_score=("mi_score", "max"),
            max_auc_gain=("auc_gain", "max"),
            max_missingness_gain=("missingness_gain", "max"),
            weak_targets_top20_count=("combined_rank_score", lambda s: int((s >= s.max() - 1e-12).sum())),
            mean_combined_rank_score=("combined_rank_score", "mean"),
        )
        .reset_index()
        .sort_values(["max_mi_score", "max_auc_gain"], ascending=False)
        .reset_index(drop=True)
    )
    return audit


def compute_label_dependency_tables(labeled: pd.DataFrame, target_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = labeled[target_cols].to_numpy(dtype=np.int8)
    n = len(labeled)
    p = y.mean(axis=0)
    rows = []
    family_rows = []
    for i, t1 in enumerate(target_cols):
        fam1 = t1.split("_")[1]
        for j, t2 in enumerate(target_cols):
            if i >= j:
                continue
            fam2 = t2.split("_")[1]
            joint = float((y[:, i] & y[:, j]).mean())
            cond_12 = joint / max(p[i], 1e-9)
            cond_21 = joint / max(p[j], 1e-9)
            lift = joint / max(p[i] * p[j], 1e-9)
            pmi = float(np.log(max(joint, 1e-12) / max(p[i] * p[j], 1e-12)))
            rows.append(
                {
                    "target_a": t1,
                    "target_b": t2,
                    "family_a": fam1,
                    "family_b": fam2,
                    "cooccurrence": joint,
                    "p_a": float(p[i]),
                    "p_b": float(p[j]),
                    "p_b_given_a": cond_12,
                    "p_a_given_b": cond_21,
                    "lift": lift,
                    "pmi": pmi,
                }
            )
    pair_df = pd.DataFrame(rows).sort_values("lift", ascending=False).reset_index(drop=True)
    for (fam_a, fam_b), group in pair_df.groupby(["family_a", "family_b"]):
        family_rows.append(
            {
                "family_a": fam_a,
                "family_b": fam_b,
                "pair_count": int(len(group)),
                "mean_lift": float(group["lift"].mean()),
                "max_lift": float(group["lift"].max()),
                "mean_pmi": float(group["pmi"].mean()),
            }
        )
    family_df = pd.DataFrame(family_rows).sort_values("mean_lift", ascending=False).reset_index(drop=True)
    return pair_df, family_df


def compute_numeric_redundancy(labeled: pd.DataFrame, num_cols: list[str], piecewise_partition_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sample = maybe_sample(labeled[["customer_id"] + num_cols], CORR_SAMPLE_ROWS, seed=7)
    data = sample[num_cols].copy()
    for col in num_cols:
        numeric = pd.to_numeric(data[col], errors="coerce")
        median = float(numeric.median()) if numeric.notna().any() else 0.0
        data[col] = numeric.fillna(median).astype("float32")
    corr = data.corr(numeric_only=True).abs()
    pairs = []
    cols = corr.columns.tolist()
    piecewise_map = piecewise_partition_df.set_index("feature")["group"].to_dict()
    for i, left in enumerate(cols):
        for j in range(i + 1, len(cols)):
            right = cols[j]
            value = float(corr.iat[i, j])
            if value >= 0.95:
                pairs.append(
                    {
                        "feature_a": left,
                        "feature_b": right,
                        "abs_corr": value,
                        "group_a": piecewise_map.get(left),
                        "group_b": piecewise_map.get(right),
                    }
                )
    pair_df = pd.DataFrame(pairs).sort_values("abs_corr", ascending=False).reset_index(drop=True)
    cluster_rows = []
    if not pair_df.empty:
        parent = {col: col for col in cols}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for row in pair_df.itertuples(index=False):
            union(row.feature_a, row.feature_b)
        groups: dict[str, list[str]] = {}
        for col in cols:
            groups.setdefault(find(col), []).append(col)
        for root, members in groups.items():
            if len(members) < 2:
                continue
            cluster_rows.append(
                {
                    "cluster_root": root,
                    "cluster_size": len(members),
                    "piecewise_bad_count": int(sum(piecewise_map.get(m) == "piecewise_bad" for m in members)),
                    "members": ", ".join(members[:15]) + (" ..." if len(members) > 15 else ""),
                }
            )
    cluster_df = pd.DataFrame(cluster_rows).sort_values("cluster_size", ascending=False).reset_index(drop=True)
    return pair_df, cluster_df


def compute_feature_stability(train_df: pd.DataFrame, val_df: pd.DataFrame, num_cols: list[str], weak_targets: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for col in num_cols:
        train_series = pd.to_numeric(train_df[col], errors="coerce")
        val_series = pd.to_numeric(val_df[col], errors="coerce")
        train_fill = train_series.fillna(float(train_series.median()) if train_series.notna().any() else 0.0).to_numpy(dtype=np.float32)
        val_fill = val_series.fillna(float(train_series.median()) if train_series.notna().any() else 0.0).to_numpy(dtype=np.float32)
        rows.append(
            {
                "feature": col,
                "psi": psi_score(train_fill, val_fill),
                "ks_stat": float(ks_2samp(train_fill, val_fill).statistic),
                "train_missing_ratio": float(train_series.isna().mean()),
                "val_missing_ratio": float(val_series.isna().mean()),
            }
        )
    overall_df = pd.DataFrame(rows).sort_values(["psi", "ks_stat"], ascending=False).reset_index(drop=True)

    targeted_rows = []
    for target_name in weak_targets[:6]:
        train_pos = train_df[train_df[target_name] == 1]
        val_pos = val_df[val_df[target_name] == 1]
        if len(train_pos) < 30 or len(val_pos) < 30:
            continue
        for col in num_cols:
            train_series = pd.to_numeric(train_pos[col], errors="coerce")
            val_series = pd.to_numeric(val_pos[col], errors="coerce")
            train_fill = train_series.fillna(float(train_series.median()) if train_series.notna().any() else 0.0).to_numpy(dtype=np.float32)
            val_fill = val_series.fillna(float(train_series.median()) if train_series.notna().any() else 0.0).to_numpy(dtype=np.float32)
            targeted_rows.append(
                {
                    "target": target_name,
                    "feature": col,
                    "positive_train_rows": int(len(train_pos)),
                    "positive_val_rows": int(len(val_pos)),
                    "psi_positive": psi_score(train_fill, val_fill),
                    "ks_positive": float(ks_2samp(train_fill, val_fill).statistic),
                }
            )
    targeted_df = pd.DataFrame(targeted_rows).sort_values(["target", "psi_positive"], ascending=[True, False]).reset_index(drop=True)
    return overall_df, targeted_df


def compute_error_geography(
    val_df: pd.DataFrame,
    weak_targets: list[str],
    top_feature_profiles_df: pd.DataFrame,
) -> pd.DataFrame:
    tabm_pred = normalize_prediction_columns(pd.read_parquet(VAL_PRED_PATH)).sort_values("customer_id").reset_index(drop=True)
    gandalf_pred = normalize_prediction_columns(pd.read_parquet(GANDALF_VAL_PATH)).sort_values("customer_id").reset_index(drop=True)
    spec_pred = normalize_prediction_columns(pd.read_parquet(SPECIALIST_VAL_PATH)).sort_values("customer_id").reset_index(drop=True)

    common_ids = set(tabm_pred["customer_id"]) & set(gandalf_pred["customer_id"]) & set(spec_pred["customer_id"]) & set(val_df["customer_id"])
    val_df = val_df[val_df["customer_id"].isin(common_ids)].sort_values("customer_id").reset_index(drop=True)
    tabm_pred = tabm_pred[tabm_pred["customer_id"].isin(common_ids)].sort_values("customer_id").reset_index(drop=True)
    gandalf_pred = gandalf_pred[gandalf_pred["customer_id"].isin(common_ids)].sort_values("customer_id").reset_index(drop=True)
    spec_pred = spec_pred[spec_pred["customer_id"].isin(common_ids)].sort_values("customer_id").reset_index(drop=True)

    rows = []
    for target_name in weak_targets[:6]:
        pred_col = target_name.replace("target_", "predict_")
        y = val_df[target_name].to_numpy(dtype=np.float32)
        tabm_err = np.abs(y - tabm_pred[pred_col].to_numpy(dtype=np.float32))
        gandalf_err = np.abs(y - gandalf_pred[pred_col].to_numpy(dtype=np.float32))
        spec_err = np.abs(y - spec_pred[pred_col].to_numpy(dtype=np.float32))

        top_features = (
            top_feature_profiles_df[top_feature_profiles_df["target"] == target_name]
            .sort_values(["mi_score", "auc_gain"], ascending=False)
            .head(3)["feature"]
            .tolist()
        )
        for feature_name in top_features:
            numeric = pd.to_numeric(val_df[feature_name], errors="coerce")
            median = float(numeric.median()) if numeric.notna().any() else 0.0
            values = numeric.fillna(median)
            try:
                bins = pd.qcut(values, q=10, duplicates="drop")
            except ValueError:
                continue
            bucket = pd.DataFrame(
                {
                    "bin": bins.astype(str),
                    "tabm_err": tabm_err,
                    "gandalf_err": gandalf_err,
                    "spec_err": spec_err,
                }
            )
            agg = (
                bucket.groupby("bin", observed=False)
                .agg(
                    rows=("tabm_err", "size"),
                    tabm_mean_abs_error=("tabm_err", "mean"),
                    gandalf_mean_abs_error=("gandalf_err", "mean"),
                    spec_mean_abs_error=("spec_err", "mean"),
                )
                .reset_index()
            )
            agg["best_alt_mean_abs_error"] = agg[["gandalf_mean_abs_error", "spec_mean_abs_error"]].min(axis=1)
            agg["tabm_minus_best_alt"] = agg["tabm_mean_abs_error"] - agg["best_alt_mean_abs_error"]
            for row in agg.sort_values("tabm_mean_abs_error", ascending=False).head(3).itertuples(index=False):
                rows.append(
                    {
                        "target": target_name,
                        "feature": feature_name,
                        "bin": row.bin,
                        "rows": int(row.rows),
                        "tabm_mean_abs_error": float(row.tabm_mean_abs_error),
                        "gandalf_mean_abs_error": float(row.gandalf_mean_abs_error),
                        "spec_mean_abs_error": float(row.spec_mean_abs_error),
                        "tabm_minus_best_alt": float(row.tabm_minus_best_alt),
                    }
                )
    return pd.DataFrame(rows).sort_values(["target", "tabm_mean_abs_error"], ascending=[True, False]).reset_index(drop=True)


def main() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    print({"stage": "load_inputs_start"})
    top_features = json.loads(TOP_FEATURES_FILE.read_text())
    labeled = load_labeled_frame(extra_feature_cols=top_features).sort_values("customer_id").reset_index(drop=True)
    target_cols = [c for c in labeled.columns if c.startswith("target_")]
    feature_cols = [c for c in labeled.columns if c not in ["customer_id"] + target_cols]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    val_ids = normalize_prediction_columns(pd.read_parquet(VAL_PRED_PATH, columns=["customer_id"]))["customer_id"]
    val_set = set(val_ids.tolist())
    val_df = labeled[labeled["customer_id"].isin(val_set)].sort_values("customer_id").reset_index(drop=True)
    train_df = labeled[~labeled["customer_id"].isin(val_set)].sort_values("customer_id").reset_index(drop=True)

    weak_df = pd.read_csv(WEAKEST_PATH)
    weak_targets = weak_df["target"].head(12).tolist()
    weakest6 = weak_targets[:6]
    print(
        {
            "stage": "load_inputs_done",
            "labeled_rows": int(len(labeled)),
            "feature_count": int(len(feature_cols)),
            "num_features": int(len(num_cols)),
            "cat_features": int(len(cat_cols)),
            "weak_targets": weak_targets,
        }
    )

    train_num_scaled, num_stats_df = prepare_numeric_train(train_df, num_cols, clip_value=float(PIECEWISE_CFG["clip_value"]))
    piecewise_partition_df = build_piecewise_partition(train_num_scaled, num_stats_df)
    feature_summary_df = compute_feature_summary(labeled, feature_cols, cat_cols, piecewise_partition_df)
    print(
        {
            "stage": "numeric_partition_ready",
            "varying_num_features": int(len(piecewise_partition_df)),
            "piecewise_ok": int((piecewise_partition_df["group"] == "piecewise_ok").sum()),
            "piecewise_bad": int((piecewise_partition_df["group"] == "piecewise_bad").sum()),
        }
    )

    signal_df = compute_feature_target_signal(labeled, feature_cols, cat_cols, weak_targets, feature_summary_df)
    print({"stage": "feature_target_signal_ready", "rows": int(len(signal_df))})
    top_feature_profiles_df = compute_top_feature_profiles(labeled, signal_df, top_k=20)
    print({"stage": "top_feature_profiles_ready", "rows": int(len(top_feature_profiles_df))})
    piecewise_bad_audit_df = compute_piecewise_bad_audit(signal_df)
    label_pair_df, label_family_df = compute_label_dependency_tables(labeled, target_cols)
    print(
        {
            "stage": "label_dependency_ready",
            "pair_rows": int(len(label_pair_df)),
            "family_rows": int(len(label_family_df)),
        }
    )
    redundancy_pairs_df, redundancy_clusters_df = compute_numeric_redundancy(labeled, num_cols, piecewise_partition_df)
    print(
        {
            "stage": "numeric_redundancy_ready",
            "pair_rows": int(len(redundancy_pairs_df)),
            "cluster_rows": int(len(redundancy_clusters_df)),
        }
    )
    stability_df, stability_positive_df = compute_feature_stability(train_df, val_df, num_cols, weak_targets)
    print(
        {
            "stage": "feature_stability_ready",
            "overall_rows": int(len(stability_df)),
            "positive_rows": int(len(stability_positive_df)),
        }
    )
    error_geo_df = compute_error_geography(val_df, weakest6, top_feature_profiles_df)
    print({"stage": "error_geography_ready", "rows": int(len(error_geo_df))})

    target_summary_df = pd.DataFrame(
        {
            "target": target_cols,
            "positive_rate": [float(labeled[t].mean()) for t in target_cols],
        }
    ).merge(weak_df[["target", "best_auc", "best_model"]], on="target", how="left")

    feature_summary_df.to_csv(ARTIFACT_DIR / "feature_summary.csv", index=False)
    target_summary_df.to_csv(ARTIFACT_DIR / "target_summary.csv", index=False)
    piecewise_partition_df.to_csv(ARTIFACT_DIR / "numeric_feature_partition.csv", index=False)
    signal_df.to_csv(ARTIFACT_DIR / "feature_target_signal_weak12.csv", index=False)
    top_feature_profiles_df.to_csv(ARTIFACT_DIR / "weak_target_top_features.csv", index=False)
    piecewise_bad_audit_df.to_csv(ARTIFACT_DIR / "piecewise_bad_feature_audit.csv", index=False)
    label_pair_df.to_csv(ARTIFACT_DIR / "label_dependency_pairwise.csv", index=False)
    label_family_df.to_csv(ARTIFACT_DIR / "label_dependency_family_summary.csv", index=False)
    redundancy_pairs_df.to_csv(ARTIFACT_DIR / "numeric_redundancy_pairs.csv", index=False)
    redundancy_clusters_df.to_csv(ARTIFACT_DIR / "numeric_redundancy_clusters.csv", index=False)
    stability_df.to_csv(ARTIFACT_DIR / "feature_stability_train_vs_val.csv", index=False)
    stability_positive_df.to_csv(ARTIFACT_DIR / "feature_stability_positive_weak_targets.csv", index=False)
    error_geo_df.to_csv(ARTIFACT_DIR / "tabm_error_geography_weak6.csv", index=False)

    top_bad_strong = piecewise_bad_audit_df.head(15)[["feature", "max_mi_score", "max_auc_gain", "max_missingness_gain"]] if not piecewise_bad_audit_df.empty else pd.DataFrame()
    weak_target_feature_head = (
        top_feature_profiles_df.groupby("target", group_keys=False)
        .head(8)[["target", "feature", "mi_score", "auc_gain", "missingness_gain", "qbin_rate_range", "is_piecewise_bad"]]
        .reset_index(drop=True)
    )
    dependency_head = label_pair_df.head(20)[["target_a", "target_b", "lift", "pmi", "p_b_given_a", "p_a_given_b"]]
    error_head = error_geo_df.head(24)[["target", "feature", "bin", "rows", "tabm_mean_abs_error", "tabm_minus_best_alt"]]
    stability_head = stability_df.head(20)[["feature", "psi", "ks_stat", "train_missing_ratio", "val_missing_ratio"]]
    cluster_head = redundancy_clusters_df.head(15)[["cluster_root", "cluster_size", "piecewise_bad_count", "members"]]

    piecewise_ok = int((piecewise_partition_df["group"] == "piecewise_ok").sum())
    piecewise_bad = int((piecewise_partition_df["group"] == "piecewise_bad").sum())
    varying_num = int(len(piecewise_partition_df))

    report = f"""# Dataset And Feature Diagnostics V1

## Scope

- Current reference backbone: `TabM longrun v3`
- Current running numeric routing hypothesis: `TabM v4 hybrid numeric path`
- Holdout definition: `customer_id` set from `TabM longrun v3` validation predictions
- Weak-target focus: first `12` rows of `weakest_targets_by_best_auc_v2_longrun.csv`

## Key Numeric Routing Facts

- Current run-level fact from `TabM v4`:
  - `224` varying numeric features
  - `90` in `piecewise_ok`
  - `134` in `piecewise_bad`
- Local reproduction from the same routing policy:
  - `varying_num_features = {varying_num}`
  - `piecewise_ok = {piecewise_ok}`
  - `piecewise_bad = {piecewise_bad}`

Interpretation:
- the numeric block is much more quasi-discrete / coarse than a naive continuous-treatment assumption would suggest
- this supports smarter numeric routing over uniformly stronger piecewise treatment

## Dataset Snapshot

- labeled rows: `{len(labeled):,}`
- train rows in holdout split: `{len(train_df):,}`
- validation rows in holdout split: `{len(val_df):,}`
- total features used by current backbone: `{len(feature_cols)}`
- numeric features: `{len(num_cols)}`
- categorical features: `{len(cat_cols)}`

## Target Summary

{markdown_table(target_summary_df.sort_values("positive_rate"), max_rows=20)}

## Per-Feature Summary

Full file:
- `artifacts/dataset_feature_diagnostics_v1/feature_summary.csv`

Top rows:

{markdown_table(feature_summary_df.head(20), max_rows=20)}

## Weak-Target Feature Signal

Weak targets:
- {", ".join(weak_targets)}

Full files:
- `artifacts/dataset_feature_diagnostics_v1/feature_target_signal_weak12.csv`
- `artifacts/dataset_feature_diagnostics_v1/weak_target_top_features.csv`

Top feature profiles for weak targets:

{markdown_table(weak_target_feature_head, max_rows=40)}

## Piecewise-Bad Audit

Full file:
- `artifacts/dataset_feature_diagnostics_v1/piecewise_bad_feature_audit.csv`

Strongest `piecewise_bad` features on the weak block:

{markdown_table(top_bad_strong, max_rows=15)}

## Missingness As Signal

Use these columns in `feature_target_signal_weak12.csv`:
- `missing_ratio`
- `missingness_auc`
- `missingness_gain`

Interpretation guide:
- if `missingness_gain` is large on weak targets, explicit missing indicators should move up in priority
- if strong weak-target features are mostly `piecewise_bad` and also have missingness signal, simple-path numeric routing becomes even more plausible

## Label Dependency Matrix

Full files:
- `artifacts/dataset_feature_diagnostics_v1/label_dependency_pairwise.csv`
- `artifacts/dataset_feature_diagnostics_v1/label_dependency_family_summary.csv`

Top pairwise dependencies:

{markdown_table(dependency_head, max_rows=20)}

Family-level summary:

{markdown_table(label_family_df.head(15), max_rows=15)}

## Numeric Redundancy Map

Full files:
- `artifacts/dataset_feature_diagnostics_v1/numeric_redundancy_pairs.csv`
- `artifacts/dataset_feature_diagnostics_v1/numeric_redundancy_clusters.csv`

Top redundant numeric clusters:

{markdown_table(cluster_head, max_rows=15)}

## Train / Validation Feature Stability

Full files:
- `artifacts/dataset_feature_diagnostics_v1/feature_stability_train_vs_val.csv`
- `artifacts/dataset_feature_diagnostics_v1/feature_stability_positive_weak_targets.csv`

Top unstable numeric features overall:

{markdown_table(stability_head, max_rows=20)}

## TabM Error Geography On Weakest 6 Targets

Weakest 6:
- {", ".join(weakest6)}

Full file:
- `artifacts/dataset_feature_diagnostics_v1/tabm_error_geography_weak6.csv`

Worst feature-bin regimes:

{markdown_table(error_head, max_rows=24)}

Interpretation guide:
- positive `tabm_minus_best_alt` means there is a local regime where another model is better on absolute error
- if these regimes are compact and repeatable, object-level gate or local specialist logic stays alive
- if these regimes are weak or diffuse, spend effort on backbone / feature-view instead

## Additional Parameters Worth Tracking

Added because they can affect metric decisions directly:
- target positive rate / class imbalance
- per-feature missingness signal, not only raw missing ratio
- quantile-bin target-rate range (`qbin_rate_range`) for weak-target top features
- redundancy cluster membership for numeric features
- train/validation PSI and KS for numeric features
- local error regimes where `TabM` loses to `GANDALF` / specialist models

## Operational Conclusions

1. If strong weak-target signal is concentrated in `piecewise_bad` features, `hybrid numeric path` gains credibility.
2. If top weak-target features show high `qbin_rate_range` and weak raw AUC, `feature-view augmentation` should move up.
3. If missingness carries signal on weak targets, explicit missing indicators deserve a test.
4. If label lift / PMI is strong inside families or around weak targets, structured decoders and target-meta layers become more plausible.
5. If error geography is compact, narrow gating remains alive; if diffuse, keep pushing backbone quality.

## Files

- report: `artifacts/dataset_feature_diagnostics_v1/dataset_feature_diagnostics.md`
- feature summary: `artifacts/dataset_feature_diagnostics_v1/feature_summary.csv`
- numeric partition: `artifacts/dataset_feature_diagnostics_v1/numeric_feature_partition.csv`
- feature-target signal: `artifacts/dataset_feature_diagnostics_v1/feature_target_signal_weak12.csv`
- weak target top features: `artifacts/dataset_feature_diagnostics_v1/weak_target_top_features.csv`
- piecewise-bad audit: `artifacts/dataset_feature_diagnostics_v1/piecewise_bad_feature_audit.csv`
- label dependency pairwise: `artifacts/dataset_feature_diagnostics_v1/label_dependency_pairwise.csv`
- label dependency family: `artifacts/dataset_feature_diagnostics_v1/label_dependency_family_summary.csv`
- redundancy pairs: `artifacts/dataset_feature_diagnostics_v1/numeric_redundancy_pairs.csv`
- redundancy clusters: `artifacts/dataset_feature_diagnostics_v1/numeric_redundancy_clusters.csv`
- feature stability: `artifacts/dataset_feature_diagnostics_v1/feature_stability_train_vs_val.csv`
- positive-row stability: `artifacts/dataset_feature_diagnostics_v1/feature_stability_positive_weak_targets.csv`
- error geography: `artifacts/dataset_feature_diagnostics_v1/tabm_error_geography_weak6.csv`
"""

    (ARTIFACT_DIR / "dataset_feature_diagnostics.md").write_text(report)
    print(f"Wrote {(ARTIFACT_DIR / 'dataset_feature_diagnostics.md')}")


if __name__ == "__main__":
    main()
