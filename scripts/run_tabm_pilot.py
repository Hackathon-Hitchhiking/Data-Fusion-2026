from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rtdl_num_embeddings import LinearReLUEmbeddings
from sklearn.metrics import roc_auc_score
from tabm import TabM
from torch.utils.data import DataLoader, TensorDataset

from lib.layout import project_root, resolve_data_dir
from lib.metrics import macro_auc
from lib.submission import normalize_prediction_columns


ROOT = project_root()
DATA_DIR = resolve_data_dir()
TOP_EXTRA_FILE = ROOT / "artifacts" / "feature_selection" / "top100_extra_gain.json"
GANDALF_VAL_FILE = ROOT / "artifacts" / "gpu_gandalf_full_gpu" / "validation_predictions.parquet"
DEFAULT_OUT_DIR = ROOT / "artifacts" / "tabm_pilot_v1"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local TabM pilot on the GANDALF holdout split")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--subsample-train", type=int, default=120000)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--eval-batch-size", type=int, default=4096)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use-num-embeddings", action="store_true")
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_frames() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str], list[str]]:
    train_main = pd.read_parquet(DATA_DIR / "train_main_features.parquet")
    train_extra = pd.read_parquet(DATA_DIR / "train_extra_features.parquet")
    train_target = pd.read_parquet(DATA_DIR / "train_target.parquet")
    top_extra = json.loads(TOP_EXTRA_FILE.read_text())

    frame = train_main.merge(train_target, on="customer_id", how="inner")
    frame = frame.merge(train_extra[["customer_id"] + top_extra], on="customer_id", how="left")

    target_cols = [c for c in frame.columns if c.startswith("target_")]
    feature_cols = [c for c in frame.columns if c not in ["customer_id"] + target_cols]
    cat_cols = [c for c in feature_cols if c.startswith("cat_feature")]
    num_cols = [c for c in feature_cols if c not in cat_cols]

    val_ids = normalize_prediction_columns(pd.read_parquet(GANDALF_VAL_FILE))[["customer_id"]]
    val = frame.merge(val_ids, on="customer_id", how="inner").sort_values("customer_id").reset_index(drop=True)
    train = frame[~frame["customer_id"].isin(set(val_ids["customer_id"]))].sort_values("customer_id").reset_index(drop=True)
    return train, val, feature_cols, num_cols, cat_cols


def prepare_arrays(
    train: pd.DataFrame,
    val: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    target_cols: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[int]]:
    train_num = train[num_cols].copy()
    val_num = val[num_cols].copy()
    train_cat = train[cat_cols].copy()
    val_cat = val[cat_cols].copy()

    for col in num_cols:
        median = float(pd.to_numeric(train_num[col], errors="coerce").median())
        train_num[col] = pd.to_numeric(train_num[col], errors="coerce").fillna(median)
        val_num[col] = pd.to_numeric(val_num[col], errors="coerce").fillna(median)
        mean = float(train_num[col].mean())
        std = float(train_num[col].std())
        std = std if std > 1e-6 else 1.0
        train_num[col] = ((train_num[col] - mean) / std).astype("float32")
        val_num[col] = ((val_num[col] - mean) / std).astype("float32")

    cat_cardinalities: list[int] = []
    for col in cat_cols:
        train_series = pd.to_numeric(train_cat[col], errors="coerce").fillna(-1).astype("int64")
        val_series = pd.to_numeric(val_cat[col], errors="coerce").fillna(-1).astype("int64")
        known_vals = sorted(v for v in train_series.unique().tolist() if v >= 0)
        mapping = {v: i for i, v in enumerate(known_vals)}
        unk = len(mapping)
        train_cat[col] = train_series.map(mapping).fillna(unk).astype("int64")
        val_cat[col] = val_series.map(mapping).fillna(unk).astype("int64")
        cat_cardinalities.append(int(unk + 1))

    train_arrays = {
        "x_num": train_num.to_numpy(dtype=np.float32, copy=True),
        "x_cat": train_cat.to_numpy(dtype=np.int64, copy=True),
        "y": train[target_cols].to_numpy(dtype=np.float32, copy=True),
    }
    val_arrays = {
        "x_num": val_num.to_numpy(dtype=np.float32, copy=True),
        "x_cat": val_cat.to_numpy(dtype=np.int64, copy=True),
        "y": val[target_cols].to_numpy(dtype=np.float32, copy=True),
    }
    return train_arrays, val_arrays, cat_cardinalities


@torch.no_grad()
def predict_proba(model: TabM, loader: DataLoader, device: str) -> np.ndarray:
    model.eval()
    parts = []
    for x_num, x_cat, _ in loader:
        x_num = x_num.to(device)
        x_cat = x_cat.to(device)
        logits = model(x_num, x_cat)
        probs = torch.sigmoid(logits).mean(1)
        parts.append(probs.cpu().numpy())
    return np.concatenate(parts, axis=0)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = pick_device()

    train, val, feature_cols, num_cols, cat_cols = load_frames()
    target_cols = [c for c in train.columns if c.startswith("target_")]
    if args.subsample_train and args.subsample_train < len(train):
        train = train.sample(args.subsample_train, random_state=args.seed).sort_values("customer_id").reset_index(drop=True)

    train_arrays, val_arrays, cat_cardinalities = prepare_arrays(train, val, num_cols, cat_cols, target_cols)

    train_ds = TensorDataset(
        torch.from_numpy(train_arrays["x_num"]),
        torch.from_numpy(train_arrays["x_cat"]),
        torch.from_numpy(train_arrays["y"]),
    )
    val_ds = TensorDataset(
        torch.from_numpy(val_arrays["x_num"]),
        torch.from_numpy(val_arrays["x_cat"]),
        torch.from_numpy(val_arrays["y"]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    num_embeddings = LinearReLUEmbeddings(len(num_cols)) if args.use_num_embeddings else None
    model = TabM.make(
        n_num_features=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        d_out=len(target_cols),
        num_embeddings=num_embeddings,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = -np.inf
    best_pred = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_num, x_cat, y in train_loader:
            x_num = x_num.to(device)
            x_cat = x_cat.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_num, x_cat)
            y_expanded = y.unsqueeze(1).expand(-1, logits.shape[1], -1)
            loss = F.binary_cross_entropy_with_logits(logits, y_expanded)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_pred = predict_proba(model, val_loader, device)
        val_pred_df = pd.DataFrame(val_pred, columns=[c.replace("target_", "predict_") for c in target_cols])
        val_pred_df.insert(0, "customer_id", val["customer_id"].values)
        val_target_df = val[["customer_id"] + target_cols].copy()
        val_macro = float(macro_auc(val_target_df, val_pred_df, target_cols))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_macro_auc": val_macro})
        print({"stage": "epoch_done", "epoch": epoch, "train_loss": float(np.mean(train_losses)), "val_macro_auc": val_macro}, flush=True)
        if val_macro > best_val:
            best_val = val_macro
            best_pred = val_pred_df

    if best_pred is None:
        raise RuntimeError("No validation predictions were produced")

    target_rows = []
    for target_name in target_cols:
        pred_col = target_name.replace("target_", "predict_")
        auc = roc_auc_score(val[target_name], best_pred[pred_col]) if val[target_name].nunique() > 1 else 0.5
        target_rows.append({"target": target_name, "oof_auc": float(auc)})

    pd.DataFrame(history).to_csv(args.out_dir / "history.csv", index=False)
    pd.DataFrame(target_rows).sort_values("oof_auc").to_csv(args.out_dir / "target_scores.csv", index=False)
    best_pred.to_parquet(args.out_dir / "validation_predictions.parquet", index=False)

    summary = {
        "device": device,
        "subsample_train": int(len(train)),
        "val_rows": int(len(val)),
        "feature_count": int(len(feature_cols)),
        "num_features": int(len(num_cols)),
        "cat_features": int(len(cat_cols)),
        "epochs": args.epochs,
        "best_val_macro_auc": float(best_val),
        "use_num_embeddings": bool(args.use_num_embeddings),
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    del model, train_loader, val_loader, train_ds, val_ds
    gc.collect()


if __name__ == "__main__":
    main()
