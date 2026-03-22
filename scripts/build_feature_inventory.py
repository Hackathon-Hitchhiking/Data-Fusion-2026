from __future__ import annotations

import argparse

from lib.feature_selection_v1 import build_feature_inventory_df, ensure_out_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build feature inventory for feature_selection_v1.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--data-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = ensure_out_dir(args.out_dir)
    inventory = build_feature_inventory_df(data_dir=args.data_dir)
    inventory.to_csv(out_dir / "feature_inventory.csv", index=False)
    print({"out_file": str(out_dir / "feature_inventory.csv"), "rows": int(len(inventory))}, flush=True)


if __name__ == "__main__":
    main()
