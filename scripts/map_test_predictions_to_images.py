"""
Map W&B test predictions CSV rows to exact image identifiers (filename or path)
and print basic analysis to the terminal.

Replicates the same data load and train/test split as get_kfold_data() in
src/dataset.py so that CSV row order matches the test set order.

Usage:
  python scripts/map_test_predictions_to_images.py \\
    --csv scripts/wandb_export_test_actual_vs_predicted_kind-sweep-45.csv \\
    --data_source spinning_disk \\
    --root_dir data/spinning_disk/processed

  root_dir can be omitted if DATA_ROOT is set; defaults by data_source otherwise.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

# Optional: tifffile/cv2 only needed for loaders that touch images.
# Spinning disk only needs CSV + split; keyence needs cv2 for path filtering.
try:
    import cv2
except ImportError:
    cv2 = None
try:
    import tifffile
except ImportError:
    tifffile = None


def _load_spinning_disk_filenames(root):
    """
    Return (pairs, filenames) in the same order as dataset._load_spinning_disk_2d_pairs.
    Pairs are (image, biomass); we need pairs only for count/order to match the split.
    """
    csv_path = Path(root) / "biofilm" / "release_biomass_mapping.csv"
    image_dir = Path(root) / "release" / "2D"
    pairs = []
    filenames = []
    skipped = 0
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            biomass_str = row["biomass"].strip()
            if not biomass_str:
                skipped += 1
                continue
            biomass = float(biomass_str)
            img_path = image_dir / filename
            try:
                img = tifffile.imread(str(img_path))
            except Exception:
                skipped += 1
                continue
            pairs.append((img, biomass))
            filenames.append(filename)
    return pairs, filenames


def _load_keyence_identifiers(root):
    """
    Return (raw_pairs, identifiers) in the same order as get_kfold_data for keyence.
    raw_pairs = list(zip(biofilm_images, release_images)); identifiers = release path per pair.
    """
    if cv2 is None:
        raise ImportError("cv2 is required for keyence data source")
    biofilm_root = Path(root) / "biofilm"
    release_root = Path(root) / "release"
    biofilm_paths = sorted(
        [p for p in biofilm_root.rglob("*.tif")],
        key=lambda p: p.as_posix().casefold(),
    )
    release_paths = sorted(
        [p for p in release_root.rglob("*.tif")],
        key=lambda p: p.as_posix().casefold(),
    )
    biofilm_images = []
    biofilm_ok_paths = []
    for p in biofilm_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            biofilm_images.append(img)
            biofilm_ok_paths.append(p)
    release_images = []
    release_ok_paths = []
    for p in release_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            release_images.append(img)
            release_ok_paths.append(p)
    n = min(len(biofilm_images), len(release_images))
    raw_pairs = list(zip(biofilm_images[:n], release_images[:n]))
    identifiers = [str(release_ok_paths[i]) for i in range(n)]
    return raw_pairs, identifiers


def get_test_identifiers(data_source, root_dir):
    """Return list of identifiers (filename or path) in test set order."""
    root = Path(root_dir)
    if data_source == "spinning_disk":
        if tifffile is None:
            raise ImportError("tifffile is required for spinning_disk data source")
        raw_pairs, identifiers = _load_spinning_disk_filenames(root)
    elif data_source == "keyence":
        raw_pairs, identifiers = _load_keyence_identifiers(root)
    else:
        raise ValueError(f"Unknown data_source: {data_source}")

    indices = np.arange(len(raw_pairs))
    _trainval_idx, test_idx = train_test_split(
        indices, train_size=0.9, random_state=42, shuffle=True
    )
    test_identifiers = [identifiers[i] for i in test_idx]
    return test_identifiers


def read_predictions_csv(csv_path):
    """Read W&B export CSV; return list of dicts with actual, predicted, percent_difference."""
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["actual"] = float(row["actual"])
            row["predicted"] = float(row["predicted"])
            row["percent_difference"] = float(row["percent_difference"])
            rows.append(row)
    return rows


def run_analysis(merged):
    """Print summary stats, worst/best N, over/under prediction analysis."""
    n = len(merged)
    pct = [m["percent_difference"] for m in merged]
    actual = np.array([m["actual"] for m in merged])
    pred = np.array([m["predicted"] for m in merged])
    abs_err = np.abs(pred - actual)

    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"  Test images: {n}")
    print(f"  percent_difference: mean={np.mean(pct):.4f}  median={np.median(pct):.4f}  "
          f"std={np.std(pct):.4f}  min={np.min(pct):.4f}  max={np.max(pct):.4f}")
    print(f"  |actual - predicted|: mean={np.mean(abs_err):.4f}  median={np.median(abs_err):.4f}  "
          f"max={np.max(abs_err):.4f}")

    print()
    print("=" * 60)
    print("WORST 10 (largest percent_difference)")
    print("=" * 60)
    sorted_by_pct = sorted(merged, key=lambda m: m["percent_difference"], reverse=True)
    for i, m in enumerate(sorted_by_pct[:10], 1):
        print(f"  {i:2}. {m['identifier']}")
        print(f"      actual={m['actual']:.4f}  predicted={m['predicted']:.4f}  "
              f"percent_difference={m['percent_difference']:.4f}")

    print()
    print("=" * 60)
    print("BEST 10 (smallest percent_difference)")
    print("=" * 60)
    for i, m in enumerate(sorted_by_pct[-10:][::-1], 1):
        print(f"  {i:2}. {m['identifier']}")
        print(f"      actual={m['actual']:.4f}  predicted={m['predicted']:.4f}  "
              f"percent_difference={m['percent_difference']:.4f}")

    over = sum(1 for m in merged if m["predicted"] > m["actual"])
    under = sum(1 for m in merged if m["predicted"] < m["actual"])
    print()
    print("=" * 60)
    print("OVER vs UNDER PREDICTION")
    print("=" * 60)
    print(f"  Over-predicted  (predicted > actual): {over}")
    print(f"  Under-predicted (predicted < actual): {under}")

    by_diff = sorted(merged, key=lambda m: m["predicted"] - m["actual"], reverse=True)
    print()
    print("  5 most over-predicted (predicted - actual):")
    for m in by_diff[:5]:
        print(f"    {m['identifier']}  diff={m['predicted'] - m['actual']:.4f}")

    print()
    print("  5 most under-predicted (predicted - actual):")
    for m in by_diff[-5:][::-1]:
        print(f"    {m['identifier']}  diff={m['predicted'] - m['actual']:.4f}")

    # Simple grouping by first filename segment (before _ or .)
    print()
    print("=" * 60)
    print("MEAN percent_difference BY FILENAME PREFIX (before first _ or .)")
    print("=" * 60)
    groups = defaultdict(list)
    for m in merged:
        name = Path(m["identifier"]).name
        prefix = name.split("_")[0].split(".")[0] or name
        groups[prefix].append(m["percent_difference"])
    for prefix in sorted(groups.keys()):
        vals = groups[prefix]
        print(f"  {prefix}: n={len(vals)}  mean={np.mean(vals):.4f}  median={np.median(vals):.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Map test prediction CSV rows to image identifiers and print analysis."
    )
    parser.add_argument("--csv", required=True, help="Path to W&B export CSV (actual, predicted, percent_difference)")
    parser.add_argument("--data_source", required=True, choices=("spinning_disk", "keyence"))
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Data root used at training (default: DATA_ROOT or data/<data_source>/processed)",
    )
    args = parser.parse_args()

    root_dir = args.root_dir or os.environ.get("DATA_ROOT")
    if root_dir is None:
        if args.data_source in ("keyence", "keyence_clearest"):
            root_dir = "data/keyence/processed"
        else:
            root_dir = "data/spinning_disk/processed"

    test_identifiers = get_test_identifiers(args.data_source, root_dir)
    rows = read_predictions_csv(args.csv)

    if len(rows) != len(test_identifiers):
        print(
            f"Error: CSV has {len(rows)} rows but test set has {len(test_identifiers)} images. "
            "Row order must match test set order.",
            file=sys.stderr,
        )
        sys.exit(1)

    merged = []
    for i, row in enumerate(rows):
        merged.append({
            "image_idx": i,
            "identifier": test_identifiers[i],
            "actual": row["actual"],
            "predicted": row["predicted"],
            "percent_difference": row["percent_difference"],
        })

    run_analysis(merged)


if __name__ == "__main__":
    main()
