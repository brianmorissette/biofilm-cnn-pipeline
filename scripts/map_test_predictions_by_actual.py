"""
Map W&B test predictions to filenames by matching "actual" to biomass in the
biofilm CSV (release_biomass_mapping.csv). Prints the same analysis as
map_test_predictions_to_images.py so you can verify the two strategies agree.

Strategy: for each row in the predictions CSV, find a row in the biofilm CSV
where biomass == actual (float tolerance). When multiple images share the same
biomass, the first unused CSV row is assigned (in CSV order).

Usage:
  python scripts/map_test_predictions_by_actual.py \\
    --csv scripts/wandb_export_test_actual_vs_predicted_kind-sweep-45.csv \\
    --root_dir data/spinning_disk/processed
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# Float tolerance for matching actual to biomass (CSV or W&B export may round)
BIOMASS_MATCH_TOL = 0.01


def load_biomass_csv(root_dir):
    """
    Load (filename, biomass) from biofilm/release_biomass_mapping.csv.
    Returns list of (filename, biomass) in CSV row order. Skips rows with
    empty biomass (same as dataset loader).
    """
    csv_path = Path(root_dir) / "biofilm" / "release_biomass_mapping.csv"
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            biomass_str = row["biomass"].strip()
            if not biomass_str:
                continue
            biomass = float(biomass_str)
            rows.append((row["filename"], biomass))
    return rows


def read_predictions_csv(csv_path):
    """Read W&B export CSV; return list of dicts with actual, predicted, percent_difference."""
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "actual": float(row["actual"]),
                "predicted": float(row["predicted"]),
                "percent_difference": float(row["percent_difference"]),
            })
    return out


def merge_by_actual(predictions, biomass_rows):
    """
    For each prediction row, find the closest unused (filename, biomass) where
    |biomass - actual| <= BIOMASS_MATCH_TOL. Use closest match to avoid
    wrong assignment when multiple rows fall in the tolerance band.
    """
    used = [False] * len(biomass_rows)
    merged = []

    for i, pred in enumerate(predictions):
        actual = pred["actual"]
        best_j = None
        best_dist = float("inf")
        for j, (filename, biomass) in enumerate(biomass_rows):
            if used[j]:
                continue
            dist = abs(biomass - actual)
            if dist <= BIOMASS_MATCH_TOL and dist < best_dist:
                best_j = j
                best_dist = dist
        if best_j is None:
            print(
                f"Error: no matching biomass for prediction row {i} (actual={actual}). "
                f"Closest in CSV: {min(abs(b - actual) for _, b in biomass_rows):.6f}.",
                file=sys.stderr,
            )
            sys.exit(1)
        filename, biomass = biomass_rows[best_j]
        used[best_j] = True
        merged.append({
            "image_idx": i,
            "identifier": filename,
            "actual": pred["actual"],
            "predicted": pred["predicted"],
            "percent_difference": pred["percent_difference"],
        })

    return merged


def run_analysis(merged):
    """Print summary stats, worst/best N, over/under prediction analysis (same as other script)."""
    n = len(merged)
    pct = [m["percent_difference"] for m in merged]
    actual = np.array([m["actual"] for m in merged])
    pred = np.array([m["predicted"] for m in merged])
    abs_err = np.abs(pred - actual)

    print("=" * 60)
    print("SUMMARY STATISTICS (strategy: match actual to biomass in biofilm CSV)")
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
        description="Map predictions to filenames by matching actual to biomass in biofilm CSV; print same analysis."
    )
    parser.add_argument("--csv", required=True, help="Path to W&B export CSV (actual, predicted, percent_difference)")
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Processed data root containing biofilm/release_biomass_mapping.csv (default: DATA_ROOT or data/spinning_disk/processed)",
    )
    args = parser.parse_args()

    root_dir = args.root_dir or os.environ.get("DATA_ROOT") or "data/spinning_disk/processed"
    biomass_rows = load_biomass_csv(root_dir)
    predictions = read_predictions_csv(args.csv)

    if len(predictions) > len(biomass_rows):
        print(
            f"Error: predictions has {len(predictions)} rows but biomass CSV has {len(biomass_rows)}. "
            "Predictions should be test set only.",
            file=sys.stderr,
        )
        sys.exit(1)

    merged = merge_by_actual(predictions, biomass_rows)
    run_analysis(merged)


if __name__ == "__main__":
    main()
