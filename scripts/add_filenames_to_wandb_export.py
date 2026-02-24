"""
Add a filename column to a W&B test predictions export CSV by matching
"actual" to biomass in the biofilm CSV (release_biomass_mapping.csv).
Writes a new CSV with the original columns plus filename.

Usage:
  python scripts/add_filenames_to_wandb_export.py \\
    --csv scripts/wandb_export_test_actual_vs_predicted_kind-sweep-45.csv \\
    --root_dir data/spinning_disk/processed \\
    --output scripts/wandb_export_with_filenames.csv
"""

import argparse
import csv
import os
import sys
from pathlib import Path

# Reuse same tolerance as map_test_predictions_by_actual.py
BIOMASS_MATCH_TOL = 0.01


def load_biomass_csv(root_dir):
    """Load (filename, biomass) from biofilm/release_biomass_mapping.csv."""
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
    """Read W&B export CSV; return list of dicts with string values for writing back."""
    out = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "actual": row["actual"],
                "predicted": row["predicted"],
                "percent_difference": row["percent_difference"],
            })
    return out


def merge_by_actual(predictions, biomass_rows):
    """Match each prediction row to a filename by actual ~ biomass; return list of dicts with filename added."""
    used = [False] * len(biomass_rows)
    merged = []

    for i, pred in enumerate(predictions):
        actual = float(pred["actual"])
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
                f"Error: no matching biomass for prediction row {i} (actual={actual}).",
                file=sys.stderr,
            )
            sys.exit(1)
        filename = biomass_rows[best_j][0]
        used[best_j] = True
        merged.append({
            "filename": filename,
            "actual": pred["actual"],
            "predicted": pred["predicted"],
            "percent_difference": pred["percent_difference"],
        })

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Add filename column to W&B predictions CSV using biomass match."
    )
    parser.add_argument("--csv", required=True, help="Path to W&B export CSV")
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Processed data root with biofilm/release_biomass_mapping.csv",
    )
    parser.add_argument("--output", required=True, help="Output CSV path (with filename column added)")
    args = parser.parse_args()

    root_dir = args.root_dir or os.environ.get("DATA_ROOT") or "data/spinning_disk/processed"
    biomass_rows = load_biomass_csv(root_dir)
    predictions = read_predictions_csv(args.csv)

    if len(predictions) > len(biomass_rows):
        print("Error: more prediction rows than biomass CSV rows.", file=sys.stderr)
        sys.exit(1)

    merged = merge_by_actual(predictions, biomass_rows)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "actual", "predicted", "percent_difference"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
