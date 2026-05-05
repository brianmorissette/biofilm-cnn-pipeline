"""
Sanity-check / dataset-summary script for the Spinning Disk dataset.

WHEN TO USE THIS SCRIPT
-----------------------
Run this whenever you want a quick health check on the spinning-disk dataset,
for example:
- on a fresh checkout of the data directory, to confirm
  `release_biomass_mapping.csv` and the `release/2D/` images are aligned,
- before kicking off a sweep, to sanity-check that the biomass distribution
  looks reasonable,
- when debugging odd training results and you want to know if a particular
  treatment or month is over/under-represented.

This script is NOT part of the training pipeline — it never trains a model.
Skip it if you just want to run sweeps.

WHAT IT DOES
------------
Reads `data/spinning_disk/processed/biofilm/release_biomass_mapping.csv` (the
ground-truth biomass labels) and prints summary tables of:
- total image counts,
- counts and biomass stats grouped by treatment (condition),
- counts and biomass stats grouped by month.

Unlike the Keyence version, this script does not need to load any TIFFs —
biomass labels are pre-computed and stored in the CSV. Optional flags can
dump the summary as CSV and produce histograms of the biomass distribution.

Usage:
  python scripts/analyze_spinning_disk_data.py
  python scripts/analyze_spinning_disk_data.py --root_dir data/spinning_disk/processed --output summary.csv --plots
"""

import argparse
import csv
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_condition(filename):
    """Extract condition; group -02 etc. with base name (e.g. ProtK-02 -> ProtK)."""
    name = Path(filename).stem
    m = re.search(r"_([^_]+?Cells[^_]*)", name)
    if m:
        raw = m.group(1).replace("Cells1to10Dilution", "").strip("-_") or "unknown"
        return re.sub(r"-\d+$", "", raw) or raw
    parts = name.split("_")
    base = parts[1] if len(parts) > 1 else "unknown"
    return re.sub(r"-\d+$", "", base) or base


def parse_month(filename):
    """Return month label from filename, e.g. Dec2025, Oct2025."""
    name = Path(filename).stem
    m = re.search(r"(\d{2}[A-Za-z]{3}\d{4})(?:-\d+)?", name)
    if m:
        d = m.group(1)
        return d[2:] if len(d) >= 8 else d
    return "unknown"


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
            rows.append({
                "filename": row["filename"],
                "biomass": float(biomass_str),
                "condition": parse_condition(row["filename"]),
                "month": parse_month(row["filename"]),
            })
    return rows


def run_analysis(rows, out_csv=None):
    """Print overview and by-condition/by-month tables; optionally write summary CSV."""
    n = len(rows)
    if n == 0:
        print("No rows in biomass CSV.")
        return

    biomass = np.array([r["biomass"] for r in rows])
    by_cond = defaultdict(list)
    by_month = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["biomass"])
        by_month[r["month"]].append(r["biomass"])

    print("=" * 70)
    print("SPINNING DISK DATA OVERVIEW (total)")
    print("=" * 70)
    print(f"  Total images: {n}")
    print(f"  Biomass: min={biomass.min():.4f}  max={biomass.max():.4f}  "
          f"mean={biomass.mean():.4f}  median={np.median(biomass):.4f}  std={biomass.std():.4f}")

    print()
    print("=" * 70)
    print("BY TREATMENT (CONDITION)")
    print("=" * 70)
    cond_summary = []
    for cond in sorted(by_cond.keys()):
        vals = np.array(by_cond[cond])
        count = len(vals)
        pct = 100.0 * count / n
        cond_summary.append({
            "condition": cond,
            "count": count,
            "pct": pct,
            "mean_biomass": vals.mean(),
            "min_biomass": vals.min(),
            "max_biomass": vals.max(),
        })
    cond_summary.sort(key=lambda x: x["count"], reverse=True)
    print(f"  {'Condition':<18}  {'Count':>6}  {'%':>7}  {'Mean biomass':>12}  {'Min':>10}  {'Max':>10}")
    print("  " + "-" * 68)
    for s in cond_summary:
        print(f"  {s['condition']:<18}  {s['count']:>6}  {s['pct']:>6.1f}%  {s['mean_biomass']:>12.4f}  {s['min_biomass']:>10.4f}  {s['max_biomass']:>10.4f}")

    print()
    print("=" * 70)
    print("BY MONTH (from filename date)")
    print("=" * 70)
    month_summary = []
    for month in sorted(by_month.keys()):
        vals = np.array(by_month[month])
        count = len(vals)
        pct = 100.0 * count / n
        month_summary.append((month, count, pct, vals.mean()))
    print(f"  {'Month':<14}  {'Count':>6}  {'%':>7}  {'Mean biomass':>12}")
    print("  " + "-" * 44)
    for month, count, pct, mean_b in month_summary:
        print(f"  {month:<14}  {count:>6}  {pct:>6.1f}%  {mean_b:>12.4f}")

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["condition", "count", "pct", "mean_biomass", "min_biomass", "max_biomass"])
            w.writeheader()
            w.writerows(cond_summary)
        print()
        print(f"Wrote condition summary to {out_csv}")

    return cond_summary, month_summary


def save_plots(rows, cond_summary, month_summary, out_dir):
    """Save bar charts: count by condition, count by month, biomass histogram."""
    import matplotlib.pyplot as plt

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    biomass = np.array([r["biomass"] for r in rows])

    # 1. Count by condition
    fig, ax = plt.subplots(figsize=(8, max(4, len(cond_summary) * 0.35)))
    conds = [s["condition"] for s in cond_summary]
    counts = [s["count"] for s in cond_summary]
    ax.barh(conds, counts, edgecolor="black", alpha=0.8)
    ax.set_xlabel("Number of images")
    ax.set_title("Spinning disk data: image count by treatment")
    fig.tight_layout()
    fig.savefig(out_dir / "spinning_disk_count_by_condition.png", dpi=120)
    plt.close(fig)

    # 2. Count by month
    fig, ax = plt.subplots(figsize=(8, 4))
    months = [m[0] for m in month_summary]
    counts = [m[1] for m in month_summary]
    ax.bar(months, counts, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Number of images")
    ax.set_title("Spinning disk data: image count by month")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "spinning_disk_count_by_month.png", dpi=120)
    plt.close(fig)

    # 3. Biomass histogram (overall)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(biomass, bins=25, edgecolor="black", alpha=0.7)
    ax.axvline(biomass.mean(), color="red", linestyle="--", label=f"Mean = {biomass.mean():.2f}")
    ax.set_xlabel("Biomass")
    ax.set_ylabel("Count")
    ax.set_title("Spinning disk data: biomass distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "spinning_disk_biomass_histogram.png", dpi=120)
    plt.close(fig)

    print(f"Saved 3 plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Overview of total spinning disk data (counts and biomass by treatment/month).")
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Processed data root (default: DATA_ROOT or data/spinning_disk/processed)",
    )
    parser.add_argument("--output", default=None, help="Optional: write condition summary to this CSV")
    parser.add_argument("--plots", action="store_true", help="Save bar charts and histogram to spinning_disk_plots/")
    args = parser.parse_args()

    root_dir = args.root_dir or os.environ.get("DATA_ROOT") or "data/spinning_disk/processed"
    rows = load_biomass_csv(root_dir)
    if not rows:
        print("No data loaded. Check root_dir and biofilm/release_biomass_mapping.csv.")
        return

    cond_summary, month_summary = run_analysis(rows, out_csv=args.output)

    if args.plots and cond_summary is not None:
        plot_dir = Path("spinning_disk_plots")
        save_plots(rows, cond_summary, month_summary, plot_dir)


if __name__ == "__main__":
    main()
