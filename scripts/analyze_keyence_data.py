"""
Comprehensive overview of total Keyence data: counts and surface area (biomass) stats
overall, by treatment (condition), and by month. Same output format as
analyze_spinning_disk_data.py. Loads biofilm/release images and computes surface
area from biofilm preprocessing.

Usage:
  python scripts/analyze_keyence_data.py
  python scripts/analyze_keyence_data.py --root_dir data/keyence/processed --output summary.csv --plots
"""

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import from src for biofilm preprocessing
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import cv2
from biofilm_preprocess import preprocess_biofilm, get_surface_area


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


def load_keyence_data(root_dir, threshold_method, blur_method, enhancement_method):
    """
    Load all Keyence pairs (biofilm path, surface area) in same order as dataset.
    Surface area is computed from biofilm image via preprocess_biofilm + get_surface_area.
    """
    biofilm_root = Path(root_dir) / "biofilm"
    release_root = Path(root_dir) / "release"
    biofilm_paths = sorted(
        [p for p in biofilm_root.rglob("*.tif")],
        key=lambda p: p.as_posix().casefold(),
    )
    release_paths = sorted(
        [p for p in release_root.rglob("*.tif")],
        key=lambda p: p.as_posix().casefold(),
    )
    biofilm_list = []
    release_list = []
    for p in biofilm_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            biofilm_list.append((p, img))
    for p in release_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is not None:
            release_list.append((p, img))
    n = min(len(biofilm_list), len(release_list))
    rows = []
    for i in range(n):
        bp, biofilm_img = biofilm_list[i]
        rp, _ = release_list[i]
        preprocessed = preprocess_biofilm(
            biofilm_img,
            enhancement_method,
            blur_method,
        )
        surface_area = get_surface_area(preprocessed, threshold_method)
        filename = bp.name
        rows.append({
            "filename": filename,
            "biomass": float(surface_area),
            "condition": parse_condition(filename),
            "month": parse_month(filename),
        })
    return rows


def run_analysis(rows, out_csv=None, data_label="KEYENCE"):
    """Print overview and by-condition/by-month tables; optionally write summary CSV. Same format as spinning disk."""
    n = len(rows)
    if n == 0:
        print(f"No {data_label} data.")
        return None, None

    biomass = np.array([r["biomass"] for r in rows])
    by_cond = defaultdict(list)
    by_month = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r["biomass"])
        by_month[r["month"]].append(r["biomass"])

    print("=" * 70)
    print(f"{data_label} DATA OVERVIEW (total)")
    print("=" * 70)
    print(f"  Total images: {n}")
    print(f"  Biomass (surface area): min={biomass.min():.4f}  max={biomass.max():.4f}  "
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


def save_plots(rows, cond_summary, month_summary, out_dir, data_label="Keyence"):
    """Save bar charts: count by condition, count by month, biomass histogram. Same as spinning disk but Keyence titles."""
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
    ax.set_title(f"{data_label} data: image count by treatment")
    fig.tight_layout()
    fig.savefig(out_dir / "keyence_count_by_condition.png", dpi=120)
    plt.close(fig)

    # 2. Count by month
    fig, ax = plt.subplots(figsize=(8, 4))
    months = [m[0] for m in month_summary]
    counts = [m[1] for m in month_summary]
    ax.bar(months, counts, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Number of images")
    ax.set_title(f"{data_label} data: image count by month")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "keyence_count_by_month.png", dpi=120)
    plt.close(fig)

    # 3. Biomass histogram (overall)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(biomass, bins=25, edgecolor="black", alpha=0.7)
    ax.axvline(biomass.mean(), color="red", linestyle="--", label=f"Mean = {biomass.mean():.2f}")
    ax.set_xlabel("Biomass (surface area)")
    ax.set_ylabel("Count")
    ax.set_title(f"{data_label} data: biomass distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "keyence_biomass_histogram.png", dpi=120)
    plt.close(fig)

    print(f"Saved 3 plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Overview of total Keyence data (counts and biomass by treatment/month). Same output as analyze_spinning_disk_data.py.")
    parser.add_argument(
        "--root_dir",
        default=None,
        help="Processed data root (default: DATA_ROOT or data/keyence/processed)",
    )
    parser.add_argument("--output", default=None, help="Optional: write condition summary to this CSV")
    parser.add_argument("--plots", action="store_true", help="Save bar charts and histogram to keyence_plots/")
    parser.add_argument(
        "--threshold_method",
        default="otsu",
        choices=("isodata", "otsu", "adaptive"),
        help="Biofilm threshold method for surface area (default: otsu)",
    )
    parser.add_argument(
        "--blur_method",
        default="gaussian",
        choices=("gaussian", "median", "none"),
        help="Biofilm blur method (default: gaussian)",
    )
    parser.add_argument(
        "--enhancement_method",
        default="clahe",
        choices=("clahe", "histogram_eq", "none"),
        help="Biofilm enhancement method (default: clahe)",
    )
    args = parser.parse_args()

    root_dir = args.root_dir or os.environ.get("DATA_ROOT") or "data/keyence/processed"
    rows = load_keyence_data(
        root_dir,
        threshold_method=args.threshold_method,
        blur_method=args.blur_method,
        enhancement_method=args.enhancement_method,
    )
    if not rows:
        print("No data loaded. Check root_dir and biofilm/ + release/ .tif images.")
        return

    cond_summary, month_summary = run_analysis(rows, out_csv=args.output, data_label="KEYENCE")

    if args.plots and cond_summary is not None:
        plot_dir = Path("keyence_plots")
        save_plots(rows, cond_summary, month_summary, plot_dir, data_label="Keyence")


if __name__ == "__main__":
    main()
