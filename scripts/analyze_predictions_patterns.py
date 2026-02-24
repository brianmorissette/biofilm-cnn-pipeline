"""
Detailed pattern analysis on the clean predictions CSV (filename, actual, predicted,
percent_difference). Parses filename structure (condition, date) and reports
error patterns by group, biomass level, and bias. Optionally saves simple plots.

Usage:
  python scripts/analyze_predictions_patterns.py scripts/kind-sweep-45_actual_vs_predicted_images.csv
  python scripts/analyze_predictions_patterns.py script.csv --out_dir ./plots
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path):
    """Load predictions CSV; return list of dicts with numeric actual, predicted, percent_difference."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "filename": row["filename"],
                "actual": float(row["actual"]),
                "predicted": float(row["predicted"]),
                "percent_difference": float(row["percent_difference"]),
            })
    return rows


def parse_condition(filename):
    """Extract condition/treatment from filename; group -02 etc. with base name (e.g. ProtK-02 -> ProtK)."""
    name = Path(filename).stem
    # RP62A_NaIO4Cells1to10Dilution_... or RP62A_ProtK-02Cells1to10Dilution_...
    m = re.search(r"_([^_]+?Cells[^_]*)", name)
    if m:
        raw = m.group(1).replace("Cells1to10Dilution", "").strip("-_") or "unknown"
        # Strip trailing -NN so ProtK-02 and ProtK go in the same bin
        return re.sub(r"-\d+$", "", raw) or raw
    # Fallback: first segment after first underscore
    parts = name.split("_")
    base = parts[1] if len(parts) > 1 else "unknown"
    return re.sub(r"-\d+$", "", base) or base


def parse_date(filename):
    """Extract date-like string (DDMonYYYY or DDMonYYYY-NN) for grouping by month."""
    name = Path(filename).stem
    # e.g. 02Dec2025, 09Oct2025-04
    m = re.search(r"(\d{2}[A-Za-z]{3}\d{4})(?:-\d+)?", name)
    if m:
        return m.group(1)
    return "unknown"


def parse_month(filename):
    """Return month label for grouping, e.g. Sept2025, Oct2025, Dec2025."""
    d = parse_date(filename)
    if d == "unknown":
        return "unknown"
    # DDMonYYYY -> MonYYYY
    return d[2:] if len(d) >= 8 else d


def enrich_rows(rows):
    """Add condition, month, abs_error, bias to each row."""
    for r in rows:
        r["condition"] = parse_condition(r["filename"])
        r["date"] = parse_date(r["filename"])
        r["month"] = parse_month(r["filename"])
        r["abs_error"] = abs(r["predicted"] - r["actual"])
        r["bias"] = r["predicted"] - r["actual"]


def run_analysis(rows):
    """Print detailed pattern analyses. Returns a summary dict for plotting."""
    n = len(rows)
    enrich_rows(rows)
    actual = np.array([r["actual"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    pct = np.array([r["percent_difference"] for r in rows])
    abs_err = np.abs(pred - actual)
    bias = pred - actual

    # ----- Build summaries for plots -----
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"]].append(r)

    cond_summary = []
    for cond in sorted(by_cond.keys()):
        sub = by_cond[cond]
        pct_vals = [r["percent_difference"] for r in sub]
        cond_summary.append({
            "condition": cond,
            "n": len(sub),
            "mean_pct": np.mean(pct_vals),
            "median_pct": np.median(pct_vals),
            "mean_abs_err": np.mean([r["abs_error"] for r in sub]),
            "mean_bias": np.mean([r["bias"] for r in sub]),
        })
    cond_summary.sort(key=lambda x: x["mean_pct"], reverse=True)

    q33 = np.percentile(actual, 33)
    q66 = np.percentile(actual, 66)
    low = [r for r in rows if r["actual"] <= q33]
    mid = [r for r in rows if q33 < r["actual"] <= q66]
    high = [r for r in rows if r["actual"] > q66]
    biomass_summary = [
        ("Low", len(low), np.mean([r["percent_difference"] for r in low])),
        ("Mid", len(mid), np.mean([r["percent_difference"] for r in mid])),
        ("High", len(high), np.mean([r["percent_difference"] for r in high])),
    ]

    by_month = defaultdict(list)
    for r in rows:
        by_month[r["month"]].append(r)
    month_summary = []
    for month in sorted(by_month.keys()):
        sub = by_month[month]
        month_summary.append((month, len(sub), np.mean([r["percent_difference"] for r in sub])))

    sorted_by_pct = sorted(rows, key=lambda r: r["percent_difference"], reverse=True)
    worst_20pct = sorted_by_pct[: max(1, n // 5)]
    best_20pct = sorted_by_pct[-max(1, n // 5) :]
    worst_cond = defaultdict(int)
    best_cond = defaultdict(int)
    for r in worst_20pct:
        worst_cond[r["condition"]] += 1
    for r in best_20pct:
        best_cond[r["condition"]] += 1

    over_under = {}
    for cond in by_cond:
        sub = by_cond[cond]
        over_under[cond] = (sum(1 for r in sub if r["bias"] > 0), sum(1 for r in sub if r["bias"] < 0))

    summary = {
        "by_cond": by_cond,
        "cond_summary": cond_summary,
        "biomass_summary": biomass_summary,
        "month_summary": month_summary,
        "worst_cond": worst_cond,
        "best_cond": best_cond,
        "over_under": over_under,
        "q33": q33,
        "q66": q66,
    }

    # ----- Overall summary -----
    print("=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)
    print(f"  N = {n}")
    print(f"  percent_difference: mean={np.mean(pct):.4f}  median={np.median(pct):.4f}  "
          f"std={np.std(pct):.4f}  min={np.min(pct):.4f}  max={np.max(pct):.4f}")
    print(f"  |pred - actual|:     mean={np.mean(abs_err):.4f}  median={np.median(abs_err):.4f}  max={np.max(abs_err):.4f}")
    print(f"  bias (pred - actual): mean={np.mean(bias):.4f}  (positive = over-predict on average)")

    # ----- By condition -----
    print()
    print("=" * 70)
    print("BY CONDITION (treatment / replicate group)")
    print("=" * 70)
    print(f"  {'Condition':<20} {'n':>4}  {'mean_%diff':>10}  {'median_%diff':>12}  {'mean_|err|':>10}  {'mean_bias':>10}")
    print("  " + "-" * 68)
    for s in cond_summary:
        print(f"  {s['condition']:<20} {s['n']:>4}  {s['mean_pct']:>10.4f}  {s['median_pct']:>12.4f}  "
              f"{s['mean_abs_err']:>10.4f}  {s['mean_bias']:>10.4f}")

    # ----- By biomass level (terciles) -----
    q33, q66 = summary["q33"], summary["q66"]
    low = [r for r in rows if r["actual"] <= q33]
    mid = [r for r in rows if q33 < r["actual"] <= q66]
    high = [r for r in rows if r["actual"] > q66]
    print()
    print("=" * 70)
    print("BY BIOMASS LEVEL (terciles of actual)")
    print("=" * 70)
    print(f"  Low  (actual <= {q33:.1f}): n={len(low)}  mean_%diff={np.mean([r['percent_difference'] for r in low]):.4f}  mean_bias={np.mean([r['bias'] for r in low]):.4f}")
    print(f"  Mid  ({q33:.1f} < actual <= {q66:.1f}): n={len(mid)}  mean_%diff={np.mean([r['percent_difference'] for r in mid]):.4f}  mean_bias={np.mean([r['bias'] for r in mid]):.4f}")
    print(f"  High (actual > {q66:.1f}): n={len(high)}  mean_%diff={np.mean([r['percent_difference'] for r in high]):.4f}  mean_bias={np.mean([r['bias'] for r in high]):.4f}")

    # ----- By month -----
    print()
    print("=" * 70)
    print("BY MONTH (from filename date)")
    print("=" * 70)
    for month, count, mean_pct in month_summary:
        sub = by_month[month]
        mean_bias = np.mean([r["bias"] for r in sub])
        print(f"  {month:<12} n={count:>3}  mean_%diff={mean_pct:.4f}  mean_bias={mean_bias:+.4f}")

    # ----- Worst / best 20% by condition -----
    print()
    print("=" * 70)
    print("CONDITION IN WORST 20% vs BEST 20% (count)")
    print("=" * 70)
    all_conds = sorted(set(summary["worst_cond"]) | set(summary["best_cond"]))
    print(f"  {'Condition':<20}  {'Worst_20%':>10}  {'Best_20%':>10}")
    print("  " + "-" * 44)
    for c in all_conds:
        print(f"  {c:<20}  {summary['worst_cond'].get(c, 0):>10}  {summary['best_cond'].get(c, 0):>10}")

    # ----- Worst case per condition -----
    print()
    print("=" * 70)
    print("WORST SINGLE IMAGE PER CONDITION (max percent_difference)")
    print("=" * 70)
    for cond in sorted(by_cond.keys()):
        sub = by_cond[cond]
        worst = max(sub, key=lambda r: r["percent_difference"])
        print(f"  {cond:<20}  {worst['filename']}")
        print(f"    actual={worst['actual']:.4f}  predicted={worst['predicted']:.4f}  %diff={worst['percent_difference']:.4f}")

    # ----- Correlation: actual vs error -----
    corr_pct = np.corrcoef(actual, pct)[0, 1] if n > 1 and np.std(pct) > 0 else float("nan")
    corr_abs = np.corrcoef(actual, abs_err)[0, 1] if n > 1 and np.std(abs_err) > 0 else float("nan")
    print()
    print("=" * 70)
    print("CORRELATION WITH ACTUAL BIOMASS")
    print("=" * 70)
    print(f"  actual vs percent_difference:  r = {corr_pct:.4f}")
    print(f"  actual vs |pred - actual|:     r = {corr_abs:.4f}")

    # ----- Over/under by condition -----
    print()
    print("=" * 70)
    print("OVER vs UNDER PREDICTION BY CONDITION (count)")
    print("=" * 70)
    print(f"  {'Condition':<20}  {'Over (pred>actual)':>18}  {'Under (pred<actual)':>18}")
    print("  " + "-" * 58)
    for cond in sorted(by_cond.keys()):
        over, under = summary["over_under"][cond]
        print(f"  {cond:<20}  {over:>18}  {under:>18}")

    return summary


def save_plots(rows, summary, out_dir):
    """Save simple matplotlib figures to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    actual = np.array([r["actual"] for r in rows])
    pred = np.array([r["predicted"] for r in rows])
    pct = np.array([r["percent_difference"] for r in rows])
    by_cond = summary["by_cond"]
    cond_summary = summary["cond_summary"]
    biomass_summary = summary["biomass_summary"]
    month_summary = summary["month_summary"]
    worst_cond = summary["worst_cond"]
    best_cond = summary["best_cond"]
    over_under = summary["over_under"]

    # 1. Actual vs predicted scatter
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(actual, pred, alpha=0.7, s=25)
    lims = [min(actual.min(), pred.min()), max(actual.max(), pred.max())]
    ax.plot(lims, lims, "k--", label="y=x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    ax.legend()
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "01_actual_vs_predicted.png", dpi=120)
    plt.close(fig)

    # 2. Percent difference histogram
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(pct, bins=20, edgecolor="black", alpha=0.7)
    ax.axvline(np.mean(pct), color="red", linestyle="--", label=f"Mean = {np.mean(pct):.3f}")
    ax.set_xlabel("Percent difference")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of percent difference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "02_percent_difference_histogram.png", dpi=120)
    plt.close(fig)

    # 3. Mean percent difference by condition (horizontal bar)
    fig, ax = plt.subplots(figsize=(7, max(4, len(cond_summary) * 0.4)))
    conds = [s["condition"] for s in cond_summary]
    means = [s["mean_pct"] for s in cond_summary]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(conds)))
    bars = ax.barh(conds, means, color=colors)
    ax.set_xlabel("Mean percent difference")
    ax.set_title("Mean percent difference by condition")
    ax.axvline(np.mean(pct), color="gray", linestyle="--", alpha=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "03_mean_pct_by_condition.png", dpi=120)
    plt.close(fig)

    # 4. Mean percent difference by biomass tercile
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [b[0] for b in biomass_summary]
    means = [b[2] for b in biomass_summary]
    ax.bar(labels, means, color=["#e74c3c", "#f1c40f", "#27ae60"], edgecolor="black")
    ax.set_ylabel("Mean percent difference")
    ax.set_title("Mean percent difference by biomass level (terciles)")
    fig.tight_layout()
    fig.savefig(out_dir / "04_mean_pct_by_biomass_tercile.png", dpi=120)
    plt.close(fig)

    # 5. Mean percent difference by month
    fig, ax = plt.subplots(figsize=(7, 4))
    months = [m[0] for m in month_summary]
    means = [m[2] for m in month_summary]
    ax.bar(months, means, edgecolor="black", alpha=0.8)
    ax.set_ylabel("Mean percent difference")
    ax.set_title("Mean percent difference by month")
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "05_mean_pct_by_month.png", dpi=120)
    plt.close(fig)

    # 6. Over vs under prediction by condition (grouped bar)
    conds = sorted(over_under.keys())
    over_counts = [over_under[c][0] for c in conds]
    under_counts = [over_under[c][1] for c in conds]
    x = np.arange(len(conds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, max(4, len(conds) * 0.35)))
    ax.barh(x - w / 2, over_counts, w, label="Over (pred > actual)", color="steelblue", alpha=0.9)
    ax.barh(x + w / 2, under_counts, w, label="Under (pred < actual)", color="coral", alpha=0.9)
    ax.set_yticks(x)
    ax.set_yticklabels(conds)
    ax.set_xlabel("Count")
    ax.set_title("Over vs under prediction by condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "06_over_under_by_condition.png", dpi=120)
    plt.close(fig)

    # 7. Worst 20% vs best 20% count by condition (grouped bar)
    all_conds = sorted(set(worst_cond) | set(best_cond))
    worst_counts = [worst_cond.get(c, 0) for c in all_conds]
    best_counts = [best_cond.get(c, 0) for c in all_conds]
    x = np.arange(len(all_conds))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, max(4, len(all_conds) * 0.35)))
    ax.barh(x - w / 2, worst_counts, w, label="Worst 20%", color="darkred", alpha=0.8)
    ax.barh(x + w / 2, best_counts, w, label="Best 20%", color="darkgreen", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(all_conds)
    ax.set_xlabel("Count")
    ax.set_title("Condition in worst 20% vs best 20% (by percent difference)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "07_worst_best_20pct_by_condition.png", dpi=120)
    plt.close(fig)

    print(f"Saved 7 plots to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Detailed pattern analysis on predictions CSV with filenames.")
    parser.add_argument("csv", help="Path to clean CSV (filename, actual, predicted, percent_difference)")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Directory to save plots (default: analysis_plots next to CSV). Omit to skip plots.",
    )
    args = parser.parse_args()

    rows = load_csv(args.csv)
    if not rows:
        print("No rows in CSV.", file=sys.stderr)
        sys.exit(1)

    summary = run_analysis(rows)

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = Path(args.csv).parent / "analysis_plots"
    save_plots(rows, summary, out_dir)


if __name__ == "__main__":
    main()
