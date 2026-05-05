"""
Pedagogical visualization script: see exactly what the spinning-disk
release-cell images look like at each stage of the training pipeline.

WHEN TO USE THIS SCRIPT
-----------------------
Run this when:
- you are new to the project and want to understand visually what the model
  actually sees as input (very recommended for onboarding),
- you are debugging a transform (fft_dct, mexican_hat, gabor) and want to see
  what its output looks like on real spinning-disk data,
- you want to sanity-check that patch extraction is producing reasonable
  patches before investing GPU time in a sweep.

This script is NOT part of the training pipeline — it never trains a model.
Skip it if you just want to run sweeps.

WHAT IT DOES
------------
For a handful of example release-cell images, it renders each preprocessing
stage that `src/dataset.py` and `src/release_preprocess.py` apply during
training:
- the raw TIFF as loaded from disk,
- the grayscale version,
- the per-image max-normalized full image,
- a single patch extracted with the same grid logic the training code uses,
- (optional) the patch after applying one of the FFT/Mexican-hat/Gabor
  transforms.

Each stage is both saved as a PNG (under `--out-dir`) and shown in an
interactive matplotlib window.

Usage (from repo root):
  uv run python scripts/visualize_spinning_disk_preprocessing.py

You can customize options, for example:
  uv run python scripts/visualize_spinning_disk_preprocessing.py \\
    --root data/spinning_disk/processed \\
    --num-examples 3 \\
    --patch-size 256 \\
    --overlap 0.10 \\
    --transform none \\
    --out-dir outputs/spinning_disk_visualization
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import cv2


# ---------------------------------------------------------------------------
# Import project modules (dataset, preprocessing) by adding src/ to sys.path
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from release_preprocess import (  # type: ignore  # noqa: E402
    extract_patches_auto,
    apply_transform,
)
from biofilm_preprocess import normalize_labels  # type: ignore  # noqa: E402

import csv as _csv  # noqa: E402
import tifffile as _tifffile  # noqa: E402


def _load_spinning_disk_filenames(root):
    """
    Load spinning-disk (image, biomass) pairs along with their filenames in the
    same order as `src/dataset._load_spinning_disk_2d_pairs`.

    Returns (pairs, filenames) where pairs is a list of (numpy_image, biomass)
    tuples and filenames is the parallel list of release-image filenames.
    Inlined here so this script has no dependency on other files in `scripts/`.
    """
    csv_path = Path(root) / "biofilm" / "release_biomass_mapping.csv"
    image_dir = Path(root) / "release" / "2D"
    pairs = []
    filenames = []
    with open(csv_path, newline="") as f:
        reader = _csv.DictReader(f)
        for row in reader:
            filename = row["filename"]
            biomass_str = row["biomass"].strip()
            if not biomass_str:
                continue
            biomass = float(biomass_str)
            img_path = image_dir / filename
            try:
                img = _tifffile.imread(str(img_path))
            except Exception:
                continue
            pairs.append((img, biomass))
            filenames.append(filename)
    return pairs, filenames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize preprocessing stages for spinning-disk release cell images, "
            "including raw, grayscale, normalized full image, and 256x256 patches."
        )
    )
    parser.add_argument(
        "--root",
        default=None,
        help=(
            "Processed data root (default: DATA_ROOT or data/spinning_disk/processed). "
            "Should contain 'biofilm/release_biomass_mapping.csv' and 'release/2D/'."
        ),
    )
    parser.add_argument(
        "--num-examples",
        type=int,
        default=3,
        help="Number of example images to visualize (default: 3).",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in pixels (height and width). Default: 256.",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.10,
        help="Target overlap percentage for patch extraction (e.g., 0.10 for 10%%).",
    )
    parser.add_argument(
        "--sample",
        action="append",
        default=None,
        help=(
            "Optional: substring/pattern to select specific filenames. "
            "Can be given multiple times to visualize particular images."
        ),
    )
    parser.add_argument(
        "--transform",
        choices=("none", "fft_dct", "mexican_hat", "gabor"),
        default="none",
        help="Optional transform to apply to the 256x256 patch (default: none).",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/spinning_disk_visualization",
        help="Output directory for PNGs (default: outputs/spinning_disk_visualization).",
    )
    return parser.parse_args()


def ensure_output_dir(base_dir: Path, example_idx: int) -> Path:
    """
    Create and return a directory for a specific example:
      base_dir/example_{idx}
    """
    example_dir = base_dir / f"example_{example_idx}"
    example_dir.mkdir(parents=True, exist_ok=True)
    return example_dir


def _normalize_token(s: str) -> str:
    """
    Normalize a string by lowercasing and removing all non-alphanumeric characters.
    """
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _pattern_tokens(s: str) -> list[str]:
    """
    Break a pattern into normalized alphanumeric tokens suitable for fuzzy matching.

    Example:
        \"DNaseI 09Dec2025-02_sample07\" -> [\"dnasei\", \"09dec2025\", \"02\", \"sample07\"]
    """
    pieces: list[str] = []
    current = []
    for ch in s.lower():
        if ch.isalnum():
            current.append(ch)
        else:
            if current:
                pieces.append("".join(current))
                current = []
    if current:
        pieces.append("".join(current))
    return pieces


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert image to grayscale using the same logic as _build_pairs_spinning_disk."""
    if image.ndim == 3:
        # Consistent with dataset._build_pairs_spinning_disk: use cv2.cvtColor
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    return gray


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Per-image max normalization to [0, 1] (if max > 0), matching _build_pairs_spinning_disk.
    """
    image = image.astype(np.float32)
    max_val = np.max(image)
    if max_val > 0:
        return image / max_val
    return image


def choose_patch(patches: list[np.ndarray]) -> tuple[int, np.ndarray]:
    """
    Choose a canonical patch for visualization.
    Currently returns the first patch in the list.
    """
    if not patches:
        raise ValueError("No patches were extracted from the image.")
    return 0, patches[0]


def save_and_show_image(
    image: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str = "gray",
) -> None:
    """
    Save an image as a PNG and display it with matplotlib.
    """
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(image, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.show()
    plt.close(fig)


def visualize_example(
    example_idx: int,
    image: np.ndarray,
    biomass: float,
    normalized_label: float | None,
    patch_size: int,
    overlap: float,
    transform_name: str,
    base_out_dir: Path,
    filename: str | None = None,
) -> None:
    """
    Run a single spinning-disk image through the preprocessing pipeline and
    save/visualize each stage.
    """
    example_dir = ensure_output_dir(base_out_dir, example_idx)

    label_info = f"biomass={biomass:.4f}"
    if filename is not None:
        label_info += f", file={filename}"
    if normalized_label is not None:
        label_info += f", norm_label={normalized_label:.4f}"

    # Stage 1: Raw image
    raw_title = f"Example {example_idx} - raw ({label_info})"
    # Display raw image in green to distinguish from later grayscale stages.
    save_and_show_image(image, raw_title, example_dir / "raw.png", cmap="Greens")

    # Stage 2: Grayscale
    gray = to_grayscale(image)
    gray_title = f"Example {example_idx} - grayscale ({label_info})"
    save_and_show_image(gray, gray_title, example_dir / "grayscale.png")

    # Stage 3: Normalized full image
    normalized_full = normalize_image(gray)
    norm_title = f"Example {example_idx} - normalized full image ({label_info})"
    save_and_show_image(normalized_full, norm_title, example_dir / "normalized.png")

    # Stage 4: Extract 256x256 patches
    patches = extract_patches_auto(
        normalized_full,
        patch_size=patch_size,
        target_overlap_pct=overlap,
    )
    patch_idx, patch = choose_patch(patches)
    patch_title = (
        f"Example {example_idx} - patch {patch_idx} "
        f"({label_info}, size={patch.shape[0]}x{patch.shape[1]})"
    )
    save_and_show_image(patch, patch_title, example_dir / f"patch_{patch_idx}.png")

    # Stage 5: Apply all key transforms (DCT, Mexican Hat, Gabor) to the patch
    for t_name in ("fft_dct", "mexican_hat", "gabor"):
        transformed = apply_transform(patch, t_name)
        transform_title = (
            f"Example {example_idx} - patch {patch_idx} "
            f"{t_name} ({label_info})"
        )
        transform_filename = example_dir / f"patch_{patch_idx}_{t_name}.png"
        save_and_show_image(transformed, transform_title, transform_filename)

    # Print a small summary to stdout
    print(
        f"[example {example_idx}] "
        f"filename={filename or 'unknown'} "
        f"biomass={biomass:.4f} "
        f"{'(norm=' + str(round(normalized_label, 4)) + ')' if normalized_label is not None else ''} "
        f"| patches_extracted={len(patches)} | chosen_patch_index={patch_idx}"
    )


def main() -> None:
    args = parse_args()

    root_dir = args.root or os.environ.get("DATA_ROOT")
    if root_dir is None:
        root_dir = "data/spinning_disk/processed"

    root_path = Path(root_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using data root: {root_path}")
    print(f"Saving visualizations under: {out_dir}")
    print(
        f"Settings: num_examples={args.num_examples}, patch_size={args.patch_size}, "
        f"overlap={args.overlap}, transform={args.transform}"
    )

    # Load raw (image, biomass) pairs and corresponding filenames.
    raw_pairs, filenames = _load_spinning_disk_filenames(root_path)
    if not raw_pairs:
        print(
            "No spinning-disk pairs loaded. "
            "Check that the root directory is correct and contains the expected files."
        )
        return

    # Determine which samples to visualize.
    selected_indices: list[int] = []
    if args.sample:
        normalized_filenames = [_normalize_token(fn) for fn in filenames]
        for pattern in args.sample:
            tokens = _pattern_tokens(pattern)
            if not tokens:
                continue
            # Match filenames that contain ALL tokens (order-independent).
            # If pattern includes "dnasei", match only "DNaseI" (exclude "DNaseI-02").
            matches = []
            for i, nf in enumerate(normalized_filenames):
                if "dnasei" in tokens and "dnasei02" in nf:
                    continue  # exclude DNaseI-02; user asked for DNaseI only
                if all(tok in nf for tok in tokens):
                    matches.append(i)
            if not matches:
                print(f"Warning: no filename matched sample pattern '{pattern}'.")
                continue
            selected_indices.append(matches[0])

        if not selected_indices:
            print(
                "No samples matched any provided patterns; "
                "falling back to the first num-examples images."
            )

    if not selected_indices:
        num_examples = min(args.num_examples, len(raw_pairs))
        selected_indices = list(range(num_examples))

    selected_pairs = [raw_pairs[i] for i in selected_indices]
    selected_filenames = [filenames[i] for i in selected_indices]
    images = [img for img, _ in selected_pairs]
    biomasses = [b for _, b in selected_pairs]

    # Optionally compute normalized labels for context in titles.
    normalized_labels: list[float] | None = None
    try:
        normalized_labels, label_min, label_max = normalize_labels(biomasses)
        print(
            f"Label normalization: min={label_min:.4f}, max={label_max:.4f} "
            "(used only for display, not for image processing)."
        )
    except Exception as e:
        print(f"Warning: could not normalize labels ({e}). Continuing without normalized labels.")

    for idx, (image, biomass) in enumerate(selected_pairs):
        norm_label = None
        if normalized_labels is not None:
            norm_label = normalized_labels[idx]
        visualize_example(
            example_idx=idx,
            image=image,
            biomass=biomass,
            normalized_label=norm_label,
            patch_size=args.patch_size,
            overlap=args.overlap,
            transform_name=args.transform,
            base_out_dir=out_dir,
            filename=selected_filenames[idx],
        )


if __name__ == "__main__":
    main()

