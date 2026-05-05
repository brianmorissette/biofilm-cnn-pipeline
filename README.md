# Biofilm CNN Pipeline

An MQP (Major Qualifying Project) research pipeline that investigates whether 2D release-cell microscopy images can predict biofilm properties using convolutional neural networks. The project is exploratory — it is not known a priori whether release cell images contain sufficient signal to predict biofilm removal.

The pipeline supports two independent microscopy datasets, runs Bayesian hyperparameter sweeps with k-fold cross-validation via Weights & Biases, and produces per-image predicted vs actual values for final evaluation.

**Research questions:**
1. Can release cell images predict biofilm removal using deep learning (CNNs)?
2. How do high-resolution data (spinning disk confocal) compare to low-resolution data (Keyence fluorescence)?
3. How do different mathematical transformations of the input images impact predictions, and why?

**Two-phase experiment design:**
1. **Phase 1 — Bayesian hyperparameter optimization:** For each dataset independently, run a 100-run Bayesian sweep over architecture and training hyperparameters with `transform_name=none` (raw images). The sweep minimizes `val/mean_loss` (mean best validation loss across k folds). This identifies the best-performing hyperparameter configuration for each dataset.
2. **Phase 2 — Transformation grid search:** Fix the best hyperparameters from Phase 1, then run a grid search over 4 transforms (none, FFT-DCT, Mexican Hat wavelet, Gabor filter) with 10 repeat runs each (40 total runs per dataset). The repeats provide statistical power to compare transform performance.

---

## Repository layout

```
.
├── src/                      # Training pipeline (the actual code)
│   ├── train.py              #   Entry point invoked by `wandb agent`
│   ├── dataset.py            #   Data loading, k-fold splitting, ImageLabelDataset
│   ├── model.py              #   DynamicCNN architecture
│   ├── biofilm_preprocess.py #   CLAHE + IsoData thresholding (Keyence labels)
│   └── release_preprocess.py #   Patch extraction + FFT/Mexican-hat/Gabor transforms
│
├── scripts/                  # Optional one-off utilities (NOT part of training)
│   ├── organize_keyence_data.py            # Raw GDrive dump -> processed/ layout
│   ├── analyze_keyence_data.py             # Dataset summary (counts, stats)
│   ├── analyze_spinning_disk_data.py       # Dataset summary (counts, stats)
│   └── visualize_spinning_disk_preprocessing.py  # Preprocessing visualizations
│
├── sweep_keyence.yml             # Phase 1: Bayesian HP sweep, Keyence
├── sweep_spinning_disk.yml       # Phase 1: Bayesian HP sweep, Spinning Disk
├── sweep_keyence_trans.yml       # Phase 2: transform grid (4 transforms × 10 repeats)
├── sweep_spinning_disk_trans.yml # Phase 2: transform grid for Spinning Disk
│
├── sbatch_sweep_keyence.sh        # SLURM submit script (Turing HPC)
├── sbatch_sweep_spinning_disk.sh
│
└── pyproject.toml / uv.lock / .python-version  # Python 3.12 + uv
```

---

## Datasets

The image data is **not** stored in this repo. It lives on the WPI Turing HPC cluster:

| Dataset | Path on Turing |
|---|---|
| Keyence (low-res fluorescence) | `/project/ejstewart/biofilm-ml-mqp/keyence/processed` |
| Spinning Disk (high-res confocal) | `/project/ejstewart/biofilm-ml-mqp/spinning_disk/processed` |

Each `processed/` directory must contain the layout described below. For local development, the code defaults to `data/<dataset>/processed/` if `DATA_ROOT` is unset.

### Keyence (low resolution)
- **Source:** Keyence fluorescence microscope
- **Images:** 179 paired image sets. Each pair has a biofilm image and a release cell image, both `.tif` files
- **Image size:** 480×640 pixels (BGR, 3-channel)
- **Labels:** Biofilm surface area in square microns, computed at runtime from the biofilm image via CLAHE preprocessing + iterative (IsoData) thresholding. Pixel size = 1.13×1.13 microns (pixel area = 1.2769 sq microns)
- **CNN input:** The release cell image (green channel extracted, normalized to [0, 1])
- **Layout:** `data/keyence/processed/biofilm/*.tif` and `data/keyence/processed/release/*.tif`, paired by filename stem

### Spinning Disk (high resolution)
- **Source:** Spinning disk confocal microscope
- **Images:** 827 release cell images, `.tif` files loaded via tifffile (not OpenCV, due to TIFF format compatibility)
- **Image size:** 2048×2048 pixels (grayscale)
- **Labels:** Pre-computed biomass values from `biofilm/release_biomass_mapping.csv`
- **CNN input:** The release cell image (grayscale, normalized to [0, max_pixel_value])
- **Layout:** `data/spinning_disk/processed/biofilm/release_biomass_mapping.csv` and `data/spinning_disk/processed/release/2D/*.tif`

If you have raw Keyence data straight from the lab's Google Drive, `scripts/organize_keyence_data.py` reorganizes it into the layout above. You only ever need to run that once.

---

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Python 3.12+ is required.

```bash
git clone <repo-url> biofilm-cnn-pipeline
cd biofilm-cnn-pipeline

uv sync               # creates .venv and installs all deps
uv run wandb login    # one-time W&B authentication
```

---

## Running a sweep

The pipeline is driven entirely by W&B sweeps. Each sweep YAML defines the hyperparameter search and points at `src/train.py` as the program.

### 1. Create the sweep

```bash
uv run wandb sweep sweep_keyence.yml
# returns a sweep ID like: <entity>/<project>/abc12def
```

### 2a. Run locally (smoke-test only — too slow for real sweeps)

```bash
export DATA_ROOT=$(pwd)/data/keyence/processed
uv run wandb agent <entity>/<project>/<sweep_id>
```

### 2b. Run on Turing HPC (SLURM)

The two `sbatch_sweep_*.sh` files set `DATA_ROOT` to the on-cluster data path and start a `wandb agent` on a node with 1 GPU, 8 CPUs, 128 GB RAM, on the `short` partition (24h limit):

```bash
sbatch sbatch_sweep_keyence.sh
sbatch sbatch_sweep_spinning_disk.sh
```

To run multiple agents in parallel against the same sweep, just `sbatch` the script multiple times.

`DATA_ROOT` points directly to the processed data directory for the specific dataset being run. It is NOT a shared root — each dataset has its own path. If `DATA_ROOT` is not set, the code defaults to `data/<dataset>/processed/` (local development).

---

## Pipeline flow (per W&B sweep run)

Each sweep run executes `src/train.py` with a hyperparameter config from W&B:

```
1. LOAD DATA
   Load all raw image pairs with filenames

2. STRATIFIED TEST SPLIT
   Hold out 20% as fixed test set (stratified by label quartiles, n_bins=4, random_state=42)
   Remaining 80% = trainval set

3. K-FOLD CROSS-VALIDATION (on the 80%)
   For each fold:
     a. Split trainval into train/val at image level (KFold, shuffle=True, random_state=42)
     b. Preprocess images
        (keyence: CLAHE+blur on biofilm, green channel on release;
         spinning disk: grayscale normalize)
     c. Compute labels
        (keyence: surface area from biofilm; spinning disk: biomass from CSV)
     d. Normalize labels to [0,1] using TRAIN fold's min/max
     e. Extract patches from release images via evenly-spaced grid
        (locked corners, no pixel loss)
     f. Apply transform if transform_name != "none"
     g. Train DynamicCNN with:
        - Loss: L1Loss (MAE) on normalized [0,1] labels
        - Optimizer: AdamW (weight_decay from config, default 1e-4)
        - Scheduler: ReduceLROnPlateau (mode="min", factor=0.5, patience=10)
        - Early stopping: patience=30 epochs
        - On-the-fly 4x rotation augmentation (0/90/180/270 degrees) on training data only
     h. Record per-epoch train/val losses for this fold
     i. Record best val loss and best epoch for this fold

4. LOG CROSS-FOLD SUMMARY TO W&B
   val/mean_loss   (PRIMARY SWEEP METRIC — Bayesian optimization minimizes this)
   val/std_loss

5. LOG MEAN LOSS CURVES
   Average train/val loss across all folds per epoch -> cv/mean_train_loss, cv/mean_val_loss
   (For visualizing overfitting across the sweep)

6. RETRAIN FINAL MODEL
   Compute avg_best_epoch = round(mean of best epochs across folds)
   Build full training set from ALL trainval data (no validation split)
   Label normalization uses full trainval min/max
   Train a fresh DynamicCNN for exactly avg_best_epoch epochs
   (no early stopping, no scheduler)

7. TEST EVALUATION
   Build test pairs using trainval's label min/max (ensures consistent normalization)
   For each test image: extract patches -> apply transform -> predict ->
                        average patch predictions = image prediction
   Denormalize predictions and actuals back to original units
   Log to W&B: predictions table (with filenames), scatter plot
   Log metrics: R², MAE (in original units)
   Save local CSV: outputs/{dataset}_{run_id}_predictions.csv
   Save model as W&B artifact
```

---

## Source files

### `src/train.py` — Main training script
- **Entry point:** `wandb.init()` then `run(wandb.config)`
- **`run(cfg)`** — Orchestrates the full pipeline: CUDA validation, data root resolution, hyperparameter summary, data loading, k-fold CV, loss curve logging, final retrain, test evaluation, output saving
- **`train_fold()`** — Single fold training loop. Uses AdamW optimizer, L1Loss, ReduceLROnPlateau (factor=0.5, patience=10), and early stopping (patience=30). Validation loss is computed at patch level. Per-epoch console logs show train loss, val loss, and LR. Returns per-epoch losses, best model state, best epoch. Does NOT log to W&B (console only)
- **`train_final()`** — Simple training loop for final retraining. AdamW optimizer, L1Loss, fixed epoch count, no early stopping, no validation, no scheduler. Logs `final/train_loss` and `final/epoch` to W&B per epoch
- **`final_evaluation()`** — Image-level evaluation: for each image, extract patches, optionally apply transform, predict on all patches, average predictions, denormalize to original units. Returns list of per-image result dicts with filename, predicted, and actual values
- **`log_detailed_results()`** — Logs prediction table and scatter plot to W&B
- **Metrics:** `calculate_r2()`, `calculate_mae()` — both handle tensors and scalars
- **`denormalize()`** — Converts normalized [0,1] values back to original units: `normalized * (max - min) + min`
- **`DATA_ROOT` env var:** Points directly to the dataset's processed directory. If not set, defaults to `data/{dataset}/processed/`

### `src/dataset.py` — Data loading and splitting
- **`load_keyence_images(root)`** — Recursively loads `.tif` images from root via `cv2.imread(IMREAD_UNCHANGED)`, returns `(image, filename_stem)` tuples sorted case-insensitively
- **`_load_spinning_disk_2d_pairs(root)`** — Loads CSV from `{root}/biofilm/release_biomass_mapping.csv` and images from `{root}/release/2D/` via `tifffile.imread()`. Skips rows with empty biomass. Returns `(image, biomass, filename)` tuples
- **`_build_pairs_keyence(raw_pairs, patch_size, transform_name, label_min=None, label_max=None)`** — Preprocesses biofilm (CLAHE+blur), computes surface area labels, extracts green channel from release images, normalizes to [0,1], normalizes labels via min-max, extracts patches, applies transform. Returns `(samples, label_min, label_max, pre_patch_pairs)` where `pre_patch_pairs` are `(full_image, normalized_label, filename)` tuples for image-level evaluation
- **`_build_pairs_spinning_disk()`** — Same return signature. Normalizes grayscale to [0,1], uses biomass from CSV as labels
- **`get_kfold_data(root, cfg, n_folds)`** — Main data function. Loads raw pairs, extracts labels for stratification, performs stratified 80/20 test split (4 quartile bins), runs `KFold(n_splits, shuffle=True, random_state=42)` on trainval. Per fold: builds train/val pairs with train fold's label bounds. Returns `(folds, test_raw, trainval_raw)`
- **`_extract_labels_for_stratification()`** — Extracts labels from raw pairs for stratified splitting (spinning disk: biomass directly; keyence: preprocess biofilm + compute surface area)
- **`ImageLabelDataset`** — PyTorch Dataset wrapper. When `rotate=True`, serves 4x samples (0/90/180/270 degree rotations on-the-fly via index mapping: `real_idx = i // 4`, `rot_variant = i % 4`). Ensures single-channel channel-first format `(1, H, W)`. Returns (float32 tensor, float32 label)
- **`_make_loaders()`** — Wraps samples in `ImageLabelDataset` and `DataLoader`s (`num_workers=2`, `pin_memory` if CUDA, `shuffle=True` for train / `False` for val). Train uses `rotate=True`, val uses `rotate=False`
- **`get_dataloaders()`** — Legacy non-CV data loader (train/val/test split without k-fold). Not used in sweep pipeline
- **Data leakage prevention:** Split at image level FIRST, then patch within each split. Label normalization bounds come from train set only

### `src/model.py` — CNN architectures
- **`DynamicCNN`** — The main model. Dynamically configurable conv layers:
  - Each layer: `Conv2d` (same padding, `padding = (kernel_size-1)//2`) → `BatchNorm2d` → `ReLU` → `MaxPool2d(2,2)`
  - Channels double per layer: `start_channels`, 2x, 4x, …
  - MaxPool halves spatial dims: patch_size → patch_size/2 → patch_size/4 → …
  - Flatten size computed automatically via dummy forward pass with zero tensor
  - Regression head: `Linear(flat_size → regressor_hidden_size)` → `ReLU` → `Dropout` → `Linear(1)`
  - Input: `(batch, 1, patch_size, patch_size)`, Output: `(batch, 1)`
- **`FixedCNN`** — 3-layer fixed architecture (32→64→128 channels, 128-unit hidden layer, dropout=0.5) for comparison. Not used in sweeps

### `src/biofilm_preprocess.py` — Biofilm image preprocessing (Keyence only)
- **`preprocess_biofilm()`** — Fixed pipeline: grayscale conversion → CLAHE (clip_limit=2.0, tile=8x8) → normalize to 0–255 → Gaussian blur (5x5)
- **`get_iterative_threshold_value()`** — IsoData iterative thresholding algorithm: initial threshold=127.0, tolerance=0.5, iterates until convergence
- **`get_surface_area()`** — Iterative (IsoData) thresholding → count foreground pixels (above threshold) → multiply by pixel area (1.2769 sq microns)
- **`normalize_labels()`** — Min-max normalization to [0,1]. Accepts optional min/max for consistent normalization across splits. Handles edge case where max=min

### `src/release_preprocess.py` — Patch extraction and transforms
- **`extract_patches_auto()`** — Calculates patch grid with locked corners and no pixel loss. Computes `n_patches` per axis = `ceil(dim / patch_size)`, uses `np.linspace` for even distribution of start positions. Returns list of patches
- **Rotation functions:** `rotate_image_90/180/270()` — OpenCV rotation, used by `ImageLabelDataset` for on-the-fly augmentation
- **`apply_transform(image, transform_name)`** — Router function dispatching to the appropriate transform. Raises `ValueError` for unknown names
- **Transforms:**
  - `fft_dct()` — Discrete Cosine Transform via `scipy.fft.dctn(type=2, norm="ortho")`
  - `mexican_hat(size=21, sigma=3.0)` — Laplacian of Gaussian wavelet: generates 2D LoG kernel, applies via `scipy.ndimage.convolve(mode="reflect")`
  - `gabor(frequency=0.5)` — Gabor filter via `skimage.filters.gabor`, returns magnitude (`sqrt(real² + imag²)`)

---

## Experiment design

### Phase 1: Baseline Hyperparameter Optimization (Bayesian)

For each dataset, a 100-run Bayesian sweep searches over architecture and training hyperparameters with raw images (no transform). The sweep uses Bayesian optimization to minimize `val/mean_loss`.

**Why Bayesian:** Efficiently explores the hyperparameter space by modeling the objective function, converging on good configurations faster than random or grid search within the 100-run budget.

### Phase 2: Transformation Comparison (Grid)

After identifying the best hyperparameters from Phase 1, a grid sweep tests whether mathematical transforms of the input patches improve prediction. The grid fixes all hyperparameters from the best Phase 1 run and varies only `transform_name` across 4 options, with 10 repeat runs each (40 total runs).

**Why 10 repeats:** Neural network training is stochastic (random weight initialization, data shuffling, dropout). Multiple repeats provide the statistical power to determine whether observed differences between transforms are significant vs due to random variation.

**Why grid search:** With only 4 transform options and the goal of comparing all of them equally, grid search ensures every transform gets the same number of runs.

### Keyence Phase 2 — Best hyperparameters from Phase 1
| Parameter | Value |
|---|---|
| patch_size | 128 |
| num_layers | 5 |
| start_channels | 16 |
| kernel_size | 5 |
| regressor_hidden_size | 256 |
| dropout | 0.1 |
| learning_rate | 0.0007163694923747869 |
| weight_decay | 0.0003676566950479888 |
| batch_size | 16 |
| n_folds | 3 |

### Spinning Disk Phase 2 — Best hyperparameters from Phase 1
| Parameter | Value |
|---|---|
| patch_size | 128 |
| num_layers | 5 |
| start_channels | 16 |
| kernel_size | 5 |
| regressor_hidden_size | 256 |
| dropout | 0.5 |
| learning_rate | 0.00004580734842721778 |
| weight_decay | 0.00030246832386003874 |
| batch_size | 64 |
| n_folds | 2 (note: should be 3 to match keyence — known discrepancy) |

---

## Sweep configurations

### Phase 1: Baseline Bayesian sweeps

Both datasets use the same hyperparameter search space with dataset-appropriate patch sizes:

#### `sweep_keyence.yml`
- Method: Bayesian optimization, minimize `val/mean_loss`
- run_cap: 100
- patch_size: [64, 128, 256] (smaller images = smaller patches)
- Fixed: data_source=keyence, transform_name=none, n_folds=3, epochs=300
- Project: `biofilm-cnn-pipeline-sweep-keyence-v3`

#### `sweep_spinning_disk.yml`
- Method: Bayesian optimization, minimize `val/mean_loss`
- run_cap: 100
- patch_size: [128, 256, 512] (larger images = larger patches)
- Fixed: data_source=spinning_disk, transform_name=none, n_folds=3, epochs=300
- Project: `biofilm-cnn-pipeline-sweep-spinning-disk-v6`

#### Shared hyperparameter search space
| Parameter | Type | Values/Range |
|---|---|---|
| num_layers | categorical | [3, 4, 5] |
| start_channels | categorical | [16, 32, 64] |
| kernel_size | categorical | [3, 5] |
| regressor_hidden_size | categorical | [64, 128, 256] |
| dropout | categorical | [0.1, 0.3, 0.5] |
| batch_size | categorical | [16, 32, 64] |
| learning_rate | log_uniform | [1e-5, 1e-3] |
| weight_decay | log_uniform | [1e-4, 1e-2] |

### Phase 2: Transformation grid sweeps

#### `sweep_keyence_trans.yml`
- Method: grid
- Grid parameters: `transform_name × repeat_id`
  - transform_name: `["none", "fft_dct", "mexican_hat", "gabor"]`
  - repeat_id: `[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]`
- Total runs: 4 × 10 = 40
- All other hyperparameters fixed from best Phase 1 run (see table above)
- Project: `biofilm-cnn-pipeline-sweep-keyence-trans-v3`

#### `sweep_spinning_disk_trans.yml`
- Method: grid
- Grid parameters: `transform_name × repeat_id` (same as keyence)
- Total runs: 4 × 10 = 40
- All other hyperparameters fixed from best Phase 1 run (see table above)
- Project: `biofilm-cnn-pipeline-sweep-spinning-disk-trans-v2`

---

## W&B metrics logged per run

### During training (after all folds complete)
- `val/mean_loss` — **Primary sweep metric.** Mean best val loss across folds
- `val/std_loss` — Cross-fold stability check (high std = unreliable mean)
- `cv/mean_train_loss`, `cv/mean_val_loss` — Per-epoch mean across folds (for overfitting analysis)
- `cv/epoch` — Epoch number (1-indexed)

### During final retrain
- `final/train_loss`, `final/epoch` — Per-epoch training loss on full trainval data

### Test evaluation
- `test/r2`, `test/mae`
- `test/predictions_table` — Table with filename, predicted, actual
- `test/pred_vs_actual` — Scatter plot

---

## Outputs

- **W&B dashboard:** All metrics, tables, plots, and artifacts are logged to the W&B project specified in the sweep YAML
- **Local CSV:** `outputs/{data_source}_{wandb_run_id}_predictions.csv` — per-image filename, predicted, actual
- **Model artifact:** `best_model.pth` saved as W&B artifact per run

---

## Key design decisions

- **L1Loss (MAE) as training loss:** L1Loss is more robust to outliers than MSE, which is important when label distributions may be skewed. It also directly optimizes the same quantity reported as the test MAE metric
- **AdamW optimizer:** Decouples weight decay from gradient updates (unlike Adam with L2 regularization), providing more principled regularization. Weight decay is a searchable hyperparameter
- **ReduceLROnPlateau scheduler:** Automatically reduces learning rate by 50% when validation loss plateaus for 10 epochs. This is more adaptive than fixed schedules. Patience (10) is shorter than early stopping patience (30) so the model gets multiple LR reductions before stopping
- **Early stopping (patience=30):** Prevents overfitting while giving the model enough runway to recover from temporary plateaus, especially after learning rate reductions
- **Stratified test split:** Labels are binned into 4 quartiles and stratified so the test set is representative of the full label distribution. This prevents the test set from being unrepresentative by chance (n_bins=4, hardcoded)
- **Retrain after CV:** The final model is retrained from scratch on all train+val data for `avg_best_epoch` epochs (no early stopping). This maximizes training data for the final model and avoids fold-selection bias
- **No per-fold W&B logging:** Only mean loss curves across folds are logged to W&B. Per-fold metrics are printed to console only. This keeps the W&B dashboard clean for sweep-level analysis
- **Image-level evaluation:** Patches are extracted from full images, predicted individually, then averaged per image. This is the primary evaluation granularity — the model never sees the full image at once
- **Two test metrics:** R² (scale-independent model quality for comparison of datasets) and MAE (physically interpretable error in original units)
- **3-fold CV:** Reduces per-run training time by ~40% vs 5-fold while providing sufficient validation set sizes (keyence: ~48 images/fold, spinning disk: ~221 images/fold)
- **On-the-fly rotation augmentation:** Training data is augmented 4x via 90-degree rotations applied in `ImageLabelDataset.__getitem__()`, avoiding RAM overhead of storing rotated copies. Only applied to training data, not validation or test
- **Patch-level validation loss:** During training, validation loss is computed on individual patches (not image-averaged). Image-level aggregation is only used during final test evaluation
- **Data leakage prevention:** Images are split into train/val/test at the image level FIRST, then patches are extracted within each split. Label normalization bounds are computed from the training set only and applied consistently to val/test

---

## Optional analysis scripts

The `scripts/` directory contains optional utilities. **None of them are needed to train models** — they're for dataset bootstrapping, sanity-checking, and visualization. Each script has a detailed module docstring at the top explaining when to use it; quick summaries:

- **`organize_keyence_data.py`** — One-time setup: takes the raw Keyence Google Drive dump and reorganizes it into `data/keyence/processed/{biofilm,release}/`. Run only once when bootstrapping the dataset.
- **`analyze_keyence_data.py`** — Dataset summary for Keyence: counts and surface-area stats overall, by treatment, and by month. Run as a sanity check before sweeps.
- **`analyze_spinning_disk_data.py`** — Dataset summary for Spinning Disk: counts and biomass stats overall, by treatment, and by month. Run as a sanity check before sweeps.
- **`visualize_spinning_disk_preprocessing.py`** — Pedagogical visualization: renders raw → grayscale → normalized → patch → transform stages for a few example images. Useful for onboarding and for sanity-checking transforms.

```bash
uv run python scripts/analyze_keyence_data.py
uv run python scripts/analyze_spinning_disk_data.py
uv run python scripts/visualize_spinning_disk_preprocessing.py
```
