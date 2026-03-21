# Biofilm CNN Pipeline

## Project Overview

This project predicts biofilm properties from microscopy images using a CNN regression pipeline. It supports two independent microscopy datasets, runs Bayesian hyperparameter sweeps with 3-fold cross-validation via Weights & Biases, and produces per-image predicted vs actual values for final evaluation.

**Goal:** Given a 2D "release cell" microscopy image, predict a continuous biofilm measurement (surface area for keyence, biomass for spinning disk).

## Datasets

### Keyence
- **Source:** Keyence fluorescence microscope
- **Images:** 179 paired image sets. Each pair has a biofilm image and a release cell image, both .tif files
- **Image size:** 480x640 pixels (BGR, 3-channel)
- **Labels:** Biofilm surface area in square microns, computed at runtime from the biofilm image via CLAHE preprocessing + iterative (IsoData) thresholding. Pixel size = 1.13x1.13 microns
- **CNN input:** The release cell image (green channel extracted, normalized to [0,1])
- **Local path:** `data/keyence/processed/` with `biofilm/` and `release/` subdirectories

### Spinning Disk
- **Source:** Spinning disk confocal microscope
- **Images:** 827 release cell images, .tif files loaded via tifffile (not OpenCV, due to compatibility)
- **Image size:** 2048x2048 pixels (grayscale)
- **Labels:** Pre-computed biomass values from `biofilm/release_biomass_mapping.csv`
- **CNN input:** The release cell image (grayscale, normalized to [0, max_pixel_value])
- **Local path:** `data/spinning_disk/processed/` with `biofilm/` (CSV) and `release/2D/` subdirectories

## Pipeline Flow (per W&B sweep run)

Each sweep run executes `src/train.py` with a hyperparameter config from W&B:

```
1. LOAD DATA
   Load all raw image pairs with filenames

2. STRATIFIED TEST SPLIT
   Hold out 20% as fixed test set (stratified by label quartiles, n_bins=4, random_state=42)
   Remaining 80% = trainval set

3. 3-FOLD CROSS-VALIDATION (on the 80%)
   For each fold:
     a. Split trainval into train/val at image level
     b. Preprocess images (keyence: CLAHE+blur on biofilm, green channel on release; SD: grayscale normalize)
     c. Compute labels (keyence: surface area from biofilm; SD: biomass from CSV)
     d. Normalize labels to [0,1] using TRAIN fold's min/max
     e. Extract patches from release images
     f. Train DynamicCNN with early stopping (patience=30) and ReduceLROnPlateau
     g. Record per-epoch train/val losses for this fold
     h. Record best val loss and best epoch for this fold

4. LOG CROSS-FOLD SUMMARY TO W&B
   val/mean_loss (THIS IS THE SWEEP METRIC - Bayesian optimization minimizes this)
   val/std_loss

5. LOG MEAN LOSS CURVES
   Average train/val loss across all folds per epoch -> cv/mean_train_loss, cv/mean_val_loss
   (For visualizing overfitting across the sweep)

6. RETRAIN FINAL MODEL
   Compute avg_best_epoch = round(mean of best epochs across 3 folds)
   Build full training set from ALL trainval data (no validation split)
   Train a fresh DynamicCNN for exactly avg_best_epoch epochs (no early stopping, no scheduler)
   Label normalization uses full trainval min/max

7. TEST EVALUATION
   Build test pairs using trainval's label min/max
   For each test image: extract patches -> predict -> average patch predictions = image prediction
   Log to W&B: predictions table (with filenames), scatter plot
   Log metrics: R², MAE
   Save local CSV: outputs/{dataset}_{run_id}_predictions.csv
   Save model as W&B artifact
```

## Source Files

### `src/train.py` — Main training script
- **Entry point:** `wandb.init()` then `run(wandb.config)`
- **`run(cfg)`** — Orchestrates the full pipeline: CUDA validation, hyperparameter summary, data loading, k-fold CV, loss curve logging, final retrain, test evaluation
- **`train_fold()`** — Single fold training loop with early stopping. Per-epoch console logs show train loss, val loss, and LR. Returns per-epoch losses, best model state, best epoch. Does NOT log to W&B (console only)
- **`train_final()`** — Simple training loop for final retraining. Fixed epoch count, no early stopping, no validation. Logs `final/train_loss` to W&B
- **`evaluate_full_images()`** — Image-level evaluation: extract patches, predict, average per image
- **`final_evaluation()`** — Detailed per-image results with filenames for test set logging
- **`log_detailed_results()`** — Logs prediction table and scatter plot to W&B
- **Metrics:** `calculate_r2()`, `calculate_mae()` — all handle both tensors and scalars
- **DATA_ROOT env var:** Points directly to the dataset's processed directory. If not set, defaults to `data/{dataset}/processed/`

### `src/dataset.py` — Data loading and splitting
- **`load_keyence_images(root)`** — Loads .tif images, returns `(image, filename_stem)` tuples
- **`_load_spinning_disk_2d_pairs(root)`** — Loads from CSV + tifffile, returns `(image, biomass, filename)` tuples
- **`_build_pairs_keyence()`** — Preprocesses biofilm (CLAHE+blur), computes surface area labels, extracts patches. Returns `(samples, label_min, label_max, pre_patch_pairs)` where `pre_patch_pairs` are `(full_image, normalized_label, filename)` tuples
- **`_build_pairs_spinning_disk()`** — Same return signature. Normalizes grayscale, uses biomass from CSV
- **`get_kfold_data(root, cfg, n_folds)`** — Main data function. Stratified test split, k-fold on trainval. Returns `(folds, test_raw, trainval_raw)`
- **`ImageLabelDataset`** — PyTorch Dataset wrapper. When `rotate=True`, serves 4x samples (0/90/180/270 degree rotations on-the-fly)
- **`get_dataloaders()`** — Legacy non-CV data loader (train/val/test split without k-fold). Not used in sweep pipeline but kept for compatibility
- **Data leakage prevention:** Split at image level FIRST, then patch within each split. Label normalization bounds come from train set only

### `src/model.py` — CNN architectures
- **`DynamicCNN`** — The main model. Dynamically configurable conv layers:
  - Each layer: Conv2d (same padding) -> BatchNorm -> ReLU -> MaxPool2d(2,2)
  - Channels double per layer: start_channels, 2x, 4x, ...
  - MaxPool halves spatial dims: patch_size -> patch_size/2 -> patch_size/4 -> ...
  - Flatten size computed automatically via dummy forward pass
  - Regression head: Linear -> ReLU -> Dropout -> Linear(1)
  - Input: (batch, 1, patch_size, patch_size), Output: (batch, 1)
- **`FixedCNN`** — 3-layer fixed architecture for comparison (not used in sweeps)

### `src/biofilm_preprocess.py` — Biofilm image preprocessing (keyence only)
- **`preprocess_biofilm()`** — Fixed pipeline: grayscale conversion -> CLAHE (clip_limit=2.0, tile=8x8) -> normalize to 0-255 -> Gaussian blur (5x5). Only `clahe` and `gaussian` are accepted (ValueError otherwise)
- **`get_surface_area()`** — Iterative (IsoData) thresholding -> count foreground pixels -> multiply by pixel area (1.2769 sq microns). Only `iterative`/`isodata` threshold method accepted
- **`normalize_labels()`** — Min-max normalization to [0,1]. Accepts optional min/max for consistent normalization across splits

### `src/release_preprocess.py` — Patch extraction and transforms
- **`extract_patches_auto()`** — Calculates patch grid with locked corners and no pixel loss. Uses `np.linspace` for even distribution
- **Rotation functions:** `rotate_image_90/180/270()` — Used by `ImageLabelDataset` for on-the-fly augmentation
- **Transforms (not used in baseline sweeps):**
  - `fft_dct()` — Discrete Cosine Transform
  - `mexican_hat()` — Laplacian of Gaussian wavelet (21x21 kernel, sigma=3)
  - `gabor()` — Gabor filter (frequency=0.5), returns magnitude

## Sweep Configurations

Both datasets use the same structure with tuned ranges:

### `sweep_keyence.yml`
- Method: Bayesian optimization, minimize `val/mean_loss`
- run_cap: 100
- patch_size: [64, 128, 256] (smaller images)
- Fixed: data_source=keyence, transform_name=none, n_folds=3, epochs=300

### `sweep_spinning_disk.yml`
- Method: Bayesian optimization, minimize `val/mean_loss`
- run_cap: 100
- patch_size: [128, 256, 512] (larger images)
- Fixed: data_source=spinning_disk, transform_name=none, n_folds=3, epochs=300

### Shared hyperparameter search space
| Parameter | Values/Range |
|---|---|
| num_layers | [3, 4, 5] |
| start_channels | [16, 32, 64] |
| kernel_size | [3, 5] |
| regressor_hidden_size | [64, 128, 256] |
| dropout | [0.1, 0.3, 0.5] |
| batch_size | [16, 32, 64] |
| learning_rate | log_uniform [1e-5, 1e-3] |
| weight_decay | log_uniform [1e-4, 1e-2] |

## W&B Metrics Logged Per Run

### During training (after all folds complete)
- `val/mean_loss` — **Primary sweep metric.** Mean best val loss across 3 folds
- `val/std_loss` — Cross-fold stability check (high std = unreliable mean)
- `cv/mean_train_loss`, `cv/mean_val_loss` — Per-epoch mean across folds (for overfitting analysis)

### During final retrain
- `final/train_loss`, `final/epoch` — Per-epoch training loss on full trainval data

### Test evaluation
- `test/r2`, `test/mae`
- `test/predictions_table` — Table with filename, predicted, actual
- `test/pred_vs_actual` — Scatter plot

## Running Sweeps

### Local (for testing)
```bash
# Create the sweep (returns a sweep ID)
wandb sweep sweep_keyence.yml

# Run an agent
uv run -- wandb agent <entity>/<project>/<sweep_id>
```

### On Turing HPC Cluster (SLURM)

Each dataset needs its own sbatch script. The key difference is `DATA_ROOT` must point to the correct dataset's processed directory on the cluster.

```bash
# In the sbatch script:
export DATA_ROOT=/path/to/dataset/processed   # must contain biofilm/ and release/ subdirs
uv run -- wandb agent <entity>/<project>/<sweep_id>
```

The sbatch scripts request: 1 GPU, 8 CPU cores, 128GB memory, short partition (24h).

**Important:** `DATA_ROOT` points directly to the processed data directory for the specific dataset being run. It is NOT a shared root — each dataset needs its own path. If `DATA_ROOT` is not set, the code defaults to `data/<dataset>/processed/` (local development).

## Outputs

- **W&B dashboard:** All metrics, tables, plots, and artifacts are logged to the W&B project specified in the sweep YAML
- **Local CSV:** `outputs/{data_source}_{wandb_run_id}_predictions.csv` — filename, predicted, actual, abs_error, pct_error
- **Model artifact:** `best_model.pth` saved as W&B artifact per run

## Key Design Decisions

- **Stratified test split:** Labels are binned into 4 quartiles and stratified so the test set is representative of the full label distribution (n_bins=4, hardcoded)
- **Retrain after CV:** The final model is retrained from scratch on all train+val data for avg_best_epoch epochs (no early stopping). This maximizes training data and avoids fold-selection bias
- **No per-fold W&B logging:** Only mean loss curves across folds are logged to W&B. Per-fold metrics are printed to console only. This keeps the W&B dashboard clean
- **Image-level evaluation:** Patches are extracted from full images, predicted individually, then averaged per image. This is the primary evaluation granularity
- **Two test metrics:** R² (scale-independent model quality) and MAE (physically interpretable error in original units). NRMSE and MAPE were dropped — NRMSE is redundant with R², and MAPE is unreliable when labels approach zero
- **3-fold CV:** Reduces per-run training time by 40% vs 5-fold. Provides sufficient validation set sizes for both datasets (keyence: ~48 images/fold, spinning disk: ~221 images/fold)
- **Transforms disabled for baseline:** `transform_name=none` in both sweep configs. Transforms (FFT-DCT, Mexican Hat, Gabor) exist in code for future A/B testing
- **On-the-fly rotation augmentation:** Training data is augmented 4x via 90-degree rotations applied in `ImageLabelDataset.__getitem__()`, avoiding RAM overhead
