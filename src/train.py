import copy
import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
import matplotlib.pyplot as plt

from model import DynamicCNN
from dataset import get_kfold_data, _build_pairs_spinning_disk, _build_pairs_keyence, ImageLabelDataset
from release_preprocess import extract_patches_auto, apply_transform


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def denormalize(normalized_label, min_val, max_val):
    """Converts normalized (0-1) values back to original units."""
    return normalized_label * (max_val - min_val) + min_val


def calculate_mape(pred, target, epsilon=1e-7):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles both tensors and scalar values.
    """
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target)
    return torch.mean(torch.abs((pred - target) / (target + epsilon))) * 100


def calculate_r2(pred, target):
    """Coefficient of determination (R²). Scale-independent."""
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def calculate_nrmse(pred, target):
    """RMSE normalized by label range. Scale-independent."""
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    rmse = torch.sqrt(torch.mean((pred - target) ** 2))
    label_range = target.max() - target.min()
    return (rmse / label_range).item() if label_range > 0 else float("inf")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_full_images(model, full_pairs, cfg, device, label_min, label_max):
    """
    Evaluates the model on full images by extracting patches, predicting,
    and averaging patch predictions per image.

    Returns:
        mean_mape: Average image-level MAPE across all images.
        mean_loss: Average image-level L1 loss (in normalized units).
    """
    model.eval()
    image_mapes = []
    image_losses = []

    for full_image, full_label, *_ in full_pairs:
        # 1. Extract patches
        patches = extract_patches_auto(
            full_image,
            patch_size=cfg["patch_size"],
            target_overlap_pct=cfg["target_overlap_pct"],
        )

        # 2. Apply transform
        transform_name = cfg.get("transform_name", "none")
        if transform_name != "none":
            patches = [apply_transform(p, transform_name) for p in patches]

        # 3. Batch and predict
        patches_np = np.array(patches)
        patches_tensor = torch.from_numpy(patches_np).float().unsqueeze(1).to(device)

        with torch.no_grad():
            patch_preds = model(patches_tensor)

        # 4. Aggregate
        avg_pred_norm = patch_preds.mean().item()

        # 5. L1 loss in normalized space
        image_losses.append(abs(avg_pred_norm - full_label))

        # 6. MAPE in original units
        pred_orig = denormalize(avg_pred_norm, label_min, label_max)
        target_orig = denormalize(full_label, label_min, label_max)
        mape = calculate_mape(pred_orig, target_orig)
        image_mapes.append(mape.item())

    return float(np.mean(image_mapes)), float(np.mean(image_losses))


def final_evaluation(model, full_pairs, cfg, device, label_min, label_max):
    """Collect detailed per-image predictions for W&B logging."""
    model.eval()
    results = []

    for idx, (full_image, full_label, filename) in enumerate(full_pairs):
        patches = extract_patches_auto(
            full_image,
            patch_size=cfg["patch_size"],
            target_overlap_pct=cfg["target_overlap_pct"],
        )

        transform_name = cfg.get("transform_name", "none")
        if transform_name != "none":
            patches = [apply_transform(p, transform_name) for p in patches]

        patches_np = np.array(patches)
        patches_tensor = torch.from_numpy(patches_np).float().unsqueeze(1).to(device)

        with torch.no_grad():
            patch_preds = model(patches_tensor)

        avg_pred_norm = patch_preds.mean().item()
        patch_std = patch_preds.std().item()

        pred_orig = denormalize(avg_pred_norm, label_min, label_max)
        target_orig = denormalize(full_label, label_min, label_max)

        abs_error = abs(pred_orig - target_orig)
        pct_error = calculate_mape(pred_orig, target_orig).item()

        results.append({
            "image_idx": idx,
            "filename": filename,
            "predicted": pred_orig,
            "actual": target_orig,
            "abs_error": abs_error,
            "pct_error": pct_error,
            "patch_std": patch_std,
        })

    return results


def log_detailed_results(results, full_pairs, prefix="val"):
    """
    Log detailed prediction results to W&B including:
    - Predictions table
    - Scatter plot of predicted vs actual
    - Error histogram
    - Best and worst predicted images
    - Summary statistics
    """
    # 1. Predictions table
    columns = ["image_idx", "filename", "predicted", "actual", "abs_error", "pct_error", "patch_std"]
    table_data = [[r[col] for col in columns] for r in results]
    table = wandb.Table(columns=columns, data=table_data)
    wandb.log({f"{prefix}/predictions_table": table})

    # 2. Scatter plot
    scatter_table = wandb.Table(
        columns=["actual", "predicted"],
        data=[[r["actual"], r["predicted"]] for r in results],
    )
    wandb.log({
        f"{prefix}/pred_vs_actual": wandb.plot.scatter(
            scatter_table, "actual", "predicted",
            title=f"{prefix.capitalize()} Predicted vs Actual",
        )
    })

    # 3. Error histogram
    errors = [r["pct_error"] for r in results]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(errors, bins=20, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Percentage Error (%)")
    ax.set_ylabel("Count")
    ax.set_title(f"{prefix.capitalize()} Error Distribution")
    ax.axvline(np.mean(errors), color="red", linestyle="--",
               label=f"Mean: {np.mean(errors):.1f}%")
    ax.legend()
    plt.tight_layout()
    wandb.log({f"{prefix}/error_histogram": wandb.Image(fig)})
    plt.close(fig)

    # 4. Best / worst images
    sorted_results = sorted(results, key=lambda x: x["pct_error"])
    for tag, subset in [("best_predictions", sorted_results[:5]),
                        ("worst_predictions", sorted_results[-5:])]:
        images = []
        for r in subset:
            img = full_pairs[r["image_idx"]][0]
            caption = (f"Pred: {r['predicted']:.1f}, "
                       f"Actual: {r['actual']:.1f}, "
                       f"Error: {r['pct_error']:.1f}%")
            images.append(wandb.Image(img, caption=caption))
        wandb.log({f"{prefix}/{tag}": images})

    # 5. Summary statistics
    wandb.log({
        f"{prefix}/final_mean_error": np.mean(errors),
        f"{prefix}/final_median_error": np.median(errors),
        f"{prefix}/final_max_error": np.max(errors),
        f"{prefix}/final_min_error": np.min(errors),
    })


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    model, train_loader, val_loader, val_full_pairs,
    cfg, device, label_min, label_max,
    fold_idx, n_folds,
):
    """
    Train a single fold and return metrics + best model state dict.

    Includes:
      - L1 loss on normalized [0,1] labels
      - ReduceLROnPlateau scheduler
      - Patience-based early stopping
      - Correct patch-level MAPE (collected across full val set then computed)
      - Image-level MAPE via evaluate_full_images
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10,
    )
    criterion = nn.L1Loss()

    patience = 30
    epochs_no_improve = 0
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    train_losses = []
    val_losses_per_epoch = []

    prefix = f"fold_{fold_idx + 1}" if n_folds > 1 else "train"

    for epoch in range(cfg["epochs"]):
        # --- TRAINING ---
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_train_loss = running_loss / len(train_loader.dataset)

        # --- VALIDATION (PATCH LEVEL) ---
        model.eval()
        val_running_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)

                preds = model(inputs)
                loss = criterion(preds, targets)
                val_running_loss += loss.item() * inputs.size(0)

                all_val_preds.append(preds.cpu())
                all_val_targets.append(targets.cpu())

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses_per_epoch.append(epoch_val_loss)

        # Correct MAPE: compute once over all predictions
        all_val_preds = torch.cat(all_val_preds)
        all_val_targets = torch.cat(all_val_targets)
        pred_orig = denormalize(all_val_preds, label_min, label_max)
        target_orig = denormalize(all_val_targets, label_min, label_max)
        epoch_patch_mape = calculate_mape(pred_orig, target_orig).item()

        # --- VALIDATION (IMAGE LEVEL) ---
        epoch_image_mape, _ = evaluate_full_images(
            model, val_full_pairs, cfg, device, label_min, label_max,
        )

        # --- LR SCHEDULER ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_loss)

        # --- LOGGING ---
        log_dict = {
            "epoch": epoch + 1,
            f"{prefix}/train_loss": epoch_train_loss,
            f"{prefix}/val_loss": epoch_val_loss,
            f"{prefix}/val_patch_mape": epoch_patch_mape,
            f"{prefix}/val_image_mape": epoch_image_mape,
            f"{prefix}/learning_rate": current_lr,
        }
        wandb.log(log_dict)

        print(f"  [{prefix}] Epoch {epoch + 1}/{cfg['epochs']} | "
              f"Train Loss: {epoch_train_loss:.5f} | "
              f"Val Loss: {epoch_val_loss:.5f} | "
              f"Patch MAPE: {epoch_patch_mape:.2f}% | "
              f"Image MAPE: {epoch_image_mape:.2f}% | "
              f"LR: {current_lr:.2e}")

        # --- EARLY STOPPING ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"  [{prefix}] Early stopping at epoch {epoch + 1} "
                      f"(best was epoch {best_epoch}).")
                break

    # Restore best model weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute final metrics with best model
    final_image_mape, final_image_loss = evaluate_full_images(
        model, val_full_pairs, cfg, device, label_min, label_max,
    )

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "final_image_mape": final_image_mape,
        "final_image_loss": final_image_loss,
        "model_state": best_state,
        "label_min": label_min,
        "label_max": label_max,
        "train_losses": train_losses,
        "val_losses": val_losses_per_epoch,
    }


# ---------------------------------------------------------------------------
# Final retraining (no validation, fixed epochs)
# ---------------------------------------------------------------------------

def train_final(model, train_loader, cfg, device, n_epochs):
    """Train on all train+val data for a fixed number of epochs.

    No early stopping, no validation, no scheduler — just train.
    Used after CV to retrain the best hyperparams on maximum data.
    """
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg.get("weight_decay", 1e-4),
    )
    criterion = nn.L1Loss()

    for epoch in range(n_epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1)
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        wandb.log({"final/train_loss": epoch_loss, "final/epoch": epoch + 1})

        if (epoch + 1) % 25 == 0 or epoch == 0:
            print(f"  [Final retrain] Epoch {epoch + 1}/{n_epochs} | "
                  f"Train Loss: {epoch_loss:.5f}")

    return model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. Determine root directory (DATA_ROOT env var overrides defaults)
    data_source = cfg["data_source"]
    root_dir = os.environ.get("DATA_ROOT")
    if root_dir is None:
        if data_source in ("keyence", "keyence_clearest"):
            root_dir = "data/keyence/processed"
        elif data_source == "spinning_disk":
            root_dir = "data/spinning_disk/processed"
        else:
            raise ValueError(f"Unknown data_source: {data_source}")

    n_folds = cfg.get("n_folds", 5)

    # Log dataset info
    wandb.config.update({"root_dir": root_dir}, allow_val_change=True)

    # 2. Load k-fold data
    folds, test_raw, trainval_raw = get_kfold_data(root_dir, cfg, n_folds=n_folds)
    print(f"Data loaded: {n_folds} folds, {len(test_raw)} test images.")

    # 3. Train each fold
    fold_results = []
    best_overall_loss = float("inf")
    best_fold_result = None

    for fold_idx, (train_loader, val_loader, val_full_pairs, label_min, label_max) in enumerate(folds):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/{n_folds}")
        print(f"{'='*60}")

        model = DynamicCNN(
            patch_size=cfg["patch_size"],
            kernel_size=cfg["kernel_size"],
            start_channels=cfg["start_channels"],
            num_layers=cfg["num_layers"],
            regressor_hidden_size=cfg["regressor_hidden_size"],
            dropout=cfg["dropout"],
        ).to(device)

        result = train_fold(
            model, train_loader, val_loader, val_full_pairs,
            cfg, device, label_min, label_max,
            fold_idx, n_folds,
        )
        fold_results.append(result)

        if result["best_val_loss"] < best_overall_loss:
            best_overall_loss = result["best_val_loss"]
            best_fold_result = result

    # 4. Log cross-fold summary
    val_losses = [r["best_val_loss"] for r in fold_results]
    image_mapes = [r["final_image_mape"] for r in fold_results]
    best_epochs = [r["best_epoch"] for r in fold_results]

    summary = {
        "val/mean_loss": float(np.mean(val_losses)),
        "val/std_loss": float(np.std(val_losses)),
        "val/mean_image_mape": float(np.mean(image_mapes)),
        "val/std_image_mape": float(np.std(image_mapes)),
        "val/best_fold_loss": best_overall_loss,
        "training/mean_best_epoch": float(np.mean(best_epochs)),
    }
    wandb.log(summary)

    print(f"\n{'='*60}")
    print(f"CROSS-FOLD SUMMARY")
    print(f"{'='*60}")
    print(f"Val Loss:      {summary['val/mean_loss']:.5f} +/- {summary['val/std_loss']:.5f}")
    print(f"Image MAPE:    {summary['val/mean_image_mape']:.2f}% +/- {summary['val/std_image_mape']:.2f}%")
    print(f"Best epochs:   {best_epochs}")

    # 5. Log mean train/val loss curves across folds
    max_epochs = max(len(r["train_losses"]) for r in fold_results)
    for epoch in range(max_epochs):
        train_vals = [r["train_losses"][epoch] for r in fold_results if epoch < len(r["train_losses"])]
        val_vals = [r["val_losses"][epoch] for r in fold_results if epoch < len(r["val_losses"])]
        wandb.log({
            "cv/mean_train_loss": float(np.mean(train_vals)),
            "cv/mean_val_loss": float(np.mean(val_vals)),
            "cv/epoch": epoch + 1,
        })

    # 6. Retrain on all train+val data
    avg_best_epoch = int(round(np.mean(best_epochs)))
    print(f"\n{'='*60}")
    print(f"FINAL RETRAIN (all train+val, {avg_best_epoch} epochs)")
    print(f"{'='*60}")

    # Build full training set from trainval_raw
    if data_source == "spinning_disk":
        full_train_samples, full_label_min, full_label_max, _ = _build_pairs_spinning_disk(
            raw_pairs=trainval_raw,
            patch_size=cfg["patch_size"],
            target_overlap_pct=cfg["target_overlap_pct"],
            transform_name=cfg.get("transform_name", "none"),
        )
    else:  # keyence
        full_train_samples, full_label_min, full_label_max, _ = _build_pairs_keyence(
            raw_pairs=trainval_raw,
            patch_size=cfg["patch_size"],
            target_overlap_pct=cfg["target_overlap_pct"],
            transform_name=cfg.get("transform_name", "none"),
            threshold_method=cfg["threshold_method"],
            blur_method=cfg["blur_method"],
            enhancement_method=cfg["enhancement_method"],
        )

    use_pin_memory = torch.cuda.is_available()
    full_train_loader = torch.utils.data.DataLoader(
        ImageLabelDataset(full_train_samples, rotate=True),
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=use_pin_memory,
    )

    final_model = DynamicCNN(
        patch_size=cfg["patch_size"],
        kernel_size=cfg["kernel_size"],
        start_channels=cfg["start_channels"],
        num_layers=cfg["num_layers"],
        regressor_hidden_size=cfg["regressor_hidden_size"],
        dropout=cfg["dropout"],
    ).to(device)

    final_model = train_final(final_model, full_train_loader, cfg, device, avg_best_epoch)

    # 7. Test evaluation with retrained model
    print(f"\n{'='*60}")
    print(f"TEST EVALUATION (retrained model)")
    print(f"{'='*60}")

    if data_source == "spinning_disk":
        _, _, _, test_full_pairs = _build_pairs_spinning_disk(
            raw_pairs=test_raw,
            patch_size=cfg["patch_size"],
            target_overlap_pct=cfg["target_overlap_pct"],
            transform_name=cfg.get("transform_name", "none"),
            label_min=full_label_min,
            label_max=full_label_max,
        )
    else:  # keyence
        _, _, _, test_full_pairs = _build_pairs_keyence(
            raw_pairs=test_raw,
            patch_size=cfg["patch_size"],
            target_overlap_pct=cfg["target_overlap_pct"],
            transform_name=cfg.get("transform_name", "none"),
            threshold_method=cfg["threshold_method"],
            blur_method=cfg["blur_method"],
            enhancement_method=cfg["enhancement_method"],
            label_min=full_label_min,
            label_max=full_label_max,
        )

    test_results = final_evaluation(
        final_model, test_full_pairs, cfg, device,
        full_label_min, full_label_max,
    )
    log_detailed_results(test_results, test_full_pairs, prefix="test")

    # Scale-independent metrics
    preds = torch.tensor([r["predicted"] for r in test_results])
    actuals = torch.tensor([r["actual"] for r in test_results])
    test_mapes = [r["pct_error"] for r in test_results]

    wandb.log({
        "test/r2": calculate_r2(preds, actuals),
        "test/nrmse": calculate_nrmse(preds, actuals),
        "test/mean_mape": float(np.mean(test_mapes)),
        "test/median_mape": float(np.median(test_mapes)),
    })
    print(f"Test R²: {calculate_r2(preds, actuals):.4f}")
    print(f"Test NRMSE: {calculate_nrmse(preds, actuals):.4f}")
    print(f"Test MAPE: {np.mean(test_mapes):.2f}% "
          f"(median {np.median(test_mapes):.2f}%)")

    # Save predictions to local CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/{data_source}_{wandb.run.id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["filename", "predicted", "actual", "abs_error", "pct_error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(test_results)
    wandb.save(csv_path)
    print(f"Predictions saved to {csv_path}")

    # Save final model as W&B artifact
    model_path = "best_model.pth"
    torch.save(final_model.state_dict(), model_path)
    artifact = wandb.Artifact("best-model", type="model")
    artifact.add_file(model_path)
    wandb.log_artifact(artifact)
    print(f"Final model saved as W&B artifact.")


if __name__ == "__main__":
    wandb.init()
    run(wandb.config)
