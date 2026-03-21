import copy
import csv
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from model import DynamicCNN
from dataset import get_kfold_data, _build_pairs_spinning_disk, _build_pairs_keyence, ImageLabelDataset
from release_preprocess import extract_patches_auto, apply_transform


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def denormalize(normalized_label, min_val, max_val):
    """Converts normalized (0-1) values back to original units."""
    return normalized_label * (max_val - min_val) + min_val


def calculate_r2(pred, target):
    """Coefficient of determination (R²). Scale-independent."""
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - torch.mean(target)) ** 2)
    return (1 - ss_res / ss_tot).item() if ss_tot > 0 else 0.0


def calculate_mae(pred, target):
    """Mean Absolute Error in original units."""
    if not isinstance(pred, torch.Tensor):
        pred = torch.tensor(pred, dtype=torch.float32)
    if not isinstance(target, torch.Tensor):
        target = torch.tensor(target, dtype=torch.float32)
    return torch.mean(torch.abs(pred - target)).item()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

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

        pred_orig = denormalize(avg_pred_norm, label_min, label_max)
        target_orig = denormalize(full_label, label_min, label_max)

        results.append({
            "image_idx": idx,
            "filename": filename,
            "predicted": pred_orig,
            "actual": target_orig,
        })

    return results


def log_detailed_results(results, prefix="test"):
    """
    Log detailed prediction results to W&B:
    - Predictions table (filename, predicted, actual)
    - Scatter plot of predicted vs actual
    """
    # 1. Predictions table
    columns = ["filename", "predicted", "actual"]
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


# ---------------------------------------------------------------------------
# Single-fold training
# ---------------------------------------------------------------------------

def train_fold(
    model, train_loader, val_loader,
    cfg, device, label_min, label_max,
    fold_idx, n_folds,
):
    """
    Train a single fold and return metrics + best model state dict.

    Includes:
      - L1 loss on normalized [0,1] labels
      - ReduceLROnPlateau scheduler
      - Patience-based early stopping
      - Per-epoch console logging: train loss, val loss, LR
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

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)

                preds = model(inputs)
                loss = criterion(preds, targets)
                val_running_loss += loss.item() * inputs.size(0)

        epoch_val_loss = val_running_loss / len(val_loader.dataset)

        train_losses.append(epoch_train_loss)
        val_losses_per_epoch.append(epoch_val_loss)

        # --- LR SCHEDULER ---
        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(epoch_val_loss)

        # --- LOGGING (console only, W&B logging done after all folds) ---
        print(f"  [{prefix}] Epoch {epoch + 1}/{cfg['epochs']} | "
              f"Train Loss: {epoch_train_loss:.5f} | "
              f"Val Loss: {epoch_val_loss:.5f} | "
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

    print(f"  [{prefix}] Best epoch: {best_epoch} | "
          f"Val Loss: {best_val_loss:.5f}")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
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
    # Validate CUDA availability early — catch GPU errors before loading data
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            device = torch.device("cuda")
        except RuntimeError as e:
            print(f"\nCUDA ERROR: GPU detected but unavailable.")
            print(f"  {e}")
            print(f"  This typically means the GPU is busy or out of memory.")
            print(f"  Try again later or check `nvidia-smi` for GPU status.")
            sys.exit(1)
    else:
        device = torch.device("cpu")
    print(f"Running on {device}")

    # 1. Determine root directory
    # DATA_ROOT points directly to the processed data for a given dataset.
    # If not set, fall back to local data/<dataset>/processed directory.
    data_source = cfg["data_source"]
    root_dir = os.environ.get("DATA_ROOT")
    if root_dir is None:
        if data_source in ("keyence", "keyence_clearest"):
            root_dir = "data/keyence/processed"
        elif data_source == "spinning_disk":
            root_dir = "data/spinning_disk/processed"
        else:
            raise ValueError(f"Unknown data_source: {data_source}")
    print(f"Data root: {root_dir}")

    n_folds = cfg.get("n_folds", 3)

    # Log dataset info
    wandb.config.update({"root_dir": root_dir}, allow_val_change=True)

    # 2. Load k-fold data
    folds, test_raw, trainval_raw = get_kfold_data(root_dir, cfg, n_folds=n_folds)
    print(f"Data loaded: {n_folds} folds, {len(test_raw)} test images.")

    # Print hyperparameter summary
    print(f"\n{'='*60}")
    print(f"HYPERPARAMETERS")
    print(f"{'='*60}")
    hp_keys = ["patch_size", "num_layers", "start_channels", "kernel_size",
               "regressor_hidden_size", "dropout", "target_overlap_pct",
               "batch_size", "learning_rate", "weight_decay"]
    for key in hp_keys:
        if key in cfg:
            print(f"  {key}: {cfg[key]}")
    print(f"  data_source: {data_source}")
    print(f"  n_folds: {n_folds}")
    print(f"  epochs: {cfg['epochs']}")

    # 3. Train each fold
    fold_results = []

    for fold_idx, (train_loader, val_loader, _val_full_pairs, label_min, label_max) in enumerate(folds):
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
            model, train_loader, val_loader,
            cfg, device, label_min, label_max,
            fold_idx, n_folds,
        )
        fold_results.append(result)

    # 4. Log cross-fold summary
    val_losses = [r["best_val_loss"] for r in fold_results]
    best_epochs = [r["best_epoch"] for r in fold_results]

    summary = {
        "val/mean_loss": float(np.mean(val_losses)),
        "val/std_loss": float(np.std(val_losses)),
    }
    wandb.log(summary)

    print(f"\n{'='*60}")
    print(f"CROSS-FOLD SUMMARY")
    print(f"{'='*60}")
    print(f"{'Fold':<6} {'Epoch':<7} {'Val Loss':<10}")
    print(f"{'-'*23}")
    for i in range(len(fold_results)):
        print(f"{i+1:<6} {best_epochs[i]:<7} {val_losses[i]:<10.5f}")
    print(f"{'-'*23}")
    print(f"{'Mean':<6} {np.mean(best_epochs):<7.1f} {np.mean(val_losses):<10.5f}")
    print(f"{'Std':<6} {'':7} {np.std(val_losses):<10.5f}")

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
    log_detailed_results(test_results, prefix="test")

    preds = torch.tensor([r["predicted"] for r in test_results])
    actuals = torch.tensor([r["actual"] for r in test_results])

    test_r2 = calculate_r2(preds, actuals)
    test_mae = calculate_mae(preds, actuals)

    wandb.log({
        "test/r2": test_r2,
        "test/mae": test_mae,
    })
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test MAE: {test_mae:.2f}")

    # Save predictions to local CSV
    os.makedirs("outputs", exist_ok=True)
    csv_path = f"outputs/{data_source}_{wandb.run.id}_predictions.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = ["filename", "predicted", "actual"]
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
