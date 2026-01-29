import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

from model import DynamicCNN
from dataset import get_dataloaders
from release_preprocess import extract_patches_auto, apply_transform


def denormalize(normalized_label, min_val, max_val):
    """
    Converts normalized (0-1) values back to original micron units.
    """
    return normalized_label * (max_val - min_val) + min_val

def calculate_mape(pred_microns, target_microns, epsilon=1e-7):
    """
    Calculates Mean Absolute Percentage Error (MAPE).
    Handles both tensors and scalar values.
    Added epsilon to avoid division by zero.
    """
    # Convert to tensors if they're scalars
    if not isinstance(pred_microns, torch.Tensor):
        pred_microns = torch.tensor(pred_microns)
    if not isinstance(target_microns, torch.Tensor):
        target_microns = torch.tensor(target_microns)
    
    return torch.mean(torch.abs((pred_microns - target_microns) / (target_microns + epsilon))) * 100


def evaluate_full_images(model, full_pairs, cfg, device, label_min, label_max):
    """
    Evaluates the model on full images by:
    1. Taking the full pre-processed image (already normalized).
    2. Extracting patches dynamically.
    3. Applying transforms if necessary.
    4. Averaging the predictions of all patches.
    """
    model.eval()
    image_mapes = []
    
    # full_pairs is a list of tuples: (normalized_release_image, normalized_label)
    for full_image, full_label in full_pairs:
        
        # 1. Extract patches
        patches = extract_patches_auto(
            full_image, 
            patch_size=cfg["patch_size"], 
            target_overlap=cfg["target_overlap"]
        )

        # 2. Apply Transform (if any)
        transform_name = cfg.get("transform_name")
        if transform_name != "none":
            patches = [apply_transform(p, transform_name) for p in patches]

        # 3. Batch patches for the model
        patches_np = np.array(patches)
        # Convert to Tensor and add Channel dim: (N, H, W) -> (N, 1, H, W)
        patches_tensor = torch.from_numpy(patches_np).float().unsqueeze(1).to(device)
        
        # 4. Predict
        with torch.no_grad():
            patch_preds = model(patches_tensor)
        
        # 5. Aggregate: Average the normalized scores
        avg_pred_norm = patch_preds.mean().item()
        
        # 6. Denormalize both to Microns for valid MAPE calculation
        pred_microns = denormalize(avg_pred_norm, label_min, label_max)
        target_microns = denormalize(full_label, label_min, label_max)
        
        # 7. Calculate MAPE
        mape = calculate_mape(pred_microns, target_microns)
        image_mapes.append(mape)

    # Return the average MAPE across all validation images
    return np.mean(image_mapes)

def run(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. Data Loading
    root_dir = "../data/processed" 
    
    (train_loader, 
     val_loader, 
     test_loader, 
     train_min, 
     train_max, 
     val_full_pairs, 
     test_full_pairs) = get_dataloaders(root_dir, cfg)

    print(f"Data Loaded.")

    # 2. Model Setup
    model = DynamicCNN(
        patch_size=cfg["patch_size"],
        kernel_size=cfg["kernel_size"],
        start_channels=cfg["start_channels"],
        num_layers=cfg["num_layers"],
        regressor_hidden_size=cfg["regressor_hidden_size"],
        dropout=cfg["dropout"]
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg.get("weight_decay"))
    criterion = nn.L1Loss() 

    # 3. Training Loop
    best_val_loss = float('inf')  # necessary when saving best model

    for epoch in range(cfg["epochs"]):
        # --- PHASE 1: TRAINING ---
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            targets = targets.view(-1, 1) # row -> column to match CNN output

            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, targets) # Loss on Normalized Data (0-1)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0) # Allows for different batch sizes to have equal weight
            
        epoch_train_loss = running_loss / len(train_loader.dataset)

        # --- PHASE 2: VALIDATION (PATCH LEVEL) ---
        model.eval()
        val_running_loss = 0.0
        val_running_mape = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                targets = targets.view(-1, 1)
                
                preds = model(inputs)
                
                # Loss (Normalized)
                loss = criterion(preds, targets)
                val_running_loss += loss.item() * inputs.size(0)
                
                # MAPE (Real Microns)
                pred_microns = denormalize(preds, train_min, train_max)
                target_microns = denormalize(targets, train_min, train_max)
                batch_mape = calculate_mape(pred_microns, target_microns)
                
                val_running_mape += batch_mape.item() * inputs.size(0)
        
        epoch_val_loss = val_running_loss / len(val_loader.dataset)
        epoch_patch_mape = val_running_mape / len(val_loader.dataset)

        # --- PHASE 3: VALIDATION (FULL IMAGE LEVEL) ---
        epoch_image_mape = evaluate_full_images(model, val_full_pairs, cfg, device, train_min, train_max)

        # --- LOGGING ---
        print(f"Epoch {epoch+1}/{cfg['epochs']} | "
              f"Train Loss: {epoch_train_loss:.5f} | "
              f"Val Loss: {epoch_val_loss:.5f} | "
              f"Val Patch MAPE: {epoch_patch_mape:.2f}% | "
              f"Val Image MAPE: {epoch_image_mape:.2f}%")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": epoch_train_loss,
            "val/loss": epoch_val_loss,
            "val/patch_mape": epoch_patch_mape,
            "val_image_mape": epoch_image_mape
        })

        # Save Best Model
        # if epoch_val_loss < best_val_loss:
        #     best_val_loss = epoch_val_loss
        #     torch.save(model.state_dict(), "best_model.pth")
        #     wandb.save("best_model.pth")

if __name__ == '__main__':
    wandb.init(project="biofilm-cnn-pipeline-sweep-v1")
    run(wandb.config)