#!/usr/bin/env python3
"""
Test script to debug data loading issues.
Tests the same data loading path as train.py and checks for common issues.
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add src to path to import modules
sys.path.insert(0, str(Path(__file__).parent))

from dataset import load_images, get_dataloaders

def test_paths():
    """Test if data directories exist."""
    print("=" * 60)
    print("TEST 1: Checking data directory paths")
    print("=" * 60)
    
    root_dir = "data/processed"
    biofilm_dir = f"{root_dir}/biofilm"
    release_dir = f"{root_dir}/release"
    
    root_path = Path(root_dir)
    biofilm_path = Path(biofilm_dir)
    release_path = Path(release_dir)
    
    print(f"Root directory: {root_path.absolute()}")
    print(f"  Exists: {root_path.exists()}")
    
    print(f"\nBiofilm directory: {biofilm_path.absolute()}")
    print(f"  Exists: {biofilm_path.exists()}")
    
    print(f"\nRelease directory: {release_path.absolute()}")
    print(f"  Exists: {release_path.exists()}")
    
    if not biofilm_path.exists() or not release_path.exists():
        print("\n❌ ERROR: One or both directories don't exist!")
        return False
    
    print("\n✅ All directories exist")
    return True


def test_image_loading():
    """Test loading images from directories."""
    print("\n" + "=" * 60)
    print("TEST 2: Loading images from directories")
    print("=" * 60)
    
    root_dir = "data/processed"
    biofilm_dir = f"{root_dir}/biofilm"
    release_dir = f"{root_dir}/release"
    
    print(f"Loading biofilm images from: {biofilm_dir}")
    biofilm_images = load_images(biofilm_dir)
    print(f"  Loaded {len(biofilm_images)} biofilm images")
    
    print(f"\nLoading release images from: {release_dir}")
    release_images = load_images(release_dir)
    print(f"  Loaded {len(release_images)} release images")
    
    if len(biofilm_images) == 0:
        print("\n❌ ERROR: No biofilm images loaded!")
        return False, None, None
    
    if len(release_images) == 0:
        print("\n❌ ERROR: No release images loaded!")
        return False, None, None
    
    # Check image shapes
    if len(biofilm_images) > 0:
        print(f"\nBiofilm image shapes:")
        for i, img in enumerate(biofilm_images[:3]):
            print(f"  Image {i}: {img.shape}, dtype={img.dtype}, min={img.min():.2f}, max={img.max():.2f}")
    
    if len(release_images) > 0:
        print(f"\nRelease image shapes:")
        for i, img in enumerate(release_images[:3]):
            print(f"  Image {i}: {img.shape}, dtype={img.dtype}, min={img.min():.2f}, max={img.max():.2f}")
    
    print("\n✅ Images loaded successfully")
    return True, biofilm_images, release_images


def test_pair_matching(biofilm_images, release_images):
    """Test if biofilm and release images can be paired correctly."""
    print("\n" + "=" * 60)
    print("TEST 3: Checking image pair matching")
    print("=" * 60)
    
    print(f"Biofilm images: {len(biofilm_images)}")
    print(f"Release images: {len(release_images)}")
    
    if len(biofilm_images) != len(release_images):
        print(f"\n⚠️  WARNING: Mismatch! Biofilm has {len(biofilm_images)} images, Release has {len(release_images)} images")
        print(f"  Difference: {abs(len(biofilm_images) - len(release_images))} images")
        print(f"  zip() will create {min(len(biofilm_images), len(release_images))} pairs")
        print(f"  {max(len(biofilm_images), len(release_images)) - min(len(biofilm_images), len(release_images))} images will be unused")
    else:
        print("\n✅ Image counts match perfectly!")
    
    raw_pairs = list(zip(biofilm_images, release_images))
    print(f"\nCreated {len(raw_pairs)} pairs")
    
    if len(raw_pairs) == 0:
        print("\n❌ ERROR: No pairs created!")
        return False
    
    print("\n✅ Pairs created successfully")
    return True


def test_get_dataloaders():
    """Test get_dataloaders function with a mock config."""
    print("\n" + "=" * 60)
    print("TEST 4: Testing get_dataloaders function")
    print("=" * 60)
    
    root_dir = "data/processed"
    
    # Create a mock config that matches what the code currently expects
    # Note: sweep.yml uses slightly different names; this is the \"internal\" schema
    mock_cfg = {
        "patch_size": 128,
        "target_overlap_pct": 0.25,  # internal name used in dataset/train
        "transform_name": "none",
        "batch_size": 16,
        # These are needed for model creation but not for data loading
        "kernel_size": 3,
        "start_channels": 16,
        "num_layers": 3,  # Note: sweep.yml has "num_conv_layers"
        "regressor_hidden_size": 128,
        "dropout": 0.2,
        "learning_rate": 0.001,
        "epochs": 1,
    }
    
    print("Mock config:")
    for key, value in mock_cfg.items():
        print(f"  {key}: {value}")
    
    try:
        print(f"\nCalling get_dataloaders(root_dir='{root_dir}', cfg=mock_cfg)...")
        result = get_dataloaders(root_dir, mock_cfg)
        
        (train_loader, 
         val_loader, 
         test_loader, 
         train_min, 
         train_max, 
         val_full_pairs, 
         test_full_pairs) = result
        
        print("\n✅ get_dataloaders completed successfully!")
        print(f"\nResults:")
        print(f"  Train loader: {len(train_loader.dataset)} samples")
        print(f"  Val loader: {len(val_loader.dataset)} samples")
        print(f"  Test loader: {len(test_loader.dataset)} samples")
        print(f"  Train min: {train_min:.6f}")
        print(f"  Train max: {train_max:.6f}")
        print(f"  Val full pairs: {len(val_full_pairs)}")
        print(f"  Test full pairs: {len(test_full_pairs)}")
        
        # Test a batch
        print(f"\nTesting a batch from train loader...")
        for inputs, targets in train_loader:
            print(f"  Batch shape - inputs: {inputs.shape}, targets: {targets.shape}")
            print(f"  Input range: [{inputs.min():.4f}, {inputs.max():.4f}]")
            print(f"  Target range: [{targets.min():.4f}, {targets.max():.4f}]")
            break
        
        return True
        
    except KeyError as e:
        print(f"\n❌ ERROR: Missing config key: {e}")
        print("This suggests a mismatch between sweep.yml parameter names and code expectations")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_config_mismatch():
    """Check for mismatches between sweep.yml and code expectations."""
    print("\n" + "=" * 60)
    print("TEST 5: Checking config parameter name mismatches")
    print("=" * 60)
    
    # Simulate what wandb.config would contain from sweep.yml
    wandb_config_keys = [
        "batch_size",
        "dropout",
        "kernel_size",
        "learning_rate",
        "num_conv_layers",  # ❌ Code expects "num_layers"
        "patch_size",
        "regressor_hidden_size",
        "start_channels",
        "target_overlap_percentage",  # ❌ Code expects "target_overlap"
        "transform_name",
        "weight_decay",
    ]
    
    # What the code actually expects (internal names)
    code_expected_keys = {
        "patch_size": "dataset.py, train.py",
        "target_overlap_pct": "dataset.py, train.py",  # ❌ sweep.yml has "target_overlap_percentage"
        "transform_name": "dataset.py, train.py",
        "batch_size": "dataset.py",
        "kernel_size": "train.py",
        "start_channels": "train.py",
        "num_layers": "train.py",  # ❌ sweep.yml has "num_conv_layers"
        "regressor_hidden_size": "train.py",
        "dropout": "train.py",
        "learning_rate": "train.py",
        "epochs": "train.py",
        "weight_decay": "train.py (optional)",
    }
    
    print("Config keys from sweep.yml:")
    for key in wandb_config_keys:
        marker = "❌" if key not in code_expected_keys else "✅"
        print(f"  {marker} {key}")
    
    print("\nConfig keys expected by code:")
    for key, location in code_expected_keys.items():
        marker = "❌" if key not in wandb_config_keys else "✅"
        print(f"  {marker} {key} (used in {location})")
    
    print("\n⚠️  MISMATCHES FOUND:")
    if "target_overlap_percentage" in wandb_config_keys and "target_overlap_pct" in code_expected_keys:
        print("  - sweep.yml has 'target_overlap_percentage' but code expects 'target_overlap_pct'")
    if "num_conv_layers" in wandb_config_keys and "num_layers" in code_expected_keys:
        print("  - sweep.yml has 'num_conv_layers' but code expects 'num_layers'")
    
    print("\n💡 SOLUTION: Add parameter mapping in train.py before calling get_dataloaders:")
    print("   cfg['target_overlap_pct'] = cfg.get('target_overlap_percentage', cfg.get('target_overlap_pct', 0.25))")
    print("   cfg['num_layers'] = cfg.get('num_conv_layers', cfg.get('num_layers', 3))")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("DATA LOADING TEST SUITE")
    print("=" * 60)
    
    results = []
    
    # Test 1: Paths
    results.append(("Paths", test_paths()))
    
    # Test 2: Image loading
    success, biofilm_images, release_images = test_image_loading()
    results.append(("Image Loading", success))
    
    if not success:
        print("\n❌ Cannot continue - image loading failed")
        return
    
    # Test 3: Pair matching
    results.append(("Pair Matching", test_pair_matching(biofilm_images, release_images)))
    
    # Test 4: get_dataloaders
    results.append(("get_dataloaders", test_get_dataloaders()))
    
    # Test 5: Config mismatch check
    check_config_mismatch()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result[1] for result in results)
    
    if all_passed:
        print("\n🎉 All tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review the output above.")


if __name__ == "__main__":
    main()
