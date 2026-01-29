import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import cv2
from pathlib import Path

from release_preprocess import extract_patches_auto, rotate_image_90, rotate_image_180, rotate_image_270, apply_transform
from biofilm_preprocess import preprocess_biofilm, get_iterative_threshold_value, get_surface_area, normalize_surface_areas


def load_images(root) -> list[np.ndarray]:
    """
    Loads all .tif images from the specified root directory and its subdirectories.
    """
    paths = sorted([*Path(root).rglob("*.tif")], key=lambda p: p.as_posix().casefold())
    return [img for p in paths if (img := cv2.imread(str(p), cv2.IMREAD_UNCHANGED)) is not None]

    
def _build_pairs(raw_pairs, patch_size, target_overlap, transform_name, rotate=True, label_min=None, label_max=None):    
    """
    Builds (patch, label) pairs from raw images (biofilm, release).

    Args:
        raw_pairs: List of (biofilm, release) image tuples.
        patch_size: Pixel size of the patches. (e.g. 64, 80, 164)
        target_overlap: Target overlap percentage for patch extraction.
        transform_name: Name of the transform to apply to images (none, fft_dct, mexican_hat).
        rotate: Boolean if the patches should be rotated (true for train, false for val & test)
        label_min: Minimum label value for normalization (optional).
        label_max: Maximum label value for normalization (optional).
    
    Returns:
        Tuple of (samples, label_min, label_max, pre_patch_pairs), where samples is a list of (patch, label) tuples,
        and pre_patch_pairs is a list of (full image, label) tuples
    """    
    # 1) Per-image preprocessing + label (no patches yet)
    pre_patch_pairs = []
    all_release = []
    all_labels = []
    for biofilm, release in raw_pairs:
        # release -> grayscale + normalize
        grayscale_release = release[:, :, 1]
        normalized_release = grayscale_release / np.max(grayscale_release)
        all_release.append(normalized_release)

        # biofilm -> preprocess + threshold + label (surface area)
        preprocessed_biofilm = preprocess_biofilm(biofilm)
        threshold = get_iterative_threshold_value(preprocessed_biofilm)
        surface_area = get_surface_area(preprocessed_biofilm, threshold)
        all_labels.append(surface_area)

    # normalize the labels 0-1
    normalized_labels, label_min, label_max = normalize_surface_areas(all_labels, min_val=label_min, max_val=label_max)

    pre_patch_pairs = list(zip(all_release, normalized_labels))

    # 2) extract patches + rotations (original + 90/180/270) (rotate only for Train)
    samples = []
    for release, biofilm_label in pre_patch_pairs:
        for patch in extract_patches_auto(release, patch_size=patch_size, target_overlap=target_overlap):
            samples.append((patch, biofilm_label))
            if rotate:
                samples.append((rotate_image_90(patch), biofilm_label))
                samples.append((rotate_image_180(patch), biofilm_label))
                samples.append((rotate_image_270(patch), biofilm_label))

    # apply transform
    if transform_name != "none":
        samples = [(apply_transform(image, transform_name), label) for image, label in samples]

    return samples, label_min, label_max, pre_patch_pairs


class ImageLabelDataset(Dataset):
    """
    Dataset wrapper for (image, label) pairs, preparing for CNN.
    """
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        img, y = self.samples[i]
        # ensure single-channel, channel-first (1,H,W)
        if img.ndim == 2:
            img = img[None, ...]
        elif img.ndim == 3 and img.shape[-1] == 1:
            img = np.transpose(img, (2, 0, 1))
        x = torch.from_numpy(img.astype(np.float32))  # image -> float32 tensor
        y = torch.tensor(y, dtype=torch.float32)  # label -> float32 tensor (regression)
        return x, y


def get_dataloaders(root, cfg):
    """
    Build train/test DataLoaders with leakage-free split:
    split on original images first, then patch/augment within each split.

    Args:
        root: Root directory containing 'biofilm' and 'release' folders.
        cfg: Configuration dictionary containing hyperparameters.

    Returns:
        Tuple of (train_loader, validation_loader, test_loader, train_min, train_max).
    """
    # load paired raw images
    biofilm_dir = f"{root}/biofilm"
    release_dir = f"{root}/release"
    biofilm_images = load_images(biofilm_dir)
    release_images = load_images(release_dir)
    raw_pairs = list(zip(biofilm_images, release_images))

    # train/test split at image level (pre-augmentation)
    train_raw, test_raw = train_test_split(raw_pairs, train_size=0.9, random_state=42, shuffle=True)
    train_raw, validation_raw = train_test_split(train_raw, train_size=0.8, random_state=42, shuffle=True)

    # build (img,label) samples for each split
    train_samples, train_min, train_max, _ = _build_pairs(
        raw_pairs=train_raw,
        patch_size=cfg["patch_size"],
        target_overlap=cfg["target_overlap"],
        transform_name=cfg["transform_name"],
    )

    # Use Training Min/Max for Validation and Test
    validation_samples, _, _, validation_full_pairs = _build_pairs(
        raw_pairs=validation_raw,
        patch_size=cfg["patch_size"],
        target_overlap=cfg["target_overlap"],
        transform_name=cfg["transform_name"],
        label_min=train_min,
        label_max=train_max,
        rotate=False
    )
    test_samples, _, _, test_full_pairs = _build_pairs(
        raw_pairs=test_raw,
        patch_size=cfg["patch_size"],
        target_overlap=cfg["target_overlap"],
        transform_name=cfg["transform_name"],
        label_min=train_min,
        label_max=train_max,
        rotate=False
    )

    # Wrap in ImageLabelDataset
    train_samples = ImageLabelDataset(train_samples)
    validation_samples = ImageLabelDataset(validation_samples)
    test_samples = ImageLabelDataset(test_samples)

    # Use pin memory if available
    use_pin_memory = torch.cuda.is_available()
    num_workers = 2

    # Create DataLoaders
    train_loader = DataLoader(
        train_samples,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    validation_loader = DataLoader(
        validation_samples,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    test_loader = DataLoader(
        test_samples,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )
    return train_loader, validation_loader, test_loader, train_min, train_max, validation_full_pairs, test_full_pairs
