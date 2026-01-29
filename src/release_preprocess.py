import numpy as np
import cv2
import scipy.fft, scipy.ndimage


def extract_patches_auto(image, patch_size, target_overlap):
    """
    Automatically calculates the number of patches needed to achieve 
    'target_overlap' while ensuring the 4 corners are locked and no pixels are lost.
    
    Args:
        image (np.ndarray): Input image.
        patch_size (int): Size of the patch (height and width).
        target_overlap (float): Desired overlap (e.g., 0.25 for 25%).

    Returns:
        List of extracted patches (np.ndarray(H,W) float in [0,1]).
    """
    h, w = image.shape[:2]
    
    def calculate_n(dimension, p_size, t_overlap):
        # 1. Calculate the theoretical stride needed for this overlap
        target_stride = p_size * (1 - t_overlap)
        
        # 2. Calculate how many steps we need to traverse the dimension
        # Formula: (Distance_to_cover) / stride + 1
        # We round to the nearest integer to find the closest valid configuration
        n_float = (dimension - p_size) / target_stride + 1
        n = int(np.round(n_float))
        
        # 3. Constraint: We must have enough patches to cover the image.
        # Minimal patches = Ceiling(Dimension / Patch_Size)
        min_required = int(np.ceil(dimension / p_size))
        return max(n, min_required)

    n_x = calculate_n(w, patch_size, target_overlap)
    n_y = calculate_n(h, patch_size, target_overlap)

    # --- Step 2: Compute Actual Overlaps (for reporting) ---
    
    # Actual stride = (Distance) / (Steps)
    # Stride = (W - P) / (n - 1)
    # We guard against divide by zero if n=1 (single patch covers whole image)
    real_stride_x = (w - patch_size) / (n_x - 1) if n_x > 1 else 0
    real_stride_y = (h - patch_size) / (n_y - 1) if n_y > 1 else 0
    
    real_overlap_x = 1 - (real_stride_x / patch_size) if patch_size > 0 else 0
    real_overlap_y = 1 - (real_stride_y / patch_size) if patch_size > 0 else 0

    # --- Step 3: Extract Patches using Linspace ---
    
    # Generate evenly spaced coordinates
    x_coords = np.linspace(0, w - patch_size, n_x, dtype=int)
    y_coords = np.linspace(0, h - patch_size, n_y, dtype=int)
    
    patches = []
    for y in y_coords:
        for x in x_coords:
            patches.append(image[y:y + patch_size, x:x + patch_size])
            
    return patches


def rotate_image_90(image):
    """Rotates an image 90 degrees clockwise."""
    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)


def rotate_image_180(image):
    """Rotates an image 180 degrees."""
    return cv2.rotate(image, cv2.ROTATE_180)


def rotate_image_270(image):
    """Rotates an image 270 degrees (90 degrees counter-clockwise)."""
    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)


def apply_transform(image, transform_name):
    """
    Applies a specified transform to an image.
    """
    if transform_name == "none":
        return image
    elif transform_name == "fft_dct":
        return fft_dct(image)
    elif transform_name == "mexican_hat":
        return mexican_hat(image)
    else:
        raise ValueError(f"Invalid transform: {transform_name}")


def fft_dct(image):
    """Applies Discrete Cosine Transform (DCT) to the image."""
    dct_image = scipy.fft.dctn(image, type=2, norm="ortho")
    return dct_image


def mexican_hat(image, size=21, sigma=3.0):
    """Applies Mexican Hat transform to the image."""
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    r2 = X ** 2 + Y ** 2
    kernel = (1 - r2 / (2 * sigma ** 2)) * np.exp(-r2 / (2 * sigma ** 2))
    kernel_sum = kernel.sum()
    kernel = kernel / (kernel_sum if kernel_sum != 0 else 1.0)
    transformed_image = scipy.ndimage.convolve(image, kernel, mode="reflect")
    return transformed_image
