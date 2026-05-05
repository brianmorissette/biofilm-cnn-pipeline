import numpy as np
import cv2
import scipy.fft, scipy.ndimage
from skimage import filters


def extract_patches_auto(image, patch_size):
    """
    Extract patches with minimal overlap while ensuring full coverage.

    Uses the minimum number of patches needed to cover each axis:
      - If image dimensions are divisible by patch_size, overlap is zero.
      - Otherwise, patches are evenly spaced with only the overlap required
        to include the image borders and avoid pixel loss.
    
    Args:
        image (np.ndarray): Input image.
        patch_size (int): Size of the patch (height and width).

    Returns:
        List of extracted patches (np.ndarray(H,W) float in [0,1]).
    """
    h, w = image.shape[:2]

    if patch_size <= 0:
        raise ValueError("patch_size must be > 0")

    # Minimum patches needed to cover each axis.
    n_x = int(np.ceil(w / patch_size))
    n_y = int(np.ceil(h / patch_size))

    # Generate evenly spaced coordinates with locked corners.
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
    elif transform_name == "gabor":
        return gabor(image)
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

def gabor(image):
    """Applies Gabor filter to the image."""
    real, imag = filters.gabor(image, frequency=0.5)
    magnitude = np.sqrt(real**2 + imag**2)
    return magnitude



