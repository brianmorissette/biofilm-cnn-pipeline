import cv2
import numpy as np


def preprocess_biofilm(
    image,
    enhancement_method,
    blur_method,
    clip_limit=2.0,
    tile_size=(8, 8),
    blur_ksize=(5, 5),
):
    """
    Preprocesses a biofilm image with configurable enhancement and blur methods.

    Args:
        image: Input image.
        enhancement_method: Contrast enhancement method ("clahe", "histogram_eq", "none").
        blur_method: Blurring method ("gaussian", "median", "none").
        clip_limit: CLAHE clip limit (only used if enhancement_method="clahe").
        tile_size: CLAHE tile grid size (only used if enhancement_method="clahe").
        blur_ksize: Blur kernel size (used for gaussian and median blur).

    Returns:
        Preprocessed image.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray_image = image

    # Apply contrast enhancement
    if enhancement_method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
        enhanced_image = clahe.apply(gray_image)
    elif enhancement_method == "histogram_eq":
        enhanced_image = cv2.equalizeHist(gray_image)
    elif enhancement_method == "none":
        enhanced_image = gray_image
    else:
        raise ValueError(f"Unknown enhancement_method: {enhancement_method}")

    # Normalize to 0-255
    normalized_image = cv2.normalize(
        src=enhanced_image,
        dst=None,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    # Apply blur
    if blur_method == "gaussian":
        preprocessed_image = cv2.GaussianBlur(normalized_image, blur_ksize, 0)
    elif blur_method == "median":
        # medianBlur requires an odd integer kernel size
        ksize = blur_ksize[0] if isinstance(blur_ksize, tuple) else blur_ksize
        preprocessed_image = cv2.medianBlur(normalized_image, ksize)
    elif blur_method == "none":
        preprocessed_image = normalized_image
    else:
        raise ValueError(f"Unknown blur_method: {blur_method}")

    return preprocessed_image


def get_iterative_threshold_value(image):
    """
    Calculates an optimal threshold using an iterative approach (IsoData).
    """
    iteration_count = 0
    current_threshold = 127.0
    last_threshold = -1.0
    tolerance = 0.5
    
    while abs(current_threshold - last_threshold) > tolerance:
        iteration_count += 1
        last_threshold = current_threshold
        background_pixels = image[image <= current_threshold]
        foreground_pixels = image[image > current_threshold]
        if background_pixels.size == 0:
            mean_bg = 0.0
        else:
            mean_bg = np.mean(background_pixels)
        if foreground_pixels.size == 0:
            mean_fg = 255.0
        else:
            mean_fg = np.mean(foreground_pixels)
        current_threshold = (mean_bg + mean_fg) / 2.0
    return int(round(current_threshold))


def get_otsu_threshold_value(image):
    """
    Calculates an optimal threshold using Otsu's method.
    
    Otsu's method maximizes inter-class variance between foreground and background.
    """
    thresh_val, _ = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return int(thresh_val)


def get_adaptive_threshold_mask(image, block_size=11, c_constant=2):
    """
    Applies adaptive thresholding and returns a binary mask.
    
    Adaptive thresholding calculates a different threshold for each pixel
    based on the local neighborhood, making it effective for images with
    varying illumination.
    
    Args:
        image: Input grayscale image (uint8).
        block_size: Size of the pixel neighborhood for threshold calculation (must be odd).
        c_constant: Constant subtracted from the mean.
        
    Returns:
        Binary mask where foreground pixels are 255.
    """
    return cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        c_constant,
    )


def get_surface_area(image, threshold_method):
    """
    Calculate the biofilm surface area in square microns using the specified threshold method.
    
    Args:
        image: Preprocessed grayscale image.
        threshold_method: Thresholding method ("isodata", "otsu", or "adaptive").
        
    Returns:
        Surface area in square microns.
    """
    # 1.13x1.13 is the pixel size in microns
    pixel_area = 1.13 * 1.13
    
    if threshold_method == "adaptive":
        # Adaptive thresholding returns a binary mask directly
        binary_mask = get_adaptive_threshold_mask(image)
        foreground_pixels = np.sum(binary_mask == 255)
    elif threshold_method == "otsu":
        threshold = get_otsu_threshold_value(image)
        foreground_pixels = np.sum(image > threshold)
    elif threshold_method == "isodata":
        threshold = get_iterative_threshold_value(image)
        foreground_pixels = np.sum(image > threshold)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    return foreground_pixels * pixel_area


def normalize_labels(labels, min_val=None, max_val=None):
    """
    Normalize an array-like of labels to [0, 1].
    """
    labels = np.array(labels, dtype=np.float32)

    if min_val is None:
        min_val = float(labels.min())
    if max_val is None:
        max_val = float(labels.max())

    if max_val - min_val == 0:
        return np.zeros_like(labels), min_val, max_val

    return (labels - min_val) / (max_val - min_val), min_val, max_val






