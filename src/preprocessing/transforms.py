"""
Image transformation functions for preprocessing.
"""

import cv2
import numpy as np
from PIL import Image
import structlog

from src.errors import PreprocessingError

logger = structlog.get_logger(__name__)


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    if image.mode == 'RGBA':
        image = image.convert('RGB')
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(image: np.ndarray) -> Image.Image:
    """Convert OpenCV image to PIL Image."""
    if len(image.shape) == 2:
        return Image.fromarray(image)
    return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))


def deskew(image: Image.Image, max_angle: float = 10.0) -> Image.Image:
    """
    Correct skew in the image using Hough transform.
    
    Args:
        image: PIL Image
        max_angle: Maximum angle to correct (degrees)
    
    Returns:
        Deskewed PIL Image
    """
    try:
        cv_image = pil_to_cv2(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Hough line detection
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
        
        if lines is None:
            logger.debug("deskew_no_lines_detected")
            return image
        
        # Calculate angles
        angles = []
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            if abs(angle) <= max_angle:
                angles.append(angle)
        
        if not angles:
            return image
        
        # Use median angle
        median_angle = np.median(angles)
        
        if abs(median_angle) < 0.5:
            return image
        
        # Rotate image
        h, w = gray.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated = cv2.warpAffine(
            cv_image, 
            rotation_matrix, 
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        
        logger.debug("deskew_applied", angle=median_angle)
        return cv2_to_pil(rotated)
        
    except Exception as e:
        logger.warning("deskew_failed", error=str(e))
        return image


def denoise(
    image: Image.Image, 
    method: str = "bilateral",
    strength: int = 10
) -> Image.Image:
    """
    Remove noise from image.
    
    Args:
        image: PIL Image
        method: 'bilateral', 'gaussian', 'median', or 'nlmeans'
        strength: Denoising strength (1-20)
    
    Returns:
        Denoised PIL Image
    """
    try:
        cv_image = pil_to_cv2(image)
        
        if method == "bilateral":
            denoised = cv2.bilateralFilter(cv_image, 9, strength * 7.5, strength * 7.5)
        elif method == "gaussian":
            ksize = max(3, (strength // 2) * 2 + 1)
            denoised = cv2.GaussianBlur(cv_image, (ksize, ksize), 0)
        elif method == "median":
            ksize = max(3, (strength // 2) * 2 + 1)
            denoised = cv2.medianBlur(cv_image, ksize)
        elif method == "nlmeans":
            if len(cv_image.shape) == 3:
                denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, strength, strength, 7, 21)
            else:
                denoised = cv2.fastNlMeansDenoising(cv_image, None, strength, 7, 21)
        else:
            raise PreprocessingError(f"Unknown denoising method: {method}")
        
        logger.debug("denoise_applied", method=method, strength=strength)
        return cv2_to_pil(denoised)
        
    except PreprocessingError:
        raise
    except Exception as e:
        logger.warning("denoise_failed", error=str(e))
        return image


def binarize(
    image: Image.Image,
    method: str = "otsu",
    block_size: int = 11,
    c: int = 2
) -> Image.Image:
    """
    Convert image to binary (black and white).
    
    Args:
        image: PIL Image
        method: 'otsu', 'adaptive_mean', or 'adaptive_gaussian'
        block_size: Block size for adaptive methods (odd number)
        c: Constant subtracted from mean for adaptive methods
    
    Returns:
        Binarized PIL Image
    """
    try:
        cv_image = pil_to_cv2(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        
        if method == "otsu":
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "adaptive_mean":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                cv2.THRESH_BINARY, block_size, c
            )
        elif method == "adaptive_gaussian":
            binary = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, c
            )
        else:
            raise PreprocessingError(f"Unknown binarization method: {method}")
        
        logger.debug("binarize_applied", method=method)
        return cv2_to_pil(binary)
        
    except PreprocessingError:
        raise
    except Exception as e:
        logger.warning("binarize_failed", error=str(e))
        return image


def enhance_contrast(
    image: Image.Image,
    clip_limit: float = 2.0,
    tile_grid_size: tuple = (8, 8)
) -> Image.Image:
    """
    Enhance image contrast using CLAHE.
    
    Args:
        image: PIL Image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
    
    Returns:
        Contrast-enhanced PIL Image
    """
    try:
        cv_image = pil_to_cv2(image)
        
        # Convert to LAB color space
        if len(cv_image.shape) == 3:
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            l_enhanced = clahe.apply(l)
            
            # Merge back
            enhanced_lab = cv2.merge([l_enhanced, a, b])
            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            enhanced = clahe.apply(cv_image)
        
        logger.debug("enhance_contrast_applied", clip_limit=clip_limit)
        return cv2_to_pil(enhanced)
        
    except Exception as e:
        logger.warning("enhance_contrast_failed", error=str(e))
        return image


def auto_crop(
    image: Image.Image,
    border_percent: float = 0.02,
    threshold: int = 250
) -> Image.Image:
    """
    Automatically crop whitespace/borders from image.
    
    Args:
        image: PIL Image
        border_percent: Minimum border to keep (as percent of dimension)
        threshold: Pixel value threshold for "white" (0-255)
    
    Returns:
        Cropped PIL Image
    """
    try:
        cv_image = pil_to_cv2(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY) if len(cv_image.shape) == 3 else cv_image
        
        # Threshold to find content
        _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Get bounding box of all contours
        all_points = np.concatenate(contours)
        x, y, w, h = cv2.boundingRect(all_points)
        
        # Add border
        height, width = gray.shape[:2]
        border_x = int(width * border_percent)
        border_y = int(height * border_percent)
        
        x = max(0, x - border_x)
        y = max(0, y - border_y)
        w = min(width - x, w + 2 * border_x)
        h = min(height - y, h + 2 * border_y)
        
        cropped = cv_image[y:y+h, x:x+w]
        
        logger.debug("auto_crop_applied", original_size=(width, height), cropped_size=(w, h))
        return cv2_to_pil(cropped)
        
    except Exception as e:
        logger.warning("auto_crop_failed", error=str(e))
        return image


def resize_for_ocr(
    image: Image.Image,
    target_dpi: int = 300,
    current_dpi: int = None,
    max_dimension: int = 4096,
    min_dimension: int = 640
) -> Image.Image:
    """
    Resize image to optimal size for OCR.
    
    Args:
        image: PIL Image
        target_dpi: Target DPI for OCR
        current_dpi: Current image DPI (estimated if not provided)
        max_dimension: Maximum dimension allowed
        min_dimension: Minimum dimension to ensure
    
    Returns:
        Resized PIL Image
    """
    try:
        width, height = image.size
        
        # Estimate current DPI if not provided
        if current_dpi is None:
            # Assume 72 DPI for most screen captures, 150 for low-res scans
            current_dpi = 72 if max(width, height) < 1000 else 150
        
        # Calculate scale factor
        scale = target_dpi / current_dpi
        
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Enforce max dimension
        max_dim = max(new_width, new_height)
        if max_dim > max_dimension:
            scale_down = max_dimension / max_dim
            new_width = int(new_width * scale_down)
            new_height = int(new_height * scale_down)
        
        # Enforce min dimension
        min_dim = min(new_width, new_height)
        if min_dim < min_dimension:
            scale_up = min_dimension / min_dim
            new_width = int(new_width * scale_up)
            new_height = int(new_height * scale_up)
        
        if new_width == width and new_height == height:
            return image
        
        # Use high-quality resampling
        resized = image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )
        
        logger.debug(
            "resize_applied",
            original_size=(width, height),
            new_size=(new_width, new_height)
        )
        return resized
        
    except Exception as e:
        logger.warning("resize_failed", error=str(e))
        return image


def sharpen(image: Image.Image, strength: float = 1.0) -> Image.Image:
    """
    Sharpen image to improve text definition.
    
    Args:
        image: PIL Image
        strength: Sharpening strength (0.5 - 2.0)
    
    Returns:
        Sharpened PIL Image
    """
    try:
        cv_image = pil_to_cv2(image)
        
        # Create sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ]) * strength
        
        sharpened = cv2.filter2D(cv_image, -1, kernel)
        
        logger.debug("sharpen_applied", strength=strength)
        return cv2_to_pil(sharpened)
        
    except Exception as e:
        logger.warning("sharpen_failed", error=str(e))
        return image
