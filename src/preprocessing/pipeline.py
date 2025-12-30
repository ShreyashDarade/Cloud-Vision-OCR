"""
Preprocessing pipeline orchestrator.
"""

from typing import Callable, List, Optional
from PIL import Image
import io
import structlog

from src.config import get_settings
from src.errors import PreprocessingError, InvalidImageError
from src.preprocessing.transforms import (
    deskew,
    denoise,
    binarize,
    enhance_contrast,
    auto_crop,
    resize_for_ocr,
    sharpen,
)

logger = structlog.get_logger(__name__)


class PreprocessingPipeline:
    """
    Configurable image preprocessing pipeline.
    
    Chains multiple preprocessing steps dynamically based on configuration.
    """
    
    SUPPORTED_FORMATS = {"PNG", "JPEG", "JPG", "TIFF", "TIF", "BMP", "GIF", "WEBP"}
    
    def __init__(self, config=None):
        """
        Initialize pipeline with configuration.
        
        Args:
            config: PreprocessingSettings instance (uses default if not provided)
        """
        self.settings = config or get_settings().preprocessing
        self._steps: List[tuple[str, Callable, dict]] = []
        self._build_pipeline()
    
    def _build_pipeline(self):
        """Build preprocessing steps based on configuration."""
        self._steps = []
        
        # Always resize first for consistent processing
        self._steps.append((
            "resize",
            resize_for_ocr,
            {
                "target_dpi": self.settings.target_dpi,
                "max_dimension": self.settings.max_dimension
            }
        ))
        
        if self.settings.deskew:
            self._steps.append(("deskew", deskew, {}))
        
        if self.settings.denoise:
            self._steps.append(("denoise", denoise, {"method": "bilateral", "strength": 10}))
        
        if self.settings.enhance_contrast:
            self._steps.append(("enhance_contrast", enhance_contrast, {}))
        
        if self.settings.auto_crop:
            self._steps.append(("auto_crop", auto_crop, {}))
        
        if self.settings.binarize:
            self._steps.append(("binarize", binarize, {"method": "adaptive_gaussian"}))
    
    def add_step(self, name: str, func: Callable, params: dict = None):
        """
        Add a custom preprocessing step.
        
        Args:
            name: Step name for logging
            func: Function that takes PIL Image and returns PIL Image
            params: Additional parameters to pass to the function
        """
        self._steps.append((name, func, params or {}))
    
    def remove_step(self, name: str):
        """Remove a preprocessing step by name."""
        self._steps = [(n, f, p) for n, f, p in self._steps if n != name]
    
    def process(self, image: Image.Image) -> Image.Image:
        """
        Process image through the pipeline.
        
        Args:
            image: PIL Image to process
        
        Returns:
            Processed PIL Image
        """
        if not self.settings.enabled:
            logger.debug("preprocessing_disabled")
            return image
        
        current = image
        
        for step_name, func, params in self._steps:
            try:
                logger.debug("preprocessing_step_start", step=step_name)
                current = func(current, **params)
                logger.debug("preprocessing_step_complete", step=step_name)
            except Exception as e:
                logger.warning(
                    "preprocessing_step_failed",
                    step=step_name,
                    error=str(e)
                )
                # Continue with current image on non-critical failures
        
        return current
    
    def process_bytes(self, image_bytes: bytes) -> bytes:
        """
        Process image bytes through the pipeline.
        
        Args:
            image_bytes: Raw image bytes
        
        Returns:
            Processed image as PNG bytes
        """
        try:
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            raise InvalidImageError(f"Failed to load image: {e}")
        
        # Validate format
        if image.format and image.format.upper() not in self.SUPPORTED_FORMATS:
            logger.warning("unsupported_format", format=image.format)
        
        # Convert to RGB if necessary
        if image.mode in ("RGBA", "P"):
            image = image.convert("RGB")
        elif image.mode == "L":
            pass  # Grayscale is fine
        elif image.mode != "RGB":
            image = image.convert("RGB")
        
        processed = self.process(image)
        
        # Convert back to bytes
        output_buffer = io.BytesIO()
        processed.save(output_buffer, format="PNG", optimize=True)
        return output_buffer.getvalue()
    
    @staticmethod
    def load_image(source) -> Image.Image:
        """
        Load image from various sources.
        
        Args:
            source: File path (str), bytes, or file-like object
        
        Returns:
            PIL Image
        """
        try:
            if isinstance(source, str):
                return Image.open(source)
            elif isinstance(source, bytes):
                return Image.open(io.BytesIO(source))
            elif hasattr(source, "read"):
                return Image.open(source)
            else:
                raise InvalidImageError(f"Unsupported source type: {type(source)}")
        except Exception as e:
            if isinstance(e, InvalidImageError):
                raise
            raise InvalidImageError(f"Failed to load image: {e}")
    
    @staticmethod
    def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
        """Convert PIL Image to bytes."""
        buffer = io.BytesIO()
        image.save(buffer, format=format)
        return buffer.getvalue()
