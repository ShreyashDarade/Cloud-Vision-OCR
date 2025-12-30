"""Custom exception classes for the OCR pipeline."""


class OCRError(Exception):
    """Base exception for OCR pipeline errors."""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class PreprocessingError(OCRError):
    """Error during image preprocessing."""
    pass


class VisionAPIError(OCRError):
    """Error from Google Cloud Vision API."""
    
    def __init__(self, message: str, status_code: int = None, details: dict = None):
        super().__init__(message, details)
        self.status_code = status_code


class CacheError(OCRError):
    """Error in caching layer."""
    pass


class RateLimitExceededError(OCRError):
    """Rate limit exceeded."""
    pass


class InvalidImageError(OCRError):
    """Invalid or corrupted image."""
    pass


class ImageTooLargeError(OCRError):
    """Image exceeds size limit."""
    
    def __init__(self, size_bytes: int, max_bytes: int):
        super().__init__(
            f"Image size {size_bytes} bytes exceeds maximum {max_bytes} bytes",
            {"size_bytes": size_bytes, "max_bytes": max_bytes}
        )
        self.size_bytes = size_bytes
        self.max_bytes = max_bytes


class UnsupportedFormatError(OCRError):
    """Unsupported image format."""
    
    def __init__(self, format: str, supported_formats: list):
        super().__init__(
            f"Format '{format}' not supported. Supported: {supported_formats}",
            {"format": format, "supported_formats": supported_formats}
        )
