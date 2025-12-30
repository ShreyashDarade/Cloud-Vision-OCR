"""
Google Cloud Vision API client wrapper with retry logic and rate limiting.
"""

import io
import time
from typing import Optional, Union, List
from pathlib import Path
import structlog

from google.cloud import vision
from google.cloud.vision_v1 import types
from google.api_core import exceptions as google_exceptions
from google.oauth2 import service_account
import json

from src.config import get_settings
from src.errors import VisionAPIError, RateLimitExceededError
from src.errors.retry import retry_with_backoff, CircuitBreaker
from src.vision.response_parser import OCRResult, parse_vision_response

logger = structlog.get_logger(__name__)


class RateLimiter:
    """Simple token bucket rate limiter."""
    
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests = []
    
    def acquire(self) -> bool:
        """Try to acquire a request slot. Returns True if allowed."""
        now = time.time()
        
        # Remove old requests outside the window
        self.requests = [t for t in self.requests if now - t < self.window_seconds]
        
        if len(self.requests) >= self.max_requests:
            return False
        
        self.requests.append(now)
        return True
    
    def wait_time(self) -> float:
        """Time to wait before next request is allowed."""
        if len(self.requests) < self.max_requests:
            return 0
        
        oldest = min(self.requests)
        return max(0, self.window_seconds - (time.time() - oldest))


class VisionClient:
    """
    Production-ready Google Cloud Vision API client.
    
    Features:
    - Automatic retry with exponential backoff
    - Rate limiting
    - Circuit breaker for cascading failure prevention
    - Support for both TEXT_DETECTION and DOCUMENT_TEXT_DETECTION
    - GCS URI support for optimized processing
    """
    
    def __init__(self, config=None):
        """
        Initialize Vision client.
        
        Args:
            config: Settings instance (uses default if not provided)
        """
        self.settings = config or get_settings()
        self._client = None
        
        # Rate limiter
        if self.settings.rate_limit.enabled:
            self.rate_limiter = RateLimiter(
                self.settings.rate_limit.requests,
                self.settings.rate_limit.window_seconds
            )
        else:
            self.rate_limiter = None
        
        # Circuit breaker
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=60.0
        )
    
    @property
    def client(self) -> vision.ImageAnnotatorClient:
        """Lazy-load Vision API client."""
        if self._client is None:
            if self.settings.google_credentials_json:
                try:
                    # Parse JSON credentials from string
                    creds_dict = json.loads(self.settings.google_credentials_json)
                    credentials = service_account.Credentials.from_service_account_info(creds_dict)
                    self._client = vision.ImageAnnotatorClient(credentials=credentials)
                    logger.info("vision_client_initialized_from_json_env")
                except Exception as e:
                    logger.error("failed_to_load_json_credentials", error=str(e))
                    raise VisionAPIError(f"Failed to load credentials from JSON env: {e}")
            else:
                # Fallback to file path or default credentials
                self._client = vision.ImageAnnotatorClient()
                logger.info("vision_client_initialized_default")
        return self._client
    
    def _check_rate_limit(self):
        """Check and wait for rate limit if needed."""
        if self.rate_limiter is None:
            return
        
        if not self.rate_limiter.acquire():
            wait_time = self.rate_limiter.wait_time()
            if wait_time > 0:
                logger.warning("rate_limit_waiting", wait_seconds=wait_time)
                time.sleep(wait_time)
                if not self.rate_limiter.acquire():
                    raise RateLimitExceededError("Rate limit exceeded")
    
    def _check_circuit_breaker(self):
        """Check circuit breaker state."""
        if not self.circuit_breaker.can_execute():
            raise VisionAPIError(
                "Circuit breaker is open - service temporarily unavailable",
                status_code=503
            )
    
    @retry_with_backoff(
        exceptions=(
            google_exceptions.ServiceUnavailable,
            google_exceptions.DeadlineExceeded,
            google_exceptions.ResourceExhausted,
        )
    )
    def _call_api(
        self,
        image: types.Image,
        features: List[types.Feature],
        image_context: Optional[types.ImageContext] = None
    ) -> types.AnnotateImageResponse:
        """
        Make API call with retry logic.
        
        Args:
            image: Vision API Image object
            features: List of features to detect
            image_context: Optional image context for hints
        
        Returns:
            API response
        """
        self._check_rate_limit()
        self._check_circuit_breaker()
        
        try:
            request = types.AnnotateImageRequest(
                image=image,
                features=features,
                image_context=image_context
            )
            
            logger.info("calling_vision_api", image_size=len(image.content) if image.content else 0)
            response = self.client.annotate_image(request)
            logger.info("vision_api_call_complete")
            
            if response.error.message:
                logger.error("vision_api_error_in_response", error=response.error.message, code=response.error.code)
                raise VisionAPIError(
                    response.error.message,
                    status_code=response.error.code
                )
            
            self.circuit_breaker.record_success()
            return response
            
        except google_exceptions.PermissionDenied as e:
            logger.error("vision_api_permission_denied", error=str(e))
            self.circuit_breaker.record_failure()
            raise VisionAPIError(
                f"Permission denied. Please enable Vision API for project and ensure service account has correct permissions. Error: {e}",
                status_code=403
            )
        except google_exceptions.NotFound as e:
            logger.error("vision_api_not_found", error=str(e))
            self.circuit_breaker.record_failure()
            raise VisionAPIError(
                f"Vision API not found. Please enable it at https://console.cloud.google.com/apis/library/vision.googleapis.com. Error: {e}",
                status_code=404
            )
        except google_exceptions.GoogleAPIError as e:
            logger.error("vision_api_google_error", error=str(e), error_type=type(e).__name__)
            self.circuit_breaker.record_failure()
            raise VisionAPIError(str(e), status_code=getattr(e, 'code', None))
        except Exception as e:
            logger.error("vision_api_unexpected_error", error=str(e), error_type=type(e).__name__)
            self.circuit_breaker.record_failure()
            raise VisionAPIError(f"Unexpected error: {type(e).__name__}: {e}")
    
    def detect_text(
        self,
        image_source: Union[bytes, str, Path],
        detection_type: str = None,
        language_hints: List[str] = None
    ) -> OCRResult:
        """
        Detect text in an image.
        
        Args:
            image_source: Image bytes, file path, or GCS URI
            detection_type: 'TEXT_DETECTION' or 'DOCUMENT_TEXT_DETECTION'
            language_hints: Optional list of language codes to prioritize
        
        Returns:
            OCRResult with extracted text and metadata
        """
        detection_type = detection_type or self.settings.default_detection_type
        
        # Build image object
        image = types.Image()
        
        if isinstance(image_source, bytes):
            image.content = image_source
            source_type = "bytes"
        elif isinstance(image_source, (str, Path)):
            source_str = str(image_source)
            if source_str.startswith("gs://"):
                image.source = types.ImageSource(gcs_image_uri=source_str)
                source_type = "gcs"
            else:
                with open(source_str, "rb") as f:
                    image.content = f.read()
                source_type = "file"
        else:
            raise VisionAPIError(f"Unsupported image source type: {type(image_source)}")
        
        # Build feature
        feature_type = (
            vision.Feature.Type.DOCUMENT_TEXT_DETECTION
            if detection_type == "DOCUMENT_TEXT_DETECTION"
            else vision.Feature.Type.TEXT_DETECTION
        )
        features = [types.Feature(type=feature_type)]
        
        # Build image context
        image_context = None
        if language_hints:
            image_context = types.ImageContext(language_hints=language_hints)
        
        logger.info(
            "vision_api_request",
            source_type=source_type,
            detection_type=detection_type,
            language_hints=language_hints
        )
        
        start_time = time.time()
        response = self._call_api(image, features, image_context)
        elapsed = time.time() - start_time
        
        result = parse_vision_response(response, detection_type)
        
        logger.info(
            "vision_api_response",
            elapsed_seconds=round(elapsed, 3),
            text_length=len(result.text),
            confidence=result.confidence
        )
        
        return result
    
    def detect_text_batch(
        self,
        image_sources: List[Union[bytes, str, Path]],
        detection_type: str = None,
        language_hints: List[str] = None
    ) -> List[OCRResult]:
        """
        Detect text in multiple images.
        
        Args:
            image_sources: List of image sources
            detection_type: Detection type for all images
            language_hints: Language hints for all images
        
        Returns:
            List of OCRResult objects
        """
        results = []
        for i, source in enumerate(image_sources):
            try:
                result = self.detect_text(source, detection_type, language_hints)
                results.append(result)
            except Exception as e:
                logger.error("batch_item_failed", index=i, error=str(e))
                # Return error result
                results.append(OCRResult(
                    text="",
                    confidence=0.0,
                    error=str(e)
                ))
        return results
    
    def close(self):
        """Close the client connection."""
        if self._client:
            self._client = None
            logger.info("vision_client_closed")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
