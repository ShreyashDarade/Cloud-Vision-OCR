"""
Prometheus metrics for monitoring.
"""

from functools import lru_cache
from prometheus_client import Counter, Histogram, Gauge, Info


class OCRMetrics:
    """OCR pipeline metrics."""
    
    def __init__(self):
        # Request metrics
        self.requests_total = Counter(
            "ocr_requests_total",
            "Total OCR requests",
            ["endpoint", "detection_type", "status"]
        )
        
        self.request_duration = Histogram(
            "ocr_request_duration_seconds",
            "OCR request duration",
            ["endpoint", "detection_type"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
        )
        
        # Processing metrics
        self.preprocessing_duration = Histogram(
            "ocr_preprocessing_duration_seconds",
            "Image preprocessing duration",
            buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
        )
        
        self.vision_api_duration = Histogram(
            "ocr_vision_api_duration_seconds",
            "Google Vision API call duration",
            ["detection_type"],
            buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
        )
        
        # Result metrics
        self.characters_extracted = Counter(
            "ocr_characters_extracted_total",
            "Total characters extracted"
        )
        
        self.words_extracted = Counter(
            "ocr_words_extracted_total", 
            "Total words extracted"
        )
        
        self.confidence_score = Histogram(
            "ocr_confidence_score",
            "OCR confidence scores",
            buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)
        )
        
        # Cache metrics
        self.cache_hits = Counter(
            "ocr_cache_hits_total",
            "Cache hits"
        )
        
        self.cache_misses = Counter(
            "ocr_cache_misses_total",
            "Cache misses"
        )
        
        # Error metrics
        self.errors_total = Counter(
            "ocr_errors_total",
            "Total errors",
            ["error_type"]
        )
        
        # System metrics
        self.active_requests = Gauge(
            "ocr_active_requests",
            "Currently active requests"
        )
        
        self.queue_size = Gauge(
            "ocr_queue_size",
            "Async job queue size"
        )
        
        # Info
        self.info = Info(
            "ocr_pipeline",
            "OCR pipeline information"
        )
        self.info.info({
            "version": "1.0.0",
            "vision_api": "google_cloud"
        })
    
    def record_request(
        self,
        endpoint: str,
        detection_type: str,
        status: str,
        duration: float
    ):
        """Record a completed request."""
        self.requests_total.labels(
            endpoint=endpoint,
            detection_type=detection_type,
            status=status
        ).inc()
        
        self.request_duration.labels(
            endpoint=endpoint,
            detection_type=detection_type
        ).observe(duration)
    
    def record_result(self, char_count: int, word_count: int, confidence: float):
        """Record OCR result metrics."""
        self.characters_extracted.inc(char_count)
        self.words_extracted.inc(word_count)
        self.confidence_score.observe(confidence)
    
    def record_error(self, error_type: str):
        """Record an error."""
        self.errors_total.labels(error_type=error_type).inc()
    
    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits.inc()
    
    def record_cache_miss(self):
        """Record a cache miss."""
        self.cache_misses.inc()


@lru_cache()
def get_metrics() -> OCRMetrics:
    """Get singleton metrics instance."""
    return OCRMetrics()
