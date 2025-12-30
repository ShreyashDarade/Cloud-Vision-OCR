"""
Pydantic schemas for API request/response validation.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# Request schemas
class OCRRequest(BaseModel):
    """Base OCR request settings."""
    detection_type: Literal["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"] = "DOCUMENT_TEXT_DETECTION"
    language_hints: Optional[List[str]] = None
    preprocessing_enabled: bool = True
    output_format: Literal["json", "text", "hocr"] = "json"


class BatchOCRRequest(BaseModel):
    """Batch OCR request."""
    detection_type: Literal["TEXT_DETECTION", "DOCUMENT_TEXT_DETECTION"] = "DOCUMENT_TEXT_DETECTION"
    language_hints: Optional[List[str]] = None
    preprocessing_enabled: bool = True


# Response schemas
class BoundingBoxResponse(BaseModel):
    """Bounding box coordinates."""
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    width: int
    height: int


class WordResponse(BaseModel):
    """Word in OCR result."""
    text: str
    confidence: float
    bounding_box: Optional[BoundingBoxResponse] = None


class LineResponse(BaseModel):
    """Line in OCR result."""
    text: str
    confidence: float
    words: List[WordResponse] = []


class BlockResponse(BaseModel):
    """Text block in OCR result."""
    text: str
    confidence: float
    block_type: str = "TEXT"
    lines: List[LineResponse] = []


class PageResponse(BaseModel):
    """Page in OCR result."""
    width: int
    height: int
    confidence: float
    blocks: List[BlockResponse] = []


class OCRResponse(BaseModel):
    """Full OCR response."""
    success: bool = True
    text: str
    confidence: float
    language: str = ""
    detection_type: str
    word_count: int
    char_count: int
    pages: List[PageResponse] = []
    processing_time_ms: float
    cached: bool = False
    error: Optional[str] = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "text": "Hello World",
                "confidence": 0.98,
                "language": "en",
                "detection_type": "DOCUMENT_TEXT_DETECTION",
                "word_count": 2,
                "char_count": 11,
                "pages": [],
                "processing_time_ms": 1234.56,
                "cached": False
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    error_type: str
    details: Optional[dict] = None


class JobStatus(BaseModel):
    """Async job status."""
    job_id: str
    status: Literal["pending", "processing", "completed", "failed"]
    created_at: datetime
    completed_at: Optional[datetime] = None
    progress: float = 0.0
    total_images: int = 0
    processed_images: int = 0
    result: Optional[List[OCRResponse]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: Literal["healthy", "degraded", "unhealthy"]
    version: str
    components: dict = Field(default_factory=dict)


class CacheStatsResponse(BaseModel):
    """Cache statistics."""
    enabled: bool
    redis_available: bool
    memory_cache_size: int
    memory_cache_max_size: int
    redis_used_memory: Optional[str] = None
