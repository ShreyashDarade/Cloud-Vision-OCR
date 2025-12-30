"""
OCR API routes.
"""

import time
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import PlainTextResponse
import uuid

from src.config import get_settings
from src.preprocessing import PreprocessingPipeline
from src.vision import VisionClient
from src.cache import get_cache
from src.observability.logging import get_logger
from src.observability.metrics import get_metrics
from src.api.schemas import OCRResponse, ErrorResponse, JobStatus, BatchOCRRequest
from src.errors import ImageTooLargeError, InvalidImageError

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

# In-memory job storage (use Redis in production)
jobs: dict = {}


@router.post(
    "/sync",
    response_model=OCRResponse,
    responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="Synchronous OCR",
    description="Process a single image synchronously and return OCR results immediately."
)
async def ocr_sync(
    file: UploadFile = File(..., description="Image file to process"),
    detection_type: str = Form("DOCUMENT_TEXT_DETECTION", description="TEXT_DETECTION or DOCUMENT_TEXT_DETECTION"),
    language_hints: Optional[str] = Form(None, description="Comma-separated language codes"),
    preprocessing_enabled: bool = Form(True, description="Enable image preprocessing"),
    output_format: str = Form("json", description="Output format: json, text, or hocr")
):
    """
    Process a single image and return OCR results.
    
    - **file**: Image file (PNG, JPEG, TIFF, BMP, GIF, WEBP)
    - **detection_type**: TEXT_DETECTION for simple text, DOCUMENT_TEXT_DETECTION for documents
    - **language_hints**: Optional language hints (e.g., "en,hi,mr")
    - **preprocessing_enabled**: Apply image preprocessing for better accuracy
    - **output_format**: Response format (json, text, hocr)
    """
    metrics = get_metrics()
    cache = get_cache()
    start_time = time.time()
    
    try:
        # Read and validate file
        content = await file.read()
        
        if len(content) > settings.max_image_size_bytes:
            raise ImageTooLargeError(len(content), settings.max_image_size_bytes)
        
        if not content:
            raise InvalidImageError("Empty file uploaded")
        
        # Check cache
        content_hash = cache.compute_hash(content)
        cached_result = cache.get(content_hash)
        
        if cached_result:
            metrics.record_cache_hit()
            logger.info("cache_hit", content_hash=content_hash[:16])
            
            processing_time = (time.time() - start_time) * 1000
            cached_result["cached"] = True
            cached_result["processing_time_ms"] = processing_time
            
            if output_format == "text":
                return PlainTextResponse(content=cached_result.get("text", ""))
            
            return OCRResponse(**cached_result)
        
        metrics.record_cache_miss()
        
        # Preprocessing
        preprocessed_content = content
        if preprocessing_enabled:
            try:
                pipeline = PreprocessingPipeline()
                preprocessed_content = pipeline.process_bytes(content)
                logger.info("preprocessing_complete")
            except Exception as e:
                logger.warning("preprocessing_failed", error=str(e))
                # Continue with original content
        
        # Parse language hints
        lang_hints = None
        if language_hints:
            lang_hints = [l.strip() for l in language_hints.split(",")]
        
        # Call Vision API
        with VisionClient() as client:
            result = client.detect_text(
                preprocessed_content,
                detection_type=detection_type,
                language_hints=lang_hints
            )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Build response
        response_data = {
            "success": True,
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "detection_type": result.detection_type,
            "word_count": result.word_count,
            "char_count": result.char_count,
            "pages": [p.to_dict() for p in result.pages],
            "processing_time_ms": processing_time,
            "cached": False
        }
        
        # Cache result
        cache.set(content_hash, response_data)
        
        # Record metrics
        metrics.record_result(result.char_count, result.word_count, result.confidence)
        metrics.record_request("sync", detection_type, "success", processing_time / 1000)
        
        logger.info(
            "ocr_complete",
            chars=result.char_count,
            words=result.word_count,
            confidence=round(result.confidence, 3),
            time_ms=round(processing_time, 2)
        )
        
        # Return based on format
        if output_format == "text":
            return PlainTextResponse(content=result.text)
        elif output_format == "hocr":
            return PlainTextResponse(content=result.to_hocr(), media_type="text/html")
        
        return OCRResponse(**response_data)
        
    except ImageTooLargeError as e:
        metrics.record_error("image_too_large")
        raise HTTPException(status_code=413, detail=str(e))
    except InvalidImageError as e:
        metrics.record_error("invalid_image")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        metrics.record_error("ocr_failed")
        logger.exception("ocr_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(e)}")


@router.post(
    "/async",
    response_model=JobStatus,
    summary="Async OCR (Submit Job)",
    description="Submit an OCR job for async processing. Returns a job ID to check status."
)
async def ocr_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    detection_type: str = Form("DOCUMENT_TEXT_DETECTION"),
    language_hints: Optional[str] = Form(None),
    preprocessing_enabled: bool = Form(True)
):
    """Submit an image for async OCR processing."""
    from datetime import datetime
    
    job_id = str(uuid.uuid4())
    content = await file.read()
    
    if len(content) > settings.max_image_size_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Image size exceeds maximum of {settings.max_image_size_mb}MB"
        )
    
    # Create job
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "progress": 0.0,
        "total_images": 1,
        "processed_images": 0,
        "result": None,
        "error": None
    }
    jobs[job_id] = job
    
    # Parse language hints
    lang_hints = None
    if language_hints:
        lang_hints = [l.strip() for l in language_hints.split(",")]
    
    # Add background task
    background_tasks.add_task(
        process_ocr_job,
        job_id,
        content,
        detection_type,
        lang_hints,
        preprocessing_enabled
    )
    
    return JobStatus(**job)


async def process_ocr_job(
    job_id: str,
    content: bytes,
    detection_type: str,
    language_hints: Optional[List[str]],
    preprocessing_enabled: bool
):
    """Background task to process OCR job."""
    from datetime import datetime
    
    job = jobs.get(job_id)
    if not job:
        return
    
    job["status"] = "processing"
    
    try:
        # Preprocessing
        preprocessed_content = content
        if preprocessing_enabled:
            try:
                pipeline = PreprocessingPipeline()
                preprocessed_content = pipeline.process_bytes(content)
            except Exception as e:
                logger.warning("preprocessing_failed", job_id=job_id, error=str(e))
        
        # Call Vision API
        with VisionClient() as client:
            result = client.detect_text(
                preprocessed_content,
                detection_type=detection_type,
                language_hints=language_hints
            )
        
        job["status"] = "completed"
        job["completed_at"] = datetime.utcnow()
        job["progress"] = 1.0
        job["processed_images"] = 1
        job["result"] = [{
            "success": True,
            "text": result.text,
            "confidence": result.confidence,
            "language": result.language,
            "detection_type": result.detection_type,
            "word_count": result.word_count,
            "char_count": result.char_count,
            "processing_time_ms": 0,
            "cached": False
        }]
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.exception("async_job_failed", job_id=job_id, error=str(e))


@router.get(
    "/jobs/{job_id}",
    response_model=JobStatus,
    summary="Get Job Status",
    description="Check the status of an async OCR job."
)
async def get_job_status(job_id: str):
    """Get the status of an async OCR job."""
    job = jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return JobStatus(**job)


@router.post(
    "/batch",
    response_model=JobStatus,
    summary="Batch OCR",
    description="Process multiple images in a batch. Returns a job ID."
)
async def ocr_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(..., description="Multiple image files"),
    detection_type: str = Form("DOCUMENT_TEXT_DETECTION"),
    language_hints: Optional[str] = Form(None),
    preprocessing_enabled: bool = Form(True)
):
    """Submit multiple images for batch OCR processing."""
    from datetime import datetime
    
    if len(files) > settings.max_batch_size:
        raise HTTPException(
            status_code=400,
            detail=f"Batch size exceeds maximum of {settings.max_batch_size}"
        )
    
    job_id = str(uuid.uuid4())
    
    # Read all files
    contents = []
    for file in files:
        content = await file.read()
        if len(content) > settings.max_image_size_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image {file.filename} exceeds maximum size"
            )
        contents.append(content)
    
    # Create job
    job = {
        "job_id": job_id,
        "status": "pending",
        "created_at": datetime.utcnow(),
        "completed_at": None,
        "progress": 0.0,
        "total_images": len(files),
        "processed_images": 0,
        "result": None,
        "error": None
    }
    jobs[job_id] = job
    
    # Parse language hints
    lang_hints = None
    if language_hints:
        lang_hints = [l.strip() for l in language_hints.split(",")]
    
    # Add background task
    background_tasks.add_task(
        process_batch_job,
        job_id,
        contents,
        detection_type,
        lang_hints,
        preprocessing_enabled
    )
    
    return JobStatus(**job)


async def process_batch_job(
    job_id: str,
    contents: List[bytes],
    detection_type: str,
    language_hints: Optional[List[str]],
    preprocessing_enabled: bool
):
    """Background task to process batch OCR job."""
    from datetime import datetime
    
    job = jobs.get(job_id)
    if not job:
        return
    
    job["status"] = "processing"
    results = []
    
    pipeline = PreprocessingPipeline() if preprocessing_enabled else None
    
    with VisionClient() as client:
        for i, content in enumerate(contents):
            try:
                # Preprocessing
                preprocessed = content
                if pipeline:
                    try:
                        preprocessed = pipeline.process_bytes(content)
                    except:
                        pass
                
                result = client.detect_text(
                    preprocessed,
                    detection_type=detection_type,
                    language_hints=language_hints
                )
                
                results.append({
                    "success": True,
                    "text": result.text,
                    "confidence": result.confidence,
                    "word_count": result.word_count,
                    "char_count": result.char_count,
                    "processing_time_ms": 0,
                    "cached": False
                })
                
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "text": "",
                    "confidence": 0
                })
            
            job["processed_images"] = i + 1
            job["progress"] = (i + 1) / len(contents)
    
    job["status"] = "completed"
    job["completed_at"] = datetime.utcnow()
    job["result"] = results
