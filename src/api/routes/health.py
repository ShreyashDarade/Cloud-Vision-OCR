"""
Health and metrics endpoints.
"""

from fastapi import APIRouter
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import get_settings
from src.cache import get_cache
from src.api.schemas import HealthResponse, CacheStatsResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Basic health check endpoint."
)
async def health_check():
    """Basic health check."""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        components={}
    )


@router.get(
    "/health/ready",
    response_model=HealthResponse,
    summary="Readiness Check",
    description="Check if the service is ready to handle requests (all dependencies available)."
)
async def readiness_check():
    """Readiness check with dependency status."""
    components = {}
    overall_status = "healthy"
    
    # Check cache
    try:
        cache = get_cache()
        cache_stats = cache.get_stats()
        components["cache"] = {
            "status": "healthy" if cache_stats["enabled"] else "disabled",
            "redis_available": cache_stats.get("redis_available", False)
        }
        if cache_stats["enabled"] and not cache_stats.get("redis_available"):
            components["cache"]["status"] = "degraded"
            overall_status = "degraded"
    except Exception as e:
        components["cache"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "degraded"
    
    # Check Vision API (lightweight check - just verify client can be created)
    try:
        from src.vision import VisionClient
        client = VisionClient()
        components["vision_api"] = {"status": "healthy"}
    except Exception as e:
        components["vision_api"] = {"status": "unhealthy", "error": str(e)}
        overall_status = "unhealthy"
    
    return HealthResponse(
        status=overall_status,
        version="1.0.0",
        components=components
    )


@router.get(
    "/health/live",
    summary="Liveness Check",
    description="Simple liveness probe."
)
async def liveness_check():
    """Simple liveness probe."""
    return {"status": "alive"}


@router.get(
    "/metrics",
    summary="Prometheus Metrics",
    description="Prometheus-format metrics for monitoring."
)
async def metrics():
    """Prometheus metrics endpoint."""
    return PlainTextResponse(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@router.get(
    "/cache/stats",
    response_model=CacheStatsResponse,
    summary="Cache Statistics",
    description="Get cache statistics."
)
async def cache_stats():
    """Get cache statistics."""
    cache = get_cache()
    stats = cache.get_stats()
    return CacheStatsResponse(**stats)


@router.delete(
    "/cache",
    summary="Clear Cache",
    description="Clear all cached OCR results."
)
async def clear_cache():
    """Clear the OCR cache."""
    cache = get_cache()
    cache.clear()
    return {"message": "Cache cleared"}
