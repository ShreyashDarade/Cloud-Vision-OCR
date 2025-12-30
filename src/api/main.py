"""
FastAPI application setup with middleware and configuration.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
import uuid

from src.config import get_settings
from src.observability.logging import setup_logging, bind_request_context, get_logger
from src.observability.metrics import get_metrics
from src.errors import OCRError
from src.api.routes import ocr, health


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    setup_logging()
    logger = get_logger(__name__)
    logger.info("application_startup", version="1.0.0")
    
    yield
    
    # Shutdown
    logger.info("application_shutdown")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Cloud Vision OCR API",
        description="Production-grade OCR pipeline using Google Cloud Vision API",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Request logging middleware
    @app.middleware("http")
    async def logging_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        bind_request_context(request_id=request_id)
        
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        
        logger = get_logger(__name__)
        logger.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2)
        )
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = str(round(duration * 1000, 2))
        
        return response
    
    # Metrics middleware
    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next):
        metrics = get_metrics()
        metrics.active_requests.inc()
        
        try:
            response = await call_next(request)
            return response
        finally:
            metrics.active_requests.dec()
    
    # Global exception handler
    @app.exception_handler(OCRError)
    async def ocr_error_handler(request: Request, exc: OCRError):
        logger = get_logger(__name__)
        logger.error(
            "ocr_error",
            error=exc.message,
            error_type=type(exc).__name__,
            details=exc.details
        )
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": exc.message,
                "error_type": type(exc).__name__,
                "details": exc.details
            }
        )
    
    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger = get_logger(__name__)
        logger.exception("unhandled_exception", error=str(exc))
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "error_type": "InternalServerError"
            }
        )
    
    # Include routers
    app.include_router(ocr.router, prefix="/ocr", tags=["OCR"])
    app.include_router(health.router, tags=["Health"])
    
    return app


# Create default app instance
app = create_app()
