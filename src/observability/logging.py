"""
Structured logging configuration using structlog.
"""

import logging
import sys
from functools import lru_cache

import structlog
from structlog.stdlib import filter_by_level

from src.config import get_settings


def setup_logging(log_level: str = None, log_format: str = None):
    """
    Configure structured logging.
    
    Args:
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR)
        log_format: Override format ('json' or 'console')
    """
    settings = get_settings()
    level = log_level or settings.log_level
    fmt = log_format or settings.log_format
    
    # Configure stdlib logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level)
    )
    
    # Shared processors
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]
    
    if fmt == "json":
        # JSON format for production
        processors = shared_processors + [
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer()
        ]
    else:
        # Console format for development
        processors = shared_processors + [
            structlog.dev.ConsoleRenderer(colors=True)
        ]
    
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


@lru_cache()
def get_logger(name: str = None):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class RequestContextManager:
    """Context manager for request-scoped logging context."""
    
    def __init__(self, **context):
        self.context = context
    
    def __enter__(self):
        for key, value in self.context.items():
            structlog.contextvars.bind_contextvars(**{key: value})
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        structlog.contextvars.clear_contextvars()
        return False


def bind_request_context(request_id: str = None, **kwargs):
    """Bind context variables for the current request."""
    import uuid
    request_id = request_id or str(uuid.uuid4())[:8]
    structlog.contextvars.bind_contextvars(request_id=request_id, **kwargs)
    return request_id
