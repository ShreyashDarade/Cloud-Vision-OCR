"""Observability module - logging and metrics."""

from src.observability.logging import setup_logging, get_logger
from src.observability.metrics import get_metrics, OCRMetrics

__all__ = ["setup_logging", "get_logger", "get_metrics", "OCRMetrics"]
