"""
Retry utilities with exponential backoff and circuit breaker pattern.
"""

import random
import time
from functools import wraps
from typing import Callable, Tuple, Type
import structlog

from src.config import get_settings

logger = structlog.get_logger(__name__)


class CircuitBreaker:
    """
    Circuit breaker pattern implementation.
    
    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Failing, requests are rejected immediately
    - HALF_OPEN: Testing, allowing limited requests through
    """
    
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        half_open_max_calls: int = 3
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        
        self.state = self.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.half_open_calls = 0
    
    def can_execute(self) -> bool:
        """Check if request can proceed."""
        if self.state == self.CLOSED:
            return True
        
        if self.state == self.OPEN:
            if time.time() - self.last_failure_time >= self.recovery_timeout:
                self.state = self.HALF_OPEN
                self.half_open_calls = 0
                logger.info("circuit_breaker_half_open")
                return True
            return False
        
        # HALF_OPEN
        if self.half_open_calls < self.half_open_max_calls:
            return True
        return False
    
    def record_success(self):
        """Record successful call."""
        if self.state == self.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.half_open_max_calls:
                self.state = self.CLOSED
                self.failure_count = 0
                logger.info("circuit_breaker_closed")
        elif self.state == self.CLOSED:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == self.HALF_OPEN:
            self.state = self.OPEN
            logger.warning("circuit_breaker_opened", reason="half_open_failure")
        elif self.failure_count >= self.failure_threshold:
            self.state = self.OPEN
            logger.warning("circuit_breaker_opened", failures=self.failure_count)


def retry_with_backoff(
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    max_attempts: int = None,
    initial_delay: float = None,
    max_delay: float = None,
    exponential_base: float = None,
    jitter: bool = None,
    on_retry: Callable = None
):
    """
    Decorator for retrying functions with exponential backoff.
    
    Args:
        exceptions: Tuple of exception types to retry on
        max_attempts: Maximum number of attempts
        initial_delay: Initial delay between retries in seconds
        max_delay: Maximum delay between retries
        exponential_base: Base for exponential backoff
        jitter: Whether to add random jitter
        on_retry: Callback function called on each retry
    """
    settings = get_settings()
    
    _max_attempts = max_attempts or settings.retry.max_attempts
    _initial_delay = initial_delay or settings.retry.initial_delay
    _max_delay = max_delay or settings.retry.max_delay
    _exponential_base = exponential_base or settings.retry.exponential_base
    _jitter = jitter if jitter is not None else settings.retry.jitter
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(1, _max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == _max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=attempt,
                            error=str(e)
                        )
                        raise
                    
                    # Calculate delay with exponential backoff
                    delay = min(
                        _initial_delay * (_exponential_base ** (attempt - 1)),
                        _max_delay
                    )
                    
                    # Add jitter
                    if _jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=_max_attempts,
                        delay=delay,
                        error=str(e)
                    )
                    
                    if on_retry:
                        on_retry(attempt, e, delay)
                    
                    time.sleep(delay)
            
            raise last_exception
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            import asyncio
            last_exception = None
            
            for attempt in range(1, _max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    
                    if attempt == _max_attempts:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=attempt,
                            error=str(e)
                        )
                        raise
                    
                    delay = min(
                        _initial_delay * (_exponential_base ** (attempt - 1)),
                        _max_delay
                    )
                    
                    if _jitter:
                        delay = delay * (0.5 + random.random())
                    
                    logger.warning(
                        "retry_attempt",
                        function=func.__name__,
                        attempt=attempt,
                        max_attempts=_max_attempts,
                        delay=delay,
                        error=str(e)
                    )
                    
                    if on_retry:
                        on_retry(attempt, e, delay)
                    
                    await asyncio.sleep(delay)
            
            raise last_exception
        
        # Return appropriate wrapper based on function type
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper
    
    return decorator
