"""
Caching layer with content-hash based keys.
Supports Redis backend with in-memory LRU fallback.
"""

import hashlib
import json
from functools import lru_cache
from typing import Optional, Any
import structlog

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from cachetools import LRUCache

from src.config import get_settings
from src.errors import CacheError

logger = structlog.get_logger(__name__)


class CacheManager:
    """
    Content-hash based cache manager.
    
    Uses Redis as primary backend with in-memory LRU as fallback.
    """
    
    def __init__(self, config=None):
        """
        Initialize cache manager.
        
        Args:
            config: CacheSettings instance
        """
        self.settings = config or get_settings().cache
        self._redis_client = None
        self._memory_cache = LRUCache(maxsize=self.settings.max_size)
        self._redis_available = False
        
        if self.settings.enabled and REDIS_AVAILABLE:
            self._init_redis()
    
    def _init_redis(self):
        """Initialize Redis connection."""
        try:
            self._redis_client = redis.from_url(
                self.settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=5
            )
            # Test connection
            self._redis_client.ping()
            self._redis_available = True
            logger.info("cache_redis_connected", url=self.settings.redis_url)
        except Exception as e:
            logger.warning("cache_redis_unavailable", error=str(e))
            self._redis_available = False
    
    @staticmethod
    def compute_hash(data: bytes) -> str:
        """
        Compute content hash for cache key.
        
        Args:
            data: Binary data to hash
        
        Returns:
            SHA-256 hash string
        """
        return hashlib.sha256(data).hexdigest()
    
    def _make_key(self, content_hash: str, prefix: str = "ocr") -> str:
        """Create cache key from content hash."""
        return f"{prefix}:{content_hash}"
    
    def get(self, content_hash: str, prefix: str = "ocr") -> Optional[dict]:
        """
        Get cached result by content hash.
        
        Args:
            content_hash: SHA-256 hash of image content
            prefix: Key prefix for namespacing
        
        Returns:
            Cached result dict or None if not found
        """
        if not self.settings.enabled:
            return None
        
        key = self._make_key(content_hash, prefix)
        
        # Try memory cache first
        if key in self._memory_cache:
            logger.debug("cache_hit_memory", key=key)
            return self._memory_cache[key]
        
        # Try Redis
        if self._redis_available:
            try:
                data = self._redis_client.get(key)
                if data:
                    result = json.loads(data)
                    # Populate memory cache
                    self._memory_cache[key] = result
                    logger.debug("cache_hit_redis", key=key)
                    return result
            except Exception as e:
                logger.warning("cache_redis_get_error", key=key, error=str(e))
        
        logger.debug("cache_miss", key=key)
        return None
    
    def set(
        self, 
        content_hash: str, 
        result: dict, 
        prefix: str = "ocr",
        ttl: int = None
    ):
        """
        Cache a result.
        
        Args:
            content_hash: SHA-256 hash of image content
            result: Result dict to cache
            prefix: Key prefix for namespacing
            ttl: Time-to-live in seconds (uses default if not specified)
        """
        if not self.settings.enabled:
            return
        
        key = self._make_key(content_hash, prefix)
        ttl = ttl or self.settings.ttl_seconds
        
        # Always update memory cache
        self._memory_cache[key] = result
        
        # Update Redis
        if self._redis_available:
            try:
                self._redis_client.setex(key, ttl, json.dumps(result))
                logger.debug("cache_set", key=key, ttl=ttl)
            except Exception as e:
                logger.warning("cache_redis_set_error", key=key, error=str(e))
    
    def delete(self, content_hash: str, prefix: str = "ocr"):
        """Delete a cached result."""
        key = self._make_key(content_hash, prefix)
        
        # Remove from memory
        self._memory_cache.pop(key, None)
        
        # Remove from Redis
        if self._redis_available:
            try:
                self._redis_client.delete(key)
                logger.debug("cache_delete", key=key)
            except Exception as e:
                logger.warning("cache_redis_delete_error", key=key, error=str(e))
    
    def clear(self, prefix: str = "ocr"):
        """Clear all cached results with given prefix."""
        # Clear memory cache
        self._memory_cache.clear()
        
        # Clear Redis (pattern-based deletion)
        if self._redis_available:
            try:
                pattern = f"{prefix}:*"
                cursor = 0
                while True:
                    cursor, keys = self._redis_client.scan(cursor, match=pattern, count=100)
                    if keys:
                        self._redis_client.delete(*keys)
                    if cursor == 0:
                        break
                logger.info("cache_cleared", prefix=prefix)
            except Exception as e:
                logger.warning("cache_redis_clear_error", error=str(e))
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        stats = {
            "enabled": self.settings.enabled,
            "redis_available": self._redis_available,
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_max_size": self._memory_cache.maxsize
        }
        
        if self._redis_available:
            try:
                info = self._redis_client.info("memory")
                stats["redis_used_memory"] = info.get("used_memory_human")
            except:
                pass
        
        return stats


@lru_cache()
def get_cache() -> CacheManager:
    """Get cached CacheManager instance."""
    return CacheManager()
