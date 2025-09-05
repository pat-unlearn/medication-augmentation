"""Base cache abstraction for web scrapers."""

import hashlib
import json
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass
from datetime import datetime
import structlog

logger = structlog.get_logger()


@dataclass
class CacheConfig:
    """Configuration for cache systems."""

    ttl_seconds: int = 3600  # 1 hour default
    max_size: Optional[int] = None  # Max items in cache
    namespace: str = "med_aug"
    enabled: bool = True


class BaseCache(ABC):
    """Abstract base class for cache implementations."""

    def __init__(self, config: CacheConfig):
        """
        Initialize cache.

        Args:
            config: Cache configuration
        """
        self.config = config
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "errors": 0}

    def make_key(self, prefix: str, *args, **kwargs) -> str:
        """
        Create a cache key from arguments.

        Args:
            prefix: Key prefix
            *args: Positional arguments to include
            **kwargs: Keyword arguments to include

        Returns:
            Cache key string
        """
        # Combine all arguments
        key_data = {"prefix": prefix, "args": args, "kwargs": kwargs}

        # Create hash for complex data
        key_json = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.md5(key_json.encode()).hexdigest()[:8]

        # Build key
        return f"{self.config.namespace}:{prefix}:{key_hash}"

    async def get_or_fetch(self, key: str, fetch_func, ttl: Optional[int] = None):
        """
        Get from cache or fetch and cache.

        Args:
            key: Cache key
            fetch_func: Async function to fetch data if not cached
            ttl: Optional TTL override

        Returns:
            Cached or fetched data
        """
        if not self.config.enabled:
            return await fetch_func()

        # Try to get from cache
        cached = await self.get(key)
        if cached is not None:
            logger.debug("cache_hit", key=key)
            return cached

        # Fetch new data
        logger.debug("cache_miss", key=key)
        data = await fetch_func()

        # Cache the result
        await self.set(key, data, ttl or self.config.ttl_seconds)

        return data

    @abstractmethod
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        pass

    @abstractmethod
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def clear(self) -> bool:
        """
        Clear all cached values.

        Returns:
            Success status
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists
        """
        pass

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Statistics dictionary
        """
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = self._stats["hits"] / total_requests if total_requests > 0 else 0

        return {**self._stats, "total_requests": total_requests, "hit_rate": hit_rate}

    def reset_stats(self):
        """Reset cache statistics."""
        self._stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "errors": 0}


class CacheEntry:
    """Entry in the cache with metadata."""

    def __init__(self, value: Any, ttl: int):
        """
        Initialize cache entry.

        Args:
            value: Cached value
            ttl: Time to live in seconds
        """
        self.value = value
        self.created_at = datetime.now()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = self.created_at

    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl <= 0:
            return False
        age = (datetime.now() - self.created_at).total_seconds()
        return age > self.ttl

    def touch(self):
        """Update access metadata."""
        self.access_count += 1
        self.last_accessed = datetime.now()
