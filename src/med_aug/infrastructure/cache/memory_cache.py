"""In-memory cache implementation."""

import asyncio
from typing import Any, Optional, Dict
from collections import OrderedDict
import structlog
from .base import BaseCache, CacheConfig, CacheEntry

logger = structlog.get_logger()


class MemoryCache(BaseCache):
    """In-memory cache implementation with TTL and size limits."""

    def __init__(self, config: CacheConfig):
        """
        Initialize memory cache.

        Args:
            config: Cache configuration
        """
        super().__init__(config)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cleanup_task = None

    async def start(self):
        """Start background cleanup task."""
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _cleanup_loop(self):
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("cache_cleanup_error", error=str(e))

    async def _cleanup_expired(self):
        """Remove expired entries."""
        async with self._lock:
            expired_keys = []
            for key, entry in self._cache.items():
                if entry.is_expired():
                    expired_keys.append(key)

            for key in expired_keys:
                del self._cache[key]
                logger.debug("cache_expired", key=key)

    async def _evict_lru(self):
        """Evict least recently used entry if cache is full."""
        if self.config.max_size and len(self._cache) >= self.config.max_size:
            # OrderedDict maintains insertion order, so first item is oldest
            key, _ = self._cache.popitem(last=False)
            logger.debug("cache_evicted", key=key)

    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        async with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            if entry.is_expired():
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Move to end (most recently used)
            self._cache.move_to_end(key)
            entry.touch()

            self._stats["hits"] += 1
            return entry.value

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
        try:
            async with self._lock:
                # Evict if necessary
                if key not in self._cache:
                    await self._evict_lru()

                # Create entry
                entry_ttl = ttl if ttl is not None else self.config.ttl_seconds
                entry = CacheEntry(value, entry_ttl)

                # Store in cache
                self._cache[key] = entry
                self._cache.move_to_end(key)

                self._stats["sets"] += 1
                return True

        except Exception as e:
            logger.error("cache_set_error", key=key, error=str(e))
            self._stats["errors"] += 1
            return False

    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.

        Args:
            key: Cache key

        Returns:
            Success status
        """
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats["deletes"] += 1
                return True
            return False

    async def clear(self) -> bool:
        """
        Clear all cached values.

        Returns:
            Success status
        """
        async with self._lock:
            self._cache.clear()
            return True

    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if exists and not expired
        """
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                return False

            if entry.is_expired():
                del self._cache[key]
                return False

            return True

    def get_size(self) -> int:
        """Get current cache size."""
        return len(self._cache)

    def get_keys(self) -> list:
        """Get all cache keys."""
        return list(self._cache.keys())

    async def get_info(self) -> Dict[str, Any]:
        """
        Get detailed cache information.

        Returns:
            Cache information dictionary
        """
        async with self._lock:
            total_size = len(self._cache)
            expired_count = sum(
                1 for entry in self._cache.values() if entry.is_expired()
            )

            access_counts = [entry.access_count for entry in self._cache.values()]
            avg_access = sum(access_counts) / len(access_counts) if access_counts else 0

            return {
                "type": "memory",
                "size": total_size,
                "max_size": self.config.max_size,
                "expired_entries": expired_count,
                "average_access_count": avg_access,
                "stats": self.get_stats(),
                "config": {
                    "ttl_seconds": self.config.ttl_seconds,
                    "namespace": self.config.namespace,
                    "enabled": self.config.enabled,
                },
            }
