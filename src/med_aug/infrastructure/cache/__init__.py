"""Caching infrastructure for web scrapers."""

from .base import BaseCache, CacheConfig
from .memory_cache import MemoryCache
from .redis_cache import RedisCache

__all__ = [
    'BaseCache',
    'CacheConfig',
    'MemoryCache',
    'RedisCache',
]