"""Caching infrastructure for web scrapers."""

from .base import BaseCache, CacheConfig
from .memory_cache import MemoryCache

__all__ = [
    "BaseCache",
    "CacheConfig",
    "MemoryCache",
]
