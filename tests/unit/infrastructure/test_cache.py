"""Unit tests for cache implementations."""

import pytest
import asyncio
from med_aug.infrastructure.cache.base import CacheConfig, CacheEntry
from med_aug.infrastructure.cache.memory_cache import MemoryCache


class TestMemoryCache:
    """Test MemoryCache functionality."""

    async def _create_cache(self):
        """Create cache instance."""
        config = CacheConfig(ttl_seconds=1, max_size=3)
        cache = MemoryCache(config)
        await cache.start()
        return cache

    @pytest.mark.asyncio
    async def test_basic_get_set(self):
        """Test basic get and set operations."""
        cache = await self._create_cache()

        # Set value
        result = await cache.set("key1", "value1")
        assert result is True

        # Get value
        value = await cache.get("key1")
        assert value == "value1"

        # Get non-existent key
        value = await cache.get("nonexistent")
        assert value is None

        await cache.stop()

    @pytest.mark.asyncio
    async def test_ttl_expiration(self):
        """Test TTL expiration."""
        cache = await self._create_cache()

        # Set with short TTL
        await cache.set("key1", "value1", ttl=1)

        # Should exist immediately
        assert await cache.exists("key1") is True
        value = await cache.get("key1")
        assert value == "value1"

        # Wait for expiration
        await asyncio.sleep(1.1)

        # Should be expired
        assert await cache.exists("key1") is False
        value = await cache.get("key1")
        assert value is None

        await cache.stop()

    @pytest.mark.asyncio
    async def test_max_size_eviction(self):
        """Test LRU eviction when max size is reached."""
        cache = await self._create_cache()

        # Fill cache to max size
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        assert cache.get_size() == 3

        # Add one more - should evict key1 (oldest)
        await cache.set("key4", "value4")

        assert cache.get_size() == 3
        assert await cache.get("key1") is None  # Evicted
        assert await cache.get("key2") == "value2"
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

        await cache.stop()

    @pytest.mark.asyncio
    async def test_lru_ordering(self):
        """Test LRU ordering with access updates."""
        cache = await self._create_cache()

        # Fill cache
        await cache.set("key1", "value1")
        await cache.set("key2", "value2")
        await cache.set("key3", "value3")

        # Access key1 to make it most recently used
        await cache.get("key1")

        # Add new key - should evict key2 (now oldest)
        await cache.set("key4", "value4")

        assert await cache.get("key1") == "value1"  # Still there
        assert await cache.get("key2") is None  # Evicted
        assert await cache.get("key3") == "value3"
        assert await cache.get("key4") == "value4"

        await cache.stop()

    @pytest.mark.asyncio
    async def test_delete(self):
        """Test delete operation."""
        cache = await self._create_cache()

        await cache.set("key1", "value1")
        assert await cache.exists("key1") is True

        # Delete existing key
        result = await cache.delete("key1")
        assert result is True
        assert await cache.exists("key1") is False

        # Delete non-existent key
        result = await cache.delete("nonexistent")
        assert result is False

        await cache.stop()

    @pytest.mark.asyncio
    async def test_clear(self):
        """Test clear operation."""
        cache = await self._create_cache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        assert cache.get_size() == 2

        result = await cache.clear()
        assert result is True
        assert cache.get_size() == 0
        assert await cache.get("key1") is None
        assert await cache.get("key2") is None

        await cache.stop()

    @pytest.mark.asyncio
    async def test_stats_tracking(self):
        """Test statistics tracking."""
        cache = await self._create_cache()

        # Reset stats
        cache.reset_stats()

        # Generate some activity
        await cache.set("key1", "value1")
        await cache.get("key1")  # Hit
        await cache.get("key2")  # Miss
        await cache.get("key3")  # Miss
        await cache.delete("key1")

        stats = cache.get_stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 2
        assert stats["sets"] == 1
        assert stats["deletes"] == 1
        assert stats["hit_rate"] == 1 / 3

        await cache.stop()

    @pytest.mark.asyncio
    async def test_cache_info(self):
        """Test cache info retrieval."""
        cache = await self._create_cache()

        await cache.set("key1", "value1")
        await cache.set("key2", "value2")

        info = await cache.get_info()
        assert info["type"] == "memory"
        assert info["size"] == 2
        assert info["max_size"] == 3
        assert "stats" in info
        assert "config" in info

        await cache.stop()

    @pytest.mark.asyncio
    async def test_get_or_fetch(self):
        """Test get_or_fetch functionality."""
        cache = await self._create_cache()

        fetch_count = 0

        async def fetch_func():
            nonlocal fetch_count
            fetch_count += 1
            return f"fetched_value_{fetch_count}"

        # First call - should fetch
        key = cache.make_key("test", "arg1", kwarg="val1")
        value = await cache.get_or_fetch(key, fetch_func, ttl=10)
        assert value == "fetched_value_1"
        assert fetch_count == 1

        # Second call - should use cache
        value = await cache.get_or_fetch(key, fetch_func, ttl=10)
        assert value == "fetched_value_1"
        assert fetch_count == 1  # Not incremented

        await cache.stop()

    @pytest.mark.asyncio
    async def test_make_key(self):
        """Test cache key generation."""
        cache = await self._create_cache()

        # Same arguments should produce same key
        key1 = cache.make_key("prefix", "arg1", "arg2", kwarg1="val1")
        key2 = cache.make_key("prefix", "arg1", "arg2", kwarg1="val1")
        assert key1 == key2

        # Different arguments should produce different keys
        key3 = cache.make_key("prefix", "arg1", "arg3", kwarg1="val1")
        assert key1 != key3

        # Key should include namespace
        assert cache.config.namespace in key1

        await cache.stop()


class TestCacheEntry:
    """Test CacheEntry functionality."""

    def test_entry_creation(self):
        """Test cache entry creation."""
        entry = CacheEntry("value", ttl=60)
        assert entry.value == "value"
        assert entry.ttl == 60
        assert entry.access_count == 0
        assert entry.created_at == entry.last_accessed

    def test_entry_expiration(self):
        """Test entry expiration check."""
        # Non-expiring entry (ttl=0)
        entry1 = CacheEntry("value", ttl=0)
        assert entry1.is_expired() is False

        # Very short TTL
        entry2 = CacheEntry("value", ttl=0.001)
        asyncio.run(asyncio.sleep(0.01))
        assert entry2.is_expired() is True

        # Long TTL
        entry3 = CacheEntry("value", ttl=3600)
        assert entry3.is_expired() is False

    def test_entry_touch(self):
        """Test touch updates metadata."""
        entry = CacheEntry("value", ttl=60)
        original_count = entry.access_count
        original_time = entry.last_accessed

        # Small delay to ensure time difference
        asyncio.run(asyncio.sleep(0.01))

        entry.touch()
        assert entry.access_count == original_count + 1
        assert entry.last_accessed > original_time
        assert entry.created_at == original_time  # Created time unchanged
