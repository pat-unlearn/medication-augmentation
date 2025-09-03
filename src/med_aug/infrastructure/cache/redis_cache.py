"""Redis cache implementation."""

import json
import pickle
from typing import Any, Optional, Dict
import structlog
from .base import BaseCache, CacheConfig

logger = structlog.get_logger()

# Optional redis import
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("redis_not_available", message="Redis library not installed")


class RedisCache(BaseCache):
    """Redis cache implementation."""
    
    def __init__(self, config: CacheConfig, redis_url: str = "redis://localhost:6379"):
        """
        Initialize Redis cache.
        
        Args:
            config: Cache configuration
            redis_url: Redis connection URL
        """
        super().__init__(config)
        self.redis_url = redis_url
        self._client = None
        
        if not REDIS_AVAILABLE:
            logger.warning("redis_disabled", message="Redis not available, cache disabled")
            self.config.enabled = False
    
    async def connect(self):
        """Connect to Redis."""
        if REDIS_AVAILABLE and not self._client:
            try:
                self._client = await redis.from_url(self.redis_url)
                await self._client.ping()
                logger.info("redis_connected", url=self.redis_url)
            except Exception as e:
                logger.error("redis_connection_failed", error=str(e))
                self.config.enabled = False
    
    async def disconnect(self):
        """Disconnect from Redis."""
        if self._client:
            await self._client.close()
            self._client = None
    
    def _serialize(self, value: Any) -> bytes:
        """Serialize value for storage."""
        try:
            # Try JSON first for simple types
            return json.dumps(value).encode()
        except (TypeError, ValueError):
            # Fall back to pickle for complex types
            return pickle.dumps(value)
    
    def _deserialize(self, data: bytes) -> Any:
        """Deserialize value from storage."""
        try:
            # Try JSON first
            return json.loads(data.decode())
        except (json.JSONDecodeError, UnicodeDecodeError):
            # Fall back to pickle
            return pickle.loads(data)
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        if not self.config.enabled or not self._client:
            self._stats['misses'] += 1
            return None
        
        try:
            data = await self._client.get(key)
            if data is None:
                self._stats['misses'] += 1
                return None
            
            self._stats['hits'] += 1
            return self._deserialize(data)
            
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            self._stats['errors'] += 1
            return None
    
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
        if not self.config.enabled or not self._client:
            return False
        
        try:
            data = self._serialize(value)
            expire = ttl if ttl is not None else self.config.ttl_seconds
            
            if expire > 0:
                await self._client.setex(key, expire, data)
            else:
                await self._client.set(key, data)
            
            self._stats['sets'] += 1
            return True
            
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
            self._stats['errors'] += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Success status
        """
        if not self.config.enabled or not self._client:
            return False
        
        try:
            result = await self._client.delete(key)
            self._stats['deletes'] += 1
            return result > 0
            
        except Exception as e:
            logger.error("redis_delete_error", key=key, error=str(e))
            self._stats['errors'] += 1
            return False
    
    async def clear(self) -> bool:
        """
        Clear all cached values in namespace.
        
        Returns:
            Success status
        """
        if not self.config.enabled or not self._client:
            return False
        
        try:
            # Get all keys in namespace
            pattern = f"{self.config.namespace}:*"
            keys = []
            async for key in self._client.scan_iter(match=pattern):
                keys.append(key)
            
            # Delete all keys
            if keys:
                await self._client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error("redis_clear_error", error=str(e))
            self._stats['errors'] += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if exists
        """
        if not self.config.enabled or not self._client:
            return False
        
        try:
            result = await self._client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error("redis_exists_error", key=key, error=str(e))
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis cache information.
        
        Returns:
            Cache information dictionary
        """
        info = {
            'type': 'redis',
            'url': self.redis_url,
            'connected': self._client is not None,
            'stats': self.get_stats(),
            'config': {
                'ttl_seconds': self.config.ttl_seconds,
                'namespace': self.config.namespace,
                'enabled': self.config.enabled
            }
        }
        
        if self._client:
            try:
                # Get Redis server info
                server_info = await self._client.info()
                info['server'] = {
                    'version': server_info.get('redis_version', 'unknown'),
                    'used_memory': server_info.get('used_memory_human', 'unknown'),
                    'connected_clients': server_info.get('connected_clients', 0),
                    'uptime_days': server_info.get('uptime_in_days', 0)
                }
                
                # Count keys in namespace
                pattern = f"{self.config.namespace}:*"
                key_count = 0
                async for _ in self._client.scan_iter(match=pattern):
                    key_count += 1
                info['namespace_keys'] = key_count
                
            except Exception as e:
                logger.error("redis_info_error", error=str(e))
        
        return info