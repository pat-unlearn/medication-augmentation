"""LLM service with caching and retry logic."""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import hashlib
import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import pickle

from ..core.logging import get_logger, PerformanceLogger
from .providers import LLMProvider, LLMResponse, LLMConfig, ProviderFactory
from .prompts import PromptManager

logger = get_logger(__name__)
perf_logger = PerformanceLogger(logger)


@dataclass
class CacheEntry:
    """Cache entry for LLM responses."""
    
    key: str
    response: LLMResponse
    timestamp: datetime
    prompt_hash: str
    hit_count: int = 0
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if cache entry is expired."""
        age = datetime.now() - self.timestamp
        return age.total_seconds() > ttl_seconds
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'key': self.key,
            'response': self.response.to_dict(),
            'timestamp': self.timestamp.isoformat(),
            'prompt_hash': self.prompt_hash,
            'hit_count': self.hit_count
        }


class ResponseCache:
    """Cache for LLM responses."""
    
    def __init__(
        self,
        cache_dir: Optional[Path] = None,
        ttl_seconds: int = 3600,
        max_entries: int = 1000
    ):
        """
        Initialize response cache.
        
        Args:
            cache_dir: Directory for persistent cache
            ttl_seconds: Time to live for cache entries
            max_entries: Maximum number of cache entries
        """
        self.cache_dir = cache_dir
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self.memory_cache: Dict[str, CacheEntry] = {}
        
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_cache()
    
    def _generate_key(self, prompt: str, system: Optional[str], **kwargs) -> str:
        """Generate cache key from prompt and parameters."""
        content = f"{system or ''}{prompt}{json.dumps(kwargs, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> Optional[LLMResponse]:
        """
        Get response from cache.
        
        Args:
            prompt: User prompt
            system: System message
            **kwargs: Additional parameters
            
        Returns:
            Cached response if available
        """
        key = self._generate_key(prompt, system, **kwargs)
        
        # Check memory cache
        if key in self.memory_cache:
            entry = self.memory_cache[key]
            if not entry.is_expired(self.ttl_seconds):
                entry.hit_count += 1
                logger.debug("cache_hit", key=key[:8], hits=entry.hit_count)
                return entry.response
            else:
                # Remove expired entry
                del self.memory_cache[key]
        
        return None
    
    def set(
        self,
        prompt: str,
        system: Optional[str],
        response: LLMResponse,
        **kwargs
    ):
        """
        Store response in cache.
        
        Args:
            prompt: User prompt
            system: System message
            response: LLM response
            **kwargs: Additional parameters
        """
        key = self._generate_key(prompt, system, **kwargs)
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            response=response,
            timestamp=datetime.now(),
            prompt_hash=hashlib.md5(prompt.encode()).hexdigest()
        )
        
        # Add to memory cache
        self.memory_cache[key] = entry
        
        # Evict old entries if needed
        if len(self.memory_cache) > self.max_entries:
            self._evict_oldest()
        
        # Save to disk if configured
        if self.cache_dir:
            self._save_entry(entry)
        
        logger.debug("cache_set", key=key[:8], cache_size=len(self.memory_cache))
    
    def _evict_oldest(self):
        """Evict oldest cache entries."""
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].timestamp
        )
        
        # Remove oldest 10% of entries
        to_remove = max(1, len(sorted_entries) // 10)
        for key, _ in sorted_entries[:to_remove]:
            del self.memory_cache[key]
    
    def _save_entry(self, entry: CacheEntry):
        """Save cache entry to disk."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / f"{entry.key}.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(entry, f)
        except Exception as e:
            logger.warning("cache_save_failed", error=str(e))
    
    def _load_cache(self):
        """Load cache from disk."""
        if not self.cache_dir or not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, 'rb') as f:
                    entry = pickle.load(f)
                    if not entry.is_expired(self.ttl_seconds):
                        self.memory_cache[entry.key] = entry
            except Exception as e:
                logger.warning("cache_load_failed", file=str(cache_file), error=str(e))
    
    def clear(self):
        """Clear all cache entries."""
        self.memory_cache.clear()
        
        if self.cache_dir and self.cache_dir.exists():
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_hits = sum(e.hit_count for e in self.memory_cache.values())
        return {
            'entries': len(self.memory_cache),
            'total_hits': total_hits,
            'ttl_seconds': self.ttl_seconds,
            'max_entries': self.max_entries
        }


class LLMService:
    """Service for managing LLM interactions."""
    
    def __init__(
        self,
        provider: Optional[LLMProvider] = None,
        config: Optional[LLMConfig] = None,
        enable_cache: bool = True,
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize LLM service.
        
        Args:
            provider: LLM provider instance
            config: LLM configuration
            enable_cache: Whether to enable response caching
            cache_dir: Directory for persistent cache
        """
        self.provider = provider or ProviderFactory.create('claude_cli', config)
        self.config = config or LLMConfig()
        self.prompt_manager = PromptManager()
        
        # Initialize cache
        self.cache = ResponseCache(cache_dir=cache_dir) if enable_cache else None
        
        # Statistics
        self.request_count = 0
        self.error_count = 0
        self.total_latency = 0.0
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        use_cache: bool = True,
        **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM with retry logic.
        
        Args:
            prompt: User prompt
            system: System message
            use_cache: Whether to use cache
            **kwargs: Additional parameters
            
        Returns:
            LLM response
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(prompt, system, **kwargs)
            if cached:
                logger.info("llm_cache_hit")
                return cached
        
        # Generate with retry logic
        logger.info("llm_generation_started", provider=type(self.provider).__name__)
        perf_logger.start_operation("llm_generation")
        
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                response = await self.provider.generate(prompt, system, **kwargs)
                
                # Cache successful response
                if use_cache and self.cache:
                    self.cache.set(prompt, system, response, **kwargs)
                
                # Update statistics
                self.request_count += 1
                latency = perf_logger.end_operation("llm_generation")
                self.total_latency += latency
                
                logger.info(
                    "llm_generation_completed",
                    attempt=attempt + 1,
                    latency=latency
                )
                
                return response
                
            except Exception as e:
                last_error = e
                self.error_count += 1
                
                logger.warning(
                    "llm_generation_failed",
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < self.config.retry_attempts - 1:
                    delay = self.config.retry_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
        
        # All retries failed
        raise RuntimeError(f"LLM generation failed after {self.config.retry_attempts} attempts: {last_error}")
    
    async def generate_with_template(
        self,
        template_name: str,
        use_cache: bool = True,
        **template_params
    ) -> LLMResponse:
        """
        Generate response using a prompt template.
        
        Args:
            template_name: Name of prompt template
            use_cache: Whether to use cache
            **template_params: Parameters for template
            
        Returns:
            LLM response
        """
        system, prompt = self.prompt_manager.format_prompt(template_name, **template_params)
        return await self.generate(prompt, system, use_cache)
    
    async def batch_generate(
        self,
        prompts: List[tuple[str, Optional[str]]],
        use_cache: bool = True,
        max_concurrent: int = 3
    ) -> List[LLMResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of (prompt, system) tuples
            use_cache: Whether to use cache
            max_concurrent: Maximum concurrent requests
            
        Returns:
            List of LLM responses
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def generate_with_semaphore(prompt: str, system: Optional[str]):
            async with semaphore:
                return await self.generate(prompt, system, use_cache)
        
        tasks = [
            generate_with_semaphore(prompt, system)
            for prompt, system in prompts
        ]
        
        return await asyncio.gather(*tasks)
    
    async def is_available(self) -> bool:
        """Check if LLM service is available."""
        return await self.provider.is_available()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get service statistics."""
        stats = {
            'requests': self.request_count,
            'errors': self.error_count,
            'error_rate': self.error_count / max(1, self.request_count),
            'avg_latency': self.total_latency / max(1, self.request_count),
            'provider': type(self.provider).__name__
        }
        
        if self.cache:
            stats['cache'] = self.cache.get_stats()
        
        return stats
    
    def clear_cache(self):
        """Clear response cache."""
        if self.cache:
            self.cache.clear()
            logger.info("llm_cache_cleared")
    
    async def __aenter__(self):
        """Async context manager entry."""
        if not await self.is_available():
            logger.warning("llm_provider_not_available")
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        pass