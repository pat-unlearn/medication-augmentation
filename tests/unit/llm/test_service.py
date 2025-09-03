"""Unit tests for LLM service."""

import pytest
import asyncio
from pathlib import Path
import tempfile

from med_aug.llm.service import (
    LLMService,
    ResponseCache,
    CacheEntry
)
from med_aug.llm.providers import MockProvider, LLMConfig, LLMResponse
from med_aug.llm.prompts import PromptManager


class TestResponseCache:
    """Test response caching."""
    
    def test_cache_initialization(self):
        """Test cache initialization."""
        cache = ResponseCache(ttl_seconds=3600, max_entries=100)
        
        assert cache.ttl_seconds == 3600
        assert cache.max_entries == 100
        assert len(cache.memory_cache) == 0
    
    def test_cache_set_and_get(self):
        """Test setting and getting from cache."""
        cache = ResponseCache()
        
        response = LLMResponse(
            content="Test response",
            model="test-model"
        )
        
        # Set in cache
        cache.set("Test prompt", "Test system", response)
        
        # Get from cache
        cached = cache.get("Test prompt", "Test system")
        
        assert cached is not None
        assert cached.content == "Test response"
        assert cached.model == "test-model"
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = ResponseCache()
        
        # Try to get non-existent entry
        result = cache.get("Non-existent prompt", None)
        
        assert result is None
    
    def test_cache_key_generation(self):
        """Test cache key generation is deterministic."""
        cache = ResponseCache()
        
        key1 = cache._generate_key("Prompt", "System", param1="value1")
        key2 = cache._generate_key("Prompt", "System", param1="value1")
        key3 = cache._generate_key("Different", "System", param1="value1")
        
        assert key1 == key2
        assert key1 != key3
    
    def test_cache_expiration(self):
        """Test cache entry expiration."""
        from datetime import datetime, timedelta
        
        cache = ResponseCache(ttl_seconds=1)
        
        response = LLMResponse(content="Test", model="test")
        cache.set("Prompt", None, response)
        
        # Should be in cache
        assert cache.get("Prompt", None) is not None
        
        # Manually expire the entry
        key = cache._generate_key("Prompt", None)
        cache.memory_cache[key].timestamp = datetime.now() - timedelta(seconds=2)
        
        # Should not be in cache (expired)
        assert cache.get("Prompt", None) is None
    
    def test_cache_eviction(self):
        """Test cache eviction when max entries exceeded."""
        cache = ResponseCache(max_entries=3)
        
        # Add more than max entries
        for i in range(5):
            response = LLMResponse(content=f"Response {i}", model="test")
            cache.set(f"Prompt {i}", None, response)
        
        # Cache should not exceed max_entries
        assert len(cache.memory_cache) <= 3
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = ResponseCache()
        
        response = LLMResponse(content="Test", model="test")
        cache.set("Prompt", None, response)
        
        # Access to increase hit count
        cache.get("Prompt", None)
        cache.get("Prompt", None)
        
        stats = cache.get_stats()
        
        assert stats['entries'] == 1
        assert stats['total_hits'] == 2
        assert stats['ttl_seconds'] == 3600
    
    def test_cache_clear(self):
        """Test clearing cache."""
        cache = ResponseCache()
        
        # Add some entries
        for i in range(3):
            response = LLMResponse(content=f"Response {i}", model="test")
            cache.set(f"Prompt {i}", None, response)
        
        assert len(cache.memory_cache) == 3
        
        # Clear cache
        cache.clear()
        
        assert len(cache.memory_cache) == 0
    
    def test_persistent_cache(self):
        """Test persistent cache with file storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Create cache and add entry
            cache1 = ResponseCache(cache_dir=cache_dir)
            response = LLMResponse(content="Persistent", model="test")
            cache1.set("Test prompt", None, response)
            
            # Create new cache instance (should load from disk)
            cache2 = ResponseCache(cache_dir=cache_dir)
            
            # Should find the cached entry
            cached = cache2.get("Test prompt", None)
            assert cached is not None
            assert cached.content == "Persistent"


class TestLLMService:
    """Test LLM service."""
    
    @pytest.fixture
    def mock_provider(self):
        """Create mock provider."""
        provider = MockProvider()
        provider.set_responses([
            "Response 1",
            "Response 2",
            "Response 3"
        ])
        return provider
    
    @pytest.fixture
    def service(self, mock_provider):
        """Create LLM service with mock provider."""
        return LLMService(
            provider=mock_provider,
            enable_cache=True
        )
    
    @pytest.mark.asyncio
    async def test_service_generate(self, service):
        """Test basic generation."""
        response = await service.generate(
            "Test prompt",
            system="Test system"
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Response 1"
        assert service.request_count == 1
    
    @pytest.mark.asyncio
    async def test_service_caching(self, service):
        """Test response caching."""
        # First call - should hit provider
        response1 = await service.generate("Same prompt", use_cache=True)
        assert response1.content == "Response 1"
        assert service.request_count == 1
        
        # Second call - should hit cache
        response2 = await service.generate("Same prompt", use_cache=True)
        assert response2.content == "Response 1"  # Same response
        assert service.request_count == 1  # No new request
        
        # Different prompt - should hit provider
        response3 = await service.generate("Different prompt", use_cache=True)
        assert response3.content == "Response 2"
        assert service.request_count == 2
    
    @pytest.mark.asyncio
    async def test_service_no_cache(self, service):
        """Test generation without caching."""
        # Multiple calls with cache disabled
        response1 = await service.generate("Same prompt", use_cache=False)
        assert response1.content == "Response 1"
        
        response2 = await service.generate("Same prompt", use_cache=False)
        assert response2.content == "Response 2"  # Different response
        
        assert service.request_count == 2
    
    @pytest.mark.asyncio
    async def test_service_with_template(self, service):
        """Test generation with template."""
        # Add a simple template
        service.prompt_manager.get_template('classification')
        
        response = await service.generate_with_template(
            'classification',
            medication='test_drug',
            disease='test_disease',
            drug_classes='test_classes'
        )
        
        assert isinstance(response, LLMResponse)
        assert response.content == "Response 1"
    
    @pytest.mark.asyncio
    async def test_batch_generate(self, service):
        """Test batch generation."""
        prompts = [
            ("Prompt 1", "System 1"),
            ("Prompt 2", "System 2"),
            ("Prompt 3", None)
        ]
        
        responses = await service.batch_generate(prompts, max_concurrent=2)
        
        assert len(responses) == 3
        assert responses[0].content == "Response 1"
        assert responses[1].content == "Response 2"
        assert responses[2].content == "Response 3"
        assert service.request_count == 3
    
    @pytest.mark.asyncio
    async def test_service_retry_logic(self):
        """Test retry logic on failures."""
        # Create a provider that fails initially
        class FailingProvider(MockProvider):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0
            
            async def generate(self, prompt, system=None, **kwargs):
                self.attempt_count += 1
                if self.attempt_count < 2:
                    raise Exception("Simulated failure")
                return LLMResponse(content="Success", model="test")
        
        provider = FailingProvider()
        config = LLMConfig(retry_attempts=3, retry_delay=0.01)
        service = LLMService(provider=provider, config=config, enable_cache=False)
        
        response = await service.generate("Test prompt")
        
        assert response.content == "Success"
        assert provider.attempt_count == 2
        assert service.error_count == 1
    
    @pytest.mark.asyncio
    async def test_service_all_retries_fail(self):
        """Test when all retries fail."""
        class AlwaysFailingProvider(MockProvider):
            async def generate(self, prompt, system=None, **kwargs):
                raise Exception("Always fails")
        
        provider = AlwaysFailingProvider()
        config = LLMConfig(retry_attempts=2, retry_delay=0.01)
        service = LLMService(provider=provider, config=config, enable_cache=False)
        
        with pytest.raises(RuntimeError, match="failed after 2 attempts"):
            await service.generate("Test prompt")
        
        assert service.error_count == 2
    
    @pytest.mark.asyncio
    async def test_service_is_available(self, service):
        """Test checking service availability."""
        available = await service.is_available()
        assert available is True
    
    def test_service_stats(self, service):
        """Test service statistics."""
        stats = service.get_stats()
        
        assert stats['requests'] == 0
        assert stats['errors'] == 0
        assert stats['error_rate'] == 0.0
        assert stats['provider'] == 'MockProvider'
        assert 'cache' in stats
    
    def test_clear_cache(self, service):
        """Test clearing service cache."""
        service.clear_cache()
        
        if service.cache:
            assert len(service.cache.memory_cache) == 0
    
    @pytest.mark.asyncio
    async def test_service_context_manager(self, mock_provider):
        """Test service as async context manager."""
        async with LLMService(provider=mock_provider) as service:
            response = await service.generate("Test prompt")
            assert response.content == "Response 1"