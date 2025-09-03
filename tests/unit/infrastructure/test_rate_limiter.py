"""Unit tests for rate limiters."""

import pytest
import asyncio
import time
from med_aug.infrastructure.rate_limiter import (
    RateLimitConfig,
    TokenBucketRateLimiter,
    SlidingWindowRateLimiter,
    DomainRateLimiter,
    AdaptiveRateLimiter
)


class TestTokenBucketRateLimiter:
    """Test TokenBucketRateLimiter functionality."""
    
    @pytest.fixture
    def limiter(self):
        """Create rate limiter instance."""
        config = RateLimitConfig(
            requests_per_second=10.0,  # 10 requests per second
            burst_size=3
        )
        return TokenBucketRateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_burst_capacity(self, limiter):
        """Test burst capacity allows multiple quick requests."""
        # Should allow burst_size requests immediately
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        
        # Fourth request should fail (no tokens left)
        assert await limiter.acquire() is False
    
    @pytest.mark.asyncio
    async def test_token_refill(self, limiter):
        """Test token refill over time."""
        # Use all tokens
        for _ in range(3):
            await limiter.acquire()
        
        # No tokens left
        assert await limiter.acquire() is False
        
        # Wait for refill (10 req/s = 0.1s per token)
        await asyncio.sleep(0.15)
        
        # Should have at least 1 token now
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_wait_and_acquire(self, limiter):
        """Test wait_and_acquire functionality."""
        # Use all tokens
        for _ in range(3):
            await limiter.acquire()
        
        # This should wait and then acquire
        start = time.time()
        wait_time = await limiter.wait_and_acquire()
        elapsed = time.time() - start
        
        assert wait_time >= 0
        assert elapsed >= 0.05  # Should have waited at least a bit
    
    @pytest.mark.asyncio
    async def test_max_tokens_cap(self, limiter):
        """Test that tokens don't exceed max capacity."""
        # Wait for potential refill
        await asyncio.sleep(1.0)
        
        # Should still be capped at burst_size
        count = 0
        while await limiter.acquire():
            count += 1
            if count > 10:  # Safety limit
                break
        
        assert count == 3  # Should equal burst_size


class TestSlidingWindowRateLimiter:
    """Test SlidingWindowRateLimiter functionality."""
    
    @pytest.fixture
    def limiter(self):
        """Create sliding window limiter."""
        config = RateLimitConfig(
            requests_per_second=2.0,
            window_seconds=1
        )
        return SlidingWindowRateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_window_limit(self, limiter):
        """Test requests within window."""
        # 2 req/s * 1s window = 2 requests allowed
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        assert await limiter.acquire() is False  # Over limit
    
    @pytest.mark.asyncio
    async def test_window_sliding(self, limiter):
        """Test sliding window behavior."""
        # Make 2 requests (at limit)
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
        
        # Wait for window to slide
        await asyncio.sleep(1.1)
        
        # Should allow new requests
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_wait_and_acquire(self, limiter):
        """Test wait_and_acquire with sliding window."""
        # Fill window
        await limiter.acquire()
        await limiter.acquire()
        
        # This should wait for window to slide
        start = time.time()
        wait_time = await limiter.wait_and_acquire()
        elapsed = time.time() - start
        
        assert wait_time > 0
        assert elapsed >= 0.5  # Should wait for window to slide


class TestDomainRateLimiter:
    """Test DomainRateLimiter functionality."""
    
    @pytest.fixture
    def limiter(self):
        """Create domain rate limiter."""
        default_config = RateLimitConfig(
            requests_per_second=1.0,
            burst_size=2
        )
        return DomainRateLimiter(default_config)
    
    @pytest.mark.asyncio
    async def test_separate_domain_limits(self, limiter):
        """Test separate limits for different domains."""
        # Different domains should have separate limits
        assert await limiter.acquire("domain1.com") is True
        assert await limiter.acquire("domain1.com") is True
        assert await limiter.acquire("domain1.com") is False  # At limit
        
        # Different domain should work
        assert await limiter.acquire("domain2.com") is True
        assert await limiter.acquire("domain2.com") is True
    
    @pytest.mark.asyncio
    async def test_custom_domain_config(self, limiter):
        """Test custom configuration for specific domain."""
        # Set custom config for specific domain
        custom_config = RateLimitConfig(
            requests_per_second=5.0,
            burst_size=5
        )
        limiter.set_domain_config("fast.com", custom_config)
        
        # Should allow more requests for this domain
        for _ in range(5):
            assert await limiter.acquire("fast.com") is True
        
        # Default domain still limited
        assert await limiter.acquire("normal.com") is True
        assert await limiter.acquire("normal.com") is True
        assert await limiter.acquire("normal.com") is False
    
    @pytest.mark.asyncio
    async def test_wait_and_acquire_per_domain(self, limiter):
        """Test wait_and_acquire for domains."""
        # Fill limit for domain1
        await limiter.acquire("domain1.com")
        await limiter.acquire("domain1.com")
        
        # Should wait for domain1
        wait_time = await limiter.wait_and_acquire("domain1.com")
        assert wait_time >= 0
        
        # domain2 should not wait (or wait very little)
        wait_time = await limiter.wait_and_acquire("domain2.com")
        assert wait_time < 0.01  # Less than 10ms
    
    def test_get_stats(self, limiter):
        """Test statistics retrieval."""
        asyncio.run(limiter.acquire("domain1.com"))
        asyncio.run(limiter.acquire("domain2.com"))
        
        stats = limiter.get_stats()
        assert "domain1.com" in stats
        assert "domain2.com" in stats
        assert "tokens" in stats["domain1.com"]
        assert "refill_rate" in stats["domain1.com"]


class TestAdaptiveRateLimiter:
    """Test AdaptiveRateLimiter functionality."""
    
    @pytest.fixture
    def limiter(self):
        """Create adaptive rate limiter."""
        config = RateLimitConfig(
            requests_per_second=2.0,
            burst_size=3
        )
        return AdaptiveRateLimiter(config)
    
    @pytest.mark.asyncio
    async def test_basic_functionality(self, limiter):
        """Test basic acquire functionality."""
        assert await limiter.acquire() is True
        assert await limiter.acquire() is True
    
    @pytest.mark.asyncio
    async def test_record_success(self, limiter):
        """Test recording successful responses."""
        # Record some successful fast responses
        for _ in range(10):
            await limiter.record_response(0.5, True)
        
        # Stats should be updated
        assert limiter.success_count == 10
        assert limiter.error_count == 0
        assert len(limiter.response_times) > 0
    
    @pytest.mark.asyncio
    async def test_record_errors(self, limiter):
        """Test recording errors."""
        # Record errors
        for _ in range(5):
            await limiter.record_response(1.0, False)
        
        assert limiter.error_count == 5
        assert limiter.success_count == 0
    
    @pytest.mark.asyncio
    async def test_rate_adjustment_on_errors(self, limiter):
        """Test rate reduction on high error rate."""
        original_rate = limiter.limiter.refill_rate
        
        # Record many errors to trigger slowdown
        for _ in range(20):
            await limiter.record_response(0.5, False)
        
        # Force adjustment
        limiter.last_adjustment = 0
        await limiter.record_response(0.5, False)
        
        # Rate should be reduced
        assert limiter.limiter.refill_rate < original_rate
    
    @pytest.mark.asyncio
    async def test_rate_adjustment_on_success(self, limiter):
        """Test rate increase on success."""
        original_rate = limiter.limiter.refill_rate
        
        # Record many fast successful responses
        for _ in range(100):
            await limiter.record_response(0.1, True)
        
        # Force adjustment
        limiter.last_adjustment = 0
        await limiter.record_response(0.1, True)
        
        # Rate should increase (but capped)
        assert limiter.limiter.refill_rate >= original_rate
        assert limiter.limiter.refill_rate <= original_rate * 2