"""Rate limiting for web scrapers."""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass
from collections import deque
import structlog

logger = structlog.get_logger()


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_second: float = 1.0
    burst_size: int = 1
    window_seconds: int = 60
    max_retries: int = 3


class TokenBucketRateLimiter:
    """Token bucket rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.tokens = float(config.burst_size)
        self.max_tokens = float(config.burst_size)
        self.refill_rate = config.requests_per_second
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """
        Acquire tokens from bucket.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            True if acquired successfully
        """
        async with self._lock:
            # Refill bucket
            now = time.time()
            elapsed = now - self.last_refill
            self.tokens = min(
                self.max_tokens,
                self.tokens + (elapsed * self.refill_rate)
            )
            self.last_refill = now
            
            # Check if enough tokens
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_and_acquire(self, tokens: int = 1) -> float:
        """
        Wait until tokens are available and acquire them.
        
        Args:
            tokens: Number of tokens to acquire
            
        Returns:
            Time waited in seconds
        """
        start = time.time()
        
        while True:
            if await self.acquire(tokens):
                waited = time.time() - start
                if waited > 0:
                    logger.debug("rate_limit_waited", seconds=waited)
                return waited
            
            # Calculate wait time
            wait_time = tokens / self.refill_rate
            await asyncio.sleep(min(wait_time, 0.1))


class SlidingWindowRateLimiter:
    """Sliding window rate limiter implementation."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.requests: deque = deque()
        self.max_requests = int(config.requests_per_second * config.window_seconds)
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """
        Try to acquire a slot.
        
        Returns:
            True if acquired successfully
        """
        async with self._lock:
            now = time.time()
            window_start = now - self.config.window_seconds
            
            # Remove old requests outside window
            while self.requests and self.requests[0] < window_start:
                self.requests.popleft()
            
            # Check if under limit
            if len(self.requests) < self.max_requests:
                self.requests.append(now)
                return True
            
            return False
    
    async def wait_and_acquire(self) -> float:
        """
        Wait until a slot is available and acquire it.
        
        Returns:
            Time waited in seconds
        """
        start = time.time()
        
        while True:
            if await self.acquire():
                waited = time.time() - start
                if waited > 0:
                    logger.debug("rate_limit_waited", seconds=waited)
                return waited
            
            # Wait until oldest request expires
            async with self._lock:
                if self.requests:
                    oldest = self.requests[0]
                    wait_time = (oldest + self.config.window_seconds) - time.time()
                    wait_time = max(0.01, min(wait_time, 1.0))
                else:
                    wait_time = 0.1
            
            await asyncio.sleep(wait_time)


class DomainRateLimiter:
    """Rate limiter that handles multiple domains."""
    
    def __init__(self, default_config: RateLimitConfig):
        """
        Initialize domain rate limiter.
        
        Args:
            default_config: Default rate limit configuration
        """
        self.default_config = default_config
        self.limiters: Dict[str, TokenBucketRateLimiter] = {}
        self.domain_configs: Dict[str, RateLimitConfig] = {}
        self._lock = asyncio.Lock()
    
    def set_domain_config(self, domain: str, config: RateLimitConfig):
        """
        Set rate limit configuration for a specific domain.
        
        Args:
            domain: Domain name
            config: Rate limit configuration
        """
        self.domain_configs[domain] = config
        if domain in self.limiters:
            del self.limiters[domain]
    
    async def get_limiter(self, domain: str) -> TokenBucketRateLimiter:
        """
        Get or create rate limiter for domain.
        
        Args:
            domain: Domain name
            
        Returns:
            Rate limiter for domain
        """
        async with self._lock:
            if domain not in self.limiters:
                config = self.domain_configs.get(domain, self.default_config)
                self.limiters[domain] = TokenBucketRateLimiter(config)
            
            return self.limiters[domain]
    
    async def acquire(self, domain: str, tokens: int = 1) -> bool:
        """
        Acquire tokens for domain.
        
        Args:
            domain: Domain name
            tokens: Number of tokens
            
        Returns:
            True if acquired
        """
        limiter = await self.get_limiter(domain)
        return await limiter.acquire(tokens)
    
    async def wait_and_acquire(self, domain: str, tokens: int = 1) -> float:
        """
        Wait and acquire tokens for domain.
        
        Args:
            domain: Domain name
            tokens: Number of tokens
            
        Returns:
            Time waited
        """
        limiter = await self.get_limiter(domain)
        return await limiter.wait_and_acquire(tokens)
    
    def get_stats(self) -> Dict[str, Dict]:
        """
        Get statistics for all domains.
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        for domain, limiter in self.limiters.items():
            stats[domain] = {
                'tokens': limiter.tokens,
                'max_tokens': limiter.max_tokens,
                'refill_rate': limiter.refill_rate
            }
        return stats


class AdaptiveRateLimiter:
    """Rate limiter that adapts based on response times and errors."""
    
    def __init__(self, initial_config: RateLimitConfig):
        """
        Initialize adaptive rate limiter.
        
        Args:
            initial_config: Initial rate limit configuration
        """
        self.config = initial_config
        self.limiter = TokenBucketRateLimiter(initial_config)
        self.response_times = deque(maxlen=100)
        self.error_count = 0
        self.success_count = 0
        self.last_adjustment = time.time()
        self._lock = asyncio.Lock()
    
    async def record_response(self, response_time: float, success: bool):
        """
        Record a response for adaptation.
        
        Args:
            response_time: Response time in seconds
            success: Whether request was successful
        """
        async with self._lock:
            self.response_times.append(response_time)
            
            if success:
                self.success_count += 1
            else:
                self.error_count += 1
            
            # Check if we should adjust rate
            now = time.time()
            if now - self.last_adjustment > 30:  # Adjust every 30 seconds
                await self._adjust_rate()
                self.last_adjustment = now
    
    async def _adjust_rate(self):
        """Adjust rate based on performance."""
        if not self.response_times:
            return
        
        # Calculate metrics
        avg_response = sum(self.response_times) / len(self.response_times)
        error_rate = self.error_count / (self.error_count + self.success_count + 1)
        
        # Adjust rate
        if error_rate > 0.1:  # More than 10% errors
            # Slow down
            new_rate = self.limiter.refill_rate * 0.8
            logger.info("rate_limiter_slowing", 
                       current_rate=self.limiter.refill_rate,
                       new_rate=new_rate,
                       error_rate=error_rate)
        elif error_rate < 0.01 and avg_response < 1.0:  # Less than 1% errors and fast responses
            # Speed up
            new_rate = min(
                self.limiter.refill_rate * 1.1,
                self.config.requests_per_second * 2  # Max 2x original rate
            )
            logger.info("rate_limiter_speeding",
                       current_rate=self.limiter.refill_rate,
                       new_rate=new_rate,
                       avg_response=avg_response)
        else:
            return
        
        # Apply new rate
        self.limiter.refill_rate = new_rate
        
        # Reset counters
        self.error_count = 0
        self.success_count = 0
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens."""
        return await self.limiter.acquire(tokens)
    
    async def wait_and_acquire(self, tokens: int = 1) -> float:
        """Wait and acquire tokens."""
        return await self.limiter.wait_and_acquire(tokens)