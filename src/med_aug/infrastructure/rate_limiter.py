"""Rate limiting utilities for web scraping and API calls."""

import asyncio
import time
from typing import Dict, Optional
from dataclasses import dataclass
from contextlib import asynccontextmanager


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    
    requests_per_second: float = 1.0
    burst_size: int = 5
    backoff_factor: float = 1.5
    max_backoff: float = 60.0


class TokenBucketRateLimiter:
    """Token bucket rate limiter for controlling request rates."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize rate limiter.
        
        Args:
            config: Rate limiting configuration
        """
        self.config = config
        self.tokens = config.burst_size
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a token, blocking if necessary."""
        async with self._lock:
            now = time.time()
            
            # Refill tokens based on time elapsed
            time_elapsed = now - self.last_refill
            tokens_to_add = time_elapsed * self.config.requests_per_second
            self.tokens = min(self.config.burst_size, self.tokens + tokens_to_add)
            self.last_refill = now
            
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return
            
            # Wait for next token
            wait_time = (1.0 - self.tokens) / self.config.requests_per_second
            await asyncio.sleep(wait_time)
            self.tokens = 0.0
    
    @asynccontextmanager
    async def limit(self):
        """Context manager for rate limiting."""
        await self.acquire()
        try:
            yield
        finally:
            pass


class AdaptiveRateLimiter:
    """Adaptive rate limiter that adjusts based on response codes."""
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize adaptive rate limiter.
        
        Args:
            config: Initial rate limiting configuration
        """
        self.base_config = config
        self.current_delay = 1.0 / config.requests_per_second
        self.consecutive_errors = 0
        self._lock = asyncio.Lock()
    
    async def acquire(self, response_code: Optional[int] = None) -> None:
        """
        Acquire a token with adaptive delay adjustment.
        
        Args:
            response_code: HTTP response code from previous request
        """
        async with self._lock:
            # Adjust delay based on response
            if response_code:
                if response_code == 429:  # Rate limited
                    self.consecutive_errors += 1
                    self.current_delay *= self.base_config.backoff_factor
                    self.current_delay = min(self.current_delay, self.base_config.max_backoff)
                elif response_code < 400:  # Success
                    if self.consecutive_errors > 0:
                        self.consecutive_errors = max(0, self.consecutive_errors - 1)
                        # Gradually reduce delay on success
                        self.current_delay = max(
                            1.0 / self.base_config.requests_per_second,
                            self.current_delay / 1.2
                        )
                else:  # Other errors
                    self.consecutive_errors += 1
                    self.current_delay *= 1.2
            
            # Apply delay
            if self.current_delay > 0:
                await asyncio.sleep(self.current_delay)
    
    @asynccontextmanager
    async def limit(self, response_code: Optional[int] = None):
        """Context manager for adaptive rate limiting."""
        await self.acquire(response_code)
        try:
            yield
        finally:
            pass


class DomainRateLimiter:
    """Per-domain rate limiting manager."""
    
    def __init__(self):
        """Initialize domain rate limiter."""
        self.domain_limiters: Dict[str, TokenBucketRateLimiter] = {}
        self._lock = asyncio.Lock()
    
    async def get_limiter(self, domain: str, config: Optional[RateLimitConfig] = None) -> TokenBucketRateLimiter:
        """
        Get or create rate limiter for domain.
        
        Args:
            domain: Domain name
            config: Rate limit config (uses default if None)
            
        Returns:
            Rate limiter for the domain
        """
        async with self._lock:
            if domain not in self.domain_limiters:
                if config is None:
                    config = RateLimitConfig()
                self.domain_limiters[domain] = TokenBucketRateLimiter(config)
            return self.domain_limiters[domain]
    
    async def acquire(self, domain: str, config: Optional[RateLimitConfig] = None) -> None:
        """
        Acquire token for domain.
        
        Args:
            domain: Domain name
            config: Rate limit config
        """
        limiter = await self.get_limiter(domain, config)
        await limiter.acquire()
    
    @asynccontextmanager
    async def limit(self, domain: str, config: Optional[RateLimitConfig] = None):
        """Context manager for domain rate limiting."""
        await self.acquire(domain, config)
        try:
            yield
        finally:
            pass


# Global domain rate limiter instance
_domain_limiter = DomainRateLimiter()


async def rate_limit(domain: str, config: Optional[RateLimitConfig] = None):
    """
    Simple rate limiting function for a domain.
    
    Args:
        domain: Domain name
        config: Rate limit configuration
    """
    await _domain_limiter.acquire(domain, config)


@asynccontextmanager
async def rate_limited(domain: str, config: Optional[RateLimitConfig] = None):
    """
    Context manager for rate limiting requests to a domain.
    
    Args:
        domain: Domain name  
        config: Rate limit configuration
        
    Example:
        async with rate_limited("api.example.com"):
            response = await client.get("https://api.example.com/data")
    """
    async with _domain_limiter.limit(domain, config):
        yield


def get_domain_from_url(url: str) -> str:
    """
    Extract domain from URL.
    
    Args:
        url: Full URL
        
    Returns:
        Domain name
    """
    from urllib.parse import urlparse
    
    parsed = urlparse(url)
    return parsed.netloc.lower()


# Default rate limit configurations for common domains
DEFAULT_CONFIGS = {
    "fda.gov": RateLimitConfig(requests_per_second=0.5, burst_size=2),
    "clinicaltrials.gov": RateLimitConfig(requests_per_second=1.0, burst_size=3),
    "api.fda.gov": RateLimitConfig(requests_per_second=0.5, burst_size=2),
    "nccn.org": RateLimitConfig(requests_per_second=0.3, burst_size=1),
    "oncokb.org": RateLimitConfig(requests_per_second=0.5, burst_size=2),
}


def get_default_config(domain: str) -> RateLimitConfig:
    """
    Get default rate limit configuration for a domain.
    
    Args:
        domain: Domain name
        
    Returns:
        Rate limit configuration
    """
    # Check exact match first
    if domain in DEFAULT_CONFIGS:
        return DEFAULT_CONFIGS[domain]
    
    # Check if any default domain is a suffix
    for default_domain, config in DEFAULT_CONFIGS.items():
        if domain.endswith(default_domain):
            return config
    
    # Return conservative default
    return RateLimitConfig(requests_per_second=0.5, burst_size=1)