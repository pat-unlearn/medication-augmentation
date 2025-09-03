# Infrastructure Module

## Overview

The infrastructure module provides the foundational components for web scraping, caching, rate limiting, and data retrieval. It implements production-ready patterns for reliable and efficient external data access.

## Structure

```
infrastructure/
├── __init__.py
├── web/                   # Web scraping components
│   ├── __init__.py
│   ├── base.py            # Base scraper interface
│   ├── fda.py             # FDA Orange Book scraper
│   ├── clinicaltrials.py  # ClinicalTrials.gov API
│   ├── nccn.py            # NCCN guidelines scraper
│   └── oncokb.py          # OncoKB database integration
├── cache.py               # Caching implementations
└── rate_limiter.py        # Rate limiting strategies
```

## Key Components

### Web Scrapers (`web/`)

#### Base Scraper Interface
```python
from med_aug.infrastructure.web import BaseScraper

class CustomScraper(BaseScraper):
    def __init__(self):
        super().__init__(
            name="custom_scraper",
            base_url="https://api.example.com",
            rate_limit=1.0  # requests per second
        )
    
    async def fetch_data(self, query: str) -> Dict:
        # Implementation with automatic rate limiting
        return await self._make_request(f"/search?q={query}")
```

#### FDA Orange Book Scraper
```python
from med_aug.infrastructure.web import FDAOrangeBookScraper

scraper = FDAOrangeBookScraper()

# Search for drug information
results = await scraper.search_drug("pembrolizumab")

# Get approval information
approval_info = await scraper.get_approval_info("pembrolizumab")
```

#### ClinicalTrials.gov Integration
```python
from med_aug.infrastructure.web import ClinicalTrialsScraper

scraper = ClinicalTrialsScraper()

# Search active trials
trials = await scraper.search_trials(
    condition="NSCLC",
    intervention="pembrolizumab",
    status="RECRUITING"
)

# Get trial details
details = await scraper.get_trial_details("NCT02220894")
```

#### NCCN Guidelines Scraper
```python
from med_aug.infrastructure.web import NCCNScraper

scraper = NCCNScraper()

# Get treatment guidelines
guidelines = await scraper.get_guidelines("nsclc")

# Extract recommended medications
medications = await scraper.get_recommended_medications("nsclc", "stage_iv")
```

#### OncoKB Integration
```python
from med_aug.infrastructure.web import OncoKBScraper

scraper = OncoKBScraper()

# Get drug annotations
annotations = await scraper.get_drug_annotations("osimertinib")

# Check resistance mutations
resistance = await scraper.get_resistance_info("EGFR", "T790M")
```

### Caching System (`cache.py`)

Multi-level caching with memory and persistent storage:

#### Cache Manager
```python
from med_aug.infrastructure import CacheManager, CacheConfig

config = CacheConfig(
    backend="hybrid",  # memory, redis, or hybrid
    ttl_seconds=3600,
    max_entries=10000,
    eviction_policy="lru"
)

cache = CacheManager(config)

# Store data
await cache.set("key", data, ttl=7200)

# Retrieve data
data = await cache.get("key")

# Batch operations
await cache.set_many({"key1": data1, "key2": data2})
results = await cache.get_many(["key1", "key2"])
```

#### Memory Cache
```python
from med_aug.infrastructure import MemoryCache

cache = MemoryCache(max_size=1000, ttl_seconds=3600)

# LRU eviction when size limit reached
cache.put("key", value)
value = cache.get("key")

# TTL-based expiration
cache.put("temp_key", value, ttl=60)
```

#### Redis Cache
```python
from med_aug.infrastructure import RedisCache

cache = RedisCache(
    host="localhost",
    port=6379,
    db=0,
    ttl_seconds=86400
)

# Async operations
await cache.set("key", value)
value = await cache.get("key")

# Atomic operations
await cache.increment("counter")
await cache.expire("key", 3600)
```

#### Hybrid Cache
```python
from med_aug.infrastructure import HybridCache

# L1: Memory (fast), L2: Redis (persistent)
cache = HybridCache(
    l1_size=100,      # Memory cache size
    l1_ttl=300,       # 5 minutes
    l2_ttl=86400      # 24 hours
)

# Automatic tiered caching
value = await cache.get("key")  # Checks L1, then L2
```

### Rate Limiting (`rate_limiter.py`)

Advanced rate limiting strategies:

#### Token Bucket
```python
from med_aug.infrastructure import TokenBucketRateLimiter

# 10 requests per second with burst of 20
limiter = TokenBucketRateLimiter(
    rate=10,
    capacity=20
)

# Check if request allowed
if await limiter.allow_request():
    # Make request
    pass
else:
    # Wait or handle rate limit
    wait_time = await limiter.time_until_available()
```

#### Sliding Window
```python
from med_aug.infrastructure import SlidingWindowRateLimiter

# 100 requests per minute
limiter = SlidingWindowRateLimiter(
    requests=100,
    window_seconds=60
)

async with limiter:
    # Automatically rate-limited
    response = await make_request()
```

#### Adaptive Rate Limiting
```python
from med_aug.infrastructure import AdaptiveRateLimiter

# Adjusts rate based on response times
limiter = AdaptiveRateLimiter(
    initial_rate=10,
    min_rate=1,
    max_rate=50,
    target_latency=0.5  # seconds
)

# Automatically adjusts rate
await limiter.record_latency(0.3)  # Increases rate
await limiter.record_error()       # Decreases rate
```

#### Per-Domain Rate Limiting
```python
from med_aug.infrastructure import DomainRateLimiter

limiter = DomainRateLimiter({
    "api.fda.gov": 2.0,        # 2 req/sec
    "clinicaltrials.gov": 1.0,  # 1 req/sec
    "oncokb.org": 0.5,          # 1 req/2sec
    "default": 1.0              # Default rate
})

# Automatic per-domain limiting
await limiter.wait_if_needed("api.fda.gov")
```

## Usage Patterns

### Web Scraping with Caching
```python
from med_aug.infrastructure import CachedScraper

scraper = CachedScraper(
    base_scraper=FDAOrangeBookScraper(),
    cache=cache_manager,
    cache_ttl=3600
)

# First call fetches from web
data = await scraper.search_drug("pembrolizumab")

# Subsequent calls use cache
data = await scraper.search_drug("pembrolizumab")  # From cache
```

### Batch Processing
```python
from med_aug.infrastructure import BatchProcessor

processor = BatchProcessor(
    scraper=scraper,
    rate_limiter=limiter,
    batch_size=10,
    max_concurrent=5
)

# Process multiple items efficiently
medications = ["drug1", "drug2", "drug3", ...]
results = await processor.process_batch(medications)
```

### Retry Logic
```python
from med_aug.infrastructure import RetryManager

retry = RetryManager(
    max_attempts=3,
    backoff_factor=2.0,
    max_delay=60
)

@retry.with_retry
async def fetch_data(url: str):
    # Automatic retry on failure
    return await scraper.fetch(url)
```

### Circuit Breaker
```python
from med_aug.infrastructure import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=60,
    expected_exception=RequestException
)

@breaker.protected
async def external_api_call():
    # Circuit opens after 5 failures
    # Waits 60 seconds before retry
    return await api.call()
```

## Configuration

### Environment Variables
```bash
# Cache settings
export MEDAUG_CACHE_BACKEND=redis
export MEDAUG_REDIS_HOST=localhost
export MEDAUG_REDIS_PORT=6379
export MEDAUG_CACHE_TTL=3600

# Rate limiting
export MEDAUG_RATE_LIMIT_DEFAULT=1.0
export MEDAUG_RATE_LIMIT_BURST=10

# Scraper settings
export MEDAUG_SCRAPER_TIMEOUT=30
export MEDAUG_SCRAPER_MAX_RETRIES=3
```

### Configuration File
```yaml
infrastructure:
  cache:
    backend: hybrid
    memory_size: 1000
    redis:
      host: localhost
      port: 6379
    ttl_seconds: 3600
  
  rate_limiting:
    strategy: adaptive
    default_rate: 1.0
    per_domain:
      api.fda.gov: 2.0
      clinicaltrials.gov: 1.0
  
  scrapers:
    timeout: 30
    max_retries: 3
    user_agent: "MedAug/1.0"
```

## Error Handling

```python
from med_aug.infrastructure import (
    ScraperError,
    RateLimitError,
    CacheError,
    NetworkError
)

try:
    data = await scraper.fetch_data(query)
except RateLimitError as e:
    logger.warning(f"Rate limited: {e.retry_after} seconds")
    await asyncio.sleep(e.retry_after)
except NetworkError as e:
    logger.error(f"Network error: {e}")
    # Use cached data if available
    data = await cache.get(cache_key)
except ScraperError as e:
    logger.error(f"Scraping failed: {e}")
    # Handle gracefully
```

## Performance Optimization

### Connection Pooling
```python
from med_aug.infrastructure import ConnectionPool

pool = ConnectionPool(
    max_connections=100,
    max_keepalive=30,
    timeout=5.0
)

# Reuses connections efficiently
async with pool.get_client() as client:
    response = await client.get(url)
```

### Async Batch Processing
```python
import asyncio

async def process_medications(medications: List[str]):
    tasks = []
    for med in medications:
        task = scraper.search_drug(med)
        tasks.append(task)
    
    # Process concurrently with rate limiting
    results = await asyncio.gather(*tasks)
    return results
```

### Cache Warming
```python
from med_aug.infrastructure import CacheWarmer

warmer = CacheWarmer(cache, scraper)

# Pre-populate cache
common_drugs = ["pembrolizumab", "nivolumab", ...]
await warmer.warm_cache(common_drugs)
```

## Monitoring and Metrics

```python
from med_aug.infrastructure import MetricsCollector

metrics = MetricsCollector()

# Track performance
metrics.record_request("fda_api", duration=0.5, status=200)
metrics.record_cache_hit("memory")
metrics.record_rate_limit("clinicaltrials.gov")

# Get statistics
stats = metrics.get_stats()
print(f"Cache hit rate: {stats.cache_hit_rate}")
print(f"Average latency: {stats.avg_latency}")
```

## Testing

Comprehensive test coverage for:
- Web scraper functionality
- Cache operations and eviction
- Rate limiting accuracy
- Retry and circuit breaker logic
- Error handling scenarios
- Performance benchmarks

## Best Practices

1. **Always use rate limiting** for external APIs
2. **Implement caching** to reduce API calls
3. **Handle errors gracefully** with retries
4. **Monitor performance** and adjust limits
5. **Use connection pooling** for efficiency
6. **Implement circuit breakers** for resilience
7. **Log all external interactions** for debugging
8. **Respect API terms of service**
9. **Use async operations** for scalability
10. **Implement cache warming** for critical data

## Future Enhancements Ideas

- GraphQL API support
- WebSocket connections for real-time data
- Distributed caching with cache coherence
- Machine learning-based rate limit optimization
- Automatic API schema discovery
- Response validation and sanitization
- Multi-region failover support
- Prometheus metrics export
