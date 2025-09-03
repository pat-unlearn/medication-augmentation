"""Base web scraper with rate limiting and retry logic."""

import asyncio
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse
import httpx
from bs4 import BeautifulSoup
import structlog

from ..cache.base import BaseCache, CacheConfig
from ..cache.memory_cache import MemoryCache
from ..rate_limiter import DomainRateLimiter, RateLimitConfig

logger = structlog.get_logger()


@dataclass
class ScraperConfig:
    """Configuration for web scrapers."""

    base_url: str
    rate_limit: float = 1.0  # Seconds between requests
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    user_agent: str = "MedicationAugmentation/1.0"
    headers: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        """Set default headers if not provided."""
        if not self.headers:
            self.headers = {
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Accept-Encoding": "gzip, deflate",
                "Connection": "keep-alive",
            }


@dataclass
class ScraperResult:
    """Result from a web scraping operation."""

    success: bool
    data: Dict[str, Any]
    url: str
    timestamp: datetime = field(default_factory=datetime.now)
    error: Optional[str] = None
    retry_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "success": self.success,
            "data": self.data,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "error": self.error,
            "retry_count": self.retry_count,
        }


class BaseScraper(ABC):
    """Abstract base class for web scrapers."""

    def __init__(
        self,
        config: ScraperConfig,
        client: Optional[httpx.AsyncClient] = None,
        cache: Optional[BaseCache] = None,
        rate_limiter: Optional[DomainRateLimiter] = None,
    ):
        """
        Initialize the scraper.

        Args:
            config: Scraper configuration
            client: Optional HTTP client (will create if not provided)
            cache: Optional cache instance
            rate_limiter: Optional rate limiter instance
        """
        self.config = config
        self._client = client
        self._last_request_time = 0.0
        self._request_count = 0

        # Initialize cache
        if cache is None:
            cache_config = CacheConfig(ttl_seconds=3600, max_size=1000)
            self.cache = MemoryCache(cache_config)
        else:
            self.cache = cache

        # Initialize rate limiter
        if rate_limiter is None:
            rate_config = RateLimitConfig(
                requests_per_second=1.0 / config.rate_limit, burst_size=3
            )
            self.rate_limiter = DomainRateLimiter(rate_config)
        else:
            self.rate_limiter = rate_limiter

    async def __aenter__(self):
        """Async context manager entry."""
        if not self._client:
            self._client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self.config.headers,
                follow_redirects=True,
            )

        # Start cache cleanup if using MemoryCache
        if isinstance(self.cache, MemoryCache):
            await self.cache.start()

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client and not hasattr(self._client, "_is_external"):
            await self._client.aclose()

        # Stop cache cleanup if using MemoryCache
        if isinstance(self.cache, MemoryCache):
            await self.cache.stop()

    async def _fetch_url(
        self, url: str, use_cache: bool = True, **kwargs
    ) -> httpx.Response:
        """
        Fetch a URL with rate limiting, caching, and retry logic.

        Args:
            url: URL to fetch
            use_cache: Whether to use cache
            **kwargs: Additional arguments for httpx request

        Returns:
            HTTP response

        Raises:
            httpx.HTTPError: If request fails after retries
        """
        # Extract domain for rate limiting
        parsed = urlparse(url)
        domain = parsed.netloc

        # Check cache first
        if use_cache:
            cache_key = self.cache.make_key("url", url, kwargs)
            cached_response = await self.cache.get(cache_key)
            if cached_response is not None:
                logger.debug("cache_hit", url=url)
                return cached_response

        # Apply rate limiting
        await self.rate_limiter.wait_and_acquire(domain)

        for attempt in range(self.config.max_retries):
            try:
                logger.debug(
                    "fetching_url",
                    url=url,
                    attempt=attempt + 1,
                    max_retries=self.config.max_retries,
                )

                response = await self._client.get(url, **kwargs)
                response.raise_for_status()

                logger.debug(
                    "fetch_success",
                    url=url,
                    status_code=response.status_code,
                    attempt=attempt + 1,
                )

                # Cache successful response
                if use_cache:
                    await self.cache.set(cache_key, response, self.config.timeout * 60)

                self._request_count += 1
                return response

            except httpx.HTTPError as e:
                logger.warning(
                    "fetch_failed", url=url, attempt=attempt + 1, error=str(e)
                )

                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                else:
                    raise

    async def _parse_html(self, html: str) -> BeautifulSoup:
        """
        Parse HTML content.

        Args:
            html: HTML content

        Returns:
            BeautifulSoup object
        """
        return BeautifulSoup(html, "html.parser")

    @abstractmethod
    async def scrape_medication_info(self, medication_name: str) -> ScraperResult:
        """
        Scrape information for a specific medication.

        Args:
            medication_name: Name of the medication

        Returns:
            Scraper result with medication data
        """
        pass

    @abstractmethod
    async def search_medications(
        self, query: str, limit: int = 10
    ) -> List[ScraperResult]:
        """
        Search for medications matching a query.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of scraper results
        """
        pass

    @abstractmethod
    async def get_recent_approvals(self, days: int = 365) -> List[ScraperResult]:
        """
        Get recently approved medications.

        Args:
            days: Number of days to look back

        Returns:
            List of recently approved medications
        """
        pass

    async def scrape_batch(self, medication_names: List[str]) -> List[ScraperResult]:
        """
        Scrape information for multiple medications.

        Args:
            medication_names: List of medication names

        Returns:
            List of scraper results
        """
        results = []

        # Use semaphore to limit concurrent requests
        semaphore = asyncio.Semaphore(5)  # Max 5 concurrent requests

        async def scrape_with_semaphore(name: str) -> ScraperResult:
            async with semaphore:
                return await self.scrape_medication_info(name)

        tasks = [scrape_with_semaphore(name) for name in medication_names]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to failed results
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(
                    ScraperResult(
                        success=False,
                        data={},
                        url=self.config.base_url,
                        error=str(result),
                    )
                )
            else:
                final_results.append(result)

        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get scraper statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "request_count": self._request_count,
            "base_url": self.config.base_url,
            "rate_limit": self.config.rate_limit,
            "last_request_time": self._last_request_time,
        }


class CompositeScraper:
    """Composite scraper that combines multiple scrapers."""

    def __init__(self, scrapers: List[BaseScraper]):
        """
        Initialize composite scraper.

        Args:
            scrapers: List of scrapers to use
        """
        self.scrapers = scrapers

    async def scrape_medication_info(
        self, medication_name: str
    ) -> Dict[str, ScraperResult]:
        """
        Scrape medication info from all scrapers.

        Args:
            medication_name: Name of medication

        Returns:
            Dictionary mapping scraper name to result
        """
        results = {}

        tasks = []
        for scraper in self.scrapers:
            task = scraper.scrape_medication_info(medication_name)
            tasks.append(task)

        scraper_results = await asyncio.gather(*tasks, return_exceptions=True)

        for scraper, result in zip(self.scrapers, scraper_results):
            scraper_name = scraper.__class__.__name__
            if isinstance(result, Exception):
                results[scraper_name] = ScraperResult(
                    success=False,
                    data={},
                    url=scraper.config.base_url,
                    error=str(result),
                )
            else:
                results[scraper_name] = result

        return results

    async def search_medications(
        self, query: str, limit: int = 10
    ) -> Dict[str, List[ScraperResult]]:
        """
        Search for medications across all scrapers.

        Args:
            query: Search query
            limit: Maximum results per scraper

        Returns:
            Dictionary mapping scraper name to results
        """
        results = {}

        tasks = []
        for scraper in self.scrapers:
            task = scraper.search_medications(query, limit)
            tasks.append(task)

        scraper_results = await asyncio.gather(*tasks, return_exceptions=True)

        for scraper, result in zip(self.scrapers, scraper_results):
            scraper_name = scraper.__class__.__name__
            if isinstance(result, Exception):
                results[scraper_name] = []
            else:
                results[scraper_name] = result

        return results

    def merge_results(self, results: Dict[str, ScraperResult]) -> Dict[str, Any]:
        """
        Merge results from multiple scrapers.

        Args:
            results: Dictionary of scraper results

        Returns:
            Merged medication information
        """
        merged = {
            "names": set(),
            "brand_names": set(),
            "generic_names": set(),
            "indications": set(),
            "drug_class": set(),
            "approval_dates": [],
            "sources": [],
        }

        for scraper_name, result in results.items():
            if result.success and result.data:
                data = result.data

                # Merge names
                if "name" in data:
                    merged["names"].add(data["name"])
                if "brand_names" in data:
                    merged["brand_names"].update(data.get("brand_names", []))
                if "generic_names" in data:
                    merged["generic_names"].update(data.get("generic_names", []))

                # Merge indications
                if "indications" in data:
                    merged["indications"].update(data.get("indications", []))

                # Merge drug class
                if "drug_class" in data:
                    merged["drug_class"].update(data.get("drug_class", []))

                # Collect approval dates
                if "approval_date" in data:
                    merged["approval_dates"].append(
                        {"date": data["approval_date"], "source": scraper_name}
                    )

                # Track sources
                merged["sources"].append(
                    {
                        "scraper": scraper_name,
                        "url": result.url,
                        "timestamp": result.timestamp.isoformat(),
                    }
                )

        # Convert sets to lists
        for key in [
            "names",
            "brand_names",
            "generic_names",
            "indications",
            "drug_class",
        ]:
            merged[key] = list(merged[key])

        return merged
