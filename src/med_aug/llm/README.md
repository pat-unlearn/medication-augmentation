# LLM Module

## Overview

The LLM (Large Language Model) module provides a flexible abstraction layer for integrating various LLM providers into the medication augmentation system. It supports both API-based and CLI-based providers with comprehensive caching, retry logic, and template management.

## Structure

```
llm/
├── __init__.py
├── providers.py        # LLM provider implementations
├── service.py          # LLM service with caching and retry
├── templates.py        # Prompt template management
└── classifier.py       # Medication classification logic
```

## Key Components

### LLM Providers (`providers.py`)

Abstraction layer supporting multiple LLM backends:

#### Base Provider Interface
```python
from med_aug.llm import LLMProvider, LLMResponse

class CustomProvider(LLMProvider):
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        # Implementation
        return LLMResponse(
            text=response_text,
            tokens_used=token_count,
            latency=response_time,
            metadata={}
        )
```

#### Claude CLI Provider
```python
from med_aug.llm import ClaudeCLIProvider, LLMConfig

# No API key required - uses subprocess
config = LLMConfig(
    provider="claude_cli",
    model="claude-3-haiku",
    temperature=0.0,
    max_tokens=2048
)

provider = ClaudeCLIProvider(config)
response = await provider.generate(
    prompt="Classify this medication: pembrolizumab",
    system="You are a medical expert."
)
```

#### Anthropic API Provider (Ready for Integration)
```python
from med_aug.llm import AnthropicProvider

# When API key is available
config = LLMConfig(
    provider="anthropic",
    api_key="sk-ant-...",
    model="claude-3-opus",
    temperature=0.0
)

provider = AnthropicProvider(config)
response = await provider.generate(prompt)
```

#### Mock Provider (Testing)
```python
from med_aug.llm import MockProvider

# For testing without LLM calls
provider = MockProvider(
    responses=["Response 1", "Response 2"],
    latency=0.1
)

response = await provider.generate("test prompt")
```

### LLM Service (`service.py`)

High-level service with caching and retry logic:

```python
from med_aug.llm import LLMService, ServiceConfig

config = ServiceConfig(
    provider_config=llm_config,
    cache_enabled=True,
    cache_ttl=3600,
    max_retries=3,
    retry_delay=1.0
)

service = LLMService(config)

# Single request with caching
response = await service.generate(prompt)

# Batch processing
prompts = ["prompt1", "prompt2", "prompt3"]
responses = await service.generate_batch(prompts, batch_size=10)
```

#### Response Caching
```python
from med_aug.llm import ResponseCache

cache = ResponseCache(
    cache_dir=Path("./cache"),
    ttl_seconds=3600,
    max_entries=1000
)

# Cache operations
cache_key = cache.get_key(prompt, system)
cached = cache.get(cache_key)

if not cached:
    response = await provider.generate(prompt)
    cache.set(cache_key, response)
```

#### Retry Logic
```python
from med_aug.llm import RetryHandler

retry_handler = RetryHandler(
    max_attempts=3,
    initial_delay=1.0,
    backoff_factor=2.0,
    max_delay=30.0
)

@retry_handler.with_retry
async def generate_with_retry(prompt: str):
    return await provider.generate(prompt)
```

### Prompt Templates (`templates.py`)

Template management for consistent prompting:

#### Built-in Templates
```python
from med_aug.llm import PromptTemplates

templates = PromptTemplates()

# Medication classification template
prompt = templates.render(
    "medication_classification",
    medication="pembrolizumab",
    context="NSCLC treatment"
)

# Drug interaction template
prompt = templates.render(
    "drug_interaction",
    drug1="pembrolizumab",
    drug2="carboplatin"
)
```

#### Custom Templates (with Jinja2)
```python
from med_aug.llm import TemplateManager

manager = TemplateManager(template_dir="./templates")

# Load custom template
template = manager.get_template("custom_classification.j2")
prompt = template.render(
    medication=medication,
    disease=disease,
    drug_classes=drug_classes
)
```

#### Template Examples
```python
# Classification template
CLASSIFICATION_TEMPLATE = """
Classify the following medication for {disease} treatment:

Medication: {medication}

Available drug classes:
{drug_classes}

Provide:
1. Most likely drug class
2. Confidence score (0-1)
3. Brief reasoning
"""

# Validation template
VALIDATION_TEMPLATE = """
Validate if "{medication}" is appropriate for {condition}.

Consider:
- FDA approval status
- Common usage patterns
- Safety profile
- Alternative names

Response format:
Valid: [yes/no]
Confidence: [0-1]
Reasoning: [brief explanation]
"""
```

### Medication Classifier (`classifier.py`)

Specialized classifier for medication analysis:

```python
from med_aug.llm import MedicationClassifier

classifier = MedicationClassifier(
    llm_service=service,
    disease_module=nsclc_module
)

# Single medication classification
result = await classifier.classify(
    medication="pembrolizumab",
    context="First-line treatment"
)

print(f"Drug class: {result.drug_class}")
print(f"Confidence: {result.confidence}")
print(f"Reasoning: {result.reasoning}")

# Batch classification
medications = ["pembrolizumab", "carboplatin", "paclitaxel"]
results = await classifier.classify_batch(
    medications,
    batch_size=10
)
```

#### Classification Result Structure
```python
@dataclass
class ClassificationResult:
    medication: str
    drug_class: str
    confidence: float
    reasoning: str
    raw_response: str
    processing_time: float
    metadata: Dict[str, Any]
```

## Provider Configuration

### Claude CLI Configuration
```yaml
llm:
  provider: claude_cli
  cli_command: claude  # or path to claude binary
  model: claude-3-haiku
  temperature: 0.0
  max_tokens: 2048
  timeout: 30
```

### API Provider Configuration
```yaml
llm:
  provider: anthropic
  api_key: ${ANTHROPIC_API_KEY}
  model: claude-3-opus
  temperature: 0.0
  max_tokens: 4096
  api_base: https://api.anthropic.com
  api_version: "2024-01-01"
```

### Environment Variables
```bash
# Provider selection
export MEDAUG_LLM_PROVIDER=claude_cli

# Claude CLI settings
export MEDAUG_CLAUDE_CLI_PATH=/usr/local/bin/claude
export MEDAUG_LLM_MODEL=claude-3-haiku

# API settings (when available)
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...

# Cache settings
export MEDAUG_LLM_CACHE_DIR=./llm_cache
export MEDAUG_LLM_CACHE_TTL=3600
```

## Usage Patterns

### Basic Classification
```python
from med_aug.llm import create_classifier

# Auto-configures from environment/config
classifier = create_classifier()

result = await classifier.classify("osimertinib")
```

### With Context
```python
result = await classifier.classify(
    medication="pembrolizumab",
    context={
        "disease": "NSCLC",
        "stage": "IV",
        "biomarkers": ["PD-L1 positive"],
        "line_of_therapy": 1
    }
)
```

### Confidence Thresholding
```python
results = await classifier.classify_batch(medications)

# Filter high-confidence results
high_confidence = [
    r for r in results
    if r.confidence >= 0.8
]

# Flag for review
needs_review = [
    r for r in results
    if 0.5 <= r.confidence < 0.8
]
```

### Custom Prompting
```python
# Override default template
custom_prompt = """
Analyze {medication} specifically for:
- Mechanism of action
- Drug class membership
- Clinical usage patterns
"""

result = await classifier.classify(
    medication="nivolumab",
    custom_prompt=custom_prompt
)
```

## Caching Strategy

### Multi-Level Cache
```python
# L1: Memory cache (fast, limited size)
# L2: Disk cache (persistent, larger)
# L3: Redis cache (shared, distributed)

cache_config = CacheConfig(
    memory_size=100,
    disk_path="./cache",
    redis_url="redis://localhost:6379",
    ttl_seconds=3600
)
```

### Cache Key Generation
```python
def generate_cache_key(prompt: str, system: str, params: dict) -> str:
    # Deterministic key generation
    content = f"{prompt}|{system}|{json.dumps(params, sort_keys=True)}"
    return hashlib.sha256(content.encode()).hexdigest()
```

### Cache Warming
```python
# Pre-populate cache with common medications
common_meds = load_common_medications()

for med in common_meds:
    await classifier.classify(med)  # Populates cache
```

## Error Handling

```python
from med_aug.llm import (
    LLMError,
    ProviderError,
    RateLimitError,
    TimeoutError,
    ParseError
)

try:
    result = await classifier.classify(medication)
except RateLimitError as e:
    # Handle rate limiting
    await asyncio.sleep(e.retry_after)
    result = await classifier.classify(medication)
except TimeoutError:
    # Use fallback classification
    result = rule_based_classify(medication)
except ParseError as e:
    # Log and use raw response
    logger.error(f"Parse error: {e}")
    result = ClassificationResult(
        medication=medication,
        drug_class="unknown",
        confidence=0.0,
        reasoning="Parse error"
    )
```

## Performance Optimization

### Batch Processing
```python
# Process medications in optimal batches
async def process_large_dataset(medications: List[str]):
    batch_size = 50  # Optimal for most providers

    results = []
    for i in range(0, len(medications), batch_size):
        batch = medications[i:i+batch_size]
        batch_results = await classifier.classify_batch(batch)
        results.extend(batch_results)

    return results
```

### Concurrent Requests
```python
# Parallel processing with rate limiting
semaphore = asyncio.Semaphore(5)  # Max 5 concurrent

async def classify_with_limit(med: str):
    async with semaphore:
        return await classifier.classify(med)

tasks = [classify_with_limit(med) for med in medications]
results = await asyncio.gather(*tasks)
```

### Response Streaming
```python
# For large responses
async for chunk in provider.generate_stream(prompt):
    process_chunk(chunk)
```

## Testing

### Unit Tests
```python
# Test with mock provider
def test_classification():
    provider = MockProvider(
        responses=["Drug class: PD-1 inhibitor\nConfidence: 0.95"]
    )
    classifier = MedicationClassifier(provider)

    result = await classifier.classify("pembrolizumab")
    assert result.drug_class == "PD-1 inhibitor"
    assert result.confidence == 0.95
```

### Integration Tests
```python
# Test with actual CLI
@pytest.mark.integration
async def test_claude_cli_integration():
    provider = ClaudeCLIProvider()
    response = await provider.generate("Test prompt")
    assert response.text
    assert response.latency > 0
```

## Monitoring

```python
from med_aug.llm import LLMMetrics

metrics = LLMMetrics()

# Track usage
metrics.record_request(
    provider="claude_cli",
    tokens=1500,
    latency=2.3,
    cache_hit=False
)

# Get statistics
stats = metrics.get_stats()
print(f"Total tokens: {stats.total_tokens}")
print(f"Cache hit rate: {stats.cache_hit_rate}")
print(f"Average latency: {stats.avg_latency}")
```

## Best Practices

1. **Always use caching** to reduce API calls and costs
2. **Implement retry logic** for transient failures
3. **Set appropriate timeouts** for long-running requests
4. **Use batch processing** for multiple items
5. **Monitor token usage** to control costs
6. **Validate responses** before processing
7. **Use templates** for consistent prompting
8. **Log all interactions** for debugging
9. **Handle errors gracefully** with fallbacks
10. **Keep prompts focused** and specific

## Migration Path

### From CLI to API
```python
# Current (CLI-based)
config = LLMConfig(provider="claude_cli")

# Future (API-based)
config = LLMConfig(
    provider="anthropic",
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

# Same interface, different backend
provider = create_provider(config)
```

## Future Enhancements Ideas

- OpenAI GPT integration
- Google Gemini support
- Local model support (Ollama, llama.cpp)
- Fine-tuning capabilities
- Prompt optimization with DSPy
- A/B testing framework
- Cost tracking and optimization
- Multi-model ensemble voting
- Automatic prompt engineering
- Response validation with schemas
