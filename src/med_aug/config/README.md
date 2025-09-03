# Config Module

## Overview

The config module manages configuration settings for the medication augmentation system. It provides centralized configuration management with support for multiple sources and validation.

## Structure

```
config/
├── __init__.py
├── settings.py         # Core settings and configuration classes
└── defaults.yaml       # Default configuration values
```

## Configuration Sources

Configuration can be loaded from multiple sources (in priority order):
1. **Command-line arguments** - Highest priority
2. **Environment variables** - `MEDAUG_*` prefix
3. **Configuration files** - YAML/JSON format
4. **Default values** - Built-in defaults

## Core Components

### Settings Class
```python
from med_aug.config import Settings

# Load configuration
settings = Settings.load()

# Access settings
settings.pipeline.batch_size
settings.llm.provider
settings.output.formats
```

### Configuration Schema

```yaml
# Pipeline Configuration
pipeline:
  batch_size: 100
  max_workers: 4
  checkpoint_interval: 1
  enable_caching: true
  timeout: 300

# Disease Module Configuration  
diseases:
  default_module: "nsclc"
  auto_discover: true
  module_path: "./diseases"

# LLM Configuration
llm:
  provider: "claude_cli"
  model: "claude-3-haiku"
  temperature: 0.0
  max_tokens: 2048
  retry_attempts: 3
  cache_responses: true

# Output Configuration
output:
  default_format: ["html", "json"]
  report_dir: "./reports"
  include_visualizations: true
  include_metrics: true

# Web Scraping Configuration
scraping:
  rate_limit: 1.0  # requests per second
  timeout: 30
  max_retries: 3
  cache_ttl: 86400  # 24 hours

# Logging Configuration
logging:
  level: "INFO"
  format: "json"
  log_file: null
  rotate: false
  max_bytes: 10485760  # 10MB
```

## Environment Variables

All settings can be overridden using environment variables:

```bash
# Pipeline settings
export MEDAUG_PIPELINE_BATCH_SIZE=200
export MEDAUG_PIPELINE_MAX_WORKERS=8

# LLM settings
export MEDAUG_LLM_PROVIDER=anthropic
export MEDAUG_LLM_API_KEY=sk-...

# Output settings
export MEDAUG_OUTPUT_REPORT_DIR=/custom/reports
export MEDAUG_OUTPUT_DEFAULT_FORMAT=excel,pdf

# Logging
export MEDAUG_LOGGING_LEVEL=DEBUG
export MEDAUG_LOGGING_LOG_FILE=/var/log/medaug.log
```

## Configuration Files

### YAML Format
```yaml
# medaug.config.yaml
pipeline:
  batch_size: 500
  enable_checkpoints: true

llm:
  provider: "anthropic"
  model: "claude-3-opus"
  
output:
  formats: ["html", "excel", "pdf"]
  include_raw_data: false
```

### JSON Format
```json
{
  "pipeline": {
    "batch_size": 500,
    "enable_checkpoints": true
  },
  "llm": {
    "provider": "anthropic",
    "model": "claude-3-opus"
  }
}
```

## Usage Examples

### Loading Configuration
```python
from med_aug.config import Settings

# Load with defaults
settings = Settings.load()

# Load from specific file
settings = Settings.load(config_file="custom.yaml")

# Override with dict
settings = Settings.load(overrides={
    "pipeline": {"batch_size": 1000},
    "llm": {"provider": "openai"}
})
```

### Accessing Settings
```python
# Direct access
batch_size = settings.pipeline.batch_size
provider = settings.llm.provider

# Get with default
timeout = settings.get("scraping.timeout", default=60)

# Check if set
if settings.has("llm.api_key"):
    # Use API provider
    pass
```

### Validating Configuration
```python
from med_aug.config import validate_config

# Validate configuration
errors = validate_config(settings)
if errors:
    print(f"Configuration errors: {errors}")
```

## Configuration Profiles

Support for multiple configuration profiles:

```yaml
# profiles.yaml
development:
  logging:
    level: DEBUG
  pipeline:
    batch_size: 10
    
production:
  logging:
    level: WARNING
  pipeline:
    batch_size: 1000
  output:
    formats: ["json"]
    
testing:
  llm:
    provider: "mock"
  pipeline:
    max_workers: 1
```

```python
# Load specific profile
settings = Settings.load(profile="production")
```

## Dynamic Configuration

Settings can be modified at runtime:

```python
# Update setting
settings.set("pipeline.batch_size", 500)

# Merge configurations
settings.merge({
    "output": {
        "formats": ["html", "pdf"],
        "report_dir": "/tmp/reports"
    }
})

# Save configuration
settings.save("current_config.yaml")
```

## Validation Rules

Configuration validation ensures:
- Required fields are present
- Values are within acceptable ranges
- File paths exist (where applicable)
- Mutually exclusive options aren't set
- Dependencies are satisfied

```python
# Validation schema example
VALIDATION_SCHEMA = {
    "pipeline.batch_size": {
        "type": int,
        "min": 1,
        "max": 10000
    },
    "llm.temperature": {
        "type": float,
        "min": 0.0,
        "max": 2.0
    },
    "output.formats": {
        "type": list,
        "choices": ["html", "pdf", "excel", "json", "markdown"]
    }
}
```

## Best Practices

1. **Use profiles** for different environments
2. **Store sensitive data** in environment variables
3. **Validate early** to catch configuration errors
4. **Document settings** with comments in YAML
5. **Version control** configuration files
6. **Use defaults** for optional settings
7. **Log configuration** at startup for debugging

## Integration with CLI

The CLI automatically loads configuration:

```python
# In CLI commands
@app.command()
def my_command(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None)
):
    # Configuration is loaded and available
    settings = ctx.obj.settings
    batch_size = settings.pipeline.batch_size
```

## Future Enhancements Ideas

- Configuration hot-reloading
- Remote configuration management
- Configuration encryption for sensitive data
- Configuration migration tools
- Web UI for configuration management
