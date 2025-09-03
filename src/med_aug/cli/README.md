# CLI Module

## Overview

The CLI (Command Line Interface) module provides a rich, user-friendly interface for interacting with the medication augmentation system. Built with Typer and Rich, it offers intuitive commands with terminal outputs.

## Structure

```
cli/
├── __init__.py
├── app.py              # Main CLI application
└── commands/           # Command groups
    ├── __init__.py
    ├── diseases.py     # Disease module commands
    └── pipeline.py     # Pipeline execution commands
```

## Main Application (`app.py`)

The main CLI application that:
- Configures the Typer app with Rich formatting
- Registers command groups (diseases, pipeline)
- Sets up logging configuration
- Provides global options (--debug, --config, --quiet)
- Displays welcome banner and help text

## Command Groups

### Diseases Commands (`commands/diseases.py`)
Manage disease modules and configurations:
- `list` - List available disease modules
- `info <name>` - Show detailed information about a disease module
- `validate <name>` - Validate a disease module configuration
- `keywords <name>` - Display drug keywords for a disease
- `export <name>` - Export disease configuration to YAML

### Pipeline Commands (`commands/pipeline.py`)
Execute and manage the augmentation pipeline:
- `run <file>` - Run the full pipeline on a data file
- `status <id>` - Check status of a pipeline run
- `list` - List all pipeline checkpoints
- `clean` - Clean old pipeline checkpoints
- `analyze <file>` - Analyze a file to identify medication columns
- `extract <file> <column>` - Extract medications from a specific column

## Usage Examples

```bash
# Basic usage
med-aug --help

# Enable debug logging
med-aug --debug pipeline run data.csv

# Run with custom config
med-aug --config custom.yaml pipeline run data.csv

# Disease module management
med-aug diseases list
med-aug diseases info nsclc
med-aug diseases keywords nsclc

# Pipeline execution
med-aug pipeline run data.csv --disease nsclc --llm
med-aug pipeline status abc123
med-aug pipeline analyze data.csv
```

## Key Features

### Rich Terminal Output
- Colored and formatted text
- Progress bars and spinners
- Tables for structured data
- Panels for important information
- Syntax highlighting for code/data

### Command Options
- Type hints for all parameters
- Automatic help generation
- Option validation
- Default values
- Environment variable support

### Error Handling
- Graceful error messages
- Helpful suggestions
- Debug mode for detailed output
- Exit codes for scripting

## Global Options

- `--version/-v` - Show version information
- `--config/-c PATH` - Specify configuration file
- `--debug/-d` - Enable debug logging
- `--log-file PATH` - Write logs to file
- `--quiet/-q` - Suppress non-essential output
- `--help/-h` - Show help message

## Configuration

The CLI can be configured through:
1. Command-line options (highest priority)
2. Configuration file (--config)
3. Environment variables
4. Default values

## Extension

To add new commands:

1. Create a new command module in `commands/`
2. Define commands using Typer decorators
3. Register the command group in `app.py`

Example:
```python
# commands/custom.py
import typer
app = typer.Typer()

@app.command()
def my_command(arg: str):
    """My custom command."""
    pass

# In app.py
from .commands import custom
app.add_typer(custom.app, name="custom")
```

## Dependencies

- **typer**: Modern CLI framework
- **rich**: Terminal formatting and styling
- **click**: Underlying CLI library (via Typer)

## Best Practices

1. **Use Rich for output** - Provides better user experience
2. **Add progress indicators** - For long-running operations
3. **Validate inputs early** - Fail fast with helpful messages
4. **Provide examples** - In help text and documentation
5. **Use consistent naming** - Follow conventions across commands
6. **Handle interrupts** - Graceful shutdown on Ctrl+C
7. **Return proper exit codes** - 0 for success, non-zero for errors
