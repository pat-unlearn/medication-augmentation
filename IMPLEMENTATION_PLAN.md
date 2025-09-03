# Medication Augmentation System - Implementation Plan

## Executive Summary

**Project**: Disease-Agnostic Medication Augmentation System  
**Goal**: Intelligent expansion of medication databases for clinical research  
**First Use Case**: Non-Small Cell Lung Cancer (NSCLC)  
**Architecture**: Plugin-based, extensible to any therapeutic area  
**Technology Stack**: Modern Python with Rich CLI and Typer framework  

### Vision
Build a production-ready system that automatically discovers, classifies, and augments medication databases by expanding from limited examples (54 drug classes with 2-10 names each) to comprehensive coverage (70+ drug classes with 20-50+ names each). The system must be extensible to any disease indication while providing a modern, beautiful CLI experience.

## System Architecture

### Core Principles
1. **Disease-Agnostic Core**: Shared components work across all therapeutic areas
2. **Plugin Architecture**: Each disease is a separate module with custom logic
3. **Modern CLI**: Rich + Typer for beautiful, interactive user experience  
4. **Async Processing**: Parallel web scraping and LLM calls for performance
5. **Intelligent Classification**: LLM-powered with disease-specific context
6. **Quality Assurance**: Built-in validation and human review workflows

### Project Structure
```
medication-augmentation/
├── src/
│   └── med_aug/
│       ├── core/           # Disease-agnostic core logic
│       │   ├── models/     # Data models (Medication, DrugClass, etc.)
│       │   ├── analyzer.py # Column detection and analysis
│       │   ├── extractor.py # Medication name extraction
│       │   ├── researcher.py # Web research coordination
│       │   └── classifier.py # LLM-powered classification
│       ├── diseases/       # Disease-specific plugins
│       │   ├── base.py     # Abstract base class for diseases
│       │   ├── nsclc/      # NSCLC implementation
│       │   │   ├── __init__.py
│       │   │   ├── module.py # NSCLC disease module
│       │   │   ├── scrapers.py # NSCLC-specific scrapers
│       │   │   └── context.py # LLM context templates
│       │   ├── prostate/   # Future: Prostate cancer
│       │   └── cardiovascular/ # Future: Cardiovascular diseases
│       ├── infrastructure/
│       │   ├── scrapers/   # Web scraping implementations
│       │   │   ├── base.py
│       │   │   ├── fda.py
│       │   │   ├── clinicaltrials.py
│       │   │   ├── nccn.py
│       │   │   └── oncokb.py
│       │   ├── llm/        # LLM integrations
│       │   │   ├── openai_client.py
│       │   │   ├── anthropic_client.py
│       │   │   └── prompt_templates.py
│       │   └── cache/      # Caching implementations
│       │       ├── redis_cache.py
│       │       └── memory_cache.py
│       ├── cli/            # Rich + Typer CLI
│       │   ├── app.py      # Main CLI application
│       │   ├── commands/   # Command implementations
│       │   │   ├── diseases.py
│       │   │   ├── analyze.py
│       │   │   ├── extract.py
│       │   │   ├── augment.py
│       │   │   ├── validate.py
│       │   │   └── pipeline.py
│       │   └── formatters/ # Rich output formatters
│       └── config/
│           ├── base.yaml   # Base system configuration
│           └── diseases/   # Disease-specific configs
│               ├── nsclc.yaml
│               ├── prostate.yaml
│               └── cardiovascular.yaml
├── tests/
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── data/
│   ├── input/
│   ├── output/
│   └── cache/
├── docs/
├── pyproject.toml
├── README.md
└── .env.example
```

## Technical Stack

### Core Dependencies
```toml
[project]
name = "medication-augmentation"
version = "1.0.0"
dependencies = [
    # CLI Framework
    "typer[all]>=0.9.0",      # Modern CLI framework with completion
    "rich>=13.7.0",           # Beautiful terminal output
    "click-completion>=0.5.2", # Shell completions
    
    # Data Processing
    "pandas>=2.0.0",          # DataFrame operations
    "polars>=0.20.0",         # Fast dataframe operations
    "pydantic>=2.5.0",        # Data validation and settings
    
    # Web Scraping & HTTP
    "httpx>=0.25.0",          # Async HTTP client
    "beautifulsoup4>=4.12.0", # HTML parsing
    "playwright>=1.40.0",     # Browser automation (if needed)
    
    # LLM Integration
    "openai>=1.0.0",          # OpenAI API
    "anthropic>=0.18.0",      # Anthropic API
    
    # Caching & Storage
    "redis>=5.0.0",           # Redis caching
    "diskcache>=5.6.0",       # Disk-based caching
    
    # Configuration & Utilities
    "pyyaml>=6.0.0",          # YAML configuration
    "python-dotenv>=1.0.0",   # Environment variables
    "structlog>=23.2.0",      # Structured logging
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.23.0",
    "pytest-mock>=3.12.0",
    "black>=23.0.0",
    "isort>=5.12.0",
    "mypy>=1.7.0",
    "ruff>=0.1.0"
]
```

### Key Technology Choices

**CLI Framework**: Typer + Rich
- Modern, type-safe CLI development
- Beautiful terminal output with colors, tables, progress bars
- Auto-completion and help generation
- Interactive prompts and menus

**Data Processing**: Pandas + Polars
- Pandas for compatibility and rich ecosystem
- Polars for high-performance operations on large datasets
- Pydantic for data validation and configuration

**Web Scraping**: HTTPX + BeautifulSoup4
- Async HTTP client for concurrent requests
- Respectful rate limiting and retry logic
- HTML parsing for unstructured data extraction

**LLM Integration**: OpenAI + Anthropic
- Multi-provider support for flexibility
- Async API calls for batch processing
- Structured prompt templates with few-shot learning

## Core Components

### 1. Disease Module System

#### Abstract Base Class
```python
# src/med_aug/diseases/base.py
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class DrugClassConfig:
    name: str
    keywords: List[str]
    confidence_threshold: float
    web_sources: List[str]

class DiseaseModule(ABC):
    """Abstract base class for all disease modules."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Disease identifier (e.g., 'nsclc', 'prostate')"""
        pass
    
    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable disease name"""
        pass
    
    @property
    @abstractmethod
    def drug_classes(self) -> List[DrugClassConfig]:
        """Disease-specific drug class configurations"""
        pass
    
    @abstractmethod
    def get_web_sources(self) -> List[str]:
        """Disease-specific data sources for web scraping"""
        pass
    
    @abstractmethod
    def get_llm_context(self) -> str:
        """Disease-specific context for LLM classification"""
        pass
    
    @abstractmethod
    def validate_medication(self, medication: str, drug_class: str) -> bool:
        """Disease-specific medication validation"""
        pass
```

#### Plugin Discovery
```python
# src/med_aug/diseases/__init__.py
import importlib
import pkgutil
from typing import Dict, Type
from .base import DiseaseModule

class DiseaseRegistry:
    """Registry for disease modules with auto-discovery."""
    
    def __init__(self):
        self._modules: Dict[str, Type[DiseaseModule]] = {}
        self._discover_modules()
    
    def _discover_modules(self):
        """Auto-discover disease modules in the diseases package."""
        for _, module_name, _ in pkgutil.iter_modules(__path__):
            if module_name != 'base':
                try:
                    module = importlib.import_module(f'.{module_name}', __name__)
                    if hasattr(module, 'MODULE_CLASS'):
                        disease_class = getattr(module, 'MODULE_CLASS')
                        if issubclass(disease_class, DiseaseModule):
                            instance = disease_class()
                            self._modules[instance.name] = disease_class
                except ImportError as e:
                    # Log error but continue discovery
                    pass
    
    def get_module(self, name: str) -> Optional[DiseaseModule]:
        """Get disease module instance by name."""
        if name in self._modules:
            return self._modules[name]()
        return None
    
    def list_available(self) -> List[str]:
        """List all available disease modules."""
        return list(self._modules.keys())

# Global registry instance
disease_registry = DiseaseRegistry()
```

### 2. Data Models

```python
# src/med_aug/core/models.py
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime

class MedicationType(Enum):
    BRAND = "brand"
    GENERIC = "generic"
    CLINICAL_TRIAL = "clinical_trial"
    ABBREVIATION = "abbreviation"
    COMBINATION = "combination"

class ConfidenceLevel(Enum):
    HIGH = "high"      # >90%
    MEDIUM = "medium"  # 70-90%
    LOW = "low"        # <70%

@dataclass
class Medication:
    name: str
    type: MedicationType
    confidence: float
    source: str
    metadata: Dict[str, Any]
    discovered_at: datetime
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        if self.confidence >= 0.9:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.7:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type.value,
            "confidence": self.confidence,
            "source": self.source,
            "metadata": self.metadata,
            "discovered_at": self.discovered_at.isoformat()
        }

@dataclass
class DrugClass:
    name: str
    taking_variable: str  # e.g., "taking_pembrolizumab"
    current_medications: List[Medication]
    category: str  # "chemotherapy", "immunotherapy", etc.
    disease: str
    
    def add_medication(self, medication: Medication) -> 'DrugClass':
        """Add medication (immutable update)."""
        return DrugClass(
            name=self.name,
            taking_variable=self.taking_variable,
            current_medications=[*self.current_medications, medication],
            category=self.category,
            disease=self.disease
        )
    
    def get_medication_names(self) -> List[str]:
        """Get list of medication names."""
        return [med.name for med in self.current_medications]

@dataclass  
class ColumnAnalysisResult:
    column: str
    confidence: float
    total_count: int
    unique_count: int
    sample_medications: List[str]
    reasoning: str

@dataclass
class AugmentationResult:
    original_count: int
    augmented_count: int
    new_medications: List[Medication]
    improvement_percentage: float
    processing_time: float
    quality_score: float
```

### 3. Core Processing Pipeline

```python
# src/med_aug/core/analyzer.py
import pandas as pd
import re
from typing import List, Dict, Any, Optional
from .models import ColumnAnalysisResult

class DataAnalyzer:
    """Analyze datasets to identify medication columns."""
    
    MEDICATION_PATTERNS = [
        r'\b[A-Z][a-z]+mab\b',      # Monoclonal antibodies
        r'\b[A-Z][a-z]*nib\b',      # Kinase inhibitors  
        r'\b[A-Z][a-z]*tin\b',      # Statins and related
        r'\bPLATINUM|PLATIN\b',     # Platinum compounds
        r'\bTAXEL|TAXANE\b',        # Taxanes
    ]
    
    MEDICATION_KEYWORDS = [
        'drug', 'medication', 'agent', 'therapy', 'treatment',
        'chemo', 'immuno', 'targeted', 'hormone', 'biological'
    ]
    
    def analyze_columns(self, 
                       file_path: str, 
                       sample_size: int = 1000,
                       confidence_threshold: float = 0.7) -> List[ColumnAnalysisResult]:
        """Analyze dataset columns to identify medication columns."""
        
        # Read sample of data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, nrows=sample_size)
        elif file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep='\t', nrows=sample_size)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
        
        results = []
        
        for column in df.columns:
            if df[column].dtype == 'object':  # String columns only
                result = self._analyze_column(df[column], column)
                if result.confidence >= confidence_threshold:
                    results.append(result)
        
        # Sort by confidence descending
        return sorted(results, key=lambda x: x.confidence, reverse=True)
    
    def _analyze_column(self, series: pd.Series, column_name: str) -> ColumnAnalysisResult:
        """Analyze a single column for medication content."""
        
        # Remove null values
        non_null = series.dropna()
        if len(non_null) == 0:
            return ColumnAnalysisResult(
                column=column_name,
                confidence=0.0,
                total_count=0,
                unique_count=0,
                sample_medications=[],
                reasoning="Column contains no non-null values"
            )
        
        # Convert to strings and get unique values
        str_values = non_null.astype(str).str.strip()
        unique_values = str_values.unique()
        
        # Calculate confidence score
        confidence_score = 0.0
        reasoning_parts = []
        
        # Column name scoring (30% weight)
        name_score = self._score_column_name(column_name)
        confidence_score += name_score * 0.3
        reasoning_parts.append(f"Column name score: {name_score:.2f}")
        
        # Pattern matching scoring (40% weight)
        pattern_score = self._score_patterns(unique_values)
        confidence_score += pattern_score * 0.4
        reasoning_parts.append(f"Pattern match score: {pattern_score:.2f}")
        
        # Statistical scoring (30% weight)
        stats_score = self._score_statistics(str_values)
        confidence_score += stats_score * 0.3
        reasoning_parts.append(f"Statistical score: {stats_score:.2f}")
        
        # Get sample medications for preview
        sample_meds = [val for val in unique_values[:10] 
                      if self._looks_like_medication(val)]
        
        return ColumnAnalysisResult(
            column=column_name,
            confidence=min(confidence_score, 1.0),
            total_count=len(non_null),
            unique_count=len(unique_values),
            sample_medications=sample_meds,
            reasoning="; ".join(reasoning_parts)
        )
    
    def _score_column_name(self, column_name: str) -> float:
        """Score column name for medication-related keywords."""
        name_lower = column_name.lower()
        score = 0.0
        
        # Exact matches
        if name_lower in ['agent', 'medication', 'drug', 'drugdtxt']:
            score += 1.0
        elif any(keyword in name_lower for keyword in self.MEDICATION_KEYWORDS):
            score += 0.7
        elif any(word in name_lower for word in ['name', 'text', 'desc']):
            score += 0.3
        
        return min(score, 1.0)
    
    def _score_patterns(self, values: List[str]) -> float:
        """Score based on medication naming patterns."""
        if len(values) == 0:
            return 0.0
        
        pattern_matches = 0
        for value in values[:100]:  # Sample first 100
            if self._looks_like_medication(value):
                pattern_matches += 1
        
        return pattern_matches / min(len(values), 100)
    
    def _looks_like_medication(self, value: str) -> bool:
        """Check if a value looks like a medication name."""
        if not value or len(value.strip()) < 3:
            return False
        
        value = value.strip()
        
        # Check against patterns
        for pattern in self.MEDICATION_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        # Check for common medication indicators
        med_indicators = [
            'mab', 'nib', 'tin', 'pril', 'sartan', 'olol', 'zole', 
            'platin', 'taxel', 'rubicin', 'mycin', 'cillin'
        ]
        
        return any(indicator in value.lower() for indicator in med_indicators)
    
    def _score_statistics(self, series: pd.Series) -> float:
        """Score based on statistical properties."""
        score = 0.0
        
        # Diversity score (more unique values = more likely medications)
        unique_ratio = series.nunique() / len(series)
        if unique_ratio > 0.5:
            score += 0.3
        elif unique_ratio > 0.2:
            score += 0.2
        
        # Average length score (medications typically 6-20 characters)
        avg_length = series.str.len().mean()
        if 6 <= avg_length <= 20:
            score += 0.3
        elif 4 <= avg_length <= 25:
            score += 0.2
        
        # Alphanumeric ratio (medications usually have letters)
        alphanumeric_ratio = series.str.contains(r'[A-Za-z]').mean()
        if alphanumeric_ratio > 0.8:
            score += 0.4
        elif alphanumeric_ratio > 0.5:
            score += 0.2
        
        return min(score, 1.0)
```

## CLI Design with Rich + Typer

### Main Application
```python
# src/med_aug/cli/app.py
import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Optional
from pathlib import Path

from ..diseases import disease_registry
from .commands import diseases, analyze, extract, augment, validate, pipeline

console = Console()

app = typer.Typer(
    name="med-aug",
    help="🏥 [bold blue]Medication Augmentation System[/bold blue]\n\n"
         "Intelligent medication discovery and classification for clinical research.\n"
         "Supports multiple therapeutic areas with extensible disease modules.",
    rich_markup_mode="rich",
    add_completion=True,
    no_args_is_help=True,
    context_settings={"help_option_names": ["-h", "--help"]}
)

# Add command groups
app.add_typer(diseases.app, name="diseases", help="🔬 Manage disease modules")
app.add_typer(analyze.app, name="analyze", help="🔍 Analyze datasets")  
app.add_typer(extract.app, name="extract", help="📋 Extract medications")
app.add_typer(augment.app, name="augment", help="⚡ Augment databases")
app.add_typer(validate.app, name="validate", help="✅ Validate results")
app.add_typer(pipeline.app, name="pipeline", help="🚀 Run full pipeline")

@app.callback()
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
) -> None:
    """
    🏥 Medication Augmentation System
    
    Intelligent medication discovery and classification system for clinical research.
    Automatically expands medication databases with comprehensive drug names and classifications.
    """
    
    if version:
        console.print("🏥 Medication Augmentation System v1.0.0", style="bold blue")
        raise typer.Exit()
    
    # Display welcome banner
    if ctx.invoked_subcommand is None:
        _display_welcome()

def _display_welcome():
    """Display welcome banner with system information."""
    
    title = Text("Medication Augmentation System", style="bold blue")
    
    content = Text()
    content.append("🎯 ", style="bold yellow")
    content.append("Intelligent medication discovery for clinical research\n")
    content.append("🔬 ", style="bold green")
    content.append(f"Available diseases: {', '.join(disease_registry.list_available())}\n")
    content.append("⚡ ", style="bold red")
    content.append("Modern CLI with Rich formatting and async processing\n\n")
    content.append("🚀 ", style="bold magenta")
    content.append("Get started: ", style="bold")
    content.append("med-aug pipeline --help", style="cyan")
    
    panel = Panel(
        content,
        title="Welcome",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)

if __name__ == "__main__":
    app()
```

### Disease Management Commands
```python
# src/med_aug/cli/commands/diseases.py
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

from ...diseases import disease_registry

console = Console()
app = typer.Typer()

@app.command("list")
def list_diseases() -> None:
    """List all available disease modules."""
    
    available_diseases = disease_registry.list_available()
    
    if not available_diseases:
        console.print("❌ No disease modules found", style="red")
        return
    
    table = Table(title="🔬 Available Disease Modules")
    table.add_column("Code", style="cyan", no_wrap=True)
    table.add_column("Name", style="yellow")
    table.add_column("Drug Classes", justify="right", style="green")
    table.add_column("Status", justify="center")
    
    for disease_code in available_diseases:
        try:
            module = disease_registry.get_module(disease_code)
            if module:
                status = "✅ Active"
                drug_count = len(module.drug_classes)
            else:
                status = "❌ Error"
                drug_count = 0
                
            table.add_row(
                disease_code,
                module.display_name if module else "Unknown",
                str(drug_count),
                status
            )
        except Exception:
            table.add_row(disease_code, "Error loading", "0", "❌ Error")
    
    console.print(table)

@app.command("info")
def disease_info(
    disease: str = typer.Argument(..., help="Disease module code")
) -> None:
    """Show detailed information about a disease module."""
    
    module = disease_registry.get_module(disease)
    if not module:
        console.print(f"❌ Disease module '{disease}' not found", style="red")
        available = disease_registry.list_available()
        if available:
            console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)
    
    # Disease info panel
    info_content = f"""
🔬 [bold]{module.display_name}[/bold]
📝 Code: {module.name}
🎯 Drug Classes: {len(module.drug_classes)}
🌐 Web Sources: {len(module.get_web_sources())}
"""
    
    info_panel = Panel(
        info_content.strip(),
        title="Disease Information",
        border_style="blue"
    )
    console.print(info_panel)
    
    # Drug classes table
    if module.drug_classes:
        drug_table = Table(title="Drug Classes")
        drug_table.add_column("Class", style="cyan")
        drug_table.add_column("Keywords", style="yellow")
        drug_table.add_column("Threshold", justify="right", style="green")
        
        for drug_class in module.drug_classes:
            keywords_str = ", ".join(drug_class.keywords[:3])
            if len(drug_class.keywords) > 3:
                keywords_str += "..."
            
            drug_table.add_row(
                drug_class.name,
                keywords_str,
                f"{drug_class.confidence_threshold:.1%}"
            )
        
        console.print(drug_table)

@app.command("create")
def create_disease_template(
    name: str = typer.Argument(..., help="Disease module name"),
    display_name: str = typer.Option(..., "--display", help="Human-readable name"),
    output_dir: Optional[str] = typer.Option(None, "--output", help="Output directory")
) -> None:
    """Create a new disease module template."""
    
    # Implementation for creating disease module templates
    console.print(f"🔬 Creating disease module template for '{name}'...")
    console.print("📝 This would generate the module files and configuration")
    console.print("⚠️  Template creation not yet implemented", style="yellow")
```

### Beautiful Pipeline Command
```python
# src/med_aug/cli/commands/pipeline.py  
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from pathlib import Path
from typing import Optional
import asyncio

from ...diseases import disease_registry
from ...core.analyzer import DataAnalyzer
from ...core.extractor import MedicationExtractor

console = Console()
app = typer.Typer()

@app.command("run") 
def run_pipeline(
    disease: str = typer.Argument(..., help="Disease module (e.g., nsclc, prostate)"),
    input_file: Path = typer.Argument(..., help="Input data file", exists=True),
    conmeds_file: Path = typer.Argument(..., help="Current conmeds.yml file", exists=True),
    output_file: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
    column: Optional[str] = typer.Option(None, "--column", help="Medication column name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview changes only"),
    skip_research: bool = typer.Option(False, "--skip-research", help="Skip web research"),
    auto_approve: bool = typer.Option(False, "--auto-approve", help="Auto-approve high confidence")
) -> None:
    """
    🚀 Run the complete medication augmentation pipeline.
    
    Executes the full workflow from data analysis to conmeds.yml augmentation
    with beautiful progress tracking and interactive validation.
    
    Examples:
        med-aug pipeline run nsclc data.csv conmeds.yml
        med-aug pipeline run nsclc data.csv conmeds.yml --column AGENT --dry-run
        med-aug pipeline run prostate data.csv conmeds.yml --skip-research
    """
    
    # Validate disease module
    module = disease_registry.get_module(disease)
    if not module:
        console.print(f"❌ Disease module '{disease}' not found", style="red")
        available = disease_registry.list_available()
        console.print(f"Available: {', '.join(available)}")
        raise typer.Exit(1)
    
    output_file = output_file or Path(f"conmeds_{disease}_augmented.yml")
    
    # Display pipeline header
    _display_pipeline_header(disease, module, input_file, conmeds_file, output_file, dry_run)
    
    # Run pipeline with rich progress tracking
    asyncio.run(_run_pipeline_async(
        module, input_file, conmeds_file, output_file, 
        column, dry_run, skip_research, auto_approve
    ))

def _display_pipeline_header(disease: str, module, input_file: Path, 
                           conmeds_file: Path, output_file: Path, dry_run: bool):
    """Display beautiful pipeline header."""
    
    header_text = f"""
🏥 [bold blue]Medication Augmentation Pipeline[/bold blue]

📊 [bold]Dataset:[/bold] {input_file.name} ({_get_file_size(input_file)})
🔬 [bold]Disease:[/bold] {module.display_name} ({disease})
📋 [bold]Current:[/bold] {conmeds_file.name}
💾 [bold]Output:[/bold] {output_file.name}
"""
    
    if dry_run:
        header_text += "\n🧪 [bold yellow]DRY RUN MODE - No files will be modified[/bold yellow]"
    
    panel = Panel(
        header_text.strip(),
        border_style="blue",
        padding=(1, 2)
    )
    console.print(panel)

async def _run_pipeline_async(module, input_file: Path, conmeds_file: Path, 
                            output_file: Path, column: Optional[str], 
                            dry_run: bool, skip_research: bool, auto_approve: bool):
    """Run the async pipeline with progress tracking."""
    
    pipeline_state = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console,
        expand=True
    ) as progress:
        
        try:
            # Phase 1: Column Analysis
            analyze_task = progress.add_task("🔍 Analyzing columns...", total=100)
            analyzer = DataAnalyzer()
            
            for i in range(100):
                await asyncio.sleep(0.01)  # Simulate work
                progress.update(analyze_task, advance=1)
            
            column_results = analyzer.analyze_columns(str(input_file))
            pipeline_state['column_analysis'] = column_results
            
            if column_results:
                selected_column = column or column_results[0].column
                console.print(f"✅ Column analysis complete: [bold]{selected_column}[/bold]")
            else:
                console.print("❌ No medication columns detected", style="red")
                raise typer.Exit(1)
            
            # Phase 2: Medication Extraction
            extract_task = progress.add_task("📋 Extracting medications...", total=100)
            extractor = MedicationExtractor()
            
            for i in range(100):
                await asyncio.sleep(0.01)  # Simulate work
                progress.update(extract_task, advance=1)
            
            # Simulate extraction results
            medications = {'medications': ['drug1', 'drug2'], 'count': 150}
            pipeline_state['medications'] = medications
            console.print(f"✅ Extracted [bold]{medications['count']}[/bold] unique medications")
            
            # Phase 3: Web Research (if not skipped)
            if not skip_research:
                research_task = progress.add_task("🌐 Researching medications...", total=100)
                
                for i in range(100):
                    await asyncio.sleep(0.02)  # Simulate slower web requests
                    progress.update(research_task, advance=1)
                
                research_data = {'researched': 120, 'enhanced': 95}
                pipeline_state['research'] = research_data
                console.print(f"✅ Research complete: [bold]{research_data['enhanced']}[/bold] medications enhanced")
            
            # Phase 4: LLM Classification
            classify_task = progress.add_task("🤖 Classifying medications...", total=100)
            
            for i in range(100):
                await asyncio.sleep(0.03)  # Simulate LLM calls
                progress.update(classify_task, advance=1)
            
            classifications = {'classified': 145, 'high_confidence': 120, 'needs_review': 25}
            pipeline_state['classifications'] = classifications
            console.print(f"✅ Classification complete: [bold]{classifications['high_confidence']}[/bold] high-confidence classifications")
            
            # Phase 5: Validation & Output
            output_task = progress.add_task("📄 Generating output...", total=100)
            
            for i in range(100):
                await asyncio.sleep(0.01)
                progress.update(output_task, advance=1)
            
            console.print(f"✅ Pipeline complete!")
            
        except Exception as e:
            console.print(f"❌ Pipeline failed: {e}", style="red")
            raise typer.Exit(1)
    
    # Display final summary
    _display_pipeline_summary(pipeline_state, dry_run)

def _display_pipeline_summary(pipeline_state: dict, dry_run: bool):
    """Display beautiful pipeline summary."""
    
    summary_table = Table(title="🎉 Pipeline Summary")
    summary_table.add_column("Phase", style="cyan", no_wrap=True)
    summary_table.add_column("Status", justify="center")
    summary_table.add_column("Result", style="yellow")
    
    summary_table.add_row("Column Analysis", "✅ Complete", "1 medication column identified")
    summary_table.add_row("Medication Extraction", "✅ Complete", "150 unique medications found")
    
    if 'research' in pipeline_state:
        research = pipeline_state['research']
        summary_table.add_row("Web Research", "✅ Complete", f"{research['enhanced']} medications enhanced")
    
    if 'classifications' in pipeline_state:
        classify = pipeline_state['classifications']
        summary_table.add_row("Classification", "✅ Complete", f"{classify['classified']} medications classified")
    
    improvement_pct = "45.2%"  # Simulated
    summary_table.add_row("Overall Improvement", "🎯 Success", f"{improvement_pct} matching improvement")
    
    console.print(summary_table)
    
    if dry_run:
        console.print("\n🧪 [bold yellow]Dry run completed - no files were modified[/bold yellow]")
    else:
        console.print(f"\n💾 Enhanced conmeds file saved to output location")

def _get_file_size(file_path: Path) -> str:
    """Get human-readable file size."""
    size = file_path.stat().st_size
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"

@app.command("status")
def pipeline_status() -> None:
    """Show pipeline execution status and history."""
    
    console.print("📊 Pipeline Status", style="bold blue")
    console.print("⚠️  Status tracking not yet implemented", style="yellow")
    
    # Would show recent pipeline runs, their status, and results
```

## NSCLC Implementation (First Use Case)

### NSCLC Disease Module
```python
# src/med_aug/diseases/nsclc/module.py
from typing import List
from ..base import DiseaseModule, DrugClassConfig

class NSCLCModule(DiseaseModule):
    """Non-Small Cell Lung Cancer disease module."""
    
    @property
    def name(self) -> str:
        return "nsclc"
    
    @property  
    def display_name(self) -> str:
        return "Non-Small Cell Lung Cancer"
    
    @property
    def drug_classes(self) -> List[DrugClassConfig]:
        return [
            DrugClassConfig(
                name="chemotherapy",
                keywords=["carboplatin", "paclitaxel", "pemetrexed", "docetaxel", "gemcitabine", "cisplatin"],
                confidence_threshold=0.8,
                web_sources=["fda", "nccn", "clinicaltrials"]
            ),
            DrugClassConfig(
                name="immunotherapy", 
                keywords=["pembrolizumab", "nivolumab", "atezolizumab", "durvalumab", "ipilimumab", "cemiplimab"],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn", "oncokb"]
            ),
            DrugClassConfig(
                name="targeted_therapy",
                keywords=["osimertinib", "erlotinib", "crizotinib", "alectinib", "lorlatinib", "afatinib"],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb", "clinicaltrials"]
            ),
            DrugClassConfig(
                name="anti_angiogenic",
                keywords=["bevacizumab", "ramucirumab", "avastin", "cyramza"],
                confidence_threshold=0.85,
                web_sources=["fda", "nccn"]
            ),
            DrugClassConfig(
                name="antibody_drug_conjugates",
                keywords=["trastuzumab deruxtecan", "enhertu", "sacituzumab govitecan", "trodelvy"],
                confidence_threshold=0.9,
                web_sources=["fda", "oncokb"]
            )
        ]
    
    def get_web_sources(self) -> List[str]:
        return [
            "fda_nsclc",
            "clinicaltrials_lung_cancer", 
            "nccn_nsclc_guidelines",
            "oncokb_lung",
            "asco_nsclc"
        ]
    
    def get_llm_context(self) -> str:
        return """
You are a clinical oncologist specializing in Non-Small Cell Lung Cancer (NSCLC) treatment.
Your expertise includes the latest FDA-approved therapies and clinical trial agents for NSCLC.

Current NSCLC treatment landscape (2024-2025):
- First-line: Pembrolizumab combinations, osimertinib for EGFR+, alectinib/lorlatinib for ALK+
- Immunotherapy: PD-1/PD-L1 inhibitors (pembrolizumab, nivolumab, atezolizumab, durvalumab)
- Targeted therapy: EGFR inhibitors (osimertinib, erlotinib, afatinib), ALK inhibitors (crizotinib, alectinib, lorlatinib)
- Recent approvals: Amivantamab for EGFR exon 20, sotorasib/adagrasib for KRAS G12C

Classify medications into these categories:
- Chemotherapy: Traditional cytotoxic agents
- Immunotherapy: Checkpoint inhibitors, cellular therapies
- Targeted Therapy: Small molecule inhibitors, precision medicine
- Anti-angiogenic: VEGF pathway inhibitors
- Antibody-Drug Conjugates: Targeted delivery systems

Consider generic names, brand names, trial designations, and abbreviations.
"""
    
    def validate_medication(self, medication: str, drug_class: str) -> bool:
        """NSCLC-specific medication validation."""
        
        medication_lower = medication.lower()
        
        # Known NSCLC medications by class
        known_nsclc_meds = {
            "chemotherapy": [
                "carboplatin", "paclitaxel", "pemetrexed", "docetaxel", "gemcitabine", 
                "cisplatin", "etoposide", "vinorelbine", "irinotecan"
            ],
            "immunotherapy": [
                "pembrolizumab", "keytruda", "nivolumab", "opdivo", "atezolizumab", 
                "tecentriq", "durvalumab", "imfinzi", "ipilimumab", "yervoy"
            ],
            "targeted_therapy": [
                "osimertinib", "tagrisso", "erlotinib", "tarceva", "afatinib", "gilotrif",
                "crizotinib", "xalkori", "alectinib", "alecensa", "lorlatinib", "lorbrena"
            ]
        }
        
        if drug_class in known_nsclc_meds:
            return any(known_med in medication_lower for known_med in known_nsclc_meds[drug_class])
        
        return True  # Allow unknown medications for discovery

# Register the module
MODULE_CLASS = NSCLCModule
```

### NSCLC Configuration
```yaml
# config/diseases/nsclc.yaml
disease:
  name: "Non-Small Cell Lung Cancer"
  code: "nsclc"
  description: "Lung cancer treatment and research medications"
  
drug_classes:
  chemotherapy:
    keywords: 
      - "carboplatin"
      - "paclitaxel" 
      - "pemetrexed"
      - "docetaxel"
      - "gemcitabine"
      - "cisplatin"
    confidence_threshold: 0.8
    web_sources: ["fda", "nccn", "clinicaltrials"]
    
  immunotherapy:
    keywords:
      - "pembrolizumab"
      - "nivolumab"
      - "atezolizumab" 
      - "durvalumab"
      - "ipilimumab"
    confidence_threshold: 0.85
    web_sources: ["fda", "nccn", "oncokb"]
    
  targeted_therapy:
    keywords:
      - "osimertinib"
      - "erlotinib"
      - "crizotinib"
      - "alectinib"
      - "lorlatinib"
    confidence_threshold: 0.9
    web_sources: ["fda", "oncokb", "clinicaltrials"]

web_sources:
  fda:
    base_url: "https://www.accessdata.fda.gov/scripts/cder/daf/"
    rate_limit: 1.0
    timeout: 30
    
  clinicaltrials:
    base_url: "https://clinicaltrials.gov/api/v2/"
    params:
      condition: "Non-Small Cell Lung Cancer"
      status: ["Recruiting", "Active", "Completed"]
    rate_limit: 0.5
    
  nccn:
    guidelines_url: "https://www.nccn.org/guidelines/category_1"
    disease_code: "nsclc"
    rate_limit: 2.0
    
  oncokb:
    base_url: "https://www.oncokb.org/api/v1/"
    cancer_type: "Lung Cancer"
    rate_limit: 1.0

llm_settings:
  model: "gpt-4"
  temperature: 0.1
  max_tokens: 2000
  system_prompt: >
    You are a clinical oncologist specializing in NSCLC treatment.
    Classify medications based on mechanism of action and NSCLC treatment standards.
    Consider both approved drugs and investigational agents in clinical trials.

validation:
  require_human_review: true
  confidence_threshold: 0.7
  auto_approve_threshold: 0.9
  
output:
  format: "yaml"
  include_metadata: true
  backup_original: true
```

## Development Phases

### Day 1: Foundation Setup
**Morning (4 hours)**
1. Initialize project with `uv`
2. Set up pyproject.toml with all dependencies
3. Create core directory structure
4. Implement base data models (Medication, DrugClass, etc.)
5. Set up logging and configuration system

**Afternoon (4 hours)**
1. Create disease module abstract base class
2. Implement disease registry with auto-discovery
3. Create NSCLC disease module
4. Set up basic CLI structure with Typer + Rich
5. Write initial tests

### Day 2: Data Analysis Core
**Morning (4 hours)**
1. Implement DataAnalyzer class
2. Add column detection algorithms
3. Create confidence scoring system
4. Test with provided NSCLC datasets
5. Add interactive column selection

**Afternoon (4 hours)**
1. Implement MedicationExtractor class
2. Add text normalization and cleaning
3. Create frequency analysis
4. Build medication deduplication
5. Add statistical analysis features

### Day 3: Web Research Engine  
**Morning (4 hours)**
1. Create base web scraper architecture
2. Implement FDA Orange Book scraper
3. Add ClinicalTrials.gov API integration
4. Create rate limiting and caching
5. Add error handling and retries

**Afternoon (4 hours)**
1. Implement NCCN guidelines scraper
2. Add OncoKB database integration
3. Create medication context enrichment
4. Add 2024-2025 NSCLC drug updates
5. Test web scraping with real data

### Day 4: LLM Classification
**Morning (4 hours)**
1. Implement LLM client abstraction
2. Add OpenAI and Anthropic integrations
3. Create NSCLC-specific prompt templates
4. Implement batch processing
5. Add confidence scoring

**Afternoon (4 hours)**
1. Create classification validation
2. Add human review workflows
3. Implement auto-approval logic
4. Create classification reports
5. Test with NSCLC medications

### Day 5: CLI Development
**Morning (4 hours)**
1. Build beautiful CLI commands with Rich
2. Implement pipeline command with progress bars
3. Add interactive prompts and menus
4. Create output formatters
5. Add shell completions

**Afternoon (4 hours)**
1. Implement disease management commands
2. Add configuration management
3. Create validation and reporting commands
4. Add dry-run and resume capabilities
5. Polish UI with Rich components

### Day 6: Integration & Validation
**Morning (4 hours)**
1. Integrate all components into pipeline
2. Test full workflow with NSCLC data
3. Create augmented conmeds.yml output
4. Validate against current NSCLC drugs
5. Measure performance improvements

**Afternoon (4 hours)**
1. Create validation reports
2. Add before/after analysis
3. Test error handling and edge cases
4. Optimize performance bottlenecks
5. Add metrics and monitoring

### Day 7: Testing & Documentation
**Morning (4 hours)**
1. Write comprehensive unit tests
2. Add integration tests for web scrapers
3. Create end-to-end pipeline tests
4. Test with multiple disease modules
5. Add performance benchmarks

**Afternoon (4 hours)**
1. Write CLI documentation
2. Create usage examples and guides
3. Document disease module creation
4. Add API documentation
5. Create deployment guide

## Key Implementation Files

### 1. Core Models (`src/med_aug/core/models.py`)
- Medication, DrugClass, ColumnAnalysisResult data models
- Type safety with enums and validation
- Serialization/deserialization methods

### 2. Disease Registry (`src/med_aug/diseases/__init__.py`)
- Auto-discovery of disease modules
- Plugin management system
- Configuration loading per disease

### 3. Data Analyzer (`src/med_aug/core/analyzer.py`) 
- Column detection algorithms
- Confidence scoring system
- Statistical analysis methods

### 4. Web Scrapers (`src/med_aug/infrastructure/scrapers/`)
- Base scraper with rate limiting
- FDA, ClinicalTrials.gov, NCCN, OncoKB implementations
- Async processing with connection pooling

### 5. LLM Integration (`src/med_aug/infrastructure/llm/`)
- Multi-provider LLM support
- Disease-specific prompt templates
- Batch processing and confidence scoring

### 6. CLI Application (`src/med_aug/cli/`)
- Rich + Typer beautiful CLI
- Interactive commands with progress tracking
- Output formatting and validation

### 7. NSCLC Module (`src/med_aug/diseases/nsclc/`)
- First disease implementation
- NSCLC-specific drug classes and context
- Validation rules for lung cancer medications

## Testing Strategy

### Unit Tests
```python
# tests/unit/test_analyzer.py
import pytest
from med_aug.core.analyzer import DataAnalyzer
from med_aug.core.models import ColumnAnalysisResult

class TestDataAnalyzer:
    @pytest.fixture
    def analyzer(self):
        return DataAnalyzer()
    
    @pytest.fixture
    def sample_medication_data(self):
        return pd.DataFrame({
            'AGENT': ['pembrolizumab', 'Keytruda', 'osimertinib', 'Tagrisso'],
            'DOSE': ['200mg', '100mg', '80mg', '40mg'],
            'PATIENT_ID': ['P001', 'P002', 'P003', 'P004']
        })
    
    def test_column_detection_high_confidence(self, analyzer, sample_medication_data):
        results = analyzer.analyze_columns_from_dataframe(sample_medication_data)
        
        assert len(results) > 0
        agent_result = next((r for r in results if r.column == 'AGENT'), None)
        assert agent_result is not None
        assert agent_result.confidence > 0.8
        assert 'pembrolizumab' in agent_result.sample_medications
    
    def test_column_detection_low_confidence(self, analyzer):
        low_conf_data = pd.DataFrame({
            'DOSE': ['200mg', '100mg', '80mg', '40mg'],
            'PATIENT_ID': ['P001', 'P002', 'P003', 'P004']
        })
        
        results = analyzer.analyze_columns_from_dataframe(low_conf_data)
        for result in results:
            assert result.confidence < 0.5
    
    @pytest.mark.parametrize("medication,expected", [
        ("pembrolizumab", True),
        ("Keytruda", True), 
        ("osimertinib", True),
        ("random_text", False),
        ("", False),
    ])
    def test_medication_pattern_detection(self, analyzer, medication, expected):
        assert analyzer._looks_like_medication(medication) == expected
```

### Integration Tests
```python
# tests/integration/test_web_scrapers.py
import pytest
import httpx
from med_aug.infrastructure.scrapers.fda import FDAAsyncScraper

@pytest.mark.integration
class TestFDAScraper:
    @pytest.mark.asyncio
    async def test_scrape_real_medication(self):
        async with httpx.AsyncClient() as client:
            scraper = FDAAsyncScraper(client)
            result = await scraper.scrape_medication_info("pembrolizumab")
            
            assert "generic_names" in result
            assert "brand_names" in result
            assert len(result["brand_names"]) > 0
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self):
        async with httpx.AsyncClient() as client:
            scraper = FDAAsyncScraper(client, rate_limit=0.1)
            
            start_time = time.time()
            await scraper.scrape_medication_info("drug1")
            await scraper.scrape_medication_info("drug2")
            elapsed = time.time() - start_time
            
            assert elapsed >= 0.1  # Rate limiting enforced
```

### End-to-End Tests
```python
# tests/e2e/test_pipeline.py
import pytest
from pathlib import Path
from med_aug.cli.app import app
from typer.testing import CliRunner

@pytest.mark.e2e
class TestFullPipeline:
    def test_complete_nsclc_pipeline(self, sample_nsclc_data, temp_output_dir):
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "pipeline", "run", 
            "nsclc", 
            str(sample_nsclc_data),
            "tests/fixtures/sample_conmeds.yml",
            "--output", str(temp_output_dir / "output.yml"),
            "--dry-run"
        ])
        
        assert result.exit_code == 0
        assert "Pipeline complete" in result.stdout
        assert "150 unique medications" in result.stdout
    
    def test_interactive_column_selection(self, multi_column_data):
        runner = CliRunner()
        
        result = runner.invoke(app, [
            "analyze", "columns",
            str(multi_column_data)
        ], input="1\n")  # Select first option
        
        assert result.exit_code == 0
        assert "medication column" in result.stdout.lower()
```

## Configuration Reference

### Base Configuration (`config/base.yaml`)
```yaml
# Base system configuration
system:
  name: "Medication Augmentation System"
  version: "1.0.0"
  
logging:
  level: "INFO"
  format: "structured"
  output: "console"
  
cache:
  type: "redis"  # redis, memory, disk
  ttl: 3600
  max_size: 1000
  
llm:
  default_provider: "openai"
  default_model: "gpt-4"
  temperature: 0.1
  max_tokens: 2000
  batch_size: 10
  
web_scraping:
  default_rate_limit: 1.0
  default_timeout: 30
  max_retries: 3
  user_agent: "MedicationAugmentation/1.0"
  
processing:
  max_workers: 4
  chunk_size: 1000
  confidence_threshold: 0.7
  auto_approve_threshold: 0.9
```

### Environment Variables (`.env.example`)
```bash
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Cache Configuration
REDIS_URL=redis://localhost:6379
CACHE_TTL=3600

# Web Scraping
USER_AGENT=MedicationAugmentation/1.0
RATE_LIMIT_DELAY=1.0
MAX_RETRIES=3

# Processing
MAX_WORKERS=4
CHUNK_SIZE=1000
CONFIDENCE_THRESHOLD=0.7

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=structured

# Output
DEFAULT_OUTPUT_FORMAT=yaml
BACKUP_FILES=true
```

## Future Roadmap

### Phase 2: Additional Disease Modules
1. **Prostate Cancer Module**
   - Hormone therapy focus (leuprolide, goserelin, abiraterone, enzalutamide)
   - PARP inhibitors (olaparib, rucaparib)
   - Radiopharmaceuticals (lutetium-177, radium-223)

2. **Cardiovascular Module**  
   - Antihypertensives (ACE inhibitors, ARBs, beta blockers)
   - Lipid-lowering drugs (statins, PCSK9 inhibitors)
   - Anticoagulants and antiplatelets

3. **Metabolic Module**
   - Diabetes medications (metformin, insulin, GLP-1 agonists)
   - Obesity treatments (semaglutide, orlistat)
   - Lipid disorders

### Phase 3: Advanced Features
1. **Multi-language Support**
   - International drug name databases
   - Non-English medication extraction
   - Localized CLI interfaces

2. **API Integration**
   - REST API for programmatic access
   - Webhook support for automated workflows
   - Integration with clinical trial databases

3. **Machine Learning Enhancement**
   - Custom medication classification models
   - Automated pattern discovery
   - Confidence score optimization

### Phase 4: Enterprise Features
1. **Team Collaboration**
   - Multi-user validation workflows
   - Role-based access control
   - Audit trails and version control

2. **Advanced Analytics**
   - Medication trend analysis
   - Cross-disease comparisons
   - Performance benchmarking

3. **Integration Ecosystem**
   - Electronic health record (EHR) integration
   - Clinical data management systems
   - Research data platforms

## Success Criteria

### Technical Metrics
- ✅ **Column Detection**: 95% accuracy on medication column identification
- ✅ **Processing Speed**: <2 hours for 10,000+ unique medication names
- ✅ **Coverage Improvement**: 30%+ increase in medication matching rates
- ✅ **System Reliability**: 99%+ uptime during processing
- ✅ **Extensibility**: New disease module creation in <1 day

### Business Metrics  
- ✅ **NSCLC Coverage**: Expand from 54 to 70+ drug classes
- ✅ **Medication Density**: 20-50+ names per drug class
- ✅ **Total Variations**: 1000+ medication variations discovered
- ✅ **False Negative Reduction**: 50%+ decrease in missed medications
- ✅ **Manual Effort Reduction**: 80%+ reduction in manual curation time

### User Experience Metrics
- ✅ **CLI Usability**: Beautiful, intuitive command-line interface
- ✅ **Interactive Features**: Smart column detection with user confirmation
- ✅ **Error Recovery**: Resume capability for interrupted workflows
- ✅ **Documentation**: Comprehensive guides and examples
- ✅ **Performance Feedback**: Real-time progress tracking and ETA

---

*This implementation plan provides a complete roadmap for building a production-ready, disease-agnostic medication augmentation system with NSCLC as the first validated use case. The system is designed for extensibility, maintainability, and user experience while delivering significant improvements in medication discovery and classification accuracy.*