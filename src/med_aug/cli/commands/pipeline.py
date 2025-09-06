"""Pipeline execution commands for the CLI."""

import asyncio
import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...core.analyzer import DataAnalyzer
from ...core.extractor import MedicationExtractor
from ...core.logging import get_logger
from ...pipeline import PipelineOrchestrator, PipelineConfig
from ...pipeline.checkpoint import CheckpointManager

logger = get_logger(__name__)
console = Console()
app = typer.Typer()


@app.command("run")
def run_pipeline(
    input_file: Path = typer.Argument(..., help="Input clinical data file"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output directory"
    ),
    conmeds_file: Optional[Path] = typer.Option(
        None, "--conmeds", help="Existing conmeds_defaults.yml to augment"
    ),
    disease: str = typer.Option(
        "nsclc",
        "--disease",
        "-d",
        help="Disease module to use (nsclc, breast_cancer, etc.)",
    ),
    confidence: float = typer.Option(
        0.5, "--confidence", "-c", help="Confidence threshold"
    ),
    no_web: bool = typer.Option(False, "--no-web", help="Disable web research"),
    no_validation: bool = typer.Option(
        False, "--no-validation", help="Disable validation"
    ),
    disable_llm: bool = typer.Option(
        False, "--no-llm", help="Disable LLM classification"
    ),
    llm_provider: str = typer.Option(
        "claude_cli", "--llm-provider", help="LLM provider to use"
    ),
    enable_evaluation: bool = typer.Option(
        False, "--evaluate", help="Enable comprehensive evaluation with LLM assistance"
    ),
    no_checkpoints: bool = typer.Option(
        False, "--no-checkpoints", help="Disable checkpoints"
    ),
    resume_from: Optional[str] = typer.Option(
        None, "--resume", help="Resume from phase"
    ),
    pipeline_id: Optional[str] = typer.Option(
        None, "--id", help="Pipeline ID for resume"
    ),
):
    """Run the medication augmentation pipeline for any disease indication."""

    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)

    # Validate conmeds file if provided
    if conmeds_file and not conmeds_file.exists():
        console.print(f"[red]Error: Conmeds file not found: {conmeds_file}[/red]")
        raise typer.Exit(1)

    # Set default conmeds file if not provided
    if conmeds_file is None:
        conmeds_file = Path("data/conmeds_defaults.yml")
        if not conmeds_file.exists():
            console.print(
                f"[yellow]Warning: Default conmeds file not found: {conmeds_file}[/yellow]"
            )
            console.print(
                "[yellow]Pipeline will create new conmeds.yml without augmenting existing entries[/yellow]"
            )

    # Set output directory with timestamped subdirectory in results/
    if output_dir is None:
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"run_{timestamp}_{Path(input_file).stem}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate pipeline ID first so we can include it in logs
    import uuid

    generated_pipeline_id = str(uuid.uuid4())[:8]

    # Set up file logging by adding a file handler to existing loggers
    log_file = output_dir / "pipeline.log"

    # Create file handler and add to both standard logging and structlog
    import logging

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)

    # Custom formatter for clean, human-readable logs
    class HumanReadableFormatter(logging.Formatter):
        def __init__(self, pipeline_id=None):
            super().__init__()
            self.pipeline_id = pipeline_id

        def format(self, record):
            # Get timestamp and module name
            timestamp = self.formatTime(record, "%Y-%m-%d %H:%M:%S")
            # Use specific module paths instead of generic names like 'phases'
            if record.name:
                # Remove 'med_aug.' prefix and use the remaining path for better specificity
                if record.name.startswith("med_aug."):
                    module = record.name.replace("med_aug.", "")
                else:
                    module = record.name
            else:
                module = "unknown"
            msg = record.getMessage()

            # Add pipeline ID prefix if available
            pipeline_prefix = f"[{self.pipeline_id}] " if self.pipeline_id else ""

            # Handle structlog messages - extract key information cleanly
            if "[info     ]" in msg and "=" in msg:
                try:
                    # Parse: [info     ] event_name param1=value1 param2=value2
                    parts = msg.split("] ", 1)[1]  # Get everything after "] "
                    event_name = parts.split()[0]  # First word is event name

                    # Format important events nicely
                    if event_name == "executing_phase":
                        phase = self._get_param(parts, "phase")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | ▶️  Starting: {phase}"

                    elif event_name == "phase_completed":
                        phase = self._get_param(parts, "phase")
                        duration = self._get_param(parts, "duration")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | ✅ Completed: {phase} ({duration})"

                    elif event_name == "llm_normalization_response":
                        med = self._get_param(parts, "medication")
                        valid = self._get_param(parts, "is_valid")
                        disease_specific = self._get_param(parts, "is_disease_specific")
                        confidence = self._get_param(parts, "confidence")
                        batch_num = self._get_param(parts, "batch_num")
                        raw_response = self._get_param(parts, "raw_llm_response")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   📊 Batch {batch_num}: {med} → valid={valid}, cancer_drug={disease_specific}, confidence={confidence}\n{timestamp} | {pipeline_prefix}{module:<35} |      Raw LLM: {raw_response[:100]}..."

                    elif event_name == "oncology_drug_accepted":
                        med = self._get_param(parts, "medication")
                        generic = self._get_param(parts, "generic_name")
                        variants = self._get_param(parts, "variants")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   ✅ ACCEPTED: {med} → {generic} ({variants} variants)"

                    elif event_name == "processing_batch":
                        batch_num = self._get_param(parts, "batch_num")
                        meds_count = self._get_param(parts, "medications")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   🔬 Processing batch {batch_num} with {meds_count} medications"

                    elif event_name == "llm_batch_request_started":
                        batch_num = self._get_param(parts, "batch_num")
                        concurrent = self._get_param(parts, "concurrent_calls")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   🤖 Sending batch {batch_num} to LLM ({concurrent} concurrent calls)"

                    elif event_name == "llm_batch_request_completed":
                        batch_num = self._get_param(parts, "batch_num")
                        responses = self._get_param(parts, "responses_received")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   ✅ LLM batch {batch_num} completed - {responses} responses received"

                    elif event_name == "batch_completed":
                        batch_num = self._get_param(parts, "batch_num")
                        valid_drugs = self._get_param(parts, "valid_drugs")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   🔄 Batch {batch_num} complete - {valid_drugs} valid drugs"

                    elif event_name == "medication_accepted_for_normalization":
                        input_med = self._get_param(parts, "input_medication")
                        generic = self._get_param(parts, "generic_name")
                        confidence = self._get_param(parts, "confidence")
                        is_disease = self._get_param(parts, "is_disease_specific")
                        batch_num = self._get_param(parts, "batch_num")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   ✅ ACCEPTED Batch {batch_num}: '{input_med}' → {generic} (confidence: {confidence}, disease-specific: {is_disease})"

                    elif event_name == "medication_rejected_from_normalization":
                        input_med = self._get_param(parts, "input_medication")
                        confidence = self._get_param(parts, "confidence")
                        is_valid = self._get_param(parts, "is_valid")
                        is_disease = self._get_param(parts, "is_disease_specific")
                        batch_num = self._get_param(parts, "batch_num")
                        reasoning = self._get_param(parts, "reasoning")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} |   ❌ REJECTED Batch {batch_num}: '{input_med}' (valid: {is_valid}, disease: {is_disease}, conf: {confidence})\n{timestamp} | {pipeline_prefix}{module:<35} |      Reason: {reasoning[:100]}..."

                    # Handle detailed data logging events
                    elif event_name == "dataset_loaded":
                        rows = self._get_param(parts, "rows")
                        cols = self._get_param(parts, "columns")
                        memory = self._get_param(parts, "memory_usage_mb")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | 📊 Dataset loaded: {rows} rows, {cols} columns, {memory}MB memory"

                    elif event_name == "dataset_sample":
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | 🔍 Sample data logged for debugging"

                    elif event_name == "column_identified_as_medication":
                        col = self._get_param(parts, "column")
                        conf = self._get_param(parts, "confidence")
                        unique = self._get_param(parts, "unique_values")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | ✅ Column '{col}' identified as medication (confidence: {conf}, unique values: {unique})"

                    elif event_name == "columns_rejected_as_medication":
                        cols = self._get_param(parts, "columns")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | ❌ Columns rejected: {cols}"

                    elif event_name == "extracted_from_column":
                        col = self._get_param(parts, "column")
                        meds = self._get_param(parts, "medications_found")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | 🧬 Extracted {meds} medications from column '{col}'"

                    elif event_name == "all_unique_medications_extracted":
                        count = self._get_param(parts, "unique_count")
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | 📋 Total unique medications extracted: {count}"

                    # Clean up other common events
                    elif event_name in [
                        "data_ingestion_completed",
                        "column_analysis_completed",
                        "medication_extraction_started",
                    ]:
                        clean_name = event_name.replace("_", " ").title()
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | 📝 {clean_name}"

                    elif event_name == "pipeline_initialized":
                        return f"{timestamp} | {pipeline_prefix}{module:<35} | 🚀 Pipeline initialized"

                    else:
                        # For other events, add appropriate emojis and context
                        clean_name = event_name.replace("_", " ")

                        # Add context for LLM service operations
                        if module in ["llm.service", "llm.providers"]:
                            # Try to extract medication or batch info from the context
                            medication = self._get_param(
                                parts, "medication"
                            ) or self._get_param(parts, "med")
                            batch_info = self._get_param(parts, "batch_num")

                            if event_name == "llm_generation_started":
                                if medication and medication != "?":
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |     🤖 Normalizing: {medication}"
                                elif batch_info and batch_info != "?":
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |     🤖 Processing batch {batch_info}"
                                else:
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |     🤖 LLM normalization started"

                            elif event_name == "llm_generation_completed":
                                if medication and medication != "?":
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |     ✅ Completed: {medication}"
                                elif batch_info and batch_info != "?":
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |     ✅ Batch {batch_info} completed"
                                else:
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |     ✅ LLM normalization completed"

                            elif event_name == "claude_cli_generation_started":
                                # Show detailed Claude CLI start with prompt preview
                                prompt_preview = self._get_param(
                                    parts, "prompt_preview"
                                )
                                model = self._get_param(parts, "model")
                                if medication and medication != "?":
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |       🔧 Claude CLI ({model}) starting: {medication}\n{timestamp} | {pipeline_prefix}{module:<35} |          Prompt: {prompt_preview}"
                                else:
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |       🔧 Claude CLI ({model}) started"

                            elif event_name == "claude_cli_generation_completed":
                                # Show response preview for debugging
                                response_preview = self._get_param(
                                    parts, "response_preview"
                                )
                                response_length = self._get_param(
                                    parts, "response_length"
                                )
                                model = self._get_param(parts, "model")
                                if medication and medication != "?":
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |       ✅ Claude CLI ({model}) completed: {medication} ({response_length} chars)\n{timestamp} | {pipeline_prefix}{module:<35} |          Response: {response_preview[:150]}..."
                                else:
                                    return f"{timestamp} | {pipeline_prefix}{module:<35} |       ✅ Claude CLI ({model}) completed ({response_length} chars)"

                        if "started" in clean_name:
                            return f"{timestamp} | {pipeline_prefix}{module:<35} | 🚦 {clean_name.title()}"
                        elif "completed" in clean_name:
                            return f"{timestamp} | {pipeline_prefix}{module:<35} | ✅ {clean_name.title()}"
                        elif "saved" in clean_name or "created" in clean_name:
                            return f"{timestamp} | {pipeline_prefix}{module:<35} | 💾 {clean_name.title()}"
                        elif "failed" in clean_name or "error" in clean_name:
                            return f"{timestamp} | {pipeline_prefix}{module:<35} | ❌ {clean_name.title()}"
                        else:
                            return f"{timestamp} | {pipeline_prefix}{module:<35} | 📝 {clean_name.title()}"

                except (IndexError, AttributeError):
                    pass  # Fall through to default formatting

            # Default formatting for regular log messages - preserve level but remove redundancy
            level_emoji = {
                "INFO": "ℹ️",
                "WARNING": "⚠️",
                "ERROR": "❌",
                "DEBUG": "🐛",
            }.get(record.levelname, "📝")

            # Clean up any remaining [info     ] patterns
            clean_msg = (
                msg.replace("[info     ]", "")
                .replace("[warning  ]", "")
                .replace("[error    ]", "")
                .strip()
            )

            # Ensure every message has context emoji
            if not level_emoji:
                level_emoji = "📝"

            return f"{timestamp} | {pipeline_prefix}{module:<35} | {level_emoji} {clean_msg}"

        def _get_param(self, text, param_name):
            """Extract parameter value from 'param=value' in text"""
            try:
                start = text.find(f"{param_name}=") + len(f"{param_name}=")
                if start == len(f"{param_name}=") - 1:  # param not found
                    return "?"

                # Find the end of the value (next space or end of string)
                end = text.find(" ", start)
                if end == -1:
                    return text[start:].strip()
                return text[start:end].strip()
            except:
                return "?"

    formatter = HumanReadableFormatter(pipeline_id=generated_pipeline_id)

    # Set the formatter on the file handler
    file_handler.setFormatter(formatter)

    # Add handler only to the med_aug logger to avoid duplication
    med_aug_logger = logging.getLogger("med_aug")
    med_aug_logger.addHandler(file_handler)
    med_aug_logger.setLevel(logging.INFO)  # Override CLI's ERROR level

    # Prevent propagation to avoid duplicate messages
    med_aug_logger.propagate = False

    # Test logging
    logger = get_logger(__name__)
    logger.info("pipeline_logging_initialized", log_file=str(log_file))
    console.print(f"[dim]📝 Logs will be written to: {log_file}[/dim]")

    # Create configuration
    config = PipelineConfig(
        input_file=str(input_file),
        output_path=str(output_dir),
        conmeds_file=(
            str(conmeds_file) if conmeds_file and conmeds_file.exists() else None
        ),
        disease_module=disease,
        confidence_threshold=confidence,
        enable_web_research=not no_web,
        enable_validation=not no_validation,
        enable_llm_classification=not disable_llm,
        llm_provider=llm_provider,
        enable_evaluation=enable_evaluation,
        enable_checkpoints=not no_checkpoints,
        display_progress=True,
        progress_mode="rich",
    )

    # Display configuration
    console.print(
        Panel(
            f"[bold blue]Pipeline Configuration[/bold blue]\n\n"
            f"Clinical Data: {input_file}\n"
            f"Conmeds File: {conmeds_file if conmeds_file and conmeds_file.exists() else '❌ None (creating new)'}\n"
            f"Output: {output_dir}\n"
            f"Disease: {disease}\n"
            f"Confidence: {confidence}\n"
            f"Web Research: {'✅' if not no_web else '❌'}\n"
            f"Validation: {'✅' if not no_validation else '❌'}\n"
            f"LLM Classification: {'✅' if not disable_llm else '❌'}\n"
            f"LLM Provider: {llm_provider if not disable_llm else 'N/A'}\n"
            f"Checkpoints: {'✅' if not no_checkpoints else '❌'}",
            title="Starting Pipeline",
        )
    )

    # Create orchestrator using the generated pipeline ID
    orchestrator = PipelineOrchestrator(config, generated_pipeline_id)

    # Run pipeline
    try:
        logger.info(
            "pipeline_run_started",
            input_file=str(input_file),
            output_dir=str(output_dir),
        )

        # Run async pipeline
        report = asyncio.run(orchestrator.run(resume_from))

        # Display results
        if report.status == "Completed":
            console.print("\n[green]✅ Pipeline completed successfully![/green]")

            # Show metrics
            metrics = orchestrator.get_metrics()
            if metrics:
                console.print("\n[bold]Pipeline Metrics:[/bold]")
                for key, value in metrics.items():
                    # Format floating point numbers to 4 decimal places
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    console.print(f"  {key}: {formatted_value}")

            # Show artifacts
            artifacts = orchestrator.get_artifacts()
            if artifacts:
                console.print("\n[bold]Generated Files:[/bold]")
                for artifact in artifacts:
                    console.print(f"  📄 {artifact}")
        else:
            console.print(f"\n[red]❌ Pipeline failed: {report.status}[/red]")

            # Show failed phases
            for phase_name, progress in report.phase_progress.items():
                if progress.error:
                    console.print(f"  {phase_name}: {progress.error}")

        logger.info(
            "pipeline_run_completed",
            status=report.status,
            duration=str(report.duration),
        )

    except Exception as e:
        console.print(f"\n[red]Pipeline error: {e}[/red]")
        logger.error("pipeline_run_failed", error=str(e))
        raise typer.Exit(1)


@app.command("status")
def pipeline_status(
    pipeline_id: str = typer.Argument(..., help="Pipeline ID to check"),
):
    """Check status of a pipeline."""

    checkpoint_manager = CheckpointManager()
    checkpoint = checkpoint_manager.load_checkpoint(pipeline_id)

    if not checkpoint:
        console.print(f"[red]No checkpoint found for pipeline: {pipeline_id}[/red]")
        raise typer.Exit(1)

    # Display checkpoint info
    console.print(
        Panel(
            f"[bold blue]Pipeline Status[/bold blue]\n\n"
            f"Pipeline ID: {checkpoint.pipeline_id}\n"
            f"Timestamp: {checkpoint.timestamp}\n"
            f"Current Phase: {checkpoint.current_phase}\n"
            f"Completed Phases: {', '.join(checkpoint.completed_phases)}",
            title="Checkpoint Information",
        )
    )

    # Show phase results
    if checkpoint.phase_results:
        table = Table(title="Phase Results")
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="yellow")

        for phase_name, result in checkpoint.phase_results.items():
            status = result.get("status", "unknown")
            duration = result.get("duration_seconds", "-")
            if isinstance(duration, (int, float)):
                duration = f"{duration:.1f}s"

            table.add_row(phase_name, status, str(duration))

        console.print(table)


@app.command("list")
def list_pipelines():
    """List all available pipeline checkpoints."""

    checkpoint_manager = CheckpointManager()
    checkpoints = checkpoint_manager.list_checkpoints()

    if not checkpoints:
        console.print("[yellow]No pipeline checkpoints found.[/yellow]")
        return

    table = Table(title="Pipeline Checkpoints")
    table.add_column("Pipeline ID", style="cyan")
    table.add_column("Timestamp", style="magenta")
    table.add_column("Size", style="yellow")
    table.add_column("Modified", style="green")

    for checkpoint in checkpoints:
        size_kb = checkpoint["size_bytes"] / 1024
        table.add_row(
            checkpoint["pipeline_id"],
            checkpoint["timestamp"],
            f"{size_kb:.1f} KB",
            checkpoint["modified"],
        )

    console.print(table)
    console.print(f"\nTotal: {len(checkpoints)} checkpoints")


@app.command("clean")
def clean_checkpoints(
    days: int = typer.Option(
        7, "--days", "-d", help="Delete checkpoints older than N days"
    ),
    confirm: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation"),
):
    """Clean old pipeline checkpoints."""

    if not confirm:
        confirm_clean = typer.confirm(f"Delete checkpoints older than {days} days?")
        if not confirm_clean:
            console.print("[yellow]Cleanup cancelled.[/yellow]")
            return

    checkpoint_manager = CheckpointManager()
    deleted = checkpoint_manager.cleanup_old_checkpoints(days)

    if deleted > 0:
        console.print(f"[green]✅ Deleted {deleted} old checkpoint(s)[/green]")
    else:
        console.print("[yellow]No old checkpoints to delete.[/yellow]")


@app.command("analyze")
def analyze_file(
    input_file: Path = typer.Argument(..., help="File to analyze"),
    threshold: float = typer.Option(
        0.5, "--threshold", "-t", help="Confidence threshold"
    ),
):
    """Analyze a file to identify medication columns."""
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Analyzing: {input_file}[/bold]")

    try:
        # Load file
        if input_file.suffix.lower() == ".csv":
            df = pd.read_csv(input_file, nrows=1000)
        elif input_file.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(input_file, nrows=1000)
        elif input_file.suffix.lower() == ".sas7bdat":
            df = pd.read_sas(input_file, format="sas7bdat")
            # Limit rows for analysis performance
            if len(df) > 1000:
                df = df.head(1000)
        else:
            console.print(f"[red]Unsupported file format: {input_file.suffix}[/red]")
            console.print(
                "[yellow]Supported formats: .csv, .xlsx, .xls, .sas7bdat[/yellow]"
            )
            raise typer.Exit(1)

        console.print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

        # Analyze
        analyzer = DataAnalyzer()
        results = analyzer.analyze_dataframe(df, threshold)

        if not results:
            console.print("[yellow]No medication columns found.[/yellow]")
            return

        # Display results
        table = Table(title="Column Analysis Results")
        table.add_column("Column", style="cyan")
        table.add_column("Confidence", style="magenta")
        table.add_column("Unique Values", style="yellow")
        table.add_column("Sample Medications", style="green")

        for result in results:
            samples = ", ".join(result.sample_medications[:3])
            if len(result.sample_medications) > 3:
                samples += "..."

            table.add_row(
                result.column,
                f"{result.confidence:.2f}",
                str(result.unique_count),
                samples,
            )

        console.print(table)

    except Exception as e:
        console.print(f"[red]Error analyzing file: {e}[/red]")
        raise typer.Exit(1)


@app.command("extract")
def extract_medications(
    input_file: Path = typer.Argument(..., help="File to extract from"),
    column: str = typer.Argument(..., help="Column name to extract from"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file"),
):
    """Extract medications from a specific column."""
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)

    try:
        # Load file
        if input_file.suffix.lower() == ".csv":
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() in [".xlsx", ".xls"]:
            df = pd.read_excel(input_file)
        elif input_file.suffix.lower() == ".sas7bdat":
            df = pd.read_sas(input_file, format="sas7bdat")
        else:
            console.print(f"[red]Unsupported file format: {input_file.suffix}[/red]")
            console.print(
                "[yellow]Supported formats: .csv, .xlsx, .xls, .sas7bdat[/yellow]"
            )
            raise typer.Exit(1)

        if column not in df.columns:
            console.print(f"[red]Column '{column}' not found in file[/red]")
            console.print(f"Available columns: {', '.join(df.columns)}")
            raise typer.Exit(1)

        # Extract medications
        extractor = MedicationExtractor()
        result = extractor.extract_from_series(df[column], column)

        console.print("\n[bold]Extraction Results:[/bold]")
        console.print(f"Total rows: {result.total_rows}")
        console.print(f"Unique medications: {result.unique_medications}")

        # Show top medications
        top_meds = result.get_top_medications(10)
        if top_meds:
            console.print("\n[bold]Top Medications:[/bold]")
            for med, count in top_meds:
                console.print(f"  {med}: {count}")

        # Save results if output specified
        if output:
            output_data = {
                "column": column,
                "total_rows": result.total_rows,
                "unique_medications": result.unique_medications,
                "medications": result.normalized_medications,
                "frequency": result.frequency_map,
                "variants": result.variants_map,
            }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)

            console.print(f"\n[green]Results saved to: {output}[/green]")

    except Exception as e:
        console.print(f"[red]Error extracting medications: {e}[/red]")
        raise typer.Exit(1)
