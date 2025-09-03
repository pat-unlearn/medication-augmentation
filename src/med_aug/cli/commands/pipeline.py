"""Pipeline execution commands for the CLI."""

import asyncio
from pathlib import Path
from typing import Optional
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from ...pipeline import PipelineOrchestrator, PipelineConfig
from ...pipeline.checkpoint import CheckpointManager
from ...core.logging import get_logger

logger = get_logger(__name__)
console = Console()
app = typer.Typer()


@app.command("run")
def run_pipeline(
    input_file: Path = typer.Argument(..., help="Input data file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    disease: str = typer.Option("nsclc", "--disease", "-d", help="Disease module to use"),
    confidence: float = typer.Option(0.5, "--confidence", "-c", help="Confidence threshold"),
    no_web: bool = typer.Option(False, "--no-web", help="Disable web research"),
    no_validation: bool = typer.Option(False, "--no-validation", help="Disable validation"),
    no_checkpoints: bool = typer.Option(False, "--no-checkpoints", help="Disable checkpoints"),
    resume_from: Optional[str] = typer.Option(None, "--resume", help="Resume from phase"),
    pipeline_id: Optional[str] = typer.Option(None, "--id", help="Pipeline ID for resume"),
):
    """Run the medication augmentation pipeline."""
    
    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd() / "output" / f"pipeline_{Path(input_file).stem}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create configuration
    config = PipelineConfig(
        input_file=str(input_file),
        output_path=str(output_dir),
        disease_module=disease,
        confidence_threshold=confidence,
        enable_web_research=not no_web,
        enable_validation=not no_validation,
        enable_checkpoints=not no_checkpoints,
        display_progress=True,
        progress_mode="rich"
    )
    
    # Display configuration
    console.print(Panel(
        f"[bold blue]Pipeline Configuration[/bold blue]\n\n"
        f"Input: {input_file}\n"
        f"Output: {output_dir}\n"
        f"Disease: {disease}\n"
        f"Confidence: {confidence}\n"
        f"Web Research: {'✅' if not no_web else '❌'}\n"
        f"Validation: {'✅' if not no_validation else '❌'}\n"
        f"Checkpoints: {'✅' if not no_checkpoints else '❌'}",
        title="Starting Pipeline"
    ))
    
    # Create orchestrator
    orchestrator = PipelineOrchestrator(config, pipeline_id)
    
    # Run pipeline
    try:
        logger.info(
            "pipeline_run_started",
            input_file=str(input_file),
            output_dir=str(output_dir)
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
                    console.print(f"  {key}: {value}")
            
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
            duration=str(report.duration)
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
    console.print(Panel(
        f"[bold blue]Pipeline Status[/bold blue]\n\n"
        f"Pipeline ID: {checkpoint.pipeline_id}\n"
        f"Timestamp: {checkpoint.timestamp}\n"
        f"Current Phase: {checkpoint.current_phase}\n"
        f"Completed Phases: {', '.join(checkpoint.completed_phases)}",
        title="Checkpoint Information"
    ))
    
    # Show phase results
    if checkpoint.phase_results:
        table = Table(title="Phase Results")
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="yellow")
        
        for phase_name, result in checkpoint.phase_results.items():
            status = result.get('status', 'unknown')
            duration = result.get('duration_seconds', '-')
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
        size_kb = checkpoint['size_bytes'] / 1024
        table.add_row(
            checkpoint['pipeline_id'],
            checkpoint['timestamp'],
            f"{size_kb:.1f} KB",
            checkpoint['modified']
        )
    
    console.print(table)
    console.print(f"\nTotal: {len(checkpoints)} checkpoints")


@app.command("clean")
def clean_checkpoints(
    days: int = typer.Option(7, "--days", "-d", help="Delete checkpoints older than N days"),
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
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Confidence threshold"),
):
    """Analyze a file to identify medication columns."""
    
    from ...core.analyzer import DataAnalyzer
    import pandas as pd
    
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Analyzing: {input_file}[/bold]")
    
    try:
        # Load file
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file, nrows=1000)
        elif input_file.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file, nrows=1000)
        else:
            console.print(f"[red]Unsupported file format: {input_file.suffix}[/red]")
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
                samples
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
    
    from ...core.extractor import MedicationExtractor
    import pandas as pd
    import json
    
    if not input_file.exists():
        console.print(f"[red]Error: File not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load file
        if input_file.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
        elif input_file.suffix.lower() in ['.xlsx', '.xls']:
            df = pd.read_excel(input_file)
        else:
            console.print(f"[red]Unsupported file format: {input_file.suffix}[/red]")
            raise typer.Exit(1)
        
        if column not in df.columns:
            console.print(f"[red]Column '{column}' not found in file[/red]")
            console.print(f"Available columns: {', '.join(df.columns)}")
            raise typer.Exit(1)
        
        # Extract medications
        extractor = MedicationExtractor()
        result = extractor.extract_from_series(df[column], column)
        
        console.print(f"\n[bold]Extraction Results:[/bold]")
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
                'column': column,
                'total_rows': result.total_rows,
                'unique_medications': result.unique_medications,
                'medications': result.normalized_medications,
                'frequency': result.frequency_map,
                'variants': result.variants_map
            }
            
            with open(output, 'w') as f:
                json.dump(output_data, f, indent=2)
            
            console.print(f"\n[green]Results saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error extracting medications: {e}[/red]")
        raise typer.Exit(1)