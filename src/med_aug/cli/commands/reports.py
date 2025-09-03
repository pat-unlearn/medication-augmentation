"""Report generation CLI commands."""

import asyncio
from pathlib import Path
from typing import Optional, List
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json

from ...output import (
    ReportGenerator,
    ReportConfig,
    ReportFormat,
    MetricsCalculator,
    ChartGenerator,
    VisualizationConfig
)
from ...core.logging import get_logger

logger = get_logger(__name__)
console = Console()
app = typer.Typer()


@app.command("generate")
def generate_report(
    input_file: Path = typer.Argument(..., help="Pipeline results JSON file"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory"),
    formats: List[str] = typer.Option(["html"], "--format", "-f", help="Output formats (html, pdf, excel, json, markdown)"),
    title: Optional[str] = typer.Option(None, "--title", help="Report title"),
    include_charts: bool = typer.Option(True, "--charts/--no-charts", help="Include visualizations"),
    include_metrics: bool = typer.Option(True, "--metrics/--no-metrics", help="Include quality metrics"),
    include_raw: bool = typer.Option(False, "--raw", help="Include raw data"),
):
    """Generate comprehensive report from pipeline results."""
    
    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd() / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse formats
    report_formats = []
    format_map = {
        'html': ReportFormat.HTML,
        'pdf': ReportFormat.PDF,
        'excel': ReportFormat.EXCEL,
        'json': ReportFormat.JSON,
        'markdown': ReportFormat.MARKDOWN
    }
    
    for fmt in formats:
        if fmt.lower() in format_map:
            report_formats.append(format_map[fmt.lower()])
        else:
            console.print(f"[yellow]Warning: Unknown format '{fmt}', skipping[/yellow]")
    
    if not report_formats:
        console.print("[red]Error: No valid formats specified[/red]")
        raise typer.Exit(1)
    
    console.print(Panel(
        f"[bold blue]Report Generation[/bold blue]\n\n"
        f"Input: {input_file}\n"
        f"Output: {output_dir}\n"
        f"Formats: {', '.join(f.value for f in report_formats)}\n"
        f"Charts: {'✅' if include_charts else '❌'}\n"
        f"Metrics: {'✅' if include_metrics else '❌'}",
        title="Configuration"
    ))
    
    try:
        # Load pipeline results
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Create report configuration
        config = ReportConfig(
            title=title or data.get('metadata', {}).get('title', 'Medication Augmentation Report'),
            subtitle=data.get('metadata', {}).get('subtitle'),
            author="Medication Augmentation System",
            formats=report_formats,
            include_visualizations=include_charts,
            include_metrics=include_metrics,
            include_raw_data=include_raw,
            output_dir=output_dir
        )
        
        # Initialize report generator
        generator = ReportGenerator(config)
        
        # Add sections from data
        if 'summary' in data:
            generator.add_summary(data['summary'])
        
        if 'medications' in data:
            generator.add_medications_table(data['medications'])
        
        if 'classifications' in data:
            generator.add_classification_results(data['classifications'])
        
        # Calculate and add metrics if requested
        if include_metrics:
            console.print("[cyan]Calculating quality metrics...[/cyan]")
            calculator = MetricsCalculator()
            
            if 'phase_results' in data:
                metrics = calculator.calculate_from_pipeline_results(data['phase_results'])
            else:
                metrics = calculator.calculate_from_data(
                    None,
                    data.get('extraction_results'),
                    data.get('classifications'),
                    data.get('validation_results')
                )
            
            generator.add_metrics(metrics.to_dict())
            
            # Show metrics summary
            summary = metrics.get_summary()
            table = Table(title="Quality Metrics Summary")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="yellow")
            
            for key, value in summary.items():
                table.add_row(key, str(value))
            
            console.print(table)
        
        # Add visualizations if requested
        if include_charts:
            console.print("[cyan]Generating visualizations...[/cyan]")
            chart_gen = ChartGenerator(VisualizationConfig())
            
            # Create relevant charts based on available data
            if 'classifications' in data and include_metrics:
                # Drug class distribution
                if metrics.drug_class_distribution:
                    chart_data = chart_gen.create_bar_chart(
                        metrics.drug_class_distribution,
                        "Drug Class Distribution",
                        "Drug Class",
                        "Count"
                    )
                    generator.add_visualization(chart_data, 'bar')
                
                # Confidence distribution
                if 'confidence_scores' in data:
                    chart_data = chart_gen.create_confidence_distribution(
                        data['confidence_scores']
                    )
                    generator.add_visualization(chart_data, 'histogram')
                
                # Metrics dashboard
                if include_metrics:
                    chart_data = chart_gen.create_metrics_dashboard(
                        metrics.to_dict()
                    )
                    generator.add_visualization(chart_data, 'dashboard')
        
        # Add raw data if requested
        if include_raw:
            generator.add_raw_data(data, "Complete Pipeline Results")
        
        # Generate reports
        console.print("\n[cyan]Generating reports...[/cyan]")
        output_files = generator.generate()
        
        # Display results
        if output_files:
            console.print("\n[green]✅ Reports generated successfully![/green]\n")
            
            table = Table(title="Generated Reports")
            table.add_column("Format", style="cyan")
            table.add_column("File", style="yellow")
            table.add_column("Size", style="green")
            
            for fmt, file_path in output_files.items():
                size_kb = file_path.stat().st_size / 1024
                table.add_row(
                    fmt.value,
                    str(file_path),
                    f"{size_kb:.1f} KB"
                )
            
            console.print(table)
        else:
            console.print("[red]❌ No reports were generated[/red]")
        
    except Exception as e:
        console.print(f"\n[red]Report generation error: {e}[/red]")
        logger.error("report_generation_failed", error=str(e))
        raise typer.Exit(1)


@app.command("metrics")
def calculate_metrics(
    input_file: Path = typer.Argument(..., help="Pipeline results JSON file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for metrics"),
    format: str = typer.Option("json", "--format", "-f", help="Output format (json, csv)"),
):
    """Calculate quality metrics from pipeline results."""
    
    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Calculate metrics
        console.print("[cyan]Calculating metrics...[/cyan]")
        calculator = MetricsCalculator()
        
        if 'phase_results' in data:
            metrics = calculator.calculate_from_pipeline_results(data['phase_results'])
        else:
            metrics = calculator.calculate_from_data(
                None,
                data.get('extraction_results'),
                data.get('classifications'),
                data.get('validation_results')
            )
        
        # Display summary
        summary = metrics.get_summary()
        
        console.print("\n[bold]Quality Metrics Summary[/bold]\n")
        for key, value in summary.items():
            console.print(f"  [cyan]{key}:[/cyan] [yellow]{value}[/yellow]")
        
        # Display detailed metrics
        detailed = metrics.to_dict()
        
        console.print("\n[bold]Detailed Metrics[/bold]\n")
        
        for category, values in detailed.items():
            if isinstance(values, dict):
                console.print(f"  [green]{category}:[/green]")
                for k, v in values.items():
                    console.print(f"    {k}: {v}")
        
        # Get recommendations
        recommendations = calculator.get_recommendations()
        if recommendations:
            console.print("\n[bold]Recommendations[/bold]\n")
            for i, rec in enumerate(recommendations, 1):
                console.print(f"  {i}. [yellow]{rec}[/yellow]")
        
        # Save if output specified
        if output:
            if format.lower() == 'json':
                with open(output, 'w') as f:
                    json.dump({
                        'summary': summary,
                        'detailed': detailed,
                        'recommendations': recommendations
                    }, f, indent=2)
            elif format.lower() == 'csv':
                import csv
                with open(output, 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(['Category', 'Metric', 'Value'])
                    for category, values in detailed.items():
                        if isinstance(values, dict):
                            for k, v in values.items():
                                writer.writerow([category, k, v])
            
            console.print(f"\n[green]Metrics saved to: {output}[/green]")
        
    except Exception as e:
        console.print(f"\n[red]Metrics calculation error: {e}[/red]")
        raise typer.Exit(1)


@app.command("export")
def export_data(
    input_file: Path = typer.Argument(..., help="Data file to export"),
    output: Path = typer.Argument(..., help="Output file path"),
    format: str = typer.Option("excel", "--format", "-f", help="Export format (excel, csv, json, html)"),
    sheet_name: Optional[str] = typer.Option(None, "--sheet", help="Sheet name for Excel export"),
):
    """Export data to various formats."""
    
    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    try:
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        console.print(f"[cyan]Exporting to {format.upper()}...[/cyan]")
        
        # Import exporters
        from ...output.exporters import (
            ExcelExporter,
            CSVExporter,
            JSONExporter,
            HTMLExporter
        )
        
        # Select exporter
        if format.lower() == 'excel':
            exporter = ExcelExporter()
            # Organize data into sheets
            if isinstance(data, dict):
                export_data = data
            else:
                export_data = {sheet_name or 'Data': data}
        elif format.lower() == 'csv':
            exporter = CSVExporter()
            # CSV expects list of dicts
            if isinstance(data, list):
                export_data = data
            elif isinstance(data, dict) and 'data' in data:
                export_data = data['data']
            else:
                export_data = [data]
        elif format.lower() == 'json':
            exporter = JSONExporter()
            export_data = data
        elif format.lower() == 'html':
            exporter = HTMLExporter()
            export_data = data if isinstance(data, dict) else {'Data': data}
        else:
            console.print(f"[red]Unsupported format: {format}[/red]")
            raise typer.Exit(1)
        
        # Export
        output_path = exporter.export(export_data, output)
        
        # Show result
        size_kb = output_path.stat().st_size / 1024
        console.print(f"\n[green]✅ Export completed![/green]")
        console.print(f"  File: {output_path}")
        console.print(f"  Size: {size_kb:.1f} KB")
        
    except Exception as e:
        console.print(f"\n[red]Export error: {e}[/red]")
        raise typer.Exit(1)


@app.command("visualize")
def visualize_data(
    input_file: Path = typer.Argument(..., help="Data file to visualize"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o", help="Output directory for charts"),
    chart_type: str = typer.Option("auto", "--type", "-t", help="Chart type (bar, pie, histogram, dashboard, auto)"),
):
    """Generate visualizations from data."""
    
    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: Input file not found: {input_file}[/red]")
        raise typer.Exit(1)
    
    # Set output directory
    if output_dir is None:
        output_dir = Path.cwd() / "charts"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        console.print("[cyan]Generating visualizations...[/cyan]")
        
        # Initialize chart generator
        from ...output.visualizations import ChartGenerator, VisualizationConfig
        
        config = VisualizationConfig(
            figure_size=(12, 8),
            save_format='png'
        )
        chart_gen = ChartGenerator(config)
        
        charts_created = []
        
        # Auto-detect chart types based on data
        if chart_type == "auto":
            # Look for specific data patterns
            if 'drug_class_distribution' in data:
                chart = chart_gen.create_bar_chart(
                    data['drug_class_distribution'],
                    "Drug Class Distribution",
                    output_path=output_dir / "drug_class_distribution.png"
                )
                charts_created.append(chart)
            
            if 'confidence_scores' in data:
                chart = chart_gen.create_confidence_distribution(
                    data['confidence_scores'],
                    output_path=output_dir / "confidence_distribution.png"
                )
                charts_created.append(chart)
            
            if 'metrics' in data:
                chart = chart_gen.create_metrics_dashboard(
                    data['metrics'],
                    output_path=output_dir / "metrics_dashboard.png"
                )
                charts_created.append(chart)
            
            # Generic charts for dict data
            for key, value in data.items():
                if isinstance(value, dict) and all(isinstance(v, (int, float)) for v in value.values()):
                    if len(value) <= 10:
                        chart = chart_gen.create_pie_chart(
                            value,
                            f"{key.replace('_', ' ').title()}",
                            output_path=output_dir / f"{key}_pie.png"
                        )
                    else:
                        chart = chart_gen.create_bar_chart(
                            value,
                            f"{key.replace('_', ' ').title()}",
                            output_path=output_dir / f"{key}_bar.png"
                        )
                    charts_created.append(chart)
        
        else:
            # Specific chart type requested
            if chart_type == "dashboard" and 'metrics' in data:
                chart = chart_gen.create_metrics_dashboard(
                    data['metrics'],
                    output_path=output_dir / "dashboard.png"
                )
                charts_created.append(chart)
            else:
                # Try to create requested chart type with available data
                for key, value in data.items():
                    if isinstance(value, dict):
                        if chart_type == "bar":
                            chart = chart_gen.create_bar_chart(
                                value,
                                key.replace('_', ' ').title(),
                                output_path=output_dir / f"{key}_bar.png"
                            )
                        elif chart_type == "pie":
                            chart = chart_gen.create_pie_chart(
                                value,
                                key.replace('_', ' ').title(),
                                output_path=output_dir / f"{key}_pie.png"
                            )
                        charts_created.append(chart)
                        break
        
        # Display results
        if charts_created:
            console.print(f"\n[green]✅ Generated {len(charts_created)} visualization(s)[/green]\n")
            
            table = Table(title="Generated Charts")
            table.add_column("Title", style="cyan")
            table.add_column("Type", style="yellow")
            table.add_column("File", style="green")
            
            for chart in charts_created:
                if 'path' in chart:
                    table.add_row(
                        chart.get('title', 'Chart'),
                        chart.get('type', 'unknown'),
                        chart.get('path', '')
                    )
            
            console.print(table)
        else:
            console.print("[yellow]No visualizations could be generated from the data[/yellow]")
        
    except Exception as e:
        console.print(f"\n[red]Visualization error: {e}[/red]")
        raise typer.Exit(1)