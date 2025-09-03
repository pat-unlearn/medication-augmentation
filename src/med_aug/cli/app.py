"""Main CLI application for medication augmentation system."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from typing import Optional
from pathlib import Path
import sys
import os

# Import disease registry to trigger auto-discovery
from ..diseases import disease_registry

# Import command modules
from .commands import diseases, pipeline

# Import logging
from ..core.logging import setup_logging, get_logger, quick_setup

console = Console()
logger = get_logger(__name__)

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
app.add_typer(pipeline.app, name="pipeline", help="🚀 Run augmentation pipeline")


@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-v", help="Show version"),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Enable debug logging"),
    log_file: Optional[Path] = typer.Option(None, "--log-file", help="Path to log file"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress output"),
) -> None:
    """
    🏥 Medication Augmentation System
    
    Intelligent medication discovery and classification system for clinical research.
    Automatically expands medication databases with comprehensive drug names and classifications.
    """
    
    if version:
        console.print("🏥 Medication Augmentation System v1.0.0", style="bold blue")
        raise typer.Exit()
    
    # Setup logging based on options
    if not ctx.invoked_subcommand:
        return

    log_level = "DEBUG" if debug else ("WARNING" if quiet else "INFO")

    # Setup logging
    if log_file:
        setup_logging(
            level=log_level,
            log_file=log_file,
            json_logs=False,
            include_timestamp=True
        )
    else:
        # Use quick setup for console logging
        quick_setup(debug=debug, log_to_file=False)

    # Log application start
    logger.info(
        "application_started",
        command=ctx.invoked_subcommand,
        debug=debug,
        config=str(config) if config else None
    )

    # Store config in context for subcommands
    if config:
        ctx.obj = {"config": config}
    
    # Display welcome banner if no subcommand
    if ctx.invoked_subcommand is None:
        _display_welcome()


def _display_welcome():
    """Display welcome banner with system information."""
    
    title = Text("Medication Augmentation System", style="bold blue")
    
    content = Text()
    content.append("🎯 ", style="bold yellow")
    content.append("Intelligent medication discovery for clinical research\n\n")
    
    # Show available disease modules
    available = disease_registry.list_available()
    if available:
        content.append("🔬 ", style="bold green")
        content.append(f"Available diseases: {', '.join(available)}\n")
    else:
        content.append("⚠️  ", style="bold yellow")
        content.append("No disease modules found - NSCLC module will be loaded on first use\n")
    
    content.append("⚡ ", style="bold red")
    content.append("Modern CLI with Rich formatting and async processing\n\n")
    content.append("🚀 ", style="bold magenta")
    content.append("Get started: ", style="bold")
    content.append("med-aug --help", style="cyan")
    content.append(" or ", style="dim")
    content.append("med-aug diseases list", style="cyan")
    
    panel = Panel(
        content,
        title="Welcome",
        border_style="blue",
        padding=(1, 2)
    )
    
    console.print(panel)


@app.command("test")
def test_command():
    """Test command to verify CLI is working."""
    logger.info("test_command_started")
    console.print("[bold green]✅ CLI is working correctly![/bold green]")
    
    # Test disease registry
    console.print("\n[bold]Testing disease registry...[/bold]")
    available = disease_registry.list_available()
    logger.debug("disease_modules_check", available=available)
    
    if available:
        console.print(f"Found {len(available)} disease module(s): {', '.join(available)}")
        
        # Try to load NSCLC module
        nsclc = disease_registry.get_module("nsclc")
        if nsclc:
            console.print(f"[green]✓[/green] NSCLC module loaded: {nsclc.display_name}")
            console.print(f"  - Drug classes: {len(nsclc.drug_classes)}")
            console.print(f"  - Total keywords: {len(nsclc.get_all_keywords())}")
            logger.info(
                "nsclc_module_loaded",
                drug_classes=len(nsclc.drug_classes),
                keywords=len(nsclc.get_all_keywords())
            )
    else:
        console.print("[yellow]No disease modules auto-discovered yet[/yellow]")
        console.print("This is normal on first run - modules will be loaded when needed")
        logger.warning("no_disease_modules_discovered")


@app.command("info")
def info_command():
    """Show system information and configuration."""
    from rich.table import Table
    import platform
    
    # System info table
    table = Table(title="System Information", show_header=True)
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="yellow")
    
    table.add_row("Version", "1.0.0")
    table.add_row("Python", platform.python_version())
    table.add_row("Platform", platform.platform())
    table.add_row("Disease Modules", str(len(disease_registry.list_available())))
    
    console.print(table)
    
    # Disease modules
    if disease_registry.list_available():
        console.print("\n[bold]Available Disease Modules:[/bold]")
        for disease_name in disease_registry.list_available():
            module = disease_registry.get_module(disease_name)
            if module:
                console.print(f"  • {module.display_name} ({disease_name})")


def main():
    """Main entry point for the CLI."""
    try:
        # Configure structlog
        import structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.dev.ConsoleRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        # Run the app
        app()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Error: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    main()