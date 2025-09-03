"""Disease module management commands."""

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
    """List all available disease modules (nsclc, breast_cancer, etc.)."""

    available_diseases = disease_registry.list_available()

    if not available_diseases:
        console.print("❌ No disease modules found", style="red")
        console.print(
            "\nTip: Disease modules are auto-discovered from the diseases package."
        )
        console.print(
            "Make sure disease modules are properly installed (NSCLC, breast cancer, etc.)."
        )
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
                display_name = module.display_name
            else:
                status = "❌ Error"
                drug_count = 0
                display_name = "Unknown"

            table.add_row(disease_code, display_name, str(drug_count), status)
        except Exception as e:
            console.print(f"Error loading {disease_code}: {e}", style="red")
            table.add_row(disease_code, "Error loading", "0", "❌ Error")

    console.print(table)


@app.command("info")
def disease_info(
    disease: str = typer.Argument(..., help="Disease module code (e.g., nsclc)"),
) -> None:
    """Show detailed information about a disease module."""

    module = disease_registry.get_module(disease)
    if not module:
        console.print(f"❌ Disease module '{disease}' not found", style="red")
        available = disease_registry.list_available()
        if available:
            console.print(f"Available: {', '.join(available)}")
        else:
            console.print("No disease modules are currently available.")
        raise typer.Exit(1)

    # Disease info panel
    info_content = f"""
🔬 [bold]{module.display_name}[/bold]
📝 Code: {module.name}
🎯 Drug Classes: {len(module.drug_classes)}
🌐 Web Sources: {len(module.get_web_sources())}
💊 Total Keywords: {len(module.get_all_keywords())}
"""

    info_panel = Panel(
        info_content.strip(), title="Disease Information", border_style="blue"
    )
    console.print(info_panel)

    # Drug classes table
    if module.drug_classes:
        drug_table = Table(title="Drug Classes")
        drug_table.add_column("Class", style="cyan")
        drug_table.add_column("Keywords", style="yellow")
        drug_table.add_column("Threshold", justify="right", style="green")

        for drug_class in module.drug_classes:
            # Show first 5 keywords
            keywords_display = drug_class.keywords[:5]
            keywords_str = ", ".join(keywords_display)
            if len(drug_class.keywords) > 5:
                keywords_str += f" ... (+{len(drug_class.keywords) - 5} more)"

            drug_table.add_row(
                drug_class.name, keywords_str, f"{drug_class.confidence_threshold:.1%}"
            )

        console.print(drug_table)

    # Web sources
    console.print("\n[bold]Web Sources:[/bold]")
    for i, source in enumerate(module.get_web_sources(), 1):
        # Truncate long URLs for display
        display_source = source if len(source) < 80 else source[:77] + "..."
        console.print(f"  {i}. {display_source}")


@app.command("keywords")
def show_keywords(
    disease: str = typer.Argument(..., help="Disease module code"),
    drug_class: Optional[str] = typer.Option(
        None, "--class", "-c", help="Filter by drug class"
    ),
) -> None:
    """Show all keywords for a disease module."""

    module = disease_registry.get_module(disease)
    if not module:
        console.print(f"❌ Disease module '{disease}' not found", style="red")
        raise typer.Exit(1)

    if drug_class:
        # Show keywords for specific drug class
        config = module.get_drug_class_by_name(drug_class)
        if not config:
            console.print(
                f"❌ Drug class '{drug_class}' not found in {disease}", style="red"
            )
            console.print(
                f"Available classes: {', '.join([dc.name for dc in module.drug_classes])}"
            )
            raise typer.Exit(1)

        console.print(
            f"\n[bold]Keywords for {drug_class} in {module.display_name}:[/bold]"
        )
        console.print(f"Total: {len(config.keywords)} keywords\n")

        # Display keywords in columns
        from rich.columns import Columns

        columns = Columns(config.keywords, equal=True, expand=False)
        console.print(columns)
    else:
        # Show all keywords grouped by drug class
        console.print(f"\n[bold]All keywords for {module.display_name}:[/bold]")

        for dc in module.drug_classes:
            console.print(f"\n[cyan]{dc.name}[/cyan] ({len(dc.keywords)} keywords):")
            # Show first 10 keywords per class
            display_keywords = dc.keywords[:10]
            for keyword in display_keywords:
                console.print(f"  • {keyword}")
            if len(dc.keywords) > 10:
                console.print(f"  ... and {len(dc.keywords) - 10} more")


@app.command("validate")
def validate_module(
    disease: str = typer.Argument(..., help="Disease module code to validate"),
) -> None:
    """Validate a disease module configuration."""

    module = disease_registry.get_module(disease)
    if not module:
        console.print(f"❌ Disease module '{disease}' not found", style="red")
        raise typer.Exit(1)

    console.print(f"[bold]Validating {module.display_name} module...[/bold]\n")

    issues = []
    warnings = []

    # Check drug classes
    if not module.drug_classes:
        issues.append("No drug classes defined")
    else:
        for dc in module.drug_classes:
            if not dc.keywords:
                issues.append(f"Drug class '{dc.name}' has no keywords")
            elif len(dc.keywords) < 3:
                warnings.append(
                    f"Drug class '{dc.name}' has only {len(dc.keywords)} keywords"
                )

            if not (0 <= dc.confidence_threshold <= 1):
                issues.append(
                    f"Drug class '{dc.name}' has invalid confidence threshold: {dc.confidence_threshold}"
                )

    # Check web sources
    if not module.get_web_sources():
        warnings.append("No web sources defined")

    # Check LLM context
    if not module.get_llm_context():
        issues.append("No LLM context defined")
    elif len(module.get_llm_context()) < 100:
        warnings.append("LLM context seems too short")

    # Display results
    if issues:
        console.print("[red]❌ Validation failed with errors:[/red]")
        for issue in issues:
            console.print(f"  • {issue}")
    else:
        console.print("[green]✅ No critical issues found[/green]")

    if warnings:
        console.print("\n[yellow]⚠️  Warnings:[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")

    if not issues and not warnings:
        console.print("[green]Perfect! Module is fully configured.[/green]")

    # Summary
    console.print("\n[bold]Summary:[/bold]")
    console.print(f"  • Drug classes: {len(module.drug_classes)}")
    console.print(f"  • Total keywords: {len(module.get_all_keywords())}")
    console.print(f"  • Web sources: {len(module.get_web_sources())}")
    console.print(f"  • LLM context length: {len(module.get_llm_context())} characters")
