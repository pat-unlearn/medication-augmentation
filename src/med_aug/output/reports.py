"""Report generation with templates."""

from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
try:
    from jinja2 import Environment, FileSystemLoader, Template
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    Environment = None
    FileSystemLoader = None
    Template = None

from ..core.logging import get_logger

logger = get_logger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""
    JSON = "json"
    HTML = "html"
    PDF = "pdf"
    EXCEL = "excel"
    MARKDOWN = "markdown"


@dataclass
class ReportConfig:
    """Configuration for report generation."""
    
    title: str = "Medication Augmentation Report"
    subtitle: Optional[str] = None
    author: str = "Medication Augmentation System"
    organization: Optional[str] = None
    formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.HTML])
    include_visualizations: bool = True
    include_metrics: bool = True
    include_raw_data: bool = False
    template_dir: Optional[Path] = None
    output_dir: Path = Path("./reports")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'author': self.author,
            'organization': self.organization,
            'formats': [f.value for f in self.formats],
            'include_visualizations': self.include_visualizations,
            'include_metrics': self.include_metrics,
            'include_raw_data': self.include_raw_data,
            'output_dir': str(self.output_dir)
        }


@dataclass
class ReportSection:
    """A section in the report."""
    
    title: str
    content: Any
    section_type: str = "text"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'title': self.title,
            'content': self.content,
            'type': self.section_type,
            'metadata': self.metadata
        }


class ReportGenerator:
    """Generator for comprehensive reports."""
    
    def __init__(self, config: ReportConfig):
        """
        Initialize report generator.
        
        Args:
            config: Report configuration
        """
        self.config = config
        self.sections: List[ReportSection] = []
        self.metadata: Dict[str, Any] = {
            'generated_at': datetime.now().isoformat(),
            'generator': 'Medication Augmentation System',
            'version': '1.0.0'
        }
        
        # Setup Jinja2 environment if available
        if JINJA2_AVAILABLE:
            template_dir = config.template_dir or Path(__file__).parent / "templates"
            if template_dir.exists():
                self.env = Environment(
                    loader=FileSystemLoader(str(template_dir)),
                    autoescape=True
                )
            else:
                self.env = Environment()
        else:
            self.env = None
            logger.warning("jinja2_not_available", 
                         message="Jinja2 not installed, using basic HTML templates")
        
        # Ensure output directory exists
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("report_generator_initialized", config=config.to_dict())
    
    def add_section(self, section: ReportSection):
        """Add a section to the report."""
        self.sections.append(section)
        logger.debug("section_added", title=section.title, type=section.section_type)
    
    def add_summary(self, summary_data: Dict[str, Any]):
        """Add executive summary section."""
        self.add_section(ReportSection(
            title="Executive Summary",
            content=summary_data,
            section_type="summary"
        ))
    
    def add_medications_table(self, medications: List[Dict[str, Any]]):
        """Add medications table section."""
        self.add_section(ReportSection(
            title="Extracted Medications",
            content=medications,
            section_type="table",
            metadata={'columns': list(medications[0].keys()) if medications else []}
        ))
    
    def add_classification_results(self, classifications: Dict[str, Any]):
        """Add classification results section."""
        self.add_section(ReportSection(
            title="Classification Results",
            content=classifications,
            section_type="classification"
        ))
    
    def add_metrics(self, metrics: Dict[str, Any]):
        """Add quality metrics section."""
        self.add_section(ReportSection(
            title="Quality Metrics",
            content=metrics,
            section_type="metrics"
        ))
    
    def add_visualization(self, chart_data: Dict[str, Any], chart_type: str):
        """Add visualization section."""
        self.add_section(ReportSection(
            title=chart_data.get('title', 'Visualization'),
            content=chart_data,
            section_type="chart",
            metadata={'chart_type': chart_type}
        ))
    
    def add_raw_data(self, data: Any, title: str = "Raw Data"):
        """Add raw data section."""
        if self.config.include_raw_data:
            self.add_section(ReportSection(
                title=title,
                content=data,
                section_type="raw_data"
            ))
    
    def generate(self) -> Dict[str, Path]:
        """
        Generate reports in configured formats.
        
        Returns:
            Dictionary mapping format to output file path
        """
        logger.info("report_generation_started", formats=[f.value for f in self.config.formats])
        
        output_files = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"report_{timestamp}"
        
        # Generate each format
        for format_type in self.config.formats:
            try:
                if format_type == ReportFormat.JSON:
                    output_file = self._generate_json(base_name)
                elif format_type == ReportFormat.HTML:
                    output_file = self._generate_html(base_name)
                elif format_type == ReportFormat.MARKDOWN:
                    output_file = self._generate_markdown(base_name)
                elif format_type == ReportFormat.PDF:
                    output_file = self._generate_pdf(base_name)
                elif format_type == ReportFormat.EXCEL:
                    output_file = self._generate_excel(base_name)
                else:
                    logger.warning("unsupported_format", format=format_type.value)
                    continue
                
                output_files[format_type] = output_file
                logger.info("report_generated", format=format_type.value, file=str(output_file))
                
            except Exception as e:
                logger.error("report_generation_failed", format=format_type.value, error=str(e))
        
        return output_files
    
    def _generate_json(self, base_name: str) -> Path:
        """Generate JSON report."""
        output_file = self.config.output_dir / f"{base_name}.json"
        
        report_data = {
            'metadata': self.metadata,
            'config': self.config.to_dict(),
            'sections': [s.to_dict() for s in self.sections]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return output_file
    
    def _generate_html(self, base_name: str) -> Path:
        """Generate HTML report."""
        output_file = self.config.output_dir / f"{base_name}.html"
        
        # Try to load template
        try:
            template = self.env.get_template("report.html")
        except:
            # Use default template if not found
            template = self._get_default_html_template()
        
        # Render HTML
        html_content = template.render(
            title=self.config.title,
            subtitle=self.config.subtitle,
            author=self.config.author,
            organization=self.config.organization,
            metadata=self.metadata,
            sections=self.sections
        )
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        return output_file
    
    def _generate_markdown(self, base_name: str) -> Path:
        """Generate Markdown report."""
        output_file = self.config.output_dir / f"{base_name}.md"
        
        lines = []
        
        # Header
        lines.append(f"# {self.config.title}")
        if self.config.subtitle:
            lines.append(f"## {self.config.subtitle}")
        lines.append("")
        lines.append(f"**Author:** {self.config.author}")
        if self.config.organization:
            lines.append(f"**Organization:** {self.config.organization}")
        lines.append(f"**Generated:** {self.metadata['generated_at']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Sections
        for section in self.sections:
            lines.append(f"## {section.title}")
            lines.append("")
            
            if section.section_type == "summary":
                for key, value in section.content.items():
                    lines.append(f"- **{key}:** {value}")
            
            elif section.section_type == "table":
                if isinstance(section.content, list) and section.content:
                    # Table header
                    headers = list(section.content[0].keys())
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                    
                    # Table rows
                    for row in section.content[:50]:  # Limit to 50 rows
                        values = [str(row.get(h, "")) for h in headers]
                        lines.append("| " + " | ".join(values) + " |")
            
            elif section.section_type == "metrics":
                for key, value in section.content.items():
                    if isinstance(value, dict):
                        lines.append(f"### {key}")
                        for k, v in value.items():
                            lines.append(f"- {k}: {v}")
                    else:
                        lines.append(f"- **{key}:** {value}")
            
            else:
                # Generic content
                lines.append(str(section.content))
            
            lines.append("")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(lines))
        
        return output_file
    
    def _generate_pdf(self, base_name: str) -> Path:
        """Generate PDF report (requires additional dependencies)."""
        # First generate HTML
        html_file = self._generate_html(f"{base_name}_temp")
        output_file = self.config.output_dir / f"{base_name}.pdf"
        
        try:
            # Try using weasyprint if available
            from weasyprint import HTML
            HTML(filename=str(html_file)).write_pdf(str(output_file))
            html_file.unlink()  # Remove temp HTML
            return output_file
            
        except ImportError:
            logger.warning("pdf_generation_unavailable", reason="weasyprint not installed")
            # Rename HTML as fallback
            output_file = html_file.rename(self.config.output_dir / f"{base_name}_pdf.html")
            return output_file
    
    def _generate_excel(self, base_name: str) -> Path:
        """Generate Excel report."""
        import pandas as pd
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment
        
        output_file = self.config.output_dir / f"{base_name}.xlsx"
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for section in self.sections:
                if section.section_type == "summary":
                    summary_data = [{'Metric': k, 'Value': v} for k, v in section.content.items()]
                    break
            
            if summary_data:
                df_summary = pd.DataFrame(summary_data)
                df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Data sheets
            sheet_num = 1
            for section in self.sections:
                if section.section_type == "table" and isinstance(section.content, list):
                    sheet_name = f"Data_{sheet_num}"
                    df = pd.DataFrame(section.content)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    sheet_num += 1
            
            # Metrics sheet
            for section in self.sections:
                if section.section_type == "metrics":
                    metrics_data = []
                    for key, value in section.content.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                metrics_data.append({'Category': key, 'Metric': k, 'Value': v})
                        else:
                            metrics_data.append({'Category': 'General', 'Metric': key, 'Value': value})
                    
                    if metrics_data:
                        df_metrics = pd.DataFrame(metrics_data)
                        df_metrics.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Format workbook
            workbook = writer.book
            for sheet in workbook.worksheets:
                # Format headers
                for cell in sheet[1]:
                    cell.font = Font(bold=True)
                    cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
                    cell.font = Font(color="FFFFFF", bold=True)
                    cell.alignment = Alignment(horizontal="center")
                
                # Adjust column widths
                for column in sheet.columns:
                    max_length = 0
                    column_letter = column[0].column_letter
                    for cell in column:
                        try:
                            if len(str(cell.value)) > max_length:
                                max_length = len(str(cell.value))
                        except:
                            pass
                    adjusted_width = min(max_length + 2, 50)
                    sheet.column_dimensions[column_letter].width = adjusted_width
        
        return output_file
    
    def _get_default_html_template(self):
        """Get default HTML template."""
        template_str = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ title }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        h1 { color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
        h2 { color: #555; margin-top: 30px; }
        .metadata { background: #f4f4f4; padding: 10px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th { background: #4CAF50; color: white; padding: 10px; text-align: left; }
        td { padding: 8px; border-bottom: 1px solid #ddd; }
        tr:hover { background: #f5f5f5; }
        .summary { background: #e8f5e9; padding: 15px; border-radius: 5px; margin: 20px 0; }
        .metric { display: inline-block; margin: 10px 20px 10px 0; }
        .metric-label { font-weight: bold; color: #555; }
        .metric-value { color: #4CAF50; font-size: 1.2em; }
    </style>
</head>
<body>
    <h1>{{ title }}</h1>
    {% if subtitle %}<h2>{{ subtitle }}</h2>{% endif %}
    
    <div class="metadata">
        <strong>Author:</strong> {{ author }}<br>
        {% if organization %}<strong>Organization:</strong> {{ organization }}<br>{% endif %}
        <strong>Generated:</strong> {{ metadata.generated_at }}
    </div>
    
    {% for section in sections %}
    <h2>{{ section.title }}</h2>
    
    {% if section.section_type == 'summary' %}
    <div class="summary">
        {% for key, value in section.content.items() %}
        <div class="metric">
            <span class="metric-label">{{ key }}:</span>
            <span class="metric-value">{{ value }}</span>
        </div>
        {% endfor %}
    </div>
    
    {% elif section.section_type == 'table' %}
    <table>
        <thead>
            <tr>
                {% for col in section.metadata.columns %}
                <th>{{ col }}</th>
                {% endfor %}
            </tr>
        </thead>
        <tbody>
            {% for row in section.content[:50] %}
            <tr>
                {% for col in section.metadata.columns %}
                <td>{{ row[col]|default('') }}</td>
                {% endfor %}
            </tr>
            {% endfor %}
        </tbody>
    </table>
    
    {% elif section.section_type == 'metrics' %}
    <div class="summary">
        {% for key, value in section.content.items() %}
        <div class="metric">
            <span class="metric-label">{{ key }}:</span>
            <span class="metric-value">{{ value }}</span>
        </div>
        {% endfor %}
    </div>
    
    {% else %}
    <div>{{ section.content }}</div>
    {% endif %}
    
    {% endfor %}
</body>
</html>
        """
        if self.env:
            return self.env.from_string(template_str)
        else:
            # Return a simple function that just returns the string
            class SimpleTemplate:
                def __init__(self, template):
                    self.template = template
                
                def render(self, **kwargs):
                    # Basic string replacement
                    result = self.template
                    for key, value in kwargs.items():
                        result = result.replace(f"{{{{ {key} }}}}", str(value))
                    return result
            
            return SimpleTemplate(template_str)