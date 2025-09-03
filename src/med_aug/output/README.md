# Output Module

## Overview

The output module provides comprehensive reporting, visualization, and export capabilities for the medication augmentation system. It generates professional reports in multiple formats with quality metrics, charts, and actionable insights.

## Structure

```
output/
├── __init__.py
├── reports.py          # Report generation and formatting
├── metrics.py          # Quality metrics calculation
├── visualizations.py   # Chart and graph generation
└── exporters.py        # Format-specific exporters
```

## Key Components

### Report Generator (`reports.py`)

Comprehensive report generation with multiple formats:

```python
from med_aug.output import ReportGenerator, ReportConfig, ReportFormat

config = ReportConfig(
    title="Medication Analysis Report",
    subtitle="Q4 2025 Analysis",
    formats=[ReportFormat.HTML, ReportFormat.EXCEL, ReportFormat.PDF],
    include_visualizations=True,
    include_metrics=True,
    include_raw_data=False
)

generator = ReportGenerator(config)

# Add report sections
generator.add_summary(summary_data)
generator.add_medications_table(medications)
generator.add_classification_results(classifications)
generator.add_validation_results(validations)
generator.add_metrics(metrics)
generator.add_visualizations(charts)
generator.add_recommendations(recommendations)

# Generate reports
output_files = generator.generate()
```

#### Report Formats

##### HTML Reports
```python
from med_aug.output import HTMLReporter

reporter = HTMLReporter(
    template="professional",  # or "simple", "detailed"
    css_theme="bootstrap",
    interactive=True
)

html = reporter.generate(data)
reporter.save("report.html")
```

##### PDF Reports
```python
from med_aug.output import PDFReporter

reporter = PDFReporter(
    page_size="A4",
    orientation="portrait",
    include_toc=True,
    include_header=True
)

pdf_bytes = reporter.generate(data)
reporter.save("report.pdf")
```

##### Excel Workbooks
```python
from med_aug.output import ExcelReporter

reporter = ExcelReporter(
    include_formatting=True,
    freeze_panes=True,
    auto_filter=True
)

# Multi-sheet workbook
reporter.add_sheet("Summary", summary_df)
reporter.add_sheet("Medications", medications_df)
reporter.add_sheet("Metrics", metrics_df)
reporter.add_chart("Drug Distribution", chart_data)

reporter.save("report.xlsx")
```

##### Markdown Reports
```python
from med_aug.output import MarkdownReporter

reporter = MarkdownReporter(
    include_toc=True,
    github_flavored=True
)

markdown = reporter.generate(data)
reporter.save("report.md")
```

### Quality Metrics (`metrics.py`)

Comprehensive metrics calculation and analysis:

```python
from med_aug.output import MetricsCalculator, QualityMetrics

calculator = MetricsCalculator()

# Calculate from pipeline results
metrics = calculator.calculate_from_pipeline_results(
    phase_results,
    include_performance=True,
    include_quality=True
)

# Access metrics
print(f"Extraction rate: {metrics.extraction_rate:.2%}")
print(f"Classification coverage: {metrics.classification_coverage:.2%}")
print(f"Average confidence: {metrics.avg_confidence:.3f}")
print(f"Processing time: {metrics.total_time:.2f}s")
```

#### Metric Categories

##### Data Quality Metrics
```python
data_metrics = metrics.data_quality
print(f"Total rows: {data_metrics.total_rows}")
print(f"Medication columns: {data_metrics.medication_columns}")
print(f"Data completeness: {data_metrics.completeness:.2%}")
print(f"Missing values: {data_metrics.missing_count}")
```

##### Extraction Metrics
```python
extraction = metrics.extraction
print(f"Total extracted: {extraction.total_medications}")
print(f"Unique medications: {extraction.unique_count}")
print(f"Normalization rate: {extraction.normalization_rate:.2%}")
print(f"Variants identified: {extraction.variant_count}")
```

##### Classification Metrics
```python
classification = metrics.classification
print(f"Classified: {classification.classified_count}")
print(f"Coverage: {classification.coverage:.2%}")
print(f"High confidence: {classification.high_confidence_count}")
print(f"Distribution: {classification.drug_class_distribution}")
```

##### Performance Metrics
```python
performance = metrics.performance
print(f"Total time: {performance.total_seconds:.2f}s")
print(f"Phase times: {performance.phase_times}")
print(f"Throughput: {performance.rows_per_second:.0f} rows/s")
print(f"Memory usage: {performance.memory_mb:.0f} MB")
```

#### Recommendations Engine
```python
from med_aug.output import RecommendationEngine

engine = RecommendationEngine()
recommendations = engine.analyze(metrics)

for rec in recommendations:
    print(f"[{rec.priority}] {rec.category}: {rec.message}")
    print(f"  Action: {rec.action}")
    print(f"  Impact: {rec.expected_impact}")
```

### Data Visualizations (`visualizations.py`)

Rich visualization capabilities:

```python
from med_aug.output import ChartGenerator, VisualizationConfig

config = VisualizationConfig(
    figure_size=(12, 8),
    color_palette="Set2",
    style="seaborn",
    dpi=100
)

chart_gen = ChartGenerator(config)
```

#### Chart Types

##### Bar Charts
```python
# Drug class distribution
chart = chart_gen.create_bar_chart(
    data=drug_class_counts,
    title="Drug Class Distribution",
    xlabel="Drug Class",
    ylabel="Count",
    color_by="confidence"
)
chart.save("drug_distribution.png")
```

##### Pie Charts
```python
# Classification coverage
chart = chart_gen.create_pie_chart(
    data=classification_stats,
    title="Classification Coverage",
    labels=["Classified", "Unclassified"],
    colors=["#2ecc71", "#e74c3c"],
    autopct="%1.1f%%"
)
```

##### Histograms
```python
# Confidence distribution
chart = chart_gen.create_histogram(
    data=confidence_scores,
    title="Confidence Score Distribution",
    xlabel="Confidence Score",
    ylabel="Frequency",
    bins=20,
    kde=True
)
```

##### Dashboard
```python
# Comprehensive metrics dashboard
dashboard = chart_gen.create_dashboard(
    metrics=metrics_dict,
    layout="2x2",
    charts=[
        ("bar", drug_classes, "Drug Classes"),
        ("pie", coverage, "Coverage"),
        ("hist", confidence, "Confidence"),
        ("line", performance, "Performance")
    ]
)
dashboard.save("dashboard.png")
```

#### Fallback Visualizations

When matplotlib is not available:

```python
# Text-based charts
from med_aug.output import TextChartGenerator

text_gen = TextChartGenerator()

# ASCII bar chart
chart = text_gen.create_bar_chart(
    data={"A": 10, "B": 20, "C": 15},
    width=50
)
print(chart)
# Output:
# A |████████          | 10
# B |████████████████  | 20
# C |████████████      | 15
```

### Data Exporters (`exporters.py`)

Specialized exporters for each format:

#### JSON Exporter
```python
from med_aug.output import JSONExporter

exporter = JSONExporter(
    pretty_print=True,
    include_metadata=True,
    date_format="iso"
)

data = {
    "results": results,
    "metrics": metrics,
    "metadata": {
        "generated": datetime.now(),
        "version": "1.0.0"
    }
}

exporter.export(data, "results.json")
```

#### CSV Exporter
```python
from med_aug.output import CSVExporter

exporter = CSVExporter(
    delimiter=",",
    quoting="minimal",
    include_index=False
)

exporter.export(dataframe, "medications.csv")
```

#### Database Export
```python
from med_aug.output import DatabaseExporter

exporter = DatabaseExporter(
    connection_string="postgresql://localhost/medaug",
    table_name="medications",
    if_exists="append"
)

exporter.export(results)
```

## Report Templates

### Using Jinja2 Templates (Optional)
```python
from med_aug.output import TemplateReportGenerator

generator = TemplateReportGenerator(
    template_dir="./templates",
    template_name="professional.html.j2"
)

html = generator.render(
    title=title,
    data=data,
    metrics=metrics,
    charts=charts
)
```

### Built-in Templates
```python
# Simple template (no Jinja2 required)
from med_aug.output import SimpleTemplate

template = SimpleTemplate()
html = template.render(
    sections=[
        ("Summary", summary_html),
        ("Data", table_html),
        ("Metrics", metrics_html)
    ]
)
```

## CLI Integration

```bash
# Generate comprehensive report
med-aug reports generate pipeline_results.json \
  --format html excel pdf \
  --title "Q4 2025 Analysis" \
  --charts --metrics

# Calculate metrics only
med-aug reports metrics pipeline_results.json \
  --output metrics.json \
  --format json

# Create visualizations
med-aug reports visualize data.json \
  --type dashboard \
  --output charts/ \
  --format png svg

# Export to specific format
med-aug reports export data.json output.xlsx \
  --format excel \
  --sheets "Summary,Data,Metrics"
```

## Configuration

### Report Configuration
```yaml
output:
  default_format: ["html", "json"]
  report_dir: "./reports"
  template_dir: "./templates"
  
  html:
    theme: "professional"
    include_toc: true
    include_charts: true
  
  excel:
    include_formatting: true
    freeze_headers: true
    auto_column_width: true
  
  pdf:
    page_size: "A4"
    orientation: "portrait"
    margins: [20, 20, 20, 20]
  
  visualizations:
    backend: "matplotlib"  # or "plotly", "text"
    style: "seaborn"
    dpi: 100
    format: "png"
```

### Environment Variables
```bash
# Output directory
export MEDAUG_OUTPUT_DIR=./reports

# Default formats
export MEDAUG_OUTPUT_FORMATS=html,json

# Visualization settings
export MEDAUG_VIZ_BACKEND=matplotlib
export MEDAUG_VIZ_DPI=150

# Template directory
export MEDAUG_TEMPLATE_DIR=./templates
```

## Usage Examples

### Complete Pipeline Report
```python
from med_aug.output import create_pipeline_report

# Generate full report from pipeline results
report = create_pipeline_report(
    pipeline_results,
    title="Medication Analysis Report",
    include_all=True
)

# Save in multiple formats
report.save_html("report.html")
report.save_excel("report.xlsx")
report.save_pdf("report.pdf")
```

### Custom Report Building
```python
from med_aug.output import ReportBuilder

builder = ReportBuilder()

# Add sections incrementally
builder.add_header("Medication Analysis")
builder.add_summary(stats)
builder.add_table(medications_df, "Medications")
builder.add_chart(chart, "Distribution")
builder.add_metrics(metrics)
builder.add_footer("Generated: " + datetime.now())

report = builder.build()
```

### Metrics-Only Analysis
```python
from med_aug.output import analyze_results

# Quick metrics calculation
metrics = analyze_results(results)
print(metrics.summary())

# Detailed analysis
analysis = analyze_results(
    results,
    include_recommendations=True,
    include_trends=True
)
```

## Quality Assurance

### Report Validation
```python
from med_aug.output import ReportValidator

validator = ReportValidator()

# Validate report completeness
issues = validator.validate(report)
if issues:
    for issue in issues:
        print(f"Warning: {issue}")
```

### Metric Validation
```python
# Ensure metrics are within expected ranges
assert metrics.extraction_rate > 0.8, "Low extraction rate"
assert metrics.classification_coverage > 0.7, "Low coverage"
assert metrics.avg_confidence > 0.6, "Low confidence"
```

## Performance Considerations

### Large Dataset Handling
```python
# Stream large datasets
from med_aug.output import StreamingReporter

reporter = StreamingReporter()
with reporter.stream("large_report.html") as stream:
    for chunk in process_data_chunks():
        stream.write(chunk)
```

### Parallel Report Generation
```python
import asyncio

async def generate_reports_parallel(data, formats):
    tasks = []
    for format in formats:
        task = generate_report_async(data, format)
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    return results
```

### Caching
```python
from med_aug.output import CachedReportGenerator

generator = CachedReportGenerator(
    cache_dir="./report_cache",
    ttl_seconds=3600
)

# Reuses cached components
report = generator.generate(data)
```

## Error Handling

```python
from med_aug.output import (
    ReportGenerationError,
    VisualizationError,
    ExportError
)

try:
    report = generator.generate()
except VisualizationError as e:
    # Fall back to text-based charts
    logger.warning(f"Visualization failed: {e}")
    report = generator.generate(visualizations=False)
except ExportError as e:
    # Try alternative format
    logger.error(f"Export failed: {e}")
    report.save_json("report.json")  # Fallback
```

## Testing

Comprehensive test coverage for:
- Report generation in all formats
- Metrics calculation accuracy
- Visualization generation
- Template rendering
- Export functionality
- Error handling
- Performance with large datasets

## Best Practices

1. **Always include metrics** for quality assurance
2. **Generate multiple formats** for different audiences
3. **Use visualizations** for better insights
4. **Include recommendations** for actionable insights
5. **Version reports** for tracking changes
6. **Validate data** before report generation
7. **Handle missing data** gracefully
8. **Cache expensive operations** for performance
9. **Use templates** for consistency
10. **Archive reports** for compliance

## Optional Dependencies

- **jinja2**: Advanced templating
- **matplotlib/seaborn**: Charts and graphs
- **plotly**: Interactive visualizations
- **weasyprint/reportlab**: PDF generation
- **openpyxl**: Excel formatting
- **xlsxwriter**: Advanced Excel features

## Future Enhancements Ideas

- Real-time report updates
- Interactive web dashboards
- Report scheduling and automation
- Email distribution
- Report comparison and diff
- Custom branding and themes
- Multi-language support
- Integration with BI tools
- Report API endpoints
- Automated insights with ML
