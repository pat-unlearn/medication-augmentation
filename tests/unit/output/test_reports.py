"""Unit tests for report generation."""

import pytest
from pathlib import Path
import json
import tempfile
from datetime import datetime

from med_aug.output.reports import (
    ReportGenerator,
    ReportConfig,
    ReportFormat,
    ReportSection
)


class TestReportConfig:
    """Test report configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        
        assert config.title == "Medication Augmentation Report"
        assert config.author == "Medication Augmentation System"
        assert ReportFormat.HTML in config.formats
        assert config.include_visualizations is True
        assert config.include_metrics is True
        assert config.include_raw_data is False
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = ReportConfig(
            title="Custom Report",
            subtitle="Test Subtitle",
            organization="Test Org",
            formats=[ReportFormat.JSON, ReportFormat.PDF],
            include_raw_data=True
        )
        
        assert config.title == "Custom Report"
        assert config.subtitle == "Test Subtitle"
        assert config.organization == "Test Org"
        assert ReportFormat.JSON in config.formats
        assert ReportFormat.PDF in config.formats
        assert config.include_raw_data is True
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = ReportConfig(
            title="Test",
            formats=[ReportFormat.HTML, ReportFormat.EXCEL]
        )
        
        config_dict = config.to_dict()
        
        assert config_dict['title'] == "Test"
        assert 'html' in config_dict['formats']
        assert 'excel' in config_dict['formats']


class TestReportSection:
    """Test report sections."""
    
    def test_section_creation(self):
        """Test creating report section."""
        section = ReportSection(
            title="Test Section",
            content={"key": "value"},
            section_type="data"
        )
        
        assert section.title == "Test Section"
        assert section.content == {"key": "value"}
        assert section.section_type == "data"
    
    def test_section_to_dict(self):
        """Test section serialization."""
        section = ReportSection(
            title="Metrics",
            content={"total": 100},
            section_type="metrics",
            metadata={"source": "test"}
        )
        
        section_dict = section.to_dict()
        
        assert section_dict['title'] == "Metrics"
        assert section_dict['content']['total'] == 100
        assert section_dict['type'] == "metrics"
        assert section_dict['metadata']['source'] == "test"


class TestReportGenerator:
    """Test report generator."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def generator(self, temp_dir):
        """Create report generator."""
        config = ReportConfig(
            title="Test Report",
            output_dir=temp_dir,
            formats=[ReportFormat.JSON]
        )
        return ReportGenerator(config)
    
    def test_generator_initialization(self, generator):
        """Test generator initialization."""
        assert generator.config.title == "Test Report"
        assert len(generator.sections) == 0
        assert 'generated_at' in generator.metadata
        assert generator.metadata['generator'] == 'Medication Augmentation System'
    
    def test_add_section(self, generator):
        """Test adding sections."""
        generator.add_section(ReportSection(
            title="Test",
            content="Content"
        ))
        
        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Test"
    
    def test_add_summary(self, generator):
        """Test adding summary section."""
        summary_data = {
            'Total Medications': 100,
            'Classified': 85,
            'Confidence': 0.92
        }
        
        generator.add_summary(summary_data)
        
        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Executive Summary"
        assert generator.sections[0].section_type == "summary"
        assert generator.sections[0].content == summary_data
    
    def test_add_medications_table(self, generator):
        """Test adding medications table."""
        medications = [
            {'name': 'pembrolizumab', 'class': 'immunotherapy'},
            {'name': 'osimertinib', 'class': 'targeted'}
        ]
        
        generator.add_medications_table(medications)
        
        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Extracted Medications"
        assert generator.sections[0].section_type == "table"
        assert len(generator.sections[0].content) == 2
    
    def test_add_metrics(self, generator):
        """Test adding metrics section."""
        metrics = {
            'extraction_rate': 0.95,
            'classification_rate': 0.88,
            'processing_time': 15.2
        }
        
        generator.add_metrics(metrics)
        
        assert len(generator.sections) == 1
        assert generator.sections[0].title == "Quality Metrics"
        assert generator.sections[0].section_type == "metrics"
    
    def test_generate_json(self, generator, temp_dir):
        """Test JSON report generation."""
        # Add some content
        generator.add_summary({'Total': 10})
        generator.add_metrics({'Score': 0.95})
        
        # Generate report
        output_files = generator.generate()
        
        assert ReportFormat.JSON in output_files
        json_file = output_files[ReportFormat.JSON]
        
        assert json_file.exists()
        assert json_file.suffix == '.json'
        
        # Verify content
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        assert 'metadata' in data
        assert 'sections' in data
        assert len(data['sections']) == 2
    
    def test_generate_html(self, temp_dir):
        """Test HTML report generation."""
        config = ReportConfig(
            title="HTML Test",
            output_dir=temp_dir,
            formats=[ReportFormat.HTML]
        )
        generator = ReportGenerator(config)
        
        # Add content
        generator.add_summary({'Medications': 50})
        generator.add_medications_table([
            {'name': 'drug1', 'confidence': 0.9},
            {'name': 'drug2', 'confidence': 0.8}
        ])
        
        # Generate report
        output_files = generator.generate()
        
        assert ReportFormat.HTML in output_files
        html_file = output_files[ReportFormat.HTML]
        
        assert html_file.exists()
        assert html_file.suffix == '.html'
        
        # Verify content
        content = html_file.read_text()
        assert 'HTML Test' in content
        assert 'Executive Summary' in content
        assert 'drug1' in content
    
    def test_generate_markdown(self, temp_dir):
        """Test Markdown report generation."""
        config = ReportConfig(
            title="Markdown Test",
            subtitle="Test Subtitle",
            output_dir=temp_dir,
            formats=[ReportFormat.MARKDOWN]
        )
        generator = ReportGenerator(config)
        
        # Add content
        generator.add_metrics({'accuracy': 0.95, 'speed': 10.5})
        
        # Generate report
        output_files = generator.generate()
        
        assert ReportFormat.MARKDOWN in output_files
        md_file = output_files[ReportFormat.MARKDOWN]
        
        assert md_file.exists()
        assert md_file.suffix == '.md'
        
        # Verify content
        content = md_file.read_text()
        assert '# Markdown Test' in content
        assert '## Test Subtitle' in content
        assert 'accuracy' in content
    
    def test_generate_excel(self, temp_dir):
        """Test Excel report generation."""
        config = ReportConfig(
            output_dir=temp_dir,
            formats=[ReportFormat.EXCEL]
        )
        generator = ReportGenerator(config)
        
        # Add table data
        generator.add_medications_table([
            {'medication': 'drug1', 'class': 'class1', 'confidence': 0.9},
            {'medication': 'drug2', 'class': 'class2', 'confidence': 0.85}
        ])
        
        # Generate report
        output_files = generator.generate()
        
        assert ReportFormat.EXCEL in output_files
        excel_file = output_files[ReportFormat.EXCEL]
        
        assert excel_file.exists()
        assert excel_file.suffix == '.xlsx'
    
    def test_generate_multiple_formats(self, temp_dir):
        """Test generating multiple formats."""
        config = ReportConfig(
            output_dir=temp_dir,
            formats=[ReportFormat.JSON, ReportFormat.HTML, ReportFormat.MARKDOWN]
        )
        generator = ReportGenerator(config)
        
        # Add content
        generator.add_summary({'Test': 'Value'})
        
        # Generate reports
        output_files = generator.generate()
        
        assert len(output_files) == 3
        assert all(fmt in output_files for fmt in config.formats)
        assert all(f.exists() for f in output_files.values())
    
    def test_add_visualization(self, generator):
        """Test adding visualization section."""
        chart_data = {
            'title': 'Test Chart',
            'data': {'A': 10, 'B': 20}
        }
        
        generator.add_visualization(chart_data, 'bar')
        
        assert len(generator.sections) == 1
        assert generator.sections[0].title == 'Test Chart'
        assert generator.sections[0].section_type == 'chart'
        assert generator.sections[0].metadata['chart_type'] == 'bar'
    
    def test_add_raw_data_respects_config(self, generator):
        """Test that raw data is only added when configured."""
        # Config has include_raw_data=False by default
        generator.add_raw_data({'raw': 'data'})
        assert len(generator.sections) == 0
        
        # Enable raw data
        generator.config.include_raw_data = True
        generator.add_raw_data({'raw': 'data'})
        assert len(generator.sections) == 1
        assert generator.sections[0].section_type == 'raw_data'