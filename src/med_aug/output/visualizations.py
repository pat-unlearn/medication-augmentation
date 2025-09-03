"""Data visualization components for reports."""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import base64
from io import BytesIO

from ..core.logging import get_logger

logger = get_logger(__name__)

# Try importing plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style("whitegrid")
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    logger.warning("plotting_libraries_unavailable", 
                  reason="matplotlib/seaborn not installed")


@dataclass
class VisualizationConfig:
    """Configuration for visualizations."""
    
    figure_size: Tuple[int, int] = (10, 6)
    dpi: int = 100
    color_palette: str = "Set2"
    save_format: str = "png"
    embed_in_html: bool = True
    max_categories: int = 15
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'figure_size': list(self.figure_size),
            'dpi': self.dpi,
            'color_palette': self.color_palette,
            'save_format': self.save_format,
            'embed_in_html': self.embed_in_html,
            'max_categories': self.max_categories
        }


class ChartGenerator:
    """Generator for data visualizations."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize chart generator.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.charts: List[Dict[str, Any]] = []
        
        if PLOTTING_AVAILABLE:
            sns.set_palette(self.config.color_palette)
        
        logger.info("chart_generator_initialized", 
                   plotting_available=PLOTTING_AVAILABLE)
    
    def create_bar_chart(
        self,
        data: Dict[str, float],
        title: str,
        xlabel: str = "Category",
        ylabel: str = "Value",
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a bar chart.
        
        Args:
            data: Dictionary of categories to values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            output_path: Optional path to save chart
            
        Returns:
            Chart metadata and data
        """
        logger.debug("creating_bar_chart", title=title)
        
        if not PLOTTING_AVAILABLE:
            return self._create_text_chart(data, title, "bar")
        
        # Limit categories if too many
        if len(data) > self.config.max_categories:
            sorted_data = dict(sorted(data.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:self.config.max_categories])
        else:
            sorted_data = data
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Plot bars
        categories = list(sorted_data.keys())
        values = list(sorted_data.values())
        
        bars = ax.bar(categories, values)
        
        # Customize
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Rotate x labels if many categories
        if len(categories) > 7:
            plt.xticks(rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.0f}' if val > 1 else f'{val:.2f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save or embed
        chart_data = self._save_or_embed_chart(fig, output_path, title)
        
        plt.close()
        
        return chart_data
    
    def create_pie_chart(
        self,
        data: Dict[str, float],
        title: str,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a pie chart.
        
        Args:
            data: Dictionary of categories to values
            title: Chart title
            output_path: Optional path to save chart
            
        Returns:
            Chart metadata and data
        """
        logger.debug("creating_pie_chart", title=title)
        
        if not PLOTTING_AVAILABLE:
            return self._create_text_chart(data, title, "pie")
        
        # Limit categories
        if len(data) > self.config.max_categories:
            sorted_data = dict(sorted(data.items(), 
                                    key=lambda x: x[1], 
                                    reverse=True)[:self.config.max_categories-1])
            other_value = sum(v for k, v in data.items() if k not in sorted_data)
            if other_value > 0:
                sorted_data['Other'] = other_value
        else:
            sorted_data = data
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Plot pie
        labels = list(sorted_data.keys())
        sizes = list(sorted_data.values())
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels, 
            autopct='%1.1f%%',
            startangle=90
        )
        
        # Customize
        ax.set_title(title, fontsize=16, fontweight='bold')
        
        # Improve text
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(10)
            autotext.set_weight('bold')
        
        plt.tight_layout()
        
        # Save or embed
        chart_data = self._save_or_embed_chart(fig, output_path, title)
        
        plt.close()
        
        return chart_data
    
    def create_histogram(
        self,
        data: List[float],
        title: str,
        xlabel: str = "Value",
        ylabel: str = "Frequency",
        bins: int = 20,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a histogram.
        
        Args:
            data: List of values
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            bins: Number of bins
            output_path: Optional path to save chart
            
        Returns:
            Chart metadata and data
        """
        logger.debug("creating_histogram", title=title)
        
        if not PLOTTING_AVAILABLE:
            return self._create_text_chart({'data_points': len(data)}, title, "histogram")
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config.figure_size, dpi=self.config.dpi)
        
        # Plot histogram
        n, bins_edges, patches = ax.hist(data, bins=bins, edgecolor='black', alpha=0.7)
        
        # Add mean line
        mean_val = sum(data) / len(data) if data else 0
        ax.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_val:.2f}')
        
        # Customize
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend()
        
        plt.tight_layout()
        
        # Save or embed
        chart_data = self._save_or_embed_chart(fig, output_path, title)
        
        plt.close()
        
        return chart_data
    
    def create_confidence_distribution(
        self,
        confidence_scores: List[float],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create confidence score distribution chart.
        
        Args:
            confidence_scores: List of confidence scores
            output_path: Optional path to save chart
            
        Returns:
            Chart metadata and data
        """
        logger.debug("creating_confidence_distribution")
        
        if not confidence_scores:
            return self._create_text_chart({}, "Confidence Distribution", "histogram")
        
        if not PLOTTING_AVAILABLE:
            bins = {'High (>0.8)': 0, 'Medium (0.5-0.8)': 0, 'Low (<0.5)': 0}
            for score in confidence_scores:
                if score > 0.8:
                    bins['High (>0.8)'] += 1
                elif score > 0.5:
                    bins['Medium (0.5-0.8)'] += 1
                else:
                    bins['Low (<0.5)'] += 1
            return self._create_text_chart(bins, "Confidence Distribution", "bar")
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=self.config.dpi)
        
        # Histogram
        ax1.hist(confidence_scores, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
        ax1.axvline(0.5, color='orange', linestyle='--', label='Medium threshold')
        ax1.axvline(0.8, color='green', linestyle='--', label='High threshold')
        ax1.set_title('Confidence Score Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Count')
        ax1.legend()
        
        # Box plot
        bp = ax2.boxplot(confidence_scores, vert=True, patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        ax2.set_title('Confidence Score Statistics', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Confidence Score')
        ax2.set_xticklabels(['All Medications'])
        
        # Add statistics text
        mean_conf = sum(confidence_scores) / len(confidence_scores)
        median_conf = sorted(confidence_scores)[len(confidence_scores)//2]
        ax2.text(1.2, mean_conf, f'Mean: {mean_conf:.3f}', va='center')
        ax2.text(1.2, median_conf, f'Median: {median_conf:.3f}', va='center')
        
        plt.suptitle('Medication Classification Confidence Analysis', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save or embed
        chart_data = self._save_or_embed_chart(fig, output_path, "Confidence Distribution")
        
        plt.close()
        
        return chart_data
    
    def create_metrics_dashboard(
        self,
        metrics: Dict[str, Any],
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create a metrics dashboard with multiple visualizations.
        
        Args:
            metrics: Dictionary of metrics
            output_path: Optional path to save chart
            
        Returns:
            Chart metadata and data
        """
        logger.debug("creating_metrics_dashboard")
        
        if not PLOTTING_AVAILABLE:
            return self._create_text_chart(metrics, "Metrics Dashboard", "dashboard")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10), dpi=self.config.dpi)
        
        # Define grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. Extraction metrics (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        extraction_data = metrics.get('extraction', {})
        if extraction_data:
            values = [extraction_data.get('total_extracted', 0),
                     extraction_data.get('unique_medications', 0),
                     extraction_data.get('normalized', 0)]
            labels = ['Total', 'Unique', 'Normalized']
            ax1.bar(labels, values, color=['#3498db', '#2ecc71', '#f39c12'])
            ax1.set_title('Extraction Metrics', fontweight='bold')
            ax1.set_ylabel('Count')
        
        # 2. Classification pie chart (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        classification = metrics.get('classification', {})
        if classification:
            sizes = [classification.get('classified', 0),
                    classification.get('unclassified', 0)]
            if sum(sizes) > 0:
                labels = ['Classified', 'Unclassified']
                colors = ['#2ecc71', '#e74c3c']
                ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                ax2.set_title('Classification Coverage', fontweight='bold')
        
        # 3. Confidence levels (top right)
        ax3 = fig.add_subplot(gs[0, 2])
        if classification:
            conf_data = [classification.get('high_confidence', 0),
                        classification.get('medium_confidence', 0),
                        classification.get('low_confidence', 0)]
            labels = ['High', 'Medium', 'Low']
            colors = ['#2ecc71', '#f39c12', '#e74c3c']
            ax3.bar(labels, conf_data, color=colors)
            ax3.set_title('Confidence Distribution', fontweight='bold')
            ax3.set_ylabel('Count')
        
        # 4. Drug class distribution (middle row, spanning 2 columns)
        ax4 = fig.add_subplot(gs[1, :2])
        distribution = metrics.get('distribution', {})
        drug_classes = distribution.get('drug_classes', {})
        if drug_classes:
            sorted_classes = dict(sorted(drug_classes.items(), 
                                       key=lambda x: x[1], 
                                       reverse=True)[:10])
            ax4.barh(list(sorted_classes.keys()), list(sorted_classes.values()))
            ax4.set_title('Top Drug Classes', fontweight='bold')
            ax4.set_xlabel('Count')
        
        # 5. Performance metrics (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        performance = metrics.get('performance', {})
        if performance:
            times = {k.replace('_time', '').title(): v 
                    for k, v in performance.items() 
                    if 'time' in k and v > 0}
            if times:
                ax5.barh(list(times.keys()), list(times.values()))
                ax5.set_title('Processing Times (s)', fontweight='bold')
                ax5.set_xlabel('Seconds')
        
        # 6. Summary statistics (bottom row)
        ax6 = fig.add_subplot(gs[2, :])
        ax6.axis('off')
        
        # Create summary text
        summary_text = "Summary Statistics\n" + "="*50 + "\n"
        
        if extraction_data:
            summary_text += f"Extraction Rate: {extraction_data.get('extraction_rate', 0)*100:.1f}%\n"
        
        if classification:
            summary_text += f"Classification Rate: {classification.get('classification_rate', 0)*100:.1f}%\n"
            summary_text += f"Average Confidence: {classification.get('average_confidence', 0):.3f}\n"
        
        validation = metrics.get('validation', {})
        if validation:
            summary_text += f"Validation Coverage: {validation.get('coverage', 0)*100:.1f}%\n"
        
        if performance:
            summary_text += f"Total Processing Time: {performance.get('total_time', 0):.1f}s\n"
        
        ax6.text(0.5, 0.5, summary_text, fontsize=12, ha='center', va='center',
                transform=ax6.transAxes, bbox=dict(boxstyle='round', facecolor='wheat'))
        
        plt.suptitle('Medication Augmentation Metrics Dashboard', 
                    fontsize=18, fontweight='bold')
        
        # Save or embed
        chart_data = self._save_or_embed_chart(fig, output_path, "Metrics Dashboard")
        
        plt.close()
        
        return chart_data
    
    def _save_or_embed_chart(
        self,
        fig,
        output_path: Optional[Path],
        title: str
    ) -> Dict[str, Any]:
        """Save chart to file or embed as base64."""
        chart_data = {
            'title': title,
            'type': 'matplotlib',
            'format': self.config.save_format
        }
        
        if output_path:
            # Save to file
            fig.savefig(output_path, format=self.config.save_format, 
                       dpi=self.config.dpi, bbox_inches='tight')
            chart_data['path'] = str(output_path)
            logger.debug("chart_saved", path=str(output_path))
            
        if self.config.embed_in_html:
            # Embed as base64
            buffer = BytesIO()
            fig.savefig(buffer, format=self.config.save_format, 
                       dpi=self.config.dpi, bbox_inches='tight')
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            chart_data['base64'] = image_base64
            chart_data['html'] = f'<img src="data:image/{self.config.save_format};base64,{image_base64}" alt="{title}">'
        
        self.charts.append(chart_data)
        return chart_data
    
    def _create_text_chart(
        self,
        data: Dict[str, Any],
        title: str,
        chart_type: str
    ) -> Dict[str, Any]:
        """Create text-based chart representation."""
        chart_data = {
            'title': title,
            'type': 'text',
            'chart_type': chart_type,
            'data': data,
            'html': self._generate_text_html(data, title, chart_type)
        }
        
        self.charts.append(chart_data)
        return chart_data
    
    def _generate_text_html(
        self,
        data: Dict[str, Any],
        title: str,
        chart_type: str
    ) -> str:
        """Generate HTML for text-based chart."""
        html = f'<div class="text-chart"><h3>{title}</h3>'
        
        if chart_type == "bar" or chart_type == "pie":
            html += '<table class="chart-table"><thead><tr><th>Category</th><th>Value</th></tr></thead><tbody>'
            for key, value in data.items():
                html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            html += '</tbody></table>'
            
        elif chart_type == "histogram":
            html += f'<p>Data points: {data.get("data_points", 0)}</p>'
            
        elif chart_type == "dashboard":
            html += '<pre>' + json.dumps(data, indent=2) + '</pre>'
        
        html += '</div>'
        return html
    
    def get_all_charts(self) -> List[Dict[str, Any]]:
        """Get all generated charts."""
        return self.charts