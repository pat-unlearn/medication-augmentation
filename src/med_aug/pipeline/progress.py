"""Progress tracking for pipeline execution."""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import asyncio

from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel

from .phases import PhaseStatus, PhaseResult
from ..core.logging import get_logger
from ..core.mixins import DictMixin

logger = get_logger(__name__)
console = Console()


@dataclass
class PhaseProgress(DictMixin):
    """Progress information for a single phase."""

    name: str
    status: PhaseStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    progress_percent: float = 0.0
    current_step: str = ""
    error: Optional[str] = None

    @property
    def duration(self) -> Optional[timedelta]:
        """Get phase duration."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return datetime.now() - self.start_time
        return None

    @property
    def duration_str(self) -> str:
        """Get formatted duration string."""
        if self.duration:
            total_seconds = int(self.duration.total_seconds())
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)

            if hours > 0:
                return f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                return f"{minutes}m {seconds}s"
            else:
                return f"{seconds}s"
        return "-"


@dataclass
class ProgressReport(DictMixin):
    """Overall progress report for pipeline."""

    pipeline_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_phases: int = 0
    completed_phases: int = 0
    failed_phases: int = 0
    current_phase: Optional[str] = None
    phase_progress: Dict[str, PhaseProgress] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)

    @property
    def overall_progress(self) -> float:
        """Calculate overall progress percentage."""
        if self.total_phases == 0:
            return 0.0
        return (self.completed_phases / self.total_phases) * 100

    @property
    def duration(self) -> timedelta:
        """Get total duration."""
        end = self.end_time or datetime.now()
        return end - self.start_time

    @property
    def status(self) -> str:
        """Get overall status."""
        if self.failed_phases > 0:
            return "Failed"
        elif self.completed_phases == self.total_phases:
            return "Completed"
        elif self.current_phase:
            return "Running"
        else:
            return "Pending"

    # to_dict() method provided by DictMixin


class ProgressTracker:
    """Tracks and displays pipeline execution progress."""

    def __init__(self, pipeline_id: str, phases: List[str], display_mode: str = "rich"):
        """
        Initialize progress tracker.

        Args:
            pipeline_id: Pipeline identifier
            phases: List of phase names
            display_mode: Display mode (rich, simple, none)
        """
        self.pipeline_id = pipeline_id
        self.phases = phases
        self.display_mode = display_mode
        self.report = ProgressReport(
            pipeline_id=pipeline_id, start_time=datetime.now(), total_phases=len(phases)
        )

        # Initialize phase progress
        for phase in phases:
            self.report.phase_progress[phase] = PhaseProgress(
                name=phase, status=PhaseStatus.PENDING
            )

        self.callbacks: List[Callable] = []
        self._live = None
        self._stop_event = asyncio.Event()

        logger.info(
            "progress_tracker_initialized",
            pipeline_id=pipeline_id,
            total_phases=len(phases),
        )

    def add_callback(self, callback: Callable[[ProgressReport], None]):
        """Add a progress callback."""
        self.callbacks.append(callback)

    def phase_started(self, phase_name: str):
        """Mark phase as started."""
        if phase_name in self.report.phase_progress:
            progress = self.report.phase_progress[phase_name]
            progress.status = PhaseStatus.RUNNING
            progress.start_time = datetime.now()
            self.report.current_phase = phase_name

            logger.info("phase_started", phase=phase_name)
            self._notify_callbacks()
            self._update_display()

    def phase_completed(self, phase_name: str, result: Optional[PhaseResult] = None):
        """Mark phase as completed."""
        if phase_name in self.report.phase_progress:
            progress = self.report.phase_progress[phase_name]
            progress.status = PhaseStatus.COMPLETED
            progress.end_time = datetime.now()
            progress.progress_percent = 100.0

            self.report.completed_phases += 1

            if result:
                self.report.metrics.update(result.metrics)

            logger.info(
                "phase_completed", phase=phase_name, duration=progress.duration_str
            )

            # Move to next phase
            self._update_current_phase()
            self._notify_callbacks()
            self._update_display()

    def phase_failed(self, phase_name: str, error: str):
        """Mark phase as failed."""
        if phase_name in self.report.phase_progress:
            progress = self.report.phase_progress[phase_name]
            progress.status = PhaseStatus.FAILED
            progress.end_time = datetime.now()
            progress.error = error

            self.report.failed_phases += 1

            logger.error("phase_failed", phase=phase_name, error=error)

            self._notify_callbacks()
            self._update_display()

    def phase_skipped(self, phase_name: str, reason: str = ""):
        """Mark phase as skipped."""
        if phase_name in self.report.phase_progress:
            progress = self.report.phase_progress[phase_name]
            progress.status = PhaseStatus.SKIPPED
            progress.current_step = reason

            logger.info("phase_skipped", phase=phase_name, reason=reason)

            self._update_current_phase()
            self._notify_callbacks()
            self._update_display()

    def update_phase_progress(self, phase_name: str, percent: float, step: str = ""):
        """Update phase progress."""
        if phase_name in self.report.phase_progress:
            progress = self.report.phase_progress[phase_name]
            progress.progress_percent = min(100.0, max(0.0, percent))
            progress.current_step = step

            self._update_display()

    def pipeline_completed(self):
        """Mark pipeline as completed."""
        self.report.end_time = datetime.now()
        self.report.current_phase = None

        logger.info(
            "pipeline_completed",
            pipeline_id=self.pipeline_id,
            duration=str(self.report.duration),
            completed_phases=self.report.completed_phases,
            failed_phases=self.report.failed_phases,
        )

        self._notify_callbacks()
        self._update_display()
        self.stop_display()

    def _update_current_phase(self):
        """Update current phase to next pending."""
        for phase_name, progress in self.report.phase_progress.items():
            if progress.status == PhaseStatus.PENDING:
                self.report.current_phase = phase_name
                return
        self.report.current_phase = None

    def _notify_callbacks(self):
        """Notify all registered callbacks."""
        for callback in self.callbacks:
            try:
                callback(self.report)
            except Exception as e:
                logger.error("callback_error", error=str(e))

    def start_display(self):
        """Start progress display."""
        if self.display_mode == "rich":
            self._start_rich_display()
        elif self.display_mode == "simple":
            self._start_simple_display()

    def stop_display(self):
        """Stop progress display."""
        self._stop_event.set()
        if self._live:
            self._live.stop()

    def _start_rich_display(self):
        """Start rich progress display."""
        layout = self._create_layout()
        self._live = Live(
            layout, refresh_per_second=2, console=console
        )  # Update twice per second
        self._live.start()

        # Start background task to update display continuously
        asyncio.create_task(self._continuous_update())

    def _start_simple_display(self):
        """Start simple progress display."""
        console.print(f"[bold]Pipeline: {self.pipeline_id}[/bold]")
        console.print(f"Total phases: {self.report.total_phases}")

    def _update_display(self):
        """Update progress display."""
        if self.display_mode == "rich" and self._live:
            self._live.update(self._create_layout())
        elif self.display_mode == "simple":
            self._print_simple_progress()

    def _create_layout(self) -> Layout:
        """Create rich layout for progress display."""
        layout = Layout()

        # Create header
        header = Panel(
            f"[bold blue]Pipeline: {self.pipeline_id}[/bold blue]\n"
            f"Status: {self._get_status_emoji()} {self.report.status}\n"
            f"Progress: {self.report.overall_progress:.1f}% "
            f"({self.report.completed_phases}/{self.report.total_phases} phases)\n"
            f"Duration: {str(self.report.duration).split('.')[0]}",
            title="Pipeline Progress",
        )

        # Create phase table
        table = Table(title="Phase Status")
        table.add_column("Phase", style="cyan")
        table.add_column("Status", style="magenta")
        table.add_column("Duration", style="yellow")

        for phase_name, progress in self.report.phase_progress.items():
            status_str = self._format_status(progress.status)

            table.add_row(phase_name, status_str, progress.duration_str)

        layout.split_column(Layout(header, size=6), Layout(table))

        return layout

    def _print_simple_progress(self):
        """Print simple progress update."""
        if self.report.current_phase:
            progress = self.report.phase_progress[self.report.current_phase]
            console.print(
                f"  {self.report.current_phase}: "
                f"{progress.status.value} "
                f"({progress.progress_percent:.0f}%)"
            )

    def _get_status_emoji(self) -> str:
        """Get emoji for current status."""
        if self.report.status == "Completed":
            return "✅"
        elif self.report.status == "Failed":
            return "❌"
        elif self.report.status == "Running":
            return "🔄"
        else:
            return "⏳"

    def _format_status(self, status: PhaseStatus) -> str:
        """Format status with color and emoji."""
        if status == PhaseStatus.COMPLETED:
            return "[green]✅ Completed[/green]"
        elif status == PhaseStatus.FAILED:
            return "[red]❌ Failed[/red]"
        elif status == PhaseStatus.RUNNING:
            return "[yellow]🔄 Running[/yellow]"
        elif status == PhaseStatus.SKIPPED:
            return "[dim]⏭️ Skipped[/dim]"
        else:
            return "[dim]⏳ Pending[/dim]"

    async def _continuous_update(self):
        """Continuously update the display for real-time duration updates."""
        while not self._stop_event.is_set():
            try:
                await asyncio.sleep(1.0)  # Update every second
                if self.display_mode == "rich" and self._live:
                    self._live.update(self._create_layout())
            except Exception as e:
                logger.debug("continuous_update_error", error=str(e))
                break

    def get_report(self) -> ProgressReport:
        """Get current progress report."""
        return self.report

    def print_summary(self):
        """Print final summary."""
        console.print("\n[bold]Pipeline Summary[/bold]")
        console.print(f"Pipeline ID: {self.pipeline_id}")
        console.print(f"Status: {self._get_status_emoji()} {self.report.status}")
        console.print(f"Duration: {str(self.report.duration).split('.')[0]}")
        console.print(
            f"Phases: {self.report.completed_phases}/{self.report.total_phases} completed"
        )

        if self.report.failed_phases > 0:
            console.print(f"[red]Failed phases: {self.report.failed_phases}[/red]")
            for phase_name, progress in self.report.phase_progress.items():
                if progress.status == PhaseStatus.FAILED:
                    console.print(f"  - {phase_name}: {progress.error}")

        if self.report.metrics:
            console.print("\n[bold]Metrics:[/bold]")
            for key, value in self.report.metrics.items():
                console.print(f"  {key}: {value}")
