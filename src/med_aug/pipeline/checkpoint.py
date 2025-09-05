"""Checkpoint management for pipeline recovery."""

import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime

from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineCheckpoint:
    """Checkpoint for pipeline state."""

    pipeline_id: str
    timestamp: datetime
    current_phase: str
    completed_phases: List[str]
    context: Dict[str, Any]
    phase_results: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pipeline_id": self.pipeline_id,
            "timestamp": self.timestamp.isoformat(),
            "current_phase": self.current_phase,
            "completed_phases": self.completed_phases,
            "context": self._serialize_context(self.context),
            "phase_results": self.phase_results,
            "metadata": self.metadata,
        }

    @staticmethod
    def _serialize_context(context: Dict[str, Any]) -> Dict[str, Any]:
        """Serialize context, handling special types."""
        serialized = {}
        for key, value in context.items():
            # Skip dataframes and other non-serializable objects
            if key in ["dataframe", "extraction_results"]:
                serialized[key] = f"<{type(value).__name__} object>"
            else:
                try:
                    json.dumps(value)  # Test if serializable
                    serialized[key] = value
                except (TypeError, ValueError):
                    serialized[key] = str(value)
        return serialized

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineCheckpoint":
        """Create from dictionary."""
        return cls(
            pipeline_id=data["pipeline_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            current_phase=data["current_phase"],
            completed_phases=data["completed_phases"],
            context=data["context"],
            phase_results=data["phase_results"],
            metadata=data.get("metadata", {}),
        )


class CheckpointManager:
    """Manages pipeline checkpoints for recovery."""

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoint files
        """
        if checkpoint_dir:
            self.checkpoint_dir = Path(checkpoint_dir)
        else:
            self.checkpoint_dir = Path.home() / ".med_aug" / "checkpoints"

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "checkpoint_manager_initialized", directory=str(self.checkpoint_dir)
        )

    def save_checkpoint(self, checkpoint: PipelineCheckpoint) -> Path:
        """
        Save a checkpoint to disk.

        Args:
            checkpoint: Checkpoint to save

        Returns:
            Path to saved checkpoint file
        """
        filename = f"{checkpoint.pipeline_id}_{checkpoint.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.checkpoint_dir / filename

        try:
            with open(filepath, "w") as f:
                json.dump(checkpoint.to_dict(), f, indent=2)

            logger.info(
                "checkpoint_saved",
                pipeline_id=checkpoint.pipeline_id,
                phase=checkpoint.current_phase,
                file=str(filepath),
            )
            return filepath

        except Exception as e:
            logger.error(
                "checkpoint_save_failed",
                pipeline_id=checkpoint.pipeline_id,
                error=str(e),
            )
            raise

    def load_checkpoint(
        self, pipeline_id: str, latest: bool = True
    ) -> Optional[PipelineCheckpoint]:
        """
        Load a checkpoint from disk.

        Args:
            pipeline_id: Pipeline ID to load
            latest: Whether to load the latest checkpoint

        Returns:
            Loaded checkpoint or None if not found
        """
        pattern = f"{pipeline_id}_*.json"
        checkpoint_files = list(self.checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            logger.warning("no_checkpoint_found", pipeline_id=pipeline_id)
            return None

        if latest:
            checkpoint_file = max(checkpoint_files, key=lambda f: f.stat().st_mtime)
        else:
            checkpoint_file = checkpoint_files[0]

        try:
            with open(checkpoint_file, "r") as f:
                data = json.load(f)

            checkpoint = PipelineCheckpoint.from_dict(data)

            logger.info(
                "checkpoint_loaded",
                pipeline_id=pipeline_id,
                phase=checkpoint.current_phase,
                file=str(checkpoint_file),
            )
            return checkpoint

        except Exception as e:
            logger.error(
                "checkpoint_load_failed",
                pipeline_id=pipeline_id,
                file=str(checkpoint_file),
                error=str(e),
            )
            return None

    def list_checkpoints(
        self, pipeline_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List available checkpoints.

        Args:
            pipeline_id: Optional pipeline ID to filter by

        Returns:
            List of checkpoint metadata
        """
        if pipeline_id:
            pattern = f"{pipeline_id}_*.json"
        else:
            pattern = "*.json"

        checkpoint_files = list(self.checkpoint_dir.glob(pattern))
        checkpoints = []

        for file in checkpoint_files:
            try:
                # Extract metadata from filename
                parts = file.stem.split("_")
                if len(parts) >= 3:
                    pipeline_id = "_".join(parts[:-2])
                    timestamp = f"{parts[-2]}_{parts[-1]}"

                    checkpoints.append(
                        {
                            "pipeline_id": pipeline_id,
                            "timestamp": timestamp,
                            "file": str(file),
                            "size_bytes": file.stat().st_size,
                            "modified": datetime.fromtimestamp(
                                file.stat().st_mtime
                            ).isoformat(),
                        }
                    )
            except Exception as e:
                logger.debug("checkpoint_parse_error", file=str(file), error=str(e))

        return sorted(checkpoints, key=lambda x: x["modified"], reverse=True)

    def delete_checkpoint(
        self, pipeline_id: str, timestamp: Optional[str] = None
    ) -> bool:
        """
        Delete checkpoint(s).

        Args:
            pipeline_id: Pipeline ID
            timestamp: Optional specific timestamp to delete

        Returns:
            True if deleted successfully
        """
        if timestamp:
            pattern = f"{pipeline_id}_{timestamp}.json"
        else:
            pattern = f"{pipeline_id}_*.json"

        checkpoint_files = list(self.checkpoint_dir.glob(pattern))

        if not checkpoint_files:
            logger.warning("no_checkpoints_to_delete", pipeline_id=pipeline_id)
            return False

        for file in checkpoint_files:
            try:
                file.unlink()
                logger.info("checkpoint_deleted", file=str(file))
            except Exception as e:
                logger.error("checkpoint_delete_failed", file=str(file), error=str(e))
                return False

        return True

    def cleanup_old_checkpoints(self, days: int = 7) -> int:
        """
        Clean up old checkpoints.

        Args:
            days: Delete checkpoints older than this many days

        Returns:
            Number of checkpoints deleted
        """
        import time

        cutoff_time = time.time() - (days * 24 * 3600)
        deleted = 0

        for file in self.checkpoint_dir.glob("*.json"):
            if file.stat().st_mtime < cutoff_time:
                try:
                    file.unlink()
                    deleted += 1
                    logger.debug("old_checkpoint_deleted", file=str(file))
                except Exception as e:
                    logger.error(
                        "checkpoint_cleanup_failed", file=str(file), error=str(e)
                    )

        if deleted > 0:
            logger.info("checkpoints_cleaned_up", count=deleted, days=days)

        return deleted

    def save_full_state(self, pipeline_id: str, state: Dict[str, Any]) -> Path:
        """
        Save full pipeline state including dataframes.

        Args:
            pipeline_id: Pipeline ID
            state: Complete pipeline state

        Returns:
            Path to saved state file
        """
        filename = f"{pipeline_id}_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
        filepath = self.checkpoint_dir / filename

        try:
            with open(filepath, "wb") as f:
                pickle.dump(state, f)

            logger.info(
                "full_state_saved",
                pipeline_id=pipeline_id,
                file=str(filepath),
                size_mb=filepath.stat().st_size / 1024 / 1024,
            )
            return filepath

        except Exception as e:
            logger.error(
                "full_state_save_failed", pipeline_id=pipeline_id, error=str(e)
            )
            raise

    def load_full_state(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """
        Load full pipeline state.

        Args:
            pipeline_id: Pipeline ID

        Returns:
            Complete pipeline state or None
        """
        pattern = f"{pipeline_id}_full_*.pkl"
        state_files = list(self.checkpoint_dir.glob(pattern))

        if not state_files:
            logger.warning("no_full_state_found", pipeline_id=pipeline_id)
            return None

        # Get latest
        state_file = max(state_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(state_file, "rb") as f:
                state = pickle.load(f)

            logger.info(
                "full_state_loaded", pipeline_id=pipeline_id, file=str(state_file)
            )
            return state

        except Exception as e:
            logger.error(
                "full_state_load_failed", pipeline_id=pipeline_id, error=str(e)
            )
            return None
