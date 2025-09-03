"""Common mixins for data classes and other utilities."""

from typing import Dict, Any
from dataclasses import asdict
from datetime import datetime
from enum import Enum


class DictMixin:
    """Mixin to provide to_dict() functionality for dataclasses."""

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert dataclass to dictionary with proper serialization.

        Handles common types like datetime, enums, and nested objects.
        """

        def _serialize_value(value):
            """Recursively serialize values to JSON-compatible types."""
            if isinstance(value, datetime):
                return value.isoformat()
            elif isinstance(value, Enum):
                return value.value
            elif hasattr(value, "to_dict"):
                return value.to_dict()
            elif isinstance(value, dict):
                return {k: _serialize_value(v) for k, v in value.items()}
            elif isinstance(value, (list, tuple)):
                return [_serialize_value(item) for item in value]
            else:
                return value

        # Use dataclasses.asdict for the base conversion
        data = asdict(self)

        # Apply custom serialization
        return {k: _serialize_value(v) for k, v in data.items()}
