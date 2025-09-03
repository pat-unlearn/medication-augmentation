"""Unit tests for disease registry."""

import pytest
from typing import List
from unittest.mock import patch, MagicMock

from med_aug.diseases import DiseaseRegistry, DiseaseModule, DrugClassConfig


class MockDiseaseModule(DiseaseModule):
    """Mock disease module for testing."""

    def __init__(self, name: str = "mock", display: str = "Mock Disease"):
        self._name = name
        self._display = display

    @property
    def name(self) -> str:
        return self._name

    @property
    def display_name(self) -> str:
        return self._display

    @property
    def drug_classes(self) -> List[DrugClassConfig]:
        return [
            DrugClassConfig(
                name="mock_class",
                keywords=["mock_drug"],
                confidence_threshold=0.8,
                web_sources=["mock_source"],
            )
        ]

    def get_web_sources(self) -> List[str]:
        return ["http://mock.com"]

    def get_llm_context(self) -> str:
        return "Mock context"

    def validate_medication(self, medication: str, drug_class: str) -> bool:
        return medication == "mock_drug" and drug_class == "mock_class"


class TestDiseaseRegistry:
    """Test DiseaseRegistry class."""

    @pytest.fixture
    def empty_registry(self):
        """Create an empty registry without auto-discovery."""
        with patch.object(DiseaseRegistry, "_discover_modules"):
            registry = DiseaseRegistry()
            registry._modules = {}
            registry._instances = {}
            return registry

    @pytest.fixture
    def populated_registry(self, empty_registry):
        """Create a registry with mock modules."""
        mock_module_class = MockDiseaseModule
        mock_instance = MockDiseaseModule("test", "Test Disease")

        empty_registry._modules["test"] = mock_module_class
        empty_registry._instances["test"] = mock_instance

        return empty_registry

    def test_registry_initialization(self):
        """Test registry initialization triggers discovery."""
        with patch.object(DiseaseRegistry, "_discover_modules") as mock_discover:
            registry = DiseaseRegistry()
            mock_discover.assert_called_once()

    def test_get_module_existing(self, populated_registry):
        """Test getting an existing module."""
        module = populated_registry.get_module("test")

        assert module is not None
        assert module.name == "test"
        assert module.display_name == "Test Disease"

    def test_get_module_nonexistent(self, empty_registry):
        """Test getting a non-existent module returns None."""
        module = empty_registry.get_module("nonexistent")
        assert module is None

    def test_get_module_creates_new_instance(self, empty_registry):
        """Test get_module creates new instance if class exists but no instance."""
        mock_class = MockDiseaseModule
        empty_registry._modules["new"] = mock_class

        # No instance yet
        assert "new" not in empty_registry._instances

        # Get module should create instance
        module = empty_registry.get_module("new")

        assert module is not None
        assert "new" in empty_registry._instances
        assert module.name == "mock"  # Default from MockDiseaseModule

    def test_list_available(self, populated_registry):
        """Test listing available disease modules."""
        available = populated_registry.list_available()

        assert len(available) == 1
        assert "test" in available

    def test_list_available_empty(self, empty_registry):
        """Test listing when no modules available."""
        available = empty_registry.list_available()
        assert available == []

    def test_get_all_modules(self, populated_registry):
        """Test getting all module instances."""
        all_modules = populated_registry.get_all_modules()

        assert len(all_modules) == 1
        assert "test" in all_modules
        assert all_modules["test"].name == "test"

        # Should return a copy
        all_modules["new"] = "something"
        assert "new" not in populated_registry._instances

    def test_register_module(self, empty_registry):
        """Test manually registering a module."""
        mock_module_class = MockDiseaseModule

        empty_registry.register_module(mock_module_class)

        assert "mock" in empty_registry._modules
        assert "mock" in empty_registry._instances
        assert empty_registry._instances["mock"].name == "mock"

    def test_register_module_invalid_class(self, empty_registry):
        """Test registering invalid class raises error."""

        class NotADiseaseModule:
            pass

        with pytest.raises(TypeError, match="must be a subclass of DiseaseModule"):
            empty_registry.register_module(NotADiseaseModule)

    def test_unregister_module(self, populated_registry):
        """Test unregistering a module."""
        # Module exists
        assert "test" in populated_registry._modules

        # Unregister
        result = populated_registry.unregister_module("test")

        assert result is True
        assert "test" not in populated_registry._modules
        assert "test" not in populated_registry._instances

    def test_unregister_nonexistent_module(self, empty_registry):
        """Test unregistering non-existent module returns False."""
        result = empty_registry.unregister_module("nonexistent")
        assert result is False

    def test_reload_modules(self, populated_registry):
        """Test reloading all modules."""
        # Modules exist before reload
        assert len(populated_registry._modules) == 1

        with patch.object(DiseaseRegistry, "_discover_modules") as mock_discover:
            populated_registry.reload_modules()

            # Should clear and re-discover
            assert len(populated_registry._modules) == 0
            mock_discover.assert_called_once()

    @patch("med_aug.diseases.pkgutil.iter_modules")
    @patch("med_aug.diseases.importlib.import_module")
    def test_discover_modules_success(
        self, mock_import, mock_iter_modules, empty_registry
    ):
        """Test successful module discovery."""
        # Mock pkgutil to find a module
        mock_iter_modules.return_value = [(None, "test_disease", True)]  # ispkg=True

        # Mock the imported module
        mock_module = MagicMock()
        mock_module.MODULE_CLASS = MockDiseaseModule
        mock_import.return_value = mock_module

        # Run discovery
        empty_registry._discover_modules()

        # Should have discovered the module
        assert "mock" in empty_registry._modules
        assert empty_registry._modules["mock"] == MockDiseaseModule

    @patch("med_aug.diseases.pkgutil.iter_modules")
    def test_discover_modules_skips_base(self, mock_iter_modules, empty_registry):
        """Test that discovery skips the base module."""
        mock_iter_modules.return_value = [(None, "base", True), (None, "nsclc", True)]

        with patch("med_aug.diseases.importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.MODULE_CLASS = MockDiseaseModule
            mock_import.return_value = mock_module

            empty_registry._discover_modules()

            # Should only import nsclc, not base
            mock_import.assert_called_once()
            call_args = mock_import.call_args[0][0]
            assert "nsclc" in call_args
            assert "base" not in call_args

    @patch("med_aug.diseases.pkgutil.iter_modules")
    @patch("med_aug.diseases.importlib.import_module")
    def test_discover_modules_handles_import_error(
        self, mock_import, mock_iter_modules, empty_registry
    ):
        """Test that discovery handles import errors gracefully."""
        mock_iter_modules.return_value = [(None, "broken_module", True)]

        # Mock import to raise error
        mock_import.side_effect = ImportError("Module not found")

        # Should not raise, just log
        empty_registry._discover_modules()

        # Registry should be empty
        assert len(empty_registry._modules) == 0

    @patch("med_aug.diseases.pkgutil.iter_modules")
    @patch("med_aug.diseases.importlib.import_module")
    def test_discover_modules_validates_module_class(
        self, mock_import, mock_iter_modules, empty_registry
    ):
        """Test that discovery validates MODULE_CLASS is a DiseaseModule."""
        mock_iter_modules.return_value = [(None, "invalid_module", True)]

        # Mock module with wrong type
        mock_module = MagicMock()
        mock_module.MODULE_CLASS = "NotAClass"
        mock_import.return_value = mock_module

        empty_registry._discover_modules()

        # Should not register the invalid module
        assert len(empty_registry._modules) == 0

    def test_multiple_modules(self, empty_registry):
        """Test registry with multiple disease modules."""

        # Register multiple modules
        class Disease1(MockDiseaseModule):
            @property
            def name(self):
                return "disease1"

        class Disease2(MockDiseaseModule):
            @property
            def name(self):
                return "disease2"

        empty_registry.register_module(Disease1)
        empty_registry.register_module(Disease2)

        # Check both are registered
        available = empty_registry.list_available()
        assert len(available) == 2
        assert "disease1" in available
        assert "disease2" in available

        # Can get both
        mod1 = empty_registry.get_module("disease1")
        mod2 = empty_registry.get_module("disease2")

        assert mod1.name == "disease1"
        assert mod2.name == "disease2"
