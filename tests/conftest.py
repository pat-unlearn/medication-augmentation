"""Shared fixtures and pytest configuration."""

import pytest
from pathlib import Path
import tempfile
import shutil

from med_aug.core.models import (
    Medication,
    MedicationType,
    DrugClass,
    ColumnAnalysisResult,
)
from med_aug.diseases.base import DrugClassConfig


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    # Cleanup
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_medications():
    """Create sample medications for testing."""
    return [
        Medication(
            name="pembrolizumab",
            type=MedicationType.GENERIC,
            confidence=0.95,
            source="FDA",
            metadata={"approval_year": 2014, "indication": "NSCLC"},
        ),
        Medication(
            name="Keytruda",
            type=MedicationType.BRAND,
            confidence=0.98,
            source="FDA",
            metadata={"generic": "pembrolizumab"},
        ),
        Medication(
            name="nivolumab",
            type=MedicationType.GENERIC,
            confidence=0.93,
            source="Clinical",
            metadata={"approval_year": 2015},
        ),
        Medication(
            name="Opdivo",
            type=MedicationType.BRAND,
            confidence=0.97,
            source="FDA",
            metadata={"generic": "nivolumab"},
        ),
        Medication(
            name="carboplatin",
            type=MedicationType.GENERIC,
            confidence=0.88,
            source="Literature",
            metadata={"class": "chemotherapy"},
        ),
    ]


@pytest.fixture
def sample_drug_classes(sample_medications):
    """Create sample drug classes."""
    return [
        DrugClass(
            name="immunotherapy",
            taking_variable="taking_pembrolizumab",
            current_medications=sample_medications[:4],  # immunotherapy drugs
            category="immunotherapy",
            disease="nsclc",
        ),
        DrugClass(
            name="chemotherapy",
            taking_variable="taking_carboplatin",
            current_medications=[sample_medications[4]],  # chemotherapy drug
            category="chemotherapy",
            disease="nsclc",
        ),
    ]


@pytest.fixture
def sample_drug_class_configs():
    """Create sample drug class configurations."""
    return [
        DrugClassConfig(
            name="immunotherapy",
            keywords=[
                "pembrolizumab",
                "nivolumab",
                "atezolizumab",
                "keytruda",
                "opdivo",
            ],
            confidence_threshold=0.85,
            web_sources=["fda", "nccn", "oncokb"],
        ),
        DrugClassConfig(
            name="chemotherapy",
            keywords=["carboplatin", "paclitaxel", "docetaxel", "cisplatin"],
            confidence_threshold=0.80,
            web_sources=["fda", "nccn"],
        ),
        DrugClassConfig(
            name="targeted_therapy",
            keywords=["osimertinib", "erlotinib", "afatinib", "tagrisso"],
            confidence_threshold=0.90,
            web_sources=["fda", "oncokb"],
        ),
    ]


@pytest.fixture
def sample_column_analysis_results():
    """Create sample column analysis results."""
    return [
        ColumnAnalysisResult(
            column="AGENT",
            confidence=0.92,
            total_count=1000,
            unique_count=150,
            sample_medications=[
                "pembrolizumab",
                "nivolumab",
                "carboplatin",
                "paclitaxel",
                "osimertinib",
            ],
            reasoning="High medication pattern match; Column name indicates drugs",
        ),
        ColumnAnalysisResult(
            column="MEDICATION",
            confidence=0.88,
            total_count=500,
            unique_count=75,
            sample_medications=["drug1", "drug2", "drug3"],
            reasoning="Strong medication indicators",
        ),
        ColumnAnalysisResult(
            column="PATIENT_ID",
            confidence=0.15,
            total_count=1000,
            unique_count=1000,
            sample_medications=[],
            reasoning="No medication patterns; appears to be identifier",
        ),
    ]


@pytest.fixture
def sample_csv_data(temp_dir):
    """Create sample CSV file with medication data."""
    import pandas as pd

    data = {
        "PATIENT_ID": ["P001", "P002", "P003", "P004", "P005"],
        "AGENT": [
            "PEMBROLIZUMAB",
            "NIVOLUMAB",
            "CARBOPLATIN + PACLITAXEL",
            "OSIMERTINIB",
            "ATEZOLIZUMAB",
        ],
        "DOSE": ["200mg", "240mg", "AUC5 + 200mg/m2", "80mg", "1200mg"],
        "START_DATE": [
            "2024-01-01",
            "2024-02-01",
            "2024-03-01",
            "2024-04-01",
            "2024-05-01",
        ],
    }

    df = pd.DataFrame(data)
    csv_path = temp_dir / "test_medications.csv"
    df.to_csv(csv_path, index=False)

    return csv_path


@pytest.fixture
def sample_yaml_config(temp_dir):
    """Create sample YAML configuration file."""
    import yaml

    config = {
        "disease": {
            "name": "Test Disease",
            "code": "test",
            "description": "Test disease for unit tests",
            "drug_classes": {
                "immunotherapy": {
                    "keywords": ["pembrolizumab", "nivolumab"],
                    "confidence_threshold": 0.85,
                    "web_sources": ["fda", "nccn"],
                },
                "chemotherapy": {
                    "keywords": ["carboplatin", "paclitaxel"],
                    "confidence_threshold": 0.80,
                    "web_sources": ["fda"],
                },
            },
        }
    }

    yaml_path = temp_dir / "test_config.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    return yaml_path


@pytest.fixture
def mock_web_research_result():
    """Create mock web research result."""
    return {
        "medication_name": "pembrolizumab",
        "generic_names": ["pembrolizumab"],
        "brand_names": ["Keytruda"],
        "drug_class_hints": ["PD-1 inhibitor", "immunotherapy", "checkpoint inhibitor"],
        "mechanism_of_action": "Binds to PD-1 receptor and blocks interaction with PD-L1/PD-L2",
        "fda_approval_date": "2014-09-04",
        "clinical_trials": ["NCT02220894", "NCT02578680", "NCT03937219"],
        "sources": ["FDA", "ClinicalTrials.gov", "OncoKB"],
        "indications": ["NSCLC", "Melanoma", "HNSCC", "Bladder Cancer"],
    }


@pytest.fixture
def mock_llm_response():
    """Create mock LLM classification response."""
    return {
        "classification": "immunotherapy",
        "confidence": 0.95,
        "reasoning": "Pembrolizumab is a PD-1 inhibitor checkpoint inhibitor used in cancer immunotherapy",
        "alternative_classes": ["checkpoint_inhibitor"],
        "notes": "First-line treatment for PD-L1 positive NSCLC",
    }


# Pytest configuration


def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line(
        "markers", "requires_network: mark test as requiring network access"
    )


def pytest_collection_modifyitems(config, items):
    """Automatically add markers based on test location."""
    for item in items:
        # Add unit marker for tests in unit directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)

        # Add integration marker for tests in integration directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)

        # Add markers based on test name
        if "slow" in item.nodeid:
            item.add_marker(pytest.mark.slow)

        if "network" in item.nodeid or "web" in item.nodeid:
            item.add_marker(pytest.mark.requires_network)


# Test data fixtures


@pytest.fixture
def nsclc_medication_names():
    """Common NSCLC medication names for testing."""
    return {
        "immunotherapy": [
            "pembrolizumab",
            "Keytruda",
            "nivolumab",
            "Opdivo",
            "atezolizumab",
            "Tecentriq",
            "durvalumab",
            "Imfinzi",
        ],
        "chemotherapy": [
            "carboplatin",
            "Paraplatin",
            "paclitaxel",
            "Taxol",
            "pemetrexed",
            "Alimta",
            "docetaxel",
            "Taxotere",
        ],
        "targeted_therapy": [
            "osimertinib",
            "Tagrisso",
            "erlotinib",
            "Tarceva",
            "alectinib",
            "Alecensa",
            "crizotinib",
            "Xalkori",
        ],
    }


@pytest.fixture
def mock_disease_module_data():
    """Mock data for a disease module."""
    return {
        "name": "mock_disease",
        "display_name": "Mock Disease for Testing",
        "drug_classes": [
            {
                "name": "class_a",
                "keywords": ["drug_a1", "drug_a2", "drug_a3"],
                "confidence_threshold": 0.85,
                "web_sources": ["source1", "source2"],
            },
            {
                "name": "class_b",
                "keywords": ["drug_b1", "drug_b2"],
                "confidence_threshold": 0.90,
                "web_sources": ["source3"],
            },
        ],
        "web_sources": ["http://example.com/mock", "http://test.com/mock"],
        "llm_context": "This is a mock disease for testing purposes.",
    }
