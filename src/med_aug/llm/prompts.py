"""Prompt templates for LLM interactions."""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from string import Template
import json


@dataclass
class PromptTemplate:
    """Template for generating prompts."""

    name: str
    system: str
    user_template: str
    output_format: Optional[str] = None
    examples: Optional[List[Dict[str, Any]]] = None

    def format(self, **kwargs) -> tuple[str, str]:
        """
        Format the prompt with given parameters.

        Args:
            **kwargs: Parameters for template substitution

        Returns:
            Tuple of (system_message, user_prompt)
        """
        # Format user prompt
        user_prompt = Template(self.user_template).safe_substitute(**kwargs)

        # Add output format if specified
        if self.output_format:
            user_prompt += f"\n\nOutput Format:\n{self.output_format}"

        # Add examples if provided
        if self.examples:
            examples_text = "\n\nExamples:\n"
            for example in self.examples:
                examples_text += f"Input: {example['input']}\n"
                examples_text += f"Output: {example['output']}\n\n"
            user_prompt += examples_text

        return self.system, user_prompt

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "system": self.system,
            "user_template": self.user_template,
            "output_format": self.output_format,
            "examples": self.examples,
        }


class MedicationPrompts:
    """Collection of medication-related prompt templates."""

    @staticmethod
    def normalization_prompt() -> PromptTemplate:
        """Prompt for normalizing medication names and finding variants."""
        return PromptTemplate(
            name="medication_normalization",
            system=(
                "You are a pharmaceutical expert specializing in medication naming and standardization. "
                "Your task is to normalize medication names to their standard generic form and "
                "identify all known brand name variants. Focus on providing the canonical drug name "
                "that would be used in clinical databases, not therapeutic classifications."
            ),
            user_template=(
                "Normalize the following medication name and find its variants:\n\n"
                "Medication: $medication\n"
                "Disease Context: $disease\n\n"
                "Tasks:\n"
                "1. Identify the standard generic name (lowercase)\n"
                "2. Find all known brand names and variants\n"
                "3. Determine if this is a valid medication for the specified disease context\n"
                "4. Provide high confidence if this is a real medication used in this therapeutic area"
            ),
            output_format=(
                "Return a JSON object with the following structure:\n"
                "{\n"
                '  "input_medication": "the input medication name",\n'
                '  "generic_name": "standard generic name (lowercase)",\n'
                '  "brand_names": ["list", "of", "brand", "names"],\n'
                '  "is_disease_specific_drug": true/false,\n'
                '  "is_valid_medication": true/false,\n'
                '  "confidence": 0.0-1.0,\n'
                '  "reasoning": "explanation for identification"\n'
                "}"
            ),
            examples=[
                {
                    "input": "Keytruda for NSCLC",
                    "output": json.dumps(
                        {
                            "input_medication": "Keytruda",
                            "generic_name": "pembrolizumab",
                            "brand_names": ["Keytruda", "KEYTRUDA"],
                            "is_disease_specific_drug": True,
                            "is_valid_medication": True,
                            "confidence": 1.0,
                            "reasoning": "Keytruda is the brand name for pembrolizumab, a PD-1 inhibitor used in cancer treatment",
                        },
                        indent=2,
                    ),
                },
                {
                    "input": "osimertinib",
                    "output": json.dumps(
                        {
                            "input_medication": "osimertinib",
                            "generic_name": "osimertinib",
                            "brand_names": ["Tagrisso"],
                            "is_disease_specific_drug": True,
                            "is_valid_medication": True,
                            "confidence": 1.0,
                            "reasoning": "Osimertinib is the generic name, marketed as Tagrisso, used for EGFR-mutated NSCLC",
                        },
                        indent=2,
                    ),
                },
            ],
        )

    @staticmethod
    def classification_prompt() -> PromptTemplate:
        """Prompt for classifying medications."""
        return PromptTemplate(
            name="medication_classification",
            system=(
                "You are a medical expert specializing in pharmaceutical classification. "
                "Your task is to classify medications into their appropriate drug classes "
                "based on their mechanism of action, therapeutic use, and chemical structure. "
                "Be precise and use standard medical terminology."
            ),
            user_template=(
                "Classify the following medication into its drug class:\n\n"
                "Medication: $medication\n"
                "Disease Context: $disease\n\n"
                "Consider the following drug classes:\n$drug_classes\n\n"
                "Provide your classification with confidence score and reasoning."
            ),
            output_format=(
                "Return a JSON object with the following structure:\n"
                "{\n"
                '  "medication": "the input medication name",\n'
                '  "classification": "primary drug class",\n'
                '  "alternative_classifications": ["other possible classes"],\n'
                '  "confidence": 0.0-1.0,\n'
                '  "reasoning": "explanation for classification",\n'
                '  "mechanism_of_action": "brief description",\n'
                '  "therapeutic_use": "primary indication"\n'
                "}"
            ),
            examples=[
                {
                    "input": "pembrolizumab for NSCLC",
                    "output": json.dumps(
                        {
                            "medication": "pembrolizumab",
                            "classification": "immunotherapy",
                            "alternative_classifications": [
                                "PD-1 inhibitor",
                                "checkpoint inhibitor",
                            ],
                            "confidence": 1.0,
                            "reasoning": "Pembrolizumab is a monoclonal antibody that blocks PD-1",
                            "mechanism_of_action": "PD-1 receptor blockade",
                            "therapeutic_use": "Non-small cell lung cancer treatment",
                        },
                        indent=2,
                    ),
                }
            ],
        )

    @staticmethod
    def validation_prompt() -> PromptTemplate:
        """Prompt for validating medication names."""
        return PromptTemplate(
            name="medication_validation",
            system=(
                "You are a pharmaceutical expert. Your task is to validate medication names, "
                "identify if they are real medications, and correct any misspellings or variations. "
                "Consider brand names, generic names, and common abbreviations."
            ),
            user_template=(
                "Validate the following medication name:\n\n"
                "Input: $medication\n\n"
                "Tasks:\n"
                "1. Determine if this is a valid medication name\n"
                "2. Provide the correct/standard name if different\n"
                "3. Identify brand vs generic name\n"
                "4. List any alternative names or formulations"
            ),
            output_format=(
                "Return a JSON object with:\n"
                "{\n"
                '  "input": "the input text",\n'
                '  "is_valid": true/false,\n'
                '  "standard_name": "correct medication name",\n'
                '  "type": "brand/generic/combination",\n'
                '  "generic_name": "generic name if brand",\n'
                '  "brand_names": ["list of brand names"],\n'
                '  "confidence": 0.0-1.0\n'
                "}"
            ),
        )

    @staticmethod
    def augmentation_prompt() -> PromptTemplate:
        """Prompt for augmenting medication information."""
        return PromptTemplate(
            name="medication_augmentation",
            system=(
                "You are a clinical pharmacology expert. Provide comprehensive information "
                "about medications including dosing, side effects, interactions, and clinical pearls. "
                "Focus on information relevant to the specified disease context."
            ),
            user_template=(
                "Provide detailed information for:\n\n"
                "Medication: $medication\n"
                "Disease Context: $disease\n"
                "Drug Class: $drug_class\n\n"
                "Include:\n"
                "1. Standard dosing regimens\n"
                "2. Common side effects and management\n"
                "3. Key drug interactions\n"
                "4. Monitoring parameters\n"
                "5. Clinical considerations specific to $disease"
            ),
            output_format=(
                "Return a structured JSON with comprehensive medication information:\n"
                "{\n"
                '  "medication": "name",\n'
                '  "dosing": {...},\n'
                '  "side_effects": {...},\n'
                '  "interactions": [...],\n'
                '  "monitoring": {...},\n'
                '  "clinical_pearls": [...]\n'
                "}"
            ),
        )

    @staticmethod
    def extraction_enhancement_prompt() -> PromptTemplate:
        """Prompt for enhancing medication extraction."""
        return PromptTemplate(
            name="extraction_enhancement",
            system=(
                "You are an expert in medical text processing and medication identification. "
                "Your task is to identify medications from various text formats including "
                "abbreviations, misspellings, and non-standard notations."
            ),
            user_template=(
                "Extract and standardize medication names from the following text:\n\n"
                "$text\n\n"
                "Consider:\n"
                "- Common abbreviations (e.g., 'pembro' for pembrolizumab)\n"
                "- Misspellings and typos\n"
                "- Combination therapies (e.g., 'carbo/taxol')\n"
                "- Dosage information that may be attached\n"
                "- Both generic and brand names"
            ),
            output_format=(
                "Return a JSON array of extracted medications:\n"
                "[\n"
                "  {\n"
                '    "original_text": "as found in input",\n'
                '    "standard_name": "standardized name",\n'
                '    "type": "single/combination",\n'
                '    "components": ["if combination"],\n'
                '    "confidence": 0.0-1.0\n'
                "  }\n"
                "]"
            ),
        )

    @staticmethod
    def batch_classification_prompt() -> PromptTemplate:
        """Prompt for classifying multiple medications at once."""
        return PromptTemplate(
            name="batch_classification",
            system=(
                "You are a medical expert. Classify multiple medications efficiently "
                "into their drug classes. Be consistent in your classifications."
            ),
            user_template=(
                "Classify the following medications for $disease treatment:\n\n"
                "$medications\n\n"
                "Available drug classes:\n$drug_classes\n\n"
                "Classify each medication and group by drug class."
            ),
            output_format=(
                "Return a JSON object grouping medications by class:\n"
                "{\n"
                '  "classifications": {\n'
                '    "drug_class_1": ["med1", "med2"],\n'
                '    "drug_class_2": ["med3"]\n'
                "  },\n"
                '  "unclassified": ["unknown_med"],\n'
                '  "summary": {\n'
                '    "total": 10,\n'
                '    "classified": 8,\n'
                '    "confidence": 0.85\n'
                "  }\n"
                "}"
            ),
        )

    @staticmethod
    def context_aware_prompt() -> PromptTemplate:
        """Prompt that uses additional context for classification."""
        return PromptTemplate(
            name="context_aware_classification",
            system=(
                "You are a clinical oncology expert. Use the provided clinical context "
                "to accurately classify medications considering their use in specific "
                "treatment protocols and patient populations."
            ),
            user_template=(
                "Given the clinical context, classify this medication:\n\n"
                "Medication: $medication\n"
                "Disease: $disease\n"
                "Treatment Line: $treatment_line\n"
                "Other Medications: $concomitant_meds\n"
                "Patient Factors: $patient_factors\n\n"
                "Determine the most appropriate drug class and therapeutic role."
            ),
            output_format=(
                "Return detailed classification with context:\n"
                "{\n"
                '  "medication": "name",\n'
                '  "classification": "primary class",\n'
                '  "therapeutic_role": "e.g., first-line, adjuvant",\n'
                '  "combination_potential": ["compatible drugs"],\n'
                '  "context_notes": "relevant considerations",\n'
                '  "confidence": 0.0-1.0\n'
                "}"
            ),
        )


class PromptManager:
    """Manager for prompt templates."""

    def __init__(self):
        """Initialize prompt manager."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_default_templates()

    def _load_default_templates(self):
        """Load default medication prompts."""
        self.templates["normalization"] = MedicationPrompts.normalization_prompt()
        self.templates["classification"] = MedicationPrompts.classification_prompt()
        self.templates["validation"] = MedicationPrompts.validation_prompt()
        self.templates["augmentation"] = MedicationPrompts.augmentation_prompt()
        self.templates["extraction"] = MedicationPrompts.extraction_enhancement_prompt()
        self.templates["batch_classification"] = (
            MedicationPrompts.batch_classification_prompt()
        )
        self.templates["context_aware"] = MedicationPrompts.context_aware_prompt()

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get a prompt template by name."""
        return self.templates.get(name)

    def add_template(self, template: PromptTemplate):
        """Add a custom prompt template."""
        self.templates[template.name] = template

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())

    def format_prompt(self, template_name: str, **kwargs) -> tuple[str, str]:
        """
        Format a prompt using a template.

        Args:
            template_name: Name of template to use
            **kwargs: Parameters for template

        Returns:
            Tuple of (system_message, user_prompt)
        """
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        return template.format(**kwargs)
