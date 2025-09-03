"""LLM provider interfaces and implementations."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import asyncio
from enum import Enum

from ..core.logging import get_logger

logger = get_logger(__name__)


class LLMError(Exception):
    """Exception raised for LLM-related errors."""

    pass


class LLMModel(Enum):
    """Available LLM models."""

    CLAUDE_3_OPUS = "claude-3-opus"
    CLAUDE_3_SONNET = "claude-3-sonnet"
    CLAUDE_4_SONNET = "claude-4-sonnet"
    GPT_4 = "gpt-4"
    GPT_35_TURBO = "gpt-3.5-turbo"
    MOCK = "mock"


@dataclass
class LLMConfig:
    """Configuration for LLM providers."""

    model: LLMModel = LLMModel.CLAUDE_3_SONNET
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout: int = 120  # Increased for complex prompts
    retry_attempts: int = 3
    retry_delay: float = 1.0
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model.value,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
            "retry_attempts": self.retry_attempts,
            "retry_delay": self.retry_delay,
            **self.extra_params,
        }


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    model: str
    usage: Optional[Dict[str, int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "metadata": self.metadata,
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, config: LLMConfig):
        """
        Initialize LLM provider.

        Args:
            config: Provider configuration
        """
        self.config = config

    @abstractmethod
    async def generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """
        Generate response from LLM.

        Args:
            prompt: User prompt
            system: System message
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        """
        Check if provider is available.

        Returns:
            True if provider is available
        """
        pass

    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        return False

    async def stream_generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ):
        """Stream responses from LLM."""
        raise NotImplementedError("Streaming not supported by this provider")


class ClaudeCLIProvider(LLMProvider):
    """LLM provider using Claude CLI via subprocess."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize Claude CLI provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config or LLMConfig(model=LLMModel.CLAUDE_4_SONNET))

    async def generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """
        Generate response using Claude CLI.

        Args:
            prompt: User prompt
            system: System message (will be prepended to prompt)
            **kwargs: Additional parameters

        Returns:
            LLM response
        """
        logger.info("claude_cli_generation_started", prompt_length=len(prompt))

        # For Claude CLI, create a simple direct prompt
        # Convert complex classification prompts to simple formats
        simple_prompt = self._simplify_prompt_for_cli(prompt, system)

        try:
            # Build simple Claude CLI command as you specified
            model_map = {
                LLMModel.CLAUDE_3_OPUS: "opus",
                LLMModel.CLAUDE_3_SONNET: "sonnet",
                LLMModel.CLAUDE_4_SONNET: "sonnet",
            }

            model = model_map.get(self.config.model, "sonnet")  # Default to sonnet

            cmd = ["claude", "--model", model, "--print", simple_prompt]

            logger.debug(
                "claude_cli_command", command=" ".join(cmd[:-1]) + " [prompt]"
            )  # Don't log full prompt

            # Execute command
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )

            # Wait for completion with timeout
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=self.config.timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                raise TimeoutError(f"Claude CLI timed out after {self.config.timeout}s")

            # Check for errors
            if process.returncode != 0:
                error_msg = stderr.decode("utf-8") if stderr else "Unknown error"
                logger.error(
                    "claude_cli_error", error=error_msg, return_code=process.returncode
                )
                raise RuntimeError(f"Claude CLI failed: {error_msg}")

            # Parse response
            response_text = stdout.decode("utf-8").strip()

            logger.info(
                "claude_cli_generation_completed",
                response_length=len(response_text),
                model=self.config.model.value,
            )

            return LLMResponse(
                content=response_text,
                model=self.config.model.value,
                metadata={
                    "provider": "claude_cli",
                    "temperature": self.config.temperature,
                    "max_tokens": self.config.max_tokens,
                },
            )

        except Exception as e:
            logger.error("claude_cli_generation_failed", error=str(e))
            raise LLMError(f"Claude CLI failed: {e}")

    def _simplify_prompt_for_cli(
        self, prompt: str, system: Optional[str] = None
    ) -> str:
        """
        Simplify complex prompts for Claude CLI to avoid conversational responses.

        Args:
            prompt: Original complex prompt
            system: System message

        Returns:
            Simplified direct prompt
        """
        # Check if this is a medication classification prompt
        if "classify" in prompt.lower() and "medication" in prompt.lower():
            # Extract medication name from prompt
            import re

            med_match = re.search(r"Medication:\s*([^\n]+)", prompt)
            if med_match:
                medication = med_match.group(1).strip()
                return f"What is the drug class for {medication}? Answer with one or two words only."

        # Check if this is a batch classification prompt
        if "classify these medications" in prompt.lower():
            # For batch prompts, use them directly as they're already simplified
            return prompt

        # Check if this is a validation prompt
        if "validate" in prompt.lower() and "medication" in prompt.lower():
            med_match = re.search(r"Input:\s*([^\n]+)", prompt)
            if med_match:
                medication = med_match.group(1).strip()
                return f"Is '{medication}' a valid medication name? Answer: Yes or No"

        # For other prompts, create a simple version
        full_prompt = prompt
        if system:
            full_prompt = f"{system}\n\n{prompt}"

        # Truncate if too long and add direct instruction
        if len(full_prompt) > 500:
            full_prompt = full_prompt[:400] + "...\n\nAnswer directly and concisely."

        return full_prompt

    async def is_available(self) -> bool:
        """Check if Claude CLI is available."""
        try:
            process = await asyncio.create_subprocess_exec(
                "claude",
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            await asyncio.wait_for(process.wait(), timeout=5.0)

            return process.returncode == 0

        except (FileNotFoundError, asyncio.TimeoutError):
            return False


class MockProvider(LLMProvider):
    """Mock LLM provider for testing."""

    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize mock provider.

        Args:
            config: Provider configuration
        """
        super().__init__(config or LLMConfig(model=LLMModel.MOCK))
        self.responses: List[str] = []
        self.response_index = 0

    def set_responses(self, responses: List[str]):
        """Set mock responses."""
        self.responses = responses
        self.response_index = 0

    async def generate(
        self, prompt: str, system: Optional[str] = None, **kwargs
    ) -> LLMResponse:
        """
        Generate mock response.

        Args:
            prompt: User prompt
            system: System message
            **kwargs: Additional parameters

        Returns:
            Mock LLM response
        """
        # Use predefined response or generate based on prompt
        if self.responses and self.response_index < len(self.responses):
            content = self.responses[self.response_index]
            self.response_index += 1
        else:
            # Generate a simple mock response
            if "classify" in prompt.lower() or "medication" in prompt.lower():
                content = json.dumps(
                    {
                        "classification": "immunotherapy",
                        "confidence": 0.95,
                        "reasoning": "Mock classification based on keywords",
                    }
                )
            else:
                content = f"Mock response to: {prompt[:50]}..."

        return LLMResponse(
            content=content,
            model=self.config.model.value,
            usage={"prompt_tokens": len(prompt), "completion_tokens": len(content)},
            metadata={"provider": "mock", "prompt_preview": prompt[:100]},
        )

    async def is_available(self) -> bool:
        """Mock provider is always available."""
        return True


class ProviderFactory:
    """Factory for creating LLM providers."""

    _providers = {
        "claude_cli": ClaudeCLIProvider,
        "mock": MockProvider,
    }

    @classmethod
    def create(
        cls, provider_type: str, config: Optional[LLMConfig] = None
    ) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            provider_type: Type of provider
            config: Provider configuration

        Returns:
            LLM provider instance
        """
        if provider_type not in cls._providers:
            raise ValueError(f"Unknown provider type: {provider_type}")

        provider_class = cls._providers[provider_type]
        return provider_class(config)

    @classmethod
    def register(cls, name: str, provider_class):
        """Register a new provider type."""
        cls._providers[name] = provider_class

    @classmethod
    def list_available(cls) -> List[str]:
        """List available provider types."""
        return list(cls._providers.keys())
