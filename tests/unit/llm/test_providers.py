"""Unit tests for LLM providers."""

import pytest
import subprocess
import json

from med_aug.llm.providers import (
    LLMProvider,
    ClaudeCLIProvider,
    MockProvider,
    LLMConfig,
    LLMModel,
    LLMResponse,
    ProviderFactory,
)


class TestLLMConfig:
    """Test LLM configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = LLMConfig()

        assert config.model == LLMModel.CLAUDE_4_SONNET
        assert config.temperature == 0.0
        assert config.max_tokens == 4096
        assert config.timeout == 30
        assert config.retry_attempts == 3
        assert config.retry_delay == 1.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = LLMConfig(
            model=LLMModel.CLAUDE_3_OPUS, temperature=0.5, max_tokens=2048, timeout=60
        )

        assert config.model == LLMModel.CLAUDE_3_OPUS
        assert config.temperature == 0.5
        assert config.max_tokens == 2048
        assert config.timeout == 60

    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = LLMConfig(
            model=LLMModel.CLAUDE_3_SONNET, extra_params={"custom": "value"}
        )

        config_dict = config.to_dict()

        assert config_dict["model"] == "claude-3-sonnet"
        assert config_dict["temperature"] == 0.0
        assert config_dict["custom"] == "value"


class TestLLMResponse:
    """Test LLM response."""

    def test_response_creation(self):
        """Test response creation."""
        response = LLMResponse(
            content="Test response",
            model="test-model",
            usage={"prompt_tokens": 10, "completion_tokens": 20},
        )

        assert response.content == "Test response"
        assert response.model == "test-model"
        assert response.usage["prompt_tokens"] == 10
        assert response.usage["completion_tokens"] == 20

    def test_response_to_dict(self):
        """Test response serialization."""
        response = LLMResponse(
            content="Test content", model="test-model", metadata={"provider": "test"}
        )

        response_dict = response.to_dict()

        assert response_dict["content"] == "Test content"
        assert response_dict["model"] == "test-model"
        assert response_dict["metadata"]["provider"] == "test"


class TestMockProvider:
    """Test mock LLM provider."""

    @pytest.mark.asyncio
    async def test_mock_generate(self):
        """Test mock response generation."""
        provider = MockProvider()

        response = await provider.generate("Test prompt", system="Test system")

        assert isinstance(response, LLMResponse)
        assert response.model == "mock"
        assert "Mock response" in response.content

    @pytest.mark.asyncio
    async def test_mock_with_predefined_responses(self):
        """Test mock provider with predefined responses."""
        provider = MockProvider()
        provider.set_responses(["First response", "Second response"])

        response1 = await provider.generate("Prompt 1")
        assert response1.content == "First response"

        response2 = await provider.generate("Prompt 2")
        assert response2.content == "Second response"

        # After exhausting predefined responses, falls back to default
        response3 = await provider.generate("Prompt 3")
        assert "Mock response" in response3.content

    @pytest.mark.asyncio
    async def test_mock_classification_response(self):
        """Test mock classification response."""
        provider = MockProvider()

        response = await provider.generate("Classify medication: pembrolizumab")

        # Should return JSON for classification prompts
        data = json.loads(response.content)
        assert "classification" in data
        assert "confidence" in data
        assert data["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_mock_is_available(self):
        """Test mock provider availability."""
        provider = MockProvider()

        available = await provider.is_available()
        assert available is True


class TestClaudeCLIProvider:
    """Test Claude CLI provider."""

    @pytest.fixture
    def provider(self):
        """Create Claude CLI provider."""
        config = LLMConfig(
            model=LLMModel.CLAUDE_4_SONNET, temperature=0.0, max_tokens=100, timeout=10
        )
        return ClaudeCLIProvider(config)

    @pytest.mark.asyncio
    async def test_provider_initialization(self, provider):
        """Test provider initialization."""
        assert provider.cli_command == "claude"
        assert provider.config.model == LLMModel.CLAUDE_4_SONNET
        assert provider.config.temperature == 0.0

    @pytest.mark.asyncio
    @pytest.mark.skipif(
        subprocess.run(["which", "claude"], capture_output=True).returncode != 0,
        reason="Claude CLI not installed",
    )
    async def test_cli_availability(self, provider):
        """Test checking CLI availability."""
        available = await provider.is_available()
        # This will be True if Claude CLI is installed, False otherwise
        assert isinstance(available, bool)

    @pytest.mark.asyncio
    async def test_generate_handles_missing_cli(self, provider):
        """Test that generate handles missing CLI gracefully."""
        # Override command to something that doesn't exist
        provider.cli_command = "nonexistent_command_xyz"

        with pytest.raises(FileNotFoundError):
            await provider.generate("Test prompt")


class TestProviderFactory:
    """Test provider factory."""

    def test_create_mock_provider(self):
        """Test creating mock provider."""
        provider = ProviderFactory.create("mock")

        assert isinstance(provider, MockProvider)
        assert provider.config.model == LLMModel.MOCK

    def test_create_claude_cli_provider(self):
        """Test creating Claude CLI provider."""
        config = LLMConfig(model=LLMModel.CLAUDE_3_OPUS)
        provider = ProviderFactory.create("claude_cli", config)

        assert isinstance(provider, ClaudeCLIProvider)
        assert provider.config.model == LLMModel.CLAUDE_3_OPUS

    def test_create_unknown_provider(self):
        """Test creating unknown provider raises error."""
        with pytest.raises(ValueError, match="Unknown provider type"):
            ProviderFactory.create("unknown_provider")

    def test_list_available_providers(self):
        """Test listing available providers."""
        providers = ProviderFactory.list_available()

        assert "mock" in providers
        assert "claude_cli" in providers

    def test_register_custom_provider(self):
        """Test registering custom provider."""

        class CustomProvider(LLMProvider):
            async def generate(self, prompt, system=None, **kwargs):
                return LLMResponse(content="Custom", model="custom")

            async def is_available(self):
                return True

        ProviderFactory.register("custom", CustomProvider)

        providers = ProviderFactory.list_available()
        assert "custom" in providers

        provider = ProviderFactory.create("custom")
        assert isinstance(provider, CustomProvider)
