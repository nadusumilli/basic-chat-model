"""
Tests for the configuration module.
"""

import os
from unittest.mock import patch

import pytest

from chat_model.config import WatsonxConfig, get_model_parameters


class TestWatsonxConfig:
    """Tests for WatsonxConfig class."""

    def test_config_creation(self):
        """Test creating a config with explicit values."""
        config = WatsonxConfig(
            api_key="test-api-key",
            project_id="test-project-id",
            url="https://test.ml.cloud.ibm.com",
        )
        assert config.api_key == "test-api-key"
        assert config.project_id == "test-project-id"
        assert config.url == "https://test.ml.cloud.ibm.com"

    def test_config_default_url(self):
        """Test that default URL is set correctly."""
        config = WatsonxConfig(
            api_key="test-api-key",
            project_id="test-project-id",
        )
        assert config.url == "https://us-south.ml.cloud.ibm.com"

    @patch.dict(
        os.environ,
        {
            "WATSONX_API_KEY": "env-api-key",
            "WATSONX_PROJECT_ID": "env-project-id",
        },
        clear=True,
    )
    def test_config_from_env(self):
        """Test loading config from environment variables."""
        config = WatsonxConfig.from_env()
        assert config.api_key == "env-api-key"
        assert config.project_id == "env-project-id"
        assert config.url == "https://us-south.ml.cloud.ibm.com"

    @patch.dict(
        os.environ,
        {
            "WATSONX_API_KEY": "env-api-key",
            "WATSONX_PROJECT_ID": "env-project-id",
            "WATSONX_URL": "https://custom.ml.cloud.ibm.com",
        },
        clear=True,
    )
    def test_config_from_env_with_custom_url(self):
        """Test loading config with custom URL from environment."""
        config = WatsonxConfig.from_env()
        assert config.url == "https://custom.ml.cloud.ibm.com"

    @patch.dict(os.environ, {}, clear=True)
    def test_config_from_env_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with pytest.raises(ValueError, match="WATSONX_API_KEY"):
            WatsonxConfig.from_env()

    @patch.dict(
        os.environ,
        {"WATSONX_API_KEY": "test-key"},
        clear=True,
    )
    def test_config_from_env_missing_project_id(self):
        """Test that missing project ID raises ValueError."""
        with pytest.raises(ValueError, match="WATSONX_PROJECT_ID"):
            WatsonxConfig.from_env()


class TestGetModelParameters:
    """Tests for get_model_parameters function."""

    def test_default_parameters(self):
        """Test default model parameters."""
        params = get_model_parameters()
        assert params["max_new_tokens"] == 500
        assert params["temperature"] == 0.7
        assert params["top_p"] == 0.9
        assert params["top_k"] == 50

    def test_custom_parameters(self):
        """Test custom model parameters."""
        params = get_model_parameters(
            max_new_tokens=1000,
            temperature=0.5,
            top_p=0.8,
            top_k=40,
        )
        assert params["max_new_tokens"] == 1000
        assert params["temperature"] == 0.5
        assert params["top_p"] == 0.8
        assert params["top_k"] == 40
