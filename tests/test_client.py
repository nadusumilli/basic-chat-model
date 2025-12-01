"""
Tests for the chat client module.
"""

from unittest.mock import MagicMock, patch

import pytest

from chat_model.client import ChatClient
from chat_model.config import WatsonxConfig


class TestChatClient:
    """Tests for ChatClient class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock configuration."""
        return WatsonxConfig(
            api_key="test-api-key",
            project_id="test-project-id",
            url="https://test.ml.cloud.ibm.com",
        )

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_client_initialization(self, mock_credentials, mock_model, mock_config):
        """Test client initialization."""
        client = ChatClient(config=mock_config)

        assert client.model_id == ChatClient.DEFAULT_MODEL
        assert client.conversation_history == []
        mock_credentials.assert_called_once_with(
            api_key="test-api-key",
            url="https://test.ml.cloud.ibm.com",
        )

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_client_custom_model(self, mock_credentials, mock_model, mock_config):
        """Test client with custom model ID."""
        client = ChatClient(config=mock_config, model_id="custom/model")
        assert client.model_id == "custom/model"

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_format_prompt_no_history(self, mock_credentials, mock_model, mock_config):
        """Test prompt formatting without history."""
        client = ChatClient(config=mock_config)
        prompt = client._format_prompt("Hello", include_history=True)
        assert prompt == "User: Hello\nAssistant:"

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_format_prompt_with_history(
        self, mock_credentials, mock_model, mock_config
    ):
        """Test prompt formatting with conversation history."""
        client = ChatClient(config=mock_config)
        client.conversation_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello! How can I help you?"},
        ]

        prompt = client._format_prompt("What is Python?", include_history=True)
        expected = (
            "User: Hi\n"
            "Assistant: Hello! How can I help you?\n"
            "User: What is Python?\n"
            "Assistant:"
        )
        assert prompt == expected

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_send_message(self, mock_credentials, mock_model_class, mock_config):
        """Test sending a message."""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_text.return_value = "Hello! I'm an AI assistant."
        mock_model_class.return_value = mock_model_instance

        client = ChatClient(config=mock_config)
        response = client.send_message("Hello")

        assert response == "Hello! I'm an AI assistant."
        assert len(client.conversation_history) == 2
        assert client.conversation_history[0] == {"role": "user", "content": "Hello"}
        assert client.conversation_history[1] == {
            "role": "assistant",
            "content": "Hello! I'm an AI assistant.",
        }

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_clear_history(self, mock_credentials, mock_model, mock_config):
        """Test clearing conversation history."""
        client = ChatClient(config=mock_config)
        client.conversation_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]

        client.clear_history()
        assert client.conversation_history == []

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_get_history(self, mock_credentials, mock_model, mock_config):
        """Test getting conversation history returns a copy."""
        client = ChatClient(config=mock_config)
        original_history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        client.conversation_history = original_history.copy()

        history = client.get_history()
        assert history == original_history
        # Verify it's a copy
        history.append({"role": "user", "content": "new message"})
        assert len(client.conversation_history) == 2

    @patch("chat_model.client.ModelInference")
    @patch("chat_model.client.Credentials")
    def test_stream_message(self, mock_credentials, mock_model_class, mock_config):
        """Test streaming message response."""
        mock_model_instance = MagicMock()
        mock_model_instance.generate_text_stream.return_value = iter(
            ["Hello", "!", " How", " can", " I", " help", "?"]
        )
        mock_model_class.return_value = mock_model_instance

        client = ChatClient(config=mock_config)

        chunks = list(client.stream_message("Hi"))

        assert chunks == ["Hello", "!", " How", " can", " I", " help", "?"]
        assert len(client.conversation_history) == 2
