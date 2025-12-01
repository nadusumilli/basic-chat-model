"""
Chat client module for interacting with IBM watsonx.ai models.
"""

from typing import Generator, List, Optional

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from .config import WatsonxConfig, get_model_parameters


class ChatClient:
    """
    A chat client for interacting with IBM watsonx.ai foundation models.

    This client provides a simple interface for sending messages to LLMs
    and receiving responses, with support for conversation history.
    """

    # Default model to use for chat
    DEFAULT_MODEL = "ibm/granite-13b-chat-v2"

    def __init__(
        self,
        config: Optional[WatsonxConfig] = None,
        model_id: Optional[str] = None,
    ):
        """
        Initialize the chat client.

        Args:
            config: WatsonxConfig object with credentials. If None, will load from env.
            model_id: The model ID to use. Defaults to granite-13b-chat-v2.
        """
        self.config = config or WatsonxConfig.from_env()
        self.model_id = model_id or self.DEFAULT_MODEL
        self.conversation_history: List[dict] = []

        # Initialize credentials
        self._credentials = Credentials(
            api_key=self.config.api_key,
            url=self.config.url,
        )

        # Initialize the model
        self._model = ModelInference(
            model_id=self.model_id,
            credentials=self._credentials,
            project_id=self.config.project_id,
        )

    def _format_prompt(self, message: str, include_history: bool = True) -> str:
        """
        Format the prompt with conversation history.

        Args:
            message: The current user message
            include_history: Whether to include conversation history

        Returns:
            str: Formatted prompt string
        """
        if not include_history or not self.conversation_history:
            return f"User: {message}\nAssistant:"

        # Build prompt with history
        prompt_parts = []
        for entry in self.conversation_history:
            role = entry["role"]
            content = entry["content"]
            if role == "user":
                prompt_parts.append(f"User: {content}")
            else:
                prompt_parts.append(f"Assistant: {content}")

        prompt_parts.append(f"User: {message}")
        prompt_parts.append("Assistant:")

        return "\n".join(prompt_parts)

    def send_message(
        self,
        message: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        include_history: bool = True,
    ) -> str:
        """
        Send a message to the chat model and get a response.

        Args:
            message: The message to send
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            include_history: Whether to include conversation history

        Returns:
            str: The model's response
        """
        prompt = self._format_prompt(message, include_history)
        params = get_model_parameters(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        response = self._model.generate_text(
            prompt=prompt,
            params=params,
        )

        # Clean up the response
        response = response.strip()

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append({"role": "assistant", "content": response})

        return response

    def stream_message(
        self,
        message: str,
        max_new_tokens: int = 500,
        temperature: float = 0.7,
        include_history: bool = True,
    ) -> Generator[str, None, None]:
        """
        Send a message and stream the response.

        Args:
            message: The message to send
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature for generation
            include_history: Whether to include conversation history

        Yields:
            str: Chunks of the model's response
        """
        prompt = self._format_prompt(message, include_history)
        params = get_model_parameters(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        full_response = []
        for chunk in self._model.generate_text_stream(
            prompt=prompt,
            params=params,
        ):
            full_response.append(chunk)
            yield chunk

        # Update conversation history with full response
        complete_response = "".join(full_response).strip()
        self.conversation_history.append({"role": "user", "content": message})
        self.conversation_history.append(
            {"role": "assistant", "content": complete_response}
        )

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.conversation_history = []

    def get_history(self) -> List[dict]:
        """
        Get the conversation history.

        Returns:
            List[dict]: List of conversation entries with 'role' and 'content' keys
        """
        return self.conversation_history.copy()
