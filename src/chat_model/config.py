"""
Configuration module for IBM watsonx.ai credentials.
"""

import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv


@dataclass
class WatsonxConfig:
    """Configuration for IBM watsonx.ai connection."""

    api_key: str
    project_id: str
    url: str = "https://us-south.ml.cloud.ibm.com"

    @classmethod
    def from_env(cls) -> "WatsonxConfig":
        """
        Load configuration from environment variables.

        Required environment variables:
            - WATSONX_API_KEY: IBM Cloud API key
            - WATSONX_PROJECT_ID: watsonx.ai project ID

        Optional environment variables:
            - WATSONX_URL: watsonx.ai service URL (defaults to us-south region)

        Returns:
            WatsonxConfig: Configuration object with credentials

        Raises:
            ValueError: If required environment variables are not set
        """
        load_dotenv()

        api_key = os.getenv("WATSONX_API_KEY")
        project_id = os.getenv("WATSONX_PROJECT_ID")
        url = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

        if not api_key:
            raise ValueError("WATSONX_API_KEY environment variable is required")
        if not project_id:
            raise ValueError("WATSONX_PROJECT_ID environment variable is required")

        return cls(api_key=api_key, project_id=project_id, url=url)


def get_model_parameters(
    max_new_tokens: int = 500,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
) -> dict:
    """
    Get model generation parameters.

    Args:
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0.0-2.0)
        top_p: Nucleus sampling probability
        top_k: Top-k sampling parameter

    Returns:
        dict: Model parameters dictionary
    """
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }
