# Basic Chat Model

A basic chat application that interacts with LLMs through IBM watsonx.ai.

## Features

- Interactive chat interface with IBM watsonx.ai foundation models
- Conversation history management
- Streaming responses
- Configurable model parameters

## Installation

```bash
# Clone the repository
git clone https://github.com/nadusumilli/basic-chat-model.git
cd basic-chat-model

# Install the package
pip install -e .

# For development (includes test dependencies)
pip install -e ".[dev]"
```

## Configuration

Set the following environment variables before running the application:

```bash
export WATSONX_API_KEY="your-ibm-cloud-api-key"
export WATSONX_PROJECT_ID="your-watsonx-project-id"
export WATSONX_URL="https://us-south.ml.cloud.ibm.com"  # Optional, defaults to us-south
```

You can also create a `.env` file in the project root:

```env
WATSONX_API_KEY=your-ibm-cloud-api-key
WATSONX_PROJECT_ID=your-watsonx-project-id
WATSONX_URL=https://us-south.ml.cloud.ibm.com
```

## Usage

### Command Line Interface

Run the interactive chat:

```bash
chat
```

Or run as a Python module:

```bash
python -m chat_model.main
```

### Available Commands

- `/clear` - Clear conversation history
- `/history` - Show conversation history
- `/quit` - Exit the application

### Python API

```python
from chat_model.client import ChatClient
from chat_model.config import WatsonxConfig

# Using environment variables
client = ChatClient()

# Or with explicit configuration
config = WatsonxConfig(
    api_key="your-api-key",
    project_id="your-project-id",
    url="https://us-south.ml.cloud.ibm.com"
)
client = ChatClient(config=config)

# Send a message
response = client.send_message("Hello, how are you?")
print(response)

# Stream a response
for chunk in client.stream_message("Tell me a joke"):
    print(chunk, end="", flush=True)

# Clear conversation history
client.clear_history()

# Get conversation history
history = client.get_history()
```

## Supported Models

The default model is `ibm/granite-13b-chat-v2`. You can specify a different model:

```python
client = ChatClient(model_id="ibm/granite-34b-code-instruct")
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Running Tests with Coverage

```bash
pytest tests/ --cov=chat_model --cov-report=term-missing
```

## License

MIT License - see [LICENSE](LICENSE) for details.
