"""
Main entry point for the chat application.
"""

import sys


def main():
    """Run the interactive chat application."""
    try:
        from .client import ChatClient
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install the package with: pip install -e .")
        sys.exit(1)

    print("=" * 60)
    print("Basic Chat Model - IBM watsonx.ai")
    print("=" * 60)
    print("\nInitializing chat client...")

    try:
        client = ChatClient()
        print(f"Connected to model: {client.model_id}")
    except ValueError as e:
        print(f"\nConfiguration Error: {e}")
        print("\nPlease set the following environment variables:")
        print("  - WATSONX_API_KEY: Your IBM Cloud API key")
        print("  - WATSONX_PROJECT_ID: Your watsonx.ai project ID")
        print("  - WATSONX_URL (optional): watsonx.ai service URL")
        sys.exit(1)
    except Exception as e:
        print(f"\nError initializing client: {e}")
        sys.exit(1)

    print("\nType your messages below. Commands:")
    print("  /clear  - Clear conversation history")
    print("  /history - Show conversation history")
    print("  /quit   - Exit the application")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() == "/quit":
                print("\nGoodbye!")
                break
            elif user_input.lower() == "/clear":
                client.clear_history()
                print("Conversation history cleared.")
                continue
            elif user_input.lower() == "/history":
                history = client.get_history()
                if not history:
                    print("No conversation history.")
                else:
                    print("\nConversation History:")
                    for entry in history:
                        role = entry["role"].capitalize()
                        content = entry["content"]
                        print(f"  {role}: {content[:100]}...")
                continue

            # Send message and get response
            print("\nAssistant: ", end="", flush=True)
            try:
                for chunk in client.stream_message(user_input):
                    print(chunk, end="", flush=True)
                print()  # New line after response
            except Exception as e:
                print(f"\nError getting response: {e}")

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\n\nGoodbye!")
            break


if __name__ == "__main__":
    main()
