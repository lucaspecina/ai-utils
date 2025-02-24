"""
A simple CLI chat interface for Ollama models.
Usage: python inference-conversation-ollama.py [--model MODEL] [--temperature TEMP]
"""

# Required imports
from langchain_ollama import ChatOllama  # Interface to Ollama models
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage  # Message types for chat
from typing import List  # Type hints
import argparse  # Command line argument parsing

# Default system message that defines the AI's behavior
DEFAULT_SYSTEM_PROMPT = """You are a witty, opinionated AI assistant with a strong personality.
You're not afraid to:
- Challenge the user's assumptions and beliefs
- Take controversial positions and defend them with logic
- Say "no" when you disagree
- Use humor and sarcasm when appropriate
- Engage in friendly debates and play devil's advocate
- Point out flaws in arguments
- Share your own unique perspectives
- Base arguments on logic and facts
- Admit when you're wrong or uncertain

You have your own personality and views - you're not just an agreeable assistant.
Feel free to express disagreement, skepticism, or enthusiasm based on your perspective.
Engage with the user as an intellectual equal in thoughtful debate and discussion."""


class ConversationalAI:
    """Main class that handles the chat interaction with the AI model"""
    
    def __init__(
        self, 
        model_name: str = "deepseek-r1:8b", 
        temperature: float = 0.2,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        """Initialize the chat interface
        Args:
            model_name: Name of the Ollama model to use
            temperature: Controls randomness in responses (0.0 = deterministic, 1.0 = creative)
            system_prompt: Instructions that define the AI's behavior
        """
        # Initialize the language model
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        # Store conversation history
        self.chat_history: List[HumanMessage | AIMessage] = []
        # Set the system message that defines AI behavior
        self.system_message = SystemMessage(content=system_prompt)

    def chat(self, user_input: str) -> str:
        """Process a single chat interaction
        Args:
            user_input: The user's message
        Returns:
            The AI's response
        """
        # Combine system message, chat history, and new input
        messages = [self.system_message] + self.chat_history + [HumanMessage(content=user_input)]
        # Get AI's response
        response = self.llm.invoke(messages)
        
        # Add the current exchange to chat history
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response.content)
        ])
        
        return response.content

    def clear_history(self):
        """Reset the conversation history"""
        self.chat_history = []

    def display_history(self):
        """Show the conversation history in a readable format"""
        if not self.chat_history:
            print("No chat history available.")
            return
            
        for message in self.chat_history:
            prefix = "You:" if isinstance(message, HumanMessage) else "AI:"
            print(f"{prefix} {message.content}\n")

    def display_full_prompt(self):
        """Display the complete prompt including system message and chat history"""
        print("\n=== FULL PROMPT ===")
        print("\n[System Message]:")
        print(self.system_message.content)
        
        print("\n[Chat History]:")
        if not self.chat_history:
            print("(Empty)")
        else:
            for message in self.chat_history:
                prefix = "User" if isinstance(message, HumanMessage) else "Assistant"
                print(f"\n{prefix}: {message.content}")
        print("\n=================")

def get_args():
    """Parse and return command line arguments"""
    parser = argparse.ArgumentParser(description='Chat with an Ollama LLM model')
    parser.add_argument(
        '--model', 
        default='deepseek-r1:8b',
        help='Ollama model name (default: deepseek-r1:8b)'
    )
    parser.add_argument(
        '--temperature', 
        type=float, 
        default=0.2,
        help='Response randomness (0.0-1.0, default: 0.2)'
    )
    parser.add_argument(
        '--system-prompt', 
        default=DEFAULT_SYSTEM_PROMPT,
        help='System prompt to set the AI\'s behavior'
    )
    return parser.parse_args()

def print_help():
    """Display available chat commands"""
    print("\nAvailable commands:")
    print("  /quit    - Exit the chat")
    print("  /clear   - Clear chat history")
    print("  /history - Show chat history")
    print("  /prompt  - Show full prompt including system message")
    print("  /help    - Show this help message")

def main():
    """Main chat loop"""
    # Get command line arguments
    args = get_args()
    
    # Initialize the chat interface
    chat_ai = ConversationalAI(
        model_name=args.model,
        temperature=args.temperature,
        system_prompt=args.system_prompt
    )
    
    # Show initial information
    print(f"\nChat started with {args.model} (temperature: {args.temperature})")
    print_help()
    
    # Main interaction loop
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            # Handle special commands (starting with '/')
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                if command == 'quit':
                    print("Goodbye!")
                    break
                elif command == 'clear':
                    chat_ai.clear_history()
                    print("Chat history cleared!")
                elif command == 'history':
                    chat_ai.display_history()
                elif command == 'prompt':
                    chat_ai.display_full_prompt()
                elif command == 'help':
                    print_help()
                else:
                    print(f"Unknown command: {user_input}")
                continue
            
            # Process normal chat messages
            if user_input:
                response = chat_ai.chat(user_input)
                print(f"\nAI: {response}")
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\nGoodbye!")
            break
        except Exception as e:
            # Handle any other errors without crashing
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()