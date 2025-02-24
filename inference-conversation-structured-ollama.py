"""
A CLI chat interface for Ollama models with structured outputs.
Usage: python inference-conversation-structured-ollama.py [--model MODEL] [--temperature TEMP]

Examples:
    # Start chat with default model
    python inference-conversation-structured-ollama.py

    # Use specific model and temperature
    python inference-conversation-structured-ollama.py --model codellama --temperature 0.2

Once started, use commands like:
    /json    - Switch to JSON output
    /kv      - Switch to Key-Value output
    /list    - Switch to List output
    /table   - Switch to Table output
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import List, Dict, Any
import argparse
import json

# Default system message that defines structured output behavior
DEFAULT_SYSTEM_PROMPT = """You are an AI assistant that provides structured responses.
Always format your responses according to the requested structure.

Available output formats:
1. JSON: When asked for JSON, respond with valid JSON only
2. Key-Value: When asked for key-value pairs, use "key: value" format
3. List: When asked for lists, use "- item" format
4. Table: When asked for tables, use markdown table format

Example formats:
JSON: {"name": "value", "items": ["a", "b"]}
Key-Value: 
name: value
category: example
List:
- First item
- Second item
Table:
| Column1 | Column2 |
|---------|---------|
| Value1  | Value2  |

Always maintain these structures strictly in your responses."""

class StructuredOutputAI:
    """Handles chat interactions with structured output formatting"""
    
    def __init__(
        self, 
        model_name: str = "deepseek-r1:8b", 
        temperature: float = 0.1,  # Lower temperature for more consistent structured output
        system_prompt: str = DEFAULT_SYSTEM_PROMPT
    ):
        self.llm = ChatOllama(
            model=model_name,
            temperature=temperature
        )
        self.chat_history: List[HumanMessage | AIMessage] = []
        self.system_message = SystemMessage(content=system_prompt)

    def parse_json_response(self, response: str) -> Dict[str, Any]:
        """Attempt to parse a JSON response"""
        try:
            # Find JSON content between curly braces
            start = response.find('{')
            end = response.rfind('}') + 1
            if start >= 0 and end > 0:
                json_str = response[start:end]
                return json.loads(json_str)
            return {"error": "No JSON found in response"}
        except json.JSONDecodeError:
            return {"error": "Invalid JSON in response"}

    def chat(self, user_input: str, format_type: str = None) -> Any:
        """Process a chat interaction with optional format parsing
        
        Args:
            user_input: The user's message
            format_type: Optional format specification ('json', 'key-value', 'list', 'table')
        """
        if format_type:
            # Add format instruction to user input
            user_input = f"Respond to this in {format_type} format: {user_input}"
        
        messages = [self.system_message] + self.chat_history + [HumanMessage(content=user_input)]
        response = self.llm.invoke(messages)
        
        # Store the interaction in chat history
        self.chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response.content)
        ])
        
        # Parse response if format specified
        if format_type == 'json':
            return self.parse_json_response(response.content)
        return response.content

    def clear_history(self):
        """Reset the conversation history"""
        self.chat_history = []

def print_help():
    """Display available commands and formats"""
    print("\nCommands:")
    print("  /quit    - Exit the chat")
    print("  /clear   - Clear chat history")
    print("  /json    - Request JSON format")
    print("  /kv      - Request Key-Value format")
    print("  /list    - Request List format")
    print("  /table   - Request Table format")
    print("  /help    - Show this help message")

def main():
    """Main chat loop"""
    parser = argparse.ArgumentParser(description='Chat with structured outputs')
    parser.add_argument('--model', default='deepseek-r1:8b', help='Ollama model name')
    parser.add_argument('--temperature', type=float, default=0.1, help='Response randomness')
    args = parser.parse_args()

    chat_ai = StructuredOutputAI(
        model_name=args.model,
        temperature=args.temperature
    )
    
    print(f"\nStructured Output Chat started with {args.model}")
    print("Type messages normally or use commands for specific formats.")
    print_help()
    
    current_format = None
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.startswith('/'):
                command = user_input[1:].lower()
                if command == 'quit':
                    print("Goodbye!")
                    break
                elif command == 'clear':
                    chat_ai.clear_history()
                    current_format = None
                    print("Chat history cleared!")
                elif command == 'json':
                    current_format = 'json'
                    print("Switched to JSON format")
                elif command == 'kv':
                    current_format = 'key-value'
                    print("Switched to Key-Value format")
                elif command == 'list':
                    current_format = 'list'
                    print("Switched to List format")
                elif command == 'table':
                    current_format = 'table'
                    print("Switched to Table format")
                elif command == 'help':
                    print_help()
                else:
                    print(f"Unknown command: {user_input}")
                continue
            
            if user_input:
                response = chat_ai.chat(user_input, current_format)
                if isinstance(response, dict):
                    print("\nAI (JSON):")
                    print(json.dumps(response, indent=2))
                else:
                    print(f"\nAI ({current_format or 'text'}):")
                    print(response)
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main() 